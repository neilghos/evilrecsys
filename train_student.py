import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from dataloaders import build_sparse_adj_matrix, load_cf_txt_dataset
from evil_model import EvilLightGCN
from metrics import evaluate_cf
from third_teacher import VALID_REF_MODES, build_third_teacher, compute_ref_scores, load_frozen_teacher
from train_evil import FastLightGCNSampler, _maybe_to_csr


def _zscore(x):
    return (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True).clamp(min=1e-6)


def sample_candidates(u, pos_i, neg_i, num_items, ref_topk, evil_topk, n_ref_sample, n_evil_sample, num_cands, device):
    """Builds per-user candidate list mixing positives, negatives, teacher-mined hard items, and randoms."""
    parts = [pos_i.unsqueeze(1), neg_i.unsqueeze(1)]
    bsz = len(u)
    if n_ref_sample > 0 and ref_topk is not None:
        idx = torch.randint(0, ref_topk.size(1), (bsz, n_ref_sample), device=device)
        parts.append(torch.gather(ref_topk[u], 1, idx))
    if n_evil_sample > 0 and evil_topk is not None:
        idx = torch.randint(0, evil_topk.size(1), (bsz, n_evil_sample), device=device)
        parts.append(torch.gather(evil_topk[u], 1, idx))
    if num_cands > 0:
        parts.append(torch.randint(0, num_items, (bsz, num_cands), device=device))
    return torch.cat(parts, dim=1)


def kd_third_teacher(u, cand_items, u_stu, i_stu, tt, ref_mode, gamma, tau, scale_student=True):
    """Distills frozen Third-Teacher scores into the student over candidate items [B, M]."""
    s_stu = torch.bmm(i_stu[cand_items], u_stu.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        s_g = torch.bmm(tt["i_good"][cand_items], tt["u_good"][u].unsqueeze(-1)).squeeze(-1)
        s_e = torch.bmm(tt["i_evil"][cand_items], tt["u_evil"][u].unsqueeze(-1)).squeeze(-1)
        s_ref = compute_ref_scores(
            s_g,
            s_e,
            ref_mode,
            gamma,
            mean_g=tt["mean_g"][u],
            std_g=tt["std_g"][u],
            mean_e=tt["mean_e"][u],
            std_e=tt["std_e"][u],
            beta_u=tt["beta_u"][u],
        )
        if scale_student:
            s_ref = _zscore(s_ref)
        p_ref = F.softmax(s_ref / tau, dim=-1)

    if scale_student:
        s_stu = _zscore(s_stu)
    log_p_stu = F.log_softmax(s_stu / tau, dim=-1)
    return F.kl_div(log_p_stu, p_ref, reduction="batchmean") * (tau ** 2)


def train_student(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_cf_txt_dataset(os.path.join(args.data_root, args.dataset))
    fair_data = (
        load_cf_txt_dataset(os.path.join(args.fair_data_root, args.dataset))
        if args.fair_data_root
        else None
    )

    num_users = data["num_users"]
    num_items = data["num_items"]
    num_nodes = num_users + num_items
    adj = build_sparse_adj_matrix(
        data["train_users"], data["train_items"], num_users, num_items
    ).coalesce().to(device)
    adj = _maybe_to_csr(adj, num_nodes, args.dim, device)

    u_evil, i_evil = load_frozen_teacher(args.evil_ckpt, adj, num_users, device)
    u_good, i_good = load_frozen_teacher(args.good_ckpt, adj, num_users, device)
    tt = build_third_teacher(
        u_good,
        i_good,
        u_evil,
        i_evil,
        ref_mode=args.ref_mode,
        gamma=args.gamma,
        train_sets=data["train_sets"],
        topk_ref=args.topk_ref,
        topk_evil=args.topk_evil,
    )
    ref_topk, evil_topk = tt["ref_topk"], tt["evil_topk"]

    student = EvilLightGCN(num_nodes=num_nodes, embedding_dim=args.dim, num_layers=args.layers).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)

    sampler = FastLightGCNSampler(
        data["train_sets"], num_users, num_items, data["item_pop"], device=device, alpha=0.0
    )

    os.makedirs(args.save_dir, exist_ok=True)
    best_recall = -1.0
    best_ckpt_path = os.path.join(args.save_dir, f"student_third_teacher_{args.dataset}_best.pt")

    print(
        f"[Student-ThirdTeacher] dataset={args.dataset} ref_mode={args.ref_mode} "
        f"gamma={args.gamma} kd_weight={args.kd_weight} bpr_weight={args.bpr_weight} "
        f"tau={args.tau} scale_student={args.scale_student} device={device}"
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        student.train()

        users_all, pos_all, neg_all = sampler.sample()
        perm = torch.randperm(len(users_all), device=device)
        users_t = users_all[perm]
        pos_t = pos_all[perm]
        neg_t = neg_all[perm]

        total_loss = 0.0
        total_kd = 0.0
        num_batches = (len(users_t) + args.batch_size - 1) // args.batch_size

        for start in range(0, len(users_t), args.batch_size):
            end = start + args.batch_size
            u = users_t[start:end]
            pos_i = pos_t[start:end]
            neg_i = neg_t[start:end]
            pos_idx = pos_i + num_users
            neg_idx = neg_i + num_users

            all_emb = student(adj)
            u_all_stu, i_all_stu = all_emb[:num_users], all_emb[num_users:]
            u_e = u_all_stu[u]
            pos_e = i_all_stu[pos_i]
            neg_e = i_all_stu[neg_i]

            pos_scores = (u_e * pos_e).sum(dim=1)
            neg_scores = (u_e * neg_e).sum(dim=1)
            bpr_loss = F.softplus(neg_scores - pos_scores).mean()

            reg_loss = 0.5 * (
                student.embedding.weight[u].pow(2).sum()
                + student.embedding.weight[pos_idx].pow(2).sum()
                + student.embedding.weight[neg_idx].pow(2).sum()
            ) / float(len(u))

            cand_items = sample_candidates(
                u,
                pos_i,
                neg_i,
                num_items,
                ref_topk,
                evil_topk,
                args.n_ref_sample,
                args.n_evil_sample,
                args.num_cands,
                device,
            )
            kd_loss = kd_third_teacher(
                u,
                cand_items,
                u_e,
                i_all_stu,
                tt,
                args.ref_mode,
                args.gamma,
                args.tau,
                scale_student=args.scale_student,
            )

            loss = args.bpr_weight * bpr_loss + args.reg * reg_loss + args.kd_weight * kd_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_kd += kd_loss.item()

        dt = time.time() - t0
        avg_loss = total_loss / max(1, num_batches)
        avg_kd = total_kd / max(1, num_batches)

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            student.eval()
            with torch.no_grad():
                final_emb = student(adj)
                u_emb, i_emb = final_emb[:num_users], final_emb[num_users:]
                res_main = evaluate_cf(
                    u_emb, i_emb, data["train_sets"], data["test_sets"], data["item_pop"], k=args.topk
                )
                main_str = " ".join(f"{k}={v:.4f}" for k, v in res_main.items())

                if fair_data is not None:
                    res_fair = evaluate_cf(
                        u_emb,
                        i_emb,
                        fair_data["train_sets"],
                        fair_data["test_sets"],
                        fair_data["item_pop"],
                        k=args.topk,
                    )
                    fair_str = " ".join(f"{k}={v:.4f}" for k, v in res_fair.items())
                    msg = f"Epoch {epoch:04d} ({dt:.1f}s) | loss={avg_loss:.4f} kd={avg_kd:.4f} | IID: {main_str} | Fair: {fair_str}"
                    tracked_recall = res_fair[f"Recall@{args.topk}"]
                else:
                    msg = f"Epoch {epoch:04d} ({dt:.1f}s) | loss={avg_loss:.4f} kd={avg_kd:.4f} | Eval: {main_str}"
                    tracked_recall = res_main[f"Recall@{args.topk}"]

                if tracked_recall > best_recall:
                    best_recall = tracked_recall
                    torch.save({"state_dict": student.state_dict(), "args": vars(args)}, best_ckpt_path)
                    msg += " *"
                print(msg)
        else:
            print(f"Epoch {epoch:04d} ({dt:.1f}s) | loss={avg_loss:.4f} kd={avg_kd:.4f}")

    ckpt_path = os.path.join(args.save_dir, f"student_third_teacher_{args.dataset}.pt")
    torch.save({"state_dict": student.state_dict(), "args": vars(args)}, ckpt_path)
    print(f"Saved Student checkpoint to {ckpt_path} (best Recall@{args.topk}={best_recall:.4f} at {best_ckpt_path})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Debiased Student with Third-Teacher KD")
    parser.add_argument("--data_root", type=str, default="PopBias/Data")
    parser.add_argument("--fair_data_root", type=str, default="")
    parser.add_argument("--dataset", type=str, default="gowalla", choices=["gowalla", "yelp2018"])
    parser.add_argument("--good_ckpt", type=str, default="checkpoints_fair_2k/baseline_gowalla.pt")
    parser.add_argument("--evil_ckpt", type=str, default="checkpoints_fair_2k/evil_a1.0_b1.0_gowalla.pt")
    parser.add_argument("--ref_mode", type=str, default="orth", choices=list(VALID_REF_MODES))
    parser.add_argument("--gamma", type=float, default=1.0, help="Evil subtraction weight for Third Teacher")
    parser.add_argument("--tau", type=float, default=0.2, help="Temperature for Third-Teacher KL")
    parser.add_argument("--kd_weight", type=float, default=1.5)
    parser.add_argument("--bpr_weight", type=float, default=0.5)
    parser.add_argument("--topk_ref", type=int, default=64, help="Top-K items mined from Third Teacher per user")
    parser.add_argument("--topk_evil", type=int, default=32, help="Top-K bias traps mined from Evil Teacher per user")
    parser.add_argument("--n_ref_sample", type=int, default=32, help="Ref-teacher hard positives sampled per batch")
    parser.add_argument("--n_evil_sample", type=int, default=16, help="Evil-teacher bias traps sampled per batch")
    parser.add_argument("--num_cands", type=int, default=64, help="Extra random background candidates per batch")
    parser.add_argument("--scale_student", action="store_true", default=True, help="Z-score student and ref logits before KL")
    parser.add_argument("--no_scale_student", dest="scale_student", action="store_false")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--save_dir", type=str, default="checkpoints_fair_v2")
    train_student(parser.parse_args())
