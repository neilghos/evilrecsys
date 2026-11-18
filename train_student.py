import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloaders import build_sparse_adj_matrix, load_cf_txt_dataset
from evil_model import EvilLightGCN
from metrics import evaluate_cf
from train_evil import FastLightGCNSampler, _maybe_to_csr


class ScoreDiscriminator(nn.Module):
    """Lightweight MLP discriminator for Adversarial KD."""
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


@torch.no_grad()
def load_frozen_teacher(ckpt_path, adj, num_users, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    c_args = ckpt["args"]
    teacher = EvilLightGCN(
        num_nodes=ckpt["num_nodes"],
        embedding_dim=c_args["dim"],
        num_layers=c_args["layers"],
    ).to(device)
    teacher.load_state_dict(ckpt["state_dict"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    emb = teacher(adj).detach()
    return emb[:num_users], emb[num_users:]


def kd_third_teacher(u, cand_items, u_stu, i_stu, u_good, i_good, u_evil, i_evil, gamma, tau):
    """
    Distills evil-referenced teacher scores: s_ref = s_good - gamma * s_evil
    into the student over candidate items [B, M].
    """
    # u_*: [B, D], i_*[cand_items]: [B, M, D]
    s_stu = torch.bmm(i_stu[cand_items], u_stu.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        s_good = torch.bmm(i_good[cand_items], u_good[u].unsqueeze(-1)).squeeze(-1)
        s_evil = torch.bmm(i_evil[cand_items], u_evil[u].unsqueeze(-1)).squeeze(-1)
        s_ref = s_good - gamma * s_evil
        p_ref = F.softmax(s_ref / tau, dim=-1)

    log_p_stu = F.log_softmax(s_stu / tau, dim=-1)
    return F.kl_div(log_p_stu, p_ref, reduction="batchmean") * (tau ** 2)


def kd_contrastive(u, u_stu, u_good, u_evil, tau):
    """
    InfoNCE contrastive KD: pulls normalized student user embedding toward
    good teacher u_good[u], and pushes away from evil teacher u_evil[u]
    plus in-batch negatives.
    """
    z_stu = F.normalize(u_stu, dim=-1)
    z_pos = F.normalize(u_good[u], dim=-1)
    z_evil = F.normalize(u_evil[u], dim=-1)

    pos_sim = (z_stu * z_pos).sum(dim=-1, keepdim=True) / tau          # [B, 1]
    evil_sim = (z_stu * z_evil).sum(dim=-1, keepdim=True) / tau        # [B, 1]
    inbatch_sim = torch.matmul(z_stu, z_pos.t()) / tau                 # [B, B]
    # Mask self-similarity on diagonal so pos isn't double-counted in denominator
    eye_mask = torch.eye(z_stu.size(0), dtype=torch.bool, device=z_stu.device)
    inbatch_sim = inbatch_sim.masked_fill(eye_mask, -1e9)

    logits = torch.cat([pos_sim, evil_sim, inbatch_sim], dim=1)        # [B, 2 + B]
    targets = torch.zeros(z_stu.size(0), dtype=torch.long, device=z_stu.device)
    return F.cross_entropy(logits, targets)


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

    student = EvilLightGCN(num_nodes=num_nodes, embedding_dim=args.dim, num_layers=args.layers).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)

    disc = None
    disc_opt = None
    if args.method == "adversarial":
        disc = ScoreDiscriminator(args.dim).to(device)
        disc_opt = torch.optim.Adam(disc.parameters(), lr=args.lr)

    sampler = FastLightGCNSampler(
        data["train_sets"], num_users, num_items, data["item_pop"], device=device, alpha=0.0
    )

    print(
        f"[Student-{args.method}] dataset={args.dataset} kd_weight={args.kd_weight} "
        f"gamma={args.gamma} tau={args.tau} device={device}"
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

            if args.method == "third_teacher":
                rand_cands = torch.randint(0, num_items, (len(u), args.num_cands), device=device)
                cand_items = torch.cat([pos_i.unsqueeze(1), neg_i.unsqueeze(1), rand_cands], dim=1)
                kd_loss = kd_third_teacher(
                    u, cand_items, u_e, i_all_stu, u_good, i_good, u_evil, i_evil, args.gamma, args.tau
                )
            elif args.method == "contrastive":
                kd_loss = kd_contrastive(u, u_e, u_good, u_evil, args.tau)
            elif args.method == "adversarial":
                # Step 1: train discriminator to separate Evil (1) from Good/Non-Evil (0)
                d_evil = disc(u_evil[u])
                d_good = disc(u_good[u])
                d_loss = 0.5 * (
                    F.binary_cross_entropy_with_logits(d_evil, torch.ones_like(d_evil))
                    + F.binary_cross_entropy_with_logits(d_good, torch.zeros_like(d_good))
                )
                disc_opt.zero_grad()
                d_loss.backward()
                disc_opt.step()

                # Step 2: student pushes its user representation to the non-evil side (0)
                d_stu = disc(u_e)
                kd_loss = F.binary_cross_entropy_with_logits(d_stu, torch.zeros_like(d_stu))
            else:
                raise ValueError(f"Unknown method: {args.method}")

            loss = bpr_loss + args.reg * reg_loss + args.kd_weight * kd_loss
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
                res_iid = evaluate_cf(
                    u_emb, i_emb, data["train_sets"], data["test_sets"], data["item_pop"], k=args.topk
                )
                iid_str = " ".join(f"{k}={v:.4f}" for k, v in res_iid.items())
                msg = f"Epoch {epoch:04d} ({dt:.1f}s) | loss={avg_loss:.4f} kd={avg_kd:.4f} | IID: {iid_str}"

                if fair_data is not None:
                    res_fair = evaluate_cf(
                        u_emb,
                        i_emb,
                        fair_data["train_sets"],
                        fair_data["test_sets"],
                        fair_data["item_pop"],
                        k=args.topk,
                    )
                    fair_str = f"Recall@{args.topk}={res_fair[f'Recall@{args.topk}']:.4f} NDCG@{args.topk}={res_fair[f'NDCG@{args.topk}']:.4f}"
                    msg += f" | Fair: {fair_str}"
                print(msg)
        else:
            print(f"Epoch {epoch:04d} ({dt:.1f}s) | loss={avg_loss:.4f} kd={avg_kd:.4f}")

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, f"student_{args.method}_{args.dataset}.pt")
    torch.save({"state_dict": student.state_dict(), "args": vars(args)}, ckpt_path)
    print(f"Saved Student checkpoint to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Debiased Student with Evil-Teacher KD")
    parser.add_argument("--data_root", type=str, default="PopBias/Baselines/LightGCN-PyTorch/data")
    parser.add_argument("--fair_data_root", type=str, default="PopBias/Data")
    parser.add_argument("--dataset", type=str, default="gowalla", choices=["gowalla", "yelp2018"])
    parser.add_argument("--good_ckpt", type=str, default="checkpoints/baseline_gowalla.pt")
    parser.add_argument("--evil_ckpt", type=str, default="checkpoints/evil_a1.0_b1.0_gowalla.pt")
    parser.add_argument("--method", type=str, default="third_teacher", choices=["third_teacher", "contrastive", "adversarial"])
    parser.add_argument("--kd_weight", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.5, help="Evil subtraction weight for third_teacher")
    parser.add_argument("--tau", type=float, default=0.2, help="Temperature for KD / InfoNCE")
    parser.add_argument("--num_cands", type=int, default=32, help="Extra random candidates per batch for third_teacher")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    train_student(parser.parse_args())
