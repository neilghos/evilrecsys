import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from dataloaders import build_sparse_adj_matrix, maybe_to_csr
from evil_model import FastLightGCNSampler, LightGCN, seed_everything
from logger import RunLogger
from teacher import (
    build_third_teacher,
    compute_ref_scores,
    load_frozen_teacher,
    make_third_teacher_score_fn,
)


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


def kd_third_teacher(
    u,
    cand_items,
    u_stu,
    i_stu,
    tt,
    ref_mode,
    gamma,
    tau,
    n_ref_sample=0,
    n_evil_sample=0,
    mse_weight=1.0,
    margin_weight=0.5,
):
    """
    Tight multi-bound Neutral-Teacher KD over candidate items [B, M]:
      1. Listwise KL (temperature-sharpened ranking match)
      2. Raw Score MSE (locks student logit magnitudes to Neutral Teacher, preventing norm drift)
      3. Pairwise Margin Bound (forces Neutral-Teacher top items above Evil-Teacher bias traps)
    """
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
        )
        p_ref = F.softmax(_zscore(s_ref) / tau, dim=-1)

    kl_loss = F.kl_div(F.log_softmax(_zscore(s_stu) / tau, dim=-1), p_ref, reduction="batchmean") * (tau ** 2)
    mse_loss = F.mse_loss(s_stu, s_ref) if mse_weight > 0 else 0.0

    margin_loss = 0.0
    if margin_weight > 0 and n_ref_sample > 0 and n_evil_sample > 0:
        # cand_items layout: [pos_i (1), neg_i (1), ref_cands (n_ref_sample), evil_cands (n_evil_sample), rand]
        ref_start = 2
        evil_start = 2 + n_ref_sample
        s_ref_stu = s_stu[:, ref_start : ref_start + n_ref_sample]
        s_evil_stu = s_stu[:, evil_start : evil_start + n_evil_sample]
        # Pairwise softplus pushing every mined neutral top item above every evil trap item
        diff = s_evil_stu.unsqueeze(1) - s_ref_stu.unsqueeze(2)  # [B, n_ref, n_evil]
        margin_loss = F.softplus(diff).mean()

    return kl_loss + mse_weight * mse_loss + margin_weight * margin_loss


class StudentKDTrainer:
    """
    Reusable Neutral-Teacher + Student KD trainer shared across all bias folders.
    1. Builds the gated Neutral Teacher (silu / swiglu) from Good + Evil checkpoints
       and evaluates it directly using the bias folder's eval_fn.
    2. Distills into a fresh LightGCN student with BPR + listwise KL over mined hard candidates.
    3. Saves and directly evaluates the best student checkpoint at the end of training.
    """
    def __init__(self, args, data, eval_fn, summary_csv="results/summary.csv"):
        self.args = args
        self.data = data
        self.eval_fn = eval_fn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.summary_csv = summary_csv

    def select_best_teacher(self, u_good, i_good, u_evil, i_evil, select_metric=None, gamma_min=0.0, gamma_max=2.0, gamma_step=0.05):
        """
        Sweeps (silu, swiglu) x gamma in [gamma_min, gamma_max] with step gamma_step
        and returns (best_mode, best_gamma, best_res).
        """
        from teacher import VALID_REF_MODES
        args = self.args
        metric_key = select_metric or f"Recall@{args.topk}"
        mask_sets = self.data.get("mask_sets", self.data["train_sets"])

        # Precompute full-catalog per-user statistics once
        tt = build_third_teacher(u_good, i_good, u_evil, i_evil, ref_mode="swiglu", gamma=1.0, mask_sets=mask_sets)

        num_steps = int(round((gamma_max - gamma_min) / gamma_step)) + 1
        gammas = [round(gamma_min + i * gamma_step, 4) for i in range(num_steps)]

        best_mode, best_gamma, best_score, best_res = "swiglu", 1.0, -float("inf"), None
        print(f"[Auto-Select] Sweeping {VALID_REF_MODES} x {len(gammas)} gammas [{gamma_min}..{gamma_max}, step={gamma_step}] by {metric_key}")

        for mode in VALID_REF_MODES:
            for g in gammas:
                res = self.eval_fn(make_third_teacher_score_fn(tt, mode, g), self.data, args.topk)
                score = res[metric_key]
                if score > best_score:
                    best_score = score
                    best_mode, best_gamma, best_res = mode, g, res
                    print(f"  * {mode:<6} gamma={g:4.2f} | " + " ".join(f"{k}={v:.4f}" for k, v in res.items()))

        print(
            f"[Auto-Select Winner] mode={best_mode} gamma={best_gamma:.2f} | "
            + " ".join(f"{k}={v:.4f}" for k, v in best_res.items())
        )
        return best_mode, best_gamma, best_res

    def train(self, good_ckpt, evil_ckpt, ckpt_path, adj_evil=None, select_metric=None, auto_select=False, evil_embs=None):
        args = self.args
        seed_everything(args.seed)

        num_users, num_items = self.data["num_users"], self.data["num_items"]
        num_nodes = num_users + num_items
        mask_sets = self.data.get("mask_sets", self.data["train_sets"])

        adj_good = build_sparse_adj_matrix(
            self.data["train_users"], self.data["train_items"], num_users, num_items
        ).coalesce().to(self.device)
        adj_good = maybe_to_csr(adj_good, num_nodes, args.dim, self.device)
        adj_e = adj_good if adj_evil is None else adj_evil

        u_good, i_good = load_frozen_teacher(good_ckpt, adj_good, num_users, self.device)
        if evil_embs is not None:
            u_evil, i_evil = evil_embs
        else:
            u_evil, i_evil = load_frozen_teacher(evil_ckpt, adj_e, num_users, self.device)

        if auto_select:
            ref_mode, gamma, neutral_res = self.select_best_teacher(
                u_good, i_good, u_evil, i_evil, select_metric=select_metric
            )
            args.ref_mode = ref_mode
            args.gamma = gamma
        else:
            ref_mode, gamma, neutral_res = args.ref_mode, args.gamma, None

        tt = build_third_teacher(
            u_good,
            i_good,
            u_evil,
            i_evil,
            ref_mode=ref_mode,
            gamma=gamma,
            mask_sets=mask_sets,
            topk_ref=args.topk_ref,
            topk_evil=args.topk_evil,
        )

        run_name = f"student_{ref_mode}_g{gamma}_{args.dataset}"
        logger = RunLogger(run_name, args.dataset, f"Student-{ref_mode}", summary_csv=self.summary_csv)

        # Evaluate Neutral Teacher directly before distillation (or reuse auto_select result)
        if neutral_res is None:
            neutral_res = self.eval_fn(make_third_teacher_score_fn(tt, ref_mode, gamma), self.data, args.topk)
        logger.info(
            f"[Neutral-Teacher({ref_mode}, g={gamma})] "
            + " ".join(f"{k}={v:.4f}" for k, v in neutral_res.items())
        )
        logger.log_summary(neutral_res, tag="neutral_teacher")

        student = LightGCN(num_nodes=num_nodes, embedding_dim=args.dim, num_layers=args.layers).to(self.device)
        optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
        sampler = FastLightGCNSampler(self.data["train_sets"], num_users, num_items, device=self.device)

        os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
        metric_key = select_metric or f"Recall@{args.topk}"
        best_score = -float("inf")
        best_res = None

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            student.train()

            u_all, p_all, n_all = sampler.sample()
            perm = torch.randperm(len(u_all), device=self.device)
            u_t, p_t, n_t = u_all[perm], p_all[perm], n_all[perm]

            total_loss, total_kd = 0.0, 0.0
            num_batches = max(1, (len(u_t) + args.batch_size - 1) // args.batch_size)

            for start in range(0, len(u_t), args.batch_size):
                end = start + args.batch_size
                u, pos_i, neg_i = u_t[start:end], p_t[start:end], n_t[start:end]
                pos_idx, neg_idx = pos_i + num_users, neg_i + num_users

                all_emb = student(adj_good)
                u_all_stu, i_all_stu = all_emb[:num_users], all_emb[num_users:]
                u_e, pos_e, neg_e = u_all_stu[u], i_all_stu[pos_i], i_all_stu[neg_i]

                bpr_loss = F.softplus((u_e * neg_e).sum(dim=1) - (u_e * pos_e).sum(dim=1)).mean()
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
                    tt["ref_topk"],
                    tt["evil_topk"],
                    args.n_ref_sample,
                    args.n_evil_sample,
                    args.num_cands,
                    self.device,
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
                    n_ref_sample=args.n_ref_sample if tt["ref_topk"] is not None else 0,
                    n_evil_sample=args.n_evil_sample if tt["evil_topk"] is not None else 0,
                    mse_weight=getattr(args, "mse_weight", 1.0),
                    margin_weight=getattr(args, "margin_weight", 0.5),
                )

                loss = args.bpr_weight * bpr_loss + args.reg * reg_loss + args.kd_weight * kd_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_kd += kd_loss.item()

            dt = time.time() - t0
            avg_loss = total_loss / num_batches
            avg_kd = total_kd / num_batches

            if epoch % args.eval_every == 0 or epoch == args.epochs:
                student.eval()
                with torch.no_grad():
                    emb = student(adj_good)
                    u_emb, i_emb = emb[:num_users], emb[num_users:]
                    res = self.eval_fn(lambda u_idx: torch.matmul(u_emb[u_idx], i_emb.t()), self.data, args.topk)
                logger.log_epoch(epoch, dt, avg_loss, res, extra=f"kd={avg_kd:.4f}")

                if res[metric_key] > best_score:
                    best_score = res[metric_key]
                    best_res = res
                    torch.save(
                        {"state_dict": student.state_dict(), "args": vars(args), "num_nodes": num_nodes},
                        ckpt_path,
                    )
            else:
                logger.log_epoch(epoch, dt, avg_loss, extra=f"kd={avg_kd:.4f}")

        # Directly reload and evaluate the best checkpoint at the end of training
        u_best, i_best = load_frozen_teacher(ckpt_path, adj_good, num_users, self.device)
        final_best_res = self.eval_fn(lambda u_idx: torch.matmul(u_best[u_idx], i_best.t()), self.data, args.topk)
        logger.info(
            f"[Best-Student] ({ckpt_path}) "
            + " ".join(f"{k}={v:.4f}" for k, v in final_best_res.items())
        )
        logger.log_summary(final_best_res, tag="best_student")
        return ckpt_path, final_best_res
