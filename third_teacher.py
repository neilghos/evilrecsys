import numpy as np
import torch
import torch.nn.functional as F

from evil_model import EvilLightGCN
from metrics import _load_tten_metrics, _tail_item_mask


VALID_REF_MODES = ("raw", "zscore", "orth", "silu", "swiglu")


@torch.no_grad()
def load_frozen_teacher(ckpt_path, adj, num_users, device):
    """Loads a frozen EvilLightGCN checkpoint and returns (u_emb, i_emb)."""
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


def _mask_seen_items_(scores, user_ids, train_sets):
    if train_sets is None:
        return
    for row_idx, u in enumerate(user_ids):
        seen = train_sets.get(u)
        if seen:
            scores[row_idx, list(seen)] = -1e9


def compute_ref_scores(
    s_good,
    s_evil,
    ref_mode,
    gamma,
    mean_g=None,
    std_g=None,
    mean_e=None,
    std_e=None,
    beta_u=None,
):
    """
    Computes Third-Teacher scores from (s_good, s_evil) for linear or gated modes:
      - raw:    s_good - gamma * s_evil
      - zscore: z(s_good) - gamma * z(s_evil)
      - orth:   s_good - gamma * beta_u * s_evil
      - silu:   z(s_good) - gamma * SiLU(z(s_evil))
      - swiglu: z(s_good) - gamma * z(s_evil) * sigmoid(z(s_evil) - z(s_good))
    """
    if ref_mode == "raw":
        return s_good - gamma * s_evil
    if ref_mode == "orth":
        return s_good - (gamma * beta_u.unsqueeze(1)) * s_evil

    zg = (s_good - mean_g.unsqueeze(1)) / std_g.unsqueeze(1)
    ze = (s_evil - mean_e.unsqueeze(1)) / std_e.unsqueeze(1)
    if ref_mode == "zscore":
        return zg - gamma * ze
    if ref_mode == "silu":
        return zg - gamma * F.silu(ze)
    if ref_mode == "swiglu":
        return zg - gamma * ze * torch.sigmoid(ze - zg)
    raise ValueError(f"Unsupported ref_mode: {ref_mode}")


@torch.no_grad()
def build_third_teacher(
    u_good,
    i_good,
    u_evil,
    i_evil,
    ref_mode="orth",
    gamma=0.8,
    train_sets=None,
    topk_ref=0,
    topk_evil=0,
    batch_size=1024,
):
    """
    Computes exact full-catalog per-user statistics for the Third Teacher and
    optionally mines top-K hard candidates (ref_topk, evil_topk) per user.
    """
    if ref_mode not in VALID_REF_MODES:
        raise ValueError(f"Unsupported ref_mode: {ref_mode}")

    num_users = u_good.size(0)
    device = u_good.device

    mean_g = torch.empty(num_users, device=device)
    std_g = torch.empty(num_users, device=device)
    mean_e = torch.empty(num_users, device=device)
    std_e = torch.empty(num_users, device=device)
    beta_u = torch.empty(num_users, device=device)

    for start in range(0, num_users, batch_size):
        end = min(start + batch_size, num_users)
        s_g = torch.matmul(u_good[start:end], i_good.t())
        s_e = torch.matmul(u_evil[start:end], i_evil.t())
        mg = s_g.mean(dim=1)
        me = s_e.mean(dim=1)
        dg = s_g - mg.unsqueeze(1)
        de = s_e - me.unsqueeze(1)

        v_e = de.pow(2).mean(dim=1).clamp(min=1e-6)
        mean_g[start:end] = mg
        mean_e[start:end] = me
        std_g[start:end] = dg.pow(2).mean(dim=1).sqrt().clamp(min=1e-6)
        std_e[start:end] = v_e.sqrt().clamp(min=1e-6)
        beta_u[start:end] = ((dg * de).mean(dim=1) / v_e).clamp(min=0.0)

    ref_topk = (
        torch.empty((num_users, topk_ref), dtype=torch.long, device=device)
        if topk_ref > 0
        else None
    )
    evil_topk = (
        torch.empty((num_users, topk_evil), dtype=torch.long, device=device)
        if topk_evil > 0
        else None
    )

    if ref_topk is not None or evil_topk is not None:
        for start in range(0, num_users, batch_size):
            end = min(start + batch_size, num_users)
            user_ids = range(start, end)
            if ref_topk is not None:
                s_g = torch.matmul(u_good[start:end], i_good.t())
                s_e = torch.matmul(u_evil[start:end], i_evil.t())
                s_ref = compute_ref_scores(
                    s_g, s_e, ref_mode, gamma,
                    mean_g=mean_g[start:end], std_g=std_g[start:end],
                    mean_e=mean_e[start:end], std_e=std_e[start:end],
                    beta_u=beta_u[start:end],
                )
                _mask_seen_items_(s_ref, user_ids, train_sets)
                ref_topk[start:end] = torch.topk(s_ref, k=topk_ref, dim=1).indices

            if evil_topk is not None:
                s_e = torch.matmul(u_evil[start:end], i_evil.t())
                _mask_seen_items_(s_e, user_ids, train_sets)
                evil_topk[start:end] = torch.topk(s_e, k=topk_evil, dim=1).indices

    return {
        "u_good": u_good,
        "i_good": i_good,
        "u_evil": u_evil,
        "i_evil": i_evil,
        "mean_g": mean_g,
        "std_g": std_g,
        "mean_e": mean_e,
        "std_e": std_e,
        "beta_u": beta_u,
        "ref_topk": ref_topk,
        "evil_topk": evil_topk,
    }


@torch.no_grad()
def evaluate_third_teacher(tt, train_sets, test_sets, item_pop, ref_mode, gamma, k=20, batch_size=1024):
    """Evaluates any linear or gated Third Teacher directly via TTEN's official evaluator."""
    tten_metrics = _load_tten_metrics()
    device = tt["i_good"].device
    tail_mask = _tail_item_mask(item_pop).to(device)
    pop_cpu = item_pop.cpu().numpy()

    test_users = sorted(test_sets.keys())
    all_user_num, all_hr, all_recall, all_ndcg = 0, 0.0, 0.0, 0.0
    arps, tails = [], []

    for start in range(0, len(test_users), batch_size):
        batch_u = test_users[start : start + batch_size]
        u_t = torch.tensor(batch_u, dtype=torch.long, device=device)

        s_g = torch.matmul(tt["u_good"][u_t], tt["i_good"].t())
        s_e = torch.matmul(tt["u_evil"][u_t], tt["i_evil"].t())
        scores = compute_ref_scores(
            s_g, s_e, ref_mode, gamma,
            mean_g=tt["mean_g"][u_t], std_g=tt["std_g"][u_t],
            mean_e=tt["mean_e"][u_t], std_e=tt["std_e"][u_t],
            beta_u=tt["beta_u"][u_t],
        )
        _mask_seen_items_(scores, batch_u, train_sets)

        _, topk_idx = torch.topk(scores, k=k, dim=1)
        topk_cpu = topk_idx.cpu().numpy()

        user_num, _, _, hr_sum, rec_sum, ndcg_sum = tten_metrics(batch_u, topk_cpu, k, test_sets)
        all_user_num += user_num
        all_hr += hr_sum
        all_recall += rec_sum
        all_ndcg += ndcg_sum

        tail_hits = tail_mask[topk_idx].float().mean(dim=1).cpu().numpy()
        for row_idx in range(len(batch_u)):
            arps.append(pop_cpu[topk_cpu[row_idx]].mean())
            tails.append(float(tail_hits[row_idx]))

    denom = max(1, all_user_num)
    return {
        f"HR@{k}": float(all_hr / denom),
        f"Recall@{k}": float(all_recall / denom),
        f"NDCG@{k}": float(all_ndcg / denom),
        f"ARP@{k}": float(np.mean(arps)),
        f"Tail@{k}": float(np.mean(tails)),
    }
