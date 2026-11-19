import torch
import torch.nn.functional as F

from evil_model import LightGCN


VALID_REF_MODES = ("silu", "swiglu")


@torch.no_grad()
def load_frozen_teacher(ckpt_path, adj, num_users, device):
    """Loads a frozen LightGCN checkpoint and returns (u_emb, i_emb)."""
    ckpt = torch.load(ckpt_path, map_location=device)
    c_args = ckpt["args"]
    num_nodes = ckpt.get("num_nodes", ckpt["state_dict"]["embedding.weight"].size(0))
    teacher = LightGCN(
        num_nodes=num_nodes,
        embedding_dim=c_args["dim"],
        num_layers=c_args["layers"],
    ).to(device)
    teacher.load_state_dict(ckpt["state_dict"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    emb = teacher(adj).detach()
    return emb[:num_users], emb[num_users:]


def _mask_seen_items_(scores, user_ids, mask_sets):
    if mask_sets is None:
        return
    for row_idx, u in enumerate(user_ids):
        seen = mask_sets.get(u)
        if seen:
            scores[row_idx, list(seen)] = -1e9


def compute_ref_scores(s_good, s_evil, ref_mode, gamma, mean_g, std_g, mean_e, std_e):
    """
    Computes gated Neutral-Teacher scores from standardized (s_good, s_evil):
      - silu:   z(s_good) - gamma * SiLU(z(s_evil))
      - swiglu: z(s_good) - gamma * z(s_evil) * sigmoid(z(s_evil) - z(s_good))
    """
    zg = (s_good - mean_g.unsqueeze(1)) / std_g.unsqueeze(1)
    ze = (s_evil - mean_e.unsqueeze(1)) / std_e.unsqueeze(1)
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
    ref_mode="swiglu",
    gamma=1.2,
    mask_sets=None,
    topk_ref=0,
    topk_evil=0,
    batch_size=1024,
):
    """
    Computes exact full-catalog per-user statistics for the gated Neutral Teacher
    and optionally mines top-K hard candidates (ref_topk, evil_topk) per user.
    """
    if ref_mode not in VALID_REF_MODES:
        raise ValueError(f"Unsupported ref_mode: {ref_mode}")

    num_users = u_good.size(0)
    device = u_good.device

    mean_g = torch.empty(num_users, device=device)
    std_g = torch.empty(num_users, device=device)
    mean_e = torch.empty(num_users, device=device)
    std_e = torch.empty(num_users, device=device)

    for start in range(0, num_users, batch_size):
        end = min(start + batch_size, num_users)
        s_g = torch.matmul(u_good[start:end], i_good.t())
        s_e = torch.matmul(u_evil[start:end], i_evil.t())
        mg = s_g.mean(dim=1)
        me = s_e.mean(dim=1)
        dg = s_g - mg.unsqueeze(1)
        de = s_e - me.unsqueeze(1)

        mean_g[start:end] = mg
        mean_e[start:end] = me
        std_g[start:end] = dg.pow(2).mean(dim=1).sqrt().clamp(min=1e-6)
        std_e[start:end] = de.pow(2).mean(dim=1).sqrt().clamp(min=1e-6)

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
                    s_g,
                    s_e,
                    ref_mode,
                    gamma,
                    mean_g[start:end],
                    std_g[start:end],
                    mean_e[start:end],
                    std_e[start:end],
                )
                _mask_seen_items_(s_ref, user_ids, mask_sets)
                ref_topk[start:end] = torch.topk(s_ref, k=topk_ref, dim=1).indices

            if evil_topk is not None:
                s_e = torch.matmul(u_evil[start:end], i_evil.t())
                _mask_seen_items_(s_e, user_ids, mask_sets)
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
        "ref_topk": ref_topk,
        "evil_topk": evil_topk,
    }


def make_third_teacher_score_fn(tt, ref_mode, gamma):
    """Returns a batched score_fn(u_tensor) -> [B, num_items] for any bias metric suite."""
    @torch.no_grad()
    def _score_fn(u_t):
        s_g = torch.matmul(tt["u_good"][u_t], tt["i_good"].t())
        s_e = torch.matmul(tt["u_evil"][u_t], tt["i_evil"].t())
        return compute_ref_scores(
            s_g,
            s_e,
            ref_mode,
            gamma,
            tt["mean_g"][u_t],
            tt["std_g"][u_t],
            tt["mean_e"][u_t],
            tt["std_e"][u_t],
        )
    return _score_fn
