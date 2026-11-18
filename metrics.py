import importlib.util
import os
import numpy as np
import torch


def _load_tten_metrics():
    tten_utils_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "PopBias",
        "Baselines",
        "TTEN",
        "utils.py",
    )
    spec = importlib.util.spec_from_file_location("tten_utils", tten_utils_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.metrics


_tten_metrics = None


def _tail_item_mask(item_pop, head_ratio=0.2):
    """Returns boolean mask where True = tail item (bottom 80% by interactions)."""
    num_items = len(item_pop)
    k_head = max(1, int(num_items * head_ratio))
    _, head_idx = torch.topk(item_pop, k=k_head)
    tail_mask = torch.ones(num_items, dtype=torch.bool)
    tail_mask[head_idx] = False
    return tail_mask


@torch.no_grad()
def evaluate_cf(
    u_emb,
    i_emb,
    train_sets,
    test_sets,
    item_pop,
    adapter=None,
    k=20,
    batch_size=1024,
):
    """
    Evaluates HR@K, Recall@K, NDCG@K via TTEN's official evaluator
    (PopBias/Baselines/TTEN/utils.py) alongside ARP@K and Tail@K.
    """
    global _tten_metrics
    if _tten_metrics is None:
        _tten_metrics = _load_tten_metrics()

    device = i_emb.device
    tail_mask = _tail_item_mask(item_pop).to(device)
    pop_cpu = item_pop.cpu().numpy()

    test_users = sorted(test_sets.keys())
    all_user_num = 0
    all_hr = 0.0
    all_recall = 0.0
    all_ndcg = 0.0
    arps, tails = [], []

    for start in range(0, len(test_users), batch_size):
        batch_u = test_users[start : start + batch_size]
        u_tensor = torch.tensor(batch_u, dtype=torch.long, device=device)
        ue = u_emb[adapter._map_users(u_tensor)] if adapter is not None else u_emb[u_tensor]

        scores = torch.matmul(ue, i_emb.t())  # [B, num_items]
        for row_idx, u in enumerate(batch_u):
            seen = train_sets.get(u)
            if seen:
                scores[row_idx, list(seen)] = -1e9

        _, topk_idx = torch.topk(scores, k=k, dim=1)
        topk_cpu = topk_idx.cpu().numpy()

        user_num, _, _, hr_sum, rec_sum, ndcg_sum = _tten_metrics(
            batch_u, topk_cpu, k, test_sets
        )
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
