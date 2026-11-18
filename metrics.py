import math
import torch
import numpy as np


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
    Evaluates full-ranking accuracy (Recall@K, NDCG@K) and popularity bias
    (ARP@K, Tail@K, Gini@K, PopCorr) on test users.
    """
    device = i_emb.device
    num_items = i_emb.size(0)
    tail_mask = _tail_item_mask(item_pop).to(device)
    pop_cpu = item_pop.cpu().numpy()
    log_pop = torch.log1p(item_pop.to(device))

    test_users = sorted(test_sets.keys())
    recalls, ndcgs, arps, tails = [], [], [], []
    item_rec_counts = np.zeros(num_items, dtype=np.int64)
    pop_corrs = []

    idcg_table = np.array([1.0 / math.log2(r + 2) for r in range(k)])

    for start in range(0, len(test_users), batch_size):
        batch_u = test_users[start : start + batch_size]
        u_tensor = torch.tensor(batch_u, dtype=torch.long, device=device)
        if adapter is not None:
            u_mapped = adapter._map_users(u_tensor)
            ue = u_emb[u_mapped]
        else:
            ue = u_emb[u_tensor]

        scores = torch.matmul(ue, i_emb.t())  # [B, num_items]

        # Track correlation with log(pop + 1) on unmasked scores
        s_centered = scores - scores.mean(dim=1, keepdim=True)
        p_centered = log_pop - log_pop.mean()
        denom = (s_centered.norm(dim=1) * p_centered.norm()).clamp(min=1e-8)
        corr = (s_centered * p_centered.unsqueeze(0)).sum(dim=1) / denom
        pop_corrs.extend(corr.cpu().tolist())

        # Mask training items
        for row_idx, u in enumerate(batch_u):
            seen = train_sets.get(u)
            if seen:
                scores[row_idx, list(seen)] = -1e9

        _, topk_idx = torch.topk(scores, k=k, dim=1)  # [B, K]
        topk_cpu = topk_idx.cpu().numpy()
        tail_hits = tail_mask[topk_idx].float().mean(dim=1).cpu().numpy()

        for row_idx, u in enumerate(batch_u):
            gt = test_sets[u]
            preds = topk_cpu[row_idx]
            hits = np.array([1.0 if i in gt else 0.0 for i in preds])

            recalls.append(hits.sum() / len(gt))
            dcg = (hits * idcg_table).sum()
            ideal_len = min(len(gt), k)
            idcg = idcg_table[:ideal_len].sum()
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

            arps.append(pop_cpu[preds].mean())
            tails.append(float(tail_hits[row_idx]))
            np.add.at(item_rec_counts, preds, 1)

    # Gini index over recommendation counts (higher = more concentrated / biased)
    sorted_counts = np.sort(item_rec_counts).astype(np.float64)
    total = sorted_counts.sum()
    if total > 0:
        idx = np.arange(1, num_items + 1)
        gini = ((2 * idx - num_items - 1) * sorted_counts).sum() / (num_items * total)
    else:
        gini = 0.0

    return {
        f"Recall@{k}": float(np.mean(recalls)),
        f"NDCG@{k}": float(np.mean(ndcgs)),
        f"ARP@{k}": float(np.mean(arps)),
        f"Tail@{k}": float(np.mean(tails)),
        f"Gini@{k}": float(gini),
        "PopCorr": float(np.mean(pop_corrs)),
    }
