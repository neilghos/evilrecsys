import math
import numpy as np
import torch


@torch.no_grad()
def evaluate_unfairness(score_fn, data, k=20, batch_size=1024):
    """
    Exact FDA (WWW'23) Top-K Utility + Demographic Fairness evaluator:
      - HR@K:        hits / min(|test_u|, K)  (FDA's HR/capped-Recall formula)
      - NDCG@K:      FDA's NDCG@K formula
      - DP@K:        Demographic Parity = mean over recommended items of |P(g=1|rec=v) - P(g=0|rec=v)| (lower = fairer)
      - EO@K:        Equal Opportunity  = mean over hit items of |P(g=1|hit=v) - P(g=0|hit=v)|         (lower = fairer)
      - Gap@K:       |NDCG@K(g=1) - NDCG@K(g=0)| (user-group utility disparity, lower = fairer)
      - FairScore@K: NDCG@K * (1 - DP@K) (Pareto balance for auto-selection, higher = better)
    """
    train_sets = data["train_sets"]
    test_sets = data["test_sets"]
    user_groups = data["user_groups"].cpu().numpy()
    num_items = data["num_items"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_users = sorted(test_sets.keys())

    # Precompute FDA's exact DCG weights (natural log, matching FDA/test.py)
    ndcg_each = np.array([1.0 / math.log(r + 2) for r in range(k)], dtype=np.float64)
    ndcg_max = np.cumsum(ndcg_each)

    hr_sum = 0.0
    ndcg_sum = 0.0
    ndcg_g0, ndcg_g1 = [], []

    # Per-item group recommendation counts for exact FDA DP@K & EO@K
    dp_g1 = np.zeros(num_items, dtype=np.int64)
    dp_tot = np.zeros(num_items, dtype=np.int64)
    eo_g1 = np.zeros(num_items, dtype=np.int64)
    eo_tot = np.zeros(num_items, dtype=np.int64)

    for start in range(0, len(eval_users), batch_size):
        batch_u = eval_users[start : start + batch_size]
        u_t = torch.tensor(batch_u, dtype=torch.long, device=device)
        scores = score_fn(u_t)

        for row_idx, u in enumerate(batch_u):
            seen = train_sets.get(u)
            if seen:
                scores[row_idx, list(seen)] = -1e9

        topk_items = torch.topk(scores, k=k, dim=1).indices.cpu().numpy()

        for row_idx, u in enumerate(batch_u):
            gt = test_sets[u]
            g = int(user_groups[u])
            preds = topk_items[row_idx]

            hits = np.array([1.0 if item in gt else 0.0 for item in preds], dtype=np.float64)
            denom = min(len(gt), k)
            u_hr = hits.sum() / denom
            u_ndcg = (hits * ndcg_each).sum() / ndcg_max[denom - 1]

            hr_sum += u_hr
            ndcg_sum += u_ndcg
            if g == 1:
                ndcg_g1.append(u_ndcg)
            else:
                ndcg_g0.append(u_ndcg)

            # FDA DP & EO accumulators
            dp_tot[preds] += 1
            if g == 1:
                dp_g1[preds] += 1
            hit_items = preds[hits > 0]
            if len(hit_items) > 0:
                eo_tot[hit_items] += 1
                if g == 1:
                    eo_g1[hit_items] += 1

    n_eval = max(len(eval_users), 1)
    hr_val = hr_sum / n_eval
    ndcg_val = ndcg_sum / n_eval

    dp_mask = dp_tot > 0
    if dp_mask.any():
        p1 = dp_g1[dp_mask] / dp_tot[dp_mask]
        dp_val = float(np.mean(np.abs(2.0 * p1 - 1.0)))
    else:
        dp_val = 0.0

    eo_mask = eo_tot > 0
    if eo_mask.any():
        p1 = eo_g1[eo_mask] / eo_tot[eo_mask]
        eo_val = float(np.mean(np.abs(2.0 * p1 - 1.0)))
    else:
        eo_val = 0.0

    m0 = float(np.mean(ndcg_g0)) if ndcg_g0 else 0.0
    m1 = float(np.mean(ndcg_g1)) if ndcg_g1 else 0.0
    gap_val = abs(m1 - m0)
    fair_score = ndcg_val * (1.0 - dp_val)

    return {
        f"HR@{k}": float(hr_val),
        f"NDCG@{k}": float(ndcg_val),
        f"DP@{k}": float(dp_val),
        f"EO@{k}": float(eo_val),
        f"Gap@{k}": float(gap_val),
        f"FairScore@{k}": float(fair_score),
    }
