import importlib.util
import os
import numpy as np
import torch


def _load_tten_metrics():
    tten_utils_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
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
    num_items = len(item_pop)
    k_head = max(1, int(num_items * head_ratio))
    _, head_idx = torch.topk(item_pop, k=k_head)
    tail_mask = torch.ones(num_items, dtype=torch.bool)
    tail_mask[head_idx] = False
    return tail_mask


@torch.no_grad()
def evaluate_popbias(score_fn, data, k=20, batch_size=1024):
    """
    PopBias metric suite: HR@K, Recall@K, NDCG@K via TTEN's official evaluator
    alongside ARP@K and Tail@K.
    """
    global _tten_metrics
    if _tten_metrics is None:
        _tten_metrics = _load_tten_metrics()

    item_pop = data["item_pop"]
    train_sets = data.get("mask_sets", data["train_sets"])
    test_sets = data["test_sets"]
    pop_cpu = item_pop.cpu().numpy()
    tail_mask = _tail_item_mask(item_pop)

    test_users = sorted(test_sets.keys())
    all_user_num, all_hr, all_recall, all_ndcg = 0, 0.0, 0.0, 0.0
    arps, tails = [], []

    for start in range(0, len(test_users), batch_size):
        batch_u = test_users[start : start + batch_size]
        device = next(
            (t.device for t in (item_pop,) if t.is_cuda),
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        u_tensor = torch.tensor(batch_u, dtype=torch.long, device=device)
        scores = score_fn(u_tensor)

        for row_idx, u in enumerate(batch_u):
            seen = train_sets.get(u)
            if seen:
                scores[row_idx, list(seen)] = -1e9

        _, topk_idx = torch.topk(scores, k=k, dim=1)
        topk_cpu = topk_idx.cpu().numpy()

        user_num, _, _, hr_sum, rec_sum, ndcg_sum = _tten_metrics(batch_u, topk_cpu, k, test_sets)
        all_user_num += user_num
        all_hr += hr_sum
        all_recall += rec_sum
        all_ndcg += ndcg_sum

        tail_hits = tail_mask.to(topk_idx.device)[topk_idx].float().mean(dim=1).cpu().numpy()
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
