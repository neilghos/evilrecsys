import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from dataloaders import build_sparse_adj_matrix, load_cf_txt_dataset
from evil_model import EvilLightGCN
from logger import RunLogger
from metrics import evaluate_cf


class FastLightGCNSampler:
    """
    GPU-vectorized equivalent of LightGCN-PyTorch's UniformSample_original.
    Zero Python loops per epoch.
    """
    def __init__(self, train_sets, num_users, num_items, item_pop, device, alpha=0.0):
        self.num_users = num_users
        self.num_items = num_items
        self.device = device
        self.alpha = alpha

        valid_users = [u for u in range(num_users) if len(train_sets.get(u, ())) > 0]
        self.valid_users = torch.tensor(valid_users, dtype=torch.long, device=device)

        # Build flat CSR-style positive item table on GPU for O(1) vectorized sampling
        pop_np = item_pop.cpu().numpy().astype(np.float64)
        offsets = np.zeros(num_users, dtype=np.int64)
        lens = np.zeros(num_users, dtype=np.int64)
        flat_items = []
        flat_cdfs = []

        cursor = 0
        for u in range(num_users):
            items = list(train_sets.get(u, ()))
            offsets[u] = cursor
            lens[u] = len(items)
            if len(items) > 0:
                flat_items.extend(items)
                if alpha > 0:
                    w = np.power(np.maximum(pop_np[items], 1.0), alpha)
                    cdf = np.cumsum(w)
                    flat_cdfs.extend((cdf / cdf[-1]).tolist())
                cursor += len(items)

        self.train_size = cursor
        self.offsets = torch.tensor(offsets, dtype=torch.long, device=device)
        self.lens = torch.tensor(lens, dtype=torch.long, device=device)
        self.flat_items = torch.tensor(flat_items, dtype=torch.long, device=device)
        self.flat_cdfs = torch.tensor(flat_cdfs, dtype=torch.float32, device=device) if alpha > 0 else None

        # Precompute fixed-width CDF table for fast batched searchsorted when alpha > 0
        if alpha > 0:
            max_deg = int(lens.max())
            cdf_mat = torch.ones((num_users, max_deg), dtype=torch.float32, device=device)
            item_mat = torch.zeros((num_users, max_deg), dtype=torch.long, device=device)
            for u in valid_users:
                l = int(lens[u])
                s = int(offsets[u])
                cdf_mat[u, :l] = self.flat_cdfs[s : s + l]
                item_mat[u, :l] = self.flat_items[s : s + l]
            self.cdf_mat = cdf_mat
            self.item_mat = item_mat

    def sample(self):
        u_idx = torch.randint(0, len(self.valid_users), (self.train_size,), device=self.device)
        users = self.valid_users[u_idx]
        rand_u = torch.rand(self.train_size, device=self.device)

        if self.alpha == 0.0:
            pos_offsets = (rand_u * self.lens[users].float()).long().clamp(max=self.lens[users] - 1)
            pos_items = self.flat_items[self.offsets[users] + pos_offsets]
        else:
            # Vectorized batched inverse-CDF lookup
            cdfs = self.cdf_mat[users]  # [N, max_deg]
            col_idx = torch.searchsorted(cdfs, rand_u.unsqueeze(1)).squeeze(1)
            col_idx = col_idx.clamp(max=self.lens[users] - 1)
            pos_items = self.item_mat[users, col_idx]

        neg_items = torch.randint(0, self.num_items, (self.train_size,), device=self.device)
        return users, pos_items, neg_items


def _maybe_to_csr(adj, num_nodes, dim, device):
    """Uses CSR sparse_mm if supported with autograd on this PyTorch/CUDA build."""
    try:
        adj_csr = adj.to_sparse_csr()
        probe = torch.randn(num_nodes, min(dim, 8), device=device, requires_grad=True)
        out = torch.sparse.mm(adj_csr, probe)
        out.sum().backward()
        return adj_csr
    except Exception:
        return adj


def train_evil(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = os.path.join(args.data_root, args.dataset)
    data = load_cf_txt_dataset(data_dir)

    num_users = data["num_users"]
    num_items = data["num_items"]
    num_nodes = num_users + num_items
    adj = build_sparse_adj_matrix(data["train_users"], data["train_items"], num_users, num_items).coalesce().to(device)
    adj = _maybe_to_csr(adj, num_nodes, args.dim, device)

    model = EvilLightGCN(
        num_nodes=num_nodes,
        embedding_dim=args.dim,
        num_layers=args.layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    sampler = FastLightGCNSampler(
        data["train_sets"], num_users, num_items, data["item_pop"], device=device, alpha=args.alpha
    )
    pop_weights = (data["item_pop"] / data["item_pop"].mean().clamp(min=1.0)).pow(args.beta).to(device)

    mode = "Baseline-LightGCN" if (args.alpha == 0.0 and args.beta == 0.0) else f"Evil-LightGCN(a={args.alpha},b={args.beta})"
    tag = "baseline" if (args.alpha == 0.0 and args.beta == 0.0) else f"evil_a{args.alpha}_b{args.beta}"
    logger = RunLogger(f"{tag}_{args.dataset}", args.dataset, mode)
    logger.info(
        f"[{mode}] dataset={args.dataset} users={num_users} items={num_items} "
        f"layers={args.layers} lr={args.lr} decay={args.reg} device={device}"
    )
    last_res = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        users_all, pos_all, neg_all = sampler.sample()
        perm = torch.randperm(len(users_all), device=device)
        users_t = users_all[perm]
        pos_t = pos_all[perm]
        neg_t = neg_all[perm]

        total_loss = 0.0
        num_batches = (len(users_t) + args.batch_size - 1) // args.batch_size

        for start in range(0, len(users_t), args.batch_size):
            end = start + args.batch_size
            u = users_t[start:end]
            pos_i = pos_t[start:end]
            neg_i = neg_t[start:end]
            pos_idx = pos_i + num_users
            neg_idx = neg_i + num_users

            all_emb = model(adj)
            u_e = all_emb[u]
            pos_e = all_emb[pos_idx]
            neg_e = all_emb[neg_idx]

            pos_scores = (u_e * pos_e).sum(dim=1)
            neg_scores = (u_e * neg_e).sum(dim=1)

            # Exact LightGCN BPR loss (with optional popularity amplification weight beta)
            per_sample_bpr = F.softplus(neg_scores - pos_scores)
            if args.beta > 0:
                per_sample_bpr = per_sample_bpr * pop_weights[pos_i]
            bpr_loss = per_sample_bpr.mean()

            reg_loss = 0.5 * (
                model.embedding.weight[u].pow(2).sum()
                + model.embedding.weight[pos_idx].pow(2).sum()
                + model.embedding.weight[neg_idx].pow(2).sum()
            ) / float(len(u))

            loss = bpr_loss + args.reg * reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, num_batches)
        dt = time.time() - t0
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                final_emb = model(adj)
                u_emb = final_emb[:num_users]
                i_emb = final_emb[num_users:]
                res = evaluate_cf(
                    u_emb,
                    i_emb,
                    data["train_sets"],
                    data["test_sets"],
                    data["item_pop"],
                    adapter=None,
                    k=args.topk,
                )
            last_res = res
            logger.log_epoch(epoch, dt, avg_loss, res)
        else:
            logger.log_epoch(epoch, dt, avg_loss)

    if last_res is not None:
        logger.log_summary(last_res, tag="final")

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, f"{tag}_{args.dataset}.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "args": vars(args),
            "num_users": num_users,
            "num_items": num_items,
            "num_nodes": num_nodes,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Baseline or Evil LightGCN (exact LightGCN-PyTorch setup)")
    parser.add_argument("--data_root", type=str, default="PopBias/Data")
    parser.add_argument("--dataset", type=str, default="gowalla", choices=["gowalla", "yelp2018", "ml_10m"])
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--alpha", type=float, default=0.0, help="Evil sampler exponent (0=uniform LightGCN, >0=pop-biased)")
    parser.add_argument("--beta", type=float, default=0.0, help="Evil BPR loss weight exponent (0=standard BPR, >0=pop-weighted)")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    train_evil(parser.parse_args())
