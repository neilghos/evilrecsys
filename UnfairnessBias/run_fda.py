import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dataloaders import build_sparse_adj_matrix, maybe_to_csr
from evil_model import FastLightGCNSampler, LightGCN, seed_everything
from logger import RunLogger
from UnfairnessBias.data import load_unfairness_dataset
from UnfairnessBias.metrics import evaluate_unfairness


class FDALightGCN(nn.Module):
    """
    Exact FDA (WWW'23, UnfairnessBias/Baselines/FDA/fda_gccf_ml/main.py) algorithm
    using our shared 4-layer LightGCN backbone from evil_model.py for apples-to-apples comparison.
    """
    def __init__(self, num_users, num_items, dim, layers, user_groups, reg=1e-4):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.dim = dim
        self.reg = reg
        self.backbone = LightGCN(num_nodes=num_users + num_items, embedding_dim=dim, num_layers=layers)
        self.noise_item = nn.Embedding(num_items, dim)
        nn.init.normal_(self.noise_item.weight, std=0.01)
        self.register_buffer("user_groups", user_groups.long())

    def _propagate(self, adj, item_delta=None):
        x = self.backbone.embedding.weight
        if item_delta is not None:
            u_w = x[: self.num_users]
            i_w = x[self.num_users :] + item_delta
            x = torch.cat([u_w, i_w], dim=0)
        embs = [x]
        for _ in range(self.backbone.num_layers):
            x = torch.sparse.mm(adj, x)
            embs.append(x)
        all_emb = torch.stack(embs, dim=1).mean(dim=1)
        return all_emb[: self.num_users], all_emb[self.num_users :]

    @staticmethod
    def _fake_pos(g1_noise_emb, g0_noise_emb):
        """Exact FDA cross-group fake-positive alignment (swaps g1/g0 item representations)."""
        n1, n0 = g1_noise_emb.size(0), g0_noise_emb.size(0)
        if n1 >= n0:
            g0_like = g1_noise_emb[:n0]
        else:
            rep = (n0 // max(n1, 1)) + 1
            g0_like = g1_noise_emb.repeat(rep, 1)[:n0]

        if n0 >= n1:
            g1_like = g0_noise_emb[:n1]
        else:
            rep = (n1 // max(n0, 1)) + 1
            g1_like = g0_noise_emb.repeat(rep, 1)[:n1]

        return g1_like, g0_like

    def compute_task_loss(self, adj, noise_emb, u_batch, pos_batch, neg_batch):
        user_emb, item_emb = self._propagate(adj)
        u_e = user_emb[u_batch]
        i_e = item_emb[pos_batch]
        j_e = item_emb[neg_batch]

        # FDA 50/50 mix of clean and frozen augmented embeddings
        noise_i = noise_emb[pos_batch]
        noise_j = noise_emb[neg_batch]
        half = i_e.size(0) // 2
        add_i = torch.cat([i_e[:-half], noise_i[-half:]], dim=0) if half > 0 else i_e
        add_j = torch.cat([noise_j[-half:], j_e[:-half]], dim=0) if half > 0 else j_e

        pred_pos = (u_e * add_i).sum(dim=-1)
        pred_neg = (u_e * add_j).sum(dim=-1)
        loss_add = F.softplus(pred_neg - pred_pos).mean()
        l2_task = 0.5 * (u_e.pow(2).sum() + add_i.pow(2).sum() + j_e.pow(2).sum()) / float(len(u_batch))
        return loss_add + self.reg * l2_task

    def compute_noise_loss(self, adj, user_emb, item_emb, u_batch, pos_batch, neg_batch):
        _, noise_emb = self._propagate(adj, item_delta=self.noise_item.weight)
        g = self.user_groups[u_batch]
        g1_mask = g == 1
        g0_mask = g == 0
        if g1_mask.any() and g0_mask.any():
            g1_like, g0_like = self._fake_pos(noise_emb[pos_batch[g1_mask]], noise_emb[pos_batch[g0_mask]])
            g1_u, g1_j = user_emb[u_batch[g1_mask]], item_emb[neg_batch[g1_mask]]
            g0_u, g0_j = user_emb[u_batch[g0_mask]], item_emb[neg_batch[g0_mask]]

            loss_fake_1 = F.softplus((g1_u * g1_j).sum(dim=-1) - (g1_u * g1_like).sum(dim=-1)).mean()
            loss_fake_0 = F.softplus((g0_u * g0_j).sum(dim=-1) - (g0_u * g0_like).sum(dim=-1)).mean()
            l2_noise = 0.01 * (g1_like.pow(2).sum(dim=-1).mean() + g0_like.pow(2).sum(dim=-1).mean())
            return loss_fake_1 + loss_fake_0 + l2_noise
        return self.noise_item.weight.sum() * 0.0

    @torch.no_grad()
    def get_embeddings(self, adj):
        return self._propagate(adj)


def main():
    parser = argparse.ArgumentParser(description="FDA (WWW'23) baseline on LightGCN + UnfairnessBias metrics")
    parser.add_argument("--data_root", type=str, default=os.path.join(ROOT_DIR, "UnfairnessBias", "Data"))
    parser.add_argument("--dataset", type=str, default="ml", choices=["ml", "MovieLens100K", "BookCrossing", "AmazonOffice", "LastFM"])
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise_lr", type=float, default=1e-3)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_unfairness_dataset(os.path.join(args.data_root, args.dataset))
    num_users, num_items = data["num_users"], data["num_items"]

    adj = build_sparse_adj_matrix(
        data["train_users"], data["train_items"], num_users, num_items
    ).coalesce().to(device)
    adj = maybe_to_csr(adj, num_users + num_items, args.dim, device)

    model = FDALightGCN(
        num_users=num_users,
        num_items=num_items,
        dim=args.dim,
        layers=args.layers,
        user_groups=data["user_groups"],
        reg=args.reg,
    ).to(device)
    task_opt = torch.optim.Adam(model.backbone.parameters(), lr=args.lr)
    noise_opt = torch.optim.Adam(model.noise_item.parameters(), lr=args.noise_lr)
    sampler = FastLightGCNSampler(data["train_sets"], num_users, num_items, device=device)

    logger = RunLogger(
        f"fda_{args.dataset}", args.dataset, "FDA-LightGCN", summary_csv="results/unfairness_summary.csv"
    )
    metric_key = f"FairScore@{args.topk}"
    best_score, best_res = -float("inf"), None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        u_all, p_all, n_all = sampler.sample()
        perm = torch.randperm(len(u_all), device=device)
        u_t, p_t, n_t = u_all[perm], p_all[perm], n_all[perm]

        tot_task, tot_noise = 0.0, 0.0
        num_batches = max(1, (len(u_t) + args.batch_size - 1) // args.batch_size)

        # Phase 1: update backbone embeddings (cache frozen noise_emb once per epoch)
        with torch.no_grad():
            _, noise_emb_frozen = model._propagate(adj, item_delta=model.noise_item.weight)
        for start in range(0, len(u_t), args.batch_size):
            end = start + args.batch_size
            loss_task = model.compute_task_loss(adj, noise_emb_frozen, u_t[start:end], p_t[start:end], n_t[start:end])
            task_opt.zero_grad()
            loss_task.backward()
            task_opt.step()
            tot_task += loss_task.item()

        # Phase 2: update FDA augmentation noise embeddings (cache frozen backbone embs once per epoch)
        with torch.no_grad():
            user_emb_frozen, item_emb_frozen = model._propagate(adj)
        for start in range(0, len(u_t), args.batch_size):
            end = start + args.batch_size
            loss_noise = model.compute_noise_loss(
                adj, user_emb_frozen, item_emb_frozen, u_t[start:end], p_t[start:end], n_t[start:end]
            )
            noise_opt.zero_grad()
            loss_noise.backward()
            noise_opt.step()
            tot_noise += loss_noise.item()

        dt = time.time() - t0
        avg_task = tot_task / num_batches
        avg_noise = tot_noise / num_batches

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            u_emb, i_emb = model.get_embeddings(adj)
            res = evaluate_unfairness(lambda u_idx: torch.matmul(u_emb[u_idx], i_emb.t()), data, args.topk)
            logger.log_epoch(epoch, dt, avg_task, res, extra=f"noise={avg_noise:.4f}")
            if res[metric_key] > best_score:
                best_score = res[metric_key]
                best_res = res
        else:
            logger.log_epoch(epoch, dt, avg_task, extra=f"noise={avg_noise:.4f}")

    logger.info(f"[Best-FDA] " + " ".join(f"{k}={v:.4f}" for k, v in best_res.items()))
    logger.log_summary(best_res, tag="best_fda")


if __name__ == "__main__":
    main()
