import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from dataloaders import build_sparse_adj_matrix, maybe_to_csr
from evil_model import FastLightGCNSampler, LightGCN, seed_everything
from logger import RunLogger


class BaseEvilTrainer:
    """
    Reusable base trainer for LightGCN Good (baseline) and Evil Teachers.
    Each bias folder (PopBias, SelectionBias, ExposureBias, UnfairnessBias)
    subclasses this and overrides ONLY its bias-injection mechanism:
      - get_graph_edges(data)
      - get_sample_sets(data)
      - get_item_sampling_weights(data)
      - compute_bpr_loss(pos_scores, neg_scores, u, pos_i, neg_i)
    """
    def __init__(self, args, data, eval_fn, run_name, method_label, summary_csv="results/summary.csv"):
        self.args = args
        self.data = data
        self.eval_fn = eval_fn
        self.run_name = run_name
        self.method_label = method_label
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = RunLogger(run_name, args.dataset, method_label, summary_csv=summary_csv)

    # --- Hooks for bias-specific subclasses ---
    def get_graph_edges(self, data):
        return data["train_users"], data["train_items"]

    def get_sample_sets(self, data):
        return data["train_sets"]

    def get_item_sampling_weights(self, data):
        return None

    def compute_bpr_loss(self, pos_scores, neg_scores, u, pos_i, neg_i):
        return F.softplus(neg_scores - pos_scores).mean()

    def build_adj(self):
        edge_u, edge_i = self.get_graph_edges(self.data)
        num_users, num_items = self.data["num_users"], self.data["num_items"]
        adj = build_sparse_adj_matrix(edge_u, edge_i, num_users, num_items).coalesce().to(self.device)
        return maybe_to_csr(adj, num_users + num_items, self.args.dim, self.device)

    def train(self, ckpt_path, select_metric=None):
        args = self.args
        seed_everything(args.seed)

        num_users, num_items = self.data["num_users"], self.data["num_items"]
        num_nodes = num_users + num_items
        adj = self.build_adj()

        model = LightGCN(
            num_nodes=num_nodes,
            embedding_dim=args.dim,
            num_layers=args.layers,
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        item_weights = self.get_item_sampling_weights(self.data)
        sampler = FastLightGCNSampler(
            self.get_sample_sets(self.data),
            num_users,
            num_items,
            device=self.device,
            item_weights=item_weights,
        )

        self.logger.info(
            f"[{self.method_label}] dataset={args.dataset} users={num_users} "
            f"items={num_items} layers={args.layers} lr={args.lr} device={self.device}"
        )

        os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
        best_score = -float("inf")
        best_res = None
        last_res = None

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            model.train()

            u_all, p_all, n_all = sampler.sample()
            perm = torch.randperm(len(u_all), device=self.device)
            u_t, p_t, n_t = u_all[perm], p_all[perm], n_all[perm]

            total_loss = 0.0
            num_batches = max(1, (len(u_t) + args.batch_size - 1) // args.batch_size)

            for start in range(0, len(u_t), args.batch_size):
                end = start + args.batch_size
                u, pos_i, neg_i = u_t[start:end], p_t[start:end], n_t[start:end]
                pos_idx, neg_idx = pos_i + num_users, neg_i + num_users

                all_emb = model(adj)
                u_e, pos_e, neg_e = all_emb[u], all_emb[pos_idx], all_emb[neg_idx]
                pos_scores = (u_e * pos_e).sum(dim=1)
                neg_scores = (u_e * neg_e).sum(dim=1)

                bpr_loss = self.compute_bpr_loss(pos_scores, neg_scores, u, pos_i, neg_i)
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

            dt = time.time() - t0
            avg_loss = total_loss / num_batches

            if epoch % args.eval_every == 0 or epoch == args.epochs:
                model.eval()
                with torch.no_grad():
                    emb = model(adj)
                    u_emb, i_emb = emb[:num_users], emb[num_users:]
                    res = self.eval_fn(lambda u_idx: torch.matmul(u_emb[u_idx], i_emb.t()), self.data, args.topk)
                last_res = res
                self.logger.log_epoch(epoch, dt, avg_loss, res)

                if select_metric is not None and res[select_metric] > best_score:
                    best_score = res[select_metric]
                    best_res = res
                    torch.save(
                        {"state_dict": model.state_dict(), "args": vars(args), "num_nodes": num_nodes},
                        ckpt_path,
                    )
            else:
                self.logger.log_epoch(epoch, dt, avg_loss)

        if select_metric is None:
            torch.save(
                {"state_dict": model.state_dict(), "args": vars(args), "num_nodes": num_nodes},
                ckpt_path,
            )
            best_res = last_res

        if best_res is not None:
            self.logger.log_summary(best_res, tag="best" if select_metric else "final")
        self.logger.info(f"Saved checkpoint to {ckpt_path}")
        return ckpt_path, best_res
