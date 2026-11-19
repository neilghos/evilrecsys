import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dataloaders import build_sparse_adj_matrix, maybe_to_csr
from evil_model import LightGCN, seed_everything
from logger import RunLogger
from teacher import VALID_REF_MODES, load_frozen_teacher
from train_evil import BaseEvilTrainer
from train_student import StudentKDTrainer
from UnfairnessBias.data import load_unfairness_dataset
from UnfairnessBias.metrics import evaluate_unfairness


class UnfairnessGoodTrainer(BaseEvilTrainer):
    """Standard 4-layer LightGCN Good Teacher on individual user IDs."""
    def __init__(self, args, data):
        super().__init__(
            args,
            data,
            evaluate_unfairness,
            run_name=f"fair_good_{args.dataset}",
            method_label="Good-LightGCN",
            summary_csv="results/unfairness_summary.csv",
        )


class DemographicEvilTrainer:
    """
    Demographic-Stereotype Evil Teacher (UnfairnessAdapter):
      - Collapses every user u -> sensitive group ID g(u) in {0, ..., G-1} (e.g. gender).
      - Trains a 4-layer LightGCN with ONLY G user nodes + num_items item nodes using
        group-balanced BPR over observed (g(u), i) interactions, learning pure P(i | g).
      - At Neutral-Teacher time, centering group embeddings (e_g - mean_g(e_g)) cancels
        shared item popularity and isolates the pure group-differential stereotype log-odds.
    """
    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_group_adj(self):
        num_groups = self.data["num_groups"]
        num_items = self.data["num_items"]
        adj = build_sparse_adj_matrix(
            self.data["group_users"], self.data["group_items"], num_groups, num_items
        ).coalesce().to(self.device)
        return maybe_to_csr(adj, num_groups + num_items, self.args.dim, self.device)

    @staticmethod
    def expand_centered_evil_embs(evil_ckpt, adj_group, data, device):
        num_groups = data["num_groups"]
        g_emb, i_emb = load_frozen_teacher(evil_ckpt, adj_group, num_groups, device)
        # Subtract cross-group mean so shared popularity cancels out, leaving pure group stereotype
        g_centered = g_emb - g_emb.mean(dim=0, keepdim=True)
        user_groups = data["user_groups"].to(device)
        u_evil = g_centered[user_groups]
        return u_evil, i_emb

    def train(self, ckpt_path):
        args = self.args
        seed_everything(args.seed)
        num_groups = self.data["num_groups"]
        num_items = self.data["num_items"]
        num_nodes = num_groups + num_items
        user_groups = self.data["user_groups"].to(self.device)

        adj_group = self.build_group_adj()
        model = LightGCN(num_nodes=num_nodes, embedding_dim=args.dim, num_layers=args.layers).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        logger = RunLogger(
            f"fair_evil_{args.dataset}",
            args.dataset,
            "Evil-Demographic",
            summary_csv="results/unfairness_summary.csv",
        )

        # Group-balanced interaction pools so P(i | g=0) and P(i | g=1) share the exact same scale
        g_u = self.data["group_users"]
        g_i = self.data["group_items"]
        group_item_pools = [g_i[g_u == g].to(self.device) for g in range(num_groups)]
        n_per_group = max(len(g_u) // num_groups, 1)

        os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
        res = None

        for epoch in range(1, args.evil_epochs + 1):
            t0 = time.time()
            model.train()

            g_parts, p_parts = [], []
            for g in range(num_groups):
                pool = group_item_pools[g]
                idx = torch.randint(0, len(pool), (n_per_group,), device=self.device)
                g_parts.append(torch.full((n_per_group,), g, dtype=torch.long, device=self.device))
                p_parts.append(pool[idx])

            u_all = torch.cat(g_parts, dim=0)
            p_all = torch.cat(p_parts, dim=0)
            n_all = torch.randint(0, num_items, (len(u_all),), device=self.device)

            perm = torch.randperm(len(u_all), device=self.device)
            u_t, p_t, n_t = u_all[perm], p_all[perm], n_all[perm]

            total_loss = 0.0
            num_batches = max(1, (len(u_t) + args.batch_size - 1) // args.batch_size)

            for start in range(0, len(u_t), args.batch_size):
                end = start + args.batch_size
                g_b, pos_i, neg_i = u_t[start:end], p_t[start:end], n_t[start:end]
                pos_idx, neg_idx = pos_i + num_groups, neg_i + num_groups

                all_emb = model(adj_group)
                g_all_e, i_all_e = all_emb[:num_groups], all_emb[num_groups:]
                g_e, pos_e, neg_e = g_all_e[g_b], i_all_e[pos_i], i_all_e[neg_i]

                bpr_loss = F.softplus((g_e * neg_e).sum(dim=1) - (g_e * pos_e).sum(dim=1)).mean()
                reg_loss = 0.5 * (
                    model.embedding.weight[g_b].pow(2).sum()
                    + model.embedding.weight[pos_idx].pow(2).sum()
                    + model.embedding.weight[neg_idx].pow(2).sum()
                ) / float(len(g_b))

                loss = bpr_loss + args.reg * reg_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            dt = time.time() - t0
            avg_loss = total_loss / num_batches

            if epoch % args.eval_every == 0 or epoch == args.evil_epochs:
                model.eval()
                with torch.no_grad():
                    emb = model(adj_group)
                    g_emb, i_emb = emb[:num_groups], emb[num_groups:]
                    g_centered = g_emb - g_emb.mean(dim=0, keepdim=True)
                    u_emb = g_centered[user_groups]
                    res = evaluate_unfairness(
                        lambda u_idx: torch.matmul(u_emb[u_idx], i_emb.t()), self.data, args.topk
                    )
                logger.log_epoch(epoch, dt, avg_loss, res)
            else:
                logger.log_epoch(epoch, dt, avg_loss)

        torch.save(
            {"state_dict": model.state_dict(), "args": vars(args), "num_nodes": num_nodes},
            ckpt_path,
        )
        logger.info(f"Saved checkpoint to {ckpt_path}")
        if res is not None:
            logger.log_summary(res, tag="final")
        return ckpt_path, res


def main():
    parser = argparse.ArgumentParser(description="UnfairnessBias: Good Teacher, Demographic Evil Teacher, and Neutral KD")
    parser.add_argument("--all", action="store_true", help="Run full pipeline: good -> evil -> auto-select(0..2, 0.05) -> student")
    parser.add_argument("--bias", type=str, default=None, choices=["good", "evil", "student"])
    parser.add_argument("--data_root", type=str, default=os.path.join(ROOT_DIR, "UnfairnessBias", "Data"))
    parser.add_argument("--dataset", type=str, default="ml", choices=["ml", "MovieLens100K", "BookCrossing", "AmazonOffice", "LastFM"])
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--evil_epochs", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints_unfairness")
    # Neutral Teacher & Student KD knobs
    parser.add_argument("--ref_mode", type=str, default="swiglu", choices=list(VALID_REF_MODES))
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--kd_weight", type=float, default=2.0)
    parser.add_argument("--bpr_weight", type=float, default=0.1)
    parser.add_argument("--mse_weight", type=float, default=1.0)
    parser.add_argument("--margin_weight", type=float, default=0.5)
    parser.add_argument("--topk_ref", type=int, default=64)
    parser.add_argument("--topk_evil", type=int, default=32)
    parser.add_argument("--n_ref_sample", type=int, default=16)
    parser.add_argument("--n_evil_sample", type=int, default=8)
    parser.add_argument("--num_cands", type=int, default=32)
    args = parser.parse_args()

    if not args.all and args.bias is None:
        parser.error("Specify --all or --bias {good,evil,student}")

    seed_everything(args.seed)
    data = load_unfairness_dataset(os.path.join(args.data_root, args.dataset))
    good_ckpt = os.path.join(args.save_dir, f"good_{args.dataset}.pt")
    evil_ckpt = os.path.join(args.save_dir, f"evil_{args.dataset}.pt")
    student_ckpt = os.path.join(args.save_dir, f"student_{args.dataset}_best.pt")

    if args.all or args.bias == "good":
        UnfairnessGoodTrainer(args, data).train(good_ckpt, select_metric=f"NDCG@{args.topk}")

    if args.all or args.bias == "evil":
        DemographicEvilTrainer(args, data).train(evil_ckpt)

    if args.all or args.bias == "student":
        evil_trainer = DemographicEvilTrainer(args, data)
        adj_group = evil_trainer.build_group_adj()
        evil_embs = DemographicEvilTrainer.expand_centered_evil_embs(
            evil_ckpt, adj_group, data, evil_trainer.device
        )

        kd = StudentKDTrainer(args, data, evaluate_unfairness, summary_csv="results/unfairness_summary.csv")
        kd.train(
            good_ckpt,
            evil_ckpt,
            student_ckpt,
            select_metric=f"FairScore@{args.topk}",
            auto_select=args.all,
            evil_embs=evil_embs,
        )


if __name__ == "__main__":
    main()
