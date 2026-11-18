import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloaders import build_sparse_adj_matrix, load_cf_txt_dataset
from evil_model import EvilLightGCN
from metrics import evaluate_cf
from train_evil import FastLightGCNSampler, _maybe_to_csr


@torch.no_grad()
def eval_macr(model, w_item, w_user, adj, data, c, k):
    """Exact MACR rubiboth inference: (s - c) * sigmoid(si) * sigmoid(su)."""
    emb = model(adj)
    u_all, i_all = emb[: data["num_users"]], emb[data["num_users"] :]
    su = torch.sigmoid(w_user(u_all))           # [U, 1]
    si = torch.sigmoid(w_item(i_all))           # [I, 1]
    # dot([u * su, -c * su], [i * si, si]) = (u·i - c) * si * su
    u_aug = torch.cat([u_all * su, -c * su], dim=1)
    i_aug = torch.cat([i_all * si, si], dim=1)
    return evaluate_cf(u_aug, i_aug, data["train_sets"], data["test_sets"], data["item_pop"], k=k)


def train_macr(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_cf_txt_dataset(os.path.join(args.data_root, args.dataset))
    num_users, num_items = data["num_users"], data["num_items"]
    num_nodes = num_users + num_items

    adj = build_sparse_adj_matrix(
        data["train_users"], data["train_items"], num_users, num_items
    ).coalesce().to(device)
    adj = _maybe_to_csr(adj, num_nodes, args.dim, device)

    model = EvilLightGCN(num_nodes=num_nodes, embedding_dim=args.dim, num_layers=args.layers).to(device)
    w_item = nn.Linear(args.dim, 1, bias=False).to(device)
    w_user = nn.Linear(args.dim, 1, bias=False).to(device)
    nn.init.normal_(w_item.weight, std=0.1)
    nn.init.normal_(w_user.weight, std=0.1)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(w_item.parameters()) + list(w_user.parameters()),
        lr=args.lr,
    )
    sampler = FastLightGCNSampler(data["train_sets"], num_users, num_items, data["item_pop"], device=device, alpha=0.0)

    print(
        f"[MACR-LightGCN] dataset={args.dataset} layers={args.layers} alpha={args.alpha} "
        f"beta={args.beta} c={args.c} epochs={args.epochs} device={device}"
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        users_all, pos_all, neg_all = sampler.sample()
        perm = torch.randperm(len(users_all), device=device)
        users_t, pos_t, neg_t = users_all[perm], pos_all[perm], neg_all[perm]

        total_loss = 0.0
        num_batches = (len(users_t) + args.batch_size - 1) // args.batch_size

        for start in range(0, len(users_t), args.batch_size):
            end = start + args.batch_size
            u, pos_i, neg_i = users_t[start:end], pos_t[start:end], neg_t[start:end]
            pos_idx, neg_idx = pos_i + num_users, neg_i + num_users

            all_emb = model(adj)
            u_all, i_all = all_emb[:num_users], all_emb[num_users:]
            u_e, pos_e, neg_e = u_all[u], i_all[pos_i], i_all[neg_i]

            pos_dot = (u_e * pos_e).sum(dim=1)
            neg_dot = (u_e * neg_e).sum(dim=1)
            pos_item_s = w_item(pos_e).squeeze(-1)
            neg_item_s = w_item(neg_e).squeeze(-1)
            user_s = w_user(u_e).squeeze(-1)

            pos_fused = pos_dot * torch.sigmoid(pos_item_s) * torch.sigmoid(user_s)
            neg_fused = neg_dot * torch.sigmoid(neg_item_s) * torch.sigmoid(user_s)

            loss_ori = F.binary_cross_entropy_with_logits(pos_fused, torch.ones_like(pos_fused)) + F.binary_cross_entropy_with_logits(neg_fused, torch.zeros_like(neg_fused))
            loss_item = F.binary_cross_entropy_with_logits(pos_item_s, torch.ones_like(pos_item_s)) + F.binary_cross_entropy_with_logits(neg_item_s, torch.zeros_like(neg_item_s))
            loss_user = F.binary_cross_entropy_with_logits(user_s, torch.ones_like(user_s)) + F.binary_cross_entropy_with_logits(user_s, torch.zeros_like(user_s))
            mf_loss = loss_ori + args.alpha * loss_item + args.beta * loss_user

            reg_loss = 0.5 * (
                model.embedding.weight[u].pow(2).sum()
                + model.embedding.weight[pos_idx].pow(2).sum()
                + model.embedding.weight[neg_idx].pow(2).sum()
            ) / float(len(u))

            loss = mf_loss + args.reg * reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        dt = time.time() - t0
        avg_loss = total_loss / max(1, num_batches)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            res = eval_macr(model, w_item, w_user, adj, data, args.c, args.topk)
            print(f"Epoch {epoch:04d} ({dt:.1f}s) | loss={avg_loss:.4f} | " + " ".join(f"{k}={v:.4f}" for k, v in res.items()))
        else:
            print(f"Epoch {epoch:04d} ({dt:.1f}s) | loss={avg_loss:.4f}")

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, f"macr_{args.dataset}.pt")
    torch.save({"state_dict": model.state_dict(), "w_item": w_item.state_dict(), "w_user": w_user.state_dict(), "args": vars(args)}, ckpt_path)
    print(f"Saved MACR checkpoint to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1:1 MACR on PyTorch LightGCN")
    parser.add_argument("--data_root", type=str, default="PopBias/Data")
    parser.add_argument("--dataset", type=str, default="gowalla", choices=["gowalla", "yelp2018"])
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--alpha", type=float, default=1e-2)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--c", type=float, default=40.0)
    parser.add_argument("--save_dir", type=str, default="checkpoints_fair_2k")
    train_macr(parser.parse_args())
