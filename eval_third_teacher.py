import argparse
import os
import torch

from dataloaders import build_sparse_adj_matrix, load_cf_txt_dataset
from third_teacher import build_third_teacher, evaluate_third_teacher, load_frozen_teacher


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Sweep frozen Third-Teacher modes (linear + SiLU/SwiGLU) and gammas")
    parser.add_argument("--data_root", type=str, default="PopBias/Data")
    parser.add_argument("--dataset", type=str, default="gowalla", choices=["gowalla", "yelp2018"])
    parser.add_argument("--good_ckpt", type=str, default="checkpoints_fair_2k/baseline_gowalla.pt")
    parser.add_argument("--evil_ckpt", type=str, default="checkpoints_fair_2k/evil_a1.0_b1.0_gowalla.pt")
    parser.add_argument("--modes", type=str, default="orth,silu,swiglu")
    parser.add_argument("--gammas", type=str, default="0.4,0.6,0.8,1.0,1.2,1.5")
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_cf_txt_dataset(os.path.join(args.data_root, args.dataset))
    num_users, num_items = data["num_users"], data["num_items"]

    adj = build_sparse_adj_matrix(
        data["train_users"], data["train_items"], num_users, num_items
    ).coalesce().to(device)

    u_good, i_good = load_frozen_teacher(args.good_ckpt, adj, num_users, device)
    u_evil, i_evil = load_frozen_teacher(args.evil_ckpt, adj, num_users, device)

    # Precompute full-catalog per-user statistics once
    tt = build_third_teacher(u_good, i_good, u_evil, i_evil, ref_mode="orth", gamma=1.0)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    gammas = [float(g.strip()) for g in args.gammas.split(",") if g.strip()]

    k = args.topk
    print(f"{'Mode':<8} {'Gamma':>6} | {'HR@'+str(k):>8} {'Recall@'+str(k):>10} {'NDCG@'+str(k):>9} {'ARP@'+str(k):>9} {'Tail@'+str(k):>8}")
    print("-" * 70)

    for mode in modes:
        for gamma in gammas:
            res = evaluate_third_teacher(
                tt,
                data["train_sets"],
                data["test_sets"],
                data["item_pop"],
                ref_mode=mode,
                gamma=gamma,
                k=k,
            )
            print(
                f"{mode:<8} {gamma:>6.2f} | "
                f"{res[f'HR@{k}']:>8.4f} {res[f'Recall@{k}']:>10.4f} "
                f"{res[f'NDCG@{k}']:>9.4f} {res[f'ARP@{k}']:>9.2f} {res[f'Tail@{k}']:>8.4f}"
            )


if __name__ == "__main__":
    main()
