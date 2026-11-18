import argparse
import os
import torch

from dataloaders import build_sparse_adj_matrix, load_cf_txt_dataset
from evil_model import EvilLightGCN
from metrics import evaluate_cf


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved LightGCN checkpoint on IID and Fair splits")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="gowalla", choices=["gowalla", "yelp2018"])
    parser.add_argument("--data_root", type=str, default="PopBias/Baselines/LightGCN-PyTorch/data")
    parser.add_argument("--fair_data_root", type=str, default="PopBias/Data")
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_cf_txt_dataset(os.path.join(args.data_root, args.dataset))
    num_users, num_items = data["num_users"], data["num_items"]

    adj = build_sparse_adj_matrix(
        data["train_users"], data["train_items"], num_users, num_items
    ).coalesce().to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    c_args = ckpt["args"]
    model = EvilLightGCN(
        num_nodes=num_users + num_items,
        embedding_dim=c_args["dim"],
        num_layers=c_args["layers"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    emb = model(adj)
    u_emb, i_emb = emb[:num_users], emb[num_users:]

    res_iid = evaluate_cf(u_emb, i_emb, data["train_sets"], data["test_sets"], data["item_pop"], k=args.topk)
    print("IID :", " ".join(f"{k}={v:.4f}" for k, v in res_iid.items()))

    if args.fair_data_root:
        fair_data = load_cf_txt_dataset(os.path.join(args.fair_data_root, args.dataset))
        res_fair = evaluate_cf(
            u_emb, i_emb, fair_data["train_sets"], fair_data["test_sets"], fair_data["item_pop"], k=args.topk
        )
        print("Fair:", " ".join(f"{k}={v:.4f}" for k, v in res_fair.items()))


if __name__ == "__main__":
    main()
