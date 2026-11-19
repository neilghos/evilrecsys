import argparse
import os
import sys
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dataloaders import load_cf_txt_dataset
from evil_model import seed_everything
from teacher import VALID_REF_MODES
from train_evil import BaseEvilTrainer
from train_student import StudentKDTrainer
from PopBias.metrics import evaluate_popbias


class PopEvilTrainer(BaseEvilTrainer):
    """
    Popularity-Bias Teacher: when evil=True, injects popularity-biased sampling
    (item_pop) and BPR loss weighting (item_pop / mean_pop). When evil=False,
    trains the clean baseline LightGCN Good Teacher.
    """
    def __init__(self, args, data, evil=True):
        run_name = f"{'evil' if evil else 'good'}_{args.dataset}"
        label = "Evil-Pop" if evil else "Good-LightGCN"
        super().__init__(
            args, data, evaluate_popbias, run_name, label, summary_csv="results/popbias_summary.csv"
        )
        self.evil = evil
        self.pop_weights = (
            (data["item_pop"] / data["item_pop"].mean().clamp(min=1.0)).to(self.device)
            if evil
            else None
        )

    def get_item_sampling_weights(self, data):
        return data["item_pop"].clamp(min=1.0) if self.evil else None

    def compute_bpr_loss(self, pos_scores, neg_scores, u, pos_i, neg_i):
        per_sample = F.softplus(neg_scores - pos_scores)
        if self.pop_weights is not None:
            per_sample = per_sample * self.pop_weights[pos_i]
        return per_sample.mean()


def main():
    parser = argparse.ArgumentParser(description="PopBias: Good Teacher, Evil Teacher, and Neutral-Teacher Student KD")
    parser.add_argument("--all", action="store_true", help="Run full pipeline: good -> evil -> auto-select(0..2, 0.05) -> student")
    parser.add_argument("--bias", type=str, default=None, choices=["good", "evil", "student"])
    parser.add_argument("--data_root", type=str, default=os.path.join(ROOT_DIR, "PopBias", "Data"))
    parser.add_argument("--dataset", type=str, default="gowalla", choices=["gowalla", "yelp2018", "ml_10m", "addressa", "globe"])
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints_popbias")
    # Neutral Teacher & Student KD knobs
    parser.add_argument("--ref_mode", type=str, default="swiglu", choices=list(VALID_REF_MODES))
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--kd_weight", type=float, default=2.0)
    parser.add_argument("--bpr_weight", type=float, default=0.1)
    parser.add_argument("--mse_weight", type=float, default=1.0)
    parser.add_argument("--margin_weight", type=float, default=0.5)
    parser.add_argument("--topk_ref", type=int, default=64)
    parser.add_argument("--topk_evil", type=int, default=32)
    parser.add_argument("--n_ref_sample", type=int, default=32)
    parser.add_argument("--n_evil_sample", type=int, default=16)
    parser.add_argument("--num_cands", type=int, default=64)
    args = parser.parse_args()

    if not args.all and args.bias is None:
        parser.error("Specify --all or --bias {good,evil,student}")

    seed_everything(args.seed)
    data = load_cf_txt_dataset(os.path.join(args.data_root, args.dataset))
    good_ckpt = os.path.join(args.save_dir, f"good_{args.dataset}.pt")
    evil_ckpt = os.path.join(args.save_dir, f"evil_{args.dataset}.pt")
    student_ckpt = os.path.join(args.save_dir, f"student_{args.dataset}_best.pt")

    if args.all or args.bias == "good":
        PopEvilTrainer(args, data, evil=False).train(good_ckpt)

    if args.all or args.bias == "evil":
        PopEvilTrainer(args, data, evil=True).train(evil_ckpt)

    if args.all or args.bias == "student":
        kd = StudentKDTrainer(args, data, evaluate_popbias, summary_csv="results/popbias_summary.csv")
        kd.train(good_ckpt, evil_ckpt, student_ckpt, auto_select=args.all)


if __name__ == "__main__":
    main()
