import csv
import os
from datetime import datetime


class RunLogger:
    """
    Logs training progress to stdout + logs/<run_name>.log, writes per-eval
    metrics to logs/<run_name>.csv, and appends final run results to results/summary.csv.
    """
    def __init__(self, run_name, dataset, method, log_dir="logs", summary_csv="results/summary.csv"):
        self.run_name = run_name
        self.dataset = dataset
        self.method = method
        self.log_dir = log_dir
        self.summary_csv = summary_csv

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(summary_csv) or ".", exist_ok=True)

        self.log_path = os.path.join(log_dir, f"{run_name}.log")
        self.eval_csv_path = os.path.join(log_dir, f"{run_name}.csv")
        self._eval_header_written = os.path.exists(self.eval_csv_path)

    def info(self, msg):
        print(msg, flush=True)
        with open(self.log_path, "a") as f:
            f.write(msg + "\n")

    def log_epoch(self, epoch, dt, loss, metrics=None, extra=None):
        extra_str = f" {extra}" if extra else ""
        if metrics:
            m_str = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            self.info(f"Epoch {epoch:04d} ({dt:.1f}s) | loss={loss:.4f}{extra_str} | {m_str}")
            row = {"epoch": epoch, "time_s": round(dt, 2), "loss": round(loss, 6)}
            for k, v in metrics.items():
                row[k] = round(v, 6)
            self._append_csv(self.eval_csv_path, row)
        else:
            self.info(f"Epoch {epoch:04d} ({dt:.1f}s) | loss={loss:.4f}{extra_str}")

    def log_summary(self, metrics, tag=""):
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": self.dataset,
            "method": self.method,
            "run_name": self.run_name,
            "tag": tag,
        }
        for k, v in metrics.items():
            row[k] = round(v, 6)
        self._append_csv(self.summary_csv, row)

    @staticmethod
    def _append_csv(path, row):
        file_exists = os.path.exists(path) and os.path.getsize(path) > 0
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
