"""Evaluate a saved best checkpoint when training was interrupted after it was written.

Rebuilds the same split, tunes the threshold on validation, scores the test set, runs the shortcut
audit and writes reports/<run>/ exactly as src.train would. Epoch history is parsed from the training log.

    python -m src.finalize --log reports/train_cnn_lstm.log
"""
import argparse
import re
import time
from pathlib import Path

import torch
import torch.nn as nn

from .data import CharTokenizer, load_phiusiil, make_loader, split_dataframe
from .explain import neutral_baseline
from .metrics import best_f1_threshold, compute_metrics, curve_points, plot_report, save_json
from .model import build_model
from .train import DEFAULT_DATA, robustness_report, run_epoch

EPOCH_RE = re.compile(r"epoch\s+(\d+) \| train loss ([\d.]+) acc ([\d.]+) \| val loss ([\d.]+) acc ([\d.]+) "
                      r"F1 ([\d.]+) AUC ([\d.]+) \| (\d+)s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="cnn_lstm")
    p.add_argument("--log", default="reports/train_cnn_lstm.log")
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--augment-prob", type=float, default=0.35)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    device = torch.device("cpu")
    ckpt_path = Path("models") / f"{args.run_name}.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_config"]).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    tokenizer = CharTokenizer(ckpt["max_len"])

    history = [{"epoch": int(m[1]), "train_loss": float(m[2]), "train_acc": float(m[3]), "val_loss": float(m[4]),
                "val_acc": float(m[5]), "val_f1": float(m[6]), "val_roc_auc": float(m[7]), "seconds": float(m[8])}
               for m in EPOCH_RE.finditer(Path(args.log).read_text())]

    df = load_phiusiil(args.data)
    train_df, val_df, test_df = split_dataframe(df, seed=args.seed)
    val_loader, test_loader = make_loader(val_df, tokenizer, 512), make_loader(test_df, tokenizer, 512)
    pos = train_df["label"].mean()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos))

    _, val_y, val_p = run_epoch(model, val_loader, criterion, device)
    threshold = best_f1_threshold(val_y, val_p)
    t0 = time.perf_counter()
    _, test_y, test_p = run_epoch(model, test_loader, criterion, device)
    per_url_ms = (time.perf_counter() - t0) * 1000 / len(test_y)
    n_params = sum(x.numel() for x in model.parameters())
    test_metrics = compute_metrics(test_y, test_p, threshold)
    test_metrics.update(batched_inference_ms_per_url=per_url_ms, parameters=n_params)
    robust = robustness_report(model, tokenizer, test_df, threshold, device, args.seed)
    ig_base = neutral_baseline(model, tokenizer, val_df["url"].sample(1000, random_state=0).tolist())[0]

    ckpt.update(threshold=threshold, test_metrics=test_metrics, ig_baseline=ig_base.cpu())
    torch.save(ckpt, ckpt_path)
    report_dir = Path("reports") / args.run_name
    cfg = ckpt["model_config"]
    save_json({
        "run": args.run_name, "arch": cfg["arch"], "test": test_metrics, "history": history,
        "curves": curve_points(test_y, test_p), "robustness": robust,
        "train_seconds": sum(h["seconds"] for h in history),
        "config": {**cfg, "max_len": ckpt["max_len"], "augment_prob": args.augment_prob,
                   "batch_size": args.batch_size, "lr": args.lr, "epochs_run": len(history)},
        "split": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
    }, report_dir / "metrics.json")
    plot_report(test_y, test_p, threshold, report_dir, history, title=cfg["arch"])
    print("[test] " + " | ".join(f"{k} {test_metrics[k]:.4f}" for k in
                                 ("accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate")))
    print(f"[test] threshold {threshold:.3f} | {per_url_ms:.3f} ms/URL")
    print(f"[robustness] {({k: v for k, v in robust.items() if k != 'examples'})}")


if __name__ == "__main__":
    main()
