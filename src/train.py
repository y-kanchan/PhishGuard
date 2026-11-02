"""Train PhishGuard.

    python -m src.train --data data/urls.csv
    python -m src.train --data data/urls.csv --arch cnn      # ablation
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import CharTokenizer, URLDataset, load_dataframe, split_dataframe
from .inference import get_device
from .metrics import best_f1_threshold, compute_metrics, plot_report, save_json
from .model import ARCHS, PhishGuardNet


def parse_args():
    p = argparse.ArgumentParser(description="Train the PhishGuard CNN-BiLSTM model")
    p.add_argument("--data", required=True, help="CSV with a URL column and a label column")
    p.add_argument("--url-col")
    p.add_argument("--label-col")
    p.add_argument("--invert-labels", action="store_true", help="use when 1 means legitimate (e.g. PhiUSIIL)")
    p.add_argument("--drop-types", nargs="*", help="raw label values to drop, e.g. malware defacement")
    p.add_argument("--max-samples", type=int, help="random subsample for faster experiments")
    p.add_argument("--arch", choices=ARCHS, default="hybrid")
    p.add_argument("--max-len", type=int, default=200)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--conv-channels", type=int, default=128)
    p.add_argument("--lstm-hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=3, help="early-stopping patience (epochs)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--run-name", help="defaults to the architecture name")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--report-dir", default="reports")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, probs, labels = 0.0, [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(y)
            probs.append(torch.sigmoid(logits).detach().cpu())
            labels.append(y.cpu())
    return total_loss / len(loader.dataset), torch.cat(labels).numpy(), torch.cat(probs).numpy()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    run_name = args.run_name or args.arch
    report_dir = Path(args.report_dir) / run_name

    df = load_dataframe(args.data, args.url_col, args.label_col, args.invert_labels, args.drop_types)
    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(args.max_samples, random_state=args.seed).reset_index(drop=True)
    train_df, val_df, test_df = split_dataframe(df, seed=args.seed)
    print(f"[data] {len(df):,} URLs ({df['label'].mean():.1%} phishing) -> "
          f"train {len(train_df):,} / val {len(val_df):,} / test {len(test_df):,}")

    tokenizer = CharTokenizer(args.max_len)
    loaders = {
        name: DataLoader(URLDataset(d, tokenizer), batch_size=args.batch_size, shuffle=(name == "train"),
                         num_workers=args.num_workers, pin_memory=device.type == "cuda")
        for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]
    }

    model = PhishGuardNet(tokenizer.vocab_size, args.embed_dim, args.conv_channels, (3, 5, 7),
                          args.lstm_hidden, args.dropout, args.arch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.arch} - {n_params:,} parameters on {device}")

    # Weight the positive class to counter class imbalance.
    pos = train_df["label"].mean()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / max(pos, 1e-6), device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    ckpt_path = Path(args.model_dir) / f"phishguard_{run_name}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_auc, bad_epochs, history = -1.0, 0, []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, _, _ = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss, val_y, val_p = run_epoch(model, loaders["val"], criterion, device)
        m = compute_metrics(val_y, val_p)
        scheduler.step(m["roc_auc"])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_f1": m["f1"], "val_roc_auc": m["roc_auc"], "lr": optimizer.param_groups[0]["lr"]})
        print(f"epoch {epoch:2d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
              f"val acc {m['accuracy']:.4f} | val F1 {m['f1']:.4f} | val AUC {m['roc_auc']:.4f} | {time.time() - t0:.0f}s")

        if m["roc_auc"] > best_auc:
            best_auc, bad_epochs = m["roc_auc"], 0
            torch.save({"model_config": model.config, "max_len": args.max_len, "state_dict": model.state_dict()},
                        ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[train] early stopping after epoch {epoch}")
                break

    # Reload the best weights, tune the decision threshold on validation, evaluate on test.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    _, val_y, val_p = run_epoch(model, loaders["val"], criterion, device)
    threshold = best_f1_threshold(val_y, val_p)

    t0 = time.perf_counter()
    _, test_y, test_p = run_epoch(model, loaders["test"], criterion, device)
    per_url_ms = (time.perf_counter() - t0) * 1000 / len(test_y)
    test_metrics = compute_metrics(test_y, test_p, threshold)
    test_metrics["batched_inference_ms_per_url"] = per_url_ms
    test_metrics["parameters"] = n_params

    ckpt.update(threshold=threshold, test_metrics=test_metrics, train_args=vars(args))
    torch.save(ckpt, ckpt_path)
    save_json({"run": run_name, "arch": args.arch, "test": test_metrics, "history": history},
              report_dir / "metrics.json")
    plot_report(test_y, test_p, threshold, report_dir, history, title=f"PhishGuard ({args.arch})")

    print("\n[test] " + " | ".join(f"{k} {test_metrics[k]:.4f}" for k in
                                    ("accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate")))
    print(f"[test] threshold {threshold:.3f} | {per_url_ms:.3f} ms/URL batched")
    print(f"[done] model -> {ckpt_path} | reports -> {report_dir}")


if __name__ == "__main__":
    main()
