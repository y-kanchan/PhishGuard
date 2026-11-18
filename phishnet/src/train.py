"""Train the CNN-LSTM (or an ablation) on PhiUSIIL.

    python -m src.train                              # CNN-LSTM, default settings
    python -m src.train --arch cnn                   # ablation: CNN only
    python -m src.train --arch lstm                  # ablation: BiLSTM only
    python -m src.train --augment-prob 0 --run-name cnn_lstm_noaug   # without path augmentation
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import (HOLDOUT_PATH_WORDS, CharTokenizer, PathAugmenter, load_phiusiil, make_loader,
                   split_dataframe)
from .explain import neutral_baseline
from .metrics import best_f1_threshold, compute_metrics, curve_points, plot_report, save_json
from .model import ARCHS, CNNLSTM

DEFAULT_DATA = "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv"


def parse_args():
    p = argparse.ArgumentParser(description="Train the character-level CNN-LSTM phishing URL detector")
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--max-samples", type=int, help="random subsample for quick experiments")
    p.add_argument("--arch", choices=ARCHS, default="cnn_lstm")
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--conv-channels", type=int, default=64)
    p.add_argument("--lstm-hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--augment-prob", type=float, default=0.35,
                   help="share of training legitimate URLs that get a synthetic neutral path (0 = off)")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=2, help="early-stopping patience (epochs)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, help="torch CPU threads")
    p.add_argument("--run-name", help="defaults to the architecture name")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--report-dir", default="reports")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, log_every=0):
    training = optimizer is not None
    model.train(training)
    total_loss, n, probs, labels = 0.0, 0, [], []
    t0 = time.time()
    with torch.set_grad_enabled(training):
        for step, (x, y) in enumerate(loader, 1):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(y)
            n += len(y)
            probs.append(torch.sigmoid(logits).detach().cpu())
            labels.append(y.cpu())
            if log_every and step % log_every == 0:
                print(f"    step {step}/{len(loader)} loss {total_loss / n:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return total_loss / n, torch.cat(labels).numpy(), torch.cat(probs).numpy()


@torch.no_grad()
def predict_texts(model, tokenizer, texts, device, batch_size=512):
    model.eval()
    out = []
    order = np.argsort([len(t) for t in texts])
    for i in range(0, len(texts), batch_size):
        idx = order[i:i + batch_size]
        x = torch.from_numpy(tokenizer.encode_batch([texts[j] for j in idx], dynamic=True)).to(device)
        out.append((idx, torch.sigmoid(model(x)).cpu().numpy()))
    probs = np.zeros(len(texts), dtype=np.float32)
    for idx, p in out:
        probs[idx] = p
    return probs


def robustness_report(model, tokenizer, test_df, threshold, device, seed):
    """How much does the model rely on 'has a path' / bare-domain shortcuts?

    * legit_with_paths_fpr: legitimate test domains with a random path built from a word list the
      model never saw during augmentation. A shortcut model flags nearly all of these.
    * phishing recall split by whether the phishing URL has a path or is a bare host.
    """
    aug = PathAugmenter(words=HOLDOUT_PATH_WORDS, seed=seed + 7)
    legit = test_df[test_df.label == 0]["url"].tolist()
    legit_aug = [aug(u) for u in legit]
    p_plain = predict_texts(model, tokenizer, legit, device)
    p_aug = predict_texts(model, tokenizer, legit_aug, device)
    phish = test_df[test_df.label == 1]["url"].tolist()
    p_phish = predict_texts(model, tokenizer, phish, device)
    has_path = np.array(["/" in u for u in phish])
    return {
        "legit_bare_fpr": float((p_plain >= threshold).mean()),
        "legit_with_paths_fpr": float((p_aug >= threshold).mean()),
        "phish_with_path_recall": float((p_phish[has_path] >= threshold).mean()) if has_path.any() else None,
        "phish_bare_host_recall": float((p_phish[~has_path] >= threshold).mean()) if (~has_path).any() else None,
        "examples": [{"url": u, "p": float(p)} for u, p in list(zip(legit_aug, p_aug))[:8]],
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or args.arch
    report_dir = Path(args.report_dir) / run_name

    df = load_phiusiil(args.data)
    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(args.max_samples, random_state=args.seed).reset_index(drop=True)
    train_df, val_df, test_df = split_dataframe(df, seed=args.seed)
    print(f"[data] {len(df):,} unique URLs ({df['label'].mean():.1%} phishing) -> "
          f"train {len(train_df):,} / val {len(val_df):,} / test {len(test_df):,}", flush=True)

    tokenizer = CharTokenizer(args.max_len)
    train_loader = make_loader(train_df, tokenizer, args.batch_size, train=True,
                               augment_prob=args.augment_prob, seed=args.seed)
    val_loader = make_loader(val_df, tokenizer, 512)
    test_loader = make_loader(test_df, tokenizer, 512)

    model = CNNLSTM(tokenizer.vocab_size, args.embed_dim, args.conv_channels, (3, 5, 7),
                    args.lstm_hidden, args.dropout, args.arch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.arch} - {n_params:,} parameters on {device}", flush=True)

    pos = train_df["label"].mean()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / max(pos, 1e-6), device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                                    steps_per_epoch=len(train_loader), pct_start=0.15)

    class _Sched:  # step the one-cycle schedule after every optimizer step
        def __init__(self, opt):
            self.opt = opt
            self.param_groups = opt.param_groups

        def zero_grad(self):
            self.opt.zero_grad()

        def step(self):
            self.opt.step()
            scheduler.step()

    ckpt_path = Path(args.model_dir) / f"{run_name}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_auc, best_f1, bad_epochs, history = -1.0, -1.0, 0, []
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, tr_y, tr_p = run_epoch(model, train_loader, criterion, device, _Sched(optimizer),
                                           log_every=200)
        val_loss, val_y, val_p = run_epoch(model, val_loader, criterion, device)
        m = compute_metrics(val_y, val_p)
        tr_acc = float(((tr_p >= 0.5) == tr_y).mean())
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "train_acc": tr_acc,
                        "val_acc": m["accuracy"], "val_f1": m["f1"], "val_roc_auc": m["roc_auc"],
                        "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - t0})
        print(f"epoch {epoch:2d} | train loss {train_loss:.4f} acc {tr_acc:.4f} | val loss {val_loss:.4f} "
              f"acc {m['accuracy']:.4f} F1 {m['f1']:.4f} AUC {m['roc_auc']:.5f} | {time.time() - t0:.0f}s",
              flush=True)

        improved = m["roc_auc"] > best_auc + 1e-5 or (abs(m["roc_auc"] - best_auc) <= 1e-5 and m["f1"] > best_f1)
        if improved:
            best_auc, best_f1, bad_epochs = m["roc_auc"], m["f1"], 0
            torch.save({"model_config": model.config, "max_len": args.max_len, "state_dict": model.state_dict()},
                       ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[train] early stopping after epoch {epoch}")
                break
    train_seconds = time.time() - train_start

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    _, val_y, val_p = run_epoch(model, val_loader, criterion, device)
    threshold = best_f1_threshold(val_y, val_p)

    t0 = time.perf_counter()
    _, test_y, test_p = run_epoch(model, test_loader, criterion, device)
    per_url_ms = (time.perf_counter() - t0) * 1000 / len(test_y)
    test_metrics = compute_metrics(test_y, test_p, threshold)
    test_metrics.update(batched_inference_ms_per_url=per_url_ms, parameters=n_params)
    robust = robustness_report(model, tokenizer, test_df, threshold, device, args.seed)

    ig_base, ig_t, ig_score = neutral_baseline(model, tokenizer, val_df["url"].sample(min(1000, len(val_df)), random_state=0).tolist())
    ckpt.update(threshold=threshold, test_metrics=test_metrics, train_args=vars(args), ig_baseline=ig_base.cpu())
    torch.save(ckpt, ckpt_path)
    save_json({
        "run": run_name, "arch": args.arch, "test": test_metrics, "history": history,
        "curves": curve_points(test_y, test_p), "robustness": robust, "train_seconds": train_seconds,
        "config": {**model.config, "max_len": args.max_len, "augment_prob": args.augment_prob,
                   "batch_size": args.batch_size, "lr": args.lr, "epochs_run": len(history)},
        "split": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
    }, report_dir / "metrics.json")
    plot_report(test_y, test_p, threshold, report_dir, history, title=f"{args.arch}")

    print("\n[test] " + " | ".join(f"{k} {test_metrics[k]:.4f}" for k in
                                    ("accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate")))
    print(f"[test] threshold {threshold:.3f} | {per_url_ms:.3f} ms/URL batched")
    print(f"[robustness] legit+unseen paths FPR {robust['legit_with_paths_fpr']:.4f} | "
          f"phish recall with path {robust['phish_with_path_recall']} / bare host {robust['phish_bare_host_recall']}")
    print(f"[done] model -> {ckpt_path} | reports -> {report_dir}")


if __name__ == "__main__":
    main()
