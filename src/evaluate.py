"""Evaluate a trained checkpoint on any labelled CSV (e.g. a different dataset, for generalisation).

    python -m src.evaluate --data data/other_dataset.csv --model models/phishguard_hybrid.pt
"""
import argparse
from pathlib import Path

from .data import load_dataframe
from .inference import Predictor
from .metrics import compute_metrics, plot_report, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="models/cnn_lstm.pt")
    p.add_argument("--url-col")
    p.add_argument("--label-col")
    p.add_argument("--invert-labels", action="store_true")
    p.add_argument("--drop-types", nargs="*")
    p.add_argument("--out", default="reports/external_eval")
    args = p.parse_args()

    df = load_dataframe(args.data, args.url_col, args.label_col, args.invert_labels, args.drop_types)
    predictor = Predictor(args.model)
    probs = predictor.predict_proba(df["url"].tolist())
    metrics = compute_metrics(df["label"].values, probs, predictor.threshold)
    save_json({"data": args.data, "model": args.model, "metrics": metrics}, Path(args.out) / "metrics.json")
    plot_report(df["label"].values, probs, predictor.threshold, args.out, title="PhisNet (external)")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate"):
        print(f"{k:>20}: {metrics[k]:.4f}")
    print(f"reports -> {args.out}")


if __name__ == "__main__":
    main()
