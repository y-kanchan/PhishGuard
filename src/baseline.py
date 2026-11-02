"""Classical baseline: character n-gram TF-IDF + logistic regression, on the same split as train.py.

    python -m src.baseline --data data/urls.csv
"""
import argparse
import time
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .data import load_dataframe, split_dataframe
from .metrics import compute_metrics, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--url-col")
    p.add_argument("--label-col")
    p.add_argument("--invert-labels", action="store_true")
    p.add_argument("--drop-types", nargs="*")
    p.add_argument("--max-samples", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report-dir", default="reports")
    args = p.parse_args()

    df = load_dataframe(args.data, args.url_col, args.label_col, args.invert_labels, args.drop_types)
    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(args.max_samples, random_state=args.seed).reset_index(drop=True)
    train_df, _, test_df = split_dataframe(df, seed=args.seed)

    clf = make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(1, 5), max_features=200_000, sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    t0 = time.time()
    clf.fit(train_df["url"], train_df["label"])
    print(f"[baseline] trained in {time.time() - t0:.0f}s")
    probs = clf.predict_proba(test_df["url"])[:, 1]
    metrics = compute_metrics(test_df["label"].values, probs)
    save_json({"run": "tfidf_logreg", "arch": "tfidf_logreg", "test": metrics},
              Path(args.report_dir) / "tfidf_logreg" / "metrics.json")
    print(" | ".join(f"{k} {metrics[k]:.4f}" for k in ("accuracy", "precision", "recall", "f1", "roc_auc")))


if __name__ == "__main__":
    main()
