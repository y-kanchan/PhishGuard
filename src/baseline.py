"""Classical baseline: char n-gram TF-IDF + logistic regression on the same split and augmentation.

    python -m src.baseline
"""
import argparse
import random
import time
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .data import PathAugmenter, load_phiusiil, split_dataframe
from .metrics import best_f1_threshold, compute_metrics, curve_points, save_json
from .train import DEFAULT_DATA


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--augment-prob", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report-dir", default="reports")
    args = p.parse_args()

    df = load_phiusiil(args.data)
    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(args.max_samples, random_state=args.seed).reset_index(drop=True)
    train_df, val_df, test_df = split_dataframe(df, seed=args.seed)

    aug, rng = PathAugmenter(seed=args.seed), random.Random(args.seed + 1)
    x_train = [aug(u) if y == 0 and rng.random() < args.augment_prob else u
               for u, y in zip(train_df["url"], train_df["label"])]

    clf = make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(1, 5), max_features=200_000, sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    t0 = time.time()
    clf.fit(x_train, train_df["label"])
    print(f"[baseline] trained in {time.time() - t0:.0f}s")
    threshold = best_f1_threshold(val_df["label"].values, clf.predict_proba(val_df["url"])[:, 1])
    probs = clf.predict_proba(test_df["url"])[:, 1]
    metrics = compute_metrics(test_df["label"].values, probs, threshold)
    save_json({"run": "tfidf_logreg", "arch": "tfidf_logreg", "test": metrics,
               "curves": curve_points(test_df["label"].values, probs), "train_seconds": time.time() - t0},
              Path(args.report_dir) / "tfidf_logreg" / "metrics.json")
    print(" | ".join(f"{k} {metrics[k]:.4f}" for k in ("accuracy", "precision", "recall", "f1", "roc_auc")))


if __name__ == "__main__":
    main()
