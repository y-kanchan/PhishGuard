"""Evaluation metrics and report plots."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)


def compute_metrics(labels, probs, threshold=0.5) -> dict:
    labels, probs = np.asarray(labels), np.asarray(probs)
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "roc_auc": roc_auc_score(labels, probs) if len(set(labels)) > 1 else float("nan"),
        "pr_auc": average_precision_score(labels, probs),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def best_f1_threshold(labels, probs) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))]) if len(thresholds) else 0.5


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def plot_report(labels, probs, threshold, out_dir, history=None, title="CNN-LSTM"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels, probs = np.asarray(labels), np.asarray(probs)

    cm = confusion_matrix(labels, (probs >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Legitimate", "Phishing"])
    ax.set_yticks([0, 1], ["Legitimate", "Phishing"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{title} - confusion matrix")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(labels, probs)
    prec, rec, _ = precision_recall_curve(labels, probs)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(fpr, tpr, label=f"AUC = {roc_auc_score(labels, probs):.4f}")
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axes[0].legend(loc="lower right")
    axes[1].plot(rec, prec, label=f"AP = {average_precision_score(labels, probs):.4f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    axes[1].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "roc_pr_curves.png", dpi=150)
    plt.close(fig)

    if history:
        epochs = [h["epoch"] for h in history]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
        axes[0].plot(epochs, [h["val_loss"] for h in history], label="validation")
        axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss")
        axes[0].legend()
        axes[1].plot(epochs, [h["val_f1"] for h in history], label="val F1")
        axes[1].plot(epochs, [h["val_roc_auc"] for h in history], label="val ROC-AUC")
        axes[1].set(xlabel="Epoch", title="Validation metrics")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(out / "training_curves.png", dpi=150)
        plt.close(fig)


def curve_points(labels, probs, n=120) -> dict:
    """Down-sampled ROC / PR curves and a score histogram, for drawing in the web UI."""
    labels, probs = np.asarray(labels), np.asarray(probs)
    fpr, tpr, _ = roc_curve(labels, probs)
    prec, rec, _ = precision_recall_curve(labels, probs)

    def thin(xs, ys):
        idx = np.unique(np.linspace(0, len(xs) - 1, min(n, len(xs))).astype(int))
        return [[round(float(xs[i]), 5), round(float(ys[i]), 5)] for i in idx]

    bins = np.linspace(0, 1, 41)
    return {
        "roc": thin(fpr, tpr),
        "pr": thin(rec[::-1], prec[::-1]),
        "hist_bins": bins.round(3).tolist(),
        "hist_legit": np.histogram(probs[labels == 0], bins)[0].tolist(),
        "hist_phish": np.histogram(probs[labels == 1], bins)[0].tolist(),
    }
