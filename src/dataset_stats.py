"""Profile PhiUSIIL for the platform's Dataset page -> reports/dataset.json

    python -m src.dataset_stats
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_phiusiil, normalize_url
from .metrics import save_json
from .train import DEFAULT_DATA


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--out", default="reports/dataset.json")
    args = p.parse_args()

    raw = pd.read_csv(args.data, usecols=["URL", "label", "TLD"])
    raw["phish"] = 1 - raw["label"]
    clean = load_phiusiil(args.data)

    def share(mask_fn):
        return {name: float(mask_fn(g["URL"]).mean()) for name, g in
                (("legitimate", raw[raw.phish == 0]), ("phishing", raw[raw.phish == 1]))}

    strip = raw["URL"].str.lower().str.replace(r"^[a-z]+://", "", regex=True).str.rstrip("/")
    raw["_has_path"] = strip.str.contains("/")
    artifacts = {
        "starts_https": share(lambda s: s.str.lower().str.startswith("https://")),
        "has_www": share(lambda s: s.str.lower().str.contains("://www.", regex=False)),
        "has_path": {"legitimate": float(raw.loc[raw.phish == 0, "_has_path"].mean()),
                     "phishing": float(raw.loc[raw.phish == 1, "_has_path"].mean())},
    }

    lens = clean["url"].str.len()
    bins = [0, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 10_000]
    length_hist = {name: np.histogram(lens[clean.label == lbl], bins)[0].tolist()
                   for name, lbl in (("legitimate", 0), ("phishing", 1))}

    tlds = {}
    for name, lbl in (("legitimate", 0), ("phishing", 1)):
        vc = raw.loc[raw.phish == lbl, "TLD"].astype(str).str.lower().value_counts()
        tlds[name] = [[k, int(v)] for k, v in vc.head(12).items()]

    norm_raw = raw["URL"].astype(str).map(normalize_url)
    out = {
        "name": "PhiUSIIL Phishing URL Dataset",
        "source": "UCI Machine Learning Repository, id 967 (Prasad & Chandra, 2023)",
        "raw_rows": int(len(raw)),
        "raw_counts": {"legitimate": int((raw.phish == 0).sum()), "phishing": int((raw.phish == 1).sum())},
        "unique_after_normalisation": int(len(clean)),
        "duplicates_removed": int(len(raw) - norm_raw.nunique()),
        "conflicting_removed": int(norm_raw.nunique() - len(clean)),
        "clean_counts": {"legitimate": int((clean.label == 0).sum()), "phishing": int((clean.label == 1).sum())},
        "artifacts": artifacts,
        "length_bins": bins[:-1] + ["200+"],
        "length_hist": length_hist,
        "median_length": {"legitimate": float(lens[clean.label == 0].median()),
                          "phishing": float(lens[clean.label == 1].median())},
        "top_tlds": tlds,
        "samples": {name: clean[clean.label == lbl]["raw_url"].sample(8, random_state=3).tolist()
                    for name, lbl in (("legitimate", 0), ("phishing", 1))},
    }
    save_json(out, Path(args.out))
    print(f"[dataset] {out['raw_rows']:,} rows -> {out['unique_after_normalisation']:,} unique -> {args.out}")


if __name__ == "__main__":
    main()
