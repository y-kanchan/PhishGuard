"""Dataset loading, URL normalisation and character-level tokenisation."""
import re
import string

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

PAD, UNK = 0, 1

_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.\-]*://")
_WWW_RE = re.compile(r"^www\d?\.")

URL_COLUMNS = ["url", "urls", "domain", "link", "website"]
LABEL_COLUMNS = ["label", "type", "status", "class", "result", "is_phishing", "phishing", "target"]

POSITIVE_LABELS = {"phishing", "phish", "malicious", "bad", "spam", "malware", "defacement", "1", "true", "yes"}
NEGATIVE_LABELS = {"benign", "legitimate", "legit", "good", "safe", "0", "false", "no", "-1"}


def normalize_url(url: str) -> str:
    """Lowercase and strip scheme / leading 'www.'.

    Many public datasets store phishing URLs with 'http://' and benign ones without
    (or vice versa). Stripping the scheme stops the model from learning that artifact.
    """
    url = str(url).strip().lower()
    url = _SCHEME_RE.sub("", url)
    url = _WWW_RE.sub("", url)
    return url.rstrip("/")


class CharTokenizer:
    """Maps each character of a normalised URL to an integer id (0 = pad, 1 = unknown)."""

    def __init__(self, max_len: int = 200):
        self.max_len = max_len
        self.alphabet = string.ascii_lowercase + string.digits + string.punctuation
        self.char2idx = {c: i + 2 for i, c in enumerate(self.alphabet)}

    @property
    def vocab_size(self) -> int:
        return len(self.alphabet) + 2

    def encode(self, url: str, normalized: bool = False) -> np.ndarray:
        text = url if normalized else normalize_url(url)
        ids = np.zeros(self.max_len, dtype=np.int64)
        for i, ch in enumerate(text[: self.max_len]):
            ids[i] = self.char2idx.get(ch, UNK)
        return ids

    def encode_batch(self, urls, normalized: bool = False) -> np.ndarray:
        return np.stack([self.encode(u, normalized) for u in urls]) if len(urls) else np.zeros((0, self.max_len), np.int64)


def _find_column(columns, candidates, override, kind):
    if override:
        if override not in columns:
            raise ValueError(f"{kind} column '{override}' not found. Available: {list(columns)}")
        return override
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    raise ValueError(f"Could not detect the {kind} column in {list(columns)}; pass it explicitly.")


def _map_label(value):
    s = str(value).strip().lower()
    if s.endswith(".0"):
        s = s[:-2]
    if s in POSITIVE_LABELS:
        return 1
    if s in NEGATIVE_LABELS:
        return 0
    return None


def load_dataframe(path, url_col=None, label_col=None, invert_labels=False, drop_types=None) -> pd.DataFrame:
    """Load a CSV of URLs and return a clean DataFrame with columns ['url', 'label'] (1 = phishing).

    invert_labels: use for datasets where 1 means legitimate (e.g. PhiUSIIL).
    drop_types: raw label values to discard (e.g. ['malware', 'defacement']).
    """
    df = pd.read_csv(path, low_memory=False)
    ucol = _find_column(df.columns, URL_COLUMNS, url_col, "URL")
    lcol = _find_column(df.columns, LABEL_COLUMNS, label_col, "label")
    df = df[[ucol, lcol]].rename(columns={ucol: "url", lcol: "raw_label"}).dropna()

    if drop_types:
        drop = {t.strip().lower() for t in drop_types}
        df = df[~df["raw_label"].astype(str).str.strip().str.lower().isin(drop)]

    df["label"] = df["raw_label"].map(_map_label)
    unmapped = df["label"].isna().sum()
    if unmapped:
        bad = df.loc[df["label"].isna(), "raw_label"].astype(str).unique()[:10]
        print(f"[data] dropping {unmapped} rows with unrecognised labels, e.g. {list(bad)}")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    if invert_labels:
        df["label"] = 1 - df["label"]

    df["url"] = df["url"].astype(str).map(normalize_url)
    df = df[df["url"].str.len() > 0]

    # Drop URLs whose duplicates carry conflicting labels, then de-duplicate.
    conflicts = df.groupby("url")["label"].nunique()
    df = df[~df["url"].isin(conflicts[conflicts > 1].index)]
    df = df.drop_duplicates(subset="url").reset_index(drop=True)
    return df[["url", "label"]]


def split_dataframe(df, val_size=0.1, test_size=0.1, seed=42):
    train_df, temp_df = train_test_split(df, test_size=val_size + test_size, stratify=df["label"], random_state=seed)
    rel_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(temp_df, test_size=rel_test, stratify=temp_df["label"], random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


class URLDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: CharTokenizer):
        self.x = torch.from_numpy(tokenizer.encode_batch(df["url"].tolist(), normalized=True))
        self.y = torch.tensor(df["label"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
