"""PhiUSIIL loading, URL normalisation, character tokenisation and path augmentation.

PhiUSIIL (Prasad & Chandra, 2023, UCI id 967) labels 1 = legitimate, 0 = phishing.
We flip it so 1 = phishing everywhere in this project.

Dataset artifact this module deals with
---------------------------------------
Every legitimate URL in PhiUSIIL is a bare homepage of the form ``https://www.<domain>``
(100 % https, 100 % www, 0 % with a path), while phishing URLs come with and without
scheme, www and paths. A model trained on the raw strings learns "starts with https://www"
and "has a path" instead of phishing patterns. We therefore

  1. strip the scheme and a leading ``www.`` (normalize_url), and
  2. optionally append synthetic, neutral paths to a share of the *training* legitimate
     URLs (PathAugmenter) so that "has a path" alone stops being a phishing signal.

The validation and test sets are never augmented.
"""
import random
import re
import string

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

PAD, UNK = 0, 1

_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.\-]*://")
_WWW_RE = re.compile(r"^www\d?\.")


def normalize_url(url: str) -> str:
    """Lowercase, strip scheme and leading 'www.', drop trailing slashes."""
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

    def encode(self, text: str, length: int | None = None) -> np.ndarray:
        """Encode an already-normalised URL, padded/truncated to `length` (default max_len)."""
        length = length or self.max_len
        ids = np.zeros(length, dtype=np.int64)
        for i, ch in enumerate(text[: min(length, self.max_len)]):
            ids[i] = self.char2idx.get(ch, UNK)
        return ids

    def encode_batch(self, texts, dynamic: bool = False) -> np.ndarray:
        """Encode normalised URLs. dynamic=True pads only to the longest URL in the batch."""
        if not len(texts):
            return np.zeros((0, self.max_len), np.int64)
        length = min(self.max_len, max(1, max(len(t) for t in texts))) if dynamic else self.max_len
        return np.stack([self.encode(t, length) for t in texts])


# --------------------------------------------------------------------------- loading

def load_phiusiil(path, url_col="URL", label_col="label") -> pd.DataFrame:
    """Return a clean DataFrame [url, label] (1 = phishing) with normalised, de-duplicated URLs.

    URLs that occur with both labels after normalisation are dropped, so the same string
    can never appear in train and test with different labels.
    """
    df = pd.read_csv(path, usecols=[url_col, label_col]).dropna()
    df = df.rename(columns={url_col: "raw_url", label_col: "label"})
    df["label"] = 1 - df["label"].astype(int)          # PhiUSIIL: 1 = legitimate
    df["url"] = df["raw_url"].astype(str).map(normalize_url)
    df = df[df["url"].str.len() > 0]
    conflicts = df.groupby("url")["label"].nunique()
    df = df[~df["url"].isin(conflicts[conflicts > 1].index)]
    df = df.drop_duplicates(subset="url").reset_index(drop=True)
    return df[["url", "raw_url", "label"]]


def split_dataframe(df, val_size=0.1, test_size=0.1, seed=42):
    train_df, temp_df = train_test_split(df, test_size=val_size + test_size, stratify=df["label"], random_state=seed)
    rel_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(temp_df, test_size=rel_test, stratify=temp_df["label"], random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# --------------------------------------------------------------------------- augmentation

# Neutral words only - no login/verify/account style words, so those stay informative.
TRAIN_PATH_WORDS = (
    "about", "contact", "products", "product", "blog", "news", "en", "us", "help", "docs", "search", "category",
    "article", "articles", "shop", "store", "events", "team", "careers", "faq", "services", "research", "index.html",
    "page", "pages", "media", "press", "gallery", "resources", "downloads", "support", "en-us", "de", "fr", "posts",
    "item", "p", "video", "videos", "collections", "tags", "wiki", "projects", "library", "courses", "info",
)
# Disjoint vocabulary used only for the robustness check on the test set.
HOLDOUT_PATH_WORDS = (
    "menu", "recipes", "location", "history", "partners", "investors", "community", "forum", "topics", "guides",
    "catalog", "brands", "stories", "reviews", "calendar", "jobs", "publications", "photos", "es", "it",
)


class PathAugmenter:
    """Appends a random neutral path / query to a legitimate URL."""

    def __init__(self, words=TRAIN_PATH_WORDS, seed=None):
        self.words = words
        self.rng = random.Random(seed)

    def _slug(self):
        r = self.rng
        kind = r.random()
        if kind < 0.35:
            return str(r.randint(1, 99999))
        if kind < 0.55:
            return "-".join(r.choice(self.words) for _ in range(r.randint(2, 4)))
        if kind < 0.65:
            return "".join(r.choices(string.ascii_lowercase + string.digits, k=r.randint(6, 12)))
        return r.choice(self.words)

    def __call__(self, url: str) -> str:
        r = self.rng
        parts = [self._slug() for _ in range(r.randint(1, 4))]
        out = url + "/" + "/".join(parts)
        if r.random() < 0.15:
            out += r.choice((".html", ".php", ".aspx", "/"))
        if r.random() < 0.2:
            out += f"?{r.choice(('id', 'page', 'ref', 'q', 'lang', 'utm_source'))}={r.choice(self.words + ('2', '17', 'home'))}"
        return out


# --------------------------------------------------------------------------- torch data

class URLDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame):
        self.urls = df["url"].tolist()
        self.labels = df["label"].astype(np.float32).tolist()

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        return self.urls[idx], self.labels[idx]


class Collator:
    """Encodes a batch with dynamic padding; optionally augments legitimate URLs (training only)."""

    def __init__(self, tokenizer: CharTokenizer, augment_prob: float = 0.0, seed: int = 0):
        self.tokenizer = tokenizer
        self.augment_prob = augment_prob
        self.augmenter = PathAugmenter(seed=seed)
        self.rng = random.Random(seed + 1)

    def __call__(self, batch):
        urls, labels = zip(*batch)
        if self.augment_prob > 0:
            urls = [self.augmenter(u) if y == 0 and self.rng.random() < self.augment_prob else u
                    for u, y in zip(urls, labels)]
        x = torch.from_numpy(self.tokenizer.encode_batch(list(urls), dynamic=True))
        return x, torch.tensor(labels, dtype=torch.float32)


class BucketSampler(torch.utils.data.Sampler):
    """Groups URLs of similar length into batches, so dynamic padding wastes little compute."""

    def __init__(self, lengths, batch_size, shuffle=True, seed=0):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        idx = self.rng.permutation(len(self.lengths)) if self.shuffle else np.arange(len(self.lengths))
        chunk = self.batch_size * 50
        batches = []
        for s in range(0, len(idx), chunk):
            part = idx[s:s + chunk]
            part = part[np.argsort(self.lengths[part], kind="stable")]
            batches += [part[i:i + self.batch_size].tolist() for i in range(0, len(part), self.batch_size)]
        if self.shuffle:
            self.rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def make_loader(df, tokenizer, batch_size, train=False, augment_prob=0.0, seed=0):
    ds = URLDataset(df)
    # Augmented URLs get longer; use a rough length estimate for bucketing.
    lengths = [min(len(u), tokenizer.max_len) for u in ds.urls]
    sampler = BucketSampler(lengths, batch_size, shuffle=train, seed=seed)
    collate = Collator(tokenizer, augment_prob if train else 0.0, seed)
    return torch.utils.data.DataLoader(ds, batch_sampler=sampler, collate_fn=collate)
