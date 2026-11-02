"""Generate a SYNTHETIC URL dataset for smoke-testing the pipeline only.

The patterns are hand-written, so scores on this data are meaningless.
Use a real dataset (see README) for any reported results.

    python scripts/make_demo_data.py --n 8000 --out data/demo_urls.csv
"""
import argparse
import csv
import random
from pathlib import Path

DOMAINS = ["google.com", "youtube.com", "wikipedia.org", "github.com", "stackoverflow.com", "amazon.com",
           "microsoft.com", "apple.com", "reddit.com", "nytimes.com", "bbc.co.uk", "linkedin.com", "paypal.com",
           "netflix.com", "python.org", "pytorch.org", "kaggle.com", "medium.com", "imdb.com", "spotify.com",
           "dropbox.com", "adobe.com", "cnn.com", "espn.com", "booking.com", "airbnb.com", "ebay.com",
           "instagram.com", "facebook.com", "twitter.com", "quora.com", "zoom.us", "slack.com", "notion.so",
           "arxiv.org", "nature.com", "mit.edu", "stanford.edu", "irctc.co.in", "flipkart.com", "zomato.com"]
WORDS = ["news", "blog", "article", "docs", "help", "products", "about", "search", "watch", "wiki", "questions",
         "careers", "store", "music", "video", "research", "papers", "learn", "tutorial", "events", "sports",
         "travel", "weather", "photos", "profile", "settings", "download", "release", "community", "pricing"]
BRANDS = ["paypal", "apple", "microsoft", "amazon", "netflix", "chase", "office365", "dhl", "icloud", "sbi", "hdfc"]
BAIT = ["login", "signin", "verify", "secure", "account", "update", "confirm", "unlock", "billing", "support"]
TLDS = ["xyz", "tk", "ml", "top", "info", "online", "site", "club", "ga", "cf", "com", "net"]


def benign(r):
    host = r.choice(DOMAINS)
    if r.random() < 0.3:
        host = r.choice(["en", "docs", "blog", "m", "support", "shop"]) + "." + host
    path = "/".join(r.choice(WORDS) for _ in range(r.randint(0, 3)))
    if r.random() < 0.3:
        path += f"/{r.randint(1, 99999)}"
    if r.random() < 0.2:
        path += f"?id={r.randint(1, 9999)}&lang=en"
    return f"{r.choice(['https://', 'http://', ''])}{host}/{path}".rstrip("/")


def phishing(r):
    brand, bait = r.choice(BRANDS), r.choice(BAIT)
    style = r.randint(0, 5)
    if style == 0:
        host = f"{brand}-{bait}-{r.choice(BAIT)}.{r.choice(TLDS)}"
    elif style == 1:
        host = f"{brand}.com.{bait}-{r.randint(10, 999)}.{r.choice(TLDS)}"
    elif style == 2:
        host = ".".join(str(r.randint(1, 255)) for _ in range(4))
    elif style == 3:
        host = f"{bait}.{brand}{r.randint(1, 99)}.{r.choice(TLDS)}"
    elif style == 4:
        host = f"{r.choice(DOMAINS)}@{brand}-{bait}.{r.choice(TLDS)}"
    else:
        host = "".join(r.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(r.randint(8, 16))) + "." + r.choice(TLDS)
    path = "/".join(r.choice(BAIT + [brand, "wp-content", "includes", "index.php", "webscr"]) for _ in range(r.randint(1, 4)))
    if r.random() < 0.4:
        path += f"?session={r.getrandbits(64):x}"
    return f"{r.choice(['http://', 'https://', ''])}{host}/{path}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8000)
    p.add_argument("--phish-ratio", type=float, default=0.4)
    p.add_argument("--out", default="data/demo_urls.csv")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    r = random.Random(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["url", "label"])
        for _ in range(args.n):
            is_phish = r.random() < args.phish_ratio
            w.writerow([phishing(r) if is_phish else benign(r), "phishing" if is_phish else "benign"])
    print(f"wrote {args.n} synthetic URLs -> {args.out}")


if __name__ == "__main__":
    main()
