"""Explain a prediction from the command line.

    python -m src.predict "http://paypal-secure-login.xyz/verify"
"""
import argparse
import json

from .inference import Predictor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("urls", nargs="+")
    p.add_argument("--model", default="models/phishguard_hybrid.pt")
    p.add_argument("--json", action="store_true", help="print the full JSON result")
    args = p.parse_args()

    predictor = Predictor(args.model)
    for url in args.urls:
        r = predictor.predict(url)
        if args.json:
            print(json.dumps(r, indent=2))
            continue
        e = r["explanation"]
        print(f"\n{url}\n  -> {r['label'].upper()}  p(phishing)={r['probability']:.4f}  ({r['latency_ms']} ms)")
        print(f"  {e['summary']}")
        for t in e["top_tokens"]:
            print(f"    token '{t['token']}': {t['score']:+.3f}")
        for i in e["indicators"]:
            print(f"    [!] {i['indicator']}: {i['detail']}")


if __name__ == "__main__":
    main()
