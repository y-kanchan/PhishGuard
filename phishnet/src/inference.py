"""Load a trained checkpoint and produce explained predictions."""
import time
from pathlib import Path

import numpy as np
import torch

from .data import CharTokenizer, normalize_url
from .explain import (integrated_gradients, lexical_indicators, neutral_baseline, summarize, token_attributions,
                      token_occlusion)
from .model import build_model


def calibration_texts(n=1000, seed=0):
    """Normalised URLs used to calibrate the IG baseline: a PhiUSIIL sample if present, else a small built-in list."""
    path = Path(__file__).resolve().parent.parent / "data" / "raw" / "PhiUSIIL_Phishing_URL_Dataset.csv"
    if path.exists():
        import pandas as pd
        urls = pd.read_csv(path, usecols=["URL"])["URL"].sample(n, random_state=seed)
        return [normalize_url(u) for u in urls]
    return [normalize_url(u) for u in (
        "https://www.wikipedia.org", "https://www.bbc.co.uk", "github.com/pytorch/pytorch", "secure-login.top/verify",
        "paypal.account-update.xyz/signin", "https://www.nasa.gov", "office365-auth.web.app", "amazon.com/gp/help")]


class Predictor:
    def __init__(self, checkpoint_path, device=None, threshold=None, ig_steps=32):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = build_model(ckpt["model_config"]).to(self.device).eval()
        self.model.load_state_dict(ckpt["state_dict"])
        self.tokenizer = CharTokenizer(max_len=ckpt["max_len"])
        self.threshold = float(threshold if threshold is not None else ckpt.get("threshold", 0.5))
        self.ig_steps = ig_steps
        base = ckpt.get("ig_baseline")
        if base is None:
            base = neutral_baseline(self.model, self.tokenizer, calibration_texts())[0]
        self.ig_baseline = torch.as_tensor(base, dtype=torch.float32, device=self.device)
        self.info = {"arch": self.model.arch, "max_len": ckpt["max_len"], "threshold": self.threshold,
                     "parameters": sum(p.numel() for p in self.model.parameters()),
                     "test_metrics": ckpt.get("test_metrics"), "device": str(self.device)}

    def _encode(self, texts):
        return torch.from_numpy(self.tokenizer.encode_batch(texts, dynamic=True)).to(self.device)

    @torch.no_grad()
    def predict_proba(self, urls, batch_size=512):
        texts = [normalize_url(u) or "?" for u in urls]
        order = np.argsort([len(t) for t in texts])
        probs = np.zeros(len(texts), dtype=np.float64)
        for i in range(0, len(texts), batch_size):
            idx = order[i:i + batch_size]
            probs[idx] = torch.sigmoid(self.model(self._encode([texts[j] for j in idx]))).cpu().numpy()
        return probs.tolist()

    def label(self, prob):
        return "phishing" if prob >= self.threshold else "legitimate"

    def predict(self, url: str, explain: bool = True) -> dict:
        start = time.perf_counter()
        norm = normalize_url(url) or "?"
        ids = self._encode([norm])
        with torch.no_grad():
            logit, attn = self.model(ids, return_attention=True)
        prob = torch.sigmoid(logit).item()
        label = self.label(prob)
        result = {"url": url, "normalized": norm, "probability": prob, "label": label,
                  "threshold": self.threshold, "truncated": len(norm) > self.tokenizer.max_len}

        if explain:
            n = int((ids != 0).sum())
            char_attr = integrated_gradients(self.model, ids, self.ig_steps, self.ig_baseline).cpu().tolist()
            analyzed = norm[:n]
            tokens = token_attributions(analyzed, char_attr)
            for t, occ in zip(tokens, token_occlusion(self.model, self.tokenizer, analyzed, tokens, logit.item())):
                t["ig"], t["score"] = t["score"], occ
            sign = 1 if label == "phishing" else -1
            top = sorted((t for t in tokens if sign * t["score"] > 0), key=lambda t: -sign * t["score"])[:5]
            indicators = lexical_indicators(url)
            result["explanation"] = {
                "analyzed_text": analyzed,
                "char_attributions": char_attr,
                "attention": attn[0, :n].cpu().tolist(),
                "top_tokens": top,
                "tokens": tokens,
                "indicators": indicators,
                "summary": summarize(label, prob, top, indicators),
            }
        result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
        return result
