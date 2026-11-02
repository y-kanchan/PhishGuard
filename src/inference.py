"""Load a trained checkpoint and produce explained predictions."""
import time

import torch

from .data import CharTokenizer, normalize_url
from .explain import integrated_gradients, lexical_indicators, summarize, token_attributions
from .model import build_model


def get_device(name=None):
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Predictor:
    def __init__(self, checkpoint_path, device=None, threshold=None, ig_steps=32):
        self.device = get_device(device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = build_model(ckpt["model_config"]).to(self.device).eval()
        self.model.load_state_dict(ckpt["state_dict"])
        self.tokenizer = CharTokenizer(max_len=ckpt["max_len"])
        self.threshold = threshold if threshold is not None else ckpt.get("threshold", 0.5)
        self.ig_steps = ig_steps
        self.info = {"arch": self.model.arch, "max_len": ckpt["max_len"], "threshold": self.threshold,
                     "test_metrics": ckpt.get("test_metrics"), "device": str(self.device)}

    def _encode(self, urls):
        return torch.from_numpy(self.tokenizer.encode_batch(urls)).to(self.device)

    @torch.no_grad()
    def predict_proba(self, urls, batch_size=512):
        probs = []
        for i in range(0, len(urls), batch_size):
            probs.append(torch.sigmoid(self.model(self._encode(urls[i:i + batch_size]))).cpu())
        return torch.cat(probs).tolist() if probs else []

    def predict(self, url: str, explain: bool = True) -> dict:
        start = time.perf_counter()
        norm = normalize_url(url)
        ids = self._encode([url])
        with torch.no_grad():
            logit, attn = self.model(ids, return_attention=True)
        prob = torch.sigmoid(logit).item()
        label = "phishing" if prob >= self.threshold else "legitimate"
        result = {"url": url, "normalized": norm, "probability": prob, "label": label, "threshold": self.threshold}

        if explain:
            n = int((ids != 0).sum())
            char_attr = integrated_gradients(self.model, ids, self.ig_steps).cpu().tolist()
            tokens = token_attributions(norm, char_attr)
            # Tokens that pushed the decision toward the predicted class.
            sign = 1 if label == "phishing" else -1
            top = sorted((t for t in tokens if sign * t["score"] > 0), key=lambda t: -sign * t["score"])[:5]
            indicators = lexical_indicators(url)
            result["explanation"] = {
                "analyzed_text": norm[:n],
                "char_attributions": char_attr,
                "attention": attn[0, :n].cpu().tolist(),
                "top_tokens": top,
                "tokens": tokens,
                "indicators": indicators,
                "summary": summarize(label, prob, top, indicators),
            }
        result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
        return result
