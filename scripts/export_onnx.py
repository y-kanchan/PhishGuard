"""Export a trained checkpoint to ONNX for the Chrome extension.

    python scripts/export_onnx.py --model models/phishguard_hybrid.pt

Writes extension/model/phishguard.onnx and extension/model/meta.json, then checks that
onnxruntime output matches PyTorch on a few URLs.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import CharTokenizer, normalize_url  # noqa: E402
from src.model import build_model  # noqa: E402

CHECK_URLS = ["http://paypal.com.verify-account.xyz/login/webscr?session=abc",
              "https://github.com/pytorch/pytorch", "http://192.168.12.4/apple/icloud/unlock", "google.com"]


class BrowserModel(nn.Module):
    """ids (batch, seq) with no padding -> (probability, attention)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ids):
        logits, attn = self.model.forward_from_embeddings(self.model.embedding(ids), ids != 0, True, pack=False)
        return torch.sigmoid(logits), attn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/phishguard_hybrid.pt")
    p.add_argument("--out-dir", default="extension/model")
    args = p.parse_args()

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model_config"]).eval()
    model.load_state_dict(ckpt["state_dict"])
    wrapper = BrowserModel(model).eval()
    tok = CharTokenizer(ckpt["max_len"])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = out / "phishguard.onnx"
    dummy = torch.from_numpy(tok.encode(CHECK_URLS[0])[None, :40])
    torch.onnx.export(wrapper, (dummy,), onnx_path, input_names=["ids"], output_names=["prob", "attention"],
                      dynamic_axes={"ids": {0: "batch", 1: "seq"}, "prob": {0: "batch"},
                                    "attention": {0: "batch", 1: "seq"}},
                      opset_version=17, dynamo=False)

    meta = {"alphabet": tok.alphabet, "max_len": ckpt["max_len"], "threshold": ckpt.get("threshold", 0.5),
            "arch": model.arch, "test_metrics": ckpt.get("test_metrics")}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=float))

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path))
    max_diff = 0.0
    for url in CHECK_URLS:
        n = min(len(normalize_url(url)), ckpt["max_len"])
        ids = tok.encode(url)[None, :n]
        with torch.no_grad():
            ref = torch.sigmoid(model(torch.from_numpy(tok.encode(url)[None]))).item()
        got = float(sess.run(None, {"ids": ids})[0][0])
        max_diff = max(max_diff, abs(ref - got))
        print(f"{ref:.6f}  {got:.6f}  {url}")
    print(f"max |pytorch - onnx| = {max_diff:.2e}")
    if max_diff > 1e-4:
        raise SystemExit("ONNX output does not match PyTorch")
    print(f"exported -> {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB) + meta.json")


if __name__ == "__main__":
    main()
