"""Explainability for PhishGuard predictions.

Three complementary views:
  1. Integrated Gradients (Sundararajan et al., 2017) on the character embeddings - signed,
     per-character contribution to the phishing logit (+ pushes toward phishing).
  2. Attention weights - where the pooling layer looked (unsigned).
  3. Lexical indicators - human-readable rule checks (IP host, '@', suspicious TLD, ...).
     These are shown to the user as supporting evidence; they are NOT model inputs.
"""
import ipaddress
import re
from urllib.parse import urlsplit

import torch

TOKEN_RE = re.compile(r"[^/.\-_?=&:@~%+#;,!]+")

SUSPICIOUS_TLDS = {"tk", "ml", "ga", "cf", "gq", "xyz", "top", "zip", "mov", "click", "country", "work",
                   "support", "buzz", "rest", "cam", "icu", "monster", "live", "fit", "loan", "review"}
SHORTENERS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "cutt.ly", "rb.gy", "shorturl.at",
              "tiny.cc", "buff.ly", "rebrand.ly"}
SENSITIVE_WORDS = ("login", "signin", "sign-in", "verify", "verification", "account", "update", "secure",
                   "banking", "confirm", "password", "wallet", "suspend", "unlock", "billing", "webscr")
BRANDS = ("paypal", "apple", "icloud", "microsoft", "office365", "outlook", "amazon", "netflix", "google",
          "facebook", "instagram", "whatsapp", "linkedin", "dropbox", "chase", "wellsfargo", "bankofamerica",
          "dhl", "fedex", "coinbase", "binance", "metamask", "steam", "adobe", "docusign")


def integrated_gradients(model, ids: torch.Tensor, steps: int = 32) -> torch.Tensor:
    """Per-character IG attributions for a single encoded URL `ids` of shape (1, L)."""
    model.eval()
    mask = ids != 0
    with torch.no_grad():
        emb = model.embedding(ids)
    baseline = torch.zeros_like(emb)  # same as the padding embedding
    alphas = torch.linspace(0, 1, steps + 1, device=ids.device)[1:].view(-1, 1, 1)
    interp = (baseline + alphas * (emb - baseline)).requires_grad_(True)
    # cuDNN RNNs refuse backward in eval mode; fall back to the native kernel here.
    with torch.backends.cudnn.flags(enabled=False):
        logits = model.forward_from_embeddings(interp, mask.expand(steps, -1))
    (grads,) = torch.autograd.grad(logits.sum(), interp)
    attr = ((emb - baseline) * grads.mean(0, keepdim=True)).sum(-1).squeeze(0)
    return attr[mask[0]].detach()


def token_attributions(text: str, char_attr):
    """Aggregate per-character attributions into URL tokens (split on URL delimiters)."""
    tokens = []
    for m in TOKEN_RE.finditer(text[: len(char_attr)]):
        score = float(sum(char_attr[m.start():m.end()]))
        tokens.append({"token": m.group(), "start": m.start(), "end": m.end(), "score": score})
    return tokens


def _registered_domain(host: str) -> str:
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def lexical_indicators(raw_url: str):
    """Rule-based red flags for a human-readable explanation."""
    raw = str(raw_url).strip()
    had_scheme = "://" in raw
    parsed = urlsplit(raw if had_scheme else "http://" + raw)
    host = (parsed.hostname or "").lower()
    full = raw.lower()
    flags = []

    def add(name, detail):
        flags.append({"indicator": name, "detail": detail})

    try:
        ipaddress.ip_address(host)
        is_ip = True
        add("IP address as host", f"'{host}' is a raw IP instead of a domain name")
    except ValueError:
        is_ip = False
    if "@" in parsed.netloc or "@" in parsed.path:
        add("'@' symbol", "browsers ignore everything before '@' in the host part")
    if "xn--" in host:
        add("Punycode domain", "may be a homograph look-alike of a real domain")
    if not is_ip and host.count(".") >= 3:
        add("Many subdomains", f"{host.count('.')} dots in host '{host}'")
    if len(raw) > 75:
        add("Long URL", f"{len(raw)} characters (long URLs often hide the real destination)")
    if host.count("-") >= 2:
        add("Hyphenated domain", f"'{host}' contains {host.count('-')} hyphens")
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    if tld in SUSPICIOUS_TLDS:
        add("Suspicious TLD", f"'.{tld}' is frequently abused for phishing")
    reg = host if is_ip else _registered_domain(host)
    if reg in SHORTENERS:
        add("URL shortener", f"'{reg}' hides the final destination")
    words = [w for w in SENSITIVE_WORDS if w in full]
    if words:
        add("Sensitive keywords", "contains " + ", ".join(f"'{w}'" for w in words[:4]))
    brands = [b for b in BRANDS if b in full and not reg.startswith(b)]
    if brands:
        add("Brand name outside the real domain", f"mentions '{brands[0]}' but the registered domain is '{reg}'")
    if had_scheme and parsed.scheme == "http":
        add("No HTTPS", "connection is not encrypted")
    digits = sum(ch.isdigit() for ch in host)
    if not is_ip and host and (digits >= 5 or digits / len(host) > 0.3):
        add("Digit-heavy host", f"'{host}' contains {digits} digits")
    if "//" in parsed.path:
        add("Embedded redirect", "'//' inside the path can redirect to another site")
    try:
        if parsed.port and parsed.port not in (80, 443):
            add("Non-standard port", f"port {parsed.port}")
    except ValueError:
        pass
    return flags


def summarize(label: str, probability: float, top_tokens, indicators) -> str:
    conf = probability if label == "phishing" else 1 - probability
    text = f"Classified as {label.upper()} ({conf:.1%} confidence)."
    if top_tokens:
        text += " The model was most influenced by " + ", ".join(f"'{t['token']}'" for t in top_tokens[:3]) + "."
    if indicators:
        text += " Red flags: " + "; ".join(i["indicator"] for i in indicators[:4]) + "."
    elif label == "legitimate":
        text += " No lexical red flags were found."
    return text
