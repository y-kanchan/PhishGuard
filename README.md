# PhishGuard

**Explainable Real-Time Phishing URL Detection Using a Hybrid CNN-BiLSTM Deep Learning Model**

PhishGuard classifies a URL as *phishing* or *legitimate* using only its characters. It needs no page
download, WHOIS lookup or third-party lookup, so it can score a URL in milliseconds, before the page loads.
Every prediction comes with an explanation.

## Architecture

```
URL ─► normalise (lowercase, strip scheme + www.) ─► char ids (max 200)
   ─► Embedding (64-d)
   ─► 3 parallel Conv1D branches, kernel 3/5/7, 128 filters each, ReLU, BatchNorm   ← local n-gram patterns
   ─► BiLSTM (128 units per direction)                                              ← sequential context
   ─► Additive attention pooling                                                    ← per-character weights
   ─► Dense(64, ReLU) ─► Dropout ─► Dense(1) ─► sigmoid = P(phishing)
```

* **CNN** detects local patterns such as `paypa1`, `-login-`, `.tk/`, `xn--` and IP octets.
* **BiLSTM** models how those patterns are arranged across the URL, for example a brand
  that appears as a subdomain of an unrelated domain.
* **Attention** pools the sequence and shows which positions the model used.
* The convolutions use "same" padding, so each time step still lines up with one character.
  This lets the explanations be drawn on the URL itself.

### Explainability

| Method | What it shows |
|---|---|
| Integrated Gradients (on embeddings, 32 steps) | Signed contribution of every character to the phishing score |
| Token aggregation | IG summed over URL tokens (`paypal`, `verify`, `tk`, …) gives the "most influential tokens" |
| Attention weights | Where the pooling layer focused |
| Lexical indicators | Rule-based red flags (IP host, `@`, punycode, suspicious TLD, brand outside the registered domain, …). Shown as supporting evidence only. **They are not model inputs.** |

### Design choices worth mentioning in a report

* **Scheme stripping:** several public datasets store phishing URLs with `http://` and benign
  URLs without a scheme. A model trained on raw strings learns that artifact and not phishing patterns.
* **De-duplication and conflict removal:** URLs are de-duplicated after normalisation, and
  URLs that appear with both labels are dropped. This stops the same URL from appearing in train and test.
* **Class imbalance:** handled with `pos_weight` in the loss. The decision threshold is tuned on the
  validation set for best F1 and stored in the checkpoint.
* **Ablations:** `--arch cnn` and `--arch bilstm` train the single-branch variants.
  `src.baseline` trains a TF-IDF char n-gram + logistic regression baseline on the same split.

## Project layout

```
src/
  data.py        CSV loading, label mapping, normalisation, char tokenizer, splits
  model.py       PhishGuardNet (hybrid / cnn / bilstm)
  train.py       training loop, early stopping, threshold tuning, test report + plots
  explain.py     Integrated Gradients, token attribution, lexical indicators
  inference.py   Predictor class used by the CLI and the API
  metrics.py     metrics + confusion matrix / ROC / PR / training-curve plots
  evaluate.py    evaluate a checkpoint on another dataset (generalisation)
  baseline.py    TF-IDF + logistic regression baseline
  compare.py     builds reports/comparison.md from all runs
  predict.py     command-line explained prediction
app/server.py    FastAPI real-time API + web UI (app/static/index.html)
extension/       Chrome extension (Manifest V3), on-device ONNX inference
scripts/
  make_demo_data.py      synthetic data, for smoke tests only
  export_onnx.py         checkpoint -> extension/model/phishguard.onnx (+ parity check)
  make_icons.py          extension icons
  package_extension.py   zip for the Chrome Web Store
```

## Setup

```bash
pip install -r requirements.txt
```

## 1. Get a dataset

Put a CSV in `data/`. Column names are auto-detected (`url`/`domain`, `label`/`type`/`status`/`class`).
String labels such as `phishing`, `malicious`, `benign` and `legitimate` and 0/1 labels are all mapped automatically.

| Dataset | Where | Notes |
|---|---|---|
| **PhiUSIIL Phishing URL** (≈235k) | UCI ML Repository | `label` uses **1 = legitimate**, so pass `--invert-labels --url-col URL --label-col label` |
| **Malicious URLs dataset** (≈651k) | Kaggle, `malicious_phish.csv` | `type` column. Add `--drop-types malware defacement` for a pure phishing-vs-benign task |
| **Phishing Site URLs** (≈549k) | Kaggle, `phishing_site_urls.csv` | `Label` = `bad` / `good` |
| PhishTank / OpenPhish + Tranco top sites | phishtank.org, tranco-list.eu | Build your own fresh dataset from live feeds |

For a quick check that everything runs, use synthetic data. **Its scores mean nothing:**

```bash
python scripts/make_demo_data.py --n 6000
```

## 2. Train

```bash
python -m src.train --data data/malicious_phish.csv --drop-types malware defacement
# fast experiment on a CPU:
python -m src.train --data data/malicious_phish.csv --max-samples 60000 --epochs 8
```

This writes `models/phishguard_hybrid.pt` and puts the following in `reports/hybrid/`:
`metrics.json`, `confusion_matrix.png`, `roc_pr_curves.png` and `training_curves.png`.
A GPU is strongly recommended for the full dataset. On a CPU, use `--max-samples` or run it on Google Colab.

## 3. Ablations, baseline and comparison table

```bash
python -m src.train    --data data/urls.csv --arch cnn
python -m src.train    --data data/urls.csv --arch bilstm
python -m src.baseline --data data/urls.csv
python -m src.compare               # -> reports/comparison.md
```

Use the same `--data`, filtering flags, `--max-samples` and `--seed` for every run so they share one split.

## 4. Explain a prediction

```bash
python -m src.predict "http://paypal.com.verify-account.tk/webscr/login.php" "https://github.com"
```

## 5. Real-time web app

```bash
uvicorn app.server:app --port 8000
```

Open http://localhost:8000 to see the verdict, the probability, a character heatmap (red pushes toward phishing,
green toward legitimate), attention weights, top tokens and red flags.

API:

```
POST /api/predict   {"url": "...", "explain": true}
POST /api/batch     {"urls": ["...", "..."]}
GET  /api/health
```

## 6. Chrome extension (runs fully on-device)

The extension contains the trained model as ONNX and runs it in the browser with onnxruntime-web (WebAssembly).
It needs no server, and no browsing data leaves the machine.

**Build after each training run**, then reload the extension:

```bash
python scripts/export_onnx.py --model models/phishguard_hybrid.pt   # -> extension/model/, checks ONNX == PyTorch
python scripts/make_icons.py                                         # once
```

**Install:** go to `chrome://extensions`, turn on **Developer mode**, click **Load unpacked** and select `extension/`.

**Features**

* Scans every page as navigation starts (`webNavigation.onBeforeNavigate`), then re-checks the final URL after redirects.
* The toolbar badge shows **OK** or **!**.
* Phishing URLs are redirected to a **warning page** that explains the verdict. The page offers *Back to safety* or
  *Continue anyway*; continuing adds the host to the trusted list.
* The **popup** shows the verdict for the current tab, a heatmap, top tokens and red flags. It also has a box for checking any URL,
  switches for protection and warnings, scan statistics and trusted-site management.

**How it works**

```
background.js (service worker)  ── navigation events, badge, warning redirect, settings, result cache
      │ chrome.runtime messages
offscreen.html + engine.js       ── onnxruntime-web + model; the service worker cannot load WASM, so the model runs in a hidden offscreen page
popup.html / warning.html        ── UI (ui.js, styles.css)
```

`engine.js` re-implements the Python normalisation, tokenizer and lexical rules. Integrated Gradients needs
gradients, which onnxruntime-web does not provide, so the extension uses **occlusion**. Each URL token is replaced
by the unknown-character id, and the change in the phishing logit is that token's score. All occluded variants
run as one batch, which takes about 50 ms per URL.

The bundled `lib/` files come from `onnxruntime-web@1.20.1` (`dist/ort.wasm.min.js`, `ort-wasm-simd-threaded.wasm`,
`ort-wasm-simd-threaded.mjs`). Chrome extensions cannot load remote code, so these files are shipped inside the extension.

**Publish:** `python scripts/package_extension.py` writes `dist/phishguard-extension-<version>.zip` for the
Chrome Web Store developer dashboard. The store listing needs a privacy statement. Because everything runs on-device,
it can state that no browsing data is collected or transmitted.

## 7. Generalisation check

Train on one dataset and test on another. This is the most honest measure of real-world performance:

```bash
python -m src.evaluate --data data/phiusiil.csv --url-col URL --invert-labels
```

## Limitations

* Only the URL is used. A phishing page hosted on a compromised legitimate domain with a clean-looking path
  is hard to catch from the URL alone.
* Attention weights are not a faithful explanation by themselves, which is why IG is the primary explanation.
* Phishing patterns change over time, so retrain periodically on fresh feeds.
