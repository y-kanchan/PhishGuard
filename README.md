# PhishNet: Explainable Deep Learning Phishing URL Detection Platform

PhisNet is an end-to-end, explainable deep learning platform that classifies URLs as **phishing** or **legitimate** using only character sequences. It evaluates links in milliseconds before the page loads—requiring no page downloading, DNS lookups, or third-party reputation feeds.

Every prediction is paired with real-time explainability, including **Integrated Gradients** attribution heatmaps, **occlusion token sensitivity**, and **lexical indicator red flags**.


---

## Key Features

- **Real-Time Detection:** Fast neural network inference scoring character sequences in milliseconds.
- **Deep Learning Architecture:** Hybrid 1D-CNN + Bidirectional LSTM with Additive Attention pooling.
- **Explainable AI (XAI):**
  - **Character Attribution Heatmaps:** Integrated Gradients on embeddings (32 steps) showing exact characters pushing toward phishing (red) vs legitimate (green).
  - **Influential Token Ranking:** Occlusion sensitivity testing which URL tokens have the strongest impact on the classification logit.
  - **Lexical Indicators:** Rule-based heuristics (punycode, suspicious TLDs, IP hosts, credentials in URL) providing transparent supporting context.
- **Interactive 5-Page Web Platform:**
  - **Scan:** Single URL instant scanner with real-time attribution and token breakdown.
  - **Batch:** Bulk URL evaluation with CSV upload, filtering, sorting, and CSV export.
  - **Model:** Live test-set metrics, confusion matrix, ROC-PR curves, training history, and architecture breakdown.
  - **Dataset:** Dataset analysis, cleaning funnel, URL length distribution, and top TLD statistics for PhiUSIIL.
  - **History:** Local SQLite scan history log (`data/history.db`) with quick search and metrics.
- **FastAPI Backend:** Fully typed REST API with interactive Swagger docs at `/docs`.

---

## Architecture

```
URL ─► normalise (lower-case, strip scheme and "www.") ─► character IDs (max 128)
    ─► Embedding (64-d)
    ─► 3 Parallel Conv1D branches (kernels 3, 5, 7 | 64 filters each | ReLU) + LayerNorm  (local n-grams)
    ─► Bidirectional LSTM (64 units per direction)                                         (sequential context)
    ─► Additive Attention Pooling                                                          (per-char weights)
    ─► Dense(64, ReLU) ─► Dropout(0.3) ─► Dense(1) ─► Sigmoid = P(phishing)
```

- **CNN branches:** Extract local character patterns such as `paypa1`, `-login-`, `.top/`, `xn--`, and IP octets.
- **BiLSTM layer:** Captures long-range syntactic relationships (e.g., brand names placed in subdomains of unrelated domains).
- **Attention pooling:** Automatically weighs influential positions across the entire sequence.
- **Character Alignment:** Same-padding ensures each time-step matches an input character for heatmaps.

---

## Evaluation (PhiUSIIL Dataset)

Trained and evaluated on the **PhiUSIIL Phishing URL Dataset** (UCI ML Repository #967, 231,747 unique URLs after deduplication; stratified 80 / 10 / 10 split).

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | False Positive Rate |
|---|---|---|---|---|---|---|---|
| **CNN-LSTM** (Hybrid with Attention) | **0.9017** | **0.9460** | **0.8112** | **0.8734** | **0.9408** | **0.9452** | **0.0333** |
| TF-IDF (1–5 grams) + Logistic Regression | 0.8948 | 0.9282 | 0.8111 | 0.8657 | 0.9406 | – | 0.0451 |

---

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Run the Web Platform

Start the web server:

```bash
python run.py
```

Or using uvicorn directly:

```bash
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000
```

Open **[http://localhost:8000](http://localhost:8000)** in your browser.  
API documentation is available at **[http://localhost:8000/docs](http://localhost:8000/docs)**.

---

## CLI Prediction

You can explain any URL directly from the terminal:

```bash
python -m src.predict "http://paypal-secure-login.xyz/verify" "https://github.com"
```

Output format with JSON:

```bash
python -m src.predict "http://paypal-secure-login.xyz/verify" --json
```

---

## Training & Evaluation

To retrain the model or test on another dataset:

1. Place the dataset in `data/raw/PhiUSIIL_Phishing_URL_Dataset.csv` (or any custom CSV).
2. Generate dataset statistics:
   ```bash
   python -m src.dataset_stats
   ```
3. Train the model:
   ```bash
   python -m src.train
   ```
   (Options: `--data <path>`, `--arch cnn_lstm|cnn|lstm`, `--epochs 10`, `--batch-size 128`)
4. Evaluate a checkpoint on an external dataset:
   ```bash
   python -m src.evaluate --data path/to/dataset.csv --url-col url --label-col label
   ```

---

## Project Structure

```
├── app/
│   ├── server.py             # FastAPI backend with REST endpoints
│   └── static/
│       ├── index.html        # Modern 5-page dashboard UI
│       ├── app.css           # Styling and layout
│       ├── app.js            # Frontend logic and API calls
│       └── charts.js         # Interactive canvas charts
├── data/
│   ├── history.db            # SQLite scan history database
│   └── raw/
│       └── README.md         # Instructions for raw dataset placement
├── models/
│   └── cnn_lstm.pt           # Trained PyTorch CNN-LSTM checkpoint
├── reports/
│   ├── cnn_lstm/             # Evaluation plots and metrics JSON
│   ├── tfidf_logreg/         # Baseline metrics
│   └── dataset.json          # Dataset statistics
├── src/
│   ├── baseline.py           # TF-IDF + Logistic Regression baseline
│   ├── data.py               # Preprocessing, normalization, tokenization
│   ├── dataset_stats.py      # Computes dataset profiling and distributions
│   ├── evaluate.py           # External evaluation script
│   ├── explain.py            # Integrated Gradients & occlusion attribution
│   ├── finalize.py           # Evaluation pipeline and reporting
│   ├── inference.py          # Fast prediction pipeline
│   ├── metrics.py            # Evaluation metrics and curve generation
│   ├── model.py              # PyTorch CNN-LSTM neural network
│   ├── predict.py            # CLI prediction utility
│   └── train.py              # Training loop with one-cycle schedule
├── requirements.txt          # Python dependencies
├── run.py                    # Server launcher
├── run.ps1                   # PowerShell launch script
└── README.md                 # Project documentation
```
