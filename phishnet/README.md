# PhishNet: Phishing URL Detection Using a CNN-LSTM

PhishNet decides whether a URL is phishing or legitimate from its characters alone. It downloads no page and
makes no WHOIS or third-party lookup, so a URL is scored in milliseconds. It is trained on the
**PhiUSIIL Phishing URL Dataset** (UCI ML Repository, id 967), and every verdict comes with an explanation.

The platform has five pages:

| Page | What it does |
|---|---|
| **Scan** | Checks one URL. Shows the verdict, P(phishing) against the tuned threshold, a per-character heatmap (Integrated Gradients or attention), the most influential tokens and rule-based red flags. `/?url=<url>` links straight to a result. |
| **Batch** | Checks up to 5,000 URLs pasted in or loaded from a CSV. Results can be filtered, sorted and exported as CSV. |
| **Model** | Test-set metrics, confusion matrix, ROC and PR curves, score distribution, training curves, the shortcut audit, ablation comparison and the architecture. |
| **Dataset** | PhiUSIIL profile: cleaning funnel, the raw-data artifacts, URL length distribution, top TLDs, sample URLs. |
| **History** | Every scan, stored in a local SQLite file (`data/history.db`). |

## Quick start

```powershell
pip install -r requirements.txt
python -m src.dataset_stats          # dataset profile for the Dataset page
python -m src.train                  # trains the CNN-LSTM -> models/cnn_lstm.pt, reports/cnn_lstm/
.\run.ps1                            # or: python -m uvicorn app.server:app --port 8000
```

Open http://localhost:8000. Interactive API docs are at http://localhost:8000/docs.

The dataset CSV goes in `data/raw/PhiUSIIL_Phishing_URL_Dataset.csv` (unzip the UCI download there).

## Model

```
URL ─► normalise (lower-case, strip scheme and "www.") ─► character ids (max 128)
    ─► Embedding (64-d)
    ─► 3 parallel Conv1D, kernel 3 / 5 / 7, 64 filters each, ReLU, LayerNorm   local n-gram patterns
    ─► BiLSTM, 64 units per direction                                         order and long-range context
    ─► additive attention pooling                                             weight per character
    ─► Dense(64, ReLU) ─► Dropout ─► Dense(1) ─► sigmoid = P(phishing)
```

* The **CNN** picks up local patterns such as `paypa1`, `-login`, `.top`, `xn--` and IP octets.
* The **BiLSTM** models how those patterns are arranged, for example a brand name used as a subdomain
  of an unrelated domain.
* **Attention** pools the sequence. The convolutions use "same" padding, so every time step still lines
  up with one character, and attention and attributions can be drawn on the URL itself.
* Training: AdamW, one-cycle learning rate, `pos_weight` for class imbalance, early stopping on validation
  ROC-AUC. The decision threshold is tuned for F1 on the validation set and stored in the checkpoint.
* Batches are bucketed by URL length and padded only to the longest URL in the batch, which makes CPU
  training practical. Because of the bucketing the model uses **LayerNorm, not BatchNorm**. In PhiUSIIL,
  URL length is strongly tied to the label, so per-batch statistics from length-sorted batches leak label
  information during training that is not available at inference. With BatchNorm, validation loss stayed
  well above training loss (see `reports/train_cnn_lstm_bn_aborted.log`). On a 40k-URL pilot, switching to
  LayerNorm raised test ROC-AUC from 0.885 to 0.913.

### Explanations

| View | Method |
|---|---|
| Character heatmap | Integrated Gradients on the embeddings (32 steps). The baseline is a scaled mean-character embedding calibrated to score near the decision boundary. The all-zero embedding is not used as the baseline because this model scores it as strongly phishing, which made every attribution look "legitimate". |
| Influential tokens | Occlusion: the change in the phishing logit when a token (`paypal`, `verify`, `top`, …) is deleted. |
| Attention | Weights from the pooling layer. They show where the model focused but not in which direction. |
| Red flags | Rule checks: IP host, `@`, punycode, suspicious TLD, brand outside the registered domain, and others. They are shown as supporting evidence and **are not model inputs**. |

## The PhiUSIIL shortcut, and how it is handled

In the raw data **every legitimate URL is a bare homepage of the form `https://www.<domain>`**: 100 % use
https, 100 % have `www.`, and 0 % have a path. Phishing URLs are mixed (49 % https, 41 % www, 28 % with a
path). A model trained on the raw strings can reach very high accuracy just by learning "starts with
`https://www.`" and "has a path". That would make it useless on real links such as
`github.com/pytorch/pytorch`.

PhishNet handles this in three ways:

1. **Normalisation** strips the scheme and a leading `www.`, so the first two giveaways disappear.
2. **Path augmentation** (`--augment-prob`, default 0.35) adds a random neutral path such as `/news/2023/item-417`
   to 35 % of the *training* legitimate URLs each epoch. The vocabulary excludes words like `login`,
   `verify` and `account`, so those stay informative. Validation and test data are never augmented.
3. **Shortcut audit** (in `reports/<run>/metrics.json` and on the Model page): the false-positive rate on
   legitimate test domains with random paths built from a *separate* word list, and phishing recall split by
   whether the URL has a path.

As a result, the accuracy here is lower than the near-100 % that is often reported for PhiUSIIL. It measures
what the model can infer from the domain and path text, not from the dataset's formatting.

Duplicates after normalisation (4,005) and URLs that appear with both labels (43) are removed before the
80 / 10 / 10 stratified split, so no URL string appears in more than one split.

## Results (held-out test set, 23,175 URLs)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---|---|---|---|---|---|
| **CNN-LSTM** (7 epochs, 215K params) | **90.17 %** | **94.60 %** | **81.12 %** | **87.34 %** | **0.9408** | **3.33 %** |
| TF-IDF char n-grams + logistic regression | 89.48 % | 92.82 % | 81.11 % | 86.57 % | 0.9406 | 4.51 % |

Shortcut audit for the CNN-LSTM: phishing recall is 98.4 % for URLs with a path and 74.3 % from the host name
alone. **On legitimate domains given random paths from an unseen vocabulary, the false-positive rate rises to
37.5 %.** Path augmentation reduces the "has a path" shortcut but does not remove it. Legitimate URLs with real
paths in the training data would be the proper fix (see Limitations).

Training was run on CPU (about 19 min per epoch). Epoch 8 was interrupted, so the epoch-7 checkpoint (best
validation AUC, 0.9404) was evaluated with `python -m src.finalize`.

## Experiments

```powershell
python -m src.train --arch cnn                                  # ablation: CNN only
python -m src.train --arch lstm                                 # ablation: BiLSTM only
python -m src.train --augment-prob 0 --run-name cnn_lstm_noaug  # without path augmentation
python -m src.baseline                                          # TF-IDF char n-grams + logistic regression
python -m src.predict "paypal.com.account-verify.top/login"      # explained prediction in the terminal
```

Every run writes `reports/<run>/metrics.json` and PNG plots (confusion matrix, ROC/PR, training curves).
The Model page compares every run in `reports/` automatically.
Use `--max-samples 40000 --epochs 3` for a quick experiment.

## API

```
POST /api/predict   {"url": "...", "explain": true}   -> verdict, probability, explanation
POST /api/batch     {"urls": ["...", "..."]}           -> probabilities and labels (max 5,000)
GET  /api/health    model status
GET  /api/model     training report (?run=cnn)
GET  /api/compare   all runs
GET  /api/dataset   dataset profile
GET  /api/history   recent scans          DELETE /api/history  clears them
```

`PHISHNET_MODEL=path/to/checkpoint.pt` selects another checkpoint. The server reloads the checkpoint
automatically when the file changes, for example after retraining.

## Project layout

```
src/
  data.py           PhiUSIIL loading, normalisation, tokenizer, path augmentation, length-bucketed loader
  model.py          CNNLSTM (cnn_lstm / cnn / lstm)
  train.py          training, threshold tuning, test report, shortcut audit, plots
  explain.py        Integrated Gradients, token occlusion, lexical red flags
  inference.py      Predictor used by the API and the CLI
  metrics.py        metrics, curves, plots
  baseline.py       TF-IDF + logistic regression baseline
  dataset_stats.py  dataset profile for the Dataset page
  predict.py        command-line prediction
app/
  server.py         FastAPI API + static web UI
  static/           index.html, app.css, app.js, charts.js (no build step, no JS dependencies)
data/raw/           PhiUSIIL CSV
models/             checkpoints
reports/            metrics.json + plots per run, dataset.json
```

## Limitations

* Only the URL is used. A phishing page on a compromised legitimate domain with a clean-looking path is hard
  to catch from the URL alone.
* PhiUSIIL's legitimate class contains homepages only. Path augmentation reduces this bias but does not
  replace real legitimate deep links. For deployment, add legitimate URLs with real paths, for example from
  a crawl of the Tranco top sites.
* Phishing patterns change over time, so retrain periodically on fresh feeds such as PhishTank or OpenPhish.

## Credits

Built on the structure of the PhishGuard reference project (`../PhishGuard-main`). Dataset: A. Prasad and
S. Chandra, "PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on
similarity index and incremental learning", *Computers & Security*, 2023.
