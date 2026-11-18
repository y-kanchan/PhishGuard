Test set: PhiUSIIL Phishing URL Dataset, 23,175 held-out URLs (see `phishnet/`).

| Model | Accuracy | Precision | Recall | F1 | Roc Auc | False Positive Rate |
|---|---|---|---|---|---|---|
| cnn_lstm | 0.9017 | 0.9460 | 0.8112 | 0.8734 | 0.9408 | 0.0333 |
| tfidf_logreg | 0.8948 | 0.9282 | 0.8111 | 0.8657 | 0.9406 | 0.0451 |
