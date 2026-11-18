"""Character-level CNN-LSTM for phishing URL detection.

    chars -> Embedding(64)
          -> 3 parallel Conv1D (kernel 3/5/7, 128 filters, ReLU) + LayerNorm   local n-gram patterns
          -> BiLSTM (128 per direction)                                        order / long-range context
          -> additive attention pooling                                       per-character weights
          -> Dense(64, ReLU) -> Dropout -> Dense(1) -> sigmoid = P(phishing)

Convolutions use 'same' padding, so every time step still lines up with one input character.
That is what lets attention weights and Integrated Gradients be drawn on the URL itself.

arch = 'cnn' or 'lstm' builds the single-branch ablations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

ARCHS = ("cnn_lstm", "cnn", "lstm")


class CNNLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, conv_channels=128, kernel_sizes=(3, 5, 7),
                 lstm_hidden=128, dropout=0.3, arch="cnn_lstm"):
        super().__init__()
        if arch not in ARCHS:
            raise ValueError(f"arch must be one of {ARCHS}")
        if any(k % 2 == 0 for k in kernel_sizes):
            raise ValueError("kernel sizes must be odd to keep per-character alignment")
        self.arch = arch
        self.config = dict(vocab_size=vocab_size, embed_dim=embed_dim, conv_channels=conv_channels,
                           kernel_sizes=list(kernel_sizes), lstm_hidden=lstm_hidden, dropout=dropout, arch=arch)
        self.use_cnn = arch in ("cnn_lstm", "cnn")
        self.use_lstm = arch in ("cnn_lstm", "lstm")

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        feat = embed_dim
        if self.use_cnn:
            self.convs = nn.ModuleList(nn.Conv1d(embed_dim, conv_channels, k, padding=k // 2) for k in kernel_sizes)
            feat = conv_channels * len(kernel_sizes)
            # LayerNorm, not BatchNorm: batches are bucketed by URL length, and length correlates with the
            # label, so batch statistics would differ between training and inference.
            self.conv_norm = nn.LayerNorm(feat)
        if self.use_lstm:
            self.lstm = nn.LSTM(feat, lstm_hidden, batch_first=True, bidirectional=True)
            feat = 2 * lstm_hidden

        self.attention = nn.Sequential(nn.Linear(feat, 64), nn.Tanh(), nn.Linear(64, 1))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat, 64), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x, return_attention=False):
        return self.forward_from_embeddings(self.embedding(x), x != 0, return_attention)

    def forward_from_embeddings(self, emb, mask, return_attention=False):
        h = emb
        if self.use_cnn:
            c = h.transpose(1, 2)
            c = torch.cat([F.relu(conv(c)) for conv in self.convs], dim=1)
            h = self.dropout(self.conv_norm(c.transpose(1, 2)))
        if self.use_lstm:
            lengths = mask.sum(1).clamp(min=1).cpu()
            packed = pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
            out, _ = self.lstm(packed)
            h, _ = pad_packed_sequence(out, batch_first=True, total_length=emb.size(1))
            h = self.dropout(h)

        scores = self.attention(h).squeeze(-1).masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), h).squeeze(1)
        logits = self.classifier(pooled).squeeze(-1)
        return (logits, weights) if return_attention else logits


def build_model(config: dict) -> CNNLSTM:
    return CNNLSTM(**config)
