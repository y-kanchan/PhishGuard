"""Hybrid CNN-BiLSTM with additive attention for character-level URL classification.

Pipeline:  chars -> embedding -> parallel 1D convs (k=3,5,7, local n-gram patterns)
           -> BiLSTM (long-range order/context) -> attention pooling -> MLP -> logit

The convs use 'same' padding so every time step still maps to one input character,
which lets attention weights and gradient attributions be shown per character.
`arch` = 'cnn' or 'bilstm' gives the single-branch ablations used for comparison.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

ARCHS = ("hybrid", "cnn", "bilstm")


class PhishGuardNet(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, conv_channels=128, kernel_sizes=(3, 5, 7),
                 lstm_hidden=128, dropout=0.3, arch="hybrid"):
        super().__init__()
        if arch not in ARCHS:
            raise ValueError(f"arch must be one of {ARCHS}")
        if any(k % 2 == 0 for k in kernel_sizes):
            raise ValueError("kernel sizes must be odd to keep per-character alignment")
        self.arch = arch
        self.config = dict(vocab_size=vocab_size, embed_dim=embed_dim, conv_channels=conv_channels,
                           kernel_sizes=list(kernel_sizes), lstm_hidden=lstm_hidden, dropout=dropout, arch=arch)

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        feat = embed_dim

        if arch in ("hybrid", "cnn"):
            self.convs = nn.ModuleList(nn.Conv1d(embed_dim, conv_channels, k, padding=k // 2) for k in kernel_sizes)
            feat = conv_channels * len(kernel_sizes)
            self.conv_norm = nn.BatchNorm1d(feat)

        if arch in ("hybrid", "bilstm"):
            self.lstm = nn.LSTM(feat, lstm_hidden, batch_first=True, bidirectional=True)
            feat = 2 * lstm_hidden

        self.attention = nn.Sequential(nn.Linear(feat, 64), nn.Tanh(), nn.Linear(64, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(feat, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, x, return_attention=False):
        mask = x != 0
        return self.forward_from_embeddings(self.embedding(x), mask, return_attention)

    def forward_from_embeddings(self, emb, mask, return_attention=False, pack=True):
        """Forward pass starting at the embedding layer (used by Integrated Gradients).

        pack=False skips sequence packing; it is only exact for inputs without padding
        and exists so the model can be exported to ONNX for the browser extension.
        """
        h = emb
        if self.arch in ("hybrid", "cnn"):
            c = h.transpose(1, 2)
            c = torch.cat([F.relu(conv(c)) for conv in self.convs], dim=1)
            h = self.dropout(self.conv_norm(c).transpose(1, 2))

        if self.arch in ("hybrid", "bilstm") and not pack:
            h = self.dropout(self.lstm(h)[0])
        elif self.arch in ("hybrid", "bilstm"):
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


def build_model(config: dict) -> PhishGuardNet:
    return PhishGuardNet(**config)
