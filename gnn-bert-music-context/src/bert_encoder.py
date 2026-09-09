"""Task 1: BERT-based multi-label tag classifier."""
import numpy as np
import torch
import torch.nn as nn


class MiniTextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, n_layers=4, n_heads=4, max_length=32, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_length, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def forward(self, input_ids):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        pad_mask = input_ids == 0
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        h = self.ln(h)
        cls = h[:, 0, :]
        return cls, h


class HFBertEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", freeze=False):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(model_name)
        self.embed_dim = self.bert.config.hidden_size
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        return h[:, 0, :], h


class BertTagClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, n_tags: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.embed_dim, n_tags)

    def forward(self, input_ids):
        cls, _ = self.encoder(input_ids)
        return self.head(cls)


def bce_loss(logits, targets, pos_weight=None):
    return nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def compute_pos_weight(y_true_stacked, device, cap=10.0):
    pos = y_true_stacked.sum(axis=0)
    neg = y_true_stacked.shape[0] - pos
    w = np.clip(neg / np.clip(pos, 1, None), 1.0, cap)
    return torch.tensor(w, dtype=torch.float32, device=device)