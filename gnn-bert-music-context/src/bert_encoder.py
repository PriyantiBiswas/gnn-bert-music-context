"""
bert_encoder.py
----------------
Task 1 (Easy): BERT-based multi-label classifier for music tags / captions
/ lyrics — no graph structure.

    t = BERT_CLS(X_text)
    y_hat_k = sigmoid(w_k^T t + b_k)
    L_BERT = -(1/K) * sum_k [ y_k log y_hat_k + (1-y_k) log(1-y_hat_k) ]

Uses HuggingFace `bert-base-uncased` / `distilbert-base-uncased`. Falls back
to a bag-of-words + linear "poor man's BERT" (`SimpleTextEncoder`) if
`transformers`/`torch` aren't installed, so the training loop stays runnable
in constrained environments (Algorithm 1 in the spec is preserved either way).
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False

import numpy as np


if _HAS_TORCH:

    class BertTagClassifier(nn.Module):
        """CLS token -> linear head -> per-tag sigmoid (Algorithm 1)."""

        def __init__(self, model_name: str = "distilbert-base-uncased",
                     num_tags: int = 50, freeze_backbone: bool = False,
                     hidden_dim: int = 768):
            super().__init__()
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert = AutoModel.from_pretrained(model_name)
            if freeze_backbone:
                for p in self.bert.parameters():
                    p.requires_grad = False
            self.head = nn.Linear(hidden_dim, num_tags)

        def encode_text(self, texts: list[str], max_length: int = 256, device="cpu"):
            enc = self.tokenizer(
                texts, padding=True, truncation=True, max_length=max_length,
                return_tensors="pt",
            ).to(device)
            out = self.bert(**enc)
            cls = out.last_hidden_state[:, 0, :]  # CLS token, t
            return cls

        def forward(self, texts: list[str], max_length: int = 256, device="cpu"):
            t = self.encode_text(texts, max_length=max_length, device=device)
            logits = self.head(t)  # pre-sigmoid; use BCEWithLogitsLoss
            return logits, t

    def bert_bce_loss(logits, targets):
        return nn.functional.binary_cross_entropy_with_logits(logits, targets)


# ---------------------------------------------------------------------------
# Dependency-free fallback: bag-of-words CLS-analogue + linear head, trained
# with plain-numpy gradient descent. Mirrors the same math (sigmoid + BCE)
# so `train.py --synthetic` runs identically end-to-end without torch.
# ---------------------------------------------------------------------------
class SimpleTextEncoder:
    """Hashing bag-of-words encoder standing in for BERT's CLS embedding."""

    def __init__(self, hidden_dim: int = 128, num_tags: int = 50, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.hidden_dim = hidden_dim
        self.W = rng.normal(0, 0.1, size=(num_tags, hidden_dim)).astype(np.float32)
        self.b = np.zeros(num_tags, dtype=np.float32)

    def _hash_embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.hidden_dim, dtype=np.float32)
        for tok in text.lower().split():
            vec[hash(tok) % self.hidden_dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def encode_text(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._hash_embed(t) for t in texts], axis=0)

    def forward(self, texts: list[str]):
        t = self.encode_text(texts)                  # "CLS" analogue
        logits = t @ self.W.T + self.b
        probs = 1 / (1 + np.exp(-logits))
        return probs, t

    def train_step(self, texts: list[str], targets: np.ndarray, lr: float = 0.1) -> float:
        probs, t = self.forward(texts)
        eps = 1e-7
        loss = -np.mean(
            targets * np.log(probs + eps) + (1 - targets) * np.log(1 - probs + eps)
        )
        grad_logits = (probs - targets) / targets.shape[0]     # dL/dlogits
        grad_W = grad_logits.T @ t
        grad_b = grad_logits.sum(axis=0)
        self.W -= lr * grad_W
        self.b -= lr * grad_b
        return float(loss)


def build_bert_tag_model(config: dict):
    """Factory: real BERT model if torch/transformers available, else fallback."""
    if _HAS_TORCH:
        return BertTagClassifier(
            model_name=config["bert"]["model_name"],
            num_tags=config["bert"]["num_tags"],
            freeze_backbone=config["bert"]["freeze_backbone"],
            hidden_dim=config["bert"]["hidden_dim"],
        )
    return SimpleTextEncoder(hidden_dim=128, num_tags=config["bert"]["num_tags"])


if __name__ == "__main__":
    model = SimpleTextEncoder(num_tags=5)
    texts = ["jazz piano slow", "metal fast guitar", "ambient synth slow"]
    targets = np.array(
        [[1, 0, 1, 0, 0], [0, 1, 0, 1, 1], [1, 1, 0, 0, 0]], dtype=np.float32
    )
    for epoch in range(50):
        loss = model.train_step(texts, targets, lr=0.5)
    print(f"final BCE loss after 50 steps: {loss:.4f}")
