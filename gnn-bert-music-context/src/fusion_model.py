"""
fusion_model.py
----------------
Task 3 (Hard): Fuse structural (GNN) and semantic (BERT) representations to
predict multi-label context: genre + mood tags + (optional) valence/arousal.

    A = softmax(Q K^T / sqrt(d)),  Q = g W_Q,  K = H_text W_K
    z = CONCAT(g, A H_text)
    y_hat = sigmoid(W z)
    L = L_tags + alpha * ||v - v_hat||^2 + beta * ||a - a_hat||^2

Also exposes the required ablation variants: BERT-only, GNN-only, early
concat, and cross-attention (Algorithm 3 + Section 4.3 deliverables).
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


if _HAS_TORCH:

    class CrossAttentionFusion(nn.Module):
        """Graph vector attends over BERT token sequence (Algorithm 3)."""

        def __init__(self, g_dim: int, text_dim: int, z_dim: int, num_heads: int = 4):
            super().__init__()
            self.q_proj = nn.Linear(g_dim, z_dim)
            self.k_proj = nn.Linear(text_dim, z_dim)
            self.v_proj = nn.Linear(text_dim, z_dim)
            self.num_heads = num_heads
            self.z_dim = z_dim
            self.out_proj = nn.Linear(g_dim + z_dim, z_dim)

        def forward(self, g, H_text):
            # g: (B, g_dim); H_text: (B, L, text_dim)
            Q = self.q_proj(g).unsqueeze(1)                      # (B, 1, z)
            K = self.k_proj(H_text)                               # (B, L, z)
            V = self.v_proj(H_text)                               # (B, L, z)
            scores = (Q @ K.transpose(-2, -1)) / (self.z_dim ** 0.5)  # (B,1,L)
            A = F.softmax(scores, dim=-1)
            attended = (A @ V).squeeze(1)                          # (B, z)
            z = self.out_proj(torch.cat([g, attended], dim=-1))
            return z, A.squeeze(1)

    class GNNBertFusionModel(nn.Module):
        """End-to-end fusion model with a multi-task head (tags + emotion)."""

        def __init__(self, gnn_encoder, bert_encoder, g_dim: int, text_dim: int,
                     z_dim: int, num_tags: int, mode: str = "cross_attention"):
            super().__init__()
            self.gnn_encoder = gnn_encoder
            self.bert_encoder = bert_encoder
            self.mode = mode  # "bert_only" | "gnn_only" | "early_concat" | "cross_attention"

            if mode == "cross_attention":
                self.fusion = CrossAttentionFusion(g_dim, text_dim, z_dim)
                fused_dim = z_dim
            elif mode == "early_concat":
                self.fusion = None
                fused_dim = g_dim + text_dim
            elif mode == "gnn_only":
                self.fusion = None
                fused_dim = g_dim
            elif mode == "bert_only":
                self.fusion = None
                fused_dim = text_dim
            else:
                raise ValueError(f"unknown fusion mode: {mode}")

            self.tag_head = nn.Linear(fused_dim, num_tags)
            self.emotion_head = nn.Linear(fused_dim, 2)  # valence, arousal

        def forward(self, graph_batch, texts, device="cpu"):
            g, _ = self.gnn_encoder(graph_batch.x, graph_batch.edge_index, graph_batch.batch)
            H_text = self.bert_encoder.bert(
                **self.bert_encoder.tokenizer(texts, padding=True, truncation=True,
                                               return_tensors="pt").to(device)
            ).last_hidden_state
            t_cls = H_text[:, 0, :]

            if self.mode == "cross_attention":
                z, attn = self.fusion(g, H_text)
            elif self.mode == "early_concat":
                z, attn = torch.cat([g, t_cls], dim=-1), None
            elif self.mode == "gnn_only":
                z, attn = g, None
            else:  # bert_only
                z, attn = t_cls, None

            tag_logits = self.tag_head(z)
            emotion_pred = self.emotion_head(z)  # (B, 2) -> valence, arousal
            return tag_logits, emotion_pred, z, attn

    def fusion_loss(tag_logits, tag_targets, emotion_pred, emotion_targets,
                     alpha: float = 1.0, beta: float = 1.0):
        l_tags = F.binary_cross_entropy_with_logits(tag_logits, tag_targets)
        v_hat, a_hat = emotion_pred[:, 0], emotion_pred[:, 1]
        v, a = emotion_targets[:, 0], emotion_targets[:, 1]
        l_emotion = alpha * F.mse_loss(v_hat, v) + beta * F.mse_loss(a_hat, a)
        return l_tags + l_emotion, l_tags.item(), l_emotion.item()


# ---------------------------------------------------------------------------
# Numpy fallback: cross-attention fusion of a GNN graph vector `g` and a
# BERT-analogue CLS vector `t`, with the same tag/emotion multi-task loss.
# ---------------------------------------------------------------------------
class NumpyFusionModel:
    def __init__(self, g_dim: int, t_dim: int, z_dim: int, num_tags: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.Wq = rng.normal(0, 0.1, size=(z_dim, g_dim)).astype(np.float32)
        self.Wk = rng.normal(0, 0.1, size=(z_dim, t_dim)).astype(np.float32)
        self.Wv = rng.normal(0, 0.1, size=(z_dim, t_dim)).astype(np.float32)
        self.Wo = rng.normal(0, 0.1, size=(z_dim, g_dim + z_dim)).astype(np.float32)
        self.tag_W = rng.normal(0, 0.1, size=(num_tags, z_dim)).astype(np.float32)
        self.tag_b = np.zeros(num_tags, dtype=np.float32)
        self.emo_W = rng.normal(0, 0.1, size=(2, z_dim)).astype(np.float32)
        self.emo_b = np.zeros(2, dtype=np.float32)
        self.z_dim = z_dim

    def forward(self, g: np.ndarray, H_text: np.ndarray):
        """g: (g_dim,) graph vector; H_text: (L, t_dim) token embeddings (single sample)."""
        Q = self.Wq @ g                                   # (z,)
        K = H_text @ self.Wk.T                             # (L, z)
        V = H_text @ self.Wv.T                              # (L, z)
        scores = (K @ Q) / np.sqrt(self.z_dim)              # (L,)
        A = np.exp(scores - scores.max())
        A /= A.sum()
        attended = A @ V                                     # (z,)
        z = self.Wo @ np.concatenate([g, attended])
        tag_probs = 1 / (1 + np.exp(-(self.tag_W @ z + self.tag_b)))
        emotion = self.emo_W @ z + self.emo_b
        return tag_probs, emotion, z, A

    def train_step(self, samples: list[tuple[np.ndarray, np.ndarray]], tags: np.ndarray,
                    emotions: np.ndarray, lr: float = 0.05, alpha: float = 1.0, beta: float = 1.0):
        total_loss = 0.0
        grad_tag_W = np.zeros_like(self.tag_W)
        grad_tag_b = np.zeros_like(self.tag_b)
        grad_emo_W = np.zeros_like(self.emo_W)
        grad_emo_b = np.zeros_like(self.emo_b)
        for (g, H_text), y_tag, y_emo in zip(samples, tags, emotions):
            probs, emo, z, _ = self.forward(g, H_text)
            eps = 1e-7
            l_tags = -np.mean(y_tag * np.log(probs + eps) + (1 - y_tag) * np.log(1 - probs + eps))
            l_emo = alpha * (emo[0] - y_emo[0]) ** 2 + beta * (emo[1] - y_emo[1]) ** 2
            total_loss += l_tags + l_emo

            grad_tag_logits = probs - y_tag
            grad_tag_W += np.outer(grad_tag_logits, z)
            grad_tag_b += grad_tag_logits
            grad_emo_out = 2 * np.array([alpha * (emo[0] - y_emo[0]), beta * (emo[1] - y_emo[1])])
            grad_emo_W += np.outer(grad_emo_out, z)
            grad_emo_b += grad_emo_out

        n = len(samples)
        self.tag_W -= lr * grad_tag_W / n
        self.tag_b -= lr * grad_tag_b / n
        self.emo_W -= lr * grad_emo_W / n
        self.emo_b -= lr * grad_emo_b / n
        return total_loss / n


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    g_dim, t_dim, z_dim, num_tags, L = 32, 16, 24, 5, 6

    model = NumpyFusionModel(g_dim, t_dim, z_dim, num_tags)
    samples = [(rng.normal(size=g_dim).astype(np.float32),
                rng.normal(size=(L, t_dim)).astype(np.float32)) for _ in range(8)]
    tags = (rng.uniform(size=(8, num_tags)) > 0.5).astype(np.float32)
    emotions = rng.uniform(2, 8, size=(8, 2)).astype(np.float32)

    for epoch in range(40):
        loss = model.train_step(samples, tags, emotions, lr=0.1)
    print(f"final fusion loss after 40 steps: {loss:.4f}")
