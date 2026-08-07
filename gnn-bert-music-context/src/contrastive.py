"""
contrastive.py
---------------
Task 4 (Advanced): Cross-modal MusicCaps alignment via InfoNCE contrastive
learning between graph embeddings and caption embeddings.

    g_i = normalize(GNN(G_i)),  t_i = normalize(BERT_CLS(caption_i))
    S_ij = g_i^T t_j / tau
    L_NCE = -(1/N) sum_i log( exp(S_ii) / sum_j exp(S_ij) )

Retrieval metrics: Caption->Audio R@1/5/10, Audio->Caption R@K
(Algorithm 4 + Section 4.4 deliverables).
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

    class DualEncoder(nn.Module):
        """Wraps a GNN encoder + BERT encoder with L2-normalized projection heads."""

        def __init__(self, gnn_encoder, bert_encoder, g_dim: int, text_dim: int,
                     embed_dim: int = 256):
            super().__init__()
            self.gnn_encoder = gnn_encoder
            self.bert_encoder = bert_encoder
            self.g_proj = nn.Linear(g_dim, embed_dim)
            self.t_proj = nn.Linear(text_dim, embed_dim)

        def encode_graph(self, graph_batch):
            g, _ = self.gnn_encoder(graph_batch.x, graph_batch.edge_index, graph_batch.batch)
            return F.normalize(self.g_proj(g), dim=-1)

        def encode_text(self, texts: list[str], device="cpu"):
            t = self.bert_encoder.encode_text(texts, device=device)
            return F.normalize(self.t_proj(t), dim=-1)

        def forward(self, graph_batch, texts, device="cpu"):
            return self.encode_graph(graph_batch), self.encode_text(texts, device=device)

    def info_nce_loss(g_embed, t_embed, temperature: float = 0.07):
        """Symmetric InfoNCE over a batch (graph->text and text->graph)."""
        logits = g_embed @ t_embed.T / temperature          # (N, N), S_ij
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_g2t = F.cross_entropy(logits, labels)
        loss_t2g = F.cross_entropy(logits.T, labels)
        return (loss_g2t + loss_t2g) / 2


# ---------------------------------------------------------------------------
# Numpy fallback: identical InfoNCE math over precomputed embedding pairs.
# ---------------------------------------------------------------------------
class NumpyDualEncoder:
    def __init__(self, g_dim: int, t_dim: int, embed_dim: int = 256, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.Wg = rng.normal(0, 0.1, size=(embed_dim, g_dim)).astype(np.float32)
        self.Wt = rng.normal(0, 0.1, size=(embed_dim, t_dim)).astype(np.float32)

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

    def encode(self, G: np.ndarray, T: np.ndarray):
        g_embed = self._normalize(G @ self.Wg.T)
        t_embed = self._normalize(T @ self.Wt.T)
        return g_embed, t_embed

    def loss_and_grad(self, G: np.ndarray, T: np.ndarray, tau: float = 0.07):
        g_embed, t_embed = self.encode(G, T)               # (N, d) each
        N = G.shape[0]
        logits = g_embed @ t_embed.T / tau                  # S_ij
        # graph -> text
        p_g2t = _softmax_rows(logits)
        loss_g2t = -np.mean(np.log(p_g2t[np.arange(N), np.arange(N)] + 1e-8))
        # text -> graph
        p_t2g = _softmax_rows(logits.T)
        loss_t2g = -np.mean(np.log(p_t2g[np.arange(N), np.arange(N)] + 1e-8))
        loss = (loss_g2t + loss_t2g) / 2

        # Gradients wrt logits (standard softmax-CE gradient), symmetrized
        grad_logits_g2t = (p_g2t - np.eye(N)) / N
        grad_logits_t2g = (p_t2g - np.eye(N)) / N
        grad_logits = (grad_logits_g2t + grad_logits_t2g.T) / 2 / tau

        grad_g_embed = grad_logits @ t_embed
        grad_t_embed = grad_logits.T @ g_embed
        grad_Wg = grad_g_embed.T @ G
        grad_Wt = grad_t_embed.T @ T
        return loss, grad_Wg, grad_Wt

    def train_step(self, G: np.ndarray, T: np.ndarray, lr: float = 0.1, tau: float = 0.07) -> float:
        loss, grad_Wg, grad_Wt = self.loss_and_grad(G, T, tau)
        self.Wg -= lr * grad_Wg
        self.Wt -= lr * grad_Wt
        return float(loss)


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Retrieval evaluation: R@1/5/10 for caption->audio and audio->caption
# ---------------------------------------------------------------------------
def recall_at_k(g_embed: np.ndarray, t_embed: np.ndarray, k_values=(1, 5, 10)) -> dict:
    """Assumes row i of g_embed and row i of t_embed are the ground-truth pair."""
    sim = g_embed @ t_embed.T  # (N, N)
    n = sim.shape[0]
    results = {}

    ranks_g2t = _ranks_of_correct(sim, np.arange(n))
    ranks_t2g = _ranks_of_correct(sim.T, np.arange(n))

    for k in k_values:
        results[f"audio2text_R@{k}"] = float(np.mean(ranks_g2t < k))
        results[f"text2audio_R@{k}"] = float(np.mean(ranks_t2g < k))
    return results


def _ranks_of_correct(sim: np.ndarray, correct_idx: np.ndarray) -> np.ndarray:
    order = np.argsort(-sim, axis=1)  # descending similarity
    ranks = np.array([
        np.where(order[i] == correct_idx[i])[0][0] for i in range(sim.shape[0])
    ])
    return ranks


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, g_dim, t_dim = 40, 32, 16
    # correlated G/T so retrieval is learnable
    latent = rng.normal(size=(N, 8))
    G = latent @ rng.normal(size=(8, g_dim)) + rng.normal(scale=0.3, size=(N, g_dim))
    T = latent @ rng.normal(size=(8, t_dim)) + rng.normal(scale=0.3, size=(N, t_dim))
    G, T = G.astype(np.float32), T.astype(np.float32)

    model = NumpyDualEncoder(g_dim, t_dim, embed_dim=32)
    for epoch in range(100):
        loss = model.train_step(G, T, lr=0.3, tau=0.2)
    g_embed, t_embed = model.encode(G, T)
    metrics = recall_at_k(g_embed, t_embed, k_values=(1, 5, 10))
    print(f"final InfoNCE loss: {loss:.4f}")
    print("retrieval metrics:", {k: round(v, 3) for k, v in metrics.items()})
