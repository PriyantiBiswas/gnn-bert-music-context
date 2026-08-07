"""
evaluate.py
-----------
Evaluation Metrics (spec Section 6):
  * Tag/genre classification: per-tag Precision/Recall/F1, Macro-F1, Micro-F1
  * AUC-PR (mean over tags)
  * Emotion regression (DEAM): MAE, R^2
  * Graph coherence score (optional structural-attention analysis)
  * Baselines (Section 8): B1 majority/random, B2 CNN-melspec, B3 BERT-only, B4 PCA+MLP
"""
from __future__ import annotations

import numpy as np


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8):
    """y_true, y_pred: binary (N, K) arrays -> per-tag P/R/F1."""
    tp = (y_true * y_pred).sum(axis=0)
    fp = ((1 - y_true) * y_pred).sum(axis=0)
    fn = (y_true * (1 - y_pred)).sum(axis=0)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1


def macro_micro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)
    macro_f1 = float(np.mean(f1))

    tp = (y_true * y_pred).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()
    micro_p = tp / (tp + fp + 1e-8)
    micro_r = tp / (tp + fn + 1e-8)
    micro_f1 = float(2 * micro_p * micro_r / (micro_p + micro_r + 1e-8))
    return {"macro_f1": macro_f1, "micro_f1": micro_f1, "per_tag_f1": f1.tolist()}


def auc_pr(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Mean AUC-PR over tags via trapezoidal integration of the PR curve."""
    K = y_true.shape[1]
    aucs = []
    for k in range(K):
        yt, ys = y_true[:, k], y_scores[:, k]
        if yt.sum() == 0:
            continue
        order = np.argsort(-ys)
        yt_sorted = yt[order]
        tp_cum = np.cumsum(yt_sorted)
        fp_cum = np.cumsum(1 - yt_sorted)
        precision = tp_cum / (tp_cum + fp_cum + 1e-8)
        recall = tp_cum / (yt.sum() + 1e-8)
        # integrate precision over recall (trapezoidal, prepend (0, 1.0))
        recall_ext = np.concatenate([[0.0], recall])
        precision_ext = np.concatenate([[1.0], precision])
        _trapz = getattr(np, "trapezoid", None) or np.trapz
        aucs.append(float(_trapz(precision_ext, recall_ext)))
    return float(np.mean(aucs)) if aucs else 0.0


def mae_r2(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + 1e-8
    r2 = float(1 - ss_res / ss_tot)
    return {"mae": mae, "r2": r2}


def graph_coherence_score(node_embeds: np.ndarray, edges: list[tuple[int, int]],
                           tau: float = 0.5) -> float:
    """Fraction of edges whose endpoint embeddings have cosine sim > tau."""
    if not edges:
        return 0.0
    Xn = node_embeds / (np.linalg.norm(node_embeds, axis=1, keepdims=True) + 1e-8)
    hits = 0
    for i, j in edges:
        if (Xn[i] @ Xn[j]) > tau:
            hits += 1
    return hits / len(edges)


# ---------------------------------------------------------------------------
# Baselines (Section 8)
# ---------------------------------------------------------------------------
def baseline_majority(y_train: np.ndarray, n_test: int) -> np.ndarray:
    """B1: predict the majority class per tag (i.e. tag prevalence > 0.5)."""
    tag_rate = y_train.mean(axis=0)
    pred_row = (tag_rate > 0.5).astype(np.float32)
    return np.tile(pred_row, (n_test, 1))


def baseline_random(n_test: int, num_tags: int, seed: int = 0) -> np.ndarray:
    """B1 variant: uniform random binary predictions."""
    rng = np.random.default_rng(seed)
    return (rng.uniform(size=(n_test, num_tags)) > 0.5).astype(np.float32)


def baseline_pca_mlp(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                      n_components: int = 32, hidden_dim: int = 64, epochs: int = 100,
                      lr: float = 0.05, seed: int = 0) -> np.ndarray:
    """B4: PCA dimensionality reduction + a tiny 1-hidden-layer MLP, numpy-only."""
    rng = np.random.default_rng(seed)
    mean = X_train.mean(axis=0)
    Xc = X_train - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:n_components]
    Z_train = Xc @ components.T
    Z_test = (X_test - mean) @ components.T

    in_dim, out_dim = Z_train.shape[1], y_train.shape[1]
    W1 = rng.normal(0, 0.1, size=(hidden_dim, in_dim)).astype(np.float32)
    b1 = np.zeros(hidden_dim, dtype=np.float32)
    W2 = rng.normal(0, 0.1, size=(out_dim, hidden_dim)).astype(np.float32)
    b2 = np.zeros(out_dim, dtype=np.float32)

    for _ in range(epochs):
        h = np.maximum(Z_train @ W1.T + b1, 0)
        logits = h @ W2.T + b2
        probs = 1 / (1 + np.exp(-logits))
        grad_logits = (probs - y_train) / y_train.shape[0]
        grad_W2 = grad_logits.T @ h
        grad_b2 = grad_logits.sum(axis=0)
        grad_h = grad_logits @ W2
        grad_h[h <= 0] = 0
        grad_W1 = grad_h.T @ Z_train
        grad_b1 = grad_h.sum(axis=0)
        W1 -= lr * grad_W1; b1 -= lr * grad_b1
        W2 -= lr * grad_W2; b2 -= lr * grad_b2

    h_test = np.maximum(Z_test @ W1.T + b1, 0)
    probs_test = 1 / (1 + np.exp(-(h_test @ W2.T + b2)))
    return probs_test


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, K = 100, 10
    y_true = (rng.uniform(size=(N, K)) > 0.6).astype(np.float32)
    y_scores = np.clip(y_true + rng.normal(0, 0.3, size=(N, K)), 0, 1)
    y_pred = (y_scores > 0.5).astype(np.float32)

    print("F1:", macro_micro_f1(y_true, y_pred))
    print("AUC-PR:", auc_pr(y_true, y_scores))

    v_true = rng.uniform(2, 8, size=50)
    v_pred = v_true + rng.normal(0, 0.5, size=50)
    print("Emotion:", mae_r2(v_true, v_pred))
