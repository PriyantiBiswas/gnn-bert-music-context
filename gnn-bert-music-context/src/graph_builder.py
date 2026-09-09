"""Music structure graph construction: chord-transition & segment graphs."""
import numpy as np


def build_segment_graph(segment_feats: np.ndarray, tau: float = 0.6):
    n = segment_feats.shape[0]
    edges, weights = [], []
    for i in range(n - 1):
        edges += [(i, i + 1), (i + 1, i)]
        weights += [1.0, 1.0]
    if n > 1:
        norm = segment_feats / (np.linalg.norm(segment_feats, axis=1, keepdims=True) + 1e-8)
        sims = norm @ norm.T
        for i in range(n):
            for j in range(n):
                if i != j and sims[i, j] > tau:
                    edges.append((i, j))
                    weights.append(float(sims[i, j]))
    edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    edge_weight = np.array(weights, dtype=np.float32)
    return segment_feats.astype(np.float32), edge_index, edge_weight


CHORD_VOCAB = ["C", "Cm", "D", "Dm", "E", "Em", "F", "Fm", "G", "Gm", "A", "Am", "B", "Bm"]


def build_chord_transition_graph(chord_sequence: list):
    idx = {c: i for i, c in enumerate(CHORD_VOCAB)}
    n = len(CHORD_VOCAB)
    counts = np.zeros((n, n), dtype=np.float32)
    for a, b in zip(chord_sequence[:-1], chord_sequence[1:]):
        if a in idx and b in idx:
            counts[idx[a], idx[b]] += 1
    edges, weights = [], []
    for i in range(n):
        for j in range(n):
            if counts[i, j] > 0:
                edges.append((i, j))
                weights.append(counts[i, j])
    edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    edge_weight = np.array(weights, dtype=np.float32)
    node_feats = np.eye(n, dtype=np.float32)
    return node_feats, edge_index, edge_weight


def graph_coherence_score(node_embeddings: np.ndarray, edge_index: np.ndarray, tau: float = 0.6):
    if edge_index.shape[1] == 0:
        return 0.0
    norm = node_embeddings / (np.linalg.norm(node_embeddings, axis=1, keepdims=True) + 1e-8)
    src, dst = edge_index
    cos = (norm[src] * norm[dst]).sum(axis=1)
    return float((cos > tau).mean())