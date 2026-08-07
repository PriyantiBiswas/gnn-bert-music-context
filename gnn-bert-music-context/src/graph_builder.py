"""
graph_builder.py
-----------------
Music structure graph construction (spec Section 3.3):

  * Chord-transition graph: nodes = unique chords, edges = observed
    transitions weighted by count.
  * Segment-similarity graph: nodes = time segments, edges = temporal
    adjacency + cosine similarity of MFCC/chroma > tau.

Graphs are returned as PyTorch Geometric `Data` objects (falls back to a
lightweight plain-Python `SimpleGraph` if torch_geometric isn't installed,
so the rest of the repo can still be exercised).
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    from torch_geometric.data import Data
    _HAS_PYG = True
except ImportError:  # pragma: no cover
    _HAS_PYG = False


CHORD_TEMPLATES = {
    # 12 major + 12 minor triads over chroma pitch classes 0-11 (C..B)
    **{f"{i}maj": [i, (i + 4) % 12, (i + 7) % 12] for i in range(12)},
    **{f"{i}min": [i, (i + 3) % 12, (i + 7) % 12] for i in range(12)},
}
CHORD_NAMES = list(CHORD_TEMPLATES.keys())


def estimate_chord_sequence(chroma: np.ndarray, hop_frames: int = 20) -> list[str]:
    """Naive template-matching chord estimation, one label per `hop_frames` block."""
    templates = np.zeros((len(CHORD_NAMES), 12))
    for i, name in enumerate(CHORD_NAMES):
        for p in CHORD_TEMPLATES[name]:
            templates[i, p] = 1.0
    templates /= np.linalg.norm(templates, axis=1, keepdims=True)

    labels = []
    for s in range(0, chroma.shape[1], hop_frames):
        block = chroma[:, s : s + hop_frames]
        if block.shape[1] == 0:
            continue
        v = block.mean(axis=1)
        v = v / (np.linalg.norm(v) + 1e-8)
        scores = templates @ v
        labels.append(CHORD_NAMES[int(np.argmax(scores))])
    return labels


def build_chord_transition_graph(chord_seq: list[str]):
    """Nodes = unique chords present; edges weighted by transition counts."""
    unique_chords = sorted(set(chord_seq))
    idx = {c: i for i, c in enumerate(unique_chords)}

    edge_counts: dict[tuple[int, int], int] = {}
    for a, b in zip(chord_seq[:-1], chord_seq[1:]):
        if a == b:
            continue
        key = (idx[a], idx[b])
        edge_counts[key] = edge_counts.get(key, 0) + 1

    node_features = np.stack(
        [_chord_one_hot(c) for c in unique_chords], axis=0
    ).astype(np.float32)

    return _to_graph(node_features, edge_counts, node_names=unique_chords)


def build_segment_similarity_graph(
    segment_embeddings: list[np.ndarray], tau: float = 0.85
):
    """Nodes = time segments. Edges = temporal adjacency + cosine sim > tau."""
    n = len(segment_embeddings)
    X = np.stack(segment_embeddings, axis=0).astype(np.float32)  # (n, d)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    sim = Xn @ Xn.T

    edge_counts: dict[tuple[int, int], float] = {}
    for i in range(n - 1):
        edge_counts[(i, i + 1)] = 1.0  # temporal adjacency

    for i in range(n):
        for j in range(i + 2, n):
            if sim[i, j] > tau:
                edge_counts[(i, j)] = float(sim[i, j])

    return _to_graph(X, edge_counts, node_names=[f"seg_{i}" for i in range(n)])


def _chord_one_hot(chord_name: str) -> np.ndarray:
    vec = np.zeros(12, dtype=np.float32)
    for p in CHORD_TEMPLATES[chord_name]:
        vec[p] = 1.0
    is_minor = 1.0 if chord_name.endswith("min") else 0.0
    return np.concatenate([vec, [is_minor]])


def _to_graph(node_features: np.ndarray, edge_counts: dict, node_names: list[str]):
    edges = list(edge_counts.keys())
    weights = list(edge_counts.values())
    # make undirected
    src = [e[0] for e in edges] + [e[1] for e in edges]
    dst = [e[1] for e in edges] + [e[0] for e in edges]
    w = weights + weights

    if _HAS_PYG:
        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.empty(
            (2, 0), dtype=torch.long
        )
        edge_weight = torch.tensor(w, dtype=torch.float32) if w else torch.empty(0)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
        data.node_names = node_names
        return data

    return SimpleGraph(node_features, list(zip(src, dst)), w, node_names)


class SimpleGraph:
    """Minimal graph container used when torch_geometric is unavailable."""

    def __init__(self, x: np.ndarray, edge_index: list[tuple[int, int]],
                 edge_weight: list[float], node_names: list[str]):
        self.x = x
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.node_names = node_names

    @property
    def num_nodes(self) -> int:
        return self.x.shape[0]

    def __repr__(self) -> str:
        return f"SimpleGraph(nodes={self.num_nodes}, edges={len(self.edge_index)})"


if __name__ == "__main__":
    from audio_features import synthetic_track, segment_windows, segment_embedding

    track = synthetic_track(seed=1)
    chords = estimate_chord_sequence(track["chroma"])
    g_chord = build_chord_transition_graph(chords)
    print("chord graph:", g_chord)

    segs = segment_windows(track["chroma"], sr=track["sr"], window_seconds=8.0)
    embeds = [segment_embedding(s) for s in segs]
    g_seg = build_segment_similarity_graph(embeds, tau=0.85)
    print("segment graph:", g_seg)
