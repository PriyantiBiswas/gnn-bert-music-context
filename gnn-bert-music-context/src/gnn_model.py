"""
gnn_model.py
------------
Task 2 (Medium): GraphSAGE / GAT encoder over segment/chord graphs, using
audio-only node features; predicts genre or top tags.

    h_i^(l+1) = sigma( W^(l) . CONCAT(h_i^(l), MEAN_{j in N(i)} h_j^(l)) )    (GraphSAGE)
    g = MEAN_{i in V} h_i^(L)                                                 (readout)
    y_hat = sigmoid(W g + b)

Implemented with PyTorch Geometric when available. A pure-numpy fallback
(`NumpyGraphSAGE`) implements the exact same recurrence for environments
without torch/torch_geometric, so `train.py --synthetic` always runs.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool
    _HAS_PYG = True
except ImportError:  # pragma: no cover
    _HAS_PYG = False


if _HAS_PYG:

    class GNNEncoder(nn.Module):
        """Stacked GraphSAGE (or GAT) layers + mean-pool readout (Algorithm 2)."""

        def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 128,
                     num_layers: int = 3, dropout: float = 0.2, gnn_type: str = "graphsage"):
            super().__init__()
            Conv = SAGEConv if gnn_type == "graphsage" else GATConv
            dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
            self.convs = nn.ModuleList(
                [Conv(dims[i], dims[i + 1]) for i in range(num_layers)]
            )
            self.dropout = dropout

        def forward(self, x, edge_index, batch):
            h = x
            for i, conv in enumerate(self.convs):
                h = conv(h, edge_index)
                if i < len(self.convs) - 1:
                    h = F.relu(h)
                    h = F.dropout(h, p=self.dropout, training=self.training)
            g = global_mean_pool(h, batch)  # graph-level readout
            return g, h

    class GNNClassifier(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                     num_layers: int, num_classes: int, dropout: float = 0.2,
                     gnn_type: str = "graphsage"):
            super().__init__()
            self.encoder = GNNEncoder(in_dim, hidden_dim, out_dim, num_layers, dropout, gnn_type)
            self.head = nn.Linear(out_dim, num_classes)

        def forward(self, x, edge_index, batch):
            g, h = self.encoder(x, edge_index, batch)
            logits = self.head(g)
            return logits, g


# ---------------------------------------------------------------------------
# Dependency-free fallback implementing the identical GraphSAGE recurrence
# with plain numpy (mean aggregation + linear + relu), for smoke tests.
# ---------------------------------------------------------------------------
class NumpyGraphSAGE:
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 128,
                 num_layers: int = 3, num_classes: int = 8, seed: int = 0):
        rng = np.random.default_rng(seed)
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = []
        for i in range(num_layers):
            # CONCAT(h_i, mean_neighbors) has dim 2*dims[i] -> dims[i+1]
            W = rng.normal(0, 0.1, size=(dims[i + 1], 2 * dims[i])).astype(np.float32)
            b = np.zeros(dims[i + 1], dtype=np.float32)
            self.layers.append([W, b])
        self.head_W = rng.normal(0, 0.1, size=(num_classes, dims[-1])).astype(np.float32)
        self.head_b = np.zeros(num_classes, dtype=np.float32)

    def _neighbors(self, edge_index: list[tuple[int, int]], n_nodes: int) -> list[list[int]]:
        adj = [[] for _ in range(n_nodes)]
        for a, b in edge_index:
            adj[a].append(b)
        return adj

    def forward(self, graph) -> tuple[np.ndarray, np.ndarray]:
        """graph: SimpleGraph from graph_builder.py"""
        h = graph.x.astype(np.float32)
        adj = self._neighbors(graph.edge_index, h.shape[0])
        for l, (W, b) in enumerate(self.layers):
            neigh_mean = np.stack(
                [h[adj[i]].mean(axis=0) if adj[i] else np.zeros_like(h[i]) for i in range(h.shape[0])],
                axis=0,
            )
            concat = np.concatenate([h, neigh_mean], axis=1)
            h_new = concat @ W.T + b
            if l < len(self.layers) - 1:
                h_new = np.maximum(h_new, 0)  # ReLU
            h = h_new
        g = h.mean(axis=0)  # mean-pool readout
        logits = self.head_W @ g + self.head_b
        probs = 1 / (1 + np.exp(-logits))
        return probs, g

    def train_step(self, graphs: list, targets: np.ndarray, lr: float = 0.05) -> float:
        """One (simple, non-batched) SGD step over a list of graphs using
        numeric gradients on the final layer only (for smoke-testing loss
        decrease; a full backprop-through-message-passing implementation
        belongs in the torch/PyG path above)."""
        total_loss = 0.0
        grad_head_W = np.zeros_like(self.head_W)
        grad_head_b = np.zeros_like(self.head_b)
        for graph, y in zip(graphs, targets):
            probs, g = self.forward(graph)
            eps = 1e-7
            loss = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
            total_loss += loss
            grad_logits = probs - y
            grad_head_W += np.outer(grad_logits, g)
            grad_head_b += grad_logits
        n = len(graphs)
        self.head_W -= lr * grad_head_W / n
        self.head_b -= lr * grad_head_b / n
        return total_loss / n


def build_gnn_model(config: dict, num_classes: int):
    if _HAS_PYG:
        g = config["gnn"]
        return GNNClassifier(
            in_dim=g["in_dim"], hidden_dim=g["hidden_dim"], out_dim=g["out_dim"],
            num_layers=g["num_layers"], num_classes=num_classes, dropout=g["dropout"],
            gnn_type=g["type"],
        )
    return NumpyGraphSAGE(in_dim=13, hidden_dim=32, out_dim=32, num_layers=2,
                           num_classes=num_classes)


if __name__ == "__main__":
    from datasets import make_synthetic_dataset

    ds = make_synthetic_dataset(n_samples=16, num_genres=4, graph_type="segment")
    model = NumpyGraphSAGE(in_dim=ds[0].graph.x.shape[1], hidden_dim=32, out_dim=32,
                            num_layers=2, num_classes=4)
    targets = np.eye(4, dtype=np.float32)[[s.genre for s in ds]]
    for epoch in range(30):
        loss = model.train_step([s.graph for s in ds], targets, lr=0.2)
    print(f"final loss after 30 steps: {loss:.4f}")
