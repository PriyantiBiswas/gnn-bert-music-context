"""Task 2: GraphSAGE encoder on music structure graphs (plain PyTorch, no PyG needed)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_mean(src, index, dim_size):
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    count = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    count.index_add_(0, index, torch.ones(src.size(0), 1, device=src.device, dtype=src.dtype))
    return out / count.clamp(min=1)


class SAGEConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x, edge_index):
        n = x.size(0)
        if edge_index.numel() == 0:
            neigh = torch.zeros_like(x)
        else:
            src, dst = edge_index[0], edge_index[1]
            neigh = scatter_mean(x[src], dst, n)
        return self.lin(torch.cat([x, neigh], dim=-1))


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, n_layers=2, dropout=0.2):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * n_layers
        self.convs = nn.ModuleList([SAGEConv(dims[i], dims[i + 1]) for i in range(n_layers)])
        self.dropout = dropout
        self.out_dim = hidden_dim

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    @staticmethod
    def readout(node_embeddings, batch_idx, n_graphs):
        return scatter_mean(node_embeddings, batch_idx, n_graphs)


class GATConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.attn = nn.Linear(2 * out_dim, 1)

    def forward(self, x, edge_index):
        n = x.size(0)
        h = self.lin(x)
        if edge_index.numel() == 0:
            return F.relu(h)
        src, dst = edge_index[0], edge_index[1]
        e = F.leaky_relu(self.attn(torch.cat([h[src], h[dst]], dim=-1)), 0.2).squeeze(-1)
        e = e - e.max()
        exp_e = torch.exp(e)
        denom = torch.zeros(n, device=x.device).index_add_(0, dst, exp_e).clamp(min=1e-8)
        alpha = exp_e / denom[dst]
        out = torch.zeros_like(h).index_add_(0, dst, alpha.unsqueeze(-1) * h[src])
        return F.relu(out)


class GNNTagClassifier(nn.Module):
    def __init__(self, in_dim, n_tags, hidden_dim=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.gnn = GraphSAGE(in_dim, hidden_dim, n_layers, dropout)
        self.head = nn.Linear(hidden_dim, n_tags)

    def forward(self, node_feats, edge_index, batch_idx, n_graphs):
        h = self.gnn(node_feats, edge_index)
        g = GraphSAGE.readout(h, batch_idx, n_graphs)
        return self.head(g), g


class CNNBaseline(nn.Module):
    def __init__(self, in_dim, n_tags, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, n_tags)

    def forward(self, mel):
        return self.head(self.net(mel))