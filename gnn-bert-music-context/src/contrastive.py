"""Task 4: contrastive GNN-BERT dual encoder (InfoNCE) for retrieval."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualEncoder(nn.Module):
    def __init__(self, gnn_encoder, text_encoder, embed_dim=128):
        super().__init__()
        self.gnn = gnn_encoder
        self.text = text_encoder
        self.g_proj = nn.Linear(gnn_encoder.out_dim, embed_dim)
        self.t_proj = nn.Linear(text_encoder.embed_dim, embed_dim)

    def encode_graph(self, node_feats, edge_index, batch_idx, n_graphs):
        from src.gnn_model import GraphSAGE
        h = self.gnn(node_feats, edge_index)
        g = GraphSAGE.readout(h, batch_idx, n_graphs)
        return F.normalize(self.g_proj(g), dim=-1)

    def encode_text(self, input_ids):
        cls, _ = self.text(input_ids)
        return F.normalize(self.t_proj(cls), dim=-1)

    def forward(self, node_feats, edge_index, batch_idx, n_graphs, input_ids):
        return self.encode_graph(node_feats, edge_index, batch_idx, n_graphs), self.encode_text(input_ids)


def info_nce_loss(g_embed, t_embed, temperature=0.07):
    logits = g_embed @ t_embed.t() / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_g2t = F.cross_entropy(logits, labels)
    loss_t2g = F.cross_entropy(logits.t(), labels)
    return (loss_g2t + loss_t2g) / 2


@torch.no_grad()
def retrieval_recall_at_k(g_embed, t_embed, ks=(1, 5, 10)):
    sims = g_embed @ t_embed.t()
    n = sims.size(0)
    results = {}
    for direction, S in [("audio_to_caption", sims), ("caption_to_audio", sims.t())]:
        ranks = []
        for i in range(n):
            order = torch.argsort(S[i], descending=True)
            rank = (order == i).nonzero(as_tuple=True)[0].item()
            ranks.append(rank)
        ranks = torch.tensor(ranks)
        for k in ks:
            if k <= n:
                results[f"{direction}_R@{k}"] = float((ranks < k).float().mean())
    return results