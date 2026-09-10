"""Task 3: GNN-BERT cross-attention fusion for multi-context understanding."""
import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    def __init__(self, graph_dim, text_dim, n_heads=4, dropout=0.1):
        super().__init__()
        assert text_dim % n_heads == 0, "text_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = text_dim // n_heads
        self.q_proj = nn.Linear(graph_dim, text_dim)
        self.k_proj = nn.Linear(text_dim, text_dim)
        self.v_proj = nn.Linear(text_dim, text_dim)
        self.out_dim = graph_dim + text_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, h_text):
        B, L, D = h_text.shape
        q = self.q_proj(g).view(B, self.n_heads, 1, self.head_dim)
        k = self.k_proj(h_text).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h_text).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        attn = torch.softmax((q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)
        attn = self.dropout(attn)
        ctx = (attn @ v).view(B, D)
        z = torch.cat([g, ctx], dim=-1)
        return z, attn.mean(dim=1).squeeze(1)


class GNNBertFusion(nn.Module):
    def __init__(self, gnn_encoder, text_encoder, n_tags, fusion_mode="cross_attention", dropout=0.2):
        super().__init__()
        self.gnn = gnn_encoder
        self.text = text_encoder
        self.fusion_mode = fusion_mode
        g_dim, t_dim = gnn_encoder.out_dim, text_encoder.embed_dim

        if fusion_mode == "cross_attention":
            self.fusion = CrossAttentionFusion(g_dim, t_dim)
            z_dim = self.fusion.out_dim
        elif fusion_mode == "concat":
            self.fusion = None
            z_dim = g_dim + t_dim
        elif fusion_mode == "gnn_only":
            self.fusion = None
            z_dim = g_dim
        elif fusion_mode == "bert_only":
            self.fusion = None
            z_dim = t_dim
        else:
            raise ValueError(fusion_mode)

        self.norm = nn.LayerNorm(z_dim)
        self.dropout = nn.Dropout(dropout)
        self.tag_head = nn.Linear(z_dim, n_tags)
        self.emotion_head = nn.Linear(z_dim, 2)

    def forward(self, node_feats, edge_index, batch_idx, n_graphs, input_ids):
        from src.gnn_model import GraphSAGE
        h_nodes = self.gnn(node_feats, edge_index)
        g = GraphSAGE.readout(h_nodes, batch_idx, n_graphs)
        cls, h_text = self.text(input_ids)

        attn = None
        if self.fusion_mode == "cross_attention":
            z, attn = self.fusion(g, h_text)
        elif self.fusion_mode == "concat":
            z = torch.cat([g, cls], dim=-1)
        elif self.fusion_mode == "gnn_only":
            z = g
        else:
            z = cls

        z = self.dropout(self.norm(z))
        tag_logits = self.tag_head(z)
        va_pred = self.emotion_head(z)
        return tag_logits, va_pred, attn


def multitask_loss(tag_logits, tags, va_pred, va_true, alpha=1.0, beta=1.0, pos_weight=None):
    tag_loss = nn.functional.binary_cross_entropy_with_logits(tag_logits, tags, pos_weight=pos_weight)
    v_loss = nn.functional.mse_loss(va_pred[:, 0], va_true[:, 0])
    a_loss = nn.functional.mse_loss(va_pred[:, 1], va_true[:, 1])
    return tag_loss + alpha * v_loss + beta * a_loss, dict(
        tag_loss=tag_loss.item(), v_loss=v_loss.item(), a_loss=a_loss.item())