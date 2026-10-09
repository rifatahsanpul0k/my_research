"""
Modular Model Architectures and Components for Spatial Multi-Omics.
Includes Models M0 through M5 and Stage 2 architectures:
- Model_H_HiRe (Hierarchical 2-Stage MLP + High-Dim Reconstruction + SAGEConv)
- Model_Attn_SAGE (Multihead Cross-Attention + SAGEConv)
- Model_Gated_ARISE (Learned Dynamic Edge Gating + GCNConv)
- Graph layers: SAGEConv, GCNConv, GATConv
"""

from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAGEConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.lin_l = nn.Linear(in_channels, out_channels, bias=True)
        self.lin_r = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        row, col = edge_index
        num_nodes = x.size(0)
        deg = torch.zeros(num_nodes, dtype=torch.float32, device=x.device)
        w = torch.ones_like(row, dtype=torch.float32) if edge_weight is None else edge_weight
        deg.scatter_add_(0, row, w)
        deg = deg.clamp(min=1).unsqueeze(1)

        msg = x[col] * (w.unsqueeze(1) if edge_weight is not None else 1.0)
        out = torch.zeros(num_nodes, self.in_channels, dtype=x.dtype, device=x.device)
        out.scatter_add_(0, row.unsqueeze(1).expand(-1, self.in_channels), msg)
        out = out / deg

        out = self.lin_l(x) + self.lin_r(out)
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out


class GCNConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        row, col = edge_index
        num_nodes = x.size(0)

        loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=x.device)
        r = torch.cat([row, loop_index])
        c = torch.cat([col, loop_index])

        if edge_weight is None:
            w = torch.ones(r.size(0), dtype=torch.float32, device=x.device)
        else:
            w = torch.cat([edge_weight, torch.ones(num_nodes, dtype=torch.float32, device=x.device)])

        deg = torch.zeros(num_nodes, dtype=torch.float32, device=x.device)
        deg.scatter_add_(0, r, w)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)

        norm_w = deg_inv_sqrt[r] * w * deg_inv_sqrt[c]
        h = self.lin(x)
        out = torch.zeros(num_nodes, h.size(1), dtype=x.dtype, device=x.device)
        out.scatter_add_(0, r.unsqueeze(1).expand(-1, h.size(1)), h[c] * norm_w.unsqueeze(1))
        return out + self.bias


class GATConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.head_dim = max(1, out_channels // heads)
        self.out_dim = heads * self.head_dim
        self.lin = nn.Linear(in_channels, self.out_dim, bias=False)
        self.att_src = nn.Parameter(torch.zeros(1, heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.zeros(1, heads, self.head_dim))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        row, col = edge_index
        num_nodes = x.size(0)
        h = self.lin(x).view(num_nodes, self.heads, self.head_dim)
        alpha_src = (h * self.att_src).sum(dim=-1)
        alpha_dst = (h * self.att_dst).sum(dim=-1)
        edge_scores = F.leaky_relu(alpha_src[row] + alpha_dst[col], negative_slope=0.2)
        if edge_weight is not None:
            edge_scores = edge_scores + torch.log(edge_weight.unsqueeze(1).clamp(min=1e-6))
        exp_scores = torch.exp(edge_scores - edge_scores.max())
        sum_exp = torch.zeros(num_nodes, self.heads, device=x.device)
        sum_exp.scatter_add_(0, row.unsqueeze(1).expand(-1, self.heads), exp_scores)
        alpha = exp_scores / (sum_exp[row] + 1e-12)
        msg = h[col] * alpha.unsqueeze(-1)
        out = torch.zeros(num_nodes, self.heads, self.head_dim, device=x.device)
        out.scatter_add_(0, row.unsqueeze(1).unsqueeze(2).expand(-1, self.heads, self.head_dim), msg)
        return out.view(num_nodes, self.out_dim)


# =========================================================================
# STAGE 2 & 3 CANDIDATE ARCHITECTURES, FUSION & CLUSTERING MODULES
# =========================================================================

class UAFFusionStage(nn.Module):
    """Uncertainty-Aware Fusion (UAF-Gaussian) per dimension."""
    def __init__(self, dim: int):
        super().__init__()
        self.lr = nn.Linear(dim, dim)
        self.la = nn.Linear(dim, dim)
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU())

    def forward(self, zr: torch.Tensor, za: torch.Tensor) -> torch.Tensor:
        wr = torch.sigmoid(-self.lr(zr))
        wa = torch.sigmoid(-self.la(za))
        return self.out((wr * zr + wa * za) / (wr + wa + 1e-9))


def sinkhorn(C: torch.Tensor, eps: float = 0.1, n_iter: int = 30) -> torch.Tensor:
    K = torch.exp(-C / eps)
    n, m = C.shape
    u = torch.full((n,), 1.0 / n, device=C.device)
    v = torch.full((m,), 1.0 / m, device=C.device)
    for _ in range(n_iter):
        u = (1.0 / n) / (K @ v + 1e-9)
        v = (1.0 / m) / (K.t() @ u + 1e-9)
    return torch.diag(u) @ K @ torch.diag(v)


def scot_loss(zr: torch.Tensor, za: torch.Tensor, m: int = 512, eps: float = 0.1) -> torch.Tensor:
    n = zr.size(0)
    if n > m:
        idx = torch.randperm(n, device=zr.device)[:m]
        zr_sub, za_sub = zr[idx], za[idx]
    else:
        zr_sub, za_sub = zr, za
    C = torch.cdist(zr_sub, za_sub)
    C_norm = C / (C.mean() + 1e-9)
    P = sinkhorn(C_norm, eps)
    return (P * C_norm).sum()


class ClusterHead(nn.Module):
    """Student-t DEC / IDEC clustering head for learnable cluster axis."""
    def __init__(self, num_clusters: int, in_features: int, alpha: float = 1.0):
        super().__init__()
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.cluster_centers = nn.Parameter(torch.Tensor(num_clusters, in_features))
        nn.init.xavier_uniform_(self.cluster_centers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_squared = torch.sum((x.unsqueeze(1) - self.cluster_centers.unsqueeze(0)) ** 2, dim=2)
        num = (1.0 + norm_squared / self.alpha) ** (-(self.alpha + 1.0) / 2.0)
        q = num / torch.sum(num, dim=1, keepdim=True)
        return q


def target_distribution(q: torch.Tensor) -> torch.Tensor:
    f = q.sum(dim=0)
    p = (q ** 2) / (f + 1e-9)
    p = p / p.sum(dim=1, keepdim=True)
    return p


def kl_clustering_loss(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return F.kl_div(q.log(), p, reduction="batchmean")


class Model_H_HiRe(nn.Module):
    """
    Stage 2 Winner & Stage 3 Base: Hierarchical High-Dim Autoencoder.
    Supports:
    - Fusion: '2stage' (MLP), 'uaf' (Uncertainty-Aware Gaussian), 'cross_attn', 'concat'
    - Backbone: 'sage' (SAGEConv), 'gat' (GATConv), 'gcn' (GCNConv)
    - Recon: 'raw' (3000-HVG), 'pca30', 'pca128', 'decoupled'
    """
    def __init__(
        self,
        in_rna_dim: int,
        in_mod2_dim: int,
        hidden_dim: int = 256,
        out_dim: int = 64,
        backbone: str = "sage",
        fusion: str = "2stage",
        recon_mode: str = "raw",
    ):
        super().__init__()
        self.fusion_type = fusion
        self.recon_mode = recon_mode
        self.backbone_type = backbone
        self.last_z_rna = None

        def make_conv(in_d, out_d):
            if backbone == "sage":
                return SAGEConv(in_d, out_d)
            elif backbone == "gat":
                return GATConv(in_d, out_d)
            else:
                return GCNConv(in_d, out_d)

        self.enc_rna_sim = make_conv(in_rna_dim, hidden_dim)
        self.enc_rna_dist = make_conv(in_rna_dim, hidden_dim)
        self.enc_mod2 = make_conv(in_mod2_dim, hidden_dim)

        self.proj_sim = nn.Linear(hidden_dim, out_dim)
        self.proj_dist = nn.Linear(hidden_dim, out_dim)
        self.proj_mod2 = nn.Linear(hidden_dim, out_dim)

        if fusion == "2stage":
            self.fusion_rna = nn.Sequential(nn.Linear(2 * out_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU())
            self.fusion_joint = nn.Sequential(nn.Linear(2 * out_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU())
        elif fusion == "uaf":
            self.fusion_rna = UAFFusionStage(out_dim)
            self.fusion_joint = UAFFusionStage(out_dim)
        elif fusion == "cross_attn":
            self.attn = nn.MultiheadAttention(embed_dim=out_dim, num_heads=4, batch_first=True)
            self.fusion_out = nn.Linear(out_dim, out_dim)
        else: # concat
            self.fusion_joint = nn.Sequential(nn.Linear(3 * out_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU())

        # Decoders
        self.dec_shared = nn.Sequential(nn.Linear(out_dim, hidden_dim), nn.ReLU())
        self.dec_rna = nn.Linear(hidden_dim, in_rna_dim)
        self.dec_mod2 = nn.Linear(hidden_dim, in_mod2_dim)
        self.dec_joint = nn.Linear(hidden_dim, in_rna_dim + in_mod2_dim)

    def forward(self, x_rna, x_mod2, e_sim, w_sim, e_dist, w_dist, e_com, w_com):
        h_sim = F.relu(self.enc_rna_sim(x_rna, e_sim, w_sim))
        h_dist = F.relu(self.enc_rna_dist(x_rna, e_dist, w_dist))
        h_mod2 = F.relu(self.enc_mod2(x_mod2, e_com, w_com))

        z_sim = self.proj_sim(h_sim)
        z_dist = self.proj_dist(h_dist)
        z_mod2 = self.proj_mod2(h_mod2)

        if self.fusion_type == "2stage":
            z_rna = self.fusion_rna(torch.cat([z_sim, z_dist], dim=1))
            z_final = self.fusion_joint(torch.cat([z_rna, z_mod2], dim=1))
        elif self.fusion_type == "uaf":
            z_rna = self.fusion_rna(z_sim, z_dist)
            z_final = self.fusion_joint(z_rna, z_mod2)
        elif self.fusion_type == "cross_attn":
            z_rna = (z_sim + z_dist) * 0.5
            stack = torch.stack([z_sim, z_dist, z_mod2], dim=1)
            attn_out, _ = self.attn(stack, stack, stack)
            z_final = self.fusion_out(attn_out.mean(dim=1))
        else:
            z_rna = (z_sim + z_dist) * 0.5
            z_final = self.fusion_joint(torch.cat([z_sim, z_dist, z_mod2], dim=1))

        self.last_z_rna = z_rna

        shared = self.dec_shared(z_final)
        rec_rna = self.dec_rna(shared)
        rec_mod2 = self.dec_mod2(shared)
        rec_joint = self.dec_joint(shared)

        return z_final, z_sim, z_dist, z_mod2, rec_rna, rec_mod2, rec_joint


class Model_Gated_ARISE(nn.Module):
    """
    Candidate 3: ARISE with Learned Dynamic Edge Gating.
    Prunes/downweights edges that cross transcriptomic boundaries.
    """
    def __init__(self, in_rna_dim: int, in_mod2_dim: int, hidden_dim: int = 256, out_dim: int = 64):
        super().__init__()
        # Edge gate networks
        self.gate_sim = nn.Sequential(nn.Linear(2 * in_rna_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.gate_dist = nn.Sequential(nn.Linear(2 * in_rna_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

        self.enc_rna1 = GCNConv(in_rna_dim, hidden_dim)
        self.enc_rna2 = GCNConv(in_rna_dim, hidden_dim)
        self.enc_mod2 = GCNConv(in_mod2_dim, out_dim)

        self.sim_conv = GCNConv(hidden_dim, out_dim)
        self.dist_conv = GCNConv(hidden_dim, out_dim)

        self.fusion1 = nn.Sequential(nn.Linear(2 * out_dim, out_dim), nn.ReLU())
        self.fusion2 = nn.Linear(2 * out_dim, out_dim)

        self.dec_shared = nn.Sequential(nn.Linear(out_dim, hidden_dim), nn.ReLU())
        self.dec_rna = nn.Linear(hidden_dim, in_rna_dim)
        self.dec_mod2 = nn.Linear(hidden_dim, in_mod2_dim)
        self.dec_joint = nn.Linear(hidden_dim, in_rna_dim + in_mod2_dim)

    def forward(self, x_rna, x_mod2, e_sim, w_sim, e_dist, w_dist, e_com, w_com):
        r_sim, c_sim = e_sim
        edge_feat_sim = torch.cat([x_rna[r_sim], x_rna[c_sim]], dim=1)
        gate_w_sim = self.gate_sim(edge_feat_sim).squeeze(-1) * (w_sim if w_sim is not None else 1.0)

        r_dist, c_dist = e_dist
        edge_feat_dist = torch.cat([x_rna[r_dist], x_rna[c_dist]], dim=1)
        gate_w_dist = self.gate_dist(edge_feat_dist).squeeze(-1) * (w_dist if w_dist is not None else 1.0)

        xs = F.relu(self.enc_rna1(x_rna, e_sim, gate_w_sim))
        xd = F.relu(self.enc_rna2(x_rna, e_dist, gate_w_dist))

        x_sim = self.sim_conv(xs, e_sim, gate_w_sim)
        x_dist = self.dist_conv(xd, e_dist, gate_w_dist)
        pro = self.enc_mod2(x_mod2, e_com, w_com)

        fused = self.fusion1(torch.cat([x_sim, x_dist], dim=1))
        z_final = self.fusion2(torch.cat([fused, pro], dim=1))

        shared = self.dec_shared(z_final)
        rec_rna = self.dec_rna(shared)
        rec_mod2 = self.dec_mod2(shared)
        rec_joint = self.dec_joint(shared)

        return z_final, x_sim, x_dist, pro, rec_rna, rec_mod2, rec_joint, gate_w_dist


# Legacy models (M0-M5) for Stage 1 reproduction and backward compatibility
class Model0_BaseSMART(nn.Module):
    def __init__(self, in_rna_dim: int, in_mod2_dim: int, hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.enc_rna = SAGEConv(in_rna_dim, hidden_dim)
        self.enc_mod2 = SAGEConv(in_mod2_dim, hidden_dim)
        self.fusion = nn.Linear(2 * hidden_dim, out_dim)
        self.dec_rna = nn.Linear(out_dim, in_rna_dim)
        self.dec_mod2 = nn.Linear(out_dim, in_mod2_dim)

    def forward(self, x_rna, x_mod2, e_sim, e_dist, e_com):
        h_rna = F.relu(self.enc_rna(x_rna, e_dist))
        h_mod2 = F.relu(self.enc_mod2(x_mod2, e_dist))
        z = self.fusion(torch.cat([h_rna, h_mod2], dim=1))
        return z, self.dec_rna(z), self.dec_mod2(z)


class Model1_DualGraphSMART(nn.Module):
    def __init__(self, in_rna_dim: int, in_mod2_dim: int, hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.enc_sim = SAGEConv(in_rna_dim, hidden_dim)
        self.enc_dist = SAGEConv(in_rna_dim, hidden_dim)
        self.enc_mod2 = SAGEConv(in_mod2_dim, hidden_dim)
        self.fusion = nn.Linear(3 * hidden_dim, out_dim)
        self.dec_sim = nn.Linear(out_dim, in_rna_dim)
        self.dec_dist = nn.Linear(out_dim, in_rna_dim)
        self.dec_mod2 = nn.Linear(out_dim, in_mod2_dim)

    def forward(self, x_rna, x_mod2, e_sim, e_dist, e_com):
        h_sim = F.relu(self.enc_sim(x_rna, e_sim))
        h_dist = F.relu(self.enc_dist(x_rna, e_dist))
        h_mod2 = F.relu(self.enc_mod2(x_mod2, e_com))
        z = self.fusion(torch.cat([h_sim, h_dist, h_mod2], dim=1))
        return z, self.dec_sim(z), self.dec_dist(z), self.dec_mod2(z)


class Model2_HierarchicalSMART(nn.Module):
    def __init__(self, in_rna_dim: int, in_mod2_dim: int, hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.enc_sim = SAGEConv(in_rna_dim, hidden_dim)
        self.enc_dist = SAGEConv(in_rna_dim, hidden_dim)
        self.enc_mod2 = SAGEConv(in_mod2_dim, hidden_dim)
        self.fusion1 = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU())
        self.fusion2 = nn.Linear(2 * hidden_dim, out_dim)
        self.dec_joint = nn.Linear(out_dim, in_rna_dim + in_mod2_dim)
        self.dec_sim = nn.Linear(hidden_dim, in_rna_dim)
        self.dec_dist = nn.Linear(hidden_dim, in_rna_dim)
        self.dec_mod2 = nn.Linear(hidden_dim, in_mod2_dim)

    def forward(self, x_rna, x_mod2, e_sim, e_dist, e_com):
        h_sim = F.relu(self.enc_sim(x_rna, e_sim))
        h_dist = F.relu(self.enc_dist(x_rna, e_dist))
        h_mod2 = F.relu(self.enc_mod2(x_mod2, e_com))
        z_rna = self.fusion1(torch.cat([h_sim, h_dist], dim=1))
        z_joint = self.fusion2(torch.cat([z_rna, h_mod2], dim=1))
        return z_joint, z_rna, self.dec_joint(z_joint), self.dec_sim(h_sim), self.dec_dist(h_dist), self.dec_mod2(h_mod2)


class Model3_ContrastiveSMART(Model2_HierarchicalSMART):
    pass


class Model4_HighDimSMART(Model2_HierarchicalSMART):
    pass


class Model5_FullARISE(nn.Module):
    def __init__(self, in_rna_dim: int, in_mod2_dim: int, hidden_dim: int = 512, out_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.x_RNA1 = GCNConv(in_rna_dim, hidden_dim)
        self.x_RNA2 = GCNConv(in_rna_dim, hidden_dim)
        self.mod2_conv = GCNConv(in_mod2_dim, out_dim)
        self.sim_conv = GCNConv(hidden_dim, out_dim)
        self.dist_conv = GCNConv(hidden_dim, out_dim)
        self.fusion1 = nn.Sequential(nn.Linear(2 * out_dim, out_dim), nn.ReLU())
        self.fusion2 = nn.Linear(2 * out_dim, out_dim)
        self.dropout = dropout
        self.deconv1 = nn.Sequential(nn.Linear(out_dim, hidden_dim), nn.ReLU())
        self.deconv_sim = nn.Linear(hidden_dim, in_rna_dim)
        self.deconv_dist = nn.Linear(hidden_dim, in_rna_dim)
        self.deconv_mod2 = nn.Linear(hidden_dim, in_mod2_dim)
        self.deconv_joint = nn.Linear(hidden_dim, in_rna_dim + in_mod2_dim)

    def forward(self, x_rna, x_mod2, e_sim, w_sim, e_dist, w_dist, e_com, w_com):
        xs = F.relu(self.x_RNA1(x_rna, e_sim, w_sim))
        xs = F.dropout(xs, self.dropout, training=self.training)
        xd = F.relu(self.x_RNA2(x_rna, e_dist, w_dist))
        xd = F.dropout(xd, self.dropout, training=self.training)
        x_sim = self.sim_conv(xs, e_sim, w_sim)
        x_dist = self.dist_conv(xd, e_dist, w_dist)
        pro = self.mod2_conv(x_mod2, e_com, w_com)
        fused = self.fusion1(torch.cat([x_sim, x_dist], dim=1))
        fused_pro = self.fusion2(torch.cat([fused, pro], dim=1))
        rec_sim = self.deconv_sim(self.deconv1(x_sim))
        rec_dist = self.deconv_dist(self.deconv1(x_dist))
        rec_pro = self.deconv_mod2(self.deconv1(pro))
        rec_joint = self.deconv_joint(self.deconv1(fused_pro))
        return fused_pro, fused, x_sim, x_dist, pro, rec_joint, rec_sim, rec_dist, rec_pro


class ModelAblation_AttentionFusion(nn.Module):
    def __init__(self, in_rna_dim: int, in_mod2_dim: int, hidden_dim: int = 512, out_dim: int = 64):
        super().__init__()
        self.x_RNA1 = GCNConv(in_rna_dim, hidden_dim)
        self.x_RNA2 = GCNConv(in_rna_dim, hidden_dim)
        self.mod2_conv = GCNConv(in_mod2_dim, out_dim)
        self.sim_conv = GCNConv(hidden_dim, out_dim)
        self.dist_conv = GCNConv(hidden_dim, out_dim)
        self.attn = nn.MultiheadAttention(embed_dim=out_dim, num_heads=4, batch_first=True)
        self.fc_out = nn.Linear(out_dim, out_dim)
        self.deconv1 = nn.Sequential(nn.Linear(out_dim, hidden_dim), nn.ReLU())
        self.deconv_joint = nn.Linear(hidden_dim, in_rna_dim + in_mod2_dim)

    def forward(self, x_rna, x_mod2, e_sim, w_sim, e_dist, w_dist, e_com, w_com):
        xs = F.relu(self.x_RNA1(x_rna, e_sim, w_sim))
        xd = F.relu(self.x_RNA2(x_rna, e_dist, w_dist))
        x_sim = self.sim_conv(xs, e_sim, w_sim)
        x_dist = self.dist_conv(xd, e_dist, w_dist)
        pro = self.mod2_conv(x_mod2, e_com, w_com)
        stack = torch.stack([x_sim, x_dist, pro], dim=1)
        attn_out, _ = self.attn(stack, stack, stack)
        fused = self.fc_out(attn_out.mean(dim=1))
        rec_joint = self.deconv_joint(self.deconv1(fused))
        return fused, fused, x_sim, x_dist, pro, rec_joint, None, None, None
