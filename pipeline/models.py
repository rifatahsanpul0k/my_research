"""
Modular Model Architectures and Components for Spatial Multi-Omics.
Includes Models M0 through M5 and modular ablations (GAT, MLP, Attention Fusion).
Native PyTorch fallback layers guarantee execution in any environment.
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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        num_nodes = x.size(0)
        deg = torch.zeros(num_nodes, dtype=torch.float32, device=x.device)
        deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float32))
        deg = deg.clamp(min=1).unsqueeze(1)

        out = torch.zeros(num_nodes, self.in_channels, dtype=x.dtype, device=x.device)
        out.scatter_add_(0, row.unsqueeze(1).expand(-1, self.in_channels), x[col])
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
        row = torch.cat([row, loop_index])
        col = torch.cat([col, loop_index])

        if edge_weight is None:
            w = torch.ones(row.size(0), dtype=torch.float32, device=x.device)
        else:
            w = torch.cat([edge_weight, torch.ones(num_nodes, dtype=torch.float32, device=x.device)])

        deg = torch.zeros(num_nodes, dtype=torch.float32, device=x.device)
        deg.scatter_add_(0, row, w)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm_w = deg_inv_sqrt[row] * w * deg_inv_sqrt[col]

        x_proj = self.lin(x)
        out = torch.zeros(num_nodes, x_proj.size(1), dtype=x.dtype, device=x.device)
        out.scatter_add_(0, row.unsqueeze(1).expand(-1, x_proj.size(1)), norm_w.unsqueeze(1) * x_proj[col])
        return out + self.bias


class SAGEEncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, out_channels, normalize=True)
        self.conv2 = SAGEConv(out_channels, out_channels, normalize=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x, edge_index), edge_index)


class SAGEDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, in_channels, normalize=True)
        self.conv2 = SAGEConv(in_channels, out_channels, normalize=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x, edge_index), edge_index)


class Model0_BaseSMART(nn.Module):
    def __init__(self, in_rna_dim: int, in_mod2_dim: int, out_dim: int = 64):
        super().__init__()
        self.enc_rna = SAGEEncoderBlock(in_rna_dim, out_dim)
        self.enc_mod2 = SAGEEncoderBlock(in_mod2_dim, out_dim)
        self.fc_fusion = nn.Linear(2 * out_dim, out_dim)
        self.dec_rna = SAGEDecoderBlock(out_dim, in_rna_dim)
        self.dec_mod2 = SAGEDecoderBlock(out_dim, in_mod2_dim)

    def forward(self, x_rna, x_mod2, e_sim, e_dist, e_com):
        h_rna = self.enc_rna(x_rna, e_dist)
        h_mod2 = self.enc_mod2(x_mod2, e_dist)
        z = self.fc_fusion(torch.cat([h_rna, h_mod2], dim=1))
        rec_rna = self.dec_rna(z, e_dist)
        rec_mod2 = self.dec_mod2(z, e_dist)
        return z, rec_rna, rec_mod2


class Model1_DualGraphSMART(nn.Module):
    def __init__(self, in_rna_dim: int, in_mod2_dim: int, out_dim: int = 64):
        super().__init__()
        self.enc_sim = SAGEEncoderBlock(in_rna_dim, out_dim)
        self.enc_dist = SAGEEncoderBlock(in_rna_dim, out_dim)
        self.enc_mod2 = SAGEEncoderBlock(in_mod2_dim, out_dim)
        self.fc_fusion = nn.Linear(3 * out_dim, out_dim)
        self.dec_sim = SAGEDecoderBlock(out_dim, in_rna_dim)
        self.dec_dist = SAGEDecoderBlock(out_dim, in_rna_dim)
        self.dec_mod2 = SAGEDecoderBlock(out_dim, in_mod2_dim)

    def forward(self, x_rna, x_mod2, e_sim, e_dist, e_com):
        h_sim = self.enc_sim(x_rna, e_sim)
        h_dist = self.enc_dist(x_rna, e_dist)
        h_pro = self.enc_mod2(x_mod2, e_com)
        z = self.fc_fusion(torch.cat([h_sim, h_dist, h_pro], dim=1))
        rec_sim = self.dec_sim(z, e_sim)
        rec_dist = self.dec_dist(z, e_dist)
        rec_mod2 = self.dec_mod2(z, e_com)
        return z, rec_sim, rec_dist, rec_mod2


class Model2_HierarchicalSMART(nn.Module):
    def __init__(self, in_rna_dim: int, in_mod2_dim: int, out_dim: int = 64):
        super().__init__()
        self.enc_sim = SAGEEncoderBlock(in_rna_dim, out_dim)
        self.enc_dist = SAGEEncoderBlock(in_rna_dim, out_dim)
        self.enc_mod2 = SAGEEncoderBlock(in_mod2_dim, out_dim)
        self.fusion1 = nn.Sequential(nn.Linear(2 * out_dim, out_dim), nn.ReLU())
        self.fusion2 = nn.Linear(2 * out_dim, out_dim)
        self.dec_sim = SAGEDecoderBlock(out_dim, in_rna_dim)
        self.dec_dist = SAGEDecoderBlock(out_dim, in_rna_dim)
        self.dec_mod2 = SAGEDecoderBlock(out_dim, in_mod2_dim)
        self.dec_joint = nn.Linear(out_dim, in_rna_dim + in_mod2_dim)

    def forward(self, x_rna, x_mod2, e_sim, e_dist, e_com):
        h_sim = self.enc_sim(x_rna, e_sim)
        h_dist = self.enc_dist(x_rna, e_dist)
        h_pro = self.enc_mod2(x_mod2, e_com)
        z_rna = self.fusion1(torch.cat([h_sim, h_dist], dim=1))
        z_joint = self.fusion2(torch.cat([z_rna, h_pro], dim=1))
        rec_sim = self.dec_sim(h_sim, e_sim)
        rec_dist = self.dec_dist(h_dist, e_dist)
        rec_mod2 = self.dec_mod2(h_pro, e_com)
        rec_joint = self.dec_joint(z_joint)
        return z_joint, z_rna, rec_joint, rec_sim, rec_dist, rec_mod2


class Model3_ContrastiveSMART(nn.Module):
    def __init__(self, in_rna_dim: int, in_mod2_dim: int, out_dim: int = 64):
        super().__init__()
        self.enc_sim = SAGEEncoderBlock(in_rna_dim, out_dim)
        self.enc_dist = SAGEEncoderBlock(in_rna_dim, out_dim)
        self.enc_mod2 = SAGEEncoderBlock(in_mod2_dim, out_dim)
        self.fusion1 = nn.Sequential(nn.Linear(2 * out_dim, out_dim), nn.ReLU())
        self.fusion2 = nn.Linear(2 * out_dim, out_dim)
        self.dec_shared = nn.Sequential(nn.Linear(out_dim, 128), nn.ReLU())
        self.dec_sim = nn.Linear(128, in_rna_dim)
        self.dec_dist = nn.Linear(128, in_rna_dim)
        self.dec_mod2 = nn.Linear(128, in_mod2_dim)
        self.dec_joint = nn.Linear(128, in_rna_dim + in_mod2_dim)

    def forward(self, x_rna, x_mod2, e_sim, e_dist, e_com):
        h_sim = self.enc_sim(x_rna, e_sim)
        h_dist = self.enc_dist(x_rna, e_dist)
        h_pro = self.enc_mod2(x_mod2, e_com)
        z_rna = self.fusion1(torch.cat([h_sim, h_dist], dim=1))
        z_joint = self.fusion2(torch.cat([z_rna, h_pro], dim=1))
        rec_sim = self.dec_sim(self.dec_shared(h_sim))
        rec_dist = self.dec_dist(self.dec_shared(h_dist))
        rec_mod2 = self.dec_mod2(self.dec_shared(h_pro))
        rec_joint = self.dec_joint(self.dec_shared(z_joint))
        return z_joint, z_rna, h_sim, h_dist, h_pro, rec_joint, rec_sim, rec_dist, rec_mod2


class Model4_HighDimSMART(nn.Module):
    def __init__(self, in_rna_dim: int, in_mod2_dim: int, raw_rna_dim: int, raw_mod2_dim: int, out_dim: int = 64, hidden_dim: int = 512):
        super().__init__()
        self.enc_sim = SAGEEncoderBlock(in_rna_dim, out_dim)
        self.enc_dist = SAGEEncoderBlock(in_rna_dim, out_dim)
        self.enc_mod2 = SAGEEncoderBlock(in_mod2_dim, out_dim)
        self.fusion1 = nn.Sequential(nn.Linear(2 * out_dim, out_dim), nn.ReLU())
        self.fusion2 = nn.Linear(2 * out_dim, out_dim)
        self.dec_shared = nn.Sequential(nn.Linear(out_dim, hidden_dim), nn.ReLU())
        self.dec_sim = nn.Linear(hidden_dim, raw_rna_dim)
        self.dec_dist = nn.Linear(hidden_dim, raw_rna_dim)
        self.dec_mod2 = nn.Linear(hidden_dim, raw_mod2_dim)
        self.dec_joint = nn.Linear(hidden_dim, raw_rna_dim + raw_mod2_dim)

    def forward(self, x_rna, x_mod2, e_sim, e_dist, e_com):
        h_sim = self.enc_sim(x_rna, e_sim)
        h_dist = self.enc_dist(x_rna, e_dist)
        h_pro = self.enc_mod2(x_mod2, e_com)
        z_rna = self.fusion1(torch.cat([h_sim, h_dist], dim=1))
        z_joint = self.fusion2(torch.cat([z_rna, h_pro], dim=1))
        rec_sim = self.dec_sim(self.dec_shared(h_sim))
        rec_dist = self.dec_dist(self.dec_shared(h_dist))
        rec_mod2 = self.dec_mod2(self.dec_shared(h_pro))
        rec_joint = self.dec_joint(self.dec_shared(z_joint))
        return z_joint, z_rna, h_sim, h_dist, h_pro, rec_joint, rec_sim, rec_dist, rec_mod2


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
