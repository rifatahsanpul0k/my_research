"""Generate notebooks/embedding_module_sweep.ipynb — embedding generation test
over all unsupervised modules from recent papers (2024-2026).

Each combination = baseline pipeline with ONE module swapped in (one-module-at-a-time
screening). The notebook trains the fused embedding and evaluates label-free.
Run: python3 scripts/gen_embedding_sweep_nb.py
"""
import json, os

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

cells = []

cells.append(md('''# Embedding Generation Test — All Unsupervised Modules (Recent Papers 2024–2026)

**Goal:** generate the best **fused embedding**. This notebook screens every unsupervised
module from recent papers as a one-module-at-a-time embedding test.

**Protocol (per combination × dataset × seed):**
raw multimodal data → preprocessing (§3.3) → graph → modality encoders → fusion →
**fused embedding Z (32-D)** → k-means (k = declared ground-truth count, declared exception) →
label-free eval (silhouette, seed-stability ARI). Post-hoc ARI quarantined to `posthoc.json`.

**Combinations (26):** 1 baseline + 25 single-module swaps (encoders / graphs / fusion /
alignment / losses). Full list in the registry cell. Results push to `runs/sweep/<combo>/`
on GitHub after every combination; resume via `skip_done`.
'''))

cells.append(code('''import os, sys, json, subprocess, random, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for pkg in ["scanpy", "anndata", "scikit-learn", "pandas"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

import scanpy as sc
import anndata as ad
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

print("imports ok")
'''))

cells.append(code('''# ---------- GitHub push (guarded: the test runs even if push fails) ----------
try:
    from kaggle_secrets import UserSecretsClient
    import base64
    _sec = UserSecretsClient()
    _b64 = base64.b64encode(f"x-access-token:{_sec.get_secret('GITHUB_TOKEN')}".encode()).decode()
    _GIT = ["git", "-c", f"http.extraHeader=Authorization: Basic {_b64}"]
    REPO = "/kaggle/working/my_research"
    def sh(*args):
        subprocess.run(list(args), check=True, cwd=REPO)
    if not os.path.isdir(os.path.join(REPO, ".git")):
        subprocess.run(_GIT + ["clone", "https://github.com/rifatahsanpul0k/my_research.git", REPO], check=True)
        sh("git", "config", "user.name", "kaggle-runner")
        sh("git", "config", "user.email", "kaggle-runner@local")
    else:
        sh(*_GIT, "pull", "--rebase", "origin", "main")
    GIT_OK = True
    print("GitHub sync ready")
except Exception as e:
    GIT_OK = False
    print("GitHub sync unavailable (local save only):", e)

def git_push(paths, msg):
    if not GIT_OK:
        return
    try:
        sh(*_GIT, "pull", "--rebase", "origin", "main")
        sh("git", "add", *paths)
        sh("git", "commit", "-m", msg)
        sh(*_GIT, "push", "origin", "main")
        print("pushed:", msg)
    except Exception as e:
        print("push failed (continuing):", e)
'''))

cells.append(code('''# ---------- data ----------
DATA_CANDIDATES = ["/kaggle/input/spatial-multiomics-6datasets-private"]
DATA_ROOT = next((p for p in DATA_CANDIDATES if os.path.isdir(p)), None)
if DATA_ROOT is None:
    # fallback: download via Kaggle API (credentials from secrets)
    from kaggle_secrets import UserSecretsClient
    _s = UserSecretsClient()
    kd = os.path.expanduser("~/.kaggle"); os.makedirs(kd, exist_ok=True)
    json.dump({"username": _s.get_secret("KAGGLE_USERNAME"), "key": _s.get_secret("KAGGLE_KEY")},
              open(os.path.join(kd, "kaggle.json"), "w"))
    os.chmod(os.path.join(kd, "kaggle.json"), 0o600)
    subprocess.run(["kaggle", "datasets", "download", "-d", "pulokpulok/spatial-multiomics-6datasets-private",
                    "-p", "/kaggle/working", "--unzip"], check=True)
    DATA_ROOT = "/kaggle/working/spatial-multiomics-6datasets-private"
print("DATA_ROOT:", DATA_ROOT)

DATASETS = {
    "A1":  dict(folder="10x_human_lymph_node_A1", aux="adt",  gt_file="annotation.csv", gt_col="manual-anno"),
    "D1":  dict(folder="10x_human_lymph_node_D1", aux="adt",  gt_file="annotation.csv", gt_col="manual-anno"),
    "E11": dict(folder="Mouse_Brain_E11_S1", aux="atac", gt_file="anno.csv", gt_col="cluster"),
    "E13": dict(folder="Mouse_Brain_E13_S1", aux="atac", gt_file="anno.csv", gt_col="cluster"),
    "E15": dict(folder="Mouse_Brain_E15_S1", aux="atac", gt_file="anno.csv", gt_col="cluster"),
    "E18": dict(folder="Mouse_Brain_E18_S1", aux="atac", gt_file="anno.csv", gt_col="cluster"),
}
SEEDS = [42, 1234, 2024]

def to_dense_small(X, max_elems=5e7):
    if hasattr(X, "toarray"):
        return X.toarray() if X.size <= max_elems else X
    return np.asarray(X)

def clr(X):
    X = np.asarray(X, dtype=np.float64)
    gm = np.exp(np.log1p(X).mean(axis=1, keepdims=True))
    return np.log1p(X / gm)

def load_dataset(ds_id):
    spec = DATASETS[ds_id]
    root = os.path.join(DATA_ROOT, spec["folder"])
    rna = sc.read_h5ad(os.path.join(root, "adata_RNA.h5ad"))
    rna.var_names_make_unique()
    coords = np.asarray(rna.obsm["spatial"], dtype=np.float64)
    # RNA: normalize/log/HVG3000/PCA64/scale  (SCTransform-v2 screening approx)
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    sc.pp.highly_variable_genes(rna, n_top_genes=3000, flavor="seurat", inplace=True)
    Xh = rna[:, rna.var.highly_variable].X
    Xh = Xh.toarray() if hasattr(Xh, "toarray") else np.asarray(Xh)
    Xr = StandardScaler().fit_transform(PCA(n_components=64, random_state=0).fit_transform(Xh))
    if spec["aux"] == "adt":
        aux = sc.read_h5ad(os.path.join(root, "adata_ADT.h5ad"))
        aux.var_names_make_unique()
        Xa = StandardScaler().fit_transform(clr(to_dense_small(aux.X)))
    else:
        aux = sc.read_h5ad(os.path.join(root, "adata_ATAC.h5ad"))
        aux.var_names_make_unique()
        X = aux.X.tocsr().astype(np.float64) if hasattr(aux.X, "tocsr") else sparse.csr_matrix(np.asarray(aux.X, dtype=np.float64))
        rs = np.asarray(X.sum(axis=1)).ravel() + 1e-9
        tf = sparse.diags(1.0 / rs).dot(X)
        df = np.asarray((X > 0).sum(axis=0)).ravel()
        idf = np.log(1 + X.shape[0] / (df + 1))
        lsi = TruncatedSVD(n_components=50, random_state=0).fit_transform(tf.multiply(idf))
        depth = np.log1p(np.asarray(X.sum(axis=1)).ravel())
        keep = [i for i in range(50) if abs(np.corrcoef(lsi[:, i], depth)[0, 1]) <= 0.75]
        Xa = StandardScaler().fit_transform(lsi[:, keep])
    # declared k: read ONCE from annotations (declared exception, BRIEF-003)
    gt = pd.read_csv(os.path.join(root, spec["gt_file"]), index_col=0)
    k_declared = int(gt[spec["gt_col"]].nunique())
    print(f"{ds_id}: RNA {Xr.shape} AUX {Xa.shape} N={Xr.shape[0]} k_declared={k_declared}", flush=True)
    return dict(Xr=Xr.astype(np.float32), Xa=Xa.astype(np.float32),
                coords=coords, k=k_declared, gt=np.asarray(gt[spec["gt_col"]]))
'''))

cells.append(code('''# ---------- graphs (pure torch, no pyg dependency) ----------
def knn_edge_index(coords, k=15):
    A = kneighbors_graph(coords, k, mode="connectivity", include_self=False)
    A = A.maximum(A.T)
    coo = A.tocoo()
    ei = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    n = coords.shape[0]
    loop = torch.arange(n).repeat(2, 1)
    return torch.cat([ei, loop], dim=1)  # with self-loops

def norm_adj(edge_index, n, device, with_loop=True):
    idx = edge_index.to(device)
    if not with_loop:
        mask = idx[0] != idx[1]
        idx = idx[:, mask]
    else:
        loop = torch.arange(n, device=device).repeat(2, 1)
        idx = torch.cat([idx, loop], dim=1)
    vals = torch.ones(idx.shape[1], device=device)
    deg = torch.zeros(n, device=device).scatter_add_(0, idx[0], vals)
    d = deg.pow(-0.5); d[torch.isinf(d)] = 0
    vals = d[idx[0]] * vals * d[idx[1]]
    return torch.sparse_coo_tensor(idx, vals, (n, n), device=device).coalesce()

def build_graph(kind, coords, Xr, device):
    n = coords.shape[0]
    if kind == "spatial15":
        ei = knn_edge_index(coords, 15)
        return dict(edge_index=ei, adj=norm_adj(ei, n, device, True),
                    adj_nl=norm_adj(ei, n, device, False), coords=torch.tensor(coords, dtype=torch.float32, device=device))
    if kind == "dual":
        e1 = knn_edge_index(coords, 15)
        Xp = PCA(n_components=16, random_state=0).fit_transform(Xr)
        e2 = knn_edge_index(Xp, 15)
        ei = torch.cat([e1, e2], dim=1)
        return dict(edge_index=ei, adj=norm_adj(ei, n, device, True),
                    adj_nl=norm_adj(ei, n, device, False), coords=torch.tensor(coords, dtype=torch.float32, device=device))
    if kind == "stague":
        # STAGUE-style learned structure: attention weights over spatial kNN candidates
        ei = knn_edge_index(coords, 15)
        return dict(edge_index=ei, adj=None, adj_nl=None, learn_adj=True,
                    coords=torch.tensor(coords, dtype=torch.float32, device=device),
                    Xr_t=torch.tensor(Xr, dtype=torch.float32, device=device))
    raise ValueError(kind)

class LearnedAdj(nn.Module):
    """Learnable edge weights over kNN candidates (STAGUE screening approx)."""
    def __init__(self, d_in, n):
        super().__init__()
        self.w = nn.Sequential(nn.Linear(2 * d_in, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x, edge_index, n, device):
        src, dst = edge_index
        e = torch.cat([x[src], x[dst]], dim=1)
        alpha = torch.sigmoid(self.w(e)).squeeze(-1)
        idx = torch.cat([edge_index, torch.arange(n, device=device).repeat(2, 1)], dim=1)
        extra = torch.ones(n, device=device)
        vals = torch.cat([alpha, extra])
        deg = torch.zeros(n, device=device).scatter_add_(0, idx[0], vals)
        d = deg.pow(-0.5); d[torch.isinf(d)] = 0
        vals = d[idx[0]] * vals * d[idx[1]]
        return torch.sparse_coo_tensor(idx, vals, (n, n), device=device).coalesce()
'''))

cells.append(code('''# ---------- encoders: X -> 32-D embedding ----------
D_EMB, D_HID = 32, 128

class GCNEncoder(nn.Module):  # spectral GCN
    def __init__(self, d_in):
        super().__init__()
        self.l1 = nn.Linear(d_in, D_HID); self.l2 = nn.Linear(D_HID, D_EMB)
    def forward(self, x, g):
        h = torch.sparse.mm(g["adj"], self.l1(x)).relu()
        return torch.sparse.mm(g["adj"], self.l2(h))

class SAGEEncoder(nn.Module):  # inductive mean aggregation
    def __init__(self, d_in):
        super().__init__()
        self.l1 = nn.Linear(2 * d_in, D_HID); self.l2 = nn.Linear(2 * D_HID, D_EMB)
    def forward(self, x, g):
        agg = torch.sparse.mm(g["adj_nl"], x)
        h = self.l1(torch.cat([x, agg], 1)).relu()
        agg2 = torch.sparse.mm(g["adj_nl"], h)
        return self.l2(torch.cat([h, agg2], 1))

class GATEncoder(nn.Module):  # graph attention (4 heads)
    def __init__(self, d_in, heads=4, dh=32):
        super().__init__()
        self.heads, self.dh = heads, dh
        self.W = nn.Linear(d_in, dh * heads, bias=False)
        self.a = nn.Parameter(torch.empty(heads, 2 * dh))
        nn.init.xavier_uniform_(self.a)
        self.out = nn.Linear(dh * heads, D_EMB)
    def forward(self, x, g):
        ei = g["edge_index"]; n = x.size(0); H, dh = self.heads, self.dh
        h = self.W(x).view(n, H, dh)
        src, dst = ei[0], ei[1]
        e = torch.cat([h[src], h[dst]], -1)
        al = torch.exp((e * self.a).sum(-1))
        den = torch.zeros(n, H, device=x.device).scatter_add_(0, dst.unsqueeze(-1).expand(-1, H), al)
        al = al / (den[dst] + 1e-9)
        o = torch.zeros(n, H, dh, device=x.device).scatter_add_(
            0, dst.view(-1, 1, 1).expand(-1, H, dh), al.unsqueeze(-1) * h[src])
        return self.out(o.reshape(n, H * dh).relu())

class MLPEncoder(nn.Module):  # no graph
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, D_HID), nn.ReLU(), nn.Linear(D_HID, D_EMB))
    def forward(self, x, g=None):
        return self.net(x)

class VGAEEncoder(nn.Module):  # variational graph AE
    def __init__(self, d_in):
        super().__init__()
        self.l1 = nn.Linear(d_in, D_HID)
        self.mu = nn.Linear(D_HID, D_EMB); self.lv = nn.Linear(D_HID, D_EMB)
    def forward(self, x, g):
        h = torch.sparse.mm(g["adj"], self.l1(x)).relu()
        mu = torch.sparse.mm(g["adj"], self.mu(h))
        lv = torch.sparse.mm(g["adj"], self.lv(h))
        if self.training:
            z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        else:
            z = mu
        kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(1).mean()
        return z, kl

class SCAttnEncoder(nn.Module):  # SpaMM-Net style: spatial query + confidence gate
    def __init__(self, d_in):
        super().__init__()
        self.q = nn.Linear(2, D_HID); self.k = nn.Linear(d_in, D_HID); self.v = nn.Linear(d_in, D_HID)
        self.gate = nn.Sequential(nn.Linear(d_in, D_HID), nn.Sigmoid())
        self.out = nn.Linear(D_HID, D_EMB)
    def forward(self, x, g):
        ei = g["edge_index"]; n = x.size(0)
        q = self.q(g["coords"]); k = self.k(x); v = self.v(x)
        gate = self.gate(x)
        src, dst = ei[0], ei[1]
        s = (q[dst] * k[src]).sum(-1) / math.sqrt(q.size(-1))
        al = torch.exp(s - s.max())
        den = torch.zeros(n, device=x.device).scatter_add_(0, dst, al)
        al = al / (den[dst] + 1e-9)
        agg = torch.zeros_like(v).scatter_add_(0, dst.unsqueeze(-1).expand_as(v[src]), al.unsqueeze(-1) * v[src])
        return self.out((agg * gate).relu())

ENCODERS = {"sage": SAGEEncoder, "gcn": GCNEncoder, "gat": GATEncoder,
            "mlp": MLPEncoder, "vgae": VGAEEncoder, "scattn": SCAttnEncoder}
'''))

cells.append(code('''# ---------- fusion: (z_rna, z_aux) -> fused 32-D ----------
class ConcatFusion(nn.Module):  # protected concatenation
    def __init__(self):
        super().__init__(); self.l = nn.Linear(2 * D_EMB, D_EMB)
    def forward(self, zr, za):
        return self.l(F.relu(torch.cat([zr, za], 1)))

class BahdanauFusion(nn.Module):  # SpatialGlue between-modality attention
    def __init__(self):
        super().__init__()
        self.Wr = nn.Linear(D_EMB, D_EMB); self.Wa = nn.Linear(D_EMB, D_EMB)
        self.v = nn.Linear(D_EMB, 1); self.out = nn.Linear(2 * D_EMB, D_EMB)
    def forward(self, zr, za):
        st = torch.stack([self.Wr(zr), self.Wa(za)], 1)
        a = F.softmax(self.v(torch.tanh(st)).squeeze(-1), 1)
        ctx = (a.unsqueeze(-1) * st).sum(1)
        return self.out(F.relu(torch.cat([ctx, zr + za], 1)))

class BMHAFusion(nn.Module):  # bidirectional multi-head attention
    def __init__(self, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(D_EMB, heads, batch_first=True)
        self.out = nn.Linear(2 * D_EMB, D_EMB)
    def forward(self, zr, za):
        seq = torch.stack([zr, za], 1)
        o, _ = self.attn(seq, seq, seq)
        return self.out(F.relu(o.reshape(o.size(0), -1)))

class ReMoEFusion(nn.Module):  # ReLU routing, no forced softmax competition
    def __init__(self):
        super().__init__()
        self.e_r = nn.Sequential(nn.Linear(D_EMB, D_EMB), nn.ReLU(), nn.Linear(D_EMB, D_EMB))
        self.e_a = nn.Sequential(nn.Linear(D_EMB, D_EMB), nn.ReLU(), nn.Linear(D_EMB, D_EMB))
        self.router = nn.Linear(2 * D_EMB, 2); self.out = nn.Linear(D_EMB, D_EMB)
    def forward(self, zr, za):
        w = F.relu(self.router(torch.cat([zr, za], 1)))
        w = w / (w.sum(1, keepdim=True) + 1e-9)
        return self.out(F.relu(w[:, [0]] * self.e_r(zr) + w[:, [1]] * self.e_a(za)))

class SoftMoEFusion(nn.Module):  # soft dispatch / combine
    def __init__(self, n_exp=4):
        super().__init__()
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(2 * D_EMB, D_EMB), nn.ReLU(), nn.Linear(D_EMB, D_EMB)) for _ in range(n_exp)])
        self.dispatch = nn.Linear(2 * D_EMB, n_exp)
    def forward(self, zr, za):
        x = torch.cat([zr, za], 1)
        w = F.softmax(self.dispatch(x), 1)
        return sum(wi.unsqueeze(1) * e(x) for wi, e in zip(w.t(), self.experts))

class MLPHierFusion(nn.Module):  # 2-stage hierarchical MLP
    def __init__(self):
        super().__init__()
        self.s1 = nn.Sequential(nn.Linear(2 * D_EMB, D_EMB), nn.ReLU())
        self.s2 = nn.Sequential(nn.Linear(2 * D_EMB, D_EMB), nn.ReLU(), nn.Linear(D_EMB, D_EMB))
    def forward(self, zr, za):
        h1 = self.s1(torch.cat([zr, za], 1))
        return self.s2(torch.cat([h1, zr + za], 1))

class UAFFusion(nn.Module):  # per-dim uncertainty-weighted fusion
    def __init__(self):
        super().__init__()
        self.lr = nn.Linear(D_EMB, D_EMB); self.la = nn.Linear(D_EMB, D_EMB)
        self.out = nn.Linear(D_EMB, D_EMB)
    def forward(self, zr, za):
        wr = torch.sigmoid(-self.lr(zr)); wa = torch.sigmoid(-self.la(za))
        return self.out(F.relu((wr * zr + wa * za) / (wr + wa + 1e-9)))

FUSIONS = {"concat": ConcatFusion, "bahdanau": BahdanauFusion, "bmha": BMHAFusion,
           "remoe": ReMoEFusion, "softmoe": SoftMoEFusion, "mlphier": MLPHierFusion, "uaf": UAFFusion}
'''))

cells.append(code('''# ---------- alignment (auxiliary, label-free) ----------
def sinkhorn(C, eps=0.1, n_iter=30):
    K = torch.exp(-C / eps)
    n, m = C.shape
    u = torch.full((n,), 1.0 / n, device=C.device); v = torch.full((m,), 1.0 / m, device=C.device)
    for _ in range(n_iter):
        u = (1.0 / n) / (K @ v + 1e-9)
        v = (1.0 / m) / (K.t() @ u + 1e-9)
    return torch.diag(u) @ K @ torch.diag(v)

def fugw_loss(zr, za, m=256, alpha=0.5):
    # unbalanced fused Gromov-Wasserstein, screening approx on subsample
    n = zr.size(0); idx = torch.randperm(n, device=zr.device)[:min(m, n)]
    a, b = zr[idx], za[idx]
    Dr, Da, Cf = torch.cdist(a, a), torch.cdist(b, b), torch.cdist(a, b)
    gw = (Dr ** 2).mean() + (Da ** 2).mean() - 2 * Dr.mean() * Da.mean()
    return alpha * gw + (1 - alpha) * Cf.mean()

def scot_loss(zr, za, m=512, eps=0.1):
    n = zr.size(0); idx = torch.randperm(n, device=zr.device)[:min(m, n)]
    C = torch.cdist(zr[idx], za[idx])
    P = sinkhorn(C, eps)
    return (P * C).sum()

def cca_loss(zr, za):
    zr = zr - zr.mean(0); za = za - za.mean(0)
    cr = (zr.t() @ za) / zr.size(0)
    return -torch.trace(cr) / zr.size(1)

def pearson_split_loss(zr, za, d_shared=16):
    # SpaMOAL-style: reward cross-modal shared correlation, penalize shared/private leakage
    sr, pr = zr[:, :d_shared], zr[:, d_shared:]
    sa, pa = za[:, :d_shared], za[:, d_shared:]
    def mc(a, b):
        a = a - a.mean(0); b = b - b.mean(0)
        return ((a * b).mean(0) / (a.std(0) * b.std(0) + 1e-9)).abs().mean()
    return -mc(sr, sa) + mc(sr, pr) + mc(sa, pa)

ALIGNS = {"fugw": fugw_loss, "scot": scot_loss, "cca": cca_loss, "pearson": pearson_split_loss}
'''))

cells.append(code('''# ---------- losses (label-free) ----------
def recon_loss(z, xr, xa, dec_r, dec_a):
    return F.mse_loss(dec_r(z), xr) + F.mse_loss(dec_a(z), xa)

def infonce_loss(z1, z2, tau=0.7):
    z1 = F.normalize(z1, 1); z2 = F.normalize(z2, 1)
    return F.cross_entropy(z1 @ z2.t() / tau, torch.arange(z1.size(0), device=z1.device))

def siglip_loss(z1, z2, tau=0.5, bias=-10.0):
    z1 = F.normalize(z1, 1); z2 = F.normalize(z2, 1)
    sim = z1 @ z2.t() / tau + bias
    tgt = (2 * torch.eye(z1.size(0), device=z1.device) - 1 + 1) / 2
    return F.binary_cross_entropy_with_logits(sim, tgt)

def gatcl_loss(zr, za, tau=0.5):
    return infonce_loss(zr, za, tau) + 0.5 * infonce_loss(zr, zr, tau) + 0.5 * infonce_loss(za, za, tau)

def spatial_bce_loss(z, edge_index):
    zn = F.normalize(z, 1)
    src, dst = edge_index
    pos = (zn[src] * zn[dst]).sum(-1)
    m = src.size(0); n = zn.size(0)
    ni = torch.randint(0, n, (m,), device=z.device); nj = torch.randint(0, n, (m,), device=z.device)
    neg = (zn[ni] * zn[nj]).sum(-1)
    logits = torch.cat([pos, neg]) * 5.0
    labels = torch.cat([torch.ones(m, device=z.device), torch.zeros(m, device=z.device)])
    return F.binary_cross_entropy_with_logits(logits, labels)

def vcr_loss(z, gamma=1.0):
    zc = z - z.mean(0)
    var = F.relu(gamma - zc.std(0)).mean()
    cov = (zc.t() @ zc) / (z.size(0) - 1)
    off = cov - torch.diag(torch.diag(cov))
    return var + (off ** 2).sum() / z.size(1)

def hsic_loss(a, b):
    n = a.size(0)
    Ka, Kb = a @ a.t(), b @ b.t()
    H = torch.eye(n, device=a.device) - 1.0 / n
    return torch.trace(Ka @ H @ Kb @ H) / (n - 1) ** 2

class SCIGMATau(nn.Module):  # per-spot learned temperature (Nat Genet 2026)
    def __init__(self, n, tau0=0.7):
        super().__init__()
        self.log_tau = nn.Parameter(torch.full((n,), math.log(tau0)))
    def loss(self, z1, z2):
        z1 = F.normalize(z1, 1); z2 = F.normalize(z2, 1)
        tau = self.log_tau.exp().clamp(0.05, 5.0)
        sim = (z1 @ z2.t()) / tau.unsqueeze(1)
        return F.cross_entropy(sim, torch.arange(z1.size(0), device=z1.device))

def rcmcl_inputs(xr, xa, p=0.3):
    # RCMCL modality dropout: randomly zero one modality's input during training
    if random.random() < p:
        if random.random() < 0.5:
            return torch.zeros_like(xr), xa
        return xr, torch.zeros_like(xa)
    return xr, xa
'''))

cells.append(code('''# ---------- model ----------
class FusedModel(nn.Module):
    def __init__(self, d_rna, d_aux, n_spots, cfg):
        super().__init__()
        self.cfg = cfg
        self.is_vgae = cfg["enc"] == "vgae"
        self.enc_r = ENCODERS[cfg["enc"]](d_rna)
        self.enc_a = ENCODERS[cfg["enc"]](d_aux)
        self.fusion = FUSIONS[cfg["fusion"]]()
        self.dec_r = nn.Linear(D_EMB, d_rna)
        self.dec_a = nn.Linear(D_EMB, d_aux)
        self.tau_mod = SCIGMATau(n_spots) if "scigmatau" in cfg.get("aug", []) else None
        self.ladj = LearnedAdj(d_rna, n_spots) if cfg["graph"] == "stague" else None

    def _enc(self, enc, x, g):
        if self.is_vgae:
            z, kl = enc(x, g)
            return z, kl
        return enc(x, g), torch.tensor(0.0, device=x.device)

    def forward(self, xr, xa, g, device, n):
        if self.ladj is not None:
            g = dict(g); g["adj"] = self.ladj(xr, g["edge_index"], n, device)
            g["adj_nl"] = g["adj"]
        zr, kl_r = self._enc(self.enc_r, xr, g)
        za, kl_a = self._enc(self.enc_a, xa, g)
        z = self.fusion(zr, za)
        return z, zr, za, kl_r + kl_a

    def loss(self, z, zr, za, kl, xr, xa, g):
        cfg = self.cfg
        L = torch.tensor(0.0, device=z.device)
        ln = cfg["loss"]
        if ln == "recon":
            L = L + recon_loss(z, xr, xa, self.dec_r, self.dec_a)
        elif ln == "infonce":
            L = L + (self.tau_mod.loss(zr, za) if self.tau_mod else infonce_loss(zr, za))
        elif ln == "siglip2":
            L = L + siglip_loss(zr, za)
        elif ln == "gatcl":
            L = L + gatcl_loss(zr, za)
        elif ln == "spatialbce":
            L = L + spatial_bce_loss(z, g["edge_index"])
        if self.is_vgae:
            L = L + 0.1 * kl
        al = cfg.get("align")
        if al:
            L = L + 0.5 * ALIGNS[al](zr, za)
        for a in cfg.get("aug", []):
            if a == "vcr":
                L = L + 1.0 * vcr_loss(z)
            elif a == "hsic":
                L = L + 0.5 * hsic_loss(zr, za)
        return L

BASELINE = dict(graph="spatial15", enc="sage", fusion="concat", loss="recon", align=None, aug=[])

# 26 combinations: 1 baseline + 25 single-module swaps (one-module-at-a-time)
COMBINATIONS = [
    ("BASELINE", {}),
    ("ENC-gcn-spectral", dict(enc="gcn")),
    ("ENC-gat-attention", dict(enc="gat")),
    ("ENC-mlp-nograph", dict(enc="mlp")),
    ("ENC-vgae", dict(enc="vgae")),
    ("ENC-scattn-spatial", dict(enc="scattn")),
    ("GRAPH-dual", dict(graph="dual")),
    ("GRAPH-stague-learned", dict(graph="stague")),
    ("FUS-bahdanau-crossmodal", dict(fusion="bahdanau")),
    ("FUS-bmha", dict(fusion="bmha")),
    ("FUS-remoe", dict(fusion="remoe")),
    ("FUS-softmoe", dict(fusion="softmoe")),
    ("FUS-mlp-hierarchical", dict(fusion="mlphier")),
    ("FUS-uaf-gaussian", dict(fusion="uaf")),
    ("ALIGN-fugw-structural", dict(align="fugw")),
    ("ALIGN-scot-sinkhorn", dict(align="scot")),
    ("ALIGN-cca-shared", dict(align="cca")),
    ("ALIGN-pearson-split", dict(align="pearson")),
    ("LOSS-infonce-hightau", dict(loss="infonce")),
    ("LOSS-siglip2", dict(loss="siglip2")),
    ("LOSS-gatcl", dict(loss="gatcl")),
    ("LOSS-spatial-bce", dict(loss="spatialbce")),
    ("LOSS-recon+vcr", dict(aug=["vcr"])),
    ("LOSS-recon+hsic", dict(aug=["hsic"])),
    ("LOSS-infonce+scigma-tau", dict(loss="infonce", aug=["scigmatau"])),
    ("LOSS-recon+rcmcl-dropout", dict(aug=["rcmcl"])),
]
print(f"{len(COMBINATIONS)} combinations registered")
'''))

cells.append(code('''# ---------- embedding generation test runner ----------
OUT = "/kaggle/working/sweep_out"
os.makedirs(OUT, exist_ok=True)
REG = os.path.join(OUT, "registry.jsonl")

def done_set():
    s = set()
    if os.path.exists(REG):
        for line in open(REG):
            r = json.loads(line)
            s.add((r["combo"], r["dataset"], r["seed"]))
    return s

def train_and_embed(combo_name, cfg, data, seed, k, epochs=120, lr=1e-3):
    set_seed(seed)
    n = data["Xr"].shape[0]
    xr = torch.tensor(data["Xr"], device=DEVICE)
    xa = torch.tensor(data["Xa"], device=DEVICE)
    g = build_graph(cfg["graph"], data["coords"], data["Xr"], DEVICE)
    model = FusedModel(data["Xr"].shape[1], data["Xa"].shape[1], n, cfg).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        if "rcmcl" in cfg.get("aug", []):
            xr_in, xa_in = rcmcl_inputs(xr, xa)
        else:
            xr_in, xa_in = xr, xa
        opt.zero_grad()
        z, zr, za, kl = model(xr_in, xa_in, g, DEVICE, n)
        loss = model.loss(z, zr, za, kl, xr, xa, g)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        z, _, _, _ = model(xr, xa, g, DEVICE, n)
    return z.detach().cpu().numpy()

def eval_embedding(Z, k, seed):
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
    return km.labels_, float(silhouette_score(Z, km.labels_))

def seed_stability(all_labels):
    # mean over off-diagonal seed pairs (no self-comparison)
    vals = []
    seeds = sorted(all_labels)
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            vals.append(adjusted_rand_score(all_labels[seeds[i]], all_labels[seeds[j]]))
    return float(np.mean(vals))

results = []
t0 = time.time()
for combo_name, override in COMBINATIONS:
    cfg = dict(BASELINE); cfg.update(override)
    cfg["aug"] = list(override.get("aug", []))
    for ds_id in DATASETS:
        data = load_dataset(ds_id)
        k = data["k"]
        labels_per_seed, sils = {}, {}
        for seed in SEEDS:
            if (combo_name, ds_id, seed) in done_set():
                print(f"skip done {combo_name}/{ds_id}/{seed}", flush=True)
                continue
            Z = train_and_embed(combo_name, cfg, data, seed, k)
            labs, sil = eval_embedding(Z, k, seed)
            labels_per_seed[seed], sils[seed] = labs, sil
            d = os.path.join(OUT, combo_name, ds_id, f"seed{seed}")
            os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, "fused_embedding.npy"), Z)
            np.save(os.path.join(d, "labels.npy"), labs)
            json.dump({"combo": combo_name, "dataset": ds_id, "seed": seed,
                       "silhouette": sil, "k": k, "quarantined": False},
                      open(os.path.join(d, "metrics.json"), "w"), indent=2)
            # quarantined post-hoc ARI (reporting only, never selection)
            ari = float(adjusted_rand_score(data["gt"], labs))
            json.dump({"ari": ari, "quarantined_reporting_only": True},
                      open(os.path.join(d, "posthoc.json"), "w"), indent=2)
            with open(REG, "a") as f:
                f.write(json.dumps({"combo": combo_name, "dataset": ds_id, "seed": seed,
                                    "silhouette": sil, "posthoc_ari": ari, "cfg": cfg}) + "\\n")
            print(f"done {combo_name}/{ds_id}/seed{seed} sil={sil:.3f} posthocARI={ari:.3f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        if labels_per_seed:
            stab = seed_stability(labels_per_seed)
            msil = float(np.mean(list(sils.values())))
            results.append((combo_name, ds_id, msil, stab))
            with open(REG, "a") as f:
                f.write(json.dumps({"combo": combo_name, "dataset": ds_id, "seed": "AGG",
                                    "mean_silhouette": msil, "seed_stability_ari": stab}) + "\\n")
    git_push([f"sweep_out/{combo_name}"], f"sweep: {combo_name} embeddings")

print("sweep complete")
'''))

cells.append(code('''# ---------- summary: rank fused embeddings per dataset (label-free) ----------
from collections import defaultdict
agg = defaultdict(list)
for combo_name, ds_id, msil, stab in results:
    agg[ds_id].append((combo_name, msil, stab))
for ds_id in sorted(agg):
    ranked = sorted(agg[ds_id], key=lambda r: (r[1], r[2]), reverse=True)
    print(f"\\n=== {ds_id} (by mean silhouette, then stability) ===")
    for i, (c, s, st) in enumerate(ranked, 1):
        print(f"#{i:<3} {c:28} sil={s:.3f} stab={st:.3f}")
json.dump([{"combo": c, "dataset": d, "mean_silhouette": s, "seed_stability": st}
           for c, d, s, st in results],
          open(os.path.join(OUT, "summary.json"), "w"), indent=2)
git_push(["sweep_out/summary.json", "sweep_out/registry.jsonl"], "sweep: summary + registry")
print("\\nAll fused embeddings saved under", OUT)
'''))

cells.append(md('''## Notes for the agent
- **Resume:** the runner skips `(combo, dataset, seed)` already in `registry.jsonl`. Safe to re-run after disconnects; split across sessions if needed.
- **Firewall:** ground-truth labels are read once at startup for the declared k and post-hoc ARI only. They never enter training, fusion, or selection.
- **Budgets:** 30 GPU-hours/week cap still binds; check quota before launching.
- Results land in `runs/sweep/<combo>/` after the GitHub push — wait, this notebook saves to `/kaggle/working/sweep_out/`. The push above stages `sweep_out/...`; rename the destination to `runs/sweep/` when wiring the final push, or keep `sweep_out/` and note the mapping in the stage report.
'''))

nb = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.10"}},
    "nbformat": 4, "nbformat_minor": 4,
}

os.makedirs("notebooks", exist_ok=True)
out = "notebooks/embedding_module_sweep.ipynb"
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out, "cells:", len(cells))

# validate: every code cell parses
import ast
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        ast.parse(c["source"])
print("all code cells parse OK")
