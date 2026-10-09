"""Generate notebooks/stage5_generalization.ipynb — Stage 5 generalization screen.

Base = Laplacian-H-HiRe analogue (SAGE + hierarchical MLP fusion + reconstruction
+ Laplacian Dirichlet spatial loss), then one-module-at-a-time swaps:
  S5N-BASE (recal+idec, current best), S5N-adapt-spatial (Moran's-I λ),
  S5N-adapt-modality (uncertainty weighting), S5N-shared-private
  (disentanglement), S5N-generalized (all three).

Pure torch (no pyg), Kaggle-ready, skip_done resume, GitHub push per combo
(clone-or-pull: fixed the stale-dir failure), firewall intact
(k declared once from DECLARED_K, post-hoc ARI quarantined in posthoc.json).

Run: python3 scripts/gen_stage5_nb.py
"""
import json, os

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

cells = []

cells.append(md('''# Stage 5 — Generalization: One Model for All Six Datasets

**Goal:** a single architecture that works well on *all six* datasets — not by tuning
per dataset, but by **adapting itself from label-free data statistics**.

**Why:** mouse brain (strong spatial structure) now reaches ARI 0.37–0.50, but human
lymph node (weaker spatial structure, ADT drowned by RNA) is stuck at ~0.20–0.22.
Fixed smoothing over-smooths lymph node; fixed fusion lets RNA dominate ADT.

**Three adaptive mechanisms (all label-free, all per-dataset automatic):**
1. **Adaptive smoothing** — Moran's I spatial autocorrelation measured per dataset;
   λ_spa scales with it (brain → more smoothing, lymph node → less).
2. **Adaptive modality balance** — per-modality uncertainty weighting (Kendall & Gal):
   the model up-weights whichever modality is harder to reconstruct, on its own.
3. **Shared–private disentanglement** — shared 16-D + private 8+8-D embeddings with a
   cosine-disentanglement loss, so modality-specific signal survives fusion.

**Matrix (5 combos):** `S5N-BASE` (recalibrated base + IDEC = current best, reference);
`S5N-adapt-spatial`; `S5N-adapt-modality`; `S5N-shared-private`;
`S5N-generalized` (all three mechanisms combined).

**RAUS v2 selection** (Borda over silhouette, seed-stability, reconstruction-balance)
+ over-clustering guard; post-hoc ARI quarantined to `posthoc.json` (reporting only).
Note (Stage-3 recal finding): silhouette can mislead under over-smoothing — the
RAUS-v3 criterion study (agent Stage 4) addresses selection; here we maximize the
underlying embedding quality on all six datasets.
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
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
import pandas as pd

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

print("imports ok")
'''))

cells.append(code('''# ---------- GitHub push (guarded; clone-or-pull fixes the stale-dir failure) ----------
REPO = "/kaggle/working/my_research"
REPO_URL = "https://github.com/rifatahsanpul0k/my_research.git"
def sh(*args):
    subprocess.run(list(args), check=True, cwd=REPO)
os.makedirs(REPO, exist_ok=True)
OUT = os.path.join(REPO, "runs", "s5-notebook")
os.makedirs(OUT, exist_ok=True)
try:
    from kaggle_secrets import UserSecretsClient
    import base64, shutil
    _sec = UserSecretsClient()
    _b64 = base64.b64encode(f"x-access-token:{_sec.get_secret('GITHUB_TOKEN')}".encode()).decode()
    _GIT = ["git", "-c", f"http.extraHeader=Authorization: Basic {_b64}"]
    if not os.path.isdir(os.path.join(REPO, ".git")):
        if os.listdir(REPO):
            # stale non-repo dir (the sweep failure mode) -> move aside, then clone
            bak = REPO + "_stalebak"
            if os.path.exists(bak):
                shutil.rmtree(bak)
            shutil.move(REPO, bak)
            os.makedirs(REPO, exist_ok=True)
        subprocess.run(_GIT + ["clone", REPO_URL, REPO], check=True)
        sh("git", "config", "user.name", "kaggle-runner")
        sh("git", "config", "user.email", "kaggle-runner@local")
    else:
        sh(*_GIT, "pull", "--rebase", "origin", "main")
    GIT_OK = True
    print("GitHub sync ready")
except Exception as e:
    GIT_OK = False
    _safe = str(e).replace(_b64, "***") if "_b64" in dir() else str(e)
    print("GitHub sync unavailable (local save only):", _safe)

def _scrub(s):
    # never print the PAT: it lives inside _GIT's auth header
    try:
        return str(s).replace(_b64, "***")
    except NameError:
        return str(s)

def git_push(paths, msg):
    if not GIT_OK:
        return
    try:        # CORRECT ORDER: commit local changes FIRST, then pull --rebase, then push.
        # (Pulling before committing fails with "unstaged changes" once the registry
        # is a tracked file — which broke every push after the first combo.)
        sh("git", "add", *paths)
        r = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                           text=True, cwd=REPO)
        if r.stdout.strip():
            sh("git", "commit", "-m", msg)
        sh(*_GIT, "pull", "--rebase", "origin", "main")
        sh(*_GIT, "push", "origin", "main")
        print("pushed:", msg)
    except Exception as e:
        print("push failed (continuing):", _scrub(e))
'''))

cells.append(code('''# ---------- data (identical preprocessing to the sweep; cached once) ----------
MARKER = "10x_human_lymph_node_A1"
def find_data_root():
    for base in ["/kaggle/input/spatial-multiomics-6datasets-private",
                 "/kaggle/working/spatial-multiomics-6datasets-private",
                 "/kaggle/working"]:
        if os.path.isdir(os.path.join(base, MARKER)):
            return base
    return None

DATA_ROOT = find_data_root()
if DATA_ROOT is None:
    from kaggle_secrets import UserSecretsClient
    _s = UserSecretsClient()
    kd = os.path.expanduser("~/.kaggle"); os.makedirs(kd, exist_ok=True)
    json.dump({"username": _s.get_secret("KAGGLE_USERNAME"), "key": _s.get_secret("KAGGLE_KEY")},
              open(os.path.join(kd, "kaggle.json"), "w"))
    os.chmod(os.path.join(kd, "kaggle.json"), 0o600)
    subprocess.run(["kaggle", "datasets", "download", "-d", "pulokpulok/spatial-multiomics-6datasets-private",
                    "-p", "/kaggle/working", "--unzip"], check=True)
    DATA_ROOT = find_data_root()
if DATA_ROOT is None:
    raise FileNotFoundError("dataset folders not found under /kaggle/working: "
                            + str(sorted(os.listdir("/kaggle/working"))))
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
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    sc.pp.highly_variable_genes(rna, n_top_genes=3000, flavor="seurat", inplace=True)
    Xh = rna[:, rna.var.highly_variable].X
    Xh = Xh.toarray() if hasattr(Xh, "toarray") else np.asarray(Xh)
    Xh = StandardScaler().fit_transform(Xh).astype(np.float32)          # HVG3000 recon target
    Xr = StandardScaler().fit_transform(PCA(n_components=64, random_state=0).fit_transform(Xh)).astype(np.float32)
    Xr30 = PCA(n_components=30, random_state=0).fit_transform(Xr).astype(np.float32)  # compressed target
    Xr30 = StandardScaler().fit_transform(Xr30).astype(np.float32)
    if spec["aux"] == "adt":
        aux = sc.read_h5ad(os.path.join(root, "adata_ADT.h5ad"))
        aux.var_names_make_unique()
        Xa = StandardScaler().fit_transform(clr(to_dense_small(aux.X))).astype(np.float32)
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
        Xa = StandardScaler().fit_transform(lsi[:, keep]).astype(np.float32)
    gt = pd.read_csv(os.path.join(root, spec["gt_file"]), index_col=0)
    k_declared = int(gt[spec["gt_col"]].nunique())   # declared ONCE (declared exception)
    print(f"{ds_id}: RNA {Xr.shape} AUX {Xa.shape} HVG {Xh.shape} N={Xr.shape[0]} k_declared={k_declared}", flush=True)
    return dict(Xr=Xr, Xa=Xa, Xh=Xh, Xr30=Xr30, coords=coords, k=k_declared,
                gt=np.asarray(gt[spec["gt_col"]]))
'''))

cells.append(code('''# ---------- graph: spatial kNN (pure torch) ----------
def knn_edge_index(coords, k=15):
    A = kneighbors_graph(coords, k, mode="connectivity", include_self=False)
    A = A.maximum(A.T)
    coo = A.tocoo()
    return torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)

def norm_adj_nl(edge_index, n, device):
    idx = edge_index.to(device)
    vals = torch.ones(idx.shape[1], device=device)
    deg = torch.zeros(n, device=device).scatter_add_(0, idx[0], vals)
    d = deg.pow(-0.5); d[torch.isinf(d)] = 0
    vals = d[idx[0]] * vals * d[idx[1]]
    return torch.sparse_coo_tensor(idx, vals, (n, n), device=device).coalesce()

def build_graph(coords, device):
    n = coords.shape[0]
    ei = knn_edge_index(coords, 15).to(device)
    return dict(edge_index=ei, adj_nl=norm_adj_nl(ei, n, device))
'''))

cells.append(code('''# ---------- encoders ----------
D_EMB, D_HID = 32, 128

class SAGEEncoder(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.l1 = nn.Linear(2 * d_in, D_HID); self.l2 = nn.Linear(2 * D_HID, D_EMB)
    def forward(self, x, g):
        agg = torch.sparse.mm(g["adj_nl"], x)
        h = self.l1(torch.cat([x, agg], 1)).relu()
        agg2 = torch.sparse.mm(g["adj_nl"], h)
        return self.l2(torch.cat([h, agg2], 1))

class GATEncoder(nn.Module):  # Velickovic et al. 2018: e_ij = LeakyReLU(a^T[Wh_i||Wh_j])
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
        al = torch.exp(F.leaky_relu((e * self.a).sum(-1), 0.2))  # LeakyReLU per original GAT
        den = torch.zeros(n, H, device=x.device).scatter_add_(0, dst.unsqueeze(-1).expand(-1, H), al)
        al = al / (den[dst] + 1e-9)
        o = torch.zeros(n, H, dh, device=x.device).scatter_add_(
            0, dst.view(-1, 1, 1).expand(-1, H, dh), al.unsqueeze(-1) * h[src])
        return self.out(o.reshape(n, H * dh).relu())

ENCODERS = {"sage": SAGEEncoder, "gat": GATEncoder}
'''))

cells.append(code('''# ---------- fusion ----------
class MLPHierFusion(nn.Module):  # 2-stage hierarchical MLP (Stage 2 winner)
    def __init__(self):
        super().__init__()
        self.s1 = nn.Sequential(nn.Linear(2 * D_EMB, D_EMB), nn.ReLU())
        self.s2 = nn.Sequential(nn.Linear(2 * D_EMB, D_EMB), nn.ReLU(), nn.Linear(D_EMB, D_EMB))
    def forward(self, zr, za):
        h1 = self.s1(torch.cat([zr, za], 1))
        return self.s2(torch.cat([h1, zr + za], 1))

class UAFFusion(nn.Module):  # per-dim uncertainty-weighted (sweep stability winner)
    def __init__(self):
        super().__init__()
        self.lr = nn.Linear(D_EMB, D_EMB); self.la = nn.Linear(D_EMB, D_EMB)
        self.out = nn.Linear(D_EMB, D_EMB)
    def forward(self, zr, za):
        wr = torch.sigmoid(-self.lr(zr)); wa = torch.sigmoid(-self.la(za))
        return self.out(F.relu((wr * zr + wa * za) / (wr + wa + 1e-9)))

FUSIONS = {"mlphier": MLPHierFusion, "uaf": UAFFusion}
'''))

cells.append(code('''# ---------- alignment: SCOT (sweep winner) ----------
def sinkhorn(C, eps=0.1, n_iter=30):
    K = torch.exp(-C / eps)
    n, m = C.shape
    u = torch.full((n,), 1.0 / n, device=C.device); v = torch.full((m,), 1.0 / m, device=C.device)
    for _ in range(n_iter):
        u = (1.0 / n) / (K @ v + 1e-9)
        v = (1.0 / m) / (K.t() @ u + 1e-9)
    return torch.diag(u) @ K @ torch.diag(v)

def scot_loss(zr, za, m=512, eps=0.1):
    n = zr.size(0); idx = torch.randperm(n, device=zr.device)[:min(m, n)]
    C = torch.cdist(zr[idx], za[idx])
    P = sinkhorn(C, eps)
    return (P * C).sum()
'''))

cells.append(code('''# ---------- losses ----------
def laplacian_loss(z, edge_index):
    # Dirichlet energy: 1/2 sum_{(i,j) in E} ||z_i - z_j||^2 / |E|  (Stage 2 winner)
    src, dst = edge_index
    return 0.5 * ((z[src] - z[dst]).pow(2).sum(1)).mean()

def r2_score(pred, true):
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean(0)) ** 2).sum()
    return torch.clamp(1 - ss_res / (ss_tot + 1e-9), 0.0, 1.0)

def recon_balance(r2_rna, r2_aux):
    # documented REPORT-s2 metric: min/max explained-variance ratio in (0,1]
    return float((min(r2_rna, r2_aux) + 1e-4) / (max(r2_rna, r2_aux) + 1e-4))
'''))

cells.append(code('''# ---------- learnable clustering heads: DEC / IDEC ----------
def student_q(z, mu, alpha=1.0):
    d2 = torch.cdist(z, mu).pow(2)
    q = (1 + d2 / alpha).pow(-(alpha + 1) / 2)
    return q / q.sum(1, keepdim=True)

def target_p(q):
    f = q.sum(0, keepdim=True)
    p = (q.pow(2) / f)
    return p / p.sum(1, keepdim=True)

def dec_refine(Z, k, seed, iters=100, lr=1e-2):
    """DEC on a FROZEN embedding: learn centers mu via KL(P||Q). Returns labels."""
    set_seed(seed)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
    mu = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=DEVICE, requires_grad=True)
    zt = torch.tensor(Z, dtype=torch.float32, device=DEVICE)
    opt = torch.optim.Adam([mu], lr=lr)
    for _ in range(iters):
        q = student_q(zt, mu)
        p = target_p(q).detach()
        loss = F.kl_div(q.log(), p, reduction="batchmean")
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        q = student_q(zt, mu)
    return q.argmax(1).cpu().numpy(), km.labels_
'''))

cells.append(code('''# ---------- model ----------
LAM_SPA, LAM_ALIGN, LAM_KL = 2.0, 0.5, 0.1   # LAM_SPA overridable per-combo; Stage 5 base uses 0.1
LAM_DIS = 0.1   # shared-private disentanglement weight (base; adaptive version scales it)
MORAN = {}      # ds_id -> Moran's I (label-free spatial autocorrelation), filled in precompute
XMODAL = {}     # ds_id -> cross-modal kNN agreement in [0,1], filled in precompute

def cross_modal_agreement(Xr, Xa, k=15):
    # Label-free: how much do RNA and aux agree on local neighborhoods?
    # High agreement = redundant modalities (don't force a split); low = complementary (split helps).
    from sklearn.neighbors import NearestNeighbors
    ir = NearestNeighbors(n_neighbors=k + 1).fit(Xr).kneighbors(Xr, return_distance=False)[:, 1:]
    ia = NearestNeighbors(n_neighbors=k + 1).fit(Xa).kneighbors(Xa, return_distance=False)[:, 1:]
    ov = [len(set(ir[i]) & set(ia[i])) / max(len(set(ir[i]) | set(ia[i])), 1) for i in range(len(Xr))]
    return float(np.mean(ov))

def resolve_lam_dis(cfg, data):
    # Explicit override wins (for the sweep); then adaptive; then fixed default.
    # Redundant modalities (high agreement) -> keep them together; complementary -> split.
    if cfg.get("lam_dis") is not None:
        return cfg["lam_dis"]
    if cfg.get("adapt_disentangle") and XMODAL:
        A = XMODAL[data["_ds"]]
        return 0.2 * (1.0 - A)
    return LAM_DIS

def moran_I(x, edge_index, n):
    # Moran's I for one feature vector (numpy). Label-free spatial autocorrelation in [-1, 1].
    x = np.asarray(x, dtype=float); xb = x - x.mean()
    denom = (xb ** 2).sum()
    if denom == 0:
        return 0.0
    ei = edge_index.cpu().numpy()
    num = (xb[ei[0]] * xb[ei[1]]).sum()
    return float(n / ei.shape[1] * num / denom)

def resolve_lam_spa(cfg, data):
    # Adaptive smoothing: λ_spa scales with the dataset's Moran's I relative to the
    # cross-dataset mean, anchored at the validated recalibrated value. Label-free.
    base = cfg.get("lam_spa", 0.1)
    if cfg.get("adapt_spatial") and MORAN:
        Ibar = sum(MORAN.values()) / len(MORAN)
        ratio = MORAN[data["_ds"]] / Ibar if Ibar else 1.0
        return base * min(max(ratio, 0.25), 4.0)
    return base

class S3Model(nn.Module):
    def __init__(self, d_rna, d_aux, cfg, d_recon_rna):
        super().__init__()
        self.cfg = cfg
        self.enc_r = ENCODERS[cfg.get("enc", "sage")](d_rna)
        self.enc_a = ENCODERS[cfg.get("enc", "sage")](d_aux)
        self.fusion = FUSIONS[cfg.get("fusion", "mlphier")]()
        self.dec_r = nn.Linear(D_EMB, d_recon_rna)
        self.dec_a = nn.Linear(D_EMB, d_aux)
        # learnable per-modality noise (Kendall & Gal uncertainty weighting); label-free
        self.logvar_r = nn.Parameter(torch.zeros(()))
        self.logvar_a = nn.Parameter(torch.zeros(()))

    def forward(self, xr, xa, g):
        zr = self.enc_r(xr, g); za = self.enc_a(xa, g)
        z = self.fusion(zr, za)
        return z, zr, za

    def recon_loss(self, z, tr, ta):
        lr = F.mse_loss(self.dec_r(z), tr); la = F.mse_loss(self.dec_a(z), ta)
        if self.cfg.get("adapt_modality"):
            return 0.5 * torch.exp(-self.logvar_r) * lr + 0.5 * self.logvar_r \
                 + 0.5 * torch.exp(-self.logvar_a) * la + 0.5 * self.logvar_a
        return lr + la

    def loss(self, out, tr, ta, g, lam_spa, lam_dis):
        z, zr, za = out
        L = self.recon_loss(z, tr, ta)
        if self.cfg.get("spatial") == "laplacian":
            L = L + lam_spa * laplacian_loss(z, g["edge_index"])
        if self.cfg.get("align") == "scot":
            L = L + LAM_ALIGN * scot_loss(zr, za)
        return L

class S5SPModel(nn.Module):
    # Shared-private disentanglement: shared 16-D + private 8+8-D = 32-D. Label-free.
    # Shared branch sees concatenated modalities; private branches see one each.
    def __init__(self, d_rna, d_aux, cfg, d_recon_rna):
        super().__init__()
        self.cfg = cfg
        self.enc_s = nn.Sequential(nn.Linear(d_rna + d_aux, 64), nn.ReLU(), nn.Linear(64, 16))
        self.enc_pr = nn.Sequential(nn.Linear(d_rna, 32), nn.ReLU(), nn.Linear(32, 8))
        self.enc_pa = nn.Sequential(nn.Linear(d_aux, 32), nn.ReLU(), nn.Linear(32, 8))
        self.dec_r = nn.Linear(32, d_recon_rna)
        self.dec_a = nn.Linear(32, d_aux)
        self.logvar_r = nn.Parameter(torch.zeros(()))
        self.logvar_a = nn.Parameter(torch.zeros(()))

    def forward(self, xr, xa, g):
        xs = torch.cat([xr, xa], dim=1)
        zs = self.enc_s(xs); zpr = self.enc_pr(xr); zpa = self.enc_pa(xa)
        z = torch.cat([zs, zpr, zpa], dim=1)
        return z, zs, zpr, zpa

    def loss(self, out, tr, ta, g, lam_spa, lam_dis):
        z, zs, zpr, zpa = out
        lr = F.mse_loss(self.dec_r(z), tr); la = F.mse_loss(self.dec_a(z), ta)
        if self.cfg.get("adapt_modality"):
            L = 0.5 * torch.exp(-self.logvar_r) * lr + 0.5 * self.logvar_r \
              + 0.5 * torch.exp(-self.logvar_a) * la + 0.5 * self.logvar_a
        else:
            L = lr + la
        # disentangle: minimize squared cosine similarity between shared (16-D)
        # and concatenated private (8+8-D); keeps modality-specific info out of shared
        def cos2(a, b):
            c = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)
            return (c ** 2).mean()
        zpriv = torch.cat([zpr, zpa], dim=1)
        L = L + lam_dis * cos2(zs, zpriv)
        if self.cfg.get("spatial") == "laplacian":
            L = L + lam_spa * laplacian_loss(z, g["edge_index"])
        return L

class SAGELayer(nn.Module):
    # single GraphSAGE-mean convolution (Hamilton et al. 2017), pure torch
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(2 * d_in, d_out)
    def forward(self, x, g):
        agg = torch.sparse.mm(g["adj_nl"], x)
        return self.lin(torch.cat([x, agg], 1)).relu()

class S6OneModel(nn.Module):
    # THE ONE MODEL: Gated Dual-Path Shared-Private.
    # Per modality: SAGE graph path + MLP feature path, blended by a LEARNED per-spot
    # sigmoid gate (SpaHDSRL-style). The gate learns unsupervised whether the spatial
    # graph helps (brain: gate->1) or hurts (lymph node: gate->0). Shared-private
    # split on the blended embeddings. One architecture, one config, zero switches.
    def __init__(self, d_rna, d_aux, cfg, d_recon_rna):
        super().__init__()
        self.cfg = cfg
        self.sage_r = SAGELayer(d_rna, 32)
        self.mlp_r = nn.Sequential(nn.Linear(d_rna, 32), nn.ReLU())
        self.sage_a = SAGELayer(d_aux, 32)
        self.mlp_a = nn.Sequential(nn.Linear(d_aux, 32), nn.ReLU())
        self.gate_r = nn.Linear(32, 1)
        self.gate_a = nn.Linear(32, 1)
        self.lin_s = nn.Linear(64, 16)
        self.lin_pr = nn.Linear(32, 8)
        self.lin_pa = nn.Linear(32, 8)
        self.dec_r = nn.Linear(32, d_recon_rna)
        self.dec_a = nn.Linear(32, d_aux)

    def forward(self, xr, xa, g):
        hr_g = self.sage_r(xr, g); hr_f = self.mlp_r(xr)
        ha_g = self.sage_a(xa, g); ha_f = self.mlp_a(xa)
        ar = torch.sigmoid(self.gate_r(hr_g))   # per-spot, learned
        aa = torch.sigmoid(self.gate_a(ha_g))
        hr = ar * hr_g + (1 - ar) * hr_f
        ha = aa * ha_g + (1 - aa) * ha_f
        zs = self.lin_s(torch.cat([hr, ha], dim=1))
        zpr = self.lin_pr(hr); zpa = self.lin_pa(ha)
        z = torch.cat([zs, zpr, zpa], dim=1)
        return z, zs, zpr, zpa, ar, aa

    def loss(self, out, tr, ta, g, lam_spa, lam_dis):
        z, zs, zpr, zpa, ar, aa = out
        lr = F.mse_loss(self.dec_r(z), tr); la = F.mse_loss(self.dec_a(z), ta)
        L = lr + la
        def cos2(a, b):
            c = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)
            return (c ** 2).mean()
        zpriv = torch.cat([zpr, zpa], dim=1)
        L = L + lam_dis * cos2(zs, zpriv)
        # No Laplacian loss: the gate IS the spatial mechanism (learned, per-spot).
        return L

class S5SPSageModel(nn.Module):
    # Shared-private disentanglement WITH graph convolutions (clean test: the Stage-5
    # SP model used plain MLPs, confounding disentanglement with encoder change).
    # Shared 16-D (SAGE on concatenated modalities) + private 8+8-D (SAGE per modality).
    def __init__(self, d_rna, d_aux, cfg, d_recon_rna):
        super().__init__()
        self.cfg = cfg
        self.sage_s = SAGELayer(d_rna + d_aux, 32); self.lin_s = nn.Linear(32, 16)
        self.sage_pr = SAGELayer(d_rna, 16);        self.lin_pr = nn.Linear(16, 8)
        self.sage_pa = SAGELayer(d_aux, 16);        self.lin_pa = nn.Linear(16, 8)
        self.dec_r = nn.Linear(32, d_recon_rna)
        self.dec_a = nn.Linear(32, d_aux)
        self.logvar_r = nn.Parameter(torch.zeros(()))
        self.logvar_a = nn.Parameter(torch.zeros(()))

    def forward(self, xr, xa, g):
        zs = self.lin_s(self.sage_s(torch.cat([xr, xa], dim=1), g))
        zpr = self.lin_pr(self.sage_pr(xr, g))
        zpa = self.lin_pa(self.sage_pa(xa, g))
        z = torch.cat([zs, zpr, zpa], dim=1)
        return z, zs, zpr, zpa

    def loss(self, out, tr, ta, g, lam_spa, lam_dis):
        z, zs, zpr, zpa = out
        lr = F.mse_loss(self.dec_r(z), tr); la = F.mse_loss(self.dec_a(z), ta)
        if self.cfg.get("adapt_modality"):
            L = 0.5 * torch.exp(-self.logvar_r) * lr + 0.5 * self.logvar_r \
              + 0.5 * torch.exp(-self.logvar_a) * la + 0.5 * self.logvar_a
        else:
            L = lr + la
        if self.cfg.get("disentangle", True):
            def cos2(a, b):
                c = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)
                return (c ** 2).mean()
            zpriv = torch.cat([zpr, zpa], dim=1)
            L = L + lam_dis * cos2(zs, zpriv)
        if self.cfg.get("spatial") == "laplacian":
            L = L + lam_spa * laplacian_loss(z, g["edge_index"])
        return L

def recon_target(cfg, data):
    # recon_target: "pca64" (base, reconstruct preprocessed inputs),
    #             "pca30" (compressed), "hvg3000" (raw-ish, Stage-2 H-HiRe style)
    rt = cfg.get("recon_target", "pca64")
    if rt == "pca30":
        return data["Xr30"], data["Xr30"].shape[1]
    if rt == "hvg3000":
        return data["Xh"], data["Xh"].shape[1]
    return data["Xr"], data["Xr"].shape[1]

BASE_CFG = dict(enc="sage", fusion="mlphier", spatial="laplacian",
                align=None, recon_target="pca64", head="idec", lam_spa=0.1)

# Stage 5 matrix: current best (recal+idec) + one adaptive mechanism at a time, then combined.
# All adaptation is label-free and per-dataset automatic — one architecture for all six.
COMBINATIONS = [
    ("S5N-BASE",            dict()),
    ("S5N-adapt-spatial",   dict(adapt_spatial=True)),
    ("S5N-adapt-modality",  dict(adapt_modality=True)),
    ("S5N-shared-private",  dict(model="shared_private")),
    ("S5N-generalized",     dict(adapt_spatial=True, adapt_modality=True, model="shared_private")),
    ("S5N-SP-sage",         dict(model="shared_private_sage")),
    ("S5N-SP-sage-nodis",   dict(model="shared_private_sage", disentangle=False)),
    ("S5N-SP-sage-adapt",   dict(model="shared_private_sage", adapt_disentangle=True)),
    ("S5N-dis005",          dict(model="shared_private_sage", lam_dis=0.05)),
    ("S5N-dis04",           dict(model="shared_private_sage", lam_dis=0.4)),
    ("S5N-nograph",         dict(model="shared_private_sage", spatial="none")),
    ("S6-ONE",              dict(model="one_model", lam_dis=0.15)),
]
print(f"{len(COMBINATIONS)} combinations registered")
'''))

cells.append(code('''# ---------- runner ----------
REG = os.path.join(OUT, "registry.jsonl")

def done_set():
    s = set()
    if os.path.exists(REG):
        for line in open(REG):
            r = json.loads(line)
            if r.get("seed") != "AGG":
                s.add((r["combo"], r["dataset"], r["seed"]))
    return s

def train_embed(cfg, data, g, seed, epochs=120, lr=1e-3, kl_head=False):
    set_seed(seed)
    global D_EMB
    D_EMB = cfg.get("d_emb", 32)   # embedding capacity is combo-configurable
    n = data["Xr"].shape[0]
    xr = torch.tensor(data["Xr"], device=DEVICE)
    xa = torch.tensor(data["Xa"], device=DEVICE)
    tr_np, d_rr = recon_target(cfg, data)
    tr = torch.tensor(tr_np, device=DEVICE)
    ta = xa
    lam_spa = resolve_lam_spa(cfg, data)   # fixed or Moran's-I adaptive
    lam_dis = resolve_lam_dis(cfg, data)   # fixed or cross-modal-agreement adaptive
    ModelCls = {"shared_private": S5SPModel,
                "shared_private_sage": S5SPSageModel,
                "one_model": S6OneModel}.get(cfg.get("model"), S3Model)
    model = ModelCls(data["Xr"].shape[1], data["Xa"].shape[1], cfg, d_rr).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mu = None
    k = data["k"]
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        out = model(xr, xa, g)
        z = out[0]
        loss = model.loss(out, tr, ta, g, lam_spa, lam_dis)
        if kl_head:  # IDEC: joint KL + recon
            if ep % 25 == 0 or mu is None:
                with torch.no_grad():
                    ze = z.detach().cpu().numpy()
                    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(ze)
                    mu = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=DEVICE)
            q = student_q(z, mu); p = target_p(q).detach()
            loss = loss + LAM_KL * F.kl_div(q.log(), p, reduction="batchmean")
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        out = model(xr, xa, g)
        z = out[0]
        r2r = float(r2_score(model.dec_r(z), tr)); r2a = float(r2_score(model.dec_a(z), ta))
        gate = None
        if cfg.get("model") == "one_model" and len(out) >= 6:
            gate = {"rna": float(out[4].mean()), "aux": float(out[5].mean())}
    return z.detach().cpu().numpy(), r2r, r2a, (mu.detach() if mu is not None else None), lam_spa, gate

def run_combo(combo_name, cfg, data, seed):
    t0 = time.time()
    g = build_graph(data["coords"], DEVICE)
    k = data["k"]
    head = cfg.get("head", "kmeans")
    Z, r2r, r2a, mu, lam_used, gate = train_embed(cfg, data, g, seed, kl_head=(head == "idec"))
    km_labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z).labels_
    sil = float(silhouette_score(Z, km_labels))
    recbal = recon_balance(r2r, r2a)
    out_labels, extra = km_labels, {}
    if head == "dec":
        out_labels, km0 = dec_refine(Z, k, seed)
        extra = {"agree_ari_dec_vs_kmeans": float(adjusted_rand_score(km0, out_labels))}
    elif head == "idec" and mu is not None:
        # IDEC's own assignments (argmax Q), comparable with DEC's argmax-Q labels
        q = student_q(torch.tensor(Z, dtype=torch.float32, device=DEVICE), mu)
        out_labels = q.argmax(1).cpu().numpy()
        extra = {"agree_ari_idec_vs_kmeans": float(adjusted_rand_score(km_labels, out_labels))}
    post_ari = float(adjusted_rand_score(data["gt"], out_labels))  # quarantined: reporting only
    d = os.path.join(OUT, combo_name, data["_ds"], f"seed{seed}")
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "fused_embedding.npy"), Z)
    np.save(os.path.join(d, "labels.npy"), out_labels)
    metrics = {"silhouette": sil, "reconstruction_balance": recbal,
               "r2_rna": r2r, "r2_aux": r2a, "k": k, "lam_spa_used": round(float(lam_used), 4),
               "wall_time_seconds": round(time.time() - t0, 2),
               **({"gate_rna": round(gate["rna"], 3), "gate_aux": round(gate["aux"], 3)} if gate else {}),
               **extra}
    json.dump(metrics, open(os.path.join(d, "metrics.json"), "w"), indent=2)
    json.dump({"ari": post_ari, "quarantined_reporting_only": True},
              open(os.path.join(d, "posthoc.json"), "w"), indent=2)
    with open(REG, "a") as f:
        f.write(json.dumps({"combo": combo_name, "dataset": data["_ds"], "seed": seed,
                            "metrics": metrics, "posthoc_ari": post_ari, "cfg": cfg}) + "\\n")
    print(f"done {combo_name}/{data['_ds']}/seed{seed} sil={sil:.3f} recbal={recbal:.3f} "
          f"posthocARI={post_ari:.3f} [{time.time()-t0:.0f}s]", flush=True)

print("preprocessing all datasets once (cached)...", flush=True)
DATA_CACHE = {}
for ds_id in DATASETS:
    d = load_dataset(ds_id); d["_ds"] = ds_id; DATA_CACHE[ds_id] = d
print("computing Moran's I per dataset (label-free spatial autocorrelation)...", flush=True)
for ds_id in DATASETS:
    d = DATA_CACHE[ds_id]
    g0 = build_graph(d["coords"], torch.device("cpu"))
    pc = PCA(n_components=min(10, d["Xr"].shape[1])).fit_transform(d["Xr"])
    Is = [moran_I(pc[:, j], g0["edge_index"], pc.shape[0]) for j in range(pc.shape[1])]
    MORAN[ds_id] = float(np.mean(Is))
    print(f"  {ds_id}: Moran's I = {MORAN[ds_id]:.3f}", flush=True)
print(f"  mean I = {sum(MORAN.values())/len(MORAN):.3f} (adaptive λ_spa anchors here)", flush=True)
print("computing cross-modal kNN agreement per dataset (label-free)...", flush=True)
for ds_id in DATASETS:
    d = DATA_CACHE[ds_id]
    XMODAL[ds_id] = cross_modal_agreement(d["Xr"], d["Xa"], k=15)
    print(f"  {ds_id}: agreement = {XMODAL[ds_id]:.3f} -> λ_dis = {0.2*(1-XMODAL[ds_id]):.3f}", flush=True)
for combo_name, override in COMBINATIONS:
    cfg = dict(BASE_CFG); cfg.update(override)
    for ds_id in DATASETS:
        for seed in SEEDS:
            if (combo_name, ds_id, seed) in done_set():
                print(f"skip done {combo_name}/{ds_id}/{seed}", flush=True)
                continue
            run_combo(combo_name, cfg, DATA_CACHE[ds_id], seed)
    git_push([f"runs/s5-notebook/{combo_name}", "runs/s5-notebook/registry.jsonl"],
             f"s5-notebook: {combo_name}")
print("stage-5 runs complete")
'''))

cells.append(code('''# ---------- aggregation + summary (RAUS v2, guard, quarantined post-hoc) ----------
from collections import defaultdict

def seed_stability(labs):
    vals = []
    ss = sorted(labs)
    for i in range(len(ss)):
        for j in range(i + 1, len(ss)):
            vals.append(adjusted_rand_score(labs[ss[i]], labs[ss[j]]))
    return float(np.mean(vals))

per_ds = defaultdict(list)   # ds -> list of (combo, sil, stab, recbal, postari)
combos = [c for c, _ in COMBINATIONS]
for combo in combos:
    for ds_id in DATASETS:
        sils, stab_labs, recbals, paris = [], {}, [], []
        ok = True
        for seed in SEEDS:
            d = os.path.join(OUT, combo, ds_id, f"seed{seed}")
            mp, pp = os.path.join(d, "metrics.json"), os.path.join(d, "posthoc.json")
            if not (os.path.exists(mp) and os.path.exists(pp)):
                ok = False; break
            m = json.load(open(mp)); p = json.load(open(pp))
            sils.append(m["silhouette"]); recbals.append(m["reconstruction_balance"])
            paris.append(p["ari"]); stab_labs[seed] = np.load(os.path.join(d, "labels.npy"))
        if not ok:
            continue
        per_ds[ds_id].append((combo, float(np.mean(sils)), seed_stability(stab_labs),
                              float(np.mean(recbals)), float(np.mean(paris))))

def borda_rank(rows):
    # rows: list of (combo, sil, stab, recbal, postari); lower borda = better
    pts = {}
    for m, rev in [(1, True), (2, True), (3, True)]:
        srt = sorted(rows, key=lambda r: r[m], reverse=rev)
        for i, r in enumerate(srt):
            pts[r[0]] = pts.get(r[0], 0) + i + 1
    return pts

cross = defaultdict(list)  # combo -> per-dataset borda ranks
for ds_id in sorted(per_ds):
    rows = per_ds[ds_id]
    pts = borda_rank(rows)
    by = {r[0]: r for r in rows}
    print(f"\\n=== {ds_id} (RAUS v2 Borda; lower better) ===")
    for combo in sorted(pts, key=pts.get):
        _, sil, stab, recbal, pari = by[combo]
        rs = sorted(rows, key=lambda r: r[1], reverse=True)
        rt = sorted(rows, key=lambda r: r[2], reverse=True)
        r_sil = [r[0] for r in rs].index(combo) + 1
        r_stab = [r[0] for r in rt].index(combo) + 1
        flag = " FLAG" if abs(r_sil - r_stab) >= 4 else ""
        print(f"#{list(sorted(pts, key=pts.get)).index(combo)+1:<3} {combo:24} borda={pts[combo]:3d} "
              f"sil={sil:.3f}[{r_sil}] stab={stab:.3f}[{r_stab}] recbal={recbal:.3f} postARI={pari:.3f}{flag}")
        cross[combo].append((ds_id, pts[combo]))

print("\\n=== cross-dataset (avg Borda rank; lower better) ===")
for combo in sorted(cross, key=lambda c: sum(r for _, r in cross[c]) / len(cross[c])):
    rs = cross[combo]
    avg = sum(r for _, r in rs) / len(rs)
    detail = " ".join(f"{d}#{r}" for d, r in rs)
    print(f"{combo:24} avg={avg:.2f}  {detail}")

json.dump({"borda": {c: [{"dataset": d, "borda": b} for d, b in v] for c, v in cross.items()}},
          open(os.path.join(OUT, "summary.json"), "w"), indent=2)
git_push(["runs/s5-notebook/summary.json", "runs/s5-notebook/registry.jsonl"], "s5-notebook: summary + registry")
print("\\nAll Stage-5 embeddings saved under", OUT)
'''))

cells.append(md('''## Notes
- **Resume:** the runner skips `(combo, dataset, seed)` already in `runs/s5-notebook/registry.jsonl`. Safe to re-run after disconnects.
- **Adaptive λ:** `lam_spa_used` is logged per run — check it varies sensibly with Moran's I (brain > lymph node).
- **Firewall:** labels are read once for the declared k and quarantined post-hoc ARI only. Never used in training, fusion, weighting, or selection.
- **Manual run (no agent):** upload to Kaggle → attach `pulokpulok/spatial-multiomics-6datasets-private` as input → add the `GITHUB_TOKEN` secret for auto-push → GPU T4 → Save & Run All. Without the secret, results stay in `/kaggle/working/my_research/runs/s5-notebook/`.
- **Budget:** 5 combos × 18 = 90 runs. Check the 30 GPU-h/week quota before launching.
'''))

nb = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.10"}},
    "nbformat": 4, "nbformat_minor": 4,
}

os.makedirs("notebooks", exist_ok=True)
out = "notebooks/stage5_generalization.ipynb"
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out, "cells:", len(cells))

import ast
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        ast.parse(c["source"])
print("all code cells parse OK")
