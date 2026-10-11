"""Generate notebooks/stage6_baseline_pca_leiden.ipynb — PCA+Leiden baseline.

Agent review Experiment 2: "You have no baseline comparison. If PCA+Leiden
achieves ARI 0.22 on A1, your best model (0.25) is only +0.03 ahead."

For each dataset: load preprocessed Xr (RNA PCA64) + Xa (aux), concatenate,
standard-scale, PCA to 32D, build kNN graph, Leiden at multiple resolutions.
Quarantined post-hoc ARI. Compare to neural model best.

CPU-only. ~10 minutes.
Run: python3 scripts/gen_baseline_nb.py
"""
import json, os

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

cells = []

cells.append(md('''# Stage 6 — Baseline: PCA + Leiden (No Neural Network)

**Goal:** Establish the floor. How well does a trivial non-neural baseline do?

**Why (agent review):** "If PCA+Leiden achieves ARI 0.22 on A1, your best model
(0.25) is only +0.03 ahead — that's within noise. You cannot evaluate progress
without knowing the floor."

**Method:** For each dataset, concatenate preprocessed RNA (PCA64) + aux features,
standard-scale, PCA to 32D (match embedding dim), kNN graph (k=15), Leiden
clustering at resolutions [0.3, 0.5, 0.8, 1.0, 1.5]. Report best ARI per dataset.

**Output:** `runs/s6-baseline/`
'''))

cells.append(code('''import os, sys, json, subprocess
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import adjusted_rand_score

print("imports ok")
'''))

cells.append(code('''# ---------- GitHub push ----------
REPO = "/kaggle/working/my_research"
REPO_URL = "https://github.com/rifatahsanpul0k/my_research.git"
def sh(*args):
    r = subprocess.run(list(args), capture_output=True, text=True, cwd=REPO)
    return r
os.makedirs(REPO, exist_ok=True)
OUT = os.path.join(REPO, "runs", "s6-baseline")
os.makedirs(OUT, exist_ok=True)
try:
    from kaggle_secrets import UserSecretsClient
    import base64, shutil
    _sec = UserSecretsClient()
    _tok = _sec.get_secret('GITHUB_TOKEN')
    _b64 = base64.b64encode(f"x-access-token:{_tok}".encode()).decode()
    _GIT = ["git", "-c", f"http.extraHeader=Authorization: Basic {_b64}"]
    if not os.path.isdir(os.path.join(REPO, ".git")):
        if os.listdir(REPO):
            bak = REPO + "_stalebak"
            if os.path.exists(bak):
                shutil.rmtree(bak)
            shutil.move(REPO, bak)
            os.makedirs(REPO, exist_ok=True)
        subprocess.run(_GIT + ["clone", REPO_URL, REPO], check=True)
        sh("git", "config", "user.name", "kaggle-runner")
        sh("git", "config", "user.email", "kaggle-runner@local")
    os.makedirs(OUT, exist_ok=True)
    print("git ready")
except Exception as e:
    print("git setup issue:", str(e)[:200])

def git_push(msg):
    try:
        os.makedirs(OUT, exist_ok=True)
        sh("git", "add", "runs/s6-baseline")
        r = sh("git", "diff", "--cached", "--quiet")
        if r.returncode == 0:
            print("nothing new to push")
            return
        sh("git", "commit", "-m", msg)
        r2 = subprocess.run(_GIT + ["pull", "--rebase", "origin", "main"],
                           capture_output=True, text=True, cwd=REPO)
        r3 = subprocess.run(_GIT + ["push", "origin", "main"],
                           capture_output=True, text=True, cwd=REPO)
        print("pushed")
    except Exception as e:
        print("push failed:", str(e)[:300])
'''))

cells.append(code('''# ---------- Data loading (EXACT copy from stage5) ----------
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
    print("dataset not found, attempting download...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)
    subprocess.run(["kaggle", "datasets", "download", "-d",
                    "pulokpulok/spatial-multiomics-6datasets-private",
                    "-p", "/kaggle/working", "--unzip"], check=True)
    DATA_ROOT = find_data_root()
print("DATA_ROOT:", DATA_ROOT)

DATASETS = {
    "A1":  dict(folder="10x_human_lymph_node_A1", aux="adt",  gt_file="annotation.csv", gt_col="manual-anno"),
    "D1":  dict(folder="10x_human_lymph_node_D1", aux="adt",  gt_file="annotation.csv", gt_col="manual-anno"),
    "E11": dict(folder="Mouse_Brain_E11_S1", aux="atac", gt_file="anno.csv", gt_col="cluster"),
    "E13": dict(folder="Mouse_Brain_E13_S1", aux="atac", gt_file="anno.csv", gt_col="cluster"),
    "E15": dict(folder="Mouse_Brain_E15_S1", aux="atac", gt_file="anno.csv", gt_col="cluster"),
    "E18": dict(folder="Mouse_Brain_E18_S1", aux="atac", gt_file="anno.csv", gt_col="cluster"),
}

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
    Xr = StandardScaler().fit_transform(PCA(n_components=64, random_state=0).fit_transform(Xh)).astype(np.float32)
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
    k_declared = int(gt[spec["gt_col"]].nunique())
    print(f"{ds_id}: RNA {Xr.shape} AUX {Xa.shape} N={Xr.shape[0]} k={k_declared}", flush=True)
    return dict(Xr=Xr, Xa=Xa, k=k_declared, gt=np.asarray(gt[spec["gt_col"]]).astype(str))
'''))

cells.append(code('''# ---------- PCA + Leiden baseline ----------
RESOLUTIONS = [0.3, 0.5, 0.8, 1.0, 1.5]
DECLARED_K = {"A1": 10, "D1": 11, "E11": 8, "E13": 12, "E15": 12, "E18": 14}

# Neural model bests for comparison (from Exp0)
NEURAL_BEST = {"A1": 0.250, "D1": 0.250, "E11": 0.445, "E13": 0.436, "E15": 0.506, "E18": 0.323}

results = {}

for ds_id in DATASETS:
    print(f"\\n{'='*50}")
    print(f"Dataset: {ds_id}")
    print(f"{'='*50}")
    data = load_dataset(ds_id)
    Xr, Xa, gt = data["Xr"], data["Xa"], data["gt"]
    k = DECLARED_K[ds_id]

    # Concatenate RNA + aux, scale, PCA to 32D (match embedding dim)
    Xcat = np.concatenate([Xr, Xa], axis=1)
    Xcat = StandardScaler().fit_transform(Xcat)
    X32 = PCA(n_components=32, random_state=0).fit_transform(Xcat)

    # Build AnnData for scanpy
    adata = sc.AnnData(X32)
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="X")
    # Also try with raw concatenated features
    adata2 = sc.AnnData(Xcat)
    sc.pp.neighbors(adata2, n_neighbors=15, use_rep="X")

    best_ari = 0
    best_cfg = ""
    for res in RESOLUTIONS:
        for name, ad in [("pca32", adata), ("concat", adata2)]:
            sc.tl.leiden(ad, resolution=res, random_state=0, key_added=f"leiden_{res}")
            labels = ad.obs[f"leiden_{res}"].values
            ari = adjusted_rand_score(gt, labels)
            if ari > best_ari:
                best_ari = ari
                best_cfg = f"{name}/res={res} (n_clust={len(np.unique(labels))})"

    neural = NEURAL_BEST[ds_id]
    gap = neural - best_ari
    print(f"\\n  Baseline best ARI: {best_ari:.3f} ({best_cfg})")
    print(f"  Neural best ARI:   {neural:.3f}")
    print(f"  Neural advantage:  {gap:+.3f} {'(meaningful)' if gap > 0.05 else '(marginal/noise)'}")

    results[ds_id] = {"baseline_ari": float(best_ari), "baseline_cfg": best_cfg,
                      "neural_ari": neural, "gap": float(gap)}

# Save
with open(os.path.join(OUT, "baseline_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\\n{'='*50}")
print("SUMMARY")
print(f"{'='*50}")
for ds_id, r in results.items():
    print(f"  {ds_id}: baseline={r['baseline_ari']:.3f} neural={r['neural_ari']:.3f} gap={r['gap']:+.3f}")

git_push("s6-baseline: PCA+Leiden baseline results")
print("\\nDone.")
'''))

# Write notebook
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "cells": cells,
}

out_path = os.path.join(os.path.dirname(__file__), "..", "notebooks", "stage6_baseline_pca_leiden.ipynb")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {out_path} ({len(cells)} cells)")

import nbformat
nbformat.read(out_path, as_version=4)
print("nbformat: valid")
