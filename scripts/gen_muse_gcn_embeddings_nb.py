#!/usr/bin/env python3
"""Generator: notebooks/muse_gcn_4encoder_1layer_embeddings.ipynb

Builds the Kaggle notebook that runs AriseSpatialGlue_4Encoder_1Layer.py
(Muse-GCN 4-Encoder, 1-Layer) at its DEFAULT settings on A1 + D1 only,
with ONE added behaviour vs the original script: every run's final fused
embedding (and its KMeans labels) is saved as .npy, then pushed to
embeddings/muse-gcn-4encoder/ in rifatahsanpul0k/my_research.
"""
import ast
import json
import os

SRC = os.path.expanduser("~/workspace/user/files/AriseSpatialGlue_4Encoder_1Layer.py")
OUT = os.path.expanduser("~/workspace/muse_gcn_nb/muse_gcn_4encoder_1layer_embeddings.ipynb")

with open(SRC) as f:
    module_src = f.read()

# --- Strip the CLI __main__ block (everything from the __main__ guard on) ---
marker = "if __name__ == '__main__':"
assert module_src.count(marker) == 1, "CLI marker not found exactly once"
module_src = module_src[: module_src.index(marker)].rstrip() + "\n"

# --- Inject the embedding save step into run_experiment ---
anchor = """            final_embeddings = result['final_embeddings']
            final_labels = result['final_labels']
"""
assert module_src.count(anchor) == 1, "embedding anchor not found exactly once"
save_block = anchor + """            emb_dir = os.path.join(output_dir, "embeddings")
            os.makedirs(emb_dir, exist_ok=True)
            np.save(os.path.join(emb_dir, f"{dataset_name}_seed{seed}_embeddings.npy"), final_embeddings)
            np.save(os.path.join(emb_dir, f"{dataset_name}_seed{seed}_labels.npy"), final_labels)
            print(f"Saved embeddings: {dataset_name} seed {seed} shape {final_embeddings.shape}")
"""
module_src = module_src.replace(anchor, save_block)
ast.parse(module_src)  # must compile as a module

# ---------------------------------------------------------------- cells ---
def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}

def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }

cells = []

cells.append(md("""
# Muse-GCN 4-Encoder 1-Layer — Embedding Extraction (A1 + D1)

Runs `AriseSpatialGlue_4Encoder_1Layer.py` (Muse-GCN: 4 x 1-layer GCN encoders — RNA spatial, RNA similarity, Aux spatial, Aux similarity — with concatenation fusion) at its **default settings**, on the two human lymph node datasets only:

- `10x_human_lymph_node_A1` (~3,500 spots, RNA + ADT)
- `10x_human_lymph_node_D1` (~3,400 spots, RNA + ADT)

**Defaults (unchanged from the script):** 20 seeds (13, 2560, 641, 1892, 1173, 69, 2024, 231, 1971, 2497, 338, 3127, 2001, 2022, 574, 2428, 999, 1187, 42, 3999), 350 epochs, lr 1e-3, beta 25 / gamma 10 / delta 1, hidden 512, **embedding dim (out_dim) = 64**, RNA PCA = 60 comps, KMeans clustering, last-epoch embeddings.

**One addition vs the original script:** the script never saved its embeddings (they existed only in memory; the dashboard export carried only UMAP coordinates and labels). This notebook saves each run's final fused embedding + labels as `.npy` and, in the last cell, pushes them to GitHub (`rifatahsanpul0k/my_research`, folder `embeddings/muse-gcn-4encoder/`).

**Kaggle setup before running:**
1. Add the input dataset **spatial-multiomics-6datasets-private** (the notebook auto-locates the A1/D1 folders under `/kaggle/input`).
2. Accelerator: **GPU** (expect roughly 1–3 hours for 40 runs; do not run on CPU).
3. Internet: **ON** (pip install + GitHub push).
4. Add a Kaggle Secret named `GITHUB_TOKEN` (fine-grained PAT with Contents read/write on `my_research`) — only needed for the final push cell; without it the embeddings still land in `/kaggle/working/results/embeddings/` (Save Version to keep them).
"""))

cells.append(code("""
# --- Install missing packages (Kaggle ships torch + scanpy; torch_geometric usually needs installing) ---
import importlib, subprocess, sys
for pkg, mod in [("torch_geometric", "torch_geometric"), ("scanpy", "scanpy")]:
    try:
        importlib.import_module(mod)
        print(f"OK: {mod} already installed")
    except ImportError:
        print(f"Installing {pkg} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
        importlib.import_module(mod)
        print(f"Installed: {mod}")
import torch
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available())
"""))

cells.append(code("""
# --- Locate the A1 / D1 dataset folders under /kaggle/input ---
import os, glob

WANT = {
    "10x_human_lymph_node_A1": ["adata_RNA.h5ad", "adata_ADT.h5ad", "annotation.csv"],
    "10x_human_lymph_node_D1": ["adata_RNA.h5ad", "adata_ADT.h5ad", "annotation.csv"],
}

DATA_DIR = None
for cand in glob.glob("/kaggle/input/**/10x_human_lymph_node_A1", recursive=True):
    parent = os.path.dirname(cand)
    if all(os.path.exists(os.path.join(parent, ds, fn)) for ds, files in WANT.items() for fn in files):
        DATA_DIR = parent
        break

assert DATA_DIR, "A1/D1 folders not found under /kaggle/input — attach the spatial-multiomics-6datasets-private dataset."
print("DATA_DIR =", DATA_DIR)
for ds, files in WANT.items():
    for fn in files:
        p = os.path.join(DATA_DIR, ds, fn)
        print(f"{os.path.getsize(p)/1e6:8.1f} MB  {p}")
"""))

cells.append(code(module_src))

cells.append(code("""
# --- Run: A1 + D1 only, everything else at the script's defaults ---
# seeds=None -> DEFAULT_SEEDS (all 20). visualize=False and eval_every_epoch=False
# match the script's own CLI defaults (plots / per-epoch eval do not change embeddings).
df = run_experiment(
    datasets=["10x_human_lymph_node_A1", "10x_human_lymph_node_D1"],
    seeds=None,
    output_dir="/kaggle/working/results",
    data_dir=DATA_DIR,
    visualize=False,
    eval_every_epoch=False,
    api_url=None,
)
df.to_csv("/kaggle/working/results/run_summary_check.csv", index=False)
print(df.groupby("dataset")[["ARI", "Silhouette", "NMI"]].mean())
"""))

cells.append(code("""
# --- Verify: every dataset x seed embedding exists, right shape, rows = spots ---
import numpy as np, pandas as pd, glob, os

emb_files = sorted(glob.glob("/kaggle/working/results/embeddings/*_embeddings.npy"))
print(f"{len(emb_files)} embedding files")
problems = []
for ds in ["10x_human_lymph_node_A1", "10x_human_lymph_node_D1"]:
    n_spots = len(pd.read_csv(os.path.join(DATA_DIR, ds, "annotation.csv"), index_col=0))
    for f in emb_files:
        if ds in f:
            arr = np.load(f)
            ok = arr.ndim == 2 and arr.shape[0] == n_spots and arr.shape[1] == 64
            if not ok:
                problems.append((f, arr.shape, n_spots))
            print(f"{'OK ' if ok else 'BAD'} {os.path.basename(f)}: shape {arr.shape} (spots={n_spots})")
assert len(emb_files) == 40, f"expected 40 embeddings (2 datasets x 20 seeds), found {len(emb_files)}"
assert not problems, f"shape mismatches: {problems}"
print("All embeddings verified: 40 files, each n_spots x 64, row order = adata_RNA obs order.")
"""))

cells.append(code("""
# --- Push embeddings (+ result CSVs) to GitHub: rifatahsanpul0k/my_research, embeddings/muse-gcn-4encoder/ ---
import os, shutil, subprocess, glob

def sh(*args, **kw):
    print("+", " ".join(args))
    return subprocess.run(args, check=True, **kw)

try:
    from kaggle_secrets import UserSecretsClient
    TOKEN = UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception as e:
    TOKEN = None
    print(f"GITHUB_TOKEN secret not available ({e}). Skipping push — embeddings are in /kaggle/working/results/embeddings/ (Save Version to keep them).")

if TOKEN:
    work = "/kaggle/working/my_research_push"
    if os.path.exists(work):
        shutil.rmtree(work)
    sh("git", "clone", "--depth", "50", f"https://x-access-token:{TOKEN}@github.com/rifatahsanpul0k/my_research.git", work)
    dest = os.path.join(work, "embeddings", "muse-gcn-4encoder")
    os.makedirs(dest, exist_ok=True)
    copied = 0
    for f in glob.glob("/kaggle/working/results/embeddings/*.npy") + glob.glob("/kaggle/working/results/*.csv"):
        shutil.copy(f, os.path.join(dest, os.path.basename(f)))
        copied += 1
    print(f"Copied {copied} files into embeddings/muse-gcn-4encoder/")
    env = dict(os.environ, GIT_AUTHOR_NAME="PULOK (Kaggle)", GIT_AUTHOR_EMAIL="pulok.github@gmail.com",
               GIT_COMMITTER_NAME="PULOK (Kaggle)", GIT_COMMITTER_EMAIL="pulok.github@gmail.com")
    sh("git", "-C", work, "add", "embeddings/muse-gcn-4encoder", env=env)
    status = subprocess.run(["git", "-C", work, "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    if status:
        sh("git", "-C", work, "commit", "-m", "Add Muse-GCN 4-encoder 1-layer embeddings for A1 + D1 (20 seeds, out_dim 64)", env=env)
        sh("git", "-C", work, "pull", "--rebase", env=env)
        sh("git", "-C", work, "push", env=env)
        print("Pushed to GitHub: embeddings/muse-gcn-4encoder/")
    else:
        print("Nothing new to push (files already in repo).")
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)

# Validate: JSON round-trip + every code cell compiles
loaded = json.load(open(OUT))
for i, c in enumerate(loaded["cells"]):
    if c["cell_type"] == "code":
        compile("".join(c["source"]), f"cell_{i}", "exec")
print(f"Notebook written: {OUT} ({len(cells)} cells, all code cells compile)")
