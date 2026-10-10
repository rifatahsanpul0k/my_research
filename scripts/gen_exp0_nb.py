#!/usr/bin/env python3
"""Generator for stage6_exp0_recluster.ipynb — Experiment 0: recluster saved embeddings.
Zero training. For each saved fused_embedding.npy in runs/s5-notebook/, try
transforms x clusterers, evaluate with unsupervised metrics + quarantined post-hoc ARI.
Runs on Kaggle (needs ground truth for ARI). Output: runs/s6-exp0/.
"""
import json

nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": []}

def md(t):
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": t.splitlines(True)})
def code(t):
    nb["cells"].append({"cell_type": "code", "metadata": {}, "execution_count": None,
                        "outputs": [], "source": t.splitlines(True)})

md("""# Stage 6 — Experiment 0: Recluster Saved Embeddings
**Zero training.** For every `fused_embedding.npy` in `runs/s5-notebook/`, try representation
transforms × clustering heads. Evaluate with unsupervised metrics + **quarantined post-hoc ARI**
(reporting only — never used for selection). Goal: find free ARI without retraining.""")

code('''# ---------- setup ----------
import os, sys, json, time, subprocess, numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

REPO = "/kaggle/working/my_research"
REPO_URL = "https://github.com/rifatahsanpul0k/my_research.git"
OUT = os.path.join(REPO, "runs", "s6-exp0")
os.makedirs(OUT, exist_ok=True)

def sh(*args):
    subprocess.run(list(args), check=True, cwd=REPO)

# github sync (same pattern as stage5)
import base64
try:
    from kaggle_secrets import UserSecretsClient
    _sec = UserSecretsClient()
    _b64 = base64.b64encode(f"x-access-token:{_sec.get_secret('GITHUB_TOKEN')}".encode()).decode()
    _GIT = ["git", "-c", f"http.extraHeader=Authorization: Basic {_b64}"]
    if not os.path.isdir(os.path.join(REPO, ".git")):
        subprocess.run(_GIT + ["clone", REPO_URL, REPO], check=True)
        sh("git", "config", "user.name", "kaggle-runner")
        sh("git", "config", "user.email", "kaggle-runner@local")
    else:
        sh(*_GIT, "pull", "--rebase", "origin", "main")
    GIT_OK = True
    print("GitHub sync ready")
except Exception as e:
    GIT_OK = False
    _s = str(e)
    try: _s = _s.replace(_b64, "***")
    except: pass
    print("GitHub sync unavailable (local save only):", _s)

def _scrub(s):
    try: return str(s).replace(_b64, "***")
    except: return str(s)

def git_push(paths, msg):
    if not GIT_OK: return
    try:
        sh("git", "add", *paths)
        r = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=REPO)
        if r.stdout.strip():
            sh("git", "commit", "-m", msg)
        sh(*_GIT, "pull", "--rebase", "origin", "main")
        sh(*_GIT, "push", "origin", "main")
        print("pushed:", msg)
    except Exception as e:
        print("push failed (continuing):", _scrub(e))
''')

code('''# ---------- ground truth (quarantined: ARI reporting only) ----------
# Labels live only in the Kaggle dataset, never in the repo.
import glob
roots = ["/kaggle/input/spatial-multiomics-6datasets-private",
         "/kaggle/working/spatial-multiomics-6datasets-private"]
root = next((r for r in roots if os.path.isdir(r)), None)
if root is None:
    # download as fallback
    subprocess.run(["kaggle","datasets","download","-d","pulokpulok/spatial-multiomics-6datasets-private",
                    "-p","/kaggle/working","--unzip"], check=True)
    root = "/kaggle/working/spatial-multiomics-6datasets-private"

import pandas as pd
GT = {}
for ds in ["A1","D1","E11","E13","E15","E18"]:
    ddir = os.path.join(root, ds)
    # annotation csv (kept out of the repo per firewall)
    cands = glob.glob(os.path.join(ddir, "*.csv"))
    assert cands, f"no csv in {ddir}"
    df = pd.read_csv(cands[0])
    # label column: first non-coordinate column
    labcol = [c for c in df.columns if c.lower() not in ("x","y","barcode","spot")][0]
    GT[ds] = df[labcol].astype(str).values
    print(ds, "gt:", GT[ds].shape, "classes:", len(np.unique(GT[ds])))
''')

code('''# ---------- transforms & clusterers ----------
def transform(Z, name):
    if name == "raw": return Z
    if name == "standard": return StandardScaler().fit_transform(Z)
    if name == "l2":
        n = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
        return Z / n
    if name == "pca24":
        return PCA(n_components=min(24, Z.shape[1]), random_state=0).fit_transform(Z)
    raise ValueError(name)

def cluster(Zt, tname, head, k, seed):
    if head == "kmeans":
        return KMeans(n_clusters=k, n_init=50, random_state=seed).fit_predict(Zt)
    if head == "spherical":
        # spherical k-means = k-means on L2-normalized (cosine geometry)
        Zn = Zt / (np.linalg.norm(Zt, axis=1, keepdims=True) + 1e-9)
        return KMeans(n_clusters=k, n_init=50, random_state=seed).fit_predict(Zn)
    if head == "gmm":
        return GaussianMixture(n_components=k, n_init=5, random_state=seed).fit_predict(Zt)
    raise ValueError(head)

TRANSFORMS = ["raw", "standard", "l2", "pca24"]
HEADS = ["kmeans", "spherical", "gmm"]
print("grid:", len(TRANSFORMS), "x", len(HEADS), "=", len(TRANSFORMS)*len(HEADS), "per embedding")
''')

code('''# ---------- run ----------
import glob
REG = os.path.join(OUT, "registry.jsonl")
done = set()
if os.path.exists(REG):
    for line in open(REG):
        r = json.loads(line)
        done.add((r["src_combo"], r["dataset"], r["seed"], r["transform"], r["head"]))

emb_files = sorted(glob.glob(os.path.join(REPO, "runs", "s5-notebook", "*", "*", "seed*", "fused_embedding.npy")))
print(len(emb_files), "embeddings")
n_new = 0
for ef in emb_files:
    parts = ef.split(os.sep)
    combo, ds, seeddir = parts[-4], parts[-3], parts[-2]
    seed = int(seeddir.replace("seed", ""))
    Z = np.load(ef)
    # k from the original metrics
    m = json.load(open(os.path.join(os.path.dirname(ef), "metrics.json")))
    k = int(m["k"]); gt = GT[ds]
    assert len(gt) == Z.shape[0], f"gt/embed mismatch {ds}"
    for tname in TRANSFORMS:
        Zt = transform(Z, tname)
        for head in HEADS:
            if (combo, ds, seed, tname, head) in done: continue
            t0 = time.time()
            lab = cluster(Zt, tname, head, k, seed)
            sil = float(silhouette_score(Zt, lab)) if len(np.unique(lab)) > 1 else -1.0
            ari = float(adjusted_rand_score(gt, lab))  # quarantined: reporting only
            # stability: agreement with a second seed
            lab2 = cluster(Zt, tname, head, k, seed + 999)
            stab = float(adjusted_rand_score(lab, lab2))
            with open(REG, "a") as f:
                f.write(json.dumps({"src_combo": combo, "dataset": ds, "seed": seed,
                    "transform": tname, "head": head, "silhouette": round(sil, 4),
                    "stability": round(stab, 4), "posthoc_ari": round(ari, 4),
                    "quarantined_reporting_only": True,
                    "time_s": round(time.time() - t0, 1)}) + "\\n")
            n_new += 1
    if n_new % 50 == 0:
        print(f"  {n_new} new...", flush=True)
print("done,", n_new, "new evaluations")
git_push(["runs/s6-exp0"], "s6-exp0: recluster evaluations")
''')

code('''# ---------- summary ----------
import pandas as pd
rows = [json.loads(l) for l in open(REG)]
df = pd.DataFrame(rows)
# mean ARI per (transform, head), averaged over all embeddings
piv = df.groupby(["transform", "head"])["posthoc_ari"].mean().unstack().round(3)
print("Mean post-hoc ARI (all embeddings):")
print(piv)
print("\\nBest per dataset:")
for ds in ["A1","D1","E11","E13","E15","E18"]:
    d = df[df.dataset == ds]
    b = d.loc[d.posthoc_ari.idxmax()]
    print(f"  {ds}: {b['transform']}+{b['head']} ARI={b['posthoc_ari']:.3f} (from {b['src_combo']})")
json.dump({"mean_ari": piv.to_dict()}, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
git_push(["runs/s6-exp0/summary.json", "runs/s6-exp0/registry.jsonl"], "s6-exp0: summary")
print("\\nAll Exp-0 results under", OUT)
''')

json.dump(nb, open("/home/hatch/workspace/my_research/notebooks/stage6_exp0_recluster.ipynb", "w"), indent=1)
# validate
import nbformat
nbformat.read("/home/hatch/workspace/my_research/notebooks/stage6_exp0_recluster.ipynb", as_version=4)
print("wrote notebooks/stage6_exp0_recluster.ipynb — valid")
