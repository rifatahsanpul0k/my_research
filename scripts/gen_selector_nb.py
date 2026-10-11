"""Generate notebooks/stage6_selector_test.ipynb — Selector validation test.

Tests whether unsupervised cross-seed stability can pick the best
(transform, head) per dataset without using ground-truth labels.

For the best encoder (S5N-dis005, mean ARI 0.355), run all 12
(transform, head) combos x 3 seeds per dataset, save full cluster labels,
compute pairwise ARI between seeds (stability, no GT), and check if
stability ranking matches quarantined ARI ranking.

Output: runs/s6-selector/
Run: python3 scripts/gen_selector_nb.py
"""
import json, os

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

cells = []

cells.append(md('''# Stage 6 — Selector Test: Can Unsupervised Stability Pick the Best Clustering?

**Goal:** validate that cross-seed stability (no labels) picks the same
(transform, head) that quarantined ARI would pick.

**Why:** Exp0 showed silhouette fails as a selector (3/18 correct). If
cross-seed stability works, we have the ONE pipeline:
fixed encoder (S5N-dis005) + stability-selected (transform, head) per dataset.

**Method:** for each dataset, run 12 (transform, head) x 3 seeds.
Compute pairwise ARI between seeds (label-free stability).
Compare stability ranking vs quarantined ARI ranking.

**Output:** `runs/s6-selector/`
'''))

cells.append(code('''import os, sys, json, subprocess, random, glob, time
import numpy as np
import pandas as pd

for pkg in ["scikit-learn", "pandas", "scipy"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import normalize
from itertools import combinations

def set_seed(s):
    random.seed(s); np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
    except ImportError:
        pass

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("imports ok | device:", DEVICE)
'''))

cells.append(code('''# ---------- GitHub push ----------
REPO = "/kaggle/working/my_research"
REPO_URL = "https://github.com/rifatahsanpul0k/my_research.git"
def sh(*args):
    r = subprocess.run(list(args), capture_output=True, text=True, cwd=REPO)
    return r
os.makedirs(REPO, exist_ok=True)
OUT = os.path.join(REPO, "runs", "s6-selector")
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
    # re-create OUT (git clone may have wiped it)
    os.makedirs(OUT, exist_ok=True)
    print("git ready")
except Exception as e:
    print("git setup issue:", str(e)[:200].replace(_tok if '_tok' in dir() else 'xxx', '***'))

def git_push(msg):
    try:
        # re-create OUT in case git ops wiped it
        os.makedirs(OUT, exist_ok=True)
        sh("git", "add", "runs/s6-selector")
        r = sh("git", "diff", "--cached", "--quiet")
        if r.returncode == 0:
            print("nothing new to push")
            return
        sh("git", "commit", "-m", msg)
        # pull before push
        r2 = subprocess.run(_GIT + ["pull", "--rebase", "origin", "main"],
                           capture_output=True, text=True, cwd=REPO)
        r3 = subprocess.run(_GIT + ["push", "origin", "main"],
                           capture_output=True, text=True, cwd=REPO)
        # scrub token from any output
        out = (r2.stdout + r2.stderr + r3.stdout + r3.stderr)
        if '_tok' in dir() and _tok:
            out = out.replace(_tok, "***")
        print(out[-500:] if out else "push done")
    except Exception as e:
        msg_e = str(e)
        if '_tok' in dir() and _tok:
            msg_e = msg_e.replace(_tok, "***")
        print("push failed:", msg_e[:300])
'''))

cells.append(code('''# ---------- Data loading (same MARKER pattern as Exp0) ----------
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
    # download fallback (same as stage5/exp0)
    print("dataset not found, attempting download...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)
    subprocess.run(["kaggle", "datasets", "download", "-d",
                    "pulokpulok/spatial-multiomics-6datasets-private",
                    "-p", "/kaggle/working", "--unzip"], check=True)
    DATA_ROOT = find_data_root()
print("DATA_ROOT:", DATA_ROOT)

# GT spec — EXACT copy from Exp0 (direct path joins, no guessing)
GT_SPEC = {
    "A1":  dict(folder="10x_human_lymph_node_A1", gt_file="annotation.csv", gt_col="manual-anno"),
    "D1":  dict(folder="10x_human_lymph_node_D1", gt_file="annotation.csv", gt_col="manual-anno"),
    "E11": dict(folder="Mouse_Brain_E11_S1", gt_file="anno.csv", gt_col="cluster"),
    "E13": dict(folder="Mouse_Brain_E13_S1", gt_file="anno.csv", gt_col="cluster"),
    "E15": dict(folder="Mouse_Brain_E15_S1", gt_file="anno.csv", gt_col="cluster"),
    "E18": dict(folder="Mouse_Brain_E18_S1", gt_file="anno.csv", gt_col="cluster"),
}
GT = {}
for ds, spec in GT_SPEC.items():
    p = os.path.join(DATA_ROOT, spec["folder"], spec["gt_file"])
    assert os.path.exists(p), f"missing {p}"
    df = pd.read_csv(p, index_col=0)
    GT[ds] = df[spec["gt_col"]].astype(str).values
    print(ds, "gt:", GT[ds].shape, "classes:", len(np.unique(GT[ds])))

# declared k (same exception as stage5: applied identically, disclosed)
DECLARED_K = {"A1": 10, "D1": 11, "E11": 8, "E13": 12, "E15": 12, "E18": 14}
# DATASETS for iteration
DATASETS = list(DECLARED_K.keys())
'''))

cells.append(code('''# ---------- GPU clustering (same as Exp0) ----------
def torch_kmeans(X, k, n_init=10, max_iter=300, seed=0):
    """GPU KMeans with k-means++ init. Returns labels."""
    set_seed(seed)
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    n, d = X_t.shape
    best_labels, best_inertia = None, float('inf')
    for _ in range(n_init):
        # k-means++ init
        idx = [torch.randint(n, (1,)).item()]
        for _ in range(1, k):
            dist = torch.cdist(X_t, X_t[idx]).min(dim=1)[0]
            probs = dist ** 2
            probs /= probs.sum()
            idx.append(torch.multinomial(probs, 1).item())
        C = X_t[idx].clone()
        for _ in range(max_iter):
            dist = torch.cdist(X_t, C)
            labels = dist.argmin(dim=1)
            C_new = torch.stack([X_t[labels == j].mean(dim=0) if (labels == j).any()
                                 else X_t[torch.randint(n, (1,)).item()]
                                 for j in range(k)])
            if torch.allclose(C, C_new, atol=1e-4):
                break
            C = C_new
        inertia = torch.cdist(X_t, C).min(dim=1)[0].pow(2).sum().item()
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.cpu().numpy()
    return best_labels

def gpu_silhouette(X, labels, max_n=2000, seed=0):
    """Silhouette on GPU, subsampled."""
    set_seed(seed)
    n = len(X)
    if n > max_n:
        idx = np.random.choice(n, max_n, replace=False)
        X, labels = X[idx], labels[idx]
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    D = torch.cdist(X_t, X_t)
    labels_t = torch.tensor(labels, device=DEVICE)
    uniq = torch.unique(labels_t)
    sil = []
    for i in range(len(X_t)):
        same = (labels_t == labels_t[i])
        same[i] = False
        a = D[i][same].mean().item() if same.any() else 0
        b = min(D[i][labels_t == u].mean().item()
                for u in uniq if u != labels_t[i] and (labels_t == u).any())
        sil.append((b - a) / max(a, b) if max(a, b) > 0 else 0)
    return float(np.mean(sil))

def apply_transform(Z, transform):
    if transform == "raw":
        return Z
    elif transform == "standard":
        return StandardScaler().fit_transform(Z)
    elif transform == "l2":
        return normalize(Z, norm='l2')
    elif transform == "pca24":
        return PCA(n_components=min(24, Z.shape[1]-1), random_state=0).fit_transform(Z)
    raise ValueError(transform)

def apply_head(Zt, head, k, seed):
    if head == "kmeans":
        return torch_kmeans(Zt, k, n_init=10, seed=seed)
    elif head == "spherical":
        Zn = normalize(Zt, norm='l2')
        return torch_kmeans(Zn, k, n_init=10, seed=seed)
    elif head == "gmm":
        set_seed(seed)
        return GaussianMixture(n_components=k, n_init=3, random_state=seed).fit_predict(Zt)
    raise ValueError(head)

print("clustering functions ready")
'''))

cells.append(code('''# ---------- Selector test ----------
ENCODER = "S5N-dis005"  # best mean ARI (0.355)
SEEDS = [42, 1234, 2024]
TRANSFORMS = ["raw", "standard", "l2", "pca24"]
HEADS = ["kmeans", "spherical", "gmm"]

# Find embedding files for the encoder
S5 = os.path.join(REPO, "runs", "s5-notebook", ENCODER)
print(f"Looking for embeddings in {S5}")
assert os.path.isdir(S5), f"Encoder dir not found: {S5}"

results = []
summary_path = os.path.join(OUT, "selector_results.jsonl")

# Load existing results for resume
done = set()
if os.path.exists(summary_path):
    with open(summary_path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                done.add((r['dataset'], r['transform'], r['head'], r['seed']))
    print(f"Resuming: {len(done)} evals already done")

fout = open(summary_path, "a")

for ds_name in DATASETS:
    k = DECLARED_K[ds_name]
    ds_dir = os.path.join(S5, ds_name)
    if not os.path.isdir(ds_dir):
        print(f"  SKIP {ds_name}: no dir")
        continue

    # GT pre-loaded above (quarantined — reporting only)
    gt = GT[ds_name]

    for transform in TRANSFORMS:
        for head in HEADS:
            # Collect labels across seeds for stability computation
            seed_labels = {}
            seed_aris = {}
            for seed in SEEDS:
                key = (ds_name, transform, head, seed)
                if key in done:
                    continue
                seed_dir = os.path.join(ds_dir, f"seed{seed}")
                emb_path = os.path.join(seed_dir, "fused_embedding.npy")
                if not os.path.exists(emb_path):
                    print(f"    MISSING {emb_path}")
                    continue
                Z = np.load(emb_path)
                Zt = apply_transform(Z, transform)
                labels = apply_head(Zt, head, k, seed)
                ari = float(adjusted_rand_score(gt, labels))
                sil = float(gpu_silhouette(Zt, labels))

                # Save labels for stability computation (compressed)
                lab_path = os.path.join(OUT, f"{ds_name}_{transform}_{head}_seed{seed}_labels.npy")
                np.save(lab_path, labels)

                rec = dict(dataset=ds_name, transform=transform, head=head,
                           seed=seed, ari=ari, silhouette=sil)
                fout.write(json.dumps(rec) + chr(10))
                fout.flush()
                seed_labels[seed] = labels
                seed_aris[seed] = ari
                done.add(key)

            # Compute cross-seed stability (label-free)
            if len(seed_labels) >= 2:
                pairs = list(combinations(sorted(seed_labels.keys()), 2))
                stab = np.mean([adjusted_rand_score(seed_labels[a], seed_labels[b])
                                for a, b in pairs])
                mean_ari = np.mean(list(seed_aris.values()))
                mean_sil = np.mean([r['silhouette'] for r in [json.loads(l) for l in open(summary_path)]
                                    if r['dataset']==ds_name and r['transform']==transform and r['head']==head])
                print(f"  {ds_name} {transform:9} {head:9} | stability={stab:.3f} | ARI={mean_ari:.3f}")

fout.close()
print("\\nDone. Results in", summary_path)
'''))

cells.append(code('''# ---------- Analyze: does stability pick the ARI winner? ----------
from collections import defaultdict

rows = [json.loads(l) for l in open(os.path.join(OUT, "selector_results.jsonl")) if l.strip()]
grp = defaultdict(list)
for r in rows:
    grp[(r['dataset'], r['transform'], r['head'])].append(r)

print("=" * 70)
print("SELECTOR VALIDATION: stability vs quarantined ARI")
print("=" * 70)

correct_stab = 0
correct_sil = 0
total = 0

for ds_name in DATASETS:
    k = DECLARED_K[ds_name]
    cands = []
    for t in TRANSFORMS:
        for h in HEADS:
            key = (ds_name, t, h)
            if key not in grp or len(grp[key]) < 2:
                continue
            rs = grp[key]
            ari = np.mean([x['ari'] for x in rs])
            sil = np.mean([x['silhouette'] for x in rs])
            # Load labels and compute stability
            labs = []
            for x in rs:
                lp = os.path.join(OUT, f"{ds_name}_{t}_{h}_seed{x['seed']}_labels.npy")
                if os.path.exists(lp):
                    labs.append(np.load(lp))
            if len(labs) >= 2:
                stab = np.mean([adjusted_rand_score(a, b)
                                for a, b in combinations(labs, 2)])
            else:
                stab = 0
            cands.append({'t': t, 'h': h, 'ari': ari, 'sil': sil, 'stab': stab})

    if not cands:
        continue
    total += 1
    by_ari = max(cands, key=lambda x: x['ari'])
    by_stab = max(cands, key=lambda x: x['stab'])
    by_sil = max(cands, key=lambda x: x['sil'])

    stab_ok = (by_stab['t'], by_stab['h']) == (by_ari['t'], by_ari['h'])
    sil_ok = (by_sil['t'], by_sil['h']) == (by_ari['t'], by_ari['h'])
    correct_stab += stab_ok
    correct_sil += sil_ok

    print(f"\\n{ds_name}:")
    print(f"  ARI winner:       {by_ari['t']}/{by_ari['h']}  ARI={by_ari['ari']:.3f}")
    print(f"  Stability picks:  {by_stab['t']}/{by_stab['h']}  {'✓' if stab_ok else '✗'}  (stab={by_stab['stab']:.3f}, ARI={by_stab['ari']:.3f})")
    print(f"  Silhouette picks: {by_sil['t']}/{by_sil['h']}  {'✓' if sil_ok else '✗'}  (sil={by_sil['sil']:.3f}, ARI={by_sil['ari']:.3f})")

print(f"\\n{'=' * 70}")
print(f"Stability correct:  {correct_stab}/{total}")
print(f"Silhouette correct: {correct_sil}/{total}")
print(f"{'=' * 70}")

# Save summary
with open(os.path.join(OUT, "selector_summary.json"), "w") as f:
    json.dump({"stability_correct": correct_stab, "silhouette_correct": correct_sil,
               "total": total, "encoder": ENCODER}, f, indent=2)

git_push("s6-selector: stability selector validation")
print("\\nPushed to GitHub")
'''))

# Write notebook
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "cells": cells,
}

out_path = os.path.join(os.path.dirname(__file__), "..", "notebooks", "stage6_selector_test.ipynb")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {out_path} ({len(cells)} cells)")

# Validate
import nbformat
nbformat.read(out_path, as_version=4)
print("nbformat: valid")
