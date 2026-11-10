"""Generate notebooks/hello_pipeline_smoke_A1_colab.ipynb

Colab-only variant of the hello-pipeline SMOKE TEST (A1 only). The
pipeline cells are NOT duplicated here: this generator loads the base
notebook (notebooks/hello_pipeline_smoke_A1.ipynb, same repo) and
replaces only the platform-specific cells:

  cell 0  notes        -> Colab instructions (Drive datasets, dummy-data
                          warnings, /content lifetime warning)
  cell 1  install      -> scanpy + statsmodels + gdown
  cell 2  config       -> /content paths; DATASET_REGISTRY with the
                          Google Drive folder link for all six datasets;
                          ensure_dataset() reuses a local folder or
                          downloads it with gdown (this run: A1 only)
  cell 9  push         -> GITHUB_TOKEN from Colab Secrets or a hidden
                          getpass prompt; clone under /content
  cell 10 (appended)   -> zip /content/smoke_outputs and download it

Cells 3-8 (prep, verbatim hello.md main code, driver with the single
n_perm edit, dummy external data, run, save) are carried over, with ONE
notebook-level fix in cell 6 (dummy external data): the A1/D1 gene panel
carries 3 duplicated symbols (TBCE, HSPA14, TMSB15B), so the dummy
Series/DataFrame built on those names have duplicated index labels and a
lookup by such a label returns a Series instead of a scalar — this
crashed Engine 1 in the first live smoke run (2026-11-10, "truth value of
a Series is ambiguous"). Cell 6 now drops duplicated index entries
(keep="first") right after generating the dummy data — the exact patch
the user ran by hand to complete the smoke test, folded in permanently.
The generated notebook contains no Kaggle code or references at all.

Run: python3 scripts/gen_hello_pipeline_smoke_colab_nb.py
"""
import json, os

MD_COLAB = """# Hello Pipeline — SMOKE TEST (A1 only, Google Colab)

**Purpose: crash-test only.** This notebook runs the main code from `hello.md` (Stages 1–5 + driver) end-to-end on **A1** to find errors. **Every external input is DUMMY, so no number that comes out of this run is science:**

- **MAGMA z-scores: random dummy** (seeded) for all 15 panel traits.
- **Gene covariates: random dummy** (gene_length, snp_density).
- **Gene annotation (chrom/start/end/biotype): DUMMY** — random chromosomes/positions, all `protein_coding`. Consequence: the −MHC track is almost identical to +MHC (dummy positions sit far from the chr6:25–34 Mb window). Real run will fetch real annotation (mygene/pybiomart).
- **S-LDSC: not present** — lookups are `None`, so Engine 2 uses its internal OLS-proxy fallback (as-is, per plan).
- **Permutations: n_perm = 10** (not 1,000). Consequence: the smallest possible spin p-value is 1/11 ≈ 0.091, so **no hit can reach "Confirmed" in this run by construction** — that is expected, not a failure.

**Real inputs (verified):** Muse-GCN embeddings for A1, seed 42 — `10x_human_lymph_node_A1_seed42_embeddings.npy` (3484×64) + `..._labels.npy` (3484 values, 10 domains 0–9, BIRCH) — downloaded from the GitHub repo `embeddings/` folder inside the notebook. RNA counts come from `adata.X`, exactly as the driver is written.

**Datasets come from Google Drive (Colab-only, no other platform involved):** the config cell holds a `DATASET_REGISTRY` with the Drive folder link for all six datasets. `ensure_dataset(name)` reuses the folder if it is already under `/content` (or a mounted Drive), otherwise downloads it with `gdown`. This smoke run downloads **only A1** (~58 MB: adata_RNA.h5ad 56.6 MB + adata_ADT.h5ad 1.1 MB + annotation.csv; contents verified 2026-11-09). The Drive folders must stay link-shared ("anyone with the link"), or `gdown` cannot fetch them — if a download fails with a permission error, that sharing setting is the first thing to check.

**GitHub push (optional):** the push cell reads `GITHUB_TOKEN` from Colab Secrets, or asks with a hidden prompt. Without a token the run still completes — outputs stay in `/content/smoke_outputs/` and the last cell zips them for download. **Download or push before closing the tab: Colab deletes `/content` when the runtime ends.**

**Runtime:** CPU runtime is fine (Runtime → Change runtime type → CPU; the code is NumPy, a GPU changes nothing). Estimate ~15–45 min for the pipeline, plus under a minute for the A1 download.

**Success =** the run cell finishes, `smoke_outputs/` contains `A1_results_df.csv` + `A1_summary.json`, and tier counts print. If any cell errors, copy the full traceback.
"""

INSTALL_COLAB = """import os, sys, json, subprocess
for pkg in ["scanpy", "statsmodels", "gdown"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
import numpy as np
import pandas as pd
import scanpy as sc
print("imports ok; numpy", np.__version__, "| pandas", pd.__version__)
"""

CONFIG_COLAB = """# ---------- CONFIG (smoke test, locked 2026-11-09) - Colab + Google Drive ----------
SEED = 42
SMOKE_N_PERM = 10            # smoke only; real run = 1000 (and 10000 recalibration)
SECTION = "10x_human_lymph_node_A1"
EMB_NAME = "10x_human_lymph_node_A1_seed42_embeddings.npy"
LAB_NAME = "10x_human_lymph_node_A1_seed42_labels.npy"
RAW_BASE = "https://raw.githubusercontent.com/rifatahsanpul0k/my_research/main/embeddings"
WORK = "/content/smoke_inputs"
OUTDIR = "/content/smoke_outputs"
DATA_HOME = "/content/datasets"
os.makedirs(WORK, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(DATA_HOME, exist_ok=True)

# ---------- Dataset registry: Google Drive folder per dataset ----------
DATASET_REGISTRY = {
    "10x_human_lymph_node_A1": {
        "index": 0,
        "url": "https://drive.google.com/drive/folders/10z1N4MwW8Y49o8GlkYGBKVx1N7fiMuyC",
        "type": "10x",
        "gt_col": "manual-anno",
        "other_file": "adata_ADT.h5ad",
        "anno_file": "annotation.csv"
    },
    "10x_human_lymph_node_D1": {
        "index": 1,
        "url": "https://drive.google.com/drive/folders/1-g_Ca2XMaMXF-MisuVY-wobWDX86O6zz",
        "type": "10x",
        "gt_col": "manual-anno",
        "other_file": "adata_ADT.h5ad",
        "anno_file": "annotation.csv"
    },
    "Mouse_Brain_E11_S1": {
        "index": 2,
        "url": "https://drive.google.com/drive/folders/1zRwDJrYnks0LRzlAVRqPU7jE_OcStgPo",
        "type": "Spatial-epigenome-transcriptome",
        "gt_col": "cluster",
        "other_file": "adata_ATAC.h5ad",
        "anno_file": "anno.csv"
    },
    "Mouse_Brain_E13_S1": {
        "index": 3,
        "url": "https://drive.google.com/drive/folders/1GOufwIRjjfcd9Bi2GKtebzKoPCg2jVud",
        "type": "Spatial-epigenome-transcriptome",
        "gt_col": "cluster",
        "other_file": "adata_ATAC.h5ad",
        "anno_file": "anno.csv"
    },
    "Mouse_Brain_E15_S1": {
        "index": 4,
        "url": "https://drive.google.com/drive/folders/1rHkTL5OF5qPsEERypRGMS51SjUQ69tdD",
        "type": "Spatial-epigenome-transcriptome",
        "gt_col": "cluster",
        "other_file": "adata_ATAC.h5ad",
        "anno_file": "anno.csv"
    },
    "Mouse_Brain_E18_S1": {
        "index": 5,
        "url": "https://drive.google.com/drive/folders/1Xj1LNIAY93biS6JIMKNRODn5GvtCKADB",
        "type": "Spatial-epigenome-transcriptome",
        "gt_col": "cluster",
        "other_file": "adata_ATAC.h5ad",
        "anno_file": "anno.csv"
    },
}

# ---------- dataset manager: find locally, else download the Drive folder ----------
import glob as _glob
def _find_dataset_dir(name):
    info = DATASET_REGISTRY[name]
    need = ["adata_RNA.h5ad", info["other_file"], info["anno_file"]]
    for cand in _glob.glob(f"/content/**/{name}", recursive=True):
        if os.path.isdir(cand) and all(os.path.exists(os.path.join(cand, f)) for f in need):
            return cand
    # fallback: any directory under /content holding all three required files
    for rna in _glob.glob("/content/**/adata_RNA.h5ad", recursive=True):
        cand = os.path.dirname(rna)
        if all(os.path.exists(os.path.join(cand, f)) for f in need):
            return cand
    return None

def ensure_dataset(name):
    found = _find_dataset_dir(name)
    if found:
        print(f"{name}: already present at {found}")
        return found
    info = DATASET_REGISTRY[name]
    print(f"{name}: downloading Drive folder {info['url']} ...")
    r = subprocess.run(["gdown", "--folder", info["url"], "-O", DATA_HOME],
                       capture_output=True, text=True)
    if r.returncode != 0:
        r = subprocess.run([sys.executable, "-m", "gdown", "--folder", info["url"], "-O", DATA_HOME],
                           capture_output=True, text=True)
    print((r.stdout or "")[-500:])
    if r.returncode != 0:
        print((r.stderr or "")[-500:])
    found = _find_dataset_dir(name)
    assert found, (f"{name}: download did not produce the required files - if gdown reported a "
                   "permission error, the Drive folder must be shared as 'anyone with the link'")
    print(f"{name}: ready at {found}")
    return found

SEC_DIR = ensure_dataset(SECTION)
_info = DATASET_REGISTRY[SECTION]
RNA_PATH = os.path.join(SEC_DIR, "adata_RNA.h5ad")
ADT_PATH = os.path.join(SEC_DIR, _info["other_file"])
ANNO_PATH = os.path.join(SEC_DIR, _info["anno_file"])
for p in (RNA_PATH, ADT_PATH, ANNO_PATH):
    assert os.path.exists(p), f"missing input file: {p}"
print("inputs:", SEC_DIR)

# ---------- download the verified Muse-GCN seed-42 embedding + labels ----------
import urllib.request
for name in (EMB_NAME, LAB_NAME):
    dst = os.path.join(WORK, name)
    if not os.path.exists(dst):
        urllib.request.urlretrieve(f"{RAW_BASE}/{name}", dst)
    print(name, os.path.getsize(dst), "bytes")
E = np.load(os.path.join(WORK, EMB_NAME))
C = np.load(os.path.join(WORK, LAB_NAME))
print("E:", E.shape, E.dtype, "| C:", C.shape, C.dtype, "| domains:", np.unique(C))
assert E.shape[0] == C.shape[0], "embedding rows != label count"
"""

PUSH_COLAB = """# ---------- OPTIONAL push of outputs to GitHub (runs/smoke-hello-pipeline) ----------
# Token: Colab Secrets entry GITHUB_TOKEN, or a hidden prompt. Any failure
# here only warns - the smoke outputs stay in /content/smoke_outputs and the
# next cell zips them for download either way.
import subprocess, base64, shutil, getpass
try:
    _tok = None
    try:
        from google.colab import userdata
        _tok = userdata.get("GITHUB_TOKEN")
    except Exception:
        pass
    if not _tok:
        _tok = getpass.getpass("GITHUB_TOKEN (hidden, Enter to skip): ").strip() or None
    assert _tok, "no GITHUB_TOKEN provided"
    _hdr = "Authorization: Basic " + base64.b64encode(f"x-access-token:{_tok}".encode()).decode()
    _GIT = ["git", "-c", f"http.extraHeader={_hdr}"]
    _REPO = "/content/my_research_push"
    if os.path.isdir(_REPO):
        shutil.rmtree(_REPO)
    subprocess.run(_GIT + ["clone", "--depth", "1", "--filter=blob:none", "--sparse",
                           "https://github.com/rifatahsanpul0k/my_research.git", _REPO],
                   check=True, capture_output=True, text=True)
    subprocess.run(["git", "sparse-checkout", "set", "runs/smoke-hello-pipeline"],
                   cwd=_REPO, check=True, capture_output=True, text=True)
    _dst = os.path.join(_REPO, "runs", "smoke-hello-pipeline")
    os.makedirs(_dst, exist_ok=True)
    for fn in os.listdir(OUTDIR):
        shutil.copy(os.path.join(OUTDIR, fn), os.path.join(_dst, fn))
    def _sh(*a):
        return subprocess.run(list(a), cwd=_REPO, capture_output=True, text=True)
    _sh("git", "config", "user.name", "colab-runner")
    _sh("git", "config", "user.email", "colab-runner@local")
    _sh("git", "add", "runs/smoke-hello-pipeline")
    if _sh("git", "diff", "--cached", "--quiet").returncode != 0:
        _sh("git", "commit", "-m", "smoke: hello-pipeline A1 outputs from Colab (dummy externals, n_perm=10)")
        subprocess.run(_GIT + ["pull", "--rebase", "origin", "main"], cwd=_REPO, capture_output=True, text=True)
        _r = subprocess.run(_GIT + ["push", "origin", "main"], cwd=_REPO, capture_output=True, text=True)
        print("pushed" if _r.returncode == 0 else "push failed: " + _r.stderr[:300])
    else:
        print("nothing new to push")
except Exception as e:
    print("push skipped/failed (outputs are safe in smoke_outputs):", str(e)[:300])
"""

ZIP_CELL = """# ---------- Download the outputs (Colab deletes /content when the runtime ends) ----------
import shutil as _shutil
_zip = "/content/smoke_outputs.zip"
if os.path.exists(_zip):
    os.remove(_zip)
_shutil.make_archive("/content/smoke_outputs", "zip", OUTDIR)
print("zipped:", _zip, os.path.getsize(_zip), "bytes")
try:
    from google.colab import files as _cf
    _cf.download(_zip)
except Exception as e:
    print("auto-download not available; download smoke_outputs.zip from the Files panel:", str(e)[:200])
"""


DEDUPE_FIX = """

# ---------- DEDUPE FIX (folded in 2026-11-10; was a hand-run patch cell) ----------
# The A1/D1 gene panel carries 3 duplicated symbols (TBCE, HSPA14, TMSB15B),
# so the dummy Series/DataFrame above have duplicated index labels. A lookup
# by a duplicated label returns a Series instead of a scalar, which crashed
# Engine 1 in the first live run ("truth value of a Series is ambiguous").
# Drop duplicated index entries (keep first) so every lookup returns a scalar.
magma_z_dict = {t: s[~s.index.duplicated(keep="first")] for t, s in magma_z_dict.items()}
gene_covariates = gene_covariates[~gene_covariates.index.duplicated(keep="first")]
print("deduped dummy externals:", len(magma_z_dict), "traits | covariates:", gene_covariates.shape)
"""


def _set(cell, text):
    cell["source"] = text.splitlines(keepends=True)

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(here, "..", "notebooks", "hello_pipeline_smoke_A1.ipynb")
    with open(src_path) as f:
        nb = json.load(f)
    cells = nb["cells"]
    assert len(cells) == 10, f"base notebook should have 10 cells, got {len(cells)}"
    _set(cells[0], MD_COLAB)
    _set(cells[1], INSTALL_COLAB)
    _set(cells[2], CONFIG_COLAB)
    _dummy = "".join(cells[6]["source"])
    assert "generate_dummy_external_data(GENES, seed=SEED)" in _dummy, "cell 6 is not the dummy-data cell"
    assert "duplicated(keep=" not in _dummy, "dedupe fix already applied to base cell 6"
    _set(cells[6], _dummy + DEDUPE_FIX)
    _set(cells[9], PUSH_COLAB)
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": ZIP_CELL.splitlines(keepends=True)})
    for i, c in enumerate(cells):
        if c["cell_type"] == "code":
            compile("".join(c["source"]), f"cell{i}", "exec")
    out_path = os.path.join(here, "..", "notebooks", "hello_pipeline_smoke_A1_colab.ipynb")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path, os.path.getsize(out_path), "bytes,", len(cells), "cells")

    # Content verification (AGENTS.md lesson: verify the OUTPUT notebook)
    import nbformat
    nbformat.read(out_path, as_version=4)
    text = open(out_path).read()
    for token in ["/content/smoke_outputs", "DATASET_REGISTRY", "ensure_dataset",
                  "gdown", "10z1N4MwW8Y49o8GlkYGBKVx1N7fiMuyC",
                  "1Xj1LNIAY93biS6JIMKNRODn5GvtCKADB",
                  "userdata", "smoke_outputs.zip", "n_perm_initial=SMOKE_N_PERM",
                  "run_single_lymph_node_section", "generate_dummy_external_data",
                  "deduped dummy externals", "TBCE, HSPA14, TMSB15B"]:
        assert token in text, f"MISSING TOKEN: {token}"
    assert "kaggle" not in text.lower(), "a Kaggle reference leaked into the Colab-only notebook"
    print("content check: Drive/Colab tokens present, zero Kaggle references")

if __name__ == "__main__":
    main()
