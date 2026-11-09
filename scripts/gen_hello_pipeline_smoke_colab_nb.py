"""Generate notebooks/hello_pipeline_smoke_A1_colab.ipynb

Colab variant of the hello-pipeline SMOKE TEST (A1 only). The pipeline
cells are NOT duplicated here: this generator loads the Kaggle notebook
(notebooks/hello_pipeline_smoke_A1.ipynb, same repo) and replaces only
the platform-specific cells:

  cell 0  notes        -> Colab instructions (Kaggle API dataset download,
                          Colab Secrets, /content lifetime warning)
  cell 1  install      -> same installs (+ json import for the config cell)
  cell 2  config       -> /content paths; dataset found in /content or a
                          mounted Drive, else downloaded via the Kaggle API
                          (Colab Secrets KAGGLE_USERNAME/KAGGLE_KEY or an
                          uploaded kaggle.json)
  cell 9  push         -> GITHUB_TOKEN from Colab Secrets or a hidden
                          getpass prompt; clone under /content
  cell 10 (appended)   -> zip /content/smoke_outputs and download it

Cells 3-8 (prep, verbatim hello.md main code, driver with the single
n_perm edit, dummy external data, run, save) are carried over untouched.

Run: python3 scripts/gen_hello_pipeline_smoke_colab_nb.py
"""
import json, os

MD_COLAB = """# Hello Pipeline — SMOKE TEST (A1 only, Google Colab)

**Purpose: crash-test only.** This is the Colab version of `hello_pipeline_smoke_A1.ipynb` — identical pipeline code from `hello.md` (Stages 1–5 + driver), identical dummy external data, identical smoke settings. **No number that comes out of this run is science:**

- **MAGMA z-scores: random dummy** (seeded) for all 15 panel traits.
- **Gene covariates: random dummy** (gene_length, snp_density).
- **Gene annotation (chrom/start/end/biotype): DUMMY** — random chromosomes/positions, all `protein_coding`. Consequence: the −MHC track is almost identical to +MHC (dummy positions sit far from the chr6:25–34 Mb window).
- **S-LDSC: not present** — lookups are `None`, so Engine 2 uses its internal OLS-proxy fallback (as-is, per plan).
- **Permutations: n_perm = 10** (not 1,000). Consequence: the smallest possible spin p-value is 1/11 ≈ 0.091, so **no hit can reach "Confirmed" in this run by construction** — expected, not a failure.

**Real inputs (verified):** Muse-GCN embeddings for A1, seed 42 — `10x_human_lymph_node_A1_seed42_embeddings.npy` (3484×64) + `..._labels.npy` (3484 values, 10 domains 0–9, BIRCH) — downloaded from the GitHub repo `embeddings/` folder inside the notebook. RNA counts come from `adata.X`, exactly as the driver is written.

**How the dataset gets in (read before running):** Colab has no Kaggle "Add data", so the config cell (1) first searches `/content` (and a mounted Drive) for the `10x_human_lymph_node_A1` folder, and (2) if it is not there, downloads your private dataset `pulokpulok/spatial-multiomics-6datasets-private` (~788 MB) with the Kaggle API. For that download you need Kaggle API credentials in the session — either add `KAGGLE_USERNAME` and `KAGGLE_KEY` as Colab Secrets (key icon in the left sidebar), or upload your `kaggle.json` when the cell prompts you (get it from kaggle.com → Settings → API → Create New Token). The credentials are used only for that download.

**GitHub push (optional):** the push cell reads `GITHUB_TOKEN` from Colab Secrets, or asks with a hidden prompt. Without a token the run still completes — outputs stay in `/content/smoke_outputs/` and the last cell zips them for download. **Download or push before closing the tab: Colab deletes `/content` when the runtime ends.**

**Runtime:** CPU runtime is fine (Runtime → Change runtime type → CPU; the code is NumPy, a GPU changes nothing). Estimate ~15–45 min for the pipeline, plus a few minutes for the dataset download.

**Success =** the run cell finishes, `smoke_outputs/` contains `A1_results_df.csv` + `A1_summary.json`, and tier counts print. If any cell errors, copy the full traceback.
"""

CONFIG_COLAB = """# ---------- CONFIG (smoke test, locked 2026-11-09) - Colab version ----------
SEED = 42
SMOKE_N_PERM = 10            # smoke only; real run = 1000 (and 10000 recalibration)
SECTION = "10x_human_lymph_node_A1"
EMB_NAME = "10x_human_lymph_node_A1_seed42_embeddings.npy"
LAB_NAME = "10x_human_lymph_node_A1_seed42_labels.npy"
RAW_BASE = "https://raw.githubusercontent.com/rifatahsanpul0k/my_research/main/embeddings"
WORK = "/content/smoke_inputs"
OUTDIR = "/content/smoke_outputs"
os.makedirs(WORK, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

# ---------- locate the A1 folder: /content, mounted Drive, else Kaggle API ----------
import glob as _glob
def find_data_root():
    for base in ["/content", "/content/drive/MyDrive", "/content/datasets"]:
        if os.path.isdir(os.path.join(base, SECTION)):
            return base
    hits = [os.path.dirname(os.path.dirname(p)) for p in
            _glob.glob(f"/content/**/{SECTION}/adata_RNA.h5ad", recursive=True)]
    return hits[0] if hits else None

DATA_ROOT = find_data_root()
if DATA_ROOT is None:
    print("dataset not found locally - downloading from Kaggle API ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)
    _kag_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(_kag_dir, exist_ok=True)
    if not os.path.exists(os.path.join(_kag_dir, "kaggle.json")):
        _got = False
        try:  # Option A: Colab Secrets KAGGLE_USERNAME / KAGGLE_KEY
            from google.colab import userdata
            _u = userdata.get("KAGGLE_USERNAME"); _k = userdata.get("KAGGLE_KEY")
            if _u and _k:
                with open(os.path.join(_kag_dir, "kaggle.json"), "w") as _f:
                    json.dump({"username": _u, "key": _k}, _f)
                _got = True
                print("Kaggle credentials: from Colab Secrets")
        except Exception:
            pass
        if not _got:  # Option B: upload kaggle.json when prompted
            from google.colab import files as _colab_files
            print("Upload your kaggle.json (kaggle.com -> Settings -> API -> Create New Token)")
            _up = _colab_files.upload()
            assert "kaggle.json" in _up, "kaggle.json was not uploaded"
            with open(os.path.join(_kag_dir, "kaggle.json"), "wb") as _f:
                _f.write(_up["kaggle.json"])
            print("Kaggle credentials: from uploaded kaggle.json")
        os.chmod(os.path.join(_kag_dir, "kaggle.json"), 0o600)
    subprocess.run(["kaggle", "datasets", "download", "-d",
                    "pulokpulok/spatial-multiomics-6datasets-private",
                    "-p", "/content/datasets", "--unzip"], check=True)
    DATA_ROOT = find_data_root()
assert DATA_ROOT, "A1 folder still not found after the Kaggle download - check the download output above"
SEC_DIR = os.path.join(DATA_ROOT, SECTION)
RNA_PATH = os.path.join(SEC_DIR, "adata_RNA.h5ad")
ADT_PATH = os.path.join(SEC_DIR, "adata_ADT.h5ad")
ANNO_PATH = os.path.join(SEC_DIR, "annotation.csv")
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

INSTALL_COLAB = """import os, sys, json, subprocess
for pkg in ["scanpy", "statsmodels"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
import numpy as np
import pandas as pd
import scanpy as sc
print("imports ok; numpy", np.__version__, "| pandas", pd.__version__)
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


def _set(cell, text):
    cell["source"] = text.splitlines(keepends=True)

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(here, "..", "notebooks", "hello_pipeline_smoke_A1.ipynb")
    with open(src_path) as f:
        nb = json.load(f)
    cells = nb["cells"]
    assert len(cells) == 10, f"Kaggle source notebook should have 10 cells, got {len(cells)}"
    _set(cells[0], MD_COLAB)
    _set(cells[1], INSTALL_COLAB)
    _set(cells[2], CONFIG_COLAB)
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
    for token in ["/content/smoke_outputs", "kaggle.json", "KAGGLE_USERNAME",
                  "userdata", "smoke_outputs.zip", "n_perm_initial=SMOKE_N_PERM",
                  "run_single_lymph_node_section", "generate_dummy_external_data"]:
        assert token in text, f"MISSING TOKEN: {token}"
    for bad in ["/kaggle/working", "/kaggle/input", "kaggle_secrets", "UserSecretsClient"]:
        assert bad not in text, f"Kaggle-only remnant in Colab notebook: {bad}"
    print("content check: Colab tokens present, no Kaggle-only remnants")

if __name__ == "__main__":
    main()
