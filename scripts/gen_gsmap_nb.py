"""Generate notebooks/stage7_disease_gsmap.ipynb — Disease association from the frozen embedding (gsMap swap).

Design (per the selected pipeline, 2026-10-31 triage):
- SKIP gsMap Step 1 (find_latent_representations): our frozen S5N-BASE fused
  embedding goes into obsm['fused'] and is passed via --latent_representation.
- KEEP: run_latent_to_gene -> run_generate_ldscore -> run_spatial_ldsc ->
  run_cauchy_combination (annotation = our domains).
- PILOT: A1 only. Traits = Height (negative control) + IQ, both ready-formatted
  in the official gsMap example bundle (no sumstats formatting risk in pilot).
  SLE etc. come in Phase 2 via gsmap format_sumstats (template cell included).
- Memory guard: --spots_per_chunk 250 (docs: ~40GB at 1000; Kaggle RAM is lower,
  and A1 has only ~3.5k spots).
- Alignment guard: domains recomputed by the fixed recipe (standard-scale +
  kmeans, declared k=10) on the loaded embedding; post-hoc ARI vs annotation is
  printed as a plumbing check (expect ~0.22, Exp0 BASE/A1). Labels never select
  anything; the recipe runs regardless.

CLI facts verified from the official gsMap docs (2026-11-01):
- binary is lowercase `gsmap`; install: pip install gsMap (python>=3.10)
- resources: https://yanglab.westlake.edu.cn/data/gsMap/gsMap_resource.tar.gz
- examples:  https://yanglab.westlake.edu.cn/data/gsMap/gsMap_example_data.tar.gz
- input h5ad: counts in layers['count'], obsm['spatial'], obs annotation column
- run_latent_to_gene has NO --data_layer flag (do not add one)

Run: python3 scripts/gen_gsmap_nb.py
"""
import json, os

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

cells = []

cells.append(md('''# Stage 7 — Disease Association from the Frozen Embedding (gsMap swap) — PILOT: A1

**Design:** gsMap (Song et al., *Nature* 2025) normally trains its own 32-d GAT
embedding in Step 1. **We skip Step 1 entirely** and supply OUR frozen fused
embedding instead: it is stored in `obsm['fused']` and passed to
`run_latent_to_gene` via `--latent_representation fused` (the documented
custom-latent route). Everything downstream (microdomains → GSS → SNP
annotation → per-spot S-LDSC → Cauchy per domain) is gsMap's own machinery,
unchanged.

**Frozen inputs:** S5N-BASE fused embedding for A1 (seed 42) + domains from the
fixed one-pipeline recipe (standard-scale → k-means, declared k=10). The
embedding and domains are never retrained or re-selected using disease data.

**Pilot traits (both ready-formatted in the official example bundle):**
- **Height** — NEGATIVE CONTROL. Expect: no domain significant. If height
  lights up, the harness is broken, not the biology.
- **IQ** — pipeline proof: a real trait producing per-spot maps + a domain table.
Phase 2 (SLE and the rest of the pre-registered panel) uses
`gsmap format_sumstats` — template at the end of this notebook.

**Output:** `runs/s7-disease/` (Cauchy domain × trait tables + run log).
**Watch out:** gsMap's documented memory is ~40 GB for the LD steps at
`--spots_per_chunk 1000`; we use 250 (A1 has only ~3.5k spots).
'''))

cells.append(code('''import os, sys, json, subprocess, time

for pkg in ["gsMap", "scanpy", "anndata", "scikit-learn", "pandas", "scipy"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

r = subprocess.run(["gsmap", "-v"], capture_output=True, text=True)
print("gsmap version:", (r.stdout or r.stderr).strip()[:100])

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

print("imports ok")
'''))

cells.append(code('''# ---------- GitHub: clone (for the saved embedding) + push helper ----------
REPO = "/kaggle/working/my_research"
REPO_URL = "https://github.com/rifatahsanpul0k/my_research.git"
def sh(*args):
    return subprocess.run(list(args), capture_output=True, text=True, cwd=REPO)
os.makedirs(REPO, exist_ok=True)
OUT = os.path.join(REPO, "runs", "s7-disease")
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
        sh("git", "add", "runs/s7-disease")
        r = sh("git", "diff", "--cached", "--quiet")
        if r.returncode == 0:
            print("nothing new to push")
            return
        sh("git", "commit", "-m", msg)
        subprocess.run(_GIT + ["pull", "--rebase", "origin", "main"],
                       capture_output=True, text=True, cwd=REPO)
        subprocess.run(_GIT + ["push", "origin", "main"],
                       capture_output=True, text=True, cwd=REPO)
        print("pushed")
    except Exception as e:
        print("push failed:", str(e)[:300])
'''))

cells.append(code('''# ---------- Load A1 data + frozen BASE embedding; recompute domains ----------
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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)
    subprocess.run(["kaggle", "datasets", "download", "-d",
                    "pulokpulok/spatial-multiomics-6datasets-private",
                    "-p", "/kaggle/working", "--unzip"], check=True)
    DATA_ROOT = find_data_root()
print("DATA_ROOT:", DATA_ROOT)

root = os.path.join(DATA_ROOT, MARKER)
adata = sc.read_h5ad(os.path.join(root, "adata_RNA.h5ad"))
adata.var_names_make_unique()
print("adata_RNA:", adata.shape, "| obsm keys:", list(adata.obsm.keys()))
assert "spatial" in adata.obsm, "obsm['spatial'] missing — cannot package for gsMap"

# Discover the saved BASE embedding (seed dir name discovered, never assumed)
base_dir = os.path.join(REPO, "runs", "s5-notebook", "S5N-BASE", "A1")
seed_dirs = [d for d in sorted(os.listdir(base_dir))
             if os.path.exists(os.path.join(base_dir, d, "fused_embedding.npy"))]
print("seed dirs with embeddings:", seed_dirs)
seed_dir = next((d for d in seed_dirs if "42" in d), seed_dirs[0])
E = np.load(os.path.join(base_dir, seed_dir, "fused_embedding.npy")).astype(np.float32)
print(f"embedding: {E.shape} from {seed_dir}")
assert E.shape[0] == adata.shape[0], "row mismatch: embedding vs adata — ALIGNMENT BROKEN"

# Domains by the FIXED one-pipeline recipe (standard-scale + kmeans, declared k)
K_A1 = 10
Ez = StandardScaler().fit_transform(E)
domains = KMeans(n_clusters=K_A1, n_init=10, random_state=42).fit_predict(Ez)
print("domain sizes:", np.bincount(domains))

# Plumbing check ONLY (post-hoc, quarantined): ARI vs manual annotation.
# Exp0 reported BASE/A1 standard/kmeans ~ 0.221. A wildly different value here
# means the embedding rows do not align with the adata rows. Labels select nothing.
gt = pd.read_csv(os.path.join(root, "annotation.csv"), index_col=0)
ari = adjusted_rand_score(np.asarray(gt["manual-anno"]).astype(str), domains)
print(f"ALIGNMENT CHECK — post-hoc ARI = {ari:.3f} (expect ~0.22)")
assert ari > 0.10, "ARI far below expected — likely row misalignment. STOP."
'''))

cells.append(code('''# ---------- Package the gsMap input h5ad ----------
# gsMap format (official docs): counts in layers['count'], coords in
# obsm['spatial'], annotation as an obs column, latent in obsm under our key.
pkg = adata.copy()
counts = pkg.X.toarray() if hasattr(pkg.X, "toarray") else np.asarray(pkg.X)
pkg.X = counts.astype(np.float32)
pkg.layers["count"] = counts.astype(np.float32)
pkg.obsm["fused"] = E
pkg.obs["domain"] = pd.Categorical([f"D{d}" for d in domains])

GSMAP_DIR = "/kaggle/working/gsmap"
os.makedirs(GSMAP_DIR, exist_ok=True)
H5AD = os.path.join(GSMAP_DIR, "A1_input.h5ad")
pkg.write_h5ad(H5AD)
print("wrote", H5AD)
print("layers:", list(pkg.layers.keys()), "| obsm:", list(pkg.obsm.keys()),
      "| obs domain counts:", pkg.obs["domain"].value_counts().to_dict())
'''))

cells.append(code('''# ---------- Download gsMap resources + example GWAS ----------
def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd)}")
    return r

def fetch(url, dest, chunk=32 * 1024 * 1024, tries=6):
    # Chunked Range download. One long stream to this server keeps dying:
    # wget truncated silently, and curl -C - resume produced corrupt files.
    # Each 32 MB chunk is a short request, retried on its own, appended in
    # order. Re-running the cell resumes — but only from a prefix THIS
    # function wrote (marker file); foreign partials are discarded.
    import urllib.request
    def _req(range_hdr=None):
        h = {"User-Agent": "Mozilla/5.0"}
        if range_hdr:
            h["Range"] = range_hdr
        return urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(_req(), timeout=60) as r:
        total = int(r.headers.get("Content-Length") or 0)
    print(f"  {dest}: expected {total/1e9:.2f} GB", flush=True)
    assert total > 0, "server gave no Content-Length"
    marker = dest + ".chunked"
    if os.path.exists(dest) and not os.path.exists(marker):
        print("  discarding partial file from an earlier download method")
        os.remove(dest)
    open(marker, "a").close()
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if have > total:
        os.remove(dest); have = 0
    with open(dest, "ab") as f:
        pos = have
        while pos < total:
            end = min(pos + chunk, total) - 1
            data = None
            for a in range(tries):
                try:
                    with urllib.request.urlopen(_req(f"bytes={pos}-{end}"),
                                                timeout=180) as resp:
                        if resp.status == 200:
                            raise IOError("server ignored the Range header")
                        data = resp.read()
                    if len(data) == end - pos + 1:
                        break
                    print(f"  chunk @{pos/1e9:.2f} GB short read, retry", flush=True)
                    data = None
                except Exception as e:
                    print(f"  chunk @{pos/1e9:.2f} GB failed ({str(e)[:80]}), "
                          f"retry {a+1}/{tries}", flush=True)
                    data = None
            if data is None:
                raise RuntimeError(f"chunk at byte {pos} failed {tries} times")
            f.write(data); f.flush()
            pos += len(data)
            if (pos // chunk) % 10 < 1:
                print(f"  {pos/1e9:.2f} / {total/1e9:.2f} GB", flush=True)
    ok = subprocess.run(["gzip", "-t", dest]).returncode == 0
    print(f"  gzip integrity: {'OK' if ok else 'BAD'}", flush=True)
    if not ok:
        os.remove(dest); os.remove(marker)
        raise RuntimeError("archive corrupt even after chunked download: " + dest)
    os.remove(marker)

os.chdir(GSMAP_DIR)
print(subprocess.run(["df", "-h", GSMAP_DIR], capture_output=True, text=True).stdout)
if not os.path.isdir("gsMap_resource"):
    fetch("https://yanglab.westlake.edu.cn/data/gsMap/gsMap_resource.tar.gz",
          "gsMap_resource.tar.gz")
    run(["tar", "-xzf", "gsMap_resource.tar.gz"])
if not os.path.isdir("gsMap_example_data"):
    fetch("https://yanglab.westlake.edu.cn/data/gsMap/gsMap_example_data.tar.gz",
          "gsMap_example_data.tar.gz")
    run(["tar", "-xzf", "gsMap_example_data.tar.gz"])

RES = os.path.join(GSMAP_DIR, "gsMap_resource")
checks = {
    "bfile bim":  f"{RES}/LD_Reference_Panel/1000G_EUR_Phase3_plink/1000G.EUR.QC.bim",
    "gtf":        f"{RES}/genome_annotation/gtf/gencode.v46lift37.basic.annotation.gtf",
    "weights":    f"{RES}/LDSC_resource/weights_hm3_no_hla/weights.1.l2.ldscore.gz",
    "height gwas": f"{GSMAP_DIR}/gsMap_example_data/GWAS/GIANT_EUR_Height_2022_Nature.sumstats.gz",
    "iq gwas":     f"{GSMAP_DIR}/gsMap_example_data/GWAS/IQ_NG_2018.sumstats.gz",
}
for name, p in checks.items():
    print(f"{name:12} {'OK' if os.path.exists(p) else 'MISSING -> ' + p}")
'''))

cells.append(code('''# ---------- gsMap Step 2: latent_to_gene with OUR embedding ----------
# (Their Step 1 — training a GAT embedding — is SKIPPED by design.)
WD = os.path.join(GSMAP_DIR, "workdir")
os.makedirs(WD, exist_ok=True)
t0 = time.time()
run(["gsmap", "run_latent_to_gene",
     "--workdir", WD,
     "--sample_name", "A1",
     "--input_hdf5_path", H5AD,
     "--annotation", "domain",
     "--latent_representation", "fused"])
print(f"latent_to_gene done in {(time.time()-t0)/60:.1f} min")
'''))

cells.append(code('''# ---------- gsMap Step 3: generate LD scores (per chromosome) ----------
# spots_per_chunk 250 (docs: ~40GB at 1000; Kaggle RAM is smaller, A1 is small)
t0 = time.time()
for chrom in range(1, 23):
    tc = time.time()
    run(["gsmap", "run_generate_ldscore",
         "--workdir", WD,
         "--sample_name", "A1",
         "--chrom", str(chrom),
         "--bfile_root", f"{RES}/LD_Reference_Panel/1000G_EUR_Phase3_plink/1000G.EUR.QC",
         "--keep_snp_root", f"{RES}/LDSC_resource/hapmap3_snps/hm",
         "--gtf_annotation_file", f"{RES}/genome_annotation/gtf/gencode.v46lift37.basic.annotation.gtf",
         "--gene_window_size", "50000",
         "--spots_per_chunk", "250"])
    print(f"  chrom {chrom} done in {(time.time()-tc)/60:.1f} min", flush=True)
print(f"ALL LD scores done in {(time.time()-t0)/60:.1f} min")
'''))

cells.append(code('''# ---------- gsMap Step 4: per-spot spatial LDSC (Height control, then IQ) ----------
TRAITS = {
    "Height": f"{GSMAP_DIR}/gsMap_example_data/GWAS/GIANT_EUR_Height_2022_Nature.sumstats.gz",
    "IQ":     f"{GSMAP_DIR}/gsMap_example_data/GWAS/IQ_NG_2018.sumstats.gz",
}
for trait, ss in TRAITS.items():
    t0 = time.time()
    run(["gsmap", "run_spatial_ldsc",
         "--workdir", WD,
         "--sample_name", "A1",
         "--trait_name", trait,
         "--sumstats_file", ss,
         "--w_file", f"{RES}/LDSC_resource/weights_hm3_no_hla/weights.",
         "--num_processes", "4"])
    print(f"spatial_ldsc [{trait}] done in {(time.time()-t0)/60:.1f} min", flush=True)
'''))

cells.append(code('''# ---------- gsMap Step 5: Cauchy combination per domain ----------
for trait in TRAITS:
    run(["gsmap", "run_cauchy_combination",
         "--workdir", WD,
         "--sample_name", "A1",
         "--trait_name", trait,
         "--annotation", "domain"])
print("cauchy done")

# Discover + collect outputs (exact filenames are workdir-internal; glob them)
import glob
found = sorted(glob.glob(os.path.join(WD, "**", "*auchy*"), recursive=True) +
               glob.glob(os.path.join(WD, "**", "*A1*"), recursive=True))
for f in found:
    print(f)
'''))

cells.append(code('''# ---------- Summarise: domain x trait table -> push to GitHub ----------
import glob, shutil
summary = {"dataset": "A1", "embedding": "S5N-BASE seed42 fused (obsm['fused'])",
           "domains": "standard+kmeans k=10 (fixed one-pipeline recipe)",
           "traits": {}, "notes": "Height = negative control (expect null)"}

cauchy_files = [f for f in glob.glob(os.path.join(WD, "**", "*.csv*"), recursive=True)
                if "auchy" in os.path.basename(f).lower()]
print("cauchy files:", cauchy_files)
for f in cauchy_files:
    try:
        df = pd.read_csv(f)
        print("\\n", os.path.basename(f)); print(df.to_string()[:1500])
        dest = os.path.join(OUT, "A1_" + os.path.basename(f))
        shutil.copy(f, dest)
    except Exception as e:
        print("could not read", f, str(e)[:150])

with open(os.path.join(OUT, "A1_gsmap_pilot_log.json"), "w") as fh:
    json.dump({"cauchy_files": [os.path.basename(f) for f in cauchy_files],
               "alignment_ari_posthoc": float(ari)}, fh, indent=2)

git_push("s7-disease: gsMap pilot A1 (BASE embedding swapped in; Height control + IQ)")
print("\\nPilot done. Read the Height table first: it should be null across domains.")
'''))

cells.append(md('''## Phase 2 — adding the real trait panel (SLE first)

For each new trait, you need sumstats with at least `SNP`, `Z`, `N` columns
(gsMap keywords). Download from the GWAS Catalog, then format and re-run
Steps 4–5 only (LD scores are trait-independent and already computed):

```bash
# 1) format (adjust column-name flags to your file's headers)
gsmap format_sumstats --sumstats <downloaded_file> --out SLE
# -> produces SLE.sumstats.gz

# 2) per-spot test + domain combination
gsmap run_spatial_ldsc --workdir <WD> --sample_name A1 --trait_name SLE \\
    --sumstats_file SLE.sumstats.gz \\
    --w_file gsMap_resource/LDSC_resource/weights_hm3_no_hla/weights. \\
    --num_processes 4
gsmap run_cauchy_combination --workdir <WD> --sample_name A1 \\
    --trait_name SLE --annotation domain
```

Pre-registered panel — lymph node: SLE, RA, MS, IBD, T1D, asthma, Hodgkin,
DLBCL, CLL. Brain datasets additionally need
`--homolog_file gsMap_resource/homologs/mouse_human_homologs.txt` in the
`run_latent_to_gene` call (gene-level only; human SNPs never map to mouse peaks).
'''))

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "cells": cells,
}

out_path = os.path.join(os.path.dirname(__file__), "..", "notebooks", "stage7_disease_gsmap.ipynb")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {out_path} ({len(cells)} cells)")

import nbformat
nbformat.read(out_path, as_version=4)
print("nbformat: valid")

# Content verification (AGENTS.md lesson: verify the OUTPUT notebook)
text = open(out_path).read()
for token in ["run_latent_to_gene", "--latent_representation", "fused",
              "run_generate_ldscore", "spots_per_chunk", "run_spatial_ldsc",
              "run_cauchy_combination", "GIANT_EUR_Height_2022",
              "gencode.v46lift37", "ALIGNMENT CHECK"]:
    assert token in text, f"MISSING TOKEN: {token}"
assert "run_find_latent_representations" not in text.replace(
    "SKIPPED by design", "").replace("Step 1", ""), "Step 1 command leaked in"
print("content check: all key tokens present")
