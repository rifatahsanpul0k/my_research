"""Generate notebooks/stage7_gsmap_exact_A1.ipynb — the EXACT gsMap pipeline,
fed with the user's own uploaded embedding file
(10x_human_lymph_node_A1_seed_42_embeddings.npy, attached to the Kaggle
notebook as an input and found by exact filename under /kaggle/input).

Derived from the v3 generator (2026-11-02) — every battle-tested mechanism
carried over unchanged; the ONLY design delta is the embedding source:
v3 pulled fused_embedding.npy out of the cloned repo's runs/s5-notebook,
this one loads the user-uploaded .npy. Fresh-session linear flow:
install -> locate inputs -> package h5ad -> verified downloads ->
latent_to_gene -> generate_ldscore -> spatial_ldsc -> cauchy -> push.

Original v3 notes follow.

REWRITE v3 (2026-11-02). v2's remaining hole, found live: the download cell
skipped re-downloading whenever the bundle DIRECTORY existed — but run 1's
disk-full crash had left a PARTIAL gsMap_example_data extraction behind, so
the GWAS sumstats were missing and spatial_ldsc died at Step 4 on a
file-not-found. v3 readiness is therefore FILE-based, never folder-based:
  - resource ready  <=>  bim + gtf + weights.1 files exist
  - example ready   <=>  BOTH GWAS sumstats files exist
A stub folder that fails its file check is deleted and re-fetched.

v3 also skips work gsMap itself has already certified:
  - latent_to_gene skipped if A1_gene_marker_score.feather exists
  - the whole LD step skipped if gsMap's own A1_generate_ldscore.done
    marker exists (in the user's current session it does: all 22 chroms x
    14 spot-chunks + baseline + w_ld completed 2026-11-02)
  - Step 4 asserts both sumstats exist BEFORE invoking gsmap

Carried over from v1/v2 (all battle-tested live):
  - chunked Range downloader (32 MB chunks, per-chunk retry, marker-guarded
    resume, gzip -t) — wget truncated silently; curl resume corrupted
  - disk discipline: tarballs deleted after extraction, example ST dropped,
    df printed around heavy steps (20 GB working disk filled in run 1 and
    killed generate_ldscore at chrom 8)
  - run_v(): gsMap calls print gsMap's own last 30 lines on failure
  - LD loop: per-chromosome progress JSON, failed chrom retries at
    --spots_per_chunk 100 (docs quote ~40 GB at 1000; Kaggle RAM smaller)

Design (selected pipeline, 2026-10-31 triage) — unchanged:
- SKIP gsMap Step 1: frozen S5N-BASE fused embedding in obsm['fused'],
  passed via --latent_representation (documented custom-latent route).
- KEEP: latent_to_gene -> generate_ldscore -> spatial_ldsc -> cauchy.
- PILOT: A1; traits Height (negative control) + IQ, ready-formatted in the
  official example bundle. SLE panel = Phase 2 (template cell at end).
- Domains: fixed recipe (standard-scale + kmeans, declared k=10) on the
  loaded embedding; post-hoc ARI printed as alignment check (expect ~0.22).

CLI facts verified from the official gsMap docs (2026-11-01): binary is
lowercase `gsmap`; pip install gsMap (python>=3.10); bundles at
yanglab.westlake.edu.cn/data/gsMap/; input h5ad = layers['count'] +
obsm['spatial'] + obs annotation column; run_latent_to_gene has NO
--data_layer flag.

Run: python3 scripts/gen_gsmap_exact_nb.py
"""
import json, os

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

cells = []

cells.append(md('''# Stage 7 — The EXACT gsMap Pipeline, Run on YOUR Embedding — A1

**Inputs you attach to this notebook (Kaggle → Add Input):**
1. Your embedding file **`10x_human_lymph_node_A1_seed_42_embeddings.npy`**
   (uploaded by you; the notebook finds it by exact filename).
2. The A1 dataset (`spatial-multiomics-6datasets-private`) — gsMap still needs
   the raw counts, gene names and spatial coordinates from `adata_RNA.h5ad`;
   the embedding alone does not contain them.

**Design:** gsMap (Song et al., *Nature* 2025) normally trains its own 32-d GAT
embedding in Step 1. **We skip Step 1 entirely** and supply YOUR embedding
instead: stored in `obsm['fused']`, passed to `run_latent_to_gene`
via `--latent_representation fused` (the documented custom-latent route).
Everything downstream (microdomains → GSS → SNP annotation → per-spot S-LDSC
→ Cauchy per domain) is gsMap's own machinery, unchanged — the exact gsMap.

**Frozen inputs:** your A1 seed-42 embedding + domains from the
fixed one-pipeline recipe (standard-scale → k-means, declared k=10). The
embedding and domains are never retrained or re-selected using disease data.
A post-hoc ARI alignment check (~0.22 expected) verifies the embedding rows
line up with the A1 spots before anything runs.

**Pilot traits (ready-formatted in the official example bundle):**
- **Height** — NEGATIVE CONTROL. Expect: no domain significant.
- **IQ** — pipeline proof: a real trait producing maps + a domain table.
Phase 2 (SLE etc.) uses `gsmap format_sumstats` — template at the end.

**v3 hardening (lessons from runs 1–2):** chunked verified downloads;
readiness judged by required FILES, not folders (a partial extraction from a
disk-full crash once slipped through and killed Step 4); tarballs deleted
immediately after extraction (20 GB disk); completed gsMap steps are skipped
via gsMap's own output markers; every gsMap call prints its own error output
on failure; the chromosome loop checkpoints and resumes.

**Output:** `runs/s7-disease/` (Cauchy domain × trait tables + run log).
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

cells.append(code('''# ---------- GitHub: clone + push helper (results -> runs/s7-disease) ----------
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

cells.append(code('''# ---------- Load A1 data + YOUR uploaded embedding; recompute domains ----------
import glob as _glob
MARKER = "10x_human_lymph_node_A1"
def find_data_root():
    for base in ["/kaggle/input/spatial-multiomics-6datasets-private",
                 "/kaggle/working/spatial-multiomics-6datasets-private",
                 "/kaggle/working"]:
        if os.path.isdir(os.path.join(base, MARKER)):
            return base
    hits = [os.path.dirname(os.path.dirname(p)) for p in
            _glob.glob(f"/kaggle/input/**/{MARKER}/adata_RNA.h5ad", recursive=True)]
    return hits[0] if hits else None

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

# YOUR embedding: found by exact filename wherever Kaggle mounted the upload
EMB_NAME = "10x_human_lymph_node_A1_seed_42_embeddings.npy"
emb_hits = (_glob.glob(f"/kaggle/input/**/{EMB_NAME}", recursive=True) +
            _glob.glob(f"/kaggle/working/**/{EMB_NAME}", recursive=True))
assert emb_hits, (f"{EMB_NAME} not found under /kaggle/input or /kaggle/working — "
                  "attach your uploaded embedding dataset to this notebook (Add Input)")
EMB_PATH = sorted(emb_hits)[0]
print("embedding file:", EMB_PATH)
E = np.load(EMB_PATH).astype(np.float32)
print(f"embedding: {E.shape}")
assert E.ndim == 2, f"embedding must be 2-D (n_spots x dims), got shape {E.shape}"
if E.shape[0] != adata.shape[0] and E.shape[1] == adata.shape[0]:
    print("WARNING: embedding looks transposed — transposing to (n_spots, dims)")
    E = E.T
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

cells.append(code('''# ---------- Download gsMap resources + example GWAS (file-verified) ----------
import glob, shutil

def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd)}")
    return r

def run_v(cmd):
    # Like run(), but captures output and prints gsMap's own last 30 lines on
    # failure — a bare wrapper traceback hid the real error in run 1.
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = "\\n".join((r.stdout + "\\n" + r.stderr).splitlines()[-30:])
        print("----- gsMap output (last 30 lines) -----")
        print(tail)
        print("----------------------------------------")
        raise RuntimeError(f"failed ({r.returncode}): {' '.join(cmd[:2])}")
    return r

def fetch(url, dest, chunk=32 * 1024 * 1024, tries=6):
    # Chunked Range download. One long stream to this server keeps dying:
    # wget truncated silently; curl -C - resume appended to corrupt prefixes.
    # Each 32 MB chunk is a short request, retried on its own, appended in
    # order. Re-running resumes — but only from a prefix THIS function wrote
    # (marker file); foreign partials are discarded. gzip -t verifies before
    # tar ever touches the file.
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

def df():
    print(subprocess.run(["df", "-h", "/kaggle/working"],
                         capture_output=True, text=True).stdout)

# --- Required files (readiness is judged on THESE, never on folders) ---
RES = os.path.join(GSMAP_DIR, "gsMap_resource")
KEY_FILES = {
    "bfile bim":  f"{RES}/LD_Reference_Panel/1000G_EUR_Phase3_plink/1000G.EUR.QC.bim",
    "gtf":        f"{RES}/genome_annotation/gtf/gencode.v46lift37.basic.annotation.gtf",
    "weights":    f"{RES}/LDSC_resource/weights_hm3_no_hla/weights.1.l2.ldscore.gz",
    "height gwas": f"{GSMAP_DIR}/gsMap_example_data/GWAS/GIANT_EUR_Height_2022_Nature.sumstats.gz",
    "iq gwas":     f"{GSMAP_DIR}/gsMap_example_data/GWAS/IQ_NG_2018.sumstats.gz",
}
def resource_ready():
    return all(os.path.exists(KEY_FILES[k]) for k in ["bfile bim", "gtf", "weights"])
def example_ready():
    return all(os.path.exists(KEY_FILES[k]) for k in ["height gwas", "iq gwas"])

os.chdir(GSMAP_DIR)
df()
if not resource_ready():
    if os.path.isdir("gsMap_resource"):
        print("gsMap_resource exists but key files are missing — removing stub")
        shutil.rmtree("gsMap_resource")
    fetch("https://yanglab.westlake.edu.cn/data/gsMap/gsMap_resource.tar.gz",
          "gsMap_resource.tar.gz")
    run(["tar", "-xzf", "gsMap_resource.tar.gz"])
    os.remove("gsMap_resource.tar.gz")  # disk discipline: 20 GB working dir
    assert resource_ready(), "resource bundle still incomplete after extraction"
if not example_ready():
    if os.path.isdir("gsMap_example_data"):
        print("gsMap_example_data exists but GWAS files are missing — removing stub")
        shutil.rmtree("gsMap_example_data")
    fetch("https://yanglab.westlake.edu.cn/data/gsMap/gsMap_example_data.tar.gz",
          "gsMap_example_data.tar.gz")
    run(["tar", "-xzf", "gsMap_example_data.tar.gz"])
    os.remove("gsMap_example_data.tar.gz")
    assert example_ready(), "example bundle still incomplete after extraction"
# Only the GWAS folder of the example bundle is needed — drop the example ST data
if os.path.isdir("gsMap_example_data/ST"):
    shutil.rmtree("gsMap_example_data/ST")
    print("dropped example ST data (not needed)")
for leftover in glob.glob("*.tar.gz"):
    os.remove(leftover)
    print("removed leftover archive:", leftover)
df()

for name, p in KEY_FILES.items():
    print(f"{name:12} {'OK' if os.path.exists(p) else 'MISSING -> ' + p}")
assert resource_ready() and example_ready(), "resource check failed — see MISSING lines"
print("all resources verified")
'''))

cells.append(code('''# ---------- gsMap Step 2: latent_to_gene with OUR embedding ----------
# (Their Step 1 — training a GAT embedding — is SKIPPED by design.)
WD = os.path.join(GSMAP_DIR, "workdir")
os.makedirs(WD, exist_ok=True)
MKSCORE = os.path.join(WD, "A1", "latent_to_gene", "A1_gene_marker_score.feather")
if os.path.exists(MKSCORE):
    print("mkscore feather found — latent_to_gene already done, skipping")
else:
    t0 = time.time()
    run_v(["gsmap", "run_latent_to_gene",
           "--workdir", WD,
           "--sample_name", "A1",
           "--input_hdf5_path", H5AD,
           "--annotation", "domain",
           "--latent_representation", "fused"])
    print(f"latent_to_gene done in {(time.time()-t0)/60:.1f} min")
'''))

cells.append(code('''# ---------- gsMap Step 3: generate LD scores (resumable, per chromosome) ----------
# Skipped entirely when gsMap's own .done marker exists (it writes one only
# after ALL chromosomes finish). Otherwise a progress JSON checkpoints each
# finished chromosome, and a failed chromosome retries once at 100
# spots/chunk (memory guard; docs quote ~40 GB at 1000, A1 has ~3.5k spots).
DONE_MARKER = os.path.join(WD, "A1", "generate_ldscore", "A1_generate_ldscore.done")
if os.path.exists(DONE_MARKER):
    print("gsMap .done marker found — LD scores already complete, skipping Step 3")
else:
    assert os.path.exists(MKSCORE), "mkscore feather missing — Step 2 did not finish"
    PROG = os.path.join(GSMAP_DIR, "ldscore_progress.json")
    done = set(json.load(open(PROG))) if os.path.exists(PROG) else set()
    print("chromosomes already done:", sorted(done))
    df()

    def ldscore_cmd(chrom, spc):
        return ["gsmap", "run_generate_ldscore",
                "--workdir", WD,
                "--sample_name", "A1",
                "--chrom", str(chrom),
                "--bfile_root", f"{RES}/LD_Reference_Panel/1000G_EUR_Phase3_plink/1000G.EUR.QC",
                "--keep_snp_root", f"{RES}/LDSC_resource/hapmap3_snps/hm",
                "--gtf_annotation_file", f"{RES}/genome_annotation/gtf/gencode.v46lift37.basic.annotation.gtf",
                "--gene_window_size", "50000",
                "--spots_per_chunk", str(spc)]

    t0 = time.time()
    for chrom in range(1, 23):
        if chrom in done:
            continue
        tc = time.time()
        try:
            run_v(ldscore_cmd(chrom, 250))
        except RuntimeError:
            print(f"  chrom {chrom} failed at 250 spots/chunk — retrying at 100")
            run_v(ldscore_cmd(chrom, 100))
        done.add(chrom)
        json.dump(sorted(done), open(PROG, "w"))
        print(f"  chrom {chrom} done in {(time.time()-tc)/60:.1f} min "
              f"({len(done)}/22)", flush=True)
    print(f"LD scores complete in {(time.time()-t0)/60:.1f} min this run")
    df()
'''))

cells.append(code('''# ---------- gsMap Step 4: per-spot spatial LDSC (Height control, then IQ) ----------
TRAITS = {
    "Height": f"{GSMAP_DIR}/gsMap_example_data/GWAS/GIANT_EUR_Height_2022_Nature.sumstats.gz",
    "IQ":     f"{GSMAP_DIR}/gsMap_example_data/GWAS/IQ_NG_2018.sumstats.gz",
}
for trait, ss in TRAITS.items():
    assert os.path.exists(ss), f"sumstats missing for {trait}: {ss}"
print("sumstats verified for:", list(TRAITS))
for trait, ss in TRAITS.items():
    t0 = time.time()
    run_v(["gsmap", "run_spatial_ldsc",
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
    run_v(["gsmap", "run_cauchy_combination",
           "--workdir", WD,
           "--sample_name", "A1",
           "--trait_name", trait,
           "--annotation", "domain"])
print("cauchy done")

# Discover + collect outputs (exact filenames are workdir-internal; glob them)
found = sorted(glob.glob(os.path.join(WD, "**", "*auchy*"), recursive=True) +
               glob.glob(os.path.join(WD, "**", "*A1*"), recursive=True))
for f in found:
    print(f)
'''))

cells.append(code('''# ---------- Summarise: domain x trait table -> push to GitHub ----------
summary = {"dataset": "A1", "embedding": "uploaded 10x_human_lymph_node_A1_seed_42_embeddings.npy -> obsm['fused']",
           "domains": "standard+kmeans k=10 (fixed one-pipeline recipe)",
           "traits": {}, "notes": "Height = negative control (expect null)"}

cauchy_files = [f for f in glob.glob(os.path.join(WD, "**", "*.csv*"), recursive=True)
                if "auchy" in os.path.basename(f).lower()]
print("cauchy files:", cauchy_files)
for f in cauchy_files:
    try:
        dfx = pd.read_csv(f)
        print("\\n", os.path.basename(f)); print(dfx.to_string()[:1500])
        dest = os.path.join(OUT, "A1_" + os.path.basename(f))
        shutil.copy(f, dest)
    except Exception as e:
        print("could not read", f, str(e)[:150])

with open(os.path.join(OUT, "A1_gsmap_pilot_log.json"), "w") as fh:
    json.dump({"cauchy_files": [os.path.basename(f) for f in cauchy_files],
               "alignment_ari_posthoc": float(ari)}, fh, indent=2)

git_push("s7-disease: exact gsMap A1 (uploaded seed42 embedding; Height control + IQ)")
print("\\nPilot done. Read the Height table first: it should be null across domains.")
df()
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

out_path = os.path.join(os.path.dirname(__file__), "..", "notebooks", "stage7_gsmap_exact_A1.ipynb")
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
              "gencode.v46lift37", "ALIGNMENT CHECK",
              "chunk=32", "ldscore_progress", "run_v",
              "resource_ready", "example_ready", "removing stub",
              "A1_generate_ldscore.done", "A1_gene_marker_score.feather",
              "sumstats verified", "gzip integrity",
              "10x_human_lymph_node_A1_seed_42_embeddings.npy", "EMB_NAME",
              "looks transposed"]:
    assert token in text, f"MISSING TOKEN: {token}"
assert '"wget"' not in text, "wget command leaked into the notebook"
assert "ALREADY_DONE" not in text, "stale ALREADY_DONE seed leaked into the notebook"
assert "s5-notebook" not in text, "old repo-runs embedding path leaked into the notebook"
assert "fused_embedding.npy" not in text, "old embedding filename leaked into the notebook"
print("content check: all key tokens present, uploaded-embedding route confirmed, no stale paths")
