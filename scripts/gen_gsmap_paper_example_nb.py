"""Generate notebooks/stage7_gsmap_paper_example.ipynb — the EXACT gsMap
paper pipeline, run on gsMap's OWN official example data, end to end.

Purpose (decided 2026-11-03): the decisive control for the A1 pilot. On A1,
the Height NEGATIVE CONTROL lit up almost everywhere (D0 p_cauchy 7e-15)
with the user's swapped-in embedding. This notebook runs gsMap exactly as
its authors publish it — their example data, their Step-1 GVAE embedding,
their parameters — so if Height also lights up here, the background is the
method's character, not our code or our embedding.

Source of truth: the official step-by-step tutorial
(https://yanglab.westlake.edu.cn/gps_data/website_docs/html/step_by_step.html),
read verbatim 2026-11-03. Every gsmap command below copies the tutorial's
flags and values exactly:
  Step 1 run_find_latent_representations --num_neighbors 200
         in  gsMap_example_data/ST/E16.5_E1S1.MOSTA.h5ad
         out .../E16.5_E1S1.MOSTA_latent.h5ad   (latent stored as obsm['latent_GVAE'])
  Step 2 run_latent_to_gene --num_neighbors 200
         --latent_representation latent_GVAE
         --homolog_file gsMap_resource/homologs/mouse_human_homologs.txt
         out .../E16.5_E1S1.MOSTA_GSS.h5ad
  Step 3 run_generate_ldscore — STRATEGY 3 (the paper's SNP->gene linking:
         TSS +-50 kb window PLUS enhancer links; the tutorial's Strategies
         1/2 are alternatives, not additional steps)
         --enhancer_annotation_file .../enhancer/by_tissue/ALL/ABC_roadmap_merged.bed
         --snp_multiple_enhancer_strategy max_mkscore
         --gene_window_enhancer_priority gene_window_first
         --gene_window_size 50000
  Step 4 run_spatial_ldsc --num_processes 4
  Step 5 run_cauchy_combination --annotation annotation
  Step 6 run_report --annotation annotation --top_corr_genes 50
Sample: E16.5_E1S1.MOSTA (mouse embryo Stereo-seq, ~120K cells per tutorial).
Workdir in the tutorial is './example/Mouse_Embryo'.

Traits: IQ (the tutorial's trait) + Height (added by us as the negative
control — identical commands, trait-independent LD scores reused). Height
is the whole point of the run; IQ alone would answer nothing.

KAGGLE BEST-EFFORT (user accepted the OOM risk 2026-11-03): the tutorial
states ~60G RAM for Steps 1 and 6, ~45G Step 2, ~40G Steps 3-4 at 120K
cells; Kaggle offers ~30G RAM and a ~20G disk. Adaptations are limited to
plumbing, never method:
  - resources located by filename, paths derived (A1 v5 lesson)
  - Step 3 chunked (--spots_per_chunk 250, retry 100 — the tutorial itself
    advises reducing this parameter on OOM) with per-chromosome checkpoints
  - completed steps skipped via their own output files / markers
  - Step 6 (marked optional in the tutorial) fails soft: a report OOM must
    not kill a completed Steps 1-5 run
  - results pushed to runs/s7-paper-example/

v2 FIX (2026-11-03, after the first Kaggle attempt failed in Step 1): the
tutorial PAGE is stale against the installed gsMap 1.73.8 CLI. Verified
from the 1.73.8 package source (config.py argument parsers + auto-paths):
  - Step 1 takes --input_hdf5_path (not the tutorial's h5ad spelling),
    REQUIRES --data_layer, has NO output-path argument (the latent h5ad is
    written to <workdir>/<sample>/find_latent_representations/
    <sample>_add_latent.h5ad with obsm['latent_GVAE']), and the neighbour
    flag is --n_neighbors (default 11; tutorial value 200 kept).
  - Step 2 takes --input_hdf5_path, has NO output-path argument, and the
    neighbour flag is --num_neighbour (default 21; tutorial value 200
    kept). With no --input_hdf5_path it defaults to Step 1's output; the
    notebook passes Step 1's auto-path explicitly.
  - Steps 3-6 flags were re-verified against the same source and were
    already correct.
  - --data_layer is chosen by a pre-flight read of the example h5ad:
    'count' layer if present, else 'counts', else 'X' (the find_latent
    source accepts a layer name or 'X'; all three feed it raw counts).
  - gsMap is pinned to ==1.73.8 in the install cell so the verified CLI
    cannot drift under the notebook again.

Run: python3 scripts/gen_gsmap_paper_example_nb.py
"""
import json, os

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

cells = []

cells.append(md('''# Stage 7 — The EXACT gsMap Paper Pipeline, on gsMap's Own Example Data

**What this is:** a faithful reproduction of gsMap's official step-by-step
tutorial (Song et al., *Nature* 2025), end to end, on the authors' own
example data — **nothing skipped, nothing swapped, their parameters**:

| Step | Command | Tutorial's stated RAM |
|---|---|---|
| 1 | `run_find_latent_representations` (their GVAE embedding) | ~60 GB |
| 2 | `run_latent_to_gene` (mouse→human homologs applied) | ~45 GB |
| 3 | `run_generate_ldscore` — **Strategy 3**: TSS ±50 kb **+** enhancer links | ~40 GB |
| 4 | `run_spatial_ldsc` | ~40 GB |
| 5 | `run_cauchy_combination` (`--annotation annotation`) | ~12 GB |
| 6 | `run_report` (`--top_corr_genes 50`) | ~60 GB |

**Data:** `E16.5_E1S1.MOSTA` — mouse embryo Stereo-seq, **~120K cells**
(A1 was ~3.5K; this is ~34× bigger). Sample name, workdir layout, neighbour
counts (200), gene window (50000), strategies — all as published.

**Why:** on A1, the Height *negative control* lit up almost everywhere with
our swapped-in embedding. If Height also lights up here — gsMap's own data,
gsMap's own embedding, gsMap's own commands — the background inflation is
the **method's character**, not our code. If Height is null here, the A1
background came from somewhere in our setup and we go hunting.

**Traits:** IQ (the tutorial's trait) + **Height** (the control — identical
commands; the LD scores from Step 3 are trait-independent and reused).

**Kaggle best-effort — read this first.** Kaggle gives ~30 GB RAM and a
~20 GB disk; the table above is what the tutorial asks for. Steps may die
with out-of-memory, most likely Step 1 or Step 6. The user has accepted
this risk. Plumbing adaptations only (the method is untouched): resources
located by filename with paths derived from the finds; Step 3 chunked at
250 spots/chunk with a retry at 100 and per-chromosome checkpoints (the
tutorial itself advises reducing `--spots_per_chunk` on OOM); finished
steps are skipped via their own outputs, so a re-run resumes; Step 6 is
optional in the tutorial and fails soft here. **Save Version when a long
step completes** — an unsaved session loses the workdir, as the SLE run
learned the hard way.

**Output:** `runs/s7-paper-example/` (Cauchy tables for Height + IQ, run log).

**v2 fix (2026-11-03):** the first attempt died in Step 1 — the tutorial
*page* is stale against the installed gsMap **1.73.8** CLI (wrong flag
spellings, a required data-layer argument the page never mentions, and
output-path arguments that no longer exist). All six steps' commands have
been rebuilt from the installed package's own argument definitions,
gsMap is pinned to 1.73.8, and Step 1's output is read from gsMap's own
workdir location. The method and every parameter value are unchanged.
'''))

cells.append(code('''import os, sys, json, subprocess, time

for pkg in ["gsMap==1.73.8", "scanpy", "anndata", "scikit-learn", "pandas", "scipy"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

r = subprocess.run(["gsmap", "-v"], capture_output=True, text=True)
print("gsmap version:", (r.stdout or r.stderr).strip()[:100])

def mem_avail_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1e6
    except Exception:
        pass
    return -1

print(f"RAM available: {mem_avail_gb():.0f} GB (tutorial wants ~60 GB for Steps 1/6)")
print("imports ok")
'''))

cells.append(code('''# ---------- GitHub: clone + push helper (results -> runs/s7-paper-example) ----------
REPO = "/kaggle/working/my_research"
REPO_URL = "https://github.com/rifatahsanpul0k/my_research.git"
def sh(*args):
    return subprocess.run(list(args), capture_output=True, text=True, cwd=REPO)
os.makedirs(REPO, exist_ok=True)
OUT = os.path.join(REPO, "runs", "s7-paper-example")
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
        sh("git", "add", "runs/s7-paper-example")
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

cells.append(code('''# ---------- Download gsMap resources + example data (discovered by filename) ----------
import glob, shutil

def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd)}")
    return r

def run_v(cmd):
    # Captures output and prints gsMap's own last 30 lines on failure — a bare
    # wrapper traceback hid the real error in the A1 runs. An out-of-memory
    # kill shows up here as returncode -9 with gsMap's last lines before it.
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

# --- Resources are DISCOVERED by filename, never trusted at fixed paths ---
# (A1 lesson, 2026-11-02: bundles can extract cleanly while a fixed-path
# readiness check still fails — e.g. keying on a merged 1000G.EUR.QC.bim
# that does not exist; the panel is per-chromosome. So find the files,
# derive every gsMap path from the finds, and never delete an extracted
# tree on a failed check — move it aside and re-fetch exactly once.)
GSMAP_DIR = "/kaggle/working/gsmap"
os.makedirs(GSMAP_DIR, exist_ok=True)
RES = os.path.join(GSMAP_DIR, "gsMap_resource")

def _find1(pat):
    hits = sorted(glob.glob(os.path.join(GSMAP_DIR, "**", pat), recursive=True))
    return hits[0] if hits else None

def _tree(p, depth=2):
    for dp, dns, fns in os.walk(p):
        lvl = dp.replace(p, "").count(os.sep)
        if lvl > depth:
            continue
        print("  " * lvl + os.path.basename(dp) + "/")
        if lvl == depth:
            for fn in sorted(fns)[:8]:
                print("  " * (lvl + 1) + fn)

BIM = BIM22 = GTF_FILE = W1 = HEIGHT_SS = IQ_SS = None
ST_H5AD = HOMOLOG = ENHANCER = None
BFILE_ROOT = HM_ROOT = W_PREFIX = None

def discover():
    global BIM, BIM22, GTF_FILE, W1, HEIGHT_SS, IQ_SS, BFILE_ROOT, HM_ROOT, W_PREFIX, RES
    global ST_H5AD, HOMOLOG, ENHANCER
    BIM = _find1("1000G.EUR.QC.1.bim")          # panel is per-chromosome;
    BIM22 = _find1("1000G.EUR.QC.22.bim")       # .1/.22 = completeness probes
    GTF_FILE = _find1("gencode*.gtf") or _find1("*.gtf")
    W1 = _find1("weights.1.l2.ldscore.gz")
    HEIGHT_SS = _find1("GIANT_EUR_Height_2022_Nature.sumstats.gz")
    IQ_SS = _find1("IQ_NG_2018.sumstats.gz")
    ST_H5AD = _find1("E16.5_E1S1.MOSTA.h5ad")   # exact name: never the _latent/_GSS outputs
    HOMOLOG = _find1("mouse_human_homologs.txt")
    ENHANCER = _find1("ABC_roadmap_merged.bed")
    if BIM:
        BFILE_ROOT = BIM[:-len(".1.bim")]       # gsMap appends .<chrom> itself
        RES = os.path.dirname(os.path.dirname(os.path.dirname(BIM)))
    if W1:
        W_PREFIX = W1[:-len("1.l2.ldscore.gz")]  # .../weights_hm3_no_hla/weights.
    hm_dirs = sorted(glob.glob(os.path.join(GSMAP_DIR, "**", "hapmap3_snps"),
                                recursive=True))
    if hm_dirs:
        HM_ROOT = os.path.join(hm_dirs[0], "hm")
    return bool(BIM and GTF_FILE and W1)

def resources_found():
    discover()
    return bool(BIM and BIM22 and GTF_FILE and W1 and HM_ROOT and HOMOLOG and ENHANCER)

def example_found():
    discover()
    return bool(ST_H5AD and HEIGHT_SS and IQ_SS)

def ensure_bundle(kind, url, tarball, std_dirname, found_fn):
    if found_fn():
        print(f"{kind}: already present and verified by discovery")
        return
    std = os.path.join(GSMAP_DIR, std_dirname)
    aside = None
    if os.path.isdir(std):
        aside = std + "_incomplete"
        if os.path.exists(aside):
            shutil.rmtree(aside)
        os.rename(std, aside)
        print(f"{std_dirname} present but key files not found — moved aside, re-fetching")
    fetch(url, tarball)
    run(["tar", "-xzf", tarball])
    os.remove(tarball)  # disk discipline: ~20 GB working dir
    if not found_fn():
        print(f"--- tree of {GSMAP_DIR} (depth 2) ---")
        _tree(GSMAP_DIR)
        raise RuntimeError(f"{kind} files not found even after fresh extraction "
                           "— send the tree listing above")
    if aside and os.path.isdir(aside):
        shutil.rmtree(aside)
        print(f"fresh {kind} bundle verified — removed the incomplete copy")

os.chdir(GSMAP_DIR)
df()
ensure_bundle("resource",
              "https://yanglab.westlake.edu.cn/data/gsMap/gsMap_resource.tar.gz",
              "gsMap_resource.tar.gz", "gsMap_resource", resources_found)
ensure_bundle("example",
              "https://yanglab.westlake.edu.cn/data/gsMap/gsMap_example_data.tar.gz",
              "gsMap_example_data.tar.gz", "gsMap_example_data", example_found)
# NOTE: unlike the A1 notebook, the example ST data is KEPT — it IS the input.
for leftover in glob.glob("*.tar.gz"):
    os.remove(leftover)
    print("removed leftover archive:", leftover)
df()

print("resource root :", RES)
print("bfile prefix  :", BFILE_ROOT)
print("gtf           :", GTF_FILE)
print("keep_snp root :", HM_ROOT)
print("weights prefix:", W_PREFIX)
print("homolog file  :", HOMOLOG)
print("enhancer bed  :", ENHANCER)
print("example ST    :", ST_H5AD)
print("height gwas   :", HEIGHT_SS)
print("iq gwas       :", IQ_SS)
assert resources_found() and example_found(), "resource check failed — see paths above"
print("all resources verified")

# Pipeline constants (tutorial values)
SAMPLE = "E16.5_E1S1.MOSTA"
WD = os.path.join(GSMAP_DIR, "example", "Mouse_Embryo")  # tutorial: ./example/Mouse_Embryo
os.makedirs(WD, exist_ok=True)
ST_DIR = os.path.dirname(ST_H5AD)
# gsMap 1.73.8 writes Step 1's output ITSELF (no output-path argument):
# <workdir>/<sample>/find_latent_representations/<sample>_add_latent.h5ad
LATENT_H5AD = os.path.join(WD, SAMPLE, "find_latent_representations",
                           SAMPLE + "_add_latent.h5ad")
ANNOT = "annotation"   # obs column in the example h5ad (tutorial, Steps 5-6)
TRAITS = {"Height": HEIGHT_SS, "IQ": IQ_SS}
PROG_STATE = os.path.join(GSMAP_DIR, "paper_progress.json")
def load_state():
    return json.load(open(PROG_STATE)) if os.path.exists(PROG_STATE) else {}
def save_state(s):
    json.dump(s, open(PROG_STATE, "w"), indent=1)
print("workdir:", WD)
print("latent out (gsMap auto-path):", LATENT_H5AD)
'''))

cells.append(code('''# ---------- Step 1: find latent representations (gsMap's OWN GVAE embedding) ----------
# Command rebuilt from the gsMap 1.73.8 package's own argument parser (the
# tutorial page's Step 1 command is stale: wrong input-flag spelling, an
# output-path argument that no longer exists, and no mention of the
# REQUIRED data-layer argument). Verified equivalents of the tutorial's
# intent: neighbours 200; latent written by gsMap itself to
# <workdir>/<sample>/find_latent_representations/<sample>_add_latent.h5ad
# with obsm['latent_GVAE']. All other GVAE settings are 1.73.8 defaults
# (epochs 300, feat 256->128, GAT 64->30, lr 0.001, dropout 0.1, PCA 300).
# Tutorial states ~60 GB RAM at ~120K cells; Kaggle has ~30 GB — this is
# the step most likely to OOM. Skipped on re-run once its output exists.
import h5py
with h5py.File(ST_H5AD, "r") as f:
    _layers = list(f["layers"].keys()) if "layers" in f else []
print("example h5ad layers:", _layers)
DATA_LAYER = "count" if "count" in _layers else ("counts" if "counts" in _layers else "X")
print("data_layer for Step 1:", DATA_LAYER, "(the raw counts gsMap's GVAE expects)")

if os.path.exists(LATENT_H5AD):
    print("latent h5ad found — Step 1 already done, skipping")
else:
    print(f"RAM available before Step 1: {mem_avail_gb():.0f} GB (tutorial: ~60 GB)")
    t0 = time.time()
    run_v(["gsmap", "run_find_latent_representations",
           "--workdir", WD,
           "--sample_name", SAMPLE,
           "--input_hdf5_path", ST_H5AD,
           "--data_layer", DATA_LAYER,
           "--n_neighbors", "200"])
    assert os.path.exists(LATENT_H5AD), "Step 1 finished but produced no latent h5ad"
    print(f"Step 1 done in {(time.time()-t0)/60:.1f} min")
    df()
'''))

cells.append(code('''# ---------- Step 2: latent_to_gene (GSS from gsMap's own latent) ----------
# Command rebuilt from the gsMap 1.73.8 parser: the neighbour flag is
# --num_neighbour (tutorial value 200 kept; spatial neighbours stay at the
# default 101), the input flag is --input_hdf5_path, and there is NO
# output-path argument — GSS lands in gsMap's own workdir files (the
# gene-marker-score feather is the skip marker, as in the A1 run). The
# homolog file is REQUIRED here (mouse example; human A1 needed none).
# Input passed explicitly: Step 1's own output h5ad (obsm['latent_GVAE']).
import glob as _g
_feather = _g.glob(os.path.join(WD, "**", "*gene_marker_score.feather"), recursive=True)
if _feather:
    print("mkscore feather found — Step 2 already done, skipping:", _feather[0])
else:
    assert os.path.exists(LATENT_H5AD), "latent h5ad missing — Step 1 did not finish"
    print(f"RAM available before Step 2: {mem_avail_gb():.0f} GB (tutorial: ~45 GB)")
    t0 = time.time()
    run_v(["gsmap", "run_latent_to_gene",
           "--workdir", WD,
           "--sample_name", SAMPLE,
           "--input_hdf5_path", LATENT_H5AD,
           "--latent_representation", "latent_GVAE",
           "--num_neighbour", "200",
           "--homolog_file", HOMOLOG])
    print(f"Step 2 done in {(time.time()-t0)/60:.1f} min")
    df()
'''))

cells.append(code('''# ---------- Step 3: generate LD scores — Strategy 3 (the paper's linking) ----------
# The tutorial offers three ALTERNATIVE Step-3 strategies. Strategy 3 is the
# paper's SNP->gene linking: TSS +-50 kb window PLUS enhancer links
# (--gene_window_enhancer_priority gene_window_first,
#  --snp_multiple_enhancer_strategy max_mkscore). Strategies 1/2 are NOT run.
# Kaggle plumbing only: per-chromosome checkpointing + --spots_per_chunk 250
# with a retry at 100 (the tutorial itself advises reducing spots_per_chunk
# on OOM; its commands pass no value, i.e. gsMap's larger default).
# Skipped entirely when gsMap's own .done marker exists.
import glob as _g
_done = _g.glob(os.path.join(WD, "**", "*generate_ldscore.done"), recursive=True)
if _done:
    print("gsMap .done marker found — LD scores already complete, skipping Step 3")
else:
    PROG = os.path.join(GSMAP_DIR, "paper_ldscore_progress.json")
    done = set(json.load(open(PROG))) if os.path.exists(PROG) else set()
    print("chromosomes already done:", sorted(done))
    df()

    def ldscore_cmd(chrom, spc):
        return ["gsmap", "run_generate_ldscore",
                "--workdir", WD,
                "--sample_name", SAMPLE,
                "--chrom", str(chrom),
                "--bfile_root", BFILE_ROOT,
                "--keep_snp_root", HM_ROOT,
                "--gtf_annotation_file", GTF_FILE,
                "--enhancer_annotation_file", ENHANCER,
                "--gene_window_size", "50000",
                "--snp_multiple_enhancer_strategy", "max_mkscore",
                "--gene_window_enhancer_priority", "gene_window_first",
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
        df()
    print(f"LD scores complete in {(time.time()-t0)/60:.1f} min this run")
'''))

cells.append(code('''# ---------- Step 4: per-spot spatial LDSC (Height control + IQ) ----------
# Tutorial command, verbatim flags (--num_processes 4). Run per trait;
# progress is checkpointed in paper_progress.json so a re-run only repeats
# the traits that never finished. LD scores from Step 3 are reused.
state = load_state()
for trait, ss in TRAITS.items():
    assert os.path.exists(ss), f"sumstats missing for {trait}: {ss}"
print("sumstats verified for:", list(TRAITS))
for trait, ss in TRAITS.items():
    if state.get(f"ldsc_{trait}"):
        print(f"spatial_ldsc [{trait}] already done — skipping")
        continue
    print(f"RAM available before ldsc [{trait}]: {mem_avail_gb():.0f} GB (tutorial: ~40 GB)")
    t0 = time.time()
    run_v(["gsmap", "run_spatial_ldsc",
           "--workdir", WD,
           "--sample_name", SAMPLE,
           "--trait_name", trait,
           "--sumstats_file", ss,
           "--w_file", W_PREFIX,
           "--num_processes", "4"])
    state[f"ldsc_{trait}"] = True
    save_state(state)
    print(f"spatial_ldsc [{trait}] done in {(time.time()-t0)/60:.1f} min", flush=True)
'''))

cells.append(code('''# ---------- Step 5: Cauchy combination per annotation ----------
# Tutorial command, verbatim: --annotation annotation (the obs column the
# example h5ad ships with). Skipped per trait when its Cauchy table exists.
import glob as _g
state = load_state()
for trait in TRAITS:
    if _g.glob(os.path.join(WD, "**", f"*{trait}.Cauchy.csv*"), recursive=True):
        print(f"cauchy [{trait}] output found — skipping")
        continue
    assert state.get(f"ldsc_{trait}"), f"spatial_ldsc for {trait} not done — run Step 4"
    run_v(["gsmap", "run_cauchy_combination",
           "--workdir", WD,
           "--sample_name", SAMPLE,
           "--trait_name", trait,
           "--annotation", ANNOT])
    print(f"cauchy [{trait}] done")
print("Step 5 complete")
'''))

cells.append(code('''# ---------- Step 6: gsMap report (OPTIONAL in the tutorial — fails soft) ----------
# Tutorial command, verbatim: --top_corr_genes 50. The tutorial marks this
# step optional and states ~60 GB RAM; on Kaggle it may OOM. A failure here
# must NOT kill the run: Steps 1-5 are the result. The exception is printed
# and the notebook continues to the summary.
state = load_state()
for trait, ss in TRAITS.items():
    if state.get(f"report_{trait}"):
        print(f"report [{trait}] already done — skipping")
        continue
    print(f"RAM available before report [{trait}]: {mem_avail_gb():.0f} GB (tutorial: ~60 GB)")
    try:
        t0 = time.time()
        run_v(["gsmap", "run_report",
               "--workdir", WD,
               "--sample_name", SAMPLE,
               "--trait_name", trait,
               "--annotation", ANNOT,
               "--sumstats_file", ss,
               "--top_corr_genes", "50"])
        state[f"report_{trait}"] = True
        save_state(state)
        print(f"report [{trait}] done in {(time.time()-t0)/60:.1f} min")
    except Exception as e:
        print(f"report [{trait}] FAILED (continuing — step is optional): {str(e)[:200]}")
'''))

cells.append(code('''# ---------- Summarise: the decisive comparison -> push to GitHub ----------
# The whole run in one table: per annotation, Height (control) vs IQ
# -log10 p_cauchy on gsMap's own data + gsMap's own embedding.
import glob as _g
import numpy as np
import pandas as pd
import shutil

def _cauchy(trait):
    fs = _g.glob(os.path.join(WD, "**", f"*{trait}.Cauchy.csv*"), recursive=True)
    return fs[0] if fs else None

tab = None
for trait in ["Height", "IQ"]:
    f = _cauchy(trait)
    if f is None:
        print(f"WARNING: no Cauchy table for {trait} — Step 5 incomplete for it")
        continue
    d = pd.read_csv(f)
    print("\\n", os.path.basename(f)); print(d.to_string()[:1200])
    shutil.copy(f, os.path.join(OUT, "EXAMPLE_" + os.path.basename(f)))
    d = d[["annotation", "p_cauchy"]].rename(columns={"p_cauchy": trait})
    tab = d if tab is None else tab.merge(d, on="annotation")

if tab is not None and "Height" in tab and "IQ" in tab:
    for trait in ["Height", "IQ"]:
        tab[trait + " -log10p"] = -np.log10(tab[trait].clip(lower=1e-300))
    tab = tab.sort_values("Height -log10p", ascending=False)
    print("\\n===== Height (control) vs IQ — gsMap example, gsMap embedding =====")
    print(tab.to_string(index=False))
    tab.to_csv(os.path.join(OUT, "EXAMPLE_height_vs_iq_comparison.csv"), index=False)
    print("\\nREAD: if Height is significant across most annotations HERE too,")
    print("the A1 background is the method's character. If Height is null here,")
    print("the A1 inflation came from our setup — go back and hunt.")

# Report HTMLs ride along only when small enough for git
for h in _g.glob(os.path.join(WD, "**", "*.html"), recursive=True):
    mb = os.path.getsize(h) / 1e6
    if mb < 20:
        shutil.copy(h, os.path.join(OUT, os.path.basename(h)))
        print(f"report html copied ({mb:.1f} MB):", os.path.basename(h))
    else:
        print(f"report html too large for git ({mb:.0f} MB), left in workdir:",
              os.path.basename(h))

with open(os.path.join(OUT, "EXAMPLE_gsmap_paper_log.json"), "w") as fh:
    json.dump({"sample": SAMPLE, "data": "gsMap official example (E16.5 mouse embryo, ~120K cells)",
               "embedding": "gsMap Step 1 GVAE (latent_GVAE) — nothing swapped",
               "step3_strategy": "Strategy 3 (TSS 50kb + enhancer, gene_window_first)",
               "traits": list(TRAITS), "state": load_state()}, fh, indent=2)

git_push("s7-paper-example: exact gsMap paper pipeline on official example data (Height control + IQ)")
df()
'''))

cells.append(md('''## How to read the result

- **Height significant across most annotations here too** → the background
  inflation we saw on A1 is gsMap's *method character* at high GWAS power:
  per-spot S-LDSC + Cauchy tilts non-null almost everywhere. Our A1 pilot
  machinery is exonerated, and Phase 2 traits must be read against their
  Height background (which is exactly how the SLE cells are built).
- **Height null here** → the inflation came from somewhere in *our* A1
  setup (embedding swap, domain recipe, or packaging). The A1 tables get
  quarantined until the difference is found.
- Either way, IQ's annotation ranking here is the authors' own reference
  result — the closest thing to a ground-truth gsMap table that exists.

**Session discipline:** Save Version whenever a long step finishes; the
workdir (latent h5ad, LD scores, progress checkpoints) is what makes a
re-run resume instead of restart.
'''))

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "cells": cells,
}

out_path = os.path.join(os.path.dirname(__file__), "..", "notebooks", "stage7_gsmap_paper_example.ipynb")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {out_path} ({len(cells)} cells)")

import nbformat
nbformat.read(out_path, as_version=4)
print("nbformat: valid")

# Content verification (AGENTS.md lesson: verify the OUTPUT notebook)
text = open(out_path).read()
for token in ["run_find_latent_representations", "latent_GVAE",
              "E16.5_E1S1.MOSTA", "_add_latent.h5ad",
              "input_hdf5_path", "data_layer", "DATA_LAYER",
              "n_neighbors", "num_neighbour", "gsMap==1.73.8",
              "mouse_human_homologs.txt",
              "run_latent_to_gene", "run_generate_ldscore",
              "ABC_roadmap_merged.bed", "max_mkscore", "gene_window_first",
              "gene_window_size", "spots_per_chunk",
              "run_spatial_ldsc", "run_cauchy_combination", "run_report",
              "top_corr_genes",
              "GIANT_EUR_Height_2022", "IQ_NG_2018",
              "1000G.EUR.QC.1.bim", "1000G.EUR.QC.22.bim",
              "paper_ldscore_progress", "paper_progress.json",
              "generate_ldscore.done", "gene_marker_score.feather",
              "runs/s7-paper-example", "EXAMPLE_height_vs_iq_comparison.csv",
              "mem_avail_gb", "Strategy 3"]:
    assert token in text, f"MISSING TOKEN: {token}"
assert '"wget"' not in text, "wget command leaked into the notebook"
assert '"--num_neighbors"' not in text, "the tutorial's stale neighbour flag leaked into the notebook"
assert '"--input_h5ad_path"' not in text, "the tutorial's stale input flag leaked into the notebook"
assert '"--output_h5ad_path"' not in text, "the removed output-path argument leaked into the notebook"
assert "MOSTA_latent.h5ad" not in text, "the tutorial's stale Step 1 output name leaked into the notebook"
assert "MOSTA_GSS.h5ad" not in text, "the tutorial's stale Step 2 output name leaked into the notebook"
assert "s5-notebook" not in text, "A1 repo-runs path leaked into the notebook"
assert "fused_embedding" not in text, "swapped-embedding route leaked into the notebook"
assert "10x_human_lymph_node" not in text, "A1 dataset leaked into the notebook"
assert '"1000G.EUR.QC.bim"' not in text, "phantom merged-bim search leaked into the notebook"
assert "enhancer_only" not in text, "Strategy 2 leaked into the notebook"
print("content check: all key tokens present, tutorial commands confirmed, no A1 leakage")
