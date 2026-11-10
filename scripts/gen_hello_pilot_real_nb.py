"""Generate notebooks/hello_pipeline_pilot_real.ipynb

REAL-DATA PILOT run of the hello.md pipeline (Branch A: A1 + D1) for Kaggle.

What is real in this run (vs the smoke test):
- MAGMA gene-level z-scores computed in-notebook from the locked GWAS files
  (Kaggle dataset pulokpulok/hello-pipeline-refs) with MAGMA v1.10 static,
  g1000_eur reference, per-build gene locations (NCBI38 / NCBI37.3),
  annotation window 10 kb up / 1.5 kb down (blueprint).
- Gene annotation (chrom/start/end/biotype) from the repo GENCODE v44 table.
- Gene covariates built from data: log10 gene length, log10 median NSNPS
  across traits, pooled mean log-expression (A1+D1).
- Permutations at the code default scale: n_perm_initial = 1000, with the
  code's own D_neg-triggered recalibration to 10000 left intact.

What stays AS-IS from hello.md (documented pilot caveats):
- Engine 2 S-LDSC is the code's internal OLS proxy (lookups = None).
- RSR audit without the P_perp projection; Engine 3 spot p-values from the
  code's own spin-null handling; ADT CLR is per-protein across spots.
- FL and DLBCL are SEPARATE traits (no combined GWAS exists); the panel
  override replaces "Follicular_Lymphoma_DLBCL" with DLBCL + FL, so the
  run has 16 traits, not 15.

The pipeline code cells are carried verbatim from the base smoke notebook
(cells 4 and 5 = hello.md blocks 1 and 2), with the single token swap
SMOKE_N_PERM -> REAL_N_PERM in the driver cell.

Every GWAS column mapping in the sumstats-prep cell was verified against
the live file headers on 2026-11-10 (see DOWNLOAD_LIST.md).
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_CANDIDATES = [
    os.path.join(HERE, "..", "notebooks", "hello_pipeline_smoke_A1.ipynb"),
    os.path.expanduser("~/workspace/hello_smoke_nb/hello_pipeline_smoke_A1.ipynb"),
]
BASE_NB = next((p for p in BASE_CANDIDATES if os.path.exists(p)), None)
assert BASE_NB, "base smoke notebook not found"

MD = """# Hello Pipeline — REAL-DATA PILOT (A1 + D1, Kaggle)

**Purpose:** run the `hello.md` pipeline end-to-end on **real** inputs to see actual results. This is a **pilot**: the numbers are real, but three engine caveats from the audit remain in the code as-is, so tiers are **provisional**:

- Engine 2's "S-LDSC" is the code's internal OLS proxy (no real S-LDSC run).
- The RSR audit does not apply the $P_\\perp$ projection; Engine 3 spot p-values use the code's own null handling.
- ADT CLR is per-protein across spots (the code's convention).

**Real inputs:** MAGMA z-scores computed in this notebook from the locked GWAS files (16 traits; FL and DLBCL separate — no combined GWAS exists), GENCODE v44 gene annotation from the repo, covariates from data (gene length, median SNP count, pooled mean expression), Muse-GCN seed-42 embeddings, n_perm = 1000 (the code recalibrates to 10000 itself if a negative control is Confirmed).

**Attach before running:** datasets `pulokpulok/spatial-multiomics-6datasets-private` and `pulokpulok/hello-pipeline-refs`, secret `GITHUB_TOKEN`, Internet ON. Then **Save Version -> Run All** (headless runs survive disconnects; interactive sessions do not).

**Runtime estimate:** MAGMA ~0.5-2.5 h for 16 traits; pipeline ~1-3 h for both sections at n_perm=1000. If the negative-control gate triggers the 10000-permutation recalibration, add several hours. Checkpoints are pushed to GitHub (`runs/hello-pilot-real`) after MAGMA, after A1, and after D1, so partial progress survives.

**Reading the result:** check the negative controls (Adult_Standing_Height, Heel_BMD, Hair_Colour) FIRST — if they light up, the discovery hits are background, not signal.
"""

INSTALL = """import os, sys, subprocess
for pkg in ["scanpy", "statsmodels"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
import numpy as np
import pandas as pd
import scanpy as sc
print("imports ok; numpy", np.__version__, "| pandas", pd.__version__)
"""

CONFIG = """# ---------- CONFIG (real pilot, locked 2026-11-10) ----------
SEED = 42
REAL_N_PERM = 1000           # code default scale; recalibration to 10000 is the code's own gate
RAW_BASE = "https://raw.githubusercontent.com/rifatahsanpul0k/my_research/main"
WORK = "/kaggle/working/pilot_inputs"
OUTDIR = "/kaggle/working/pilot_outputs"
REFS = "/kaggle/working/refs"
MAGMA_WORK = "/kaggle/working/magma_work"
for d in (WORK, OUTDIR, REFS, MAGMA_WORK):
    os.makedirs(d, exist_ok=True)

SECTIONS = {
    "A1": {"folder": "10x_human_lymph_node_A1",
           "emb": "10x_human_lymph_node_A1_seed42_embeddings.npy",
           "lab": "10x_human_lymph_node_A1_seed42_labels.npy"},
    "D1": {"folder": "10x_human_lymph_node_D1",
           "emb": "10x_human_lymph_node_D1_seed42_embeddings.npy",
           "lab": "10x_human_lymph_node_D1_seed42_labels.npy"},
}

import glob as _glob
def find_refs_root():
    for pat in ("/kaggle/input/*/magma.zip", "/kaggle/input/*/*/magma.zip", "/kaggle/input/*/*/*/magma.zip"):
        for hit in _glob.glob(pat):
            root = os.path.dirname(hit)
            if os.path.exists(os.path.join(root, "gwas.zip")):
                return root
    return None

REFS_IN = find_refs_root()
assert REFS_IN, "hello-pipeline-refs dataset not attached (magma.zip not found under /kaggle/input)"
print("refs dataset at:", REFS_IN)

def find_section_dir(folder):
    for pat in (f"/kaggle/input/**/{folder}/adata_RNA.h5ad",):
        hits = _glob.glob(pat, recursive=True)
        if hits:
            return os.path.dirname(hits[0])
    return None

SEC_DIRS = {}
for sec, spec in SECTIONS.items():
    d = find_section_dir(spec["folder"])
    assert d, f"{spec['folder']} not found - attach pulokpulok/spatial-multiomics-6datasets-private"
    SEC_DIRS[sec] = d
    print(sec, "data:", d)

# ---------- Muse-GCN seed-42 embeddings + labels from the GitHub repo ----------
import urllib.request
EMB = {}
for sec, spec in SECTIONS.items():
    paths = {}
    for kind, name in (("E", spec["emb"]), ("C", spec["lab"])):
        dst = os.path.join(WORK, name)

        if not os.path.exists(dst):
            urllib.request.urlretrieve(f"{RAW_BASE}/embeddings/{name}", dst)
        paths[kind] = np.load(dst)
    assert paths["E"].shape[0] == paths["C"].shape[0], f"{sec}: E rows != C length"
    EMB[sec] = paths
    print(sec, "E:", paths["E"].shape, "| domains:", np.unique(paths["C"]))
"""

PUSH_HELPER = """# ---------- GitHub checkpoint push helper (runs/hello-pilot-real) ----------
# Failures only warn: outputs also stay in /kaggle/working/pilot_outputs and in
# the Kaggle version output. Token comes from the GITHUB_TOKEN Kaggle secret.
import subprocess, base64, shutil
def push_outputs(message):
    try:
        from kaggle_secrets import UserSecretsClient
        _tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
        _hdr = "Authorization: Basic " + base64.b64encode(f"x-access-token:{_tok}".encode()).decode()
        _GIT = ["git", "-c", f"http.extraHeader={_hdr}"]
        _REPO = "/kaggle/working/my_research_push"
        if os.path.isdir(_REPO):
            shutil.rmtree(_REPO)
        subprocess.run(_GIT + ["clone", "--depth", "1", "--filter=blob:none", "--sparse",
                               "https://github.com/rifatahsanpul0k/my_research.git", _REPO],
                       check=True, capture_output=True, text=True)
        subprocess.run(["git", "sparse-checkout", "set", "runs/hello-pilot-real"],
                       cwd=_REPO, check=True, capture_output=True, text=True)
        _dst = os.path.join(_REPO, "runs", "hello-pilot-real")
        os.makedirs(_dst, exist_ok=True)
        for fn in os.listdir(OUTDIR):
            src = os.path.join(OUTDIR, fn)
            if os.path.isfile(src):
                shutil.copy(src, os.path.join(_dst, fn))
            elif os.path.isdir(src):
                shutil.copytree(src, os.path.join(_dst, fn), dirs_exist_ok=True)
        def _sh(*a):
            return subprocess.run(list(a), cwd=_REPO, capture_output=True, text=True)
        _sh("git", "config", "user.name", "kaggle-runner")
        _sh("git", "config", "user.email", "kaggle-runner@local")
        _sh("git", "add", "runs/hello-pilot-real")
        if _sh("git", "diff", "--cached", "--quiet").returncode != 0:
            _sh("git", "commit", "-m", message)
            subprocess.run(_GIT + ["pull", "--rebase", "origin", "main"], cwd=_REPO, capture_output=True, text=True)
            _r = subprocess.run(_GIT + ["push", "origin", "main"], cwd=_REPO, capture_output=True, text=True)
            print("pushed:", message if _r.returncode == 0 else "FAILED " + _r.stderr[:300])
        else:
            print("push: nothing new")
    except Exception as e:
        print("push skipped/failed (outputs safe in pilot_outputs):", str(e)[:300])
"""

REFS_SETUP = """# ---------- Unzip reference bundles + MAGMA binary ----------
import zipfile, shutil, urllib.request, stat
def _unzip(zpath, dest):
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(dest)

def _find1(pat):
    hits = _glob.glob(pat, recursive=True)
    assert hits, f"not found: {pat}"
    return hits[0]

if not os.path.exists(os.path.join(REFS, "gwas")):
    for z in ("magma.zip", "gwas.zip"):
        zp = os.path.join(REFS_IN, z)
        print("unzipping", z, "...")
        _unzip(zp, REFS)
print("refs tree:", sorted(os.listdir(REFS)))

# inner MAGMA zips (search recursively: dir-mode zip may or may not keep the folder level)
_unzip(_find1(os.path.join(REFS, "**", "magma_v1.10_static.zip")), os.path.join(REFS, "magma_bin"))
_unzip(_find1(os.path.join(REFS, "**", "NCBI38.zip")), os.path.join(REFS, "gene_loc"))
_unzip(_find1(os.path.join(REFS, "**", "g1000_eur.zip")), os.path.join(REFS, "g1000"))
MAGMA = _find1(os.path.join(REFS, "magma_bin", "**", "magma"))
os.chmod(MAGMA, os.stat(MAGMA).st_mode | stat.S_IEXEC)
GENE_LOC_38 = _find1(os.path.join(REFS, "gene_loc", "**", "NCBI38.gene.loc"))
BFILE = _find1(os.path.join(REFS, "g1000", "**", "g1000_eur.bim"))[:-4]  # strip .bim
assert os.path.exists(BFILE + ".bed") and os.path.exists(BFILE + ".fam"), "g1000_eur plink files incomplete"

# NCBI37.3 gene locations were not in the download bundle - fetch (0.35 MB, link verified 2026-11-10)
_ncbi37 = os.path.join(REFS, "NCBI37.3.zip")
if not os.path.exists(_ncbi37):
    urllib.request.urlretrieve("https://vu.data.surfsara.nl/index.php/s/Pj2orwuF2JYyKxq/download", _ncbi37)
_unzip(_ncbi37, os.path.join(REFS, "gene_loc"))
GENE_LOC_37 = _find1(os.path.join(REFS, "gene_loc", "**", "NCBI37.3.gene.loc"))

_r = subprocess.run([MAGMA], capture_output=True, text=True)
print("MAGMA binary responds:", (("MAGMA" in _r.stdout) or ("magma" in _r.stdout.lower())))
print("BFILE:", BFILE)
print("GENE_LOC_38:", GENE_LOC_38)
print("GENE_LOC_37:", GENE_LOC_37)
# locate the extracted gwas folder (holds one subfolder per trait)
_cand = [d for d in _glob.glob(os.path.join(REFS, "**", "SLE"), recursive=True) if os.path.isdir(d)]
assert _cand, "extracted gwas/SLE folder not found"
GWAS_ROOT = os.path.dirname(_cand[0])
print("GWAS_ROOT:", GWAS_ROOT)
"""

SUMSTATS_PREP = """# ---------- Sumstats -> MAGMA inputs (snp-loc + pval with N column) ----------
# Column mappings verified against the live file headers on 2026-11-10.
# build: '38' -> NCBI38 gene locations, '37' -> NCBI37.3 (annotation uses each
# file's OWN chr/pos; gene analysis matches the EUR reference by rsID).
# MS and Hair_Colour carry no rsID column -> mapped through the g1000_eur .bim
# (chr:pos, GRCh37 - both files are GRCh37, so the mapping is build-consistent).
TRAIT_SPEC = {
 # key                    folder               file                                                              sep     rsid          chr             pos                    p / neglog10        N fixed  N col  build
 "SLE":                   ("SLE",               "bentham_2015_26502338_sle_efo0002690_1_gwas.sumstats.tsv.gz",       "\\t",  "rsid",       "chrom",        "pos",                 "p",                23210,   None,  "37"),
 "RA":                    ("RA",                "GCST90132223.h.tsv.gz",                                               "\\t",  "rsid",       "chromosome",   "base_pair_location",  "p_value",          97173,   None,  "38"),
 "IBD":                   ("IBD",               "28067908-GCST004131-EFO_0003767.h.tsv.gz",                              "\\t",  "variant_id", "chromosome",   "base_pair_location",  "p_value",          59957,   None,  "38"),
 "MS":                    ("MS",                "GCST90558092.tsv",                                                    "\\t",  None,         "chromosome",   "base_pair_location",  "p_value",          360004,  None,  "37"),
 "T1D":                   ("T1D",               "34012112-GCST90014023-EFO_0001359.h.tsv.gz",                            "\\t",  "variant_id", "chromosome",   "base_pair_location",  "p_value",          520580,  None,  "38"),
 "Celiac":                ("CELIAC",            "20190752-GCST000612-EFO_0001060.h.tsv.gz",                              "\\t",  "variant_id", "chromosome",   "base_pair_location",  "p_value",          15283,   None,  "38"),
 "DLBCL":                 ("DLBCL",             "GCST90624739.h.tsv.gz",                                               "\\t",  "rsid",       "chromosome",   "base_pair_location",  "p_value",          658535,  None,  "38"),
 "FL":                    ("FL",                "GCST90624741.h.tsv.gz",                                               "\\t",  "rsid",       "chromosome",   "base_pair_location",  "p_value",          657982,  None,  "38"),
 "CLL":                   ("CLL",               "GCST90624738.h.tsv.gz",                                               "\\t",  "rsid",       "chromosome",   "base_pair_location",  "p_value",          1108777, None,  "38"),
 "Lymphocyte_Count":      ("LYMPHOCYTE_COUNT",  "32888493-GCST90002316-EFO_0004587.h.tsv.gz",                            "\\t",  "variant_id", "chromosome",   "base_pair_location",  "p_value",          524923,  None,  "38"),
 "Adult_Standing_Height": ("HEIGHT",            "GIANT_HEIGHT_YENGO_2022_GWAS_SUMMARY_STATS_EUR.gz",                   "\\t",  "RSID",       "CHR",          "POS",                 "P",                None,    "N",   "37"),
 "Heel_BMD":              ("HEEL_BMD",          "30598549-GCST006979-EFO_0009270.h.tsv.gz",                              "\\t",  "variant_id", "chromosome",   "base_pair_location",  "p_value",          None,    "n",   "38"),
 "Hair_Colour":           ("HAIR_COLOUR",       "hair_panukbb_1747_blonde.tsv.bgz",                                    "\\t",  None,         "chr",          "pos",                 "neglog10_pval_EUR",419469,  None,  "37"),
 "ASD":                   ("ASD",               "iPSYCH-PGC_ASD_Nov2017.gz",                                           "\\t",  "SNP",        "CHR",          "BP",                  "P",                46350,   None,  "37"),
 "Cortical_Surface_Area": ("CORTICAL_SURFACE_AREA", "ENIGMA3_mixed_se_wo_Mean_Full_SurfArea_20190429.txt.gz",          "\\\\s+", "SNP",        "CHR",          "BP",                  "P",                None,    "N",   "37"),
 "Cortical_Thickness":    ("CORTICAL_THICKNESS",    "ENIGMA3_mixed_se_wo_Mean_Full_Thickness_20190429.txt.gz",          "\\\\s+", "SNP",        "CHR",          "BP",                  "P",                None,    "N",   "37"),
}

BIM = None
def load_bim():
    global BIM
    if BIM is None:
        bim = pd.read_csv(BFILE + ".bim", sep="\\t", header=None,
                          names=["CHR", "SNP", "CM", "BP", "A1", "A2"], dtype={"CHR": str, "SNP": str, "BP": np.int64})
        BIM = bim.drop_duplicates(subset=["CHR", "BP"])[["CHR", "BP", "SNP"]]
    return BIM

def prep_trait(key, spec):
    folder, fname, sep, rsid_c, chr_c, pos_c, p_c, n_fixed, n_col, build = spec
    outdir = os.path.join(MAGMA_WORK, key)
    os.makedirs(outdir, exist_ok=True)
    snp_loc_f = os.path.join(outdir, key + ".snp_loc.txt")
    pval_f = os.path.join(outdir, key + ".pval.txt")
    if os.path.exists(snp_loc_f) and os.path.exists(pval_f):
        print(key, ": already prepped"); return snp_loc_f, pval_f, build
    path = os.path.join(GWAS_ROOT, folder, fname)
    assert os.path.exists(path), f"{key}: sumstats file missing: {path}"
    want = {chr_c, pos_c, p_c} | ({rsid_c} if rsid_c else set()) | ({n_col} if n_col else set())
    _comp = "gzip" if (fname.endswith(".gz") or fname.endswith(".bgz")) else None
    df = pd.read_csv(path, sep=sep, engine="python" if sep == "\\\\s+" else "c", compression=_comp,
                     usecols=lambda c: c in want or c == "low_confidence_EUR")
    df = df.rename(columns={chr_c: "CHR", pos_c: "BP", p_c: "PRAW"})
    if key == "Hair_Colour":
        if "low_confidence_EUR" in df.columns:
            df = df[df["low_confidence_EUR"].astype(str).str.lower() == "false"]
        df["P"] = np.power(10.0, -df["PRAW"].astype(float))
    else:
        df["P"] = df["PRAW"].astype(float)
    df["CHR"] = df["CHR"].astype(str).str.replace("^chr", "", regex=True)
    df["BP"] = pd.to_numeric(df["BP"], errors="coerce")
    df = df.dropna(subset=["BP", "P"])
    df["BP"] = df["BP"].astype(np.int64)
    if rsid_c:
        df["SNP"] = df[rsid_c].astype(str)
    else:
        bim = load_bim()
        df = df.merge(bim, on=["CHR", "BP"], how="inner")
        print(f"{key}: bim-mapped {len(df)} rows")
    df = df[df["SNP"].str.startswith("rs", na=False)]
    df["P"] = df["P"].clip(lower=1e-300, upper=1.0)
    df["N"] = pd.to_numeric(df[n_col], errors="coerce").fillna(n_fixed) if n_col else float(n_fixed)
    df = df.sort_values("P").drop_duplicates(subset=["SNP"], keep="first")
    df[["SNP", "CHR", "BP"]].to_csv(snp_loc_f, sep="\\t", index=False, header=False)
    df[["SNP", "P", "N"]].to_csv(pval_f, sep="\\t", index=False)
    print(f"{key}: {len(df)} SNPs prepped (build {build})")
    return snp_loc_f, pval_f, build

PREPPED = {}
for _k, _spec in TRAIT_SPEC.items():
    PREPPED[_k] = prep_trait(_k, _spec)
print("prepped traits:", len(PREPPED))
"""

MAGMA_RUN = """# ---------- MAGMA: annotate + gene analysis per trait (checkpointed) ----------
import time
MAGMA_OUT = os.path.join(OUTDIR, "magma")
os.makedirs(MAGMA_OUT, exist_ok=True)
GENES_OUT = {}
for key, (snp_loc_f, pval_f, build) in PREPPED.items():
    gene_loc = GENE_LOC_38 if build == "38" else GENE_LOC_37
    prefix = os.path.join(MAGMA_WORK, key, key)
    t0 = time.time()
    if not os.path.exists(prefix + ".genes.annot"):
        r = subprocess.run([MAGMA, "--annotate", "window=10,1.5", "--snp-loc", snp_loc_f,
                            "--gene-loc", gene_loc, "--out", prefix],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"{key} annotate failed: {r.stdout[-500:]} {r.stderr[-500:]}"
    if not os.path.exists(prefix + ".genes.out"):
        r = subprocess.run([MAGMA, "--bfile", BFILE, "--pval", pval_f, "ncol=N",
                            "--gene-annot", prefix + ".genes.annot", "--out", prefix],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"{key} gene analysis failed: {r.stdout[-800:]} {r.stderr[-500:]}"
    dst = os.path.join(MAGMA_OUT, key + ".genes.out")
    shutil.copy(prefix + ".genes.out", dst)
    GENES_OUT[key] = dst
    if os.path.exists(prefix + ".genes.annot"):
        os.remove(prefix + ".genes.annot")  # reclaim disk; re-annotate on rerun if needed
    print(f"{key}: genes.out ok ({(time.time()-t0)/60:.1f} min)")
print("MAGMA done:", len(GENES_OUT), "traits")
push_outputs("pilot-real: MAGMA genes.out for 16 traits (A1+D1 run)")
"""

SECTION_PREP = """# ---------- PREP A1 + D1: REAL gene annotation join + ADT/annotation alignment ----------
# Annotation: repo GENCODE v44 table (gene_id, gene_symbol, chrom, start, end, biotype).
# Join on the h5ad var['gene_ids'] (Ensembl), unique-symbol fallback, as verified
# in the Step-2 prep (18,079 / 18,085 matched). Unmatched genes get biotype
# 'unannotated' + chrom '.' so Stage 1 drops them.
import urllib.request
_anno_csv = os.path.join(WORK, "gene_annotation_GRCh38_gencode_v44.csv")
if not os.path.exists(_anno_csv):
    urllib.request.urlretrieve(f"{RAW_BASE}/resources/gene_annotation_GRCh38_gencode_v44.csv", _anno_csv)
ANNO = pd.read_csv(_anno_csv)
_by_id = {c: dict(zip(ANNO["gene_id"].astype(str), ANNO[c])) for c in ("chrom", "start", "end", "biotype")}
_by_sym = {c: dict(zip(ANNO["gene_symbol"].astype(str), ANNO[c])) for c in ("chrom", "start", "end", "biotype")}

SEC_PREP = {}
ADATAS = {}
for sec, spec in SECTIONS.items():
    sdir = SEC_DIRS[sec]
    rna_path = os.path.join(sdir, "adata_RNA.h5ad")
    adt_path = os.path.join(sdir, "adata_ADT.h5ad")
    anno_path = os.path.join(sdir, "annotation.csv")
    E, C = EMB[sec]["E"], EMB[sec]["C"]
    adata_rna = sc.read_h5ad(rna_path)
    assert adata_rna.n_obs == E.shape[0], f"{sec}: RNA spots != embedding rows - ALIGNMENT BROKEN, STOP"
    gids = adata_rna.var["gene_ids"].astype(str).values
    syms = adata_rna.var_names.astype(str).values
    for col in ("chrom", "start", "end", "biotype"):
        vals = np.array([_by_id[col].get(g, _by_sym[col].get(s)) for g, s in zip(gids, syms)], dtype=object)
        adata_rna.var[col] = vals
    n_unmatched = int(adata_rna.var["chrom"].isna().sum())
    adata_rna.var["chrom"] = adata_rna.var["chrom"].fillna(".").astype(str)
    adata_rna.var["start"] = pd.to_numeric(adata_rna.var["start"], errors="coerce").fillna(0).astype(np.int64)
    adata_rna.var["end"] = pd.to_numeric(adata_rna.var["end"], errors="coerce").fillna(0).astype(np.int64)
    adata_rna.var["biotype"] = adata_rna.var["biotype"].fillna("unannotated").astype(str)
    print(f"{sec}: annotation joined; unmatched genes: {n_unmatched} / {adata_rna.n_vars}")
    out_rna = os.path.join(WORK, f"adata_RNA_{sec}_annot.h5ad")
    adata_rna.write_h5ad(out_rna)
    ADATAS[sec] = adata_rna

    adata_adt = sc.read_h5ad(adt_path)
    assert adata_adt.n_obs == adata_rna.n_obs, f"{sec}: ADT spots != RNA spots - STOP"
    if list(adata_adt.obs_names) != list(adata_rna.obs_names):
        assert set(adata_adt.obs_names) == set(adata_rna.obs_names), f"{sec}: ADT barcodes mismatch - STOP"
        adata_adt = adata_adt[list(adata_rna.obs_names)]
        print(f"{sec}: ADT reordered to RNA spot order")
    out_adt = os.path.join(WORK, f"adata_ADT_{sec}_prep.h5ad")
    adata_adt.write_h5ad(out_adt)

    anno = pd.read_csv(anno_path, index_col=0)
    assert "manual-anno" in anno.columns, f"{sec}: annotation.csv has no manual-anno column"
    if set(anno.index.astype(str)) == set(adata_rna.obs_names.astype(str)):
        anno = anno.reindex(adata_rna.obs_names)
        assert anno["manual-anno"].notna().all(), f"{sec}: annotation reindex NaNs - STOP"
        out_anno = os.path.join(WORK, f"annotation_{sec}_aligned.csv")
        anno.to_csv(out_anno)
        print(f"{sec}: annotation aligned to RNA spot order")
    else:
        out_anno = anno_path
        print(f"{sec}: WARNING annotation index != RNA barcodes; driver will read it positionally")
    SEC_PREP[sec] = {"rna": out_rna, "adt": out_adt, "anno": out_anno}
print("section prep done:", SEC_PREP)
"""

EXTERNALS = """# ---------- Build magma_z_dict + gene_covariates from MAGMA outputs ----------
import pickle
def _read_gene_loc(path):
    m = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\\n").split("\\t")
            if len(parts) >= 6:
                m[parts[0]] = parts[5]
    return m
LOC_MAP = {"38": _read_gene_loc(GENE_LOC_38), "37": _read_gene_loc(GENE_LOC_37)}

magma_z_dict = {}
_nsnps = {}
for key, (_, _, build) in PREPPED.items():
    go = pd.read_csv(GENES_OUT[key], sep="\\\\s+", engine="python")
    go["symbol"] = go["GENE"].astype(str).map(LOC_MAP[build])
    go = go.dropna(subset=["symbol"])
    go["_absz"] = go["ZSTAT"].abs()
    go = go.sort_values("_absz", ascending=False).drop_duplicates(subset=["symbol"], keep="first")
    magma_z_dict[key] = pd.Series(go["ZSTAT"].values, index=go["symbol"].values)
    _nsnps[key] = pd.Series(go["NSNPS"].values, index=go["symbol"].values)
    print(f"{key}: {len(magma_z_dict[key])} genes with z-scores")

_panel_genes = set()
for sec in SECTIONS:
    _panel_genes |= set(ADATAS[sec].var_names.astype(str))
_cov_syms = sorted(_panel_genes | set().union(*[set(s.index) for s in magma_z_dict.values()]))
_len_map = {}
for g, s, e in zip(ANNO["gene_symbol"].astype(str), ANNO["start"], ANNO["end"]):
    _len_map.setdefault(g, float(e) - float(s) + 1.0)
_nsnps_med = pd.DataFrame(_nsnps).median(axis=1, skipna=True)
_expr = {}
for sec in SECTIONS:
    X = ADATAS[sec].X
    if hasattr(X, "toarray"):
        import scipy.sparse as _sp
        Xl = X.copy()
        Xl.data = np.log1p(Xl.data)
        m = np.asarray(Xl.mean(axis=0)).ravel()
    else:
        m = np.log1p(np.asarray(X)).mean(axis=0)
    _expr[sec] = pd.Series(m, index=ADATAS[sec].var_names.astype(str))
_expr_mean = pd.DataFrame(_expr).mean(axis=1)
gene_covariates = pd.DataFrame(index=_cov_syms)
gene_covariates["log10_gene_length"] = np.log10(pd.Series(_len_map).reindex(_cov_syms).fillna(30000.0))
gene_covariates["log10_median_nsnps"] = np.log10(_nsnps_med.reindex(_cov_syms).fillna(_nsnps_med.median()))
gene_covariates["mean_log_expr"] = _expr_mean.reindex(_cov_syms).fillna(0.0)
gene_covariates = gene_covariates[~gene_covariates.index.duplicated(keep="first")]
print("covariates:", gene_covariates.shape)

with open(os.path.join(OUTDIR, "magma_z_dict.pkl"), "wb") as f:
    pickle.dump(magma_z_dict, f)
gene_covariates.to_csv(os.path.join(OUTDIR, "gene_covariates.csv"))
for key, s in magma_z_dict.items():
    cov = len(set(s.index) & _panel_genes)
    print(f"coverage {key}: {cov} / {len(_panel_genes)} panel genes")
"""

PANEL_OVERRIDE = """# ---------- PANEL OVERRIDE (pilot): FL and DLBCL as separate traits ----------
# hello.md's panel names a combined "Follicular_Lymphoma_DLBCL" trait, but no
# combined GWAS exists - MAGMA produced separate DLBCL and FL z-scores.
# The triangulation reads TRAIT_PANELS_LYMPH_NODE at call time, so mutating
# D_disc here is enough; D_neg and D_spec are untouched.
TRAIT_PANELS_LYMPH_NODE["D_disc"] = ["SLE", "RA", "IBD", "MS", "T1D", "Celiac",
                                     "DLBCL", "FL", "CLL", "Lymphocyte_Count"]
_all = (TRAIT_PANELS_LYMPH_NODE["D_disc"] + TRAIT_PANELS_LYMPH_NODE["D_neg"]
        + TRAIT_PANELS_LYMPH_NODE["D_spec"])
assert set(_all) == set(magma_z_dict.keys()), (set(_all) ^ set(magma_z_dict.keys()))
print("panel traits (16):", _all)
"""

RUNLIB = """# ---------- Run one section + save + checkpoint ----------
import time, json
RES = {}
def run_and_save(sec):
    t0 = time.time()
    res = run_single_lymph_node_section(
        section_id=sec,
        rna_h5ad_path=SEC_PREP[sec]["rna"],
        adt_h5ad_path=SEC_PREP[sec]["adt"],
        embedding_E=EMB[sec]["E"],
        domain_C=EMB[sec]["C"],
        annotation_csv_path=SEC_PREP[sec]["anno"],
        magma_z_dict=magma_z_dict,
        gene_covariates=gene_covariates,
        sldsc_lookup_plus=None,     # as-is: Engine 2 internal OLS-proxy fallback
        sldsc_lookup_minus=None,
    )
    RES[sec] = res
    df = res["results_df"]
    mins = (time.time() - t0) / 60
    print(f"{sec}: run finished in {mins:.1f} min | results_df {df.shape}")
    print("tier counts:\\n", df["tier"].value_counts())
    print("M_eff:", res["M_eff"], "| recalibrated:", res["recalibrated"])
    print("GT post-hoc: ARI =", res["ground_truth_eval"]["ARI"], "| NMI =", res["ground_truth_eval"]["NMI"])
    neg = df[df["trait"].isin(TRAIT_PANELS_LYMPH_NODE["D_neg"])]
    print("NEGATIVE CONTROLS (read these first):")
    print(neg[["domain", "trait", "tier", "q_omni"]].to_string(index=False))
    df.to_csv(os.path.join(OUTDIR, f"{sec}_results_df.csv"), index=False)
    res["ground_truth_eval"]["jaccard_table"].to_csv(os.path.join(OUTDIR, f"{sec}_gt_jaccard.csv"))
    summary = {
        "run": "hello-pipeline REAL-DATA PILOT - real MAGMA z + covariates + GENCODE v44 annotation; engines as-is (E2 OLS proxy, no Pp RSR, E3 code null); FL/DLBCL separate traits (16 total)",
        "section": sec, "n_perm_initial": REAL_N_PERM,
        "runtime_min": round(mins, 1),
        "E_shape": list(EMB[sec]["E"].shape), "n_domains": int(len(np.unique(EMB[sec]["C"]))),
        "M_eff": float(res["M_eff"]), "recalibrated": bool(res["recalibrated"]),
        "tier_counts": {k: int(v) for k, v in df["tier"].value_counts().items()},
        "gt_ARI": float(res["ground_truth_eval"]["ARI"]), "gt_NMI": float(res["ground_truth_eval"]["NMI"]),
    }
    with open(os.path.join(OUTDIR, f"{sec}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    push_outputs(f"pilot-real: {sec} results (real MAGMA, as-is engines)")
    return res
"""

RUN_A1 = """res_a1 = run_and_save("A1")"""
RUN_D1 = """res_d1 = run_and_save("D1")"""

STAGE4C = """# ---------- Stage 4C: canonical matching + replication (A1 x D1) ----------
s4c = stage4c_canonical_donor_matching_and_replication(
    RES["A1"]["v_domain_rna"], RES["A1"]["genes_plus"], RES["A1"]["v_domain_adt"], RES["A1"]["adt_names"], RES["A1"]["results_df"],
    RES["D1"]["v_domain_rna"], RES["D1"]["genes_plus"], RES["D1"]["v_domain_adt"], RES["D1"]["adt_names"], RES["D1"]["results_df"],
    tau_match=0.40,
)
s4c["M_corr"].to_csv(os.path.join(OUTDIR, "stage4c_M_corr.csv"))
s4c["canonical_matches"].to_csv(os.path.join(OUTDIR, "stage4c_canonical_matches.csv"), index=False)
s4c["replication_table"].to_csv(os.path.join(OUTDIR, "stage4c_replication_table.csv"), index=False)
print("canonical matches:")
print(s4c["canonical_matches"].to_string(index=False))
print("final replication tiers:\\n", s4c["replication_table"]["final_replication_tier"].value_counts())
push_outputs("pilot-real: Stage 4C matching + replication (A1 x D1)")
print("PILOT RUN COMPLETE - outputs in", OUTDIR)
"""


def _code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}

def _md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def main():
    with open(BASE_NB) as f:
        base = json.load(f)
    cells = base["cells"]
    assert len(cells) == 10, f"base notebook should have 10 cells, got {len(cells)}"
    code_block1 = "".join(cells[4]["source"])
    driver = "".join(cells[5]["source"])
    assert "def stage1_harmonize_lymph_node" in code_block1, "cell 4 is not hello.md block 1"
    assert "def run_single_lymph_node_section" in driver, "cell 5 is not the driver"
    n_tok = driver.count("SMOKE_N_PERM")
    assert n_tok >= 2, f"expected >=2 SMOKE_N_PERM tokens in driver, got {n_tok}"
    driver = driver.replace("SMOKE_N_PERM", "REAL_N_PERM")

    nb_cells = [
        _md(MD),
        _code(INSTALL),
        _code(CONFIG),
        _code(PUSH_HELPER),
        _code(REFS_SETUP),
        _code(SUMSTATS_PREP),
        _code(MAGMA_RUN),
        _code(SECTION_PREP),
        _code(EXTERNALS),
        _code(code_block1),
        _code(driver),
        _code(PANEL_OVERRIDE),
        _code(RUNLIB),
        _code(RUN_A1),
        _code(RUN_D1),
        _code(STAGE4C),
    ]
    for _i, _c in enumerate(nb_cells):
        _c["id"] = f"pilot-{_i:02d}"
    nb = {"cells": nb_cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.10"}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb_cells):
        if c["cell_type"] == "code":
            compile("".join(c["source"]), f"cell{i}", "exec")
    out_path = os.path.join(HERE, "hello_pipeline_pilot_real.ipynb")
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path, os.path.getsize(out_path), "bytes,", len(nb_cells), "cells")

    import nbformat
    nbformat.read(out_path, as_version=4)
    text = open(out_path).read()
    for token in ["TRAIT_SPEC", "NCBI37.3.gene.loc", "window=10,1.5", "ncol=N",
                  "hello-pipeline-refs", "spatial-multiomics-6datasets-private",
                  "gene_annotation_GRCh38_gencode_v44.csv", "REAL_N_PERM = 1000",
                  "n_perm_initial=REAL_N_PERM", "stage4c_canonical_donor_matching_and_replication",
                  "Follicular_Lymphoma_DLBCL", "neglog10_pval_EUR", "res_a1 = run_and_save",
                  "res_d1 = run_and_save", "tau_match=0.40"]:
        assert token in text, f"MISSING TOKEN: {token}"
    for bad in ["SMOKE_N_PERM", "dummy", "DUMMY", "generate_dummy_external_data"]:
        assert bad not in text, f"STALE TOKEN PRESENT: {bad}"
    print("content check: all real-run tokens present, zero smoke/dummy tokens")

if __name__ == "__main__":
    main()
