"""Generate notebooks/stage8_combo1_lymph_node_calibrated.ipynb

Stage 8, Notebook 1 (Combo 1): Calibrated v2 disease-association pipeline
on the human lymph node pair (A1 + D1), built to PULOK's locked answers
(2026-11-05):

- Scope: Combo 1 ONLY (A1 K=10, D1 K=11). Mouse deferred to Stage 8b.
- Frozen S5N-BASE/standard/kmeans seed-42 embeddings, found by exact
  filename (same convention as the Stage 7 exact notebook); domains by the
  fixed recipe (StandardScaler + KMeans(DECLARED_K, seed 42)).
- Data: pulokpulok/spatial-multiomics-6datasets-private; per folder
  adata_RNA.h5ad + adata_ADT.h5ad + annotation.csv (Stage 5/7 spec).
- ADT: Stage 5 CLR convention reused verbatim
  (clr(X) = log1p(X / exp(mean(log1p(X), axis=1)))); name cleaning +
  explicit alias->HGNC dictionary; isotype controls dropped; the full
  raw->HGNC mapping table is printed and saved.
- Specificity: cosine kNN graph (k=15) on the frozen embedding;
  GSS = per-gene z-score (across spots) of graph-smoothed expression.
- Calibration (the core fix): cluster-level Monte Carlo nulls sampled
  from 5x5 mean x variance expression bins (P=1000, seed 42), computed on
  the CLUSTER null distribution itself (never naive Stouffer across
  autocorrelated spots). Ablation built in: the same scores are also run
  against an UNMATCHED uniform null, so the Height fix is shown as
  unmatched -> matched on identical gene sets.
- Trait gene sets: Stage 7 (verified 2026-11-05) contains NO gene-set
  files - runs/s7-disease holds only the Height/IQ Cauchy tables, and
  Stage 7's traits were GWAS sumstats for S-LDSC, a different statistic.
  The notebook therefore (1) uses a repo-cached trait_gene_sets.tsv if
  one exists, else (2) fetches sets live from the OpenTargets Platform
  API v4 (disease search -> associatedTargets ranked by the
  genetic_association datatype score, top 200; API verified live
  2026-11-05 incl. body height OBA_VT0001253 and heel BMD EFO_0009270),
  and caches everything into runs/s8-disease so the sets are versioned.
- A1<->D1 matching: Spearman rho on concatenated standardised
  [GSS_RNA ; GSS_ADT] cluster profiles; Hungarian assignment + reciprocal
  best hits; pairs kept at rho >= 0.4 (the 11th D1 cluster is never
  force-matched); cross-donor trait-Z concordance on matched pairs.
- Drugs: DGIdb v5 GraphQL (verified live) + OpenTargets tractability /
  clinical candidates (verified live), fetched for ADT surface proteins
  and top RNA hub genes of (replicated) vulnerable clusters; raw API
  responses cached into the output folder.
- Output: runs/s8-disease/ (tables, plots, caches, run log), pushed by
  the notebook itself (commit -> pull --rebase -> push).

Terminology: outputs say "spatial enrichment of disease-relevant
programmes" / "cellular vulnerability mapping" only.

Run: python3 scripts/gen_stage8_combo1_nb.py
"""
import json, os

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

cells = []

cells.append(md("""# Stage 8 (Combo 1) — Calibrated Disease Association on Human Lymph Node (A1 + D1)

**What this notebook does.** Takes the *frozen* S5N-BASE / standard / kmeans seed-42 embeddings for A1 and D1 and runs the Calibrated v2 post-hoc pipeline: embedding-smoothed gene/protein specificity (GSS) -> trait scoring against **expression-matched Monte Carlo nulls** -> A1<->D1 replication in canonical feature space -> ADT surfaceome drug prioritisation (DGIdb v5 + OpenTargets). All results are framed as **spatial enrichment of disease-relevant programmes** (cellular vulnerability mapping) in healthy reference tissue. Association is reporting-only: nothing here feeds back into training, clustering, or model selection.

**Why calibration is the headline.** In the Stage 7 gsMap pilot, the negative-control trait Height came back at Cauchy p = 7.3e-15 on A1 — a false positive produced by expression/housekeeping bias. This notebook runs an explicit ablation on identical gene sets: every trait score is tested against (a) an **unmatched** uniform null and (b) a **matched** null (5x5 mean x variance bins). The pipeline passes QC only if Height and Heel BMD stay non-significant (FDR >= 0.05 in every cluster) under the matched null.

**Locked parameters:** P = 1,000 permutations, seed 42, cosine kNN graph k = 15, bins 5x5 = 25, gene universe = expressed in >= 5% of spots (MT-/RPS/RPL excluded), cluster Z computed against the cluster-level null distribution (no Stouffer across autocorrelated spots), A1<->D1 matching by Spearman rho on [GSS_RNA ; GSS_ADT] with Hungarian + reciprocal-best-hit at rho >= 0.4.

**Inputs you must attach to this Kaggle notebook:** (1) the dataset `pulokpulok/spatial-multiomics-6datasets-private`, (2) a dataset containing the two embedding files `10x_human_lymph_node_A1_seed_42_embeddings.npy` and `10x_human_lymph_node_D1_seed_42_embeddings.npy` (the same upload used in Stage 7), (3) the Kaggle secret `GITHUB_TOKEN`, (4) internet ON (OpenTargets + DGIdb are fetched live and cached into the output).

**Trait gene sets — read this.** Stage 7 has no gene-set files to reuse (its traits were GWAS sumstats for S-LDSC; `runs/s7-disease/` holds only the Height/IQ Cauchy tables). So this notebook fetches trait gene sets from the OpenTargets Platform API v4 — disease search, then associated targets ranked by the **genetic_association** datatype score, top 200 genes — and caches them into `runs/s8-disease/trait_gene_sets/`, versioned in GitHub. On any later run the cached file is reused, so the sets stay frozen from then on. The Stage 7 Height result (different statistic) is kept as a reference line in the final summary, not as the ablation baseline — the ablation here is unmatched-vs-matched on the same sets.

**Runtime estimate:** roughly 30-60 minutes on a Kaggle CPU session (calibration dominates; the API stages a few minutes). No GPU needed."""))

cells.append(code("""import os, sys, json, subprocess, time

for pkg in ["scanpy", "anndata", "scikit-learn", "scipy", "pandas", "matplotlib", "requests"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

print("imports ok")
"""))

cells.append(code("""# ---------- GitHub: clone + push helper (results -> runs/s8-disease) ----------
REPO = "/kaggle/working/my_research"
REPO_URL = "https://github.com/rifatahsanpul0k/my_research.git"
def sh(*args):
    return subprocess.run(list(args), capture_output=True, text=True, cwd=REPO)
os.makedirs(REPO, exist_ok=True)
OUT = os.path.join(REPO, "runs", "s8-disease")
os.makedirs(OUT, exist_ok=True)
try:
    from kaggle_secrets import UserSecretsClient
    import base64, shutil
    _sec = UserSecretsClient()
    _tok = _sec.get_secret("GITHUB_TOKEN")
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
        sh("git", "add", "runs/s8-disease")
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
"""))

cells.append(code("""# ---------- CONFIG (locked answers, 2026-11-05) ----------
SEED = 42
P_PERM = 1000               # Monte Carlo permutations
KNN_K = 15                  # cosine kNN graph on the frozen embedding
N_BINS = 5                  # 5 mean x 5 variance = 25 expression bins
MIN_PCT = 0.05              # gene universe: expressed in >= 5% of spots
TOP_N_GENES = 200           # per trait, ranked by genetic_association score
RHO_MIN = 0.4               # A1<->D1 matched-pair threshold

DATASETS = {
    "A1": dict(folder="10x_human_lymph_node_A1", K=10,
               emb="10x_human_lymph_node_A1_seed_42_embeddings.npy"),
    "D1": dict(folder="10x_human_lymph_node_D1", K=11,
               emb="10x_human_lymph_node_D1_seed_42_embeddings.npy"),
}

# Full pre-registered trait list. kind: "positive" or "control".
# Stage 7's only negative control was Height (IQ was its test trait);
# Heel BMD is the second control from the locked spec.
TRAITS = [
    dict(name="DLBCL",                 query="diffuse large B-cell lymphoma",        kind="positive"),
    dict(name="Follicular Lymphoma",   query="follicular lymphoma",                  kind="positive"),
    dict(name="Hodgkin Lymphoma",      query="Hodgkin lymphoma",                     kind="positive"),
    dict(name="CLL",                   query="chronic lymphocytic leukemia",         kind="positive"),
    dict(name="SLE",                   query="systemic lupus erythematosus",         kind="positive"),
    dict(name="RA",                    query="rheumatoid arthritis",                 kind="positive"),
    dict(name="MS",                    query="multiple sclerosis",                   kind="positive"),
    dict(name="IBD",                   query="inflammatory bowel disease",           kind="positive"),
    dict(name="T1D",                   query="type 1 diabetes",                      kind="positive"),
    dict(name="Asthma",                query="asthma",                               kind="positive"),
    dict(name="T-ALL",                 query="T-cell acute lymphoblastic leukemia",  kind="positive"),
    dict(name="Height",                query="height",                               kind="control"),
    dict(name="Heel BMD",              query="heel bone mineral density",            kind="control"),
]
CONTROLS = [t["name"] for t in TRAITS if t["kind"] == "control"]

# ADT alias -> HGNC (applied AFTER generic prefix/suffix cleaning).
ADT_ALIAS = {
    "CD20": "MS4A1", "PD-1": "PDCD1", "PD1": "PDCD1", "CD21": "CR2",
    "CD45RA": "PTPRC", "CD45RO": "PTPRC", "CD45": "PTPRC",
    "HLA-DR": "HLA-DRA", "CD3": "CD3E", "CD8A": "CD8A", "CD8a": "CD8A",
    "CD4": "CD4", "CD19": "CD19", "CD22": "CD22", "CD40": "CD40",
    "CXCR5": "CXCR5", "CD16": "FCGR3A", "CD56": "NCAM1", "CD57": "B3GAT1",
    "CD123": "IL3RA", "CD127": "IL7R", "CD25": "IL2RA", "CD11C": "ITGAX",
    "CD11c": "ITGAX", "CD11B": "ITGAM", "CD11b": "ITGAM",
    "CD103": "ITGAE", "CD23": "FCER2", "CD35": "CR1", "CD138": "SDC1",
    "CD79A": "CD79A", "CD79a": "CD79A", "CD90": "THY1", "CD31": "PECAM1",
    "CD34": "CD34", "CD38": "CD38", "CD27": "CD27", "CD28": "CD28",
    "CD44": "CD44", "CD69": "CD69", "CD5": "CD5", "CD7": "CD7",
    "CD14": "CD14", "CCR7": "CCR7", "CXCR3": "CXCR3", "PDPN": "PDPN",
    "PODOPLANIN": "PDPN", "IGM": "IGHM", "IGD": "IGHD",
}
ISOTYPE_TOKENS = ["isotype", "igg1", "igg2a", "igg2b", "igg2", "hashtag"]

print("config locked:", list(DATASETS), "| traits:", len(TRAITS), "| controls:", CONTROLS)
"""))

cells.append(code("""# ---------- Data discovery + defensive loader (Stage 7 conventions) ----------
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

def to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)

def looks_like_raw_counts(X):
    # integer-valued, non-negative sample -> raw counts
    if sp.issparse(X):
        vals = X.data[:200000]
    else:
        vals = np.asarray(X).ravel()[:200000]
    if vals.size == 0:
        return False
    return bool(vals.min() >= 0 and np.allclose(vals, np.round(vals), atol=1e-4))

def clr_stage5(X):
    # EXACT Stage 5 convention: clr(X) = log1p(X / exp(mean(log1p(X), axis=1)))
    X = np.asarray(X, dtype=np.float64)
    gm = np.exp(np.log1p(X).mean(axis=1, keepdims=True))
    return np.log1p(X / gm)

def load_dataset(ds):
    spec = DATASETS[ds]
    root = os.path.join(DATA_ROOT, spec["folder"])
    rna = sc.read_h5ad(os.path.join(root, "adata_RNA.h5ad"))
    rna.var_names_make_unique()

    # counts: layers['counts'] first, else X (verified raw or flagged)
    if "counts" in rna.layers:
        counts = rna.layers["counts"]
        counts_src = "layers['counts']"
    else:
        counts = rna.X
        counts_src = "X"
    counts_raw = looks_like_raw_counts(counts)
    print(f"{ds}: counts from {counts_src}, raw-count-like = {counts_raw}")
    if not counts_raw:
        print(f"  WARNING {ds}: count matrix does not look like raw counts; proceeding as-is (flagged in log)")
    counts = sp.csr_matrix(counts, dtype=np.float32)

    # gene symbols: var_names, or a symbol column if var_names are Ensembl IDs
    genes = np.asarray(rna.var_names).astype(str)
    ens_frac = np.mean([g.startswith("ENSG") for g in genes])
    if ens_frac > 0.5:
        for col in ["feature_name", "gene_name", "symbol", "gene_symbol", "gene_ids"]:
            if col in rna.var.columns:
                cand = np.asarray(rna.var[col]).astype(str)
                if np.mean([c.startswith("ENSG") for c in cand]) < 0.5:
                    genes = cand
                    print(f"{ds}: var_names are Ensembl; symbols taken from var['{col}']")
                    break
        else:
            print(f"{ds}: WARNING no symbol column found; trait sets (HGNC) will intersect poorly")

    assert "spatial" in rna.obsm, f"{ds}: obsm['spatial'] missing"
    coords = np.asarray(rna.obsm["spatial"], dtype=np.float64)

    # ADT, aligned to RNA spots
    adt = sc.read_h5ad(os.path.join(root, "adata_ADT.h5ad"))
    adt.var_names_make_unique()
    if set(adt.obs_names) == set(rna.obs_names):
        adt = adt[rna.obs_names]
    elif adt.shape[0] == rna.shape[0]:
        print(f"{ds}: WARNING ADT obs_names differ; aligning positionally")
    else:
        raise AssertionError(f"{ds}: ADT rows {adt.shape[0]} cannot align to RNA rows {rna.shape[0]}")
    adt_X = to_dense(adt.X).astype(np.float64)
    adt_raw = looks_like_raw_counts(adt.X)
    if adt_raw:
        adt_X = clr_stage5(np.maximum(adt_X, 0))
        print(f"{ds}: ADT raw counts -> Stage-5 CLR applied")
    else:
        print(f"{ds}: ADT not raw-count-like -> assumed already transformed, used as-is")

    # frozen embedding, exact filename (Stage 7 convention)
    emb_hits = (_glob.glob(f"/kaggle/input/**/{spec['emb']}", recursive=True) +
                _glob.glob(f"/kaggle/working/**/{spec['emb']}", recursive=True))
    assert emb_hits, (f"{spec['emb']} not found under /kaggle/input or /kaggle/working - "
                      "attach the embedding upload used in Stage 7 as a notebook input")
    E = np.load(sorted(emb_hits)[0]).astype(np.float32)
    print(f"{ds}: embedding {E.shape} from {sorted(emb_hits)[0]}")
    assert E.ndim == 2
    if E.shape[0] != rna.shape[0] and E.shape[1] == rna.shape[0]:
        print(f"{ds}: WARNING embedding looks transposed - transposing")
        E = E.T
    assert E.shape[0] == rna.shape[0], f"{ds}: embedding/adata row mismatch - ALIGNMENT BROKEN"

    # domains: FIXED recipe (standard-scale + KMeans, DECLARED_K, seed 42)
    Ez = StandardScaler().fit_transform(E)
    domains = KMeans(n_clusters=spec["K"], n_init=10, random_state=SEED).fit_predict(Ez)
    print(f"{ds}: domain sizes:", np.bincount(domains))

    # plumbing check ONLY (post-hoc, quarantined): ARI vs manual annotation
    ari = None
    gt = pd.read_csv(os.path.join(root, "annotation.csv"), index_col=0)
    gt = gt.reindex(rna.obs_names)
    ok = gt["manual-anno"].notna().values
    if ok.sum() > 100:
        ari = adjusted_rand_score(gt["manual-anno"].values[ok].astype(str), domains[ok])
        print(f"{ds}: ALIGNMENT CHECK - post-hoc ARI = {ari:.3f} (A1 expect ~0.22, D1 ~0.21)")
        assert ari > 0.10, f"{ds}: ARI far below expected - likely row misalignment. STOP."

    return dict(counts=counts, genes=genes, coords=coords,
                adt_X=adt_X.astype(np.float32), adt_names=np.asarray(adt.var_names).astype(str),
                E=E, domains=domains, obs_names=np.asarray(rna.obs_names).astype(str),
                ari=ari, counts_raw=counts_raw, adt_was_raw=adt_raw,
                n_spots=rna.shape[0])

DATA = {ds: load_dataset(ds) for ds in DATASETS}
validator = pd.DataFrame([{ "dataset": ds, "n_spots": d["n_spots"], "n_genes_rna": len(d["genes"]),
                            "n_proteins_adt_raw": len(d["adt_names"]), "embedding_dim": d["E"].shape[1],
                            "K": DATASETS[ds]["K"], "posthoc_ARI": d["ari"],
                            "counts_raw": d["counts_raw"], "adt_was_raw": d["adt_was_raw"]}
                          for ds, d in DATA.items()])
validator.to_csv(os.path.join(OUT, "io_validator_report.csv"), index=False)
print(validator)
"""))

cells.append(code("""# ---------- ADT name cleaning + HGNC mapping (nothing silently dropped) ----------
import re
def clean_adt_name(raw):
    s = str(raw).strip()
    s = re.sub(r"^(anti[- ]human\\s+|anti[- ]mouse\\s+|anti\\s+|Hu\\.?\\s+)", "", s, flags=re.I)
    s = re.sub(r"(_TotalSeq[ABC]?|_TotalSeq|TotalSeq[ABC])$", "", s, flags=re.I)
    s = re.sub(r"_[0-9]+$", "", s)
    return s.strip(" ._-")

for ds, d in DATA.items():
    rows, keep_idx, hgnc_names = [], [], []
    agg = {}
    for j, raw in enumerate(d["adt_names"]):
        cleaned = clean_adt_name(raw)
        low = cleaned.lower()
        if any(tok in low for tok in ISOTYPE_TOKENS):
            rows.append((raw, cleaned, "", "DROPPED isotype/control"))
            continue
        hgnc = ADT_ALIAS.get(cleaned, ADT_ALIAS.get(cleaned.upper(), cleaned.upper()))
        rows.append((raw, cleaned, hgnc, "kept"))
        if hgnc in agg:
            agg[hgnc].append(j)
        else:
            agg[hgnc] = [j]
    maptab = pd.DataFrame(rows, columns=["raw_name", "cleaned", "hgnc", "status"])
    maptab.to_csv(os.path.join(OUT, f"adt_mapping_{ds}.csv"), index=False)
    print(f"===== {ds} ADT mapping ({len(rows)} raw -> {len(agg)} HGNC) =====")
    print(maptab.to_string(index=False))
    # aggregate duplicates (e.g. CD45RA + CD45RO -> PTPRC) by mean
    cols, names = [], []
    for hgnc, idxs in agg.items():
        cols.append(d["adt_X"][:, idxs].mean(axis=1))
        names.append(hgnc)
    d["adt_hgnc_X"] = np.stack(cols, axis=1).astype(np.float32)
    d["adt_hgnc"] = np.asarray(names)
    print(f"{ds}: ADT matrix for specificity: {d['adt_hgnc_X'].shape}")
"""))

cells.append(code("""# ---------- Specificity: cosine-kNN smoothing on the frozen embedding -> GSS ----------
from sklearn.neighbors import NearestNeighbors

def quintile_bins(v, nb=N_BINS):
    v = np.asarray(v, dtype=np.float64)
    order = np.argsort(v, kind="mergesort")
    b = np.empty(len(v), dtype=int)
    b[order] = (np.arange(len(v)) * nb) // len(v)
    return b

def knn_smooth_W(E, k=KNN_K):
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(E)
    dist, idx = nn.kneighbors(E)          # includes self (distance 0)
    sim = np.clip(1.0 - dist, 0.0, None).astype(np.float32)
    rows = np.repeat(np.arange(E.shape[0]), k)
    W = sp.csr_matrix((sim.ravel(), (rows, idx.ravel())), shape=(E.shape[0], E.shape[0]))
    rs = np.asarray(W.sum(axis=1)).ravel()
    rs[rs == 0] = 1.0
    return W.multiply(1.0 / rs[:, None]).tocsr()

def zscore_cols(M):
    mu = M.mean(axis=0)
    sd = M.std(axis=0)
    sd[sd == 0] = 1.0
    return ((M - mu) / sd).astype(np.float32), mu, sd

for ds, d in DATA.items():
    counts = d["counts"]
    genes = d["genes"]
    frac_expr = np.asarray((counts > 0).mean(axis=0)).ravel()
    upper = np.char.upper(genes.astype(str))
    bad_prefix = np.array([g.startswith(("MT-", "RPS", "RPL")) for g in upper])
    uni = (frac_expr >= MIN_PCT) & (~bad_prefix)
    print(f"{ds}: gene universe {uni.sum()} / {len(genes)} (min_pct={MIN_PCT}, MT/RPS/RPL excluded)")
    d["uni_mask"] = uni
    d["uni_genes"] = genes[uni]
    d["gene_index"] = {g: i for i, g in enumerate(d["uni_genes"])}

    rowsum = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    rowsum[rowsum == 0] = 1.0
    Xnorm = np.log1p(counts[:, uni].toarray().astype(np.float32) / rowsum[:, None] * 1e4)
    W = knn_smooth_W(d["E"])
    smoothed = np.asarray(W @ Xnorm, dtype=np.float32)
    gss, mu_s, _ = zscore_cols(smoothed)
    d["gss_rna"] = gss
    d["bin_id"] = quintile_bins(mu_s) * N_BINS + quintile_bins(smoothed.var(axis=0))
    pd.DataFrame({"gene": d["uni_genes"], "mean_smoothed": mu_s,
                  "var_smoothed": smoothed.var(axis=0), "bin_id": d["bin_id"]}
                 ).to_csv(os.path.join(OUT, f"gene_bins_{ds}.csv"), index=False)

    adt_scaled = StandardScaler().fit_transform(d["adt_hgnc_X"])
    adt_smooth = np.asarray(W @ adt_scaled, dtype=np.float32)
    gss_adt, _, _ = zscore_cols(adt_smooth)
    d["gss_adt"] = gss_adt

    dom = d["domains"]
    prof_rna = np.stack([gss[dom == k].mean(axis=0) for k in range(DATASETS[ds]["K"])])
    prof_adt = np.stack([gss_adt[dom == k].mean(axis=0) for k in range(DATASETS[ds]["K"])])
    d["prof_rna"], d["prof_adt"] = prof_rna, prof_adt
    pd.DataFrame(prof_rna, columns=d["uni_genes"],
                 index=[f"{ds}_C{k}" for k in range(DATASETS[ds]["K"])]
                 ).to_csv(os.path.join(OUT, f"cluster_GSS_RNA_{ds}.csv"))
    pd.DataFrame(prof_adt, columns=d["adt_hgnc"],
                 index=[f"{ds}_C{k}" for k in range(DATASETS[ds]["K"])]
                 ).to_csv(os.path.join(OUT, f"cluster_GSS_ADT_{ds}.csv"))
    print(f"{ds}: GSS_RNA {gss.shape}, GSS_ADT {gss_adt.shape}, bins {len(np.unique(d['bin_id']))} unique")
"""))

cells.append(code("""# ---------- Trait gene sets: repo cache first, else OpenTargets API v4 (live-verified) ----------
import requests
OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"
SET_DIR = os.path.join(OUT, "trait_gene_sets")
CACHE_DIR = os.path.join(OUT, "cache", "opentargets")
os.makedirs(SET_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
SET_TSV = os.path.join(SET_DIR, "trait_gene_sets.tsv")

def gql(query, variables, tries=3):
    last = None
    for i in range(tries):
        try:
            r = requests.post(OT_URL, json={"query": query, "variables": variables}, timeout=60)
            r.raise_for_status()
            out = r.json()
            if "errors" in out:
                raise RuntimeError(str(out["errors"])[:300])
            return out["data"]
        except Exception as e:
            last = e
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"OpenTargets call failed after {tries} tries: {last}")

Q_SEARCH = "query($q: String!){ search(queryString: $q, entityNames: [\\"disease\\"], page: {index: 0, size: 5}) { hits { id name } } }"
Q_ASSOC = ("query($id: String!, $idx: Int!){ disease(efoId: $id){ name "
           "associatedTargets(page: {index: $idx, size: 250}){ count rows { target { approvedSymbol } "
           "score datatypeScores { id score } } } } }")

def fetch_trait_set(tr):
    data = gql(Q_SEARCH, {"q": tr["query"]})
    hits = data["search"]["hits"]
    assert hits, f"no OpenTargets disease hit for {tr['query']}"
    exact = [h for h in hits if h["name"].lower() == tr["query"].lower()]
    chosen = exact[0] if exact else hits[0]
    # Page through associatedTargets until TOP_N_GENES genetic-association
    # genes are collected (cap 12 pages = 3000 rows). This matters: for
    # cancers the overall ranking is dominated by somatic/literature
    # evidence (verified 2026-11-05: only 5 of DLBCL's top-500 overall
    # targets have genetic_association > 0), so a fixed 500-row window
    # would silently mix evidence types across traits.
    rows, page, total = [], 0, None
    while page < 12:
        d = gql(Q_ASSOC, {"id": chosen["id"], "idx": page})
        at = d["disease"]["associatedTargets"]
        total = at["count"]
        for r in at["rows"]:
            gen = next((x["score"] for x in r["datatypeScores"] if x["id"] == "genetic_association"), 0.0)
            rows.append(dict(gene=r["target"]["approvedSymbol"], overall=r["score"], genetic=gen))
        page += 1
        n_gen = sum(1 for r in rows if r["genetic"] > 0)
        if n_gen >= TOP_N_GENES or len(rows) >= total:
            break
    raw = dict(trait=tr["name"], query=tr["query"], search_hits=hits, chosen=chosen, n_targets_total=total, rows=rows)
    with open(os.path.join(CACHE_DIR, tr["name"].replace(" ", "_").replace("/", "-") + ".json"), "w") as fh:
        json.dump(raw, fh)
    df = pd.DataFrame(rows).drop_duplicates("gene")
    df = df.sort_values(["genetic", "overall"], ascending=False)
    filt = df[df["genetic"] > 0]
    src = "genetic_association_paged"
    if len(filt) < 10:
        raise RuntimeError(f"only {len(filt)} genetic_association genes found for {tr['name']} in {len(rows)} rows - insufficient GWAS-evidence set, trait excluded rather than mixing evidence types")
    genes = filt["gene"].head(TOP_N_GENES).tolist()
    print(f"{tr['name']}: {chosen['id']} ({chosen['name']}) total targets {total} -> set {len(genes)} genes ({src})")
    return genes, chosen, src, filt.head(TOP_N_GENES)

TRAIT_SETS, trait_meta_rows, trait_long = {}, [], []
if os.path.exists(SET_TSV):
    _prev = pd.read_csv(SET_TSV)
    if set(_prev["trait"].unique()) >= {t["name"] for t in TRAITS}:
        for name, grp in _prev.groupby("trait"):
            TRAIT_SETS[name] = grp["gene"].tolist()
        trait_meta_rows = _prev.drop_duplicates("trait")[["trait", "kind", "disease_id", "disease_name", "set_source"]].to_dict("records")
        print("trait gene sets loaded from repo cache:", {k: len(v) for k, v in TRAIT_SETS.items()})

if not TRAIT_SETS:
    for tr in TRAITS:
        try:
            genes, chosen, src, top = fetch_trait_set(tr)
        except Exception as e:
            print(f"TRAIT FAILED: {tr['name']}: {str(e)[:200]}")
            if tr["kind"] == "control":
                raise RuntimeError(f"negative control {tr['name']} could not be resolved - run is invalid without it")
            continue
        TRAIT_SETS[tr["name"]] = genes
        trait_meta_rows.append(dict(trait=tr["name"], kind=tr["kind"], disease_id=chosen["id"],
                                    disease_name=chosen["name"], set_source=src))
        for _, r in top.iterrows():
            trait_long.append(dict(trait=tr["name"], kind=tr["kind"], disease_id=chosen["id"],
                                   disease_name=chosen["name"], set_source=src,
                                   gene=r["gene"], genetic_score=r["genetic"], overall_score=r["overall"]))
    pd.DataFrame(trait_long).to_csv(SET_TSV, index=False)
    with open(os.path.join(SET_DIR, "trait_gene_sets.gmt"), "w") as fh:
        for name, genes in TRAIT_SETS.items():
            fh.write(name + chr(9) + "OpenTargets genetic_association top" + str(TOP_N_GENES) + chr(9) + chr(9).join(genes) + chr(10))

TRAIT_META = pd.DataFrame(trait_meta_rows)
TRAIT_META.to_csv(os.path.join(SET_DIR, "trait_meta.csv"), index=False)
print(TRAIT_META)
"""))

cells.append(code("""# ---------- Monte Carlo calibration engine (cluster-level nulls; matched + unmatched ablation) ----------
def bh_qvals(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out

def null_spot_scores(gss, trait_idx, bin_id, matched, rng):
    G = gss.shape[1]
    M = len(trait_idx)
    in_set = np.zeros(G, dtype=bool)
    in_set[trait_idx] = True
    samples = np.empty((M, P_PERM), dtype=np.int64)
    if matched:
        for j, g in enumerate(trait_idx):
            pool = np.where((bin_id == bin_id[g]) & (~in_set))[0]
            if len(pool) == 0:
                pool = np.where(~in_set)[0]
            samples[j] = rng.choice(pool, size=P_PERM, replace=True)
    else:
        pool_all = np.where(~in_set)[0]
        samples[:] = rng.choice(pool_all, size=(M, P_PERM), replace=True)
    rows = samples.T.reshape(-1)                       # permutation-major
    cols = np.repeat(np.arange(P_PERM), M)
    B = sp.csc_matrix((np.full(len(rows), 1.0 / M, dtype=np.float32), (rows, cols)),
                      shape=(G, P_PERM))
    return np.asarray(B.T @ gss.T).T.astype(np.float32)   # (n_spots, P)

def calibrate_dataset(ds):
    d = DATA[ds]
    gss, dom, K = d["gss_rna"], d["domains"], DATASETS[ds]["K"]
    records = {}
    spot_z = pd.DataFrame({"obs": d["obs_names"], "domain": dom})
    trait_names = [t["name"] for t in TRAITS if t["name"] in TRAIT_SETS]
    for ti, tname in enumerate(trait_names):
        idx = [d["gene_index"][g] for g in TRAIT_SETS[tname] if g in d["gene_index"]]
        if len(idx) < 10:
            print(f"{ds}/{tname}: only {len(idx)} set genes in universe - skipped")
            continue
        obs = gss[:, idx].mean(axis=1)
        for matched in (True, False):
            tag = "matched" if matched else "unmatched"
            rng = np.random.default_rng(np.random.SeedSequence([SEED, ti, int(matched)]))
            nsp = null_spot_scores(gss, np.asarray(idx), d["bin_id"], matched, rng)
            if matched:
                sz = (obs - nsp.mean(axis=1)) / (nsp.std(axis=1) + 1e-8)
                spot_z[tname] = sz
            for k in range(K):
                mask = dom == k
                S = float(obs[mask].mean())
                null_k = nsp[mask].mean(axis=0)
                Z = (S - float(null_k.mean())) / (float(null_k.std(ddof=1)) + 1e-8)
                p_emp = (1.0 + float((null_k >= S).sum())) / (P_PERM + 1.0)
                rec = records.setdefault((tname, k), dict(
                    dataset=ds, trait=tname,
                    kind=next(t["kind"] for t in TRAITS if t["name"] == tname),
                    cluster=k, n_genes_set=len(TRAIT_SETS[tname]), n_genes_in_universe=len(idx),
                    S_actual=S, S_actual_median=float(np.median(obs[mask]))))
                rec[f"Z_{tag}"] = float(Z)
                rec[f"p_{tag}"] = float(p_emp)
                if matched:
                    rec["median_spot_Z"] = float(np.median(spot_z[tname].values[mask]))
        print(f"{ds}/{tname}: calibrated ({len(idx)} genes in universe)")
    df = pd.DataFrame(list(records.values()))
    df["FDR_matched"] = bh_qvals(df["p_matched"].values)
    df["FDR_unmatched"] = bh_qvals(df["p_unmatched"].values)
    df = df.sort_values(["trait", "cluster"]).reset_index(drop=True)
    df.to_csv(os.path.join(OUT, f"calibrated_trait_scores_{ds}.csv"), index=False)
    spot_z.to_csv(os.path.join(OUT, f"spot_Z_{ds}.csv.gz"), index=False, compression="gzip")
    return df

CALIB = {ds: calibrate_dataset(ds) for ds in DATASETS}
for ds, df in CALIB.items():
    print(f"===== {ds} negative controls (matched null) =====")
    print(df[df["kind"] == "control"][["trait", "cluster", "Z_unmatched", "FDR_unmatched", "Z_matched", "FDR_matched"]].to_string(index=False))
"""))

cells.append(code("""# ---------- Calibration QC + plots (the Height fix, shown explicitly) ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
PLOTS = os.path.join(OUT, "plots")
os.makedirs(PLOTS, exist_ok=True)

qc_rows = []
for ds, df in CALIB.items():
    for cname in CONTROLS:
        sub = df[df["trait"] == cname]
        if sub.empty:
            continue
        qc_rows.append(dict(dataset=ds, control=cname,
                            min_FDR_unmatched=float(sub["FDR_unmatched"].min()),
                            min_FDR_matched=float(sub["FDR_matched"].min()),
                            pass_matched=bool(sub["FDR_matched"].min() >= 0.05)))
QC = pd.DataFrame(qc_rows)
QC.to_csv(os.path.join(OUT, "calibration_qc.csv"), index=False)
print(QC)
print("CALIBRATION QC:", "PASS" if QC["pass_matched"].all() else "FAIL - a negative control is significant under the matched null")

for ds, df in CALIB.items():
    h = df[df["trait"] == "Height"].sort_values("cluster")
    if not h.empty:
        x = np.arange(len(h))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x - 0.2, h["Z_unmatched"], width=0.4, label="unmatched null", color="#999999")
        ax.bar(x + 0.2, h["Z_matched"], width=0.4, label="matched null (calibrated)", color="#2166ac")
        ax.set_xticks(x)
        ax.set_xticklabels([f"C{int(c)}" for c in h["cluster"]])
        ax.set_ylabel("cluster Z")
        ax.set_title(f"{ds} - Height (negative control): unmatched vs calibrated")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS, f"Height_calibration_fix_{ds}.png"), dpi=150)
        plt.close(fig)
    piv = df.pivot(index="trait", columns="cluster", values="Z_matched")
    sig = df.pivot(index="trait", columns="cluster", values="FDR_matched") < 0.05
    fig, ax = plt.subplots(figsize=(1.1 * len(piv.columns) + 3, 0.55 * len(piv.index) + 2))
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-6, vmax=6)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"C{int(c)}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            if sig.values[i, j]:
                ax.text(j, i, "*", ha="center", va="center", fontsize=13)
    ax.set_title(f"{ds} - calibrated trait Z by cluster (* = FDR < 0.05)")
    fig.colorbar(im, ax=ax, shrink=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, f"calibrated_Z_heatmap_{ds}.png"), dpi=150)
    plt.close(fig)
print("plots saved to", PLOTS)
"""))

cells.append(code("""# ---------- A1<->D1 cluster matching in canonical feature space (never latent space) ----------
from scipy.stats import rankdata
from scipy.optimize import linear_sum_assignment

def std_cols(M):
    mu = M.mean(axis=0)
    sd = M.std(axis=0)
    sd[sd == 0] = 1.0
    return (M - mu) / sd

shared_genes = sorted(set(DATA["A1"]["uni_genes"]) & set(DATA["D1"]["uni_genes"]))
shared_prots = sorted(set(DATA["A1"]["adt_hgnc"]) & set(DATA["D1"]["adt_hgnc"]))
print(f"shared features: {len(shared_genes)} genes, {len(shared_prots)} proteins")
PROF = {}
for ds in DATASETS:
    d = DATA[ds]
    gi = [d["gene_index"][g] for g in shared_genes]
    pi = [list(d["adt_hgnc"]).index(p) for p in shared_prots]
    PROF[ds] = std_cols(np.concatenate([d["prof_rna"][:, gi], d["prof_adt"][:, pi]], axis=1))

KA, KD = DATASETS["A1"]["K"], DATASETS["D1"]["K"]
rho = np.zeros((KA, KD))
for a in range(KA):
    ra = rankdata(PROF["A1"][a])
    for b in range(KD):
        rb = rankdata(PROF["D1"][b])
        rho[a, b] = np.corrcoef(ra, rb)[0, 1]
rho_df = pd.DataFrame(rho, index=[f"A1_C{a}" for a in range(KA)], columns=[f"D1_C{b}" for b in range(KD)])
rho_df.to_csv(os.path.join(OUT, "matching_rho_matrix.csv"))
print(rho_df.round(2))

rows_a, cols_d = linear_sum_assignment(-rho)
pair_rows = []
for a, b in zip(rows_a, cols_d):
    rbh = (np.argmax(rho[a]) == b) and (np.argmax(rho[:, b]) == a)
    pair_rows.append(dict(A1_cluster=int(a), D1_cluster=int(b), rho=float(rho[a, b]),
                          reciprocal_best_hit=bool(rbh), kept=bool(rho[a, b] >= RHO_MIN)))
PAIRS = pd.DataFrame(pair_rows).sort_values("A1_cluster")
PAIRS.to_csv(os.path.join(OUT, "matched_pairs.csv"), index=False)
print(PAIRS)
print(f"D1 clusters unmatched by Hungarian: {sorted(set(range(KD)) - set(PAIRS['D1_cluster']))}")

# cross-donor trait concordance on matched pairs
zA = CALIB["A1"].pivot(index="trait", columns="cluster", values="Z_matched")
zD = CALIB["D1"].pivot(index="trait", columns="cluster", values="Z_matched")
common_traits = sorted(set(zA.index) & set(zD.index))
conc_rows, all_a, all_d = [], [], []
kept_pairs = PAIRS[PAIRS["kept"]]
for _, pr in kept_pairs.iterrows():
    va = zA.loc[common_traits, int(pr["A1_cluster"])].values.astype(float)
    vd = zD.loc[common_traits, int(pr["D1_cluster"])].values.astype(float)
    r = np.corrcoef(rankdata(va), rankdata(vd))[0, 1]
    conc_rows.append(dict(A1_cluster=int(pr["A1_cluster"]), D1_cluster=int(pr["D1_cluster"]),
                          rho_profile=float(pr["rho"]), trait_Z_spearman=float(r)))
    all_a.extend(va)
    all_d.extend(vd)
CONC = pd.DataFrame(conc_rows)
global_rho = float(np.corrcoef(rankdata(all_a), rankdata(all_d))[0, 1]) if all_a else float("nan")
CONC.to_csv(os.path.join(OUT, "trait_concordance.csv"), index=False)
print(CONC)
print(f"global trait-Z concordance across matched pairs: Spearman rho = {global_rho:.3f}")

# replication: same trait significant in matched niches in BOTH donors
fA = CALIB["A1"].pivot(index="trait", columns="cluster", values="FDR_matched")
fD = CALIB["D1"].pivot(index="trait", columns="cluster", values="FDR_matched")
rep_rows = []
for _, pr in kept_pairs.iterrows():
    a, b = int(pr["A1_cluster"]), int(pr["D1_cluster"])
    for tname in common_traits:
        if fA.loc[tname, a] < 0.05 and fD.loc[tname, b] < 0.05:
            rep_rows.append(dict(trait=tname, A1_cluster=a, D1_cluster=b,
                                 Z_A1=float(zA.loc[tname, a]), Z_D1=float(zD.loc[tname, b]),
                                 FDR_A1=float(fA.loc[tname, a]), FDR_D1=float(fD.loc[tname, b]),
                                 rho_profile=float(pr["rho"])))
REP = pd.DataFrame(rep_rows, columns=["trait", "A1_cluster", "D1_cluster", "Z_A1", "Z_D1", "FDR_A1", "FDR_D1", "rho_profile"])
REP.to_csv(os.path.join(OUT, "replicated_hits.csv"), index=False)
print(f"replicated trait-cluster associations: {len(REP)}")
print(REP)

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(rho, cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(KD)); ax.set_xticklabels([f"D1_C{b}" for b in range(KD)])
ax.set_yticks(range(KA)); ax.set_yticklabels([f"A1_C{a}" for a in range(KA)])
for _, pr in kept_pairs.iterrows():
    ax.add_patch(plt.Rectangle((int(pr["D1_cluster"]) - 0.5, int(pr["A1_cluster"]) - 0.5), 1, 1,
                               fill=False, edgecolor="red", linewidth=2))
ax.set_title("A1<->D1 profile Spearman rho (red = matched pair)")
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "matching_rho_heatmap.png"), dpi=150)
plt.close(fig)
"""))

cells.append(code("""# ---------- Druggable surfaceome: DGIdb v5 + OpenTargets (live, cached) ----------
DGIDB_URL = "https://dgidb.org/api/graphql"
STAGE_ORDER = {"APPROVAL": 5, "PHASE_3": 4, "PHASE_2_3": 3.5, "PHASE_2": 3, "PHASE_1_2": 2.5, "PHASE_1": 2, "PRECLINICAL": 1}

def gql2(url, query, variables, tries=3):
    last = None
    for i in range(tries):
        try:
            r = requests.post(url, json={"query": query, "variables": variables}, timeout=60)
            r.raise_for_status()
            out = r.json()
            if "errors" in out:
                raise RuntimeError(str(out["errors"])[:300])
            return out["data"]
        except Exception as e:
            last = e
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"API call failed: {last}")

# candidate (dataset, cluster, trait) entries
entries = []
if len(REP) > 0:
    for _, r in REP.iterrows():
        entries.append(("A1", int(r["A1_cluster"]), r["trait"], True))
        entries.append(("D1", int(r["D1_cluster"]), r["trait"], True))
    cand_source = "replicated in both donors"
else:
    sig = pd.concat([CALIB[ds].assign(dataset=ds) for ds in DATASETS])
    sig = sig[sig["FDR_matched"] < 0.05]
    if len(sig) > 0:
        for _, r in sig.iterrows():
            entries.append((r["dataset"], int(r["cluster"]), r["trait"], False))
        cand_source = "single-donor FDR<0.05 (no replicated hits)"
    else:
        top = pd.concat([CALIB[ds].assign(dataset=ds) for ds in DATASETS]).sort_values("Z_matched", ascending=False).head(3)
        for _, r in top.iterrows():
            entries.append((r["dataset"], int(r["cluster"]), r["trait"], False))
        cand_source = "exploratory top-Z (no FDR<0.05 hits)"
print("drug-stage candidates from:", cand_source, "| entries:", len(entries))

cand = {}   # gene -> list of occurrence dicts
for ds, cl, tname, repl in entries:
    d = DATA[ds]
    top_adt = np.argsort(-d["prof_adt"][cl])[:5]
    for j in top_adt:
        cand.setdefault(d["adt_hgnc"][j], []).append(
            dict(dataset=ds, cluster=cl, trait=tname, replicated=repl, modality="ADT",
                 gene=d["adt_hgnc"][j], GSS=float(d["prof_adt"][cl, j])))
    top_rna = np.argsort(-d["prof_rna"][cl])[:10]
    for j in top_rna:
        cand.setdefault(d["uni_genes"][j], []).append(
            dict(dataset=ds, cluster=cl, trait=tname, replicated=repl, modality="RNA",
                 gene=d["uni_genes"][j], GSS=float(d["prof_rna"][cl, j])))
genes_all = sorted(cand)
print(f"unique candidate genes for API lookup: {len(genes_all)}")

# DGIdb batch (schema verified live 2026-11-05)
Q_DGI = ("query($names: [String!]!){ genes(names: $names){ nodes { name "
         "interactions { drug { name approved } interactionScore } } } }")
dgi = {}
try:
    data = gql2(DGIDB_URL, Q_DGI, {"names": genes_all})
    for node in data["genes"]["nodes"]:
        if node:
            dgi[node["name"]] = node["interactions"]
except Exception as e:
    print("DGIdb batch failed, falling back to per-gene:", str(e)[:150])
    for g in genes_all:
        try:
            data = gql2(DGIDB_URL, Q_DGI, {"names": [g]})
            for node in data["genes"]["nodes"]:
                if node:
                    dgi[node["name"]] = node["interactions"]
        except Exception as e2:
            print("  DGIdb failed for", g, str(e2)[:100])

# OpenTargets per gene: tractability + clinical candidates
Q_TSEARCH = "query($q: String!){ search(queryString: $q, entityNames: [\\"target\\"], page: {index: 0, size: 3}) { hits { id name } } }"
Q_TRACT = ("query($id: String!){ target(ensemblId: $id){ approvedSymbol "
           "tractability { label modality value } "
           "drugAndClinicalCandidates { count rows { drug { name maximumClinicalStage } } } } }")
ot_info = {}
for g in genes_all:
    try:
        hits = gql(Q_TSEARCH, {"q": g})["search"]["hits"]
        exact = [h for h in hits if h["name"] == g]
        if not hits:
            ot_info[g] = {}
            continue
        tid = (exact[0] if exact else hits[0])["id"]
        t = gql(Q_TRACT, {"id": tid})["target"]
        ab = [x["label"] for x in t["tractability"] if x["modality"] == "AB" and x["value"]]
        sm = [x["label"] for x in t["tractability"] if x["modality"] == "SM" and x["value"]]
        candrows = t["drugAndClinicalCandidates"]["rows"]
        best = max(candrows, key=lambda r: STAGE_ORDER.get(r["drug"]["maximumClinicalStage"], 0), default=None)
        approved = sorted({r["drug"]["name"] for r in candrows if r["drug"]["maximumClinicalStage"] == "APPROVAL"})
        ot_info[g] = dict(ensembl=tid, tractability_AB="; ".join(ab), tractability_SM="; ".join(sm),
                          max_stage=(best["drug"]["maximumClinicalStage"] if best else ""),
                          ot_approved_drugs="; ".join(approved[:8]),
                          n_candidates=t["drugAndClinicalCandidates"]["count"])
    except Exception as e:
        print("  OpenTargets failed for", g, str(e)[:120])
        ot_info[g] = {}

os.makedirs(os.path.join(OUT, "cache"), exist_ok=True)
with open(os.path.join(OUT, "cache", "drug_apis.json"), "w") as fh:
    json.dump(dict(dgidb=dgi, opentargets=ot_info), fh)

out_rows = []
for g, occs in cand.items():
    inter = dgi.get(g, [])
    appr = sorted({i["drug"]["name"] for i in inter if i["drug"].get("approved")},
                  key=lambda n: -max(i["interactionScore"] for i in inter if i["drug"]["name"] == n))
    info = ot_info.get(g, {})
    for o in occs:
        out_rows.append(dict(**o, n_dgidb_interactions=len(inter),
                             dgidb_approved_drugs="; ".join(appr[:8]),
                             ot_max_stage=info.get("max_stage", ""),
                             ot_approved_drugs=info.get("ot_approved_drugs", ""),
                             tractability_AB=info.get("tractability_AB", ""),
                             tractability_SM=info.get("tractability_SM", "")))
DRUGS = pd.DataFrame(out_rows)
if len(DRUGS):
    DRUGS["_mod"] = (DRUGS["modality"] != "ADT").astype(int)
    DRUGS = DRUGS.sort_values(["replicated", "_mod", "GSS"], ascending=[False, True, False]).drop(columns="_mod")
DRUGS.to_csv(os.path.join(OUT, "drug_targets.csv"), index=False)
print(DRUGS.head(30).to_string(index=False) if len(DRUGS) else "no drug candidates")
"""))

cells.append(code("""# ---------- Summary + push ----------
height_lines = {}
for ds, df in CALIB.items():
    h = df[df["trait"] == "Height"]
    if not h.empty:
        height_lines[ds] = dict(min_FDR_unmatched=float(h["FDR_unmatched"].min()),
                                min_FDR_matched=float(h["FDR_matched"].min()),
                                max_Z_matched=float(h["Z_matched"].abs().max()))
run_log = dict(
    stage="stage8_combo1_lymph_node_calibrated",
    params=dict(P=P_PERM, seed=SEED, knn_k=KNN_K, bins=f"{N_BINS}x{N_BINS}", min_pct=MIN_PCT,
                top_n_genes=TOP_N_GENES, rho_min=RHO_MIN),
    datasets={ds: dict(n_spots=int(DATA[ds]["n_spots"]), K=DATASETS[ds]["K"],
                       universe=int(len(DATA[ds]["uni_genes"])),
                       adt_proteins=int(len(DATA[ds]["adt_hgnc"])),
                       posthoc_ARI=DATA[ds]["ari"]) for ds in DATASETS},
    trait_set_sizes={k: len(v) for k, v in TRAIT_SETS.items()},
    calibration_qc=QC.to_dict("records") if len(QC) else [],
    height_headline=height_lines,
    stage7_reference=("Stage 7 gsMap S-LDSC Height Cauchy p = 7.33e-15 on A1 domains "
                      "(different statistic - reference only, not the ablation baseline)"),
    matched_pairs=int(PAIRS["kept"].sum()) if len(PAIRS) else 0,
    global_trait_concordance=global_rho,
    n_replicated_hits=int(len(REP)),
    drug_candidate_source=cand_source,
    n_drug_rows=int(len(DRUGS)),
)
with open(os.path.join(OUT, "run_log.json"), "w") as fh:
    json.dump(run_log, fh, indent=2, default=str)
print(json.dumps(run_log, indent=2, default=str)[:3000])
print()
print("HEADLINE - Height negative control on A1:")
hl = height_lines.get("A1", {})
print(f"  unmatched null: min FDR = {hl.get('min_FDR_unmatched')}")
print(f"  matched null (calibrated): min FDR = {hl.get('min_FDR_matched')}")
print("  Stage 7 reference (S-LDSC, different statistic): Cauchy p = 7.33e-15")
print("CALIBRATION QC:", "PASS" if (len(QC) and QC["pass_matched"].all()) else "CHECK calibration_qc.csv")

git_push("stage8 combo1: calibrated A1+D1 trait association, replication, ADT drug targets")
"""))

cells.append(md("""## How to read the outputs (all in `runs/s8-disease/`)

- `calibrated_trait_scores_{A1,D1}.csv` — the main result: per trait x cluster, the observed score, **Z_matched / FDR_matched** (the calibrated numbers to report), the unmatched ablation columns, and the secondary median spot-Z. Wording for any write-up: "spatial enrichment of disease-relevant programmes" / "cellular vulnerability mapping".
- `calibration_qc.csv` + `plots/Height_calibration_fix_*.png` — the QC gate: Height and Heel BMD must be null under the matched null in both donors. If QC fails, the calibrated numbers are not publishable yet — that is the point of the gate.
- `matching_rho_matrix.csv`, `matched_pairs.csv`, `trait_concordance.csv`, `replicated_hits.csv` — cross-donor replication in canonical feature space. A trait-cluster hit counts as replicated only if it is FDR < 0.05 in matched niches in **both** A1 and D1.
- `drug_targets.csv` — ADT surface proteins first (they are the actual antibody-accessible targets), then RNA hubs, with DGIdb approved/clinical drugs and OpenTargets tractability (AB = antibody/surface, SM = small molecule) and max clinical stage. Raw API responses are cached under `cache/` and trait sets under `trait_gene_sets/`, so this exact run is reproducible offline.
- Known limitations: trait sets are OpenTargets genetic-association top-200 (not MAGMA-on-GWAS sets — none exist from Stage 7); small-GWAS traits (DLBCL, FL, T-ALL) may be underpowered, so a null there is not proof of absence; the ADT panel is small, so its permutation matching is coarse and A1<->D1 replication is its real evidence.
- Next: Stage 8b repeats this for Mouse Brain E11-E18 with the gene-level ortholog route (no peak liftOver), producing the canonical Disease x Stage matrix."""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "cells": cells,
}

out_path = os.path.join(os.path.dirname(__file__), "..", "notebooks", "stage8_combo1_lymph_node_calibrated.ipynb")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {out_path} ({len(cells)} cells)")

import nbformat
nbformat.read(out_path, as_version=4)
print("nbformat: valid")

# Content verification (AGENTS.md lesson: verify the OUTPUT notebook)
text = open(out_path).read()
for token in ["runs/s8-disease", "adata_RNA.h5ad", "adata_ADT.h5ad",
              "10x_human_lymph_node_A1_seed_42_embeddings.npy",
              "10x_human_lymph_node_D1_seed_42_embeddings.npy",
              "clr_stage5", "ADT_ALIAS", "adt_mapping_",
              "null_spot_scores", "SeedSequence", "FDR_matched", "FDR_unmatched",
              "Heel BMD", "genetic_association", "trait_gene_sets.tsv",
              "api.platform.opentargets.org", "dgidb.org/api/graphql",
              "linear_sum_assignment", "RHO_MIN", "reciprocal_best_hit",
              "Height_calibration_fix", "calibration_qc.csv", "replicated_hits.csv",
              "drug_targets.csv", "run_log.json", "git_push",
              "spatial enrichment of disease-relevant programmes"]:
    assert token in text, f"MISSING TOKEN: {token}"
assert 'os.path.join(REPO, "runs", "s7-disease")' not in text, "Stage 7 output path leaked as Stage 8 OUT"
assert "Mouse_Brain" not in text, "mouse scope leaked into Combo 1 notebook"
assert "liftOver" not in text.replace("no peak liftOver", ""), "unexpected liftOver mention"
assert "run_spatial_ldsc" not in text, "gsMap LDSC route leaked into Stage 8"
print("content check: all key tokens present, Combo-1 scope clean, no stale Stage 7 output path")
