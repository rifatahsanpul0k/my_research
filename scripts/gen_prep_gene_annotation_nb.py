"""Generate notebooks/prep_gene_annotation_colab.ipynb

Step 2 of the post-smoke plan (built 2026-11-10): give the hello.md
pipeline driver the REAL gene annotation it reads from adata_rna.var
(chrom / start / end / biotype) — the columns the 10x h5ads do not carry
(the smoke test faked them with random values).

The annotation table resources/gene_annotation_GRCh38_gencode_v44.csv was
built from GENCODE v44 (GRCh38) and joined to the panel through the
Ensembl gene_ids stored in var["gene_ids"]:
  18,068 / 18,085 genes matched by Ensembl ID, 11 more by unique-symbol
  fallback (IDs retired since the Cell Ranger reference was built, e.g.
  SOD2), 6 unmatched (clone-named lncRNAs / novel genes: GCSAML-AS1,
  AC004593.3, TIAF1, ELOA3, AC119396.1, AC007244.1) -> 99.97% coverage,
  17,987 protein_coding, 211 genes overlapping the chr6:25-34 Mb MHC
  window. A1 and D1 share the identical 18,085-gene panel (verified from
  the local h5ads), so one table serves both sections; the notebook still
  asserts the panel match per section at runtime.

The notebook (Google Colab, Drive-registry pattern like the smoke
notebook): for A1 and D1 it fetches adata_RNA.h5ad via gdown, downloads
the annotation CSV from this repo, joins on gene_ids, writes
<SECTION>_adata_RNA_annotated.h5ad under /content/annotated/, and prints
coverage asserts. An optional last cell copies the annotated files to the
user's own Drive (MyDrive/hello_pipeline_annotated/) so they survive the
Colab runtime.

Run: python3 scripts/gen_prep_gene_annotation_nb.py
"""
import json, os

MD = """# Prep — Real Gene Annotation for A1 + D1 (Google Colab)

**Why this exists:** the hello-pipeline driver reads `chrom`, `start`, `end`, `biotype` from the RNA h5ad's `var` — and the 10x files don't carry those columns. The smoke test faked them (random chromosomes, everything `protein_coding`, which also made the −MHC track meaningless). This notebook builds the **real** thing.

**What it does, per section (A1, D1):**
1. Fetches `adata_RNA.h5ad` from the Google Drive dataset folder (same registry pattern as the smoke notebook).
2. Downloads the annotation table from this repo: `resources/gene_annotation_GRCh38_gencode_v44.csv` — GENCODE v44 (GRCh38) joined to the panel through the Ensembl `gene_ids` already stored in `var`.
3. Adds `chrom / start / end / biotype` to `var` and writes `<section>_adata_RNA_annotated.h5ad` to `/content/annotated/`.

**Join quality (verified when the table was built, re-asserted in this notebook):** 18,085 panel genes → 18,068 matched by Ensembl ID + 11 by unique-symbol fallback (IDs retired since the Cell Ranger reference, e.g. SOD2) + **6 unmatched** (clone-named lncRNAs / novel genes — they get empty annotation and are treated as non-MHC, non-protein-coding downstream). Coverage **99.97%**; 17,987 protein-coding; 211 genes overlap the chr6:25–34 Mb MHC window. A1 and D1 share the identical gene panel, so one table serves both — the notebook asserts this per section anyway.

**After running:** the annotated files live in `/content/annotated/` (gone when the runtime ends). The real-run notebook can either re-run this prep inline or read these files — use the **optional last cell** to copy them into your own Drive (`MyDrive/hello_pipeline_annotated/`) if you want them kept.
"""

INSTALL = """import os, sys, subprocess
for pkg in ["scanpy", "gdown"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
import numpy as np
import pandas as pd
import scanpy as sc
print("imports ok; pandas", pd.__version__)
"""

CONFIG = """# ---------- CONFIG ----------
DATA_HOME = "/content/datasets"
OUTDIR = "/content/annotated"
os.makedirs(DATA_HOME, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)
ANNO_URL = ("https://raw.githubusercontent.com/rifatahsanpul0k/my_research/main/"
            "resources/gene_annotation_GRCh38_gencode_v44.csv")
ANNO_CSV = "/content/gene_annotation_GRCh38_gencode_v44.csv"

DATASET_REGISTRY = {
    "10x_human_lymph_node_A1": {
        "url": "https://drive.google.com/drive/folders/10z1N4MwW8Y49o8GlkYGBKVx1N7fiMuyC",
        "other_file": "adata_ADT.h5ad",
        "anno_file": "annotation.csv",
    },
    "10x_human_lymph_node_D1": {
        "url": "https://drive.google.com/drive/folders/1-g_Ca2XMaMXF-MisuVY-wobWDX86O6zz",
        "other_file": "adata_ADT.h5ad",
        "anno_file": "annotation.csv",
    },
}
SECTIONS = list(DATASET_REGISTRY)

import glob as _glob
def _find_dataset_dir(name):
    info = DATASET_REGISTRY[name]
    need = ["adata_RNA.h5ad", info["other_file"], info["anno_file"]]
    for cand in _glob.glob(f"/content/**/{name}", recursive=True):
        if os.path.isdir(cand) and all(os.path.exists(os.path.join(cand, f)) for f in need):
            return cand
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
    print(f"{name}: downloading Drive folder ...")
    r = subprocess.run(["gdown", "--folder", info["url"], "-O", DATA_HOME],
                       capture_output=True, text=True)
    if r.returncode != 0:
        r = subprocess.run([sys.executable, "-m", "gdown", "--folder", info["url"], "-O", DATA_HOME],
                           capture_output=True, text=True)
    found = _find_dataset_dir(name)
    assert found, (f"{name}: download did not produce the required files - if gdown reported a "
                   "permission error, the Drive folder must be shared as 'anyone with the link'")
    print(f"{name}: ready at {found}")
    return found

import urllib.request
if not os.path.exists(ANNO_CSV):
    urllib.request.urlretrieve(ANNO_URL, ANNO_CSV)
anno = pd.read_csv(ANNO_CSV, dtype={"chrom": str})
print("annotation table:", anno.shape, "| columns:", list(anno.columns))
assert len(anno) == 18085, f"expected the 18,085-gene panel table, got {len(anno)} rows"
"""

PREP = """# ---------- PREP: add chrom/start/end/biotype to var, per section ----------
def prep_section(section):
    sec_dir = ensure_dataset(section)
    adata = sc.read_h5ad(os.path.join(sec_dir, "adata_RNA.h5ad"))
    var = adata.var
    assert "gene_ids" in var.columns, "var has no gene_ids column - cannot join annotation"
    panel_ids = var["gene_ids"].astype(str).tolist()
    assert set(panel_ids) == set(anno["gene_id"].astype(str)), \\
        f"{section}: panel gene_ids differ from the annotation table - do not force the join"
    m = anno.set_index("gene_id").loc[panel_ids]
    assert (m["gene_symbol"].astype(str).values == var.index.astype(str).values).mean() > 0.999, \\
        f"{section}: gene_symbol order does not match var_names - join is misaligned"
    var["chrom"] = m["chrom"].fillna("").values
    var["start"] = pd.to_numeric(m["start"], errors="coerce").values
    var["end"] = pd.to_numeric(m["end"], errors="coerce").values
    var["biotype"] = m["biotype"].fillna("").values
    n_match = int((m["match_source"] != "unmatched").sum())
    n_pc = int((var["biotype"] == "protein_coding").sum())
    mhc = var[(var["chrom"] == "chr6") & (var["start"] <= 34000000) & (var["end"] >= 25000000)]
    print(f"{section}: genes {len(var)} | annotated {n_match} ({n_match/len(var)*100:.2f}%) "
          f"| protein_coding {n_pc} | chr6:25-34Mb overlap {len(mhc)}")
    assert n_match / len(var) > 0.999, f"{section}: annotation coverage below 99.9%"
    out = os.path.join(OUTDIR, f"{section}_adata_RNA_annotated.h5ad")
    adata.write_h5ad(out)
    print("wrote", out, os.path.getsize(out), "bytes")
    return out

outputs = {s: prep_section(s) for s in SECTIONS}
print("done:", outputs)
"""

DRIVE_SAVE = """# ---------- OPTIONAL: keep the annotated files in your own Drive ----------
# Mounts Google Drive and copies /content/annotated/*.h5ad to
# MyDrive/hello_pipeline_annotated/. Skip this cell if you will re-run the
# prep inside the real-run notebook instead.
from google.colab import drive as _drive
_drive.mount("/content/drive")
import glob as _g, shutil as _sh
_dst = "/content/drive/MyDrive/hello_pipeline_annotated"
os.makedirs(_dst, exist_ok=True)
for _p in _g.glob("/content/annotated/*.h5ad"):
    _sh.copy(_p, os.path.join(_dst, os.path.basename(_p)))
    print("saved to Drive:", os.path.join(_dst, os.path.basename(_p)))
"""


def _cell(kind, text):
    c = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c


def main():
    cells = [_cell("markdown", MD), _cell("code", INSTALL), _cell("code", CONFIG),
             _cell("code", PREP), _cell("code", DRIVE_SAVE)]
    for i, c in enumerate(cells):
        if c["cell_type"] == "code":
            compile("".join(c["source"]), f"cell{i}", "exec")
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3"}},
          "nbformat": 4, "nbformat_minor": 5}
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, "..", "notebooks", "prep_gene_annotation_colab.ipynb")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path, os.path.getsize(out_path), "bytes,", len(cells), "cells")

    import nbformat
    nbformat.read(out_path, as_version=4)
    text = open(out_path).read()
    for token in ["gene_annotation_GRCh38_gencode_v44", "chrom", "biotype",
                  "10z1N4MwW8Y49o8GlkYGBKVx1N7fiMuyC", "1-g_Ca2XMaMXF-MisuVY-wobWDX86O6zz",
                  "99.97", "prep_section", "match_source"]:
        assert token in text, f"MISSING TOKEN: {token}"
    assert "kaggle" not in text.lower(), "unexpected Kaggle reference"
    print("content check: annotation/prep tokens present, zero Kaggle references")


if __name__ == "__main__":
    main()
