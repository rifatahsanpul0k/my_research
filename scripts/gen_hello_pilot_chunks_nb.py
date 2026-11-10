#!/usr/bin/env python3
"""Build the 3 CHUNKED pilot notebooks from the verified monolithic pilot notebook.

Source of truth: hello_pipeline_pilot_real.ipynb (commit d32a524) — its cells are
sliced/reordered, never retyped. Chunking (user decision 2026-11-10): fast parts
first, each chunk pushes its outputs to runs/hello-pilot-real before the next runs.

  chunk 1: refs + FAST MAGMA batch (8 traits) + A1/D1 prep sanity -> reports
  chunk 2: refs + REST MAGMA batch (8 traits) + fetch chunk 1 -> combined report
  chunk 3: refs + prep + fetch all 16 genes.out -> externals -> A1 -> D1 -> 4C

Only transforms on carried cells:
  - SUMSTATS_PREP: loop restricted to CHUNK_TRAITS (prepended literal).
  - MAGMA_RUN: checkpoint push message made chunk-specific.
Everything else is verbatim.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC_NB = HERE.parent / "hello_pilot_nb" / "hello_pipeline_pilot_real.ipynb"

FAST = ["Celiac", "Adult_Standing_Height", "Cortical_Thickness", "ASD",
        "Cortical_Surface_Area", "FL", "DLBCL", "CLL"]
REST = ["SLE", "IBD", "RA", "Heel_BMD", "Hair_Colour", "MS",
        "Lymphocyte_Count", "T1D"]

nb_src = json.loads(SRC_NB.read_text())
CELLS = ["".join(c["source"]) for c in nb_src["cells"]]
assert len(CELLS) == 16, f"pilot notebook should have 16 cells, got {len(CELLS)}"

# --- BUILD_OF: trait -> genome build, parsed from the pilot's TRAIT_SPEC ---
BUILD_OF = {}
for line in CELLS[5].splitlines():
    s = line.strip()
    if s.startswith('"'):
        toks = re.findall(r'"([^"]*)"', s)
        if len(toks) >= 8 and toks[-1] in ("37", "38"):
            BUILD_OF[toks[0]] = toks[-1]
assert len(BUILD_OF) == 16, f"BUILD_OF parse failed: {BUILD_OF}"
assert set(FAST) | set(REST) == set(BUILD_OF), "FAST/REST do not cover TRAIT_SPEC"
assert not (set(FAST) & set(REST)), "FAST/REST overlap"


def code(text, cid):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


def md(text, cid):
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": text.splitlines(keepends=True)}


def filtered_sumstats(traits):
    s = CELLS[5]
    old = "for _k, _spec in TRAIT_SPEC.items():"
    assert old in s
    s = s.replace(old, "for _k, _spec in [(k, TRAIT_SPEC[k]) for k in CHUNK_TRAITS]:")
    return "CHUNK_TRAITS = " + repr(traits) + "  # this chunk only\n" + s


def magma_cell(message):
    s = CELLS[6]
    old = "pilot-real: MAGMA genes.out for 16 traits (A1+D1 run)"
    assert old in s
    return s.replace(old, message)


def override_cell(folder):
    return ("# ---------- Chunk output folder (overrides the pilot OUTDIR) ----------\n"
            f'OUTDIR = "/kaggle/working/{folder}"\n'
            "os.makedirs(OUTDIR, exist_ok=True)\n"
            'print("chunk outputs ->", OUTDIR)\n')


MD1 = """# Hello Pipeline — PILOT, CHUNK 1 of 3: fast MAGMA batch + prep sanity

Part of the chunked real-data pilot (A1 + D1, engines as-is from hello.md).
This chunk: unzips the refs, runs MAGMA for the **8 fast traits** (Celiac, Height,
Cortical Thickness, ASD, Cortical Surface Area, FL, DLBCL, CLL), then the A1+D1
section prep (GENCODE v44 annotation join + ADT/annotation alignment) as a sanity
check. Outputs are pushed to `runs/hello-pilot-real` in the repo.

**Setup:** attach BOTH datasets (`spatial-multiomics-6datasets-private` +
`hello-pipeline-refs`), `GITHUB_TOKEN` secret set, Internet ON, then
**Save Version -> Save & Run All**. Expected: ~45-75 min.

Then run chunk 2. Read the two report CSVs in the repo after this finishes."""

MD2 = """# Hello Pipeline — PILOT, CHUNK 2 of 3: remaining MAGMA batch

Runs MAGMA for the **8 remaining traits** (SLE, IBD, RA, Heel BMD, Hair Colour, MS,
Lymphocyte Count, T1D), fetches chunk 1's `genes.out` files from the repo, and
pushes a combined 16-trait report. Run AFTER chunk 1 has finished and pushed.

**Setup:** same as chunk 1 (both datasets attached, `GITHUB_TOKEN`, Internet ON,
**Save Version -> Save & Run All**). Expected: ~1-2 h (T1D + Lymphocyte Count
are the big files). Then run chunk 3."""

MD3 = """# Hello Pipeline — PILOT, CHUNK 3 of 3: externals + A1 + D1 + Stage 4C

Fetches all 16 `genes.out` from the repo (chunks 1+2), rebuilds the section prep,
builds the real MAGMA z-scores + covariates, then runs the hello.md pipeline
(engines AS-IS: Engine 2 = OLS proxy, RSR without the projection, Engine 3 with
its own null) on A1, then D1, then Stage 4C matching/replication.
**Read the negative controls (Height, Heel BMD, Hair Colour) FIRST in the results.**
If a negative control comes out Confirmed, the code itself recalibrates at
n_perm=10000 — several extra hours; do not stop it.

**Setup:** same as chunk 1. Run AFTER chunk 2's combined report shows 16 traits.
Expected: ~1-3 h without recalibration."""

REPORT1 = r"""# ---------- Chunk 1 report: fast MAGMA batch + prep sanity ----------
rows = []
for p in sorted(os.listdir(MAGMA_OUT)):
    if p.endswith(".genes.out"):
        df = pd.read_csv(os.path.join(MAGMA_OUT, p), sep="\s+", engine="python")
        rows.append({"trait": p.replace(".genes.out", ""), "genes": len(df),
                     "median_abs_Z": float(df["ZSTAT"].abs().median())})
magma_report = pd.DataFrame(rows)
magma_report.to_csv(os.path.join(OUTDIR, "chunk1_magma_report.csv"), index=False)
print(magma_report)
assert len(magma_report) == 8, f"fast batch should have 8 genes.out, got {len(magma_report)}"

prep_rows = []
for sec in SECTIONS:
    v = ADATAS[sec].var
    prep_rows.append({"section": sec, "spots": int(ADATAS[sec].n_obs),
                      "genes": int(ADATAS[sec].n_vars),
                      "unmatched_genes": int((v["chrom"] == ".").sum()),
                      "protein_coding": int((v["biotype"] == "protein_coding").sum()),
                      "mhc_genes": int(((v["chrom"] == "chr6") & (v["start"] >= 25000000)
                                        & (v["end"] <= 34000000)).sum())})
prep_report = pd.DataFrame(prep_rows)
prep_report.to_csv(os.path.join(OUTDIR, "chunk1_prep_report.csv"), index=False)
print(prep_report)
push_outputs("chunk 1: reports (fast magma + prep sanity)")
"""

FETCH2 = r"""# ---------- Fetch chunk 1 genes.out from the repo + combined report ----------
import urllib.request
os.makedirs(MAGMA_OUT, exist_ok=True)
for key in CHUNK1_TRAITS:
    dst = os.path.join(MAGMA_OUT, key + ".genes.out")
    if not os.path.exists(dst):
        urllib.request.urlretrieve(
            f"{RAW_BASE}/runs/hello-pilot-real/magma/{key}.genes.out", dst)
    print("have:", key)
rows = []
for p in sorted(os.listdir(MAGMA_OUT)):
    if p.endswith(".genes.out"):
        with open(os.path.join(MAGMA_OUT, p)) as f:
            n = sum(1 for _ in f) - 1
        rows.append({"trait": p.replace(".genes.out", ""), "genes": n})
combined = pd.DataFrame(rows)
combined.to_csv(os.path.join(OUTDIR, "magma_combined_report.csv"), index=False)
print(combined)
assert len(combined) == 16, f"expected 16 genes.out after fetch, got {len(combined)}"
push_outputs("chunk 2: magma combined report (16 traits)")
"""

FETCH3_TMPL = r"""# ---------- Fetch all 16 genes.out from the repo (chunks 1+2) ----------
import urllib.request
MAGMA_OUT = os.path.join(OUTDIR, "magma")
os.makedirs(MAGMA_OUT, exist_ok=True)
BUILD_OF = __BUILD_OF__
for key in BUILD_OF:
    dst = os.path.join(MAGMA_OUT, key + ".genes.out")
    if not os.path.exists(dst):
        urllib.request.urlretrieve(
            f"{RAW_BASE}/runs/hello-pilot-real/magma/{key}.genes.out", dst)
    print("have:", key)
PREPPED = {k: (None, None, b) for k, b in BUILD_OF.items()}
GENES_OUT = {k: os.path.join(MAGMA_OUT, k + ".genes.out") for k in BUILD_OF}
print("externals inputs ready:", len(GENES_OUT), "traits")
"""


def assemble(cells):
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3",
                                        "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.10"}},
            "nbformat": 4, "nbformat_minor": 5}


def check(nb, name, n_cells, tokens):
    assert len(nb["cells"]) == n_cells, f"{name}: {len(nb['cells'])} cells != {n_cells}"
    import nbformat
    nbformat.validate(nbformat.from_dict(nb))
    full = ""
    for c in nb["cells"]:
        s = "".join(c["source"])
        full += s
        if c["cell_type"] == "code":
            compile(s, f"{name}:{c['id']}", "exec")
    for t in tokens:
        assert t in full, f"{name}: missing token {t!r}"
    for bad in ("SMOKE", "dummy"):
        assert bad not in full, f"{name}: stale token {bad!r} present"
    print(f"{name}: OK ({n_cells} cells, {len(json.dumps(nb))} bytes)")


def main():
    fetch3 = FETCH3_TMPL.replace("__BUILD_OF__", repr(BUILD_OF))

    chunk1 = assemble([
        md(MD1, "chunk1-00"),
        code(CELLS[1], "chunk1-01"),                      # install
        code(CELLS[2], "chunk1-02"),                      # config
        code(CELLS[3], "chunk1-03"),                      # push helper
        code(override_cell("chunk1_outputs"), "chunk1-04"),
        code(CELLS[4], "chunk1-05"),                      # refs setup
        code(filtered_sumstats(FAST), "chunk1-06"),
        code(magma_cell("chunk 1: fast MAGMA batch (8 traits)"), "chunk1-07"),
        code(CELLS[7], "chunk1-08"),                      # section prep
        code(REPORT1, "chunk1-09"),
    ])
    check(chunk1, "chunk1", 10, ["CHUNK_TRAITS", "chunk1_outputs",
                                 "chunk1_magma_report.csv", "chunk1_prep_report.csv",
                                 "Celiac", "DLBCL"])

    chunk2 = assemble([
        md(MD2, "chunk2-00"),
        code(CELLS[1], "chunk2-01"),
        code(CELLS[2], "chunk2-02"),
        code(CELLS[3], "chunk2-03"),
        code(override_cell("chunk2_outputs"), "chunk2-04"),
        code(CELLS[4], "chunk2-05"),
        code(filtered_sumstats(REST), "chunk2-06"),
        code(magma_cell("chunk 2: remaining MAGMA batch (8 traits)"), "chunk2-07"),
        code("CHUNK1_TRAITS = " + repr(FAST) + "\n" + FETCH2, "chunk2-08"),
    ])
    check(chunk2, "chunk2", 9, ["CHUNK_TRAITS", "chunk2_outputs", "CHUNK1_TRAITS",
                                "magma_combined_report.csv", "T1D", "Lymphocyte_Count"])

    chunk3 = assemble([
        md(MD3, "chunk3-00"),
        code(CELLS[1], "chunk3-01"),
        code(CELLS[2], "chunk3-02"),
        code(CELLS[3], "chunk3-03"),
        code(override_cell("chunk3_outputs"), "chunk3-04"),
        code(CELLS[4], "chunk3-05"),                      # refs setup
        code(CELLS[7], "chunk3-06"),                      # section prep
        code(fetch3, "chunk3-07"),
        code(CELLS[8], "chunk3-08"),                      # externals
        code(CELLS[9], "chunk3-09"),                      # pipeline block 1
        code(CELLS[10], "chunk3-10"),                     # driver
        code(CELLS[11], "chunk3-11"),                     # panel override
        code(CELLS[12], "chunk3-12"),                     # runlib
        code(CELLS[13], "chunk3-13"),                     # run A1
        code(CELLS[14], "chunk3-14"),                     # run D1
        code(CELLS[15], "chunk3-15"),                     # stage 4C
    ])
    check(chunk3, "chunk3", 16, ["chunk3_outputs", "BUILD_OF", "PREPPED",
                                 "run_and_save", "Adult_Standing_Height"])

    out = {"hello_pilot_chunk1_fast.ipynb": chunk1,
           "hello_pilot_chunk2_rest.ipynb": chunk2,
           "hello_pilot_chunk3_driver.ipynb": chunk3}
    for name, nb in out.items():
        (HERE / name).write_text(json.dumps(nb, indent=1))
        print("wrote", HERE / name)


if __name__ == "__main__":
    main()
