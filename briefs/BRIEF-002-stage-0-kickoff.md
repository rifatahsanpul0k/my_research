# BRIEF-002 — Stage 0 kickoff: data ingestion + GitHub push smoke test

**Status:** APPROVED — start immediately. This is the first job. Do not start Stage 1 (Phase 1 matrix) until BRIEF-003.

## Read first (in order)

1. `HANDOFF.md` — the bridge rules (you read this every session).
2. `briefs/BRIEF-001-research-brief.md` — the full research brief. §1 (the unsupervised firewall) is law.
3. `DATASETS.md` in your local repo root — the six Drive links. **Private: never commit it, never paste the links into chat, the notebook, or any committed file.** Inject them into notebook code only.
4. `KAGGLE_GITHUB_SETUP.md` — the exact notebook cells for the GitHub push pattern.

## Task 1 — connect to the notebook

Through your Kaggle MCP: `kernels_list` with search `research_notebook` (exact title, your own account) → take its `kernel_slug` → `kernel_pull` to read the current code. Check your MCP panel's tool list first — exact tool names vary by server. Pushing a new version (`kernel_push`) is what executes it on Kaggle; poll `kernel_status`, fetch `kernel_output`.

## Task 2 — one-time data ingestion

Push a notebook version (Internet ON; Secrets `GITHUB_TOKEN`, `KAGGLE_USERNAME`, `KAGGLE_KEY` attached) that:

1. `pip install gdown`.
2. Downloads the six Drive folders from DATASETS.md into `/kaggle/working/data/<ID>/` for A1, D1, E11, E13, E15, E18 — inside Kaggle, server-to-server. Your Mac never downloads the data.
3. Writes `~/.kaggle/kaggle.json` from the two Kaggle secrets (`chmod 600`) and creates **one private Kaggle dataset** from the download. This dataset persists — every later run mounts it as a data source; nothing is ever downloaded twice.
4. Verifies: list the dataset's files and confirm all six folders with their expected contents (`adata_RNA.h5ad`; human sets also `adata_ADT.h5ad` + `annotation.csv`; mouse sets also `adata_ATAC.h5ad` + `anno.csv`).

Firewall note: the annotation CSVs download along as cargo — your ingestion code must not compute anything from them. They are post-hoc-only (§1).

## Task 3 — smoke test the GitHub push

Per `KAGGLE_GITHUB_SETUP.md`: write `runs/_smoke/ping.json` (`{"ok": true}`), push it with the notebook's git pattern, confirm it appears on github.com, then remove it. **If the smoke test fails, STOP — do not proceed.** Report the error in your stage report instead.

## Task 4 — report

Write `reports/REPORT-stage0.md`: the dataset slug, the file-verification table, the smoke test result, wall-clock time. Commit + push. Phoenix reviews the report and replies with BRIEF-003.

## Explicitly out of scope for this stage

- No training, no encoders, no clustering, no metrics yet.
- The two firewall fixes flagged in PULOK's training snippet (k derived from label counts; `ground_truth` passed into `train_model`) come before Stage 1 — first locate the full pipeline code; the snippet was incomplete.
- Compute budget and run budget (§9) are still PULOK's decisions; the loop's termination conditions are set in BRIEF-003.
