# BRIEF-003 — Stage 1: Phase 1 screening matrix (APPROVED)

**Status:** APPROVED — start as soon as the pre-flight fixes below are done. Stage 0 is green (data ingested, private dataset live, push pattern verified).

## Budgets (PULOK's decisions, 2026-10-08)

- **Compute:** Kaggle **T4 x2, 30 GPU-hours per week — hard cap**. Before every run batch, check the accelerator quota via the Kaggle MCP (`get_accelerator_quota`) and stop the loop before exceeding the weekly quota. The quota resets weekly; plan Stage 1 to fit inside one week.
- **Run budget:** Stage 1 = the designed Phase 1 matrix — **15 experiments × screening seeds (50 runs), ≥3 seeds each**. The loop terminates when the matrix is exhausted **or** the weekly GPU quota runs out, whichever comes first. No new configs beyond this matrix without a new brief.

## Pre-flight fixes (BEFORE the first Stage 1 run)

1. **k comes from ground-truth class counts — DECLARED EXCEPTION (decided by PULOK 2026-10-08).** PULOK confirms: in his pipeline `n_clusters` is always the label counts from ground truth. This is honestly label use in the decision loop (the field-standard form of leakage), so it is recorded here as an explicit exception to §1, not a hidden violation:
   - It is applied **identically to every config** in the matrix, so it cannot bias RAUS *ranking* between configs — selection stays label-free.
   - It keeps absolute ARI numbers comparable to published baselines, which all do the same.
   - The final report must disclose it as a limitation: the "unsupervised" claim covers training and model selection; k uses annotation counts, as in the compared literature.
   - The original snippet's pattern (`ground_truth.nunique()` computed inline at runtime) is still replaced: k is read once from the dataset registry as a declared constant, never recomputed by inspecting an annotation file mid-pipeline.
   - *If PULOK later wants the strict version:* a label-free k-sweep (silhouette-selected) can be added as an ablation. Default stays with the declared exception.
2. **`ground_truth` inside `train_model`.** Audit the full pipeline code — PULOK confirms the ground-truth paths are given (`annotation.csv` / `anno.csv` per dataset, see DATASETS.md). `ground_truth` may enter **only** the post-hoc logging path (`posthoc.json`, reporting-only). Refactor so the training and selection code cannot import it. Encode as a runtime assertion (`test_firewall.py`): any selection-path access to labels raises.
3. The annotation loader must expose labels through a separate, clearly-marked post-hoc-only channel from the start — no shared dataframe columns that training code could accidentally read.

## Stage 1 execution

- **Data:** mount the private dataset `pulokpulok/spatial-multiomics-6datasets-private` as a notebook data source. Never re-download.
- **Notebook:** continue in `research_notebook` (or a dedicated Stage 1 notebook — agent's choice; results land in the same repo paths).
- **Runs:** the 15-experiment screening matrix, ≥3 seeds per experiment, `skip_done=True`. Preprocessing defaults per BRIEF-001 §3.3.
- **Sync:** push after EVERY run — `runs/s1/<exp_id>/` (metrics, embeddings, assignments, logs, plots) plus the appended `registry.jsonl` — via the notebook's direct GitHub push. GitHub is the single source of truth.
- **Resume:** on Kaggle disconnect, pull the repo, read the registry, rerun only the interrupted experiment.
- **Selection:** RAUS Borda over {silhouette, seed-stability ARI, kNN-overlap} — label-free, always. Post-hoc ARI/NMI vs annotations go in `posthoc.json`, quarantined, reporting-only.

## Report

Write `reports/REPORT-s1.md`: RAUS ranking table per dataset, metric distributions across seeds (mean ± std, not best-seed only), what failed and why, the proposed Stage 2 (configs + one-line justifications), unresolved questions. Commit + push. **Do not start Stage 2 without BRIEF-004.**
