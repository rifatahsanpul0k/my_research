# BRIEF-007 — Stage 4: Champion Validation, Selection-Criterion Repair, First New Idea

## 0. Status gate
- Stage 3 is **accepted**. `reports/REPORT-s3.md` verified line-by-line against `runs/s3/registry.jsonl` (144 runs; Borda tables, SCOT ARIs, IDEC 11.00 all match). One nit (does not block): "average rank 3.00" for IDEC — mean per-dataset placement recomputes to ~2.17.
- **Do not re-litigate:** SCOT is a stable collapse (post-hoc ARI ≈ 0 on all four mouse sets; permanently disqualified). UAF is a mirage on mouse under Laplacian (sil ~0.9, stab ~0). Both are dead — do not build on them.

## 1. Objective
Three bounded goals: (a) validate the evidence-based champion in this pipeline; (b) repair the selection criterion — RAUS v2 demonstrably prefers worse models (see §2); (c) test exactly one genuinely new idea (uniformity anti-collapse).

## 2. Evidence going in (do not re-litigate)
- **Stage 3 champion (agent pipeline):** Laplacian-H-HiRe + PCA-64 recon + IDEC head (`S3-ABL-clus-idec`), cross-dataset Borda **11.00** vs Base 12.17, mean post-hoc ARI 0.254.
- **Recalibration finding (Phoenix notebook, `runs/s3-notebook/`):** at λ_spa=2.0 the Laplacian outweighed reconstruction ~16:1 at init and RNA R² collapsed to 1–3% — the embedding was smoothing-only. At λ_spa=0.1, R² recovers to 10–31% and post-hoc ARI jumps: E15 0.385→**0.503**, E13 0.281→**0.440**, E11 0.317→**0.374**. First 0.5+ ARI. Human sets move modestly (A1 0.222, D1 0.202).
- **The selection problem:** on the recalibrated models, silhouette *falls* (0.13–0.36) while ARI *rises*. RAUS v2 (which votes silhouette) would select the old smooth-but-wrong embeddings over the better ones. Silhouette is actively misleading under over-smoothing. Consensus-across-seeds was tested and **failed** (deltas −0.15 to +0.01) — do not pursue it.

## 3. Stage 4 matrix
All six datasets, seeds 42/1234/2024, same preprocessing as Stage 3.

**3A. Champion validation (2 configs).** Evidence-based champion: **GAT encoder + hierarchical-MLP fusion + Laplacian (λ per §4) + PCA30 recon + IDEC head**, and a **SAGEConv-encoder variant** as ablation. (GAT was the most robust Stage-3 encoder and won D1; PCA30 gave the best single ARI, E15 0.385; IDEC is the cross-dataset winner.)

**3B. RAUS v3 criterion study (no GPU — meta-analysis).** Using the registries from `runs/s3/` and `runs/s3-notebook/` (all configs, all datasets): for each candidate criterion below, rank the configs per dataset and compute the rank-correlation (Spearman) between the criterion's ranking and the quarantined post-hoc ARI ranking. Candidates:
- v2 (baseline): Borda{silhouette, seed-stability, recbal}
- v3a: Borda{seed-stability, recbal} (drop silhouette)
- v3b: Borda{seed-stability, recbal, R²_rna} (reward alive reconstruction; R²_rna is label-free)
- v3c: your proposal, if you have a better label-free idea — document it.
Report per-dataset and mean Spearman ρ for each candidate. The winner becomes the selection criterion going forward. This is the most important deliverable of Stage 4 after the champion.

**3C. One new idea: uniformity anti-collapse (1 config).** VCR failed in the sweep (destroyed geometry). Test the gentler alternative on the recalibrated base: add the Wang & Isola uniformity loss L_unif = log E_{i≠j}[exp(−t·||z_i − z_j||²)] (t=2.0 default; subsample pairs per batch for cost), weight λ_unif=0.1. Verdict criteria: no geometry destruction (RAUS + quarantined ARI vs recalibrated base), no new mirage signatures (sil ~1 with stab ~0, or stab high with ARI ~0).

Total ≤ 3 configs × 18 runs + λ-calibration runs (§4) ≈ **≤ 63 runs**.

## 4. λ calibration protocol (do NOT copy λ=0.1 blindly)
Loss scales differ between pipelines. Calibrate λ_spa in *this* pipeline with a label-free criterion: binary-search λ_spa on **E15 only, seed 42**, base config, until mean **R²_rna ∈ [0.10, 0.35]** (the "reconstruction-alive band" — embedding encodes RNA without the Laplacian going silent). Then freeze that λ for all six datasets and every Stage-4 config. Log the calibration runs. If no λ reaches the band, stop and report.

## 5. Fixed protocol (unchanged)
- **k**: `DECLARED_K` (A1=10, D1=11, E11=8, E13=12, E15=12, E18=14), identical for every config. Declared exception, disclosed in final report.
- **Firewall:** `tests/test_firewall.py` must pass before any run. Labels touch only `posthoc.json` (`quarantined_reporting_only: true`). Never print or commit secrets; never embed tokens in command strings.
- **Sync:** push `runs/s4/registry.jsonl` + per-run folders after **every** run (resume via `skip_done`). Check Kaggle GPU quota first (30 h/week cap).
- **Stop:** firewall failure, quota exhaustion, NaN/collapse in >1 config, or λ calibration fails → stop, report, await instruction.

## 6. Deliverables
- `runs/s4/registry.jsonl` (+ per-run folders, same layout as Stage 3).
- `reports/REPORT-s4.md`: champion validation (both encoder variants vs Stage-3 IDEC baseline), **RAUS v3 verdict** (Spearman table per candidate criterion; name the new selection rule), uniformity verdict, and the named final architecture with full config. Explicitly state which findings replicate the Phoenix notebook and which differ.
- No Stage 5 until Phoenix reviews REPORT-s4.md and issues the next brief.
