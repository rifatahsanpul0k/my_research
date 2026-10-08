# BRIEF-005 — Stage 2 APPROVED: robust architecture search + ablations

**Status:** APPROVED. REPORT-s1.md v2 is ACCEPTED — Phoenix independently verified the per-dataset Borda rankings, the corrected seed-stability values, and the silhouette↔ARI correlations against `runs/s1/registry.jsonl`. All match. Proceed to Stage 2.

## What Stage 2 runs

**Candidates** (from the report §8, grounded in cross-dataset robustness):
1. **H-HiRe** — hierarchical 2-stage MLP fusion (EXP03) + raw 3000-HVG reconstruction (EXP05) + calibrated spatial loss (λ_spa ≤ 2.0, λ_rec = 15.0). No dense BCE.
2. **Attn-SAGE** — cross-modality multihead attention fusion (EXP09) on a SAGEConv backbone (EXP01), avoiding spectral-GCN over-smoothing.
3. **Gated-ARISE** — ARISE with learned dynamic edge gates to prune edges crossing transcriptomic boundaries.

**Ablation matrix** (one module at a time from the strongest base):
- Fusion: concat vs 2-stage MLP vs cross-attention
- Reconstruction: PCA-30 vs raw 3000 HVGs vs decoupled per-modality recon
- Spatial loss: zero vs contrastive BCE vs graph Laplacian vs edge-gated
- Backbone: spectral GCNConv vs SAGEConv vs GAT

Bound the matrix BEFORE running: state the exact config count upfront. One-module-at-a-time, no combinatorial explosion. Stop at budget exhaustion.

## Datasets & selection (changed per Stage 1 findings)

- **All six datasets**: A1, D1, E11, **E13, E15, E18**. No more 3-dataset screening subset.
- **RAUS v2 for selection**: Borda over **{silhouette, seed-stability ARI, reconstruction-balance}** — kNN-overlap is DROPPED from the selection formula (degenerate, per the report §5). Keep logging it, but it doesn't vote.
- **Reconstruction-balance metric**: define it label-free (e.g., per-modality reconstruction loss ratio or relative explained variance — RNA vs AUX). The agent defines it precisely in the Stage 2 report; it must not touch labels.
- **Over-clustering guard (open research problem from Stage 1)**: the 2-metric Borda would still have selected EXP04 on E11 — the pathological config. For every candidate, report silhouette-rank vs stability-rank divergence as a diagnostic red flag, and discuss any config where geometric tightness and stability disagree. Do not "fix" this by peeking at ARI.

## Standing rules (unchanged)

- ≥3 seeds per config, `skip_done=True`; push after EVERY run (`runs/s2/<exp_id>/` + `registry.jsonl`); resume-only-reruns on disconnect.
- Firewall: `test_firewall.py` must pass before the first Stage 2 run and stay green. Labels → `posthoc.json` only. Declared-k exception stands, read once from registry.
- Budget: 30 GPU-hours/week hard cap, quota checked before each batch.
- Mount `pulokpulok/spatial-multiomics-6datasets-private`; never re-download.

## Report

`reports/REPORT-s2.md` in the v2 format: per-dataset Borda tables with ranks, corrected findings, the over-clustering diagnostic, and a named winning architecture (or a justified "no single winner" with the evidence). Commit + push. **Do not start Stage 3 without BRIEF-006.**
