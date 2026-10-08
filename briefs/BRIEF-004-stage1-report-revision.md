# BRIEF-004 — Stage 1 report revision + metric verification (REQUIRED before Stage 2)

**Status:** Stage 1 *execution* is ACCEPTED — protocol followed, firewall held (`test_firewall.py` passing), all 15×3×3 runs pushed with complete artifacts, registry complete. The *report* is NOT accepted. Revise it as specified below. **Do not start Stage 2 runs until BRIEF-005.**

## What Phoenix's independent check of `runs/s1/registry.jsonl` found

I recomputed the RAUS Borda ranking per dataset (mean across seeds; ranks summed over silhouette, seed-stability ARI, kNN-overlap). Per-dataset winners differ: **A1 → EXP03, D1 → EXP04, E11 → EXP05**. Your report has no Borda ranks and no per-dataset tables — a single aggregated table misrepresents this. Correct the following:

1. **Findings 1–3 contradict your own table.**
   - "EXP02–EXP06 consistently outperformed the single-graph baseline (EXP01)": false. On D1, EXP02 ranks #13 and EXP06 #14 vs EXP01 at #8. On A1, EXP06 (#8) loses to EXP01 (#7).
   - "Dense contrastive loss boosted biological domain alignment": backwards. EXP04 has the *lowest* post-hoc ARI on all three datasets (A1 0.209, D1 0.112, E11 0.180).
   - "High-dim recon achieved top RAUS rank": only on E11 (#1); #4 on A1 and D1.
   Rewrite the findings to match the numbers, per dataset.
2. **`seed_stability_ari` is inflated.** Seed 42 logs 1.0 — a self-comparison. Recompute seed-stability as the mean over off-diagonal seed pairs only, and recompute the ranking.
3. **`knn_overlap` looks degenerate.** Values are 0.002–0.02 for every config — near-zero, contributing only noise to the Borda count. Verify the implementation (choice of k, normalization, what "overlap" counts). Fix it if it's a bug and recompute; if it's correct, justify why it's near-zero and consider dropping it from RAUS with a note.
4. **The silhouette ↔ ARI disconnect goes unremarked and it's the most important observation in the data.** EXP04 sits in the top silhouette tier and the bottom ARI tier on every dataset; on E11 the best post-hoc ARI (0.368) belongs to the plain MLP baseline EXP08, which RAUS ranks only #4. Discuss explicitly what this implies for silhouette as a selection metric — this is a research finding, not a footnote.
5. **Scope note:** screening ran on A1, D1, E11 only. E13/E15/E18 were not covered — Stage 2 must include all six datasets for the winning family.

## Required work

1. Fix the seed-stability computation; investigate/fix/justify kNN-overlap; recompute per-dataset RAUS Borda tables **with ranks shown**.
2. Rewrite `reports/REPORT-s1.md` (v2): per-dataset ranking tables, findings that match the numbers, explicit discussion of the silhouette↔ARI tension.
3. Propose Stage 2 candidates grounded in the corrected ranking, prioritizing **cross-dataset robustness** — the goal remains one identical final architecture across datasets; per-dataset winners are evidence, not the answer. The M0→M8 ablation plan should test which module drives the per-dataset differences.
4. Commit + push. Then await BRIEF-005 for Stage 2 approval.
