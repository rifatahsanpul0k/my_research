# BRIEF-006 — Stage 3: Synthesis. Best Fused Embedding.

## 0. Status gate
- Stage 2 is **accepted**. `reports/REPORT-s2.md` verified line-by-line against `runs/s2/registry.jsonl` (all six Borda tables, ranks, ARI means match).
- **Two amendments required to REPORT-s2.md** (do these first, same commit discipline):
  1. §5: Attn-SAGE raised the over-clustering flag on **6 of 6** datasets, not 5 — E15 also flags (|sil rank 3 − stab rank 10| = 7).
  2. §3/§5: Gated-ARISE does **not** tie for highest average post-hoc ARI. `S2-EXP05-ABL-recon-pca30` averages **0.187**, the actual highest; Gated-ARISE is 0.173.
- (Rhetorical: "resolved the over-clustering pathology" → "reduced"; EXP09 itself is flagged on 4/6.)

## 1. Objective
Build the best **single fused embedding** across all six datasets by combining the verified winners of Stage 2 and the embedding-module sweep. One identical final architecture after modality-appropriate preprocessing (per BRIEF-001 mission). This is the synthesis stage — breadth first, then the champion.

## 2. Evidence going in (do not re-litigate)
**Stage 2 (verified):** Winner `S2-EXP09-ABL-spa-laplacian` (Laplacian-H-HiRe: SAGEConv + hierarchical 2-stage MLP fusion + 3000-HVG recon + Laplacian Dirichlet spatial loss), cross-dataset avg rank **2.00**. Runner-up Gated-ARISE (avg 4.50, zero flags). Dead: Attn-SAGE (6/6 over-clustering flags, worst post-hoc ARI nearly everywhere), concat fusion (dead last, avg 8.17), dense spatial BCE on mouse (collapses E13/E15/E18; wins D1 only).
**Module sweep (26 combos × 6 datasets × 3 seeds, Phoenix-analyzed):** `ALIGN-scot-sinkhorn` is the dominant cross-dataset module (cross-dataset Borda 4.7 vs next-best 11.0; mean sil 0.762, mean stab 0.774). **Open question:** SCOT's silhouette reaches 0.95+ — verify with post-hoc ARI (quarantined) that it is not a stable collapse mode. Also verified: SCAttn-spatial is an extreme mirage (sil ~0.99, stab ~0.15); VCR and SigLIP2 are dead; dual-graph dead; UAF-Gaussian has top-tier stability (esp. E11/E15); GAT encoder is stable and competitive; SCIGMA-τ ≈ fixed high-τ (drop it); contrastive losses (InfoNCE/GATCL) are crisp-but-unstable in 32-D.

## 3. Stage 3 matrix (bounded; one-module-at-a-time, then champion)
Base = `S2-EXP09` (Laplacian-H-HiRe). All six datasets, seeds 42/1234/2024.
- `S3-CAND-A` — Base + **SCOT-sinkhorn alignment** (auxiliary OT loss). Answers the open question: does the sweep's best module transfer? Post-hoc ARI on this config is the SCOT verdict.
- `S3-CAND-B` — Base with fusion **hierarchical-MLP → UAF-Gaussian**.
- `S3-CAND-C` — Base with encoder **SAGEConv → GAT**.
- `S3-ABL-recon` — on the best of {A,B,C} by cross-dataset Borda: reconstruction target **3000-HVG vs PCA30 vs PCA128**.
- `S3-ABL-cluster` — **learnable clustering axis** (parked per user instruction, now activated): k-means vs **DEC** vs **IDEC** heads on the champion embedding. Clustering only; selection stays label-free; post-hoc ARI reporting-only.
- `S3-CHAMPION` (conditional) — combine the winning modules from A/B/C + winning recon target **only if** each individually beats Base on cross-dataset Borda. Otherwise champion = best single candidate. No blind stacking.

Total ≤ 7 configs × 6 datasets × 3 seeds = 126 runs.

## 4. Fixed protocol (unchanged)
- **k**: read once from `DECLARED_K` per dataset (A1=10, D1=11, E11=8, E13=12, E15=12, E18=14), applied identically to every config. Declared exception, disclosed in final report.
- **Selection:** RAUS v2 — Borda over {silhouette, seed-stability ARI, reconstruction-balance}; kNN-overlap logged, not voted. Over-clustering guard: |Rank_sil − Rank_stab| ≥ 4 → flag.
- **Firewall:** `tests/test_firewall.py` must pass before any run. Labels touch only `posthoc.json` (`quarantined_reporting_only: true`). Never print or commit secrets; never embed tokens in command strings.
- **Sync:** push `runs/s3/registry.jsonl` + per-run folders after **every** run (resume via `skip_done`). Check Kaggle GPU quota before the batch (30 h/week cap); Stage 2 used ~70 min for 180 runs — this stage is smaller.
- **Stop:** any firewall failure, quota exhaustion, or NaN/collapse in >1 config → stop, report, await instruction.

## 5. Deliverables
- `runs/s3/registry.jsonl` (+ per-run folders, same layout as Stage 2).
- `reports/REPORT-s3.md`: RAUS v2 tables per dataset (same format as REPORT-s2.md), cross-dataset summary, explicit **SCOT post-hoc ARI verdict** (breakthrough vs stable-collapse), ablation analysis, named champion architecture with full config (λ values, dims), and the DEC/IDEC clustering comparison.
- No Stage 4 until Phoenix reviews REPORT-s3.md and issues the next brief.

## 6. Note on sweep artifacts
The module-sweep results are currently Kaggle-local (push failed: pre-existing `/kaggle/working/my_research` broke the clone step). **Not blocking** — Phoenix already analyzed the completed run's statistics. Do not re-run the sweep. (PULOK: re-running the notebook's push cell with the stale dir removed will land `runs/sweep/`.)
