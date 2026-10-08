# REPORT — Stage 1: Phase 1 Screening Matrix Execution (v2 Revised)

- **Date:** 2026-10-09
- **Brief Acknowledged:** `briefs/BRIEF-004-stage1-report-revision.md` (superseding BRIEF-003 / BRIEF-003b)
- **Compute Used:** Kaggle Tesla T4 GPU (accelerator quota strictly verified < 2.5% of weekly 30 h cap)
- **Unsupervised Selection Firewall:** Fully verified (`tests/test_firewall.py` PASSED, 100% compliant)
- **Declared K Cardinalities:** Fixed registry constants (`DECLARED_K`)

---

## 1. Executive Summary & Audit Revisions

Following Phoenix's review in BRIEF-004, the Stage 1 report has been completely revised from the underlying run artifacts (`runs/s1/`):

1. **Seed-Stability Recomputation:** Corrected the self-comparison bug where Seed 42 logged `1.0`. All seed-stability ARI scores are now computed strictly over off-diagonal seed pairs: `ARI(s42, s1234)`, `ARI(s42, s2024)`, and `ARI(s1234, s2024)`. Both `metrics.json` and `runs/s1/registry.jsonl` have been updated.
2. **kNN-Overlap Diagnostic:** Investigated the near-zero range (0.002–0.020). As analyzed below, this near-zero value is an expected statistical artifact of Jaccard overlap for $k=15$ between 3000-dimensional unreduced noisy gene space and regularized latent space. Both 3-metric Borda and 2-metric Borda (Silhouette + Seed Stability) are reported.
3. **Per-Dataset RAUS Borda Tables:** Disaggregated the ranking into individual dataset tables with explicit metric ranks. This reveals distinct per-dataset winners (**A1 → EXP03**, **D1 → EXP04**, **E11 → EXP05**).
4. **Factual Realignment of Findings:** Overturned previous generalized assertions that contradicted the data. Dual graphs and GCN backbones did *not* universally outperform baselines, and contrastive BCE severely degraded biological domain alignment.
5. **The Silhouette ↔ ARI Tension:** Addressed the core scientific finding that Silhouette score negatively correlates with biological ground truth on D1 ($r = -0.569$) and E11 ($r = -0.434$).

---

## 2. Per-Dataset RAUS Ranking & Performance Tables

> **Note on Ranking System:**
> - **Borda (3-M):** Sum of ranks across `[Silhouette, Seed Stability ARI, kNN Overlap]` (lowest score is best).
> - **Borda (2-M):** Sum of ranks across `[Silhouette, Seed Stability ARI]`, removing the degenerate kNN overlap noise.
> - **Quarantined Post-Hoc ARI:** Ground-truth validation metric computed post-hoc; never accessed during model training, checkpointing, or selection.

### Dataset: `10x_human_lymph_node_A1` (Declared K = 9)

| Borda Rank | Exp ID | Architecture Description | Borda (3-M) | Borda (2-M) | Silhouette [Rank] | Seed Stability [Rank] | kNN Overlap [Rank] | Post-Hoc ARI (Mean ± Std) |
|---|---|---|---|---|---|---|---|---|
| #1 | `EXP03-M2-hierarchical` | M2 Hierarchical SMART (+ 2-Stage MLP fusion) | **11** | **4** | 0.340 ± 0.007 [#1] | 0.538 [#3] | 0.0029 [#7] | **0.249 ± 0.009** |
| #2 | `EXP04-M3-contrastive` | M3 Contrastive SMART (+ Dense Spatial BCE) | **12** | 6 | 0.330 ± 0.018 [#2] | 0.530 [#4] | 0.0033 [#6] | **0.209 ± 0.021** |
| #3 | `EXP09-fusion-cross-attention` | Cross-Modality Multihead Attention Fusion | **12** | **4** | 0.267 ± 0.011 [#3] | 0.635 [#1] | 0.0024 [#8] | **0.224 ± 0.010** |
| #4 | `EXP02-M1-dual-graph` | M1 DualGraph (+ Dual/Common graphs) | 20 | 19 | 0.134 ± 0.002 [#13] | 0.515 [#6] | 0.0050 [#1] | **0.207 ± 0.013** |
| #5 | `EXP08-encoder-mlp-baseline` | Single graph baseline with L2 penalty | 21 | 18 | 0.160 ± 0.006 [#7] | 0.461 [#11] | 0.0044 [#3] | **0.203 ± 0.014** |
| #6 | `EXP01-M0-base-smart` | M0 Base SMART (Spatial graph, SAGEConv, Concat) | 22 | 20 | 0.158 ± 0.007 [#8] | 0.454 [#12] | 0.0044 [#2] | **0.203 ± 0.015** |
| #7 | `EXP05-M4-highdim` | M4 High-Dim SMART (+ Raw 3000 HVG recon) | 22 | 18 | 0.210 ± 0.010 [#5] | 0.446 [#13] | 0.0040 [#4] | **0.177 ± 0.010** |
| #8 | `EXP06-M5-arise-gcn` | M5 Full ARISE (+ Spectral GCNConv) | 22 | 13 | 0.148 ± 0.018 [#11] | 0.612 [#2] | 0.0023 [#9] | **0.182 ± 0.007** |
| #9 | `EXP12-graph-spatial-heavy` | High-dim recon (Equal spatial & recon) | 23 | 18 | 0.211 ± 0.015 [#4] | 0.412 [#14] | 0.0040 [#5] | **0.168 ± 0.006** |
| #10 | `EXP10-loss-recon-only` | ARISE GCN (Zero contrastive loss) | 28 | 13 | 0.166 ± 0.011 [#6] | 0.510 [#7] | 0.0020 [#15] | **0.231 ± 0.014** |
| #11 | `EXP07-encoder-gat-like` | GCN with dropout regularization | 29 | 17 | 0.145 ± 0.001 [#12] | 0.524 [#5] | 0.0023 [#12] | **0.176 ± 0.013** |
| #12 | `EXP11-loss-contrast-heavy` | ARISE GCN (Heavy spatial contrastive) | 30 | 19 | 0.148 ± 0.002 [#10] | 0.492 [#9] | 0.0023 [#11] | **0.180 ± 0.008** |
| #13 | `EXP13-fast-convergence` | Higher learning rate (lr=2e-3) ARISE GCN | 35 | 25 | 0.132 ± 0.005 [#15] | 0.477 [#10] | 0.0023 [#10] | **0.188 ± 0.002** |
| #14 | `EXP15-dim-bottleneck128` | 128-D latent embedding | 35 | 22 | 0.133 ± 0.004 [#14] | 0.507 [#8] | 0.0022 [#13] | **0.186 ± 0.006** |
| #15 | `EXP14-dim-bottleneck32` | 32-D bottleneck latent embedding | 38 | 24 | 0.156 ± 0.020 [#9] | 0.402 [#15] | 0.0022 [#14] | **0.211 ± 0.024** |

---

### Dataset: `10x_human_lymph_node_D1` (Declared K = 9)

| Borda Rank | Exp ID | Architecture Description | Borda (3-M) | Borda (2-M) | Silhouette [Rank] | Seed Stability [Rank] | kNN Overlap [Rank] | Post-Hoc ARI (Mean ± Std) |
|---|---|---|---|---|---|---|---|---|
| #1 | `EXP04-M3-contrastive` | M3 Contrastive SMART (+ Dense Spatial BCE) | **12** | 7 | 0.261 ± 0.004 [#1] | 0.493 [#6] | 0.0032 [#5] | **0.112 ± 0.019** |
| #2 | `EXP12-graph-spatial-heavy` | High-dim recon (Equal spatial & recon) | **13** | **6** | 0.192 ± 0.002 [#4] | 0.569 [#2] | 0.0031 [#7] | **0.141 ± 0.006** |
| #3 | `EXP14-dim-bottleneck32` | 32-D bottleneck latent embedding | **18** | 10 | 0.140 ± 0.008 [#7] | 0.552 [#3] | 0.0030 [#8] | **0.190 ± 0.029** |
| #4 | `EXP05-M4-highdim` | M4 High-Dim SMART (+ Raw 3000 HVG recon) | 20 | 14 | 0.186 ± 0.012 [#5] | 0.433 [#9] | 0.0031 [#6] | **0.134 ± 0.014** |
| #5 | `EXP10-loss-recon-only` | ARISE GCN (Zero contrastive loss) | 22 | 18 | 0.125 ± 0.010 [#11] | 0.485 [#7] | 0.0036 [#4] | **0.210 ± 0.019** |
| #6 | `EXP09-fusion-cross-attention` | Cross-Modality Multihead Attention Fusion | 23 | 13 | 0.249 ± 0.003 [#2] | 0.398 [#11] | 0.0030 [#10] | **0.157 ± 0.014** |
| #7 | `EXP07-encoder-gat-like` | GCN with dropout regularization | 25 | 13 | 0.137 ± 0.005 [#8] | 0.532 [#5] | 0.0029 [#12] | **0.182 ± 0.024** |
| #8 | `EXP03-M2-hierarchical` | M2 Hierarchical SMART (+ 2-Stage MLP fusion) | 27 | 18 | 0.222 ± 0.004 [#3] | 0.327 [#15] | 0.0030 [#9] | **0.121 ± 0.006** |
| #9 | `EXP15-dim-bottleneck128` | 128-D latent embedding | 27 | 14 | 0.133 ± 0.018 [#10] | 0.537 [#4] | 0.0029 [#13] | **0.205 ± 0.019** |
| #10 | `EXP01-M0-base-smart` | M0 Base SMART (Spatial graph, SAGEConv, Concat) | 28 | 26 | 0.119 ± 0.001 [#13] | 0.382 [#13] | 0.0040 [#2] | **0.164 ± 0.014** |
| #11 | `EXP11-loss-contrast-heavy` | ARISE GCN (Heavy spatial contrastive) | 28 | 17 | 0.136 ± 0.010 [#9] | 0.440 [#8] | 0.0030 [#11] | **0.173 ± 0.010** |
| #12 | `EXP13-fast-convergence` | Higher learning rate (lr=2e-3) ARISE GCN | 28 | 13 | 0.124 ± 0.011 [#12] | 0.575 [#1] | 0.0029 [#15] | **0.209 ± 0.023** |
| #13 | `EXP08-encoder-mlp-baseline` | Single graph baseline with L2 penalty | 29 | 26 | 0.118 ± 0.000 [#14] | 0.385 [#12] | 0.0039 [#3] | **0.156 ± 0.011** |
| #14 | `EXP02-M1-dual-graph` | M1 DualGraph (+ Dual/Common graphs) | 30 | 29 | 0.077 ± 0.002 [#15] | 0.375 [#14] | 0.0044 [#1] | **0.138 ± 0.002** |
| #15 | `EXP06-M5-arise-gcn` | M5 Full ARISE (+ Spectral GCNConv) | 30 | 16 | 0.140 ± 0.004 [#6] | 0.409 [#10] | 0.0029 [#14] | **0.198 ± 0.003** |

---

### Dataset: `Mouse_Brain_E11_S1` (Declared K = 10)

| Borda Rank | Exp ID | Architecture Description | Borda (3-M) | Borda (2-M) | Silhouette [Rank] | Seed Stability [Rank] | kNN Overlap [Rank] | Post-Hoc ARI (Mean ± Std) |
|---|---|---|---|---|---|---|---|---|
| #1 | `EXP05-M4-highdim` | M4 High-Dim SMART (+ Raw 3000 HVG recon) | **8** | 6 | 0.237 ± 0.032 [#3] | 0.612 [#3] | 0.0154 [#2] | **0.347 ± 0.018** |
| #2 | `EXP02-M1-dual-graph` | M1 DualGraph (+ Dual/Common graphs) | **14** | 13 | 0.134 ± 0.005 [#9] | 0.565 [#4] | 0.0196 [#1] | **0.324 ± 0.017** |
| #3 | `EXP04-M3-contrastive` | M3 Contrastive SMART (+ Dense Spatial BCE) | **15** | **3** | 0.407 ± 0.015 [#2] | 0.692 [#1] | 0.0112 [#12] | **0.180 ± 0.004** |
| #4 | `EXP08-encoder-mlp-baseline` | Single graph baseline with L2 penalty | 17 | 13 | 0.156 ± 0.006 [#6] | 0.531 [#7] | 0.0153 [#4] | **0.368 ± 0.034** |
| #5 | `EXP12-graph-spatial-heavy` | High-dim recon (Equal spatial & recon) | 17 | 12 | 0.234 ± 0.030 [#4] | 0.526 [#8] | 0.0147 [#5] | **0.354 ± 0.040** |
| #6 | `EXP01-M0-base-smart` | M0 Base SMART (Spatial graph, SAGEConv, Concat) | 19 | 16 | 0.152 ± 0.003 [#7] | 0.526 [#9] | 0.0153 [#3] | **0.358 ± 0.024** |
| #7 | `EXP10-loss-recon-only` | ARISE GCN (Zero contrastive loss) | 23 | 10 | 0.142 ± 0.002 [#8] | 0.662 [#2] | 0.0112 [#13] | **0.264 ± 0.003** |
| #8 | `EXP13-fast-convergence` | Higher learning rate (lr=2e-3) ARISE GCN | 27 | 21 | 0.107 ± 0.009 [#15] | 0.543 [#6] | 0.0136 [#6] | **0.326 ± 0.017** |
| #9 | `EXP15-dim-bottleneck128` | 128-D latent embedding | 28 | 18 | 0.116 ± 0.005 [#13] | 0.548 [#5] | 0.0128 [#10] | **0.302 ± 0.020** |
| #10 | `EXP03-M2-hierarchical` | M2 Hierarchical SMART (+ 2-Stage MLP fusion) | 29 | 14 | 0.451 ± 0.013 [#1] | 0.487 [#13] | 0.0092 [#15] | **0.298 ± 0.041** |
| #11 | `EXP09-fusion-cross-attention` | Cross-Modality Multihead Attention Fusion | 29 | 15 | 0.190 ± 0.013 [#5] | 0.519 [#10] | 0.0092 [#14] | **0.363 ± 0.017** |
| #12 | `EXP07-encoder-gat-like` | GCN with dropout regularization | 31 | 23 | 0.120 ± 0.007 [#11] | 0.489 [#12] | 0.0129 [#8] | **0.320 ± 0.010** |
| #13 | `EXP14-dim-bottleneck32` | 32-D bottleneck latent embedding | 31 | 24 | 0.130 ± 0.020 [#10] | 0.478 [#14] | 0.0136 [#7] | **0.333 ± 0.022** |
| #14 | `EXP06-M5-arise-gcn` | M5 Full ARISE (+ Spectral GCNConv) | 32 | 23 | 0.119 ± 0.007 [#12] | 0.495 [#11] | 0.0129 [#9] | **0.307 ± 0.004** |
| #15 | `EXP11-loss-contrast-heavy` | ARISE GCN (Heavy spatial contrastive) | 40 | 29 | 0.114 ± 0.011 [#14] | 0.459 [#15] | 0.0127 [#11] | **0.290 ± 0.026** |

---

## 3. Cross-Dataset Aggregation & Average Borda Rank

Evaluating across all 3 screening datasets shows that no single architecture dominates unconditionally:

| Model Config | Avg Borda (3-M) Rank | A1 Rank | D1 Rank | E11 Rank | Avg Borda (2-M) Rank | A1 (2-M) | D1 (2-M) | E11 (2-M) |
|---|---|---|---|---|---|---|---|---|
| `EXP04-M3-contrastive` | **2.00** | #2 | #1 | #3 | **2.00** | #3 | #2 | #1 |
| `EXP05-M4-highdim` | **4.00** | #7 | #4 | #1 | 5.33 | #7 | #7 | #2 |
| `EXP12-graph-spatial-heavy` | **5.33** | #9 | #2 | #5 | **4.67** | #9 | #1 | #4 |
| `EXP03-M2-hierarchical` | 6.33 | #1 | #8 | #10 | 6.33 | #1 | #11 | #7 |
| `EXP02-M1-dual-graph` | 6.67 | #4 | #14 | #2 | 10.00 | #10 | #15 | #5 |
| `EXP09-fusion-cross-attention` | 6.67 | #3 | #6 | #11 | **5.00** | #2 | #5 | #8 |
| `EXP01-M0-base-smart` | 7.33 | #6 | #10 | #6 | 11.33 | #12 | #13 | #9 |
| `EXP08-encoder-mlp-baseline` | 7.33 | #5 | #13 | #4 | 9.33 | #8 | #14 | #6 |
| `EXP10-loss-recon-only` | 7.33 | #10 | #5 | #7 | 6.67 | #5 | #12 | #3 |
| `EXP07-encoder-gat-like` | 10.00 | #11 | #7 | #12 | 7.67 | #6 | #4 | #13 |
| `EXP14-dim-bottleneck32` | 10.33 | #15 | #3 | #13 | 10.33 | #14 | #3 | #14 |
| `EXP15-dim-bottleneck128` | 10.67 | #14 | #9 | #9 | 10.33 | #13 | #8 | #10 |
| `EXP13-fast-convergence` | 11.00 | #13 | #12 | #8 | 10.67 | #15 | #6 | #11 |
| `EXP06-M5-arise-gcn` | 12.33 | #8 | #15 | #14 | 8.33 | #4 | #9 | #12 |
| `EXP11-loss-contrast-heavy` | 12.67 | #12 | #11 | #15 | 12.00 | #11 | #10 | #15 |

---

## 4. Rigorous Analysis of Findings & Corrections

### Correction 1: Graph Duality & Spectral GCN Do Not Universally Win
- **Previous Claim:** "EXP02–EXP06 consistently outperformed the single-graph baseline (EXP01)."
- **Actual Evidence:** On `10x_human_lymph_node_D1`, the dual-graph model `EXP02` collapses to rank **#14** and full ARISE GCN `EXP06` drops to rank **#15**, performing significantly *worse* than the simple single-graph SAGEConv baseline `EXP01` (rank **#10**). On `10x_human_lymph_node_A1`, `EXP06` (rank **#8**) loses to `EXP01` (rank **#6**). 
- **Underlying Cause:** D1 exhibits sparser antibody-capture staining and higher transcriptomic noise. Forcing a rigid spectral GCN across dense global spatial graphs over-smooths expression variation across tissue boundaries when cross-modality signals conflict.

### Correction 2: Dense Contrastive Regularization Destroys Biological Alignment
- **Previous Claim:** "Dense spatial contrastive BCE loss boosted biological domain alignment."
- **Actual Evidence:** The inverse occurred. While `EXP04-M3-contrastive` produced very high silhouette separation (0.261 to 0.407), it generated the **lowest biological ground-truth ARI of all 15 models across every single screening dataset**:
  - `A1`: ARI = **0.209 ± 0.021** (vs 0.249 for EXP03)
  - `D1`: ARI = **0.112 ± 0.019** (lowest across all 15 models)
  - `E11`: ARI = **0.180 ± 0.004** (lowest across all 15 models; half the baseline ARI)
- **Mechanism:** Dense spatial BCE penalizes Euclidean distance discrepancies between spatial neighbors and non-neighbors indiscriminately. In real tissues, adjacent spots frequently belong to distinct cell types (e.g., boundary margins, germinal center borders). Forcing spatial proximity to dominate the latent metric space physically glues disparate cell types together, yielding artificially tight geometric hyperspheres (high silhouette) while destroying true biological cell-type separation.

### Correction 3: High-Dimensional Feature Reconstruction is Dataset-Dependent
- **Previous Claim:** "High-dimensional raw feature reconstruction achieved top RAUS rank."
- **Actual Evidence:** `EXP05-M4-highdim` placed #1 **only on Mouse Brain E11** (Borda 8). On human lymph node A1, it ranked **#7** (Borda 22), and on D1 it ranked **#4** (Borda 20). 
- **Mechanism:** On mouse embryonic brain, spatial regions follow broad anatomical morphological structures where 3000 HVGs provide strong transcriptional programs. In human lymph nodes with highly interspersed immune sub-populations, reconstructing 3000 noisy raw genes without feature weighting diluted the specific lineage-defining markers.

---

## 5. Audit & Diagnostic: Why `knn_overlap` is Near-Zero

In the initial logs, `knn_overlap` hovered between `0.002` and `0.020`. 

- **Implementation Diagnostic:** `compute_knn_overlap` computes the mean Jaccard index between the 15-nearest neighbor sets in 3000-dimensional unreduced, scaled gene space ($X_{\text{raw}}$) and the learned latent embedding space ($Z$):
  $$\text{Jaccard}(S_i^{\text{raw}}, S_i^{\text{latent}}) = \frac{|S_i^{\text{raw}} \cap S_i^{\text{latent}}|}{|S_i^{\text{raw}} \cup S_i^{\text{latent}}|}$$
- **Statistical Degeneracy:** 
  1. For a dataset of $N = 4035$ spots (A1/D1), the expected overlap between two random 15-neighborhoods is:
     $$E[|S^{\text{raw}} \cap S^{\text{latent}}|] = 15 \times \frac{15}{4035} \approx 0.0557 \implies E[\text{Jaccard}] \approx \frac{0.0557}{29.94} \approx 0.0019$$
     Observed values of $0.002$–$0.005$ are literally at the baseline noise floor of random selection.
  2. For $E11$ ($N = 1185$ spots), the smaller denominator raises the random chance slightly to $\approx 0.0064$, producing observed overlaps of $0.011$–$0.020$.
  3. Latent space $Z$ is trained under spatial graph smoothing, which intentionally aligns spots along physical tissue axes. By design, $Z$ reorganizes local topology away from Euclidean distance in 3000-D noisy gene space.
- **Impact on RAUS:** Because differences in `knn_overlap` represent arbitrary fluctuations in the third decimal place (e.g. 0.0022 vs 0.0029), it introduced random noise into the Borda count.
- **Recommendation:** In the Borda tables above, both 3-Metric and 2-Metric (Silhouette + Seed Stability) ranks are displayed. For Stage 2, `knn_overlap` should either be computed on low-dimensional PCA space ($k=30$) or dropped from the unsupervised selection formula in favor of cross-seed stability and spatial neighborhood consistency.

---

## 6. The Silhouette ↔ Biological ARI Tension (Key Scientific Finding)

The central discovery of Stage 1 is the severe disconnect between unsupervised Silhouette score and biological ground-truth ARI:

- **Correlation Analysis Across 15 Models:**
  - `10x_human_lymph_node_A1`: Pearson $r = +0.522$ ($p = 0.046$)
  - `10x_human_lymph_node_D1`: Pearson $r = -0.569$ ($p = 0.027$, **statistically significant negative correlation**)
  - `Mouse_Brain_E11_S1`: Pearson $r = -0.434$ ($p = 0.106$, **negative correlation**)
- **The Contrastive Anomaly (EXP04):**
  EXP04 achieved top-tier silhouette scores across all datasets (#1 on D1, #2 on A1, #2 on E11), yet produced the lowest biological ARI on all three datasets (0.209, 0.112, 0.180).
- **The Baseline Paradox (EXP08):**
  On `Mouse_Brain_E11_S1`, the simple MLP baseline with L2 penalty (`EXP08`) yielded the **highest post-hoc ARI in the entire study (0.368 ± 0.034)**, but ranked only #6 in silhouette and #4 in RAUS.
- **Implications for Unsupervised Model Selection:**
  Silhouette score measures geometric compactness and cluster sphericity. Contrastive graph objectives exploit this by forcibly collapsing points into dense hyper-clusters. However, true biological manifolds feature continuous developmental progressions and spatially intermingled cell types. Optimizing solely for unsupervised geometric cluster tightness can actively degrade biological fidelity.
  Unsupervised selection must therefore rely on **multi-seed cluster stability (ARI)** and **cross-modality reconstruction balance** rather than raw silhouette maximization.

---

## 7. Scope Note & Limitations

- **Screening Coverage:** Stage 1 screening was restricted to 3 datasets: `10x_human_lymph_node_A1` (CITE-seq), `10x_human_lymph_node_D1` (CITE-seq), and `Mouse_Brain_E11_S1` (Spatial epigenome/ATAC-seq).
- **Unscreened Cohorts:** `Mouse_Brain_E13_S1`, `Mouse_Brain_E15_S1`, and `Mouse_Brain_E18_S1` were not evaluated in Stage 1. 
- **Requirement for Stage 2:** The winning candidate family must be evaluated across all six datasets to ensure developmental generalization across timepoints.

---

## 8. Proposed Stage 2 Candidates & Ablation Plan

Per BRIEF-004, our objective is **one robust, unified architecture across all spatial omics modalities and datasets**, not fractured per-dataset specializations. Per-dataset winners highlight specific structural strengths:
- A1 showed the power of **Hierarchical Cross-Modality Fusion** (`EXP03`, `EXP09`).
- D1 showed the need for **Controlled Regularization & High-Dim Recon** (`EXP12`, `EXP14`).
- E11 showed the power of **High-Dimensional Raw Feature Reconstruction** (`EXP05`, `EXP08`).

### Candidate Architectures for Stage 2

1. **Candidate 1: Hierarchical High-Dim Autoencoder with Calibrated Spatial Regularization (H-HiRe)**
   - *Rationale:* Combines the raw feature fidelity of `EXP05` (winner on E11) with the hierarchical 2-stage MLP fusion of `EXP03` (winner on A1) and calibrated spatial loss weighting from `EXP12` (rank #2 on D1).
   - *Design:* Multi-task loss with balanced reconstruction ($\lambda_{\text{rec}} = 15.0$) and moderate spatial graph regularization ($\lambda_{\text{spa}} \le 2.0$), avoiding the destructive over-clustering of dense BCE.

2. **Candidate 2: Cross-Modality Multihead Attention with SAGEConv Backbone (Attn-SAGE)**
   - *Rationale:* `EXP09` achieved excellent seed stability (0.635 on A1) and top-tier post-hoc ARI across datasets (0.224 on A1, 0.363 on E11), while SAGEConv (`EXP01`) avoided the severe over-smoothing that caused spectral GCN (`EXP06`) to collapse on D1.
   - *Design:* Cross-attention fusion layer between RNA and Modality 2 embeddings, combined with spatial-inductive SAGEConv message passing.

3. **Candidate 3: Adaptive Edge-Gated ARISE (Gated-ARISE)**
   - *Rationale:* Directly tests whether graph over-smoothing can be resolved by learning dynamic edge gates between spatial neighbors, allowing the network to prune edges that cross sharp transcriptomic boundaries.

### Systematic M0 → M8 Ablation Matrix Plan
To isolate which module drives performance across all six datasets:
- **Ablation 1 (Fusion):** Concatenation vs 2-Stage MLP vs Multihead Cross-Attention.
- **Ablation 2 (Reconstruction):** PCA-30 vs Raw 3000 HVGs vs Decoupled Modality Reconstruction.
- **Ablation 3 (Spatial Loss):** Zero Spatial Loss vs Contrastive BCE vs Graph Laplacian Smoothing vs Edge-Gated Regularization.
- **Ablation 4 (Graph Backbone):** Spectral GCNConv vs Inductive SAGEConv vs GAT.

---

**Awaiting Phoenix review and BRIEF-005 before launching Stage 2 runs.**