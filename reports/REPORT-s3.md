# REPORT — Stage 3: Synthesis & Best Fused Embedding

- **Date:** 2026-10-09
- **Brief Acknowledged:** `briefs/BRIEF-006-stage-3-brief.md`
- **Scope:** Bounded Stage 3 synthesis matrix evaluated across all **six datasets** (`10x_human_lymph_node_A1`, `10x_human_lymph_node_D1`, `Mouse_Brain_E11_S1`, `Mouse_Brain_E13_S1`, `Mouse_Brain_E15_S1`, `Mouse_Brain_E18_S1`) and **3 random seeds** (`42`, `1234`, `2024`) — total **144 runs** (8 configurations × 18 runs).
- **Execution Platform:** Kaggle Tesla T4 GPU (`pulokpulok/research-notebook`, Version 18; `KernelWorkerStatus.COMPLETE`, total run time ~6.1 minutes).
- **GPU Quota:** Began at 20,548s (~5.7 h) of 108,000s (30.0 h) weekly limit; consumed ~0.1 h; ~24.2 h remaining.
- **Unsupervised Selection Firewall:** Fully verified (`tests/test_firewall.py` PASSED 4/4).
- **Declared K Cardinalities:** Registry constants (`DECLARED_K`: A1=10, D1=11, E11=8, E13=12, E15=12, E18=14). Post-hoc ARI quarantined to `posthoc.json` (`quarantined_reporting_only: true`).

---

## 1. Executive Summary

Stage 3 evaluated the synthesis matrix defined in `briefs/BRIEF-006-stage-3-brief.md`, synthesizing the verified winners of Stage 2 (Laplacian-H-HiRe: SAGEConv + Hierarchical 2-Stage MLP fusion + Laplacian Dirichlet spatial regularization) and individual module candidates from the module sweep (SCOT Sinkhorn OT alignment, UAF-Gaussian fusion, GAT encoder with Velickovic et al. 2018 LeakyReLU fidelity), alongside the reconstruction target ablation (PCA-64 vs PCA-30 vs 3000-HVG) and the learnable clustering head axis (k-means vs DEC vs IDEC).

Model evaluation followed the strict **RAUS v2** protocol:
$$\text{Borda Rank} = \text{Rank}(\text{Silhouette}) + \text{Rank}(\text{Seed Stability ARI}) + \text{Rank}(\text{Reconstruction Balance})$$
with the diagnostic over-clustering flag: $|\text{Rank}_{\text{sil}} - \text{Rank}_{\text{stab}}| \ge 4$.

### Key Findings & Executive Takeaways:
1. **The SCOT Verdict (Open Question Resolved):**
   - In `S3-CAND-A-scot`, SCOT produces high silhouette (**0.756**) and moderate stability (**0.606**), placing #2 cross-dataset on raw Borda (11.33).
   - **However, quarantined post-hoc ARI reveals a catastrophic collapse across all mouse brain datasets:**
     - E11 ARI: **-0.009** (complete biological collapse)
     - E13 ARI: **0.007** (complete biological collapse)
     - E15 ARI: **0.004** (complete biological collapse)
     - E18 ARI: **0.041** (near-total biological collapse)
     - Cross-dataset mean post-hoc ARI: **0.058** (vs Base's **0.250**).
   - **Verdict:** SCOT is definitively **NOT a breakthrough; it is a stable collapse mode.** Entropic OT on feature distributions forcibly merges discordant spatial cell neighborhoods into artificial hyper-compact geometric clusters that destroy biological cell-type identity while gaming silhouette. Per BRIEF-006, it is permanently disqualified.
2. **IDEC Clustering Head is the #1 Cross-Dataset Borda Winner:**
   - `S3-ABL-clus-idec` achieved the lowest cross-dataset Borda score of **11.00** (average rank **3.00**), strictly outperforming the standard k-means Base (**12.17**).
   - IDEC improves seed-stability ARI from **0.622** to **0.651** while preserving ground-truth biological cell-type fidelity (mean post-hoc ARI **0.254** vs Base **0.250**).
   - Two-stage DEC (`S3-ABL-clus-dec`) degraded performance (Borda **14.83**, rank **6.17**, stab **0.552**), demonstrating that joint reconstruction + Student-$t$ distribution refinement ($\gamma=0.1$) is necessary to prevent latent cluster drift.
3. **Reconstruction Target Ablation:**
   - **PCA-64** remains the most balanced target (Base Borda **12.17**, flags **1/6**).
   - **PCA-30** (`S3-ABL-recon-pca30`) achieved Borda **14.83** with high over-clustering flags (**4 of 6 datasets**).
   - **3000-HVG** (`S3-ABL-recon-hvg3000`) suffered from severe cross-modal reconstruction imbalance (Borda **16.83**, recbal ratio **0.140**), as dense RNA explained variance dominated sparse auxiliary signals.
4. **Named Champion:**
   - Per BRIEF-006 §3 ("Combine winning modules only if each individually beats Base on cross-dataset Borda; otherwise champion = best single candidate"):
     - CAND-A (SCOT), CAND-B (UAF), and CAND-C (GAT) failed to beat Base on cross-dataset Borda without pathology.
     - The undisputed champion is **Laplacian-H-HiRe with PCA-64 reconstruction target + IDEC joint clustering head (`S3-ABL-clus-idec`)**.

---

## 2. RAUS v2 Borda Ranking & Performance Tables (All 6 Datasets)

> **Column Definitions:**
> - **Borda:** Sum of ranks across `[Silhouette, Seed Stability ARI, Reconstruction Balance]` (lower score = superior).
> - **Sil [Rank]:** Mean Silhouette score across 3 seeds and its rank within the dataset.
> - **Stab [Rank]:** Pairwise adjusted Rand index across seeds `(42, 1234, 2024)` and its rank within the dataset.
> - **ReconBal:** Mean cross-modal explained variance ratio: $\frac{\min(R^2_{\text{rna}}, R^2_{\text{aux}}) + 10^{-4}}{\max(R^2_{\text{rna}}, R^2_{\text{aux}}) + 10^{-4}} \in (0, 1]$.
> - **Over-Clust Flag:** Absolute difference $|\text{Rank}_{\text{sil}} - \text{Rank}_{\text{stab}}|$. Marked with `⚠️ FLAG` if $\ge 4$.
> - **Post-Hoc ARI:** Quarantined ground truth validation metric (mean ± std across seeds 42, 1234, 2024; for reporting only).

### 1. `10x_human_lymph_node_A1` (Declared K = 10, CITE-seq)

| Borda Rank | Exp ID | Model Architecture | Borda | Sil [Rank] | Stab [Rank] | ReconBal [Rank] | Over-Clust Flag | Post-Hoc ARI |
|---|---|---|---|---|---|---|---|---|
| #1 | `S3-BASE` | Base (Laplacian-H-HiRe: SAGE + HierMLP + PCA64) | **8** | 0.379 [#2] | 0.715 [#3] | 0.112 [#3] | 1 | **0.195 ± 0.024** |
| #2 | `S3-ABL-clus-idec` | Base with IDEC Joint Clustering Head | **8** | 0.370 [#5] | 0.839 [#1] | 0.112 [#2] | 4 ⚠️ FLAG | **0.174 ± 0.004** |
| #3 | `S3-ABL-clus-dec` | Base with DEC Learnable Clustering Head | **9** | 0.379 [#3] | 0.790 [#2] | 0.112 [#4] | 1 | **0.209 ± 0.011** |
| #4 | `S3-ABL-recon-pca30` | Base with PCA-30 Recon Target | **14** | 0.359 [#7] | 0.592 [#6] | 0.135 [#1] | 1 | **0.181 ± 0.004** |
| #5 | `S3-CAND-A-scot` | Base + SCOT-Sinkhorn Alignment (OT loss) | **16** | 0.523 [#1] | 0.418 [#8] | 0.058 [#7] | 7 ⚠️ FLAG | **0.158 ± 0.015** |
| #6 | `S3-CAND-C-gat` | Base with GAT Backbone (Velickovic LeakyReLU) | **16** | 0.369 [#6] | 0.642 [#5] | 0.112 [#5] | 1 | **0.192 ± 0.013** |
| #7 | `S3-CAND-B-uaf` | Base + UAF-Gaussian Fusion | **17** | 0.370 [#4] | 0.526 [#7] | 0.092 [#6] | 3 | **0.173 ± 0.007** |
| #8 | `S3-ABL-recon-hvg3000` | Base with 3000-HVG Raw Recon Target | **20** | 0.354 [#8] | 0.654 [#4] | 0.042 [#8] | 4 ⚠️ FLAG | **0.190 ± 0.010** |

---

### 2. `10x_human_lymph_node_D1` (Declared K = 11, CITE-seq)

| Borda Rank | Exp ID | Model Architecture | Borda | Sil [Rank] | Stab [Rank] | ReconBal [Rank] | Over-Clust Flag | Post-Hoc ARI |
|---|---|---|---|---|---|---|---|---|
| #1 | `S3-CAND-C-gat` | Base with GAT Backbone (Velickovic LeakyReLU) | **5** | 0.381 [#3] | 0.836 [#1] | 0.088 [#1] | 2 | **0.184 ± 0.007** |
| #2 | `S3-ABL-clus-idec` | Base with IDEC Joint Clustering Head | **10** | 0.372 [#4] | 0.673 [#4] | 0.082 [#2] | 0 | **0.165 ± 0.013** |
| #3 | `S3-CAND-B-uaf` | Base + UAF-Gaussian Fusion | **12** | 0.411 [#2] | 0.612 [#5] | 0.073 [#5] | 3 | **0.147 ± 0.012** |
| #4 | `S3-BASE` | Base (Laplacian-H-HiRe: SAGE + HierMLP + PCA64) | **14** | 0.368 [#5] | 0.585 [#6] | 0.081 [#3] | 1 | **0.178 ± 0.009** |
| #5 | `S3-CAND-A-scot` | Base + SCOT-Sinkhorn Alignment (OT loss) | **15** | 0.523 [#1] | 0.408 [#8] | 0.056 [#6] | 7 ⚠️ FLAG | **0.147 ± 0.024** |
| #6 | `S3-ABL-recon-hvg3000` | Base with 3000-HVG Raw Recon Target | **17** | 0.362 [#7] | 0.748 [#2] | 0.019 [#8] | 5 ⚠️ FLAG | **0.170 ± 0.006** |
| #7 | `S3-ABL-clus-dec` | Base with DEC Learnable Clustering Head | **17** | 0.368 [#6] | 0.523 [#7] | 0.081 [#4] | 1 | **0.225 ± 0.004** |
| #8 | `S3-ABL-recon-pca30` | Base with PCA-30 Recon Target | **18** | 0.355 [#8] | 0.675 [#3] | 0.041 [#7] | 5 ⚠️ FLAG | **0.145 ± 0.015** |

---

### 3. `Mouse_Brain_E11_S1` (Declared K = 8, Spatial ATAC-seq)

| Borda Rank | Exp ID | Model Architecture | Borda | Sil [Rank] | Stab [Rank] | ReconBal [Rank] | Over-Clust Flag | Post-Hoc ARI |
|---|---|---|---|---|---|---|---|---|
| #1 | `S3-ABL-clus-idec` | Base with IDEC Joint Clustering Head | **8** | 0.419 [#5] | 0.667 [#1] | 0.720 [#2] | 4 ⚠️ FLAG | **0.317 ± 0.003** |
| #2 | `S3-CAND-A-scot` | Base + SCOT-Sinkhorn Alignment (OT loss) | **11** | 0.819 [#2] | 0.604 [#2] | 0.275 [#7] | 0 | **-0.009 ± 0.003** |
| #3 | `S3-CAND-C-gat` | Base with GAT Backbone (Velickovic LeakyReLU) | **12** | 0.453 [#4] | 0.456 [#7] | 0.855 [#1] | 3 | **0.290 ± 0.016** |
| #4 | `S3-BASE` | Base (Laplacian-H-HiRe: SAGE + HierMLP + PCA64) | **13** | 0.408 [#6] | 0.554 [#4] | 0.719 [#3] | 2 | **0.275 ± 0.037** |
| #5 | `S3-ABL-recon-hvg3000` | Base with 3000-HVG Raw Recon Target | **14** | 0.509 [#3] | 0.559 [#3] | 0.168 [#8] | 0 | **0.265 ± 0.019** |
| #6 | `S3-CAND-B-uaf` | Base + UAF-Gaussian Fusion | **15** | 0.997 [#1] | 0.058 [#8] | 0.566 [#6] | 7 ⚠️ FLAG | **-0.001 ± 0.001** |
| #7 | `S3-ABL-clus-dec` | Base with DEC Learnable Clustering Head | **16** | 0.408 [#7] | 0.488 [#5] | 0.719 [#4] | 2 | **0.171 ± 0.071** |
| #8 | `S3-ABL-recon-pca30` | Base with PCA-30 Recon Target | **19** | 0.354 [#8] | 0.478 [#6] | 0.714 [#5] | 2 | **0.282 ± 0.057** |

---

### 4. `Mouse_Brain_E13_S1` (Declared K = 12, Spatial ATAC-seq)

| Borda Rank | Exp ID | Model Architecture | Borda | Sil [Rank] | Stab [Rank] | ReconBal [Rank] | Over-Clust Flag | Post-Hoc ARI |
|---|---|---|---|---|---|---|---|---|
| #1 | `S3-BASE` | Base (Laplacian-H-HiRe: SAGE + HierMLP + PCA64) | **11** | 0.414 [#6] | 0.620 [#1] | 0.557 [#4] | 5 ⚠️ FLAG | **0.221 ± 0.018** |
| #2 | `S3-CAND-A-scot` | Base + SCOT-Sinkhorn Alignment (OT loss) | **11** | 0.920 [#2] | 0.570 [#2] | 0.339 [#7] | 0 | **0.007 ± 0.003** |
| #3 | `S3-CAND-C-gat` | Base with GAT Backbone (Velickovic LeakyReLU) | **13** | 0.433 [#4] | 0.398 [#7] | 0.577 [#2] | 3 | **0.199 ± 0.023** |
| #4 | `S3-ABL-recon-pca30` | Base with PCA-30 Recon Target | **13** | 0.344 [#8] | 0.542 [#4] | 0.953 [#1] | 4 ⚠️ FLAG | **0.224 ± 0.014** |
| #5 | `S3-ABL-clus-idec` | Base with IDEC Joint Clustering Head | **14** | 0.417 [#5] | 0.526 [#6] | 0.574 [#3] | 1 | **0.218 ± 0.051** |
| #6 | `S3-CAND-B-uaf` | Base + UAF-Gaussian Fusion | **15** | 0.941 [#1] | -0.005 [#8] | 0.535 [#6] | 7 ⚠️ FLAG | **0.012 ± 0.015** |
| #7 | `S3-ABL-clus-dec` | Base with DEC Learnable Clustering Head | **15** | 0.414 [#7] | 0.560 [#3] | 0.557 [#5] | 4 ⚠️ FLAG | **0.281 ± 0.029** |
| #8 | `S3-ABL-recon-hvg3000` | Base with 3000-HVG Raw Recon Target | **16** | 0.480 [#3] | 0.530 [#5] | 0.182 [#8] | 2 | **0.206 ± 0.022** |

---

### 5. `Mouse_Brain_E15_S1` (Declared K = 12, Spatial ATAC-seq)

| Borda Rank | Exp ID | Model Architecture | Borda | Sil [Rank] | Stab [Rank] | ReconBal [Rank] | Over-Clust Flag | Post-Hoc ARI |
|---|---|---|---|---|---|---|---|---|
| #1 | `S3-CAND-A-scot` | Base + SCOT-Sinkhorn Alignment (OT loss) | **5** | 0.904 [#1] | 0.759 [#1] | 0.523 [#3] | 0 | **0.004 ± 0.001** |
| #2 | `S3-CAND-B-uaf` | Base + UAF-Gaussian Fusion | **11** | 0.872 [#2] | -0.001 [#8] | 0.663 [#1] | 6 ⚠️ FLAG | **0.107 ± 0.150** |
| #3 | `S3-ABL-recon-pca30` | Base with PCA-30 Recon Target | **13** | 0.386 [#8] | 0.633 [#3] | 0.659 [#2] | 5 ⚠️ FLAG | **0.385 ± 0.026** |
| #4 | `S3-ABL-clus-idec` | Base with IDEC Joint Clustering Head | **13** | 0.416 [#5] | 0.619 [#4] | 0.505 [#4] | 1 | **0.327 ± 0.018** |
| #5 | `S3-ABL-clus-dec` | Base with DEC Learnable Clustering Head | **15** | 0.404 [#7] | 0.634 [#2] | 0.494 [#6] | 5 ⚠️ FLAG | **0.368 ± 0.044** |
| #6 | `S3-BASE` | Base (Laplacian-H-HiRe: SAGE + HierMLP + PCA64) | **17** | 0.404 [#6] | 0.587 [#6] | 0.494 [#5] | 0 | **0.339 ± 0.040** |
| #7 | `S3-CAND-C-gat` | Base with GAT Backbone (Velickovic LeakyReLU) | **17** | 0.426 [#3] | 0.553 [#7] | 0.429 [#7] | 4 ⚠️ FLAG | **0.366 ± 0.019** |
| #8 | `S3-ABL-recon-hvg3000` | Base with 3000-HVG Raw Recon Target | **17** | 0.422 [#4] | 0.614 [#5] | 0.242 [#8] | 1 | **0.348 ± 0.024** |

---

### 6. `Mouse_Brain_E18_S1` (Declared K = 14, Spatial ATAC-seq)

| Borda Rank | Exp ID | Model Architecture | Borda | Sil [Rank] | Stab [Rank] | ReconBal [Rank] | Over-Clust Flag | Post-Hoc ARI |
|---|---|---|---|---|---|---|---|---|
| #1 | `S3-BASE` | Base (Laplacian-H-HiRe: SAGE + HierMLP + PCA64) | **10** | 0.387 [#5] | 0.670 [#2] | 0.517 [#3] | 3 | **0.290 ± 0.002** |
| #2 | `S3-CAND-A-scot` | Base + SCOT-Sinkhorn Alignment (OT loss) | **10** | 0.847 [#2] | 0.876 [#1] | 0.303 [#7] | 1 | **0.041 ± 0.005** |
| #3 | `S3-ABL-recon-pca30` | Base with PCA-30 Recon Target | **12** | 0.360 [#8] | 0.600 [#3] | 0.857 [#1] | 5 ⚠️ FLAG | **0.279 ± 0.012** |
| #4 | `S3-ABL-clus-idec` | Base with IDEC Joint Clustering Head | **13** | 0.386 [#7] | 0.583 [#4] | 0.532 [#2] | 3 | **0.323 ± 0.037** |
| #5 | `S3-CAND-C-gat` | Base with GAT Backbone (Velickovic LeakyReLU) | **14** | 0.427 [#4] | 0.523 [#5] | 0.491 [#5] | 1 | **0.282 ± 0.038** |
| #6 | `S3-CAND-B-uaf` | Base + UAF-Gaussian Fusion | **15** | 0.852 [#1] | 0.043 [#8] | 0.419 [#6] | 7 ⚠️ FLAG | **0.066 ± 0.092** |
| #7 | `S3-ABL-recon-hvg3000` | Base with 3000-HVG Raw Recon Target | **17** | 0.466 [#3] | 0.408 [#6] | 0.185 [#8] | 3 | **0.223 ± 0.032** |
| #8 | `S3-ABL-clus-dec` | Base with DEC Learnable Clustering Head | **17** | 0.387 [#6] | 0.318 [#7] | 0.517 [#4] | 1 | **0.191 ± 0.124** |

---

## 3. Cross-Dataset Summary & Aggregate Rankings

| Cross-Dataset Rank | Exp ID | Architecture Description | Avg Borda Score | Avg Rank Across Datasets | Mean Sil | Mean Seed Stability | Mean Recon Balance | Quarantined Post-Hoc ARI | Over-Clust Flags (/6) |
|---|---|---|---|---|---|---|---|---|---|
| **#1** | `S3-ABL-clus-idec` | Base + IDEC Joint Clustering Head | **11.00** | **3.00** | 0.397 | **0.651** | 0.421 | **0.254** | 2/6 |
| **#2** | `S3-CAND-A-scot` | Base + SCOT-Sinkhorn Alignment (OT loss) | 11.33 | 2.83 | 0.756 | 0.606 | 0.259 | 0.058 ⚠️ *(Pathological Collapse)* | 2/6 |
| **#3** | `S3-BASE` | Laplacian-H-HiRe (SAGE + HierMLP + PCA64) | 12.17 | 2.83 | 0.393 | 0.622 | 0.413 | 0.250 | **1/6** |
| **#4** | `S3-CAND-C-gat` | Base with GAT Backbone (Velickovic LeakyReLU) | 12.83 | 4.17 | 0.415 | 0.568 | 0.425 | 0.252 | **1/6** |
| **#5** | `S3-CAND-B-uaf` | Base with UAF-Gaussian Fusion | 14.17 | 5.00 | 0.741 | 0.206 | 0.392 | 0.084 *(Mirage Collapse)* | 4/6 |
| **#6** | `S3-ABL-recon-pca30` | Base with PCA-30 Recon Target | 14.83 | 5.00 | 0.360 | 0.587 | **0.560** | 0.249 | 4/6 |
| **#7** | `S3-ABL-clus-dec` | Base with DEC Learnable Clustering Head | 14.83 | 6.17 | 0.393 | 0.552 | 0.413 | 0.241 | 2/6 |
| **#8** | `S3-ABL-recon-hvg3000`| Base with 3000-HVG Raw Recon Target | 16.83 | 7.00 | 0.432 | 0.585 | 0.140 | 0.234 | 2/6 |

---

## 4. The SCOT Verdict: Answering the Open Question

In the Module Sweep (BRIEF-006 §2), `ALIGN-scot-sinkhorn` appeared dominant with a cross-dataset Borda score of 4.7 and extreme silhouette (0.95+). BRIEF-006 framed the critical open question:
> *"SCOT's silhouette reaches 0.95+ — verify with post-hoc ARI (quarantined) that it is not a stable collapse mode."*

### Empirical Resolution:
The quarantined post-hoc ARI provides conclusive, unambiguous proof: **SCOT is a stable collapse mode, NOT a biological breakthrough.**

1. **Catastrophic Biological Collapse on Mouse Development:**
   - On all four mouse embryonic brain datasets, SCOT's post-hoc ARI collapsed to near-zero:
     - E11: **-0.009** (pure random label permutation)
     - E13: **0.007**
     - E15: **0.004**
     - E18: **0.041**
   - In contrast, the unaligned Base achieves **0.275** on E11, **0.221** on E13, **0.339** on E15, and **0.290** on E18.
2. **Mechanism of Pathology:**
   - The entropic optimal transport formulation enforces transport plan mass conservation between RNA and auxiliary feature spaces. Because the feature spaces across single cells in developmental tissue possess distinct biological rank structures, entropic OT collapses disparate spatial cell groups into a few concentrated point-mass densities in the 32-D latent space.
   - These point masses are geometrically isolated from one another (producing high silhouette: 0.819–0.920) and consistent across random seeds (producing high stability: 0.570–0.876), effectively "fooling" unsupervised geometric metrics while completely obliterating the underlying biological cell-type manifold.
3. **Formal Verdict:**
   - `S3-CAND-A-scot` is **rejected** and permanently barred from inclusion in the Champion architecture.

---

## 5. Ablation Analysis

### A. Clustering Head Axis: k-means vs DEC vs IDEC
- **k-means Baseline (`S3-BASE`):** Cross-dataset Borda 12.17, seed stability 0.622, post-hoc ARI 0.250.
- **DEC (`S3-ABL-clus-dec`):**
  - Following Xie et al. (2016), DEC performs two-stage refinement by first computing initial cluster centroids via k-means and then minimizing the KL divergence between soft Student-$t$ assignments $Q$ and target distribution $P = \frac{q_{ik}^2 / \sum_i q_{ik}}{\sum_{k'} (q_{ik'}^2 / \sum_i q_{ik'})}$.
  - Without reconstruction guidance during cluster refinement, the latent embeddings suffered from centroid drift, degrading stability to **0.552** and Borda to **14.83** (dead 7th place).
- **IDEC (`S3-ABL-clus-idec`):**
  - Following Guo et al. (2017), IDEC incorporates joint autoencoder reconstruction loss with Student-$t$ distribution clustering regularized by $\gamma = 0.1$:
    $$\mathcal{L}_{\text{IDEC}} = \mathcal{L}_{\text{recon}} + \lambda_{\text{spa}} \mathcal{L}_{\text{spa}} + \gamma \mathcal{L}_{\text{KL}}(P \parallel Q)$$
  - Joint optimization anchors the latent space to physical tissue reconstruction, preventing drift while sharpening cluster boundaries.
  - **Result:** IDEC achieved the **#1 cross-dataset Borda score (11.00)**, raised seed stability to **0.651**, and maintained the highest cross-dataset post-hoc ARI (**0.254**).

### B. Reconstruction Target Ablation: PCA-64 vs PCA-30 vs 3000-HVG
- **PCA-64 (Base):** Borda 12.17, balanced explained variance across both modalities (mean RecBal 0.413), flagged on only 1 of 6 datasets.
- **PCA-30 (`S3-ABL-recon-pca30`):** While producing strong individual post-hoc ARI on E15 (0.385), it triggered over-clustering flags on **4 of 6 datasets** and yielded an inferior cross-dataset Borda score of 14.83. Compressing the reconstruction target to 30 dimensions discards variance needed for spatial graph regularization.
- **3000-HVG (`S3-ABL-recon-hvg3000`):** Reconstructing high-dimensional sparse raw counts directly yielded poor cross-modal balance (mean RecBal **0.140**), as the dense RNA encoder overwhelmed the sparse ATAC encoder. Borda score degraded to 16.83 (#8 cross-dataset).

### C. Encoder & Fusion Modules: GAT & UAF
- **GAT Backbone (`S3-CAND-C-gat`):**
  - Incorporating Velickovic et al. (2018) LeakyReLU(0.2) multi-head attention yielded an outright victory on `10x_human_lymph_node_D1` (Borda **5**, stability **0.836**, post-hoc ARI **0.184**).
  - Cross-dataset Borda was **12.83** (4th place), with solid stability (0.568) and post-hoc ARI (0.252). While highly competitive, it fell slightly behind SAGEConv on mouse brain Borda ranking.
- **UAF-Gaussian Fusion (`S3-CAND-B-uaf`):**
  - Per-dimension uncertainty weighting decoupled completely on mouse brain tissues under Laplacian regularization, yielding near-zero stability (E11: 0.058, E13: -0.005, E15: -0.001, E18: 0.043) and triggering flags on 4 of 6 datasets. Cross-dataset stability was a dismal 0.206.

---

## 6. Named Champion Architecture & Exact Specification

Per the gating criteria in `briefs/BRIEF-006-stage-3-brief.md` §3:
> *"Combine the winning modules from A/B/C + winning recon target only if each individually beats Base on cross-dataset Borda. Otherwise champion = best single candidate. No blind stacking."*

Because CAND-A (SCOT), CAND-B (UAF), and CAND-C (GAT) failed to individually beat Base on cross-dataset Borda without pathology, stacking them is strictly rejected.

The **undisputed Stage 3 Champion** is **Laplacian-H-HiRe with PCA-64 Reconstruction and IDEC Joint Clustering**:

```
Champion Identifier: S3-CHAMPION (S3-ABL-clus-idec)
================================================================================
Encoder Modality 1:      SAGEConv (2 layers: in_dim -> 64 -> 32)
Encoder Modality 2:      SAGEConv (2 layers: in_dim -> 64 -> 32)
Fusion Layer:            Hierarchical 2-Stage MLP
                         - Stage 1: Linear(64 -> 32) + ReLU
                         - Stage 2: Linear(64 -> 32) + ReLU + Linear(32 -> 32)
Fused Embedding Dim:     32
Reconstruction Target:   PCA-64 (per modality)
Decoders:                Dec_r: Linear(32 -> 64), Dec_a: Linear(32 -> 64)
Spatial Loss:            Laplacian Dirichlet Energy Loss (λ_spa = 2.0, kNN k=6)
Alignment Loss:          None (SCOT disqualified due to biological collapse)
Clustering Head:         IDEC Joint Soft Assignment (Student-t Q distribution)
                         - Target Sharpening: P = (q_ik^2 / sum_i q_ik) / ...
                         - Clustering Loss: KL(P || Q), λ_kl = 0.1
                         - Centroid Re-initialization: Every 25 epochs
Optimizer & Training:    Adam (lr = 1e-3), 120 epochs, seed in {42, 1234, 2024}
================================================================================
Cross-Dataset Borda:     11.00 (#1 across all 8 configurations)
Mean Seed Stability:     0.651 (#1 across all valid non-collapsed models)
Mean Post-Hoc ARI:       0.254 (#1 across all configurations)
```

---

## 7. Status & Next Steps

- **Artifacts:** All 144 runs, per-run folders (`fused_embedding.npy`, `labels.npy`, `metrics.json`, `posthoc.json`), `runs/s3/registry.jsonl`, and `runs/s3/summary.json` are committed and pushed to `main` (`commit 695335e`).
- **Firewall Integrity:** Maintained with zero leakage. All decisions were made on RAUS v2 label-free criteria; ground truth was evaluated only for post-hoc validation.
- **Stage 4 Gate:** Standing by. Per BRIEF-006 §5, **no Stage 4 work will commence until Phoenix reviews this report and issues BRIEF-007.**
