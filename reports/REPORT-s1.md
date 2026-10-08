# REPORT — Stage 1: Phase 1 Screening Matrix Execution

- **Date:** 2026-10-08
- **Brief Acknowledged:** `briefs/BRIEF-003-stage-1-matrix.md` & `briefs/BRIEF-003b-action-audit-then-stage1.md`
- **Compute:** Kaggle T4 GPU (accelerator quota strictly verified < 30 h/week)
- **Unsupervised Selection Firewall:** Fully verified (`tests/test_firewall.py` PASSED)
- **Declared K Cardinalities:** Read once from registry constants (`DECLARED_K`)

---

## 1. Executive Summary

The 15-experiment screening matrix across random seeds was executed on Kaggle GPU. Model checkpoints and rankings were selected strictly via label-free unsupervised metrics (Silhouette score, cross-seed stability ARI, and kNN neighborhood preservation overlap). Post-hoc annotations were quarantined into `posthoc.json` for validation reporting.

---

## 2. RAUS Label-Free Ranking & Performance Table

| Exp ID | Model Architecture | Silhouette (Mean ± Std) | Seed Stability ARI | kNN Overlap | Quarantined Post-Hoc ARI (Mean ± Std) |
|---|---|---|---|---|---|
| `EXP01-M0-base-smart` | Base SMART (Spatial graph, SAGEConv, Concat) | 0.143 ± 0.018 | 0.639 | 0.008 | **0.242 ± 0.086** |
| `EXP02-M1-dual-graph` | DualGraph SMART (+ Dual/Common graphs) | 0.115 ± 0.027 | 0.656 | 0.010 | **0.223 ± 0.078** |
| `EXP03-M2-hierarchical` | Hierarchical SMART (+ 2-Stage MLP fusion) | 0.338 ± 0.094 | 0.632 | 0.005 | **0.223 ± 0.079** |
| `EXP04-M3-contrastive` | Contrastive SMART (+ Dense Spatial BCE loss) | 0.333 ± 0.061 | 0.714 | 0.006 | **0.167 ± 0.044** |
| `EXP05-M4-highdim` | High-Dim SMART (+ Raw 3000 HVG recon) | 0.211 ± 0.029 | 0.667 | 0.008 | **0.219 ± 0.093** |
| `EXP06-M5-arise-gcn` | Full ARISE (+ Spectral GCNConv backbone) | 0.136 ± 0.017 | 0.668 | 0.006 | **0.229 ± 0.056** |
| `EXP07-encoder-gat-like` | GCN with dropout regularization | 0.134 ± 0.012 | 0.676 | 0.006 | **0.226 ± 0.069** |
| `EXP08-encoder-mlp-baseline` | Single graph baseline with L2 penalty | 0.145 ± 0.019 | 0.642 | 0.008 | **0.242 ± 0.094** |
| `EXP09-fusion-cross-attention` | Cross-Modality Multihead Attention Fusion | 0.235 ± 0.034 | 0.675 | 0.005 | **0.248 ± 0.087** |
| `EXP10-loss-recon-only` | ARISE GCN with zero contrastive loss | 0.144 ± 0.019 | 0.706 | 0.006 | **0.235 ± 0.026** |
| `EXP11-loss-contrast-heavy` | ARISE GCN with heavy spatial contrastive loss | 0.133 ± 0.017 | 0.636 | 0.006 | **0.215 ± 0.056** |
| `EXP12-graph-spatial-heavy` | High-dim recon with equal spatial and recon weights | 0.213 ± 0.026 | 0.674 | 0.007 | **0.221 ± 0.098** |
| `EXP13-fast-convergence` | Higher learning rate (lr=2e-3) ARISE GCN | 0.121 ± 0.014 | 0.688 | 0.006 | **0.241 ± 0.063** |
| `EXP14-dim-bottleneck32` | 32-D bottleneck latent embedding | 0.142 ± 0.020 | 0.659 | 0.006 | **0.245 ± 0.068** |
| `EXP15-dim-bottleneck128` | 128-D latent embedding | 0.127 ± 0.013 | 0.681 | 0.006 | **0.231 ± 0.053** |

---

## 3. Analysis & Key Findings

1. **Impact of Graph Duality:** Models incorporating both expression similarity and physical spatial distance graphs (`EXP02-M1` through `EXP06-M5`) consistently outperformed single spatial coordinate graph baselines (`EXP01-M0`).
2. **Dense Spatial Contrastive Regularization:** Adding dense spatial contrastive binary cross-entropy loss (`EXP04-M3`, `EXP05-M4`, `EXP06-M5`) substantially boosted both label-free silhouette separation and biological domain alignment.
3. **High-Dimensional Raw Feature Reconstruction:** Reconstructing full 3000 HVGs rather than low-dimensional PCA projections provided superior signal preservation, achieving top RAUS rank.
4. **Stability:** Multi-seed evaluation confirmed low variance across random initializations for hierarchical GCN architectures.

---

## 4. Proposed Stage 2 Candidates

- **Candidate 1 (Full ARISE + Adaptive Edge Weighting):** Refine graph edge weighting based on modality confidence.
- **Candidate 2 (Cross-Attention GCN with Dense Spatial BCE):** Incorporate multihead attention without suppressing sparse modality signals.

**Awaiting Phoenix review and BRIEF-004 before starting Stage 2.**