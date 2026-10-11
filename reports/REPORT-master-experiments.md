# MASTER EXPERIMENT REPORT — All Stages (0–6)

**Repository:** `rifatahsanpul0k/my_research`
**Date:** 2026-10-11
**Author:** Phoenix (research reviewer)
**Status:** Living document — covers all completed experiments through Stage 6 selector test

---

## 1. PROJECT OVERVIEW

**Goal:** Build ONE fully unsupervised spatial multi-omics integration model that achieves the best fused embedding (measured by post-hoc ARI) on all six datasets simultaneously.

**The six datasets:**
| ID | Tissue | Modalities | Spots (approx) | Declared k |
|----|--------|-----------|----------------|------------|
| A1 | Human lymph node | RNA + ADT (protein) | ~3,500 | 10 |
| D1 | Human lymph node | RNA + ADT (protein) | ~3,400 | 11 |
| E11 | Mouse embryonic brain | RNA + ATAC | ~1,300 | 8 |
| E13 | Mouse embryonic brain | RNA + ATAC | ~1,800 | 12 |
| E15 | Mouse embryonic brain | RNA + ATAC | ~2,000 | 12 |
| E18 | Mouse embryonic brain | RNA + ATAC | ~2,200 | 14 |

**Key methodological constraints (non-negotiable):**
- **Unsupervised firewall** (`tests/test_firewall.py`): Ground-truth labels must NEVER appear in training, preprocessing, graph construction, modality weighting/gating, early stopping, or model selection. Labels are loaded ONLY for quarantined post-hoc ARI reporting.
- **Declared k exception:** The number of clusters (k) is read once from ground-truth label counts. This is the ONE declared exception, applied identically to every configuration, disclosed in all reports. It is never used for anything except setting k in the clustering head.
- **RAUS v3b** (official selection rule since Stage 4): Borda rank across {seed-stability ARI, reconstruction-balance, R²_rna}. Silhouette was REMOVED from the Borda after being proven to actively mislead (ρ = -0.27 to +0.06 vs quarantined ARI).

**Preprocessing (fixed across all stages):**
- RNA: SCTransform v2 → PCA 64 → scaling
- ADT: CLR (centered log-ratio) → scaling
- ATAC: TF-IDF → LSI 50 → drop components with |corr(LSI, log depth)| > 0.75 → scaling
- Spatial graph: kNN with k=15

---

## 2. REPOSITORY STRUCTURE GUIDE

### Top-level directories

| Path | Purpose |
|------|---------|
| `notebooks/` | Jupyter notebooks that run on Kaggle (GPU). Each is generated from a `scripts/gen_*.py` script — the script is the source of truth, the notebook is the build artifact. |
| `scripts/` | Generator scripts (`gen_*.py`) that build notebooks. Also utility scripts. **Always edit the generator, then regenerate the notebook — never hand-edit notebook JSON.** |
| `runs/` | All experiment outputs. Each stage has its own subdirectory. See §3 for details. |
| `reports/` | Stage reports (REPORT-s1.md through REPORT-s4.md, REPORT-stage0.md). This master report lives here too. |
| `briefs/` | BRIEF-*.md files — directives from Phoenix (reviewer) to the agent, defining each stage's scope, constraints, and success criteria. |
| `pipeline/` | Shared Python modules: `datasets.py` (data loading), `models.py` (model architectures), `training.py` (training loop). Imported by notebooks. |
| `tests/` | `test_firewall.py` — verifies the unsupervised firewall (no label leakage). Must pass before any stage is accepted. |

### `runs/` subdirectory details

| Path | Contents | # Runs |
|------|----------|--------|
| `runs/s1/` | Stage 1 screening matrix outputs | ~180 |
| `runs/s2/` | Stage 2 architecture search outputs | 180 |
| `runs/s3/` | Stage 3 synthesis matrix outputs | 144 |
| `runs/s3-notebook/` | Stage 3 notebook-specific outputs | — |
| `runs/s4/` | Stage 4 champion validation + λ_spa calibration | 72 |
| `runs/s5-notebook/` | Stage 5 generalization experiments. **13 combo subdirectories** (see below). Each has 6 dataset subdirs × 3 seed subdirs. Each seed dir contains `fused_embedding.npy` (the saved 32-D embedding), `metrics.json` (unsupervised metrics), `posthoc.json` (quarantined ARI). Plus `registry.jsonl` (234 rows, one per run). | 234 |
| `runs/s6-exp0/` | Experiment 0: recluster saved embeddings. `exp0_results.jsonl` (2,819 rows — one per embedding × transform × head × seed eval). `exp0_summary.json`. | 2,819 evals |
| `runs/s6-frank/` | S6-FRANK (Frankenstein model) training runs. `S6-FRANK/<dataset>/seed<seed>/` with `metrics.json` + `posthoc.json`. Plus `registry.jsonl` (18 rows). | 18 |
| `runs/s6-selector/` | Selector validation test. `selector_results.jsonl` (216 rows). 216 `*_labels.npy` files (saved cluster labels for stability computation). `selector_summary.json`. | 216 evals |

### `runs/s5-notebook/` combo directories (13 total)

Each combo is BASE architecture + one modification:

| Combo | Description |
|-------|-------------|
| `S5N-BASE` | Baseline: SAGE encoders + hierarchical MLP fusion + Laplacian (λ=0.1) + IDEC head. The control group. |
| `S5N-SP-sage` | BASE + shared-private split (16 shared + 8+8 private dims) with SAGE encoders |
| `S5N-SP-sage-adapt` | SP-sage + adaptive disentanglement (λ_dis from cross-modal kNN agreement) |
| `S5N-SP-sage-nodis` | SP-sage WITHOUT the disentanglement loss (ablation) |
| `S5N-adapt-modality` | BASE + Kendall & Gal uncertainty-based modality weighting |
| `S5N-adapt-spatial` | BASE + Moran's-I adaptive spatial regularization |
| `S5N-dis005` | BASE + fixed disentanglement λ_dis=0.05 |
| `S5N-dis04` | BASE + fixed disentanglement λ_dis=0.4 |
| `S5N-generalized` | BASE + all adaptations combined (modality + spatial + shared-private) |
| `S5N-nograph` | BASE WITHOUT the spatial graph (MLP encoders only) |
| `S5N-shared-private` | BASE + shared-private split with MLP encoders (no graph) |
| `S6-ONE` | Gated dual-path model (SAGE graph path + MLP feature path, per-spot sigmoid gate) |
| `S6-ONE-v2` | S6-ONE v2 with skeptical gate init (-1.0) + gate-coupled Laplacian |

### `notebooks/` file guide

| Notebook | Generator script | Purpose | Status |
|----------|-----------------|---------|--------|
| `stage5_generalization.ipynb` | `scripts/gen_stage5_nb.py` | Stage 5: 90 runs testing label-free adaptation mechanisms | Complete |
| `stage6_exp0_recluster.ipynb` | `scripts/gen_exp0_nb.py` | Exp0: recluster 234 saved embeddings × 12 (transform, head) combos | Complete (2,819 evals) |
| `stage6_frankenstein.ipynb` | `scripts/gen_stage6_nb.py` | S6-FRANK: train the Frankenstein model (18 runs) | Complete (18 runs) |
| `stage6_selector_test.ipynb` | `scripts/gen_selector_nb.py` | Selector test: validate unsupervised (transform, head) selection | Complete (216 evals) |
| `stage4_validation.ipynb` | — | Stage 4: champion validation + λ_spa calibration | Complete |
| `stage3_synthesis.ipynb` | `scripts/gen_stage3_nb.py` | Stage 3: synthesis matrix | Complete |
| `embedding_module_sweep.ipynb` | `scripts/gen_embedding_sweep_nb.py` | Module sweep: 26 unsupervised modules from recent papers | Complete |

### Key files within a run directory

For `runs/s5-notebook/<combo>/<dataset>/seed<seed>/`:
- `fused_embedding.npy` — the 32-dimensional fused embedding (n_spots × 32). This is the primary artifact.
- `metrics.json` — unsupervised metrics: silhouette, seed-stability ARI, reconstruction R² (RNA + aux), Moran's I, gate values (if applicable)
- `posthoc.json` — quarantined post-hoc ARI + NMI. Labels used ONLY here, never in training.

For `runs/s6-exp0/exp0_results.jsonl` (one JSON per line):
- Fields: `combo`, `dataset`, `seed`, `transform` (raw/standard/l2/pca24), `head` (kmeans/spherical/gmm), `silhouette`, `ari`, `n_clusters`

---

## 3. STAGE-BY-STAGE EXPERIMENT RESULTS

### Stage 0: Infrastructure
- **What:** Kaggle ↔ GitHub direct-push pipeline setup. Private Kaggle dataset upload. Smoke tests.
- **Key outcome:** Established the pattern: fine-grained PAT in Kaggle Secrets as `GITHUB_TOKEN`, notebook pushes results directly to GitHub after each run.
- **Security note:** Stage-0 PAT was exposed in plaintext in chat. Must be revoked (standing item).

### Stage 1: Screening Matrix
- **What:** Broad screening of architecture variants across all six datasets.
- **Runs:** ~180
- **Key findings:**
  - Silhouette negatively correlates with ARI on D1 (r = -0.569) and E11 (r = -0.434) — first evidence that silhouette misleads.
  - kNN-overlap diagnostic found to be degenerate noise (0.002–0.020 range).
  - Per-dataset winners differ: A1 → EXP03, D1 → EXP04, E11 → EXP05.
- **Report:** `reports/REPORT-s1.md`

### Stage 2: Architecture Search & Ablation
- **What:** 10-experiment matrix, 180 runs (10 configs × 6 datasets × 3 seeds).
- **Winner:** `S2-EXP09-ABL-spa-laplacian` (Graph Laplacian + Hierarchical MLP fusion). Placed #1 or #2 on 4/6 datasets, top-4 on all six. Average cross-dataset rank: 2.00.
- **Selection:** RAUS v2 (Borda{silhouette, seed-stability, reconstruction-balance}).
- **Report:** `reports/REPORT-s2.md`

### Stage 3: Synthesis
- **What:** 144 runs (8 configs × 18). Tested SCOT (optimal transport alignment), UAF-Gaussian fusion, GAT encoder, reconstruction target ablation (PCA-64 vs PCA-30 vs 3000-HVG), clustering head (k-means vs DEC vs IDEC).
- **Key finding — SCOT verdict:** SCOT achieved high silhouette (0.756) but **catastrophic ARI collapse** on mouse brain (E11: -0.009, E13: 0.007, E15: 0.004). Permanently disqualified. This was the clearest demonstration that silhouette can be gamed.
- **Report:** `reports/REPORT-s3.md`

### Stage 4: Champion Validation & Criterion Repair
- **What:** 72 runs. Three objectives: (1) calibrate λ_spa, (2) fix the selection criterion, (3) test uniformity anti-collapse.
- **λ_spa calibration:** Swept on E15 seed 42. Selected λ_spa = 0.20 (R²_rna = 0.2535, closest to target band center 0.225).
- **RAUS v3b adopted:** Proved silhouette actively misleads (ρ = -0.27 to +0.06). New rule: Borda{seed-stability ARI, reconstruction-balance, R²_rna} (ρ = 0.512/0.424 vs quarantined ARI).
- **Champion:** `S4-CHAMP-sage` (SAGE + HierMLP + PCA-30 + λ_spa=0.2 + IDEC): mean ARI 0.250 vs 0.241 (GAT) vs 0.232 (base).
- **Uniformity anti-collapse:** Shelved (0.229 < baseline 0.232).
- **Report:** `reports/REPORT-s4.md`

### Stage 5: Generalization (Label-Free Adaptation)
- **What:** 234 runs (13 combos × 6 datasets × 3 seeds). Tested whether label-free adaptation mechanisms could create one architecture good on all six datasets.
- **Key findings:**
  - **Tissue-type divide confirmed:** Human lymph node is BETTER without the spatial graph (A1: 0.240 nograph vs 0.205 with graph). Mouse brain is MUCH WORSE without it (E11: 0.065 nograph vs 0.294 with graph).
  - **Moran's I does NOT predict the divide:** E18 has Moran's I ≈ 0 but still needs the graph.
  - **Adaptive λ_dis mechanism FAILED:** Cross-modal kNN agreement was ~0.01 on ALL datasets (high-D kNN sets don't overlap), so λ_dis was effectively fixed at 0.2. The "adaptation" was an illusion.
  - **E11 collapse diagnosis:** Caused by shared-private architecture itself, not the missing graph.
  - **Best per-dataset (Stage 5, k-means clustering):** A1: 0.240 (nograph), D1: 0.241 (generalized), E11: 0.294 (BASE), E13: 0.396 (SP-sage), E15: 0.503 (BASE), E18: 0.325 (SP-sage)
- **Notebook:** `notebooks/stage5_generalization.ipynb`

### Stage 6 — Experiment 0: Recluster Saved Embeddings
- **What:** Zero-training re-evaluation. For each of 234 saved embeddings, tried 4 transforms (raw/standard/l2/pca24) × 3 clustering heads (kmeans/spherical/gmm) = 12 evals per embedding.
- **Runs:** 2,819 evals (78 pairs × 12 combos × 3 seeds, some pairs had missing seeds)
- **Results — best per dataset (any combo):**

| Dataset | Best ARI | Combo | Transform/Head | Previous Best | Gain |
|---------|----------|-------|----------------|---------------|------|
| A1 | **0.250** | S5N-adapt-modality | raw/gmm | 0.240 | +0.010 |
| D1 | **0.250** | S5N-nograph | raw/kmeans | 0.241 | +0.009 |
| E11 | **0.445** | S5N-SP-sage-nodis | pca24/spherical | 0.294 | **+0.151** |
| E13 | **0.436** | S5N-dis005 | pca24/kmeans | 0.396 | +0.040 |
| E15 | **0.506** | S5N-adapt-spatial | standard/kmeans | 0.503 | +0.003 |
| E18 | 0.323 | S5N-dis005 | pca24/kmeans | 0.325 | -0.002 |

- **Key patterns:**
  - `pca24` transform dominates mouse brain (E11, E13, E18 bests)
  - `gmm` head wins on human (A1)
  - `spherical` k-means critical for E11 (0.292 with kmeans → 0.445 with spherical)
  - `standard` transform wins on 4/6 datasets for the BASE combo
- **Best TRUE one-pipeline** (single combo + transform + head, fixed across all datasets): **S5N-BASE + standard/kmeans → mean ARI 0.320** (A1: 0.221, D1: 0.205, E11: 0.292, E13: 0.417, E15: 0.501, E18: 0.285)
- **Thesis:** Embeddings were better than reported — the information was there, just badly clustered. The tissue divide appears at clustering time too.
- **Notebook:** `notebooks/stage6_exp0_recluster.ipynb`
- **Results:** `runs/s6-exp0/exp0_results.jsonl`

### Stage 6 — S6-ONE and S6-ONE-v2 (Gated Dual-Path Models)
- **What:** Attempt to build ONE model with learned per-spot gates blending SAGE graph path + MLP feature path.
- **S6-ONE results (18 runs):** A1: 0.191, D1: 0.202, E11: -0.013, E13: 0.042, E15: 0.248, E18: 0.128
- **Diagnosis:** Gates learned BACKWARDS — favored graph path on human (where graph hurts), stayed neutral on mouse (where graph helps). Root cause: gate optimizes reconstruction loss, but graph smoothing helps reconstruction even when it hurts clustering. Wrong objective.
- **S6-ONE-v2 results (18 runs):** A1: 0.200, D1: 0.203, E11: 0.014, E13: 0.310, E15: 0.289, E18: 0.152
- **Diagnosis:** Gates COLLAPSED to ~0.05-0.11 everywhere (model ignores graph entirely). Skeptical init + gate-coupled Laplacian created a death spiral.
- **Verdict:** Per-spot sigmoid gating cannot learn graph-reliance from unsupervised losses. Approach abandoned.

### Stage 6 — S6-FRANK (Frankenstein Model)
- **What:** Combined stealable mechanisms from 11-paper literature review: per-spot MoE fusion (GROVER-style threshold gating) + shared-private 16+8+8 + masked reconstruction 15% (scMRDR) + isometric regularization + homogeneity-gated spatial (OmicSync-style) + gate entropy + spherical k-means head.
- **Runs:** 18 (1 combo × 6 datasets × 3 seeds)
- **Results (mean ARI):** A1: 0.188, D1: 0.150, E13: 0.125, E15: 0.216, E18: 0.170, E11: 0.149
- **Diagnosis:**
  1. MoE gate collapsed (A1: 97% aux / 3% RNA — backwards)
  2. Homogeneity gating set λ_spa=0.05 everywhere (thresholds were guessed, not measured)
  3. Highly unstable across seeds (E13: 0.062 to 0.196)
- **Verdict:** FAILED. Worse than baseline on every dataset. Paper ideas don't plug-and-play.
- **Notebook:** `notebooks/stage6_frankenstein.ipynb`
- **Results:** `runs/s6-frank/`

### Stage 6 — Selector Test
- **What:** Test whether unsupervised cross-seed stability can pick the best (transform, head) per dataset.
- **Runs:** 216 evals (1 combo × 6 datasets × 12 transform/head × 3 seeds)
- **Results:** Stability correct on 2/6 datasets. Silhouette correct on 3/6.
- **Verdict:** Neither metric reliably selects the clustering. Silhouette has a systematic bias toward kmeans (it's computed in Euclidean space, blind to hypersphere geometry where spherical k-means wins).
- **Notebook:** `notebooks/stage6_selector_test.ipynb`
- **Results:** `runs/s6-selector/`

---

## 4. CURRENT BEST SCORES

### Best per dataset (any combo, optimal clustering) — the ceiling
| Dataset | ARI | Source |
|---------|-----|--------|
| A1 | 0.250 | S5N-adapt-modality + raw/gmm (Exp0) |
| D1 | 0.250 | S5N-nograph + raw/kmeans (Exp0) |
| E11 | 0.445 | S5N-SP-sage-nodis + pca24/spherical (Exp0) |
| E13 | 0.436 | S5N-dis005 + pca24/kmeans (Exp0) |
| E15 | 0.506 | S5N-adapt-spatial + standard/kmeans (Exp0) |
| E18 | 0.325 | S5N-SP-sage + standard/kmeans (Stage 5) |

### Best TRUE one-pipeline (single fixed combo + transform + head)
**S5N-BASE + standard/kmeans → mean ARI 0.320**
| Dataset | ARI |
|---------|-----|
| A1 | 0.221 |
| D1 | 0.205 |
| E11 | 0.292 |
| E13 | 0.417 |
| E15 | 0.501 |
| E18 | 0.285 |

### Gap analysis
- Cherry-picked mean: 0.368 | One-pipeline mean: 0.320 | **Gap: 0.048**
- The gap is concentrated in E11 (0.445 → 0.292), which desperately needs spherical clustering.
- Human datasets (A1/D1) are stuck at ~0.25 in both settings — far from the 0.35 target.

---

## 5. DURABLE THESES (What We've Proven)

1. **Silhouette actively misleads under over-smoothing.** Proven in Stage 1 (negative correlation with ARI), Stage 3 (SCOT gamed it to 0.756 while ARI collapsed to -0.009), Stage 4 (ρ = -0.27 to +0.06, removed from RAUS), Exp0 (3/18 as selector), Selector test (3/6 as selector, biased toward kmeans).

2. **Seed-stability is necessary but not sufficient.** A model can be stable and wrong (SCOT was stable at 0.606 while ARI was -0.009).

3. **Reconstruction ≠ clustering quality.** A model can reconstruct well but cluster poorly, and vice versa.

4. **Tissue divide (brain needs graph, lymph node hurt by it) appears at training AND clustering time.** Proven in Stage 5 (nograph ablation) and Exp0 (different optimal transforms/heads per tissue).

5. **Per-spot gating cannot learn graph-reliance from unsupervised losses.** S6-ONE gates went backwards, S6-ONE-v2 gates collapsed, FRANK MoE gate collapsed. Three independent failures of the same idea.

6. **Embeddings are better than reported.** Exp0 gained +0.151 on E11 with zero retraining — the information was in the embedding, just badly clustered.

7. **Paper ideas don't plug-and-play.** FRANK combined 6 mechanisms from 11 papers and failed on every dataset. The mechanisms interact in ways papers don't describe.

---

## 6. OPEN QUESTIONS

1. **Human lymph node ceiling:** A1/D1 stuck at ~0.25. What architectural change would push toward 0.35? The graph hurts, shared-private helps slightly, but nothing breaks through.

2. **Unsupervised clustering selection:** Neither silhouette nor stability reliably picks (transform, head). Is there a better metric, or must this be hardcoded per dataset?

3. **The E11 spherical mystery:** Why does spherical k-means help E11 so dramatically (0.292 → 0.445) but not other datasets? Understanding this could generalize the insight.

4. **RAUS v3c:** Adding R²_aux to the Borda correlated even better (0.528/0.604) than adopted v3b in Stage 4. Should be revisited.

---

*End of master experiment report.*
