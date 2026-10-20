# RESEARCH HANDOFF — Complete Guide for the Next Agent

**Repository:** `rifatahsanpul0k/my_research`
**Owner:** Md Rifat Ahasan Pulock (PULOK), CSE student, UIU
**Written:** 2026-10-20 by Phoenix (outgoing research reviewer)
**Purpose:** Everything a new agent needs to take over this research without re-deriving two weeks of experiments. Read this whole file before touching anything.

Companion documents already in the repo:
- `reports/REPORT-master-experiments.md` — full experiment log, stages 0–6 (Oct 11)
- `reports/REPORT-codebase-understanding.md` — codebase walkthrough + open validity questions
- `reports/AGENT-REVIEW.md` — independent critique by the previous local agent (read its caveats in §10 below)
- `reports/REPORT-s1.md` … `REPORT-s4.md`, `REPORT-stage0.md` — per-stage reports
- `HANDOFF.md` — the GitHub-bridge collaboration protocol (how runs/results flow)

---

## 1. THE GOAL (the owner's words, do not soften them)

> "Our goal is to generate the best fused embedding… we need better ARI on ALL the datasets… I need a generalized model. Which works for all the datasets perfectly. Not only one or two or three — all the six."
> "No, this is not called generalized. One model… I need strictly the 6 datasets doing best. Do it however you can. Being honest. One pipeline can do all."

Concretely: **ONE single fully-unsupervised pipeline** (one architecture, one training recipe, one clustering recipe) that achieves the best fused-embedding clustering (post-hoc ARI) on **all six datasets simultaneously**. A "model zoo with per-dataset switches" was explicitly proposed and **rejected by the owner**. Per-dataset cherry-picking is useful diagnostically (it defines the ceiling) but is NOT the deliverable.

Realistic target: human lymph node ~0.35+ ARI (currently stuck at 0.25); mouse brain is already 0.32–0.51.

---

## 2. THE DATA

| ID | Tissue | Modalities | Spots ≈ | Declared k |
|----|--------|-----------|---------|-----------|
| A1 | Human lymph node | RNA + ADT (protein, ~200 markers) | ~3,500 | 10 |
| D1 | Human lymph node | RNA + ADT (protein, ~200 markers) | ~3,400 | 11 |
| E11 | Mouse embryonic brain | RNA + ATAC | ~1,300 | 8 |
| E13 | Mouse embryonic brain | RNA + ATAC | ~1,800 | 12 |
| E15 | Mouse embryonic brain | RNA + ATAC | ~2,000 | 12 |
| E18 | Mouse embryonic brain | RNA + ATAC | ~2,200 | 14 |

**Fixed preprocessing (identical for every experiment — do not change casually):**
- RNA: SCTransform v2 → PCA 64 → scale
- ADT: CLR (centered log-ratio) → scale
- ATAC: TF-IDF → LSI 50 → drop components with |corr(LSI, log depth)| > 0.75 → scale
- Spatial graph: kNN, k = 15

**Dataset spec (copy this exactly — guessing paths has already burned two full run cycles):**
- Kaggle private dataset: `pulokpulok/spatial-multiomics-6datasets-private`
- A1 folder `10x_human_lymph_node_A1`, D1 folder `10x_human_lymph_node_D1`: GT file `annotation.csv`, column `manual-anno`
- E11/E13/E15/E18 folders `Mouse_Brain_E11_S1` / `E13_S1` / `E15_S1` / `E18_S1`: GT file `anno.csv`, column `cluster`
- Each folder contains `adata_RNA.h5ad` + `adata_ADT.h5ad` (human) or `adata_ATAC.h5ad` (mouse)
- The working loaders live inside the Stage 5/6 notebooks (see §4) — copy them, never rewrite from memory.

---

## 3. THE RULES (non-negotiable)

1. **Unsupervised firewall** (`tests/test_firewall.py`): ground-truth labels must NEVER enter training, preprocessing, graph construction, modality weighting/gating, early stopping, or model selection. Labels are loaded ONLY to compute quarantined post-hoc ARI/NMI into `posthoc.json` (reporting only). Known limitation: the current test only statically inspects limited files with shallow AST checks — hardening it is an open task (§9).
2. **Declared-k exception:** cluster count k is read once from ground-truth label counts (table in §2). This is the ONE declared exception: applied identically to every config, disclosed in every report, used for nothing except setting k in the clustering head. (An eigengap-based k estimator was suggested as a cleaner replacement — not yet built.)
3. **RAUS v3b is the official selection rule** (since Stage 4): Borda rank over {seed-stability ARI, reconstruction-balance, R²_rna}. **Silhouette is banned from selection** — proven to actively mislead (ρ = −0.27 to +0.06 vs ARI). Note: RAUS v3c (adding R²_aux) correlated even better (0.528/0.604) in Stage 4 and was never adopted — worth revisiting.
4. **Honesty rules:** report mean ± std across seeds (42, 1234, 2024), never just the best seed. A negative result is a result — every failed model below is documented on purpose. Do not re-run a dead end (§8) without a new mechanism that addresses its diagnosed root cause.
5. **Generator convention:** `scripts/gen_*.py` are the source of truth; `notebooks/*.ipynb` are build artifacts. Edit the generator → regenerate → verify the OUTPUT notebook's content (grep for stale paths/tokens) → push both.
6. **Owner's standing instructions:** never assume environment details — ask before building when uncertain. Every build proposal leads with an honest runtime estimate. Nothing is pushed/published without the owner's word (pushes go through approval cards; a denial is final).

---

## 4. HOW THE MACHINE WORKS (infrastructure)

- **Compute:** Kaggle GPU notebooks. Datasets are mounted from the private Kaggle dataset above. Auth: fine-grained GitHub PAT stored in Kaggle Secrets as `GITHUB_TOKEN`. **After changing the secret, the Kaggle kernel must be restarted** or the old token keeps being used.
- **Result flow:** each notebook clones the repo, runs experiments with `skip_done` semantics, and after EVERY run pushes results to GitHub: `git add` (ALL changed files incl. `registry.jsonl`) → `git commit` → `git pull --rebase` → `git push`. Order matters — committing after the pull breaks every subsequent push with "unstaged changes". The token is scrubbed (`***`) from all error output — a past version printed it in plaintext (see §11).
- **GitHub is the source of truth for runs.** Local clones fall behind; pull before trusting them.
- **Per-run artifacts** (in `runs/<stage>/<combo>/<dataset>/seed<seed>/`): `fused_embedding.npy` (n×32 fused embedding — the primary artifact), `metrics.json` (unsupervised metrics), `posthoc.json` (quarantined ARI/NMI), plus a stage-level `registry.jsonl` (one row per run).
- **Notebook template:** install-first cell (Kaggle does NOT preinstall scanpy/leidenalg — every notebook must pip-install its imports in cell 1) → git setup cell → data cell (MARKER-based root finding + download fallback) → model/eval cells → final `git_push(...)` call.
- **CRITICAL codebase caveat:** the Stage 5/6 notebooks define their models **inline** and do **not** import `pipeline/`. The shared modules in `pipeline/` (`datasets.py`, `models.py`, `training.py`) belong to the earlier agent pipeline and diverge from the notebook code (e.g., the notebooks already implement the LSI depth filter and do NOT apply the SAGEConv L2-normalization that `pipeline/` has). When reviewing or reusing code, the notebooks are ground truth. See §10.

---

## 5. CODEBASE MAP

| Path | What it is |
|------|-----------|
| `notebooks/` | The 10 experiment notebooks (below) |
| `scripts/` | `gen_*.py` notebook generators + utilities — source of truth |
| `pipeline/` | Earlier shared modules — NOT used by Stage 5/6 notebooks (see caveat §4) |
| `runs/` | All results: `s1/`, `s2/`, `s3/`, `s3-notebook/`, `s4/`, `s5-notebook/` (234 runs), `s6-exp0/` (2,819 evals), `s6-frank/`, `s6-selector/`, `s6-baseline/`, `s6-concat/` |
| `reports/` | Stage reports + master report + codebase report + agent review |
| `briefs/` | Directives that scoped each stage (BRIEF-*.md) |
| `tests/` | `test_firewall.py` — the unsupervised firewall test |
| `HANDOFF.md`, `KAGGLE_GITHUB_SETUP.md` | Collaboration protocol + setup instructions |

**Notebooks:**
| Notebook | Purpose | Status |
|----------|---------|--------|
| `stage5_generalization.ipynb` | Stage 5: 13 combos × 6 datasets × 3 seeds | Complete (234 runs) |
| `stage6_exp0_recluster.ipynb` | Exp0: recluster 234 saved embeddings × 4 transforms × 3 heads | Complete (2,819 evals) |
| `stage6_frankenstein.ipynb` | S6-FRANK combined-mechanism model | Complete — FAILED |
| `stage6_selector_test.ipynb` | Can stability/silhouette pick (transform, head) unsupervised? | Complete — NO |
| `stage6_baseline_pca_leiden.ipynb` | PCA+Leiden classical baseline | Complete |
| `stage6_concat_baseline.ipynb` | Concatenation baselines (naive/var-scaled/single-modality) | Complete |
| `stage6_human_push.ipynb` | Five cheap ideas for human data (see §9) | **Built, NOT YET RUN** |
| `stage3_synthesis.ipynb`, `stage4_validation.ipynb`, `embedding_module_sweep.ipynb` | Earlier stages | Complete |

---

## 6. EXPERIMENT HISTORY (what was tried, what happened)

### Stage 0 — Infrastructure
Kaggle↔GitHub direct-push pattern established; six datasets uploaded as one private Kaggle dataset. Report: `reports/REPORT-stage0.md`.

### Stage 1 — Screening matrix (~180 runs)
Broad architecture screening. First evidence that **silhouette misleads**: negatively correlated with ARI on D1 (r = −0.569) and E11 (r = −0.434). kNN-overlap diagnostic found to be degenerate noise. Per-dataset winners already differed. Report: `reports/REPORT-s1.md`.

### Stage 2 — Architecture search (180 runs)
10 configs × 6 datasets × 3 seeds. **Winner: `S2-EXP09-ABL-spa-laplacian`** — Graph Laplacian + hierarchical MLP fusion; #1 or #2 on 4/6 datasets, top-4 on all six, average rank 2.00. This became the ancestor of the BASE architecture. Selected under RAUS v2 (which still included silhouette — later repaired). Report: `reports/REPORT-s2.md`.

### Stage 3 — Synthesis (144 runs)
Tested SCOT (optimal-transport alignment), UAF-Gaussian fusion, GAT encoder, reconstruction-target ablations, k-means vs DEC vs IDEC heads. **SCOT verdict:** silhouette 0.756 but ARI collapsed on mouse brain (E11 −0.009, E13 0.007, E15 0.004) — stable, confident, and completely wrong. Permanently disqualified; the canonical proof that silhouette can be gamed. Report: `reports/REPORT-s3.md`.

### Stage 4 — Champion validation & criterion repair (72 runs)
- λ_spa calibrated → **0.20** (swept on E15; R²_rna = 0.2535).
- **RAUS v3b adopted**: silhouette removed from selection (proven ρ = −0.27…+0.06); Borda{seed-stability, reconstruction-balance, R²_rna} correlates ρ = 0.512/0.424 with quarantined ARI.
- **Champion `S4-CHAMP-sage`** (SAGE + HierMLP + PCA30 + λ 0.2 + IDEC): mean ARI 0.250 vs GAT 0.241 vs base 0.232.
- Uniformity anti-collapse shelved (0.229 < base). Report: `reports/REPORT-s4.md`.

### Stage 5 — Generalization (234 runs, 13 combos)
BASE = SAGE encoders + hierarchical MLP fusion + Laplacian (λ=0.1) + IDEC. Combos: BASE, SP-sage (shared-private 16+8+8), SP-sage-adapt, SP-sage-nodis, adapt-modality (Kendall & Gal uncertainty weighting), adapt-spatial (Moran's-I adaptive λ_spa), dis005 (λ_dis=0.05), dis04 (λ_dis=0.4), generalized (all adaptations), nograph (MLP encoders only), shared-private (MLP), + S6-ONE/v2 (below).

**Findings:**
1. **THE TISSUE DIVIDE (the project's central empirical fact):** human lymph node clusters BETTER without the spatial graph (A1: nograph 0.240 vs graph 0.205); mouse brain collapses without it (E11: 0.065 vs 0.294). Brain needs spatial smoothing; lymph node is hurt by it.
2. **Moran's I does NOT predict the divide** (E18 has I ≈ 0 yet needs the graph). Measured I: A1 0.014, D1 0.013, E11 0.006, E13 0.020, E15 0.020, E18 −0.000.
3. **The "adaptive λ_dis" mechanism was an illusion:** cross-modal kNN agreement was ~0.01 on ALL datasets (high-dimensional kNN sets barely overlap), so λ_dis was effectively fixed at 0.2. What looked like adaptation was a fixed stronger value. Optimal λ_dis is genuinely dataset-specific: E11 peaks at 0.2 (inverted-U: 0.05→0.160, 0.1→0.121, 0.2→0.322, 0.4→0.143); E13 peaks at 0.1 (0.396).
4. **E11 collapse diagnosis:** the shared-private architecture itself causes it, not the missing graph (SP-sage 0.121 ≈ MLP-SP; nodis 0.145 also collapses).
5. Stage-5 bests (k-means head): A1 0.240 (nograph), D1 0.241 (generalized), E11 0.294 (BASE), E13 0.396 (SP-sage), E15 0.503 (BASE), E18 0.325 (SP-sage).
Full data: `runs/s5-notebook/registry.jsonl` (234 rows) + saved embeddings for every run.

### Stage 6, Experiment 0 — Recluster everything (2,819 evals, zero training)
Every saved Stage-5 embedding × 4 transforms (raw / standard / l2 / pca24) × 3 heads (kmeans / spherical / GMM). **The embeddings were better than reported — the information was there, badly clustered.**

Best per dataset (current all-time ceiling):
| Dataset | ARI | Combo | Transform/Head |
|---------|-----|-------|----------------|
| A1 | **0.250** | S5N-adapt-modality | raw / gmm |
| D1 | **0.250** | S5N-nograph | raw / kmeans |
| E11 | **0.445** | S5N-SP-sage-nodis | pca24 / spherical |
| E13 | **0.436** | S5N-dis005 | pca24 / kmeans |
| E15 | **0.506** | S5N-adapt-spatial | standard / kmeans |
| E18 | **0.323** | S5N-dis005 | pca24 / kmeans |

Patterns: pca24 dominates mouse; GMM wins on human A1; spherical k-means is *critical* for E11 (0.292 → 0.445). **Best TRUE one-pipeline** (one combo + transform + head fixed for all six): **S5N-BASE + standard/kmeans → mean 0.320** (A1 0.221, D1 0.205, E11 0.292, E13 0.417, E15 0.501, E18 0.285). Cherry-picked mean: 0.368. The 0.048 gap is concentrated in E11.

### Stage 6 — S6-ONE and S6-ONE-v2 (gated dual-path) — FAILED, abandoned
Idea: per-spot sigmoid gate blends a SAGE graph path with an MLP feature path, learning graph-reliance unsupervised. v1 (18 runs): gates learned **backwards** (graph favored on human where it hurts; A1 0.191, E11 −0.013). v2 (skeptical init −1.0 + gate-coupled Laplacian): gates **collapsed** to ~0.05–0.11 everywhere — the model ignored the graph even on brain (E11 0.014). Root cause, proven twice: gates optimize reconstruction, and smoothing helps reconstruction even when it destroys clusters. **Per-spot gating from unsupervised losses does not work. Do not rebuild this.**

### Stage 6 — S6-FRANK (Frankenstein) — FAILED, abandoned
Six mechanisms combined from an 11-paper literature review (MoE fusion, shared-private, masked reconstruction, isometric regularization, homogeneity-gated spatial, spherical head). 18 runs. Worse than baseline on EVERY dataset (mean ARI: A1 0.188, D1 0.150, E11 0.149, E13 0.125, E15 0.216, E18 0.170). MoE gate collapsed (A1: 97% aux), homogeneity thresholds were guessed, seed-unstable. Lesson: **paper mechanisms do not plug-and-play**; they interact in ways papers don't describe.

### Stage 6 — Selector test — negative result
Can an unsupervised metric pick the best (transform, head) per dataset? Cross-seed stability: **2/6 correct**. Silhouette: **3/6 correct**, with a systematic bias toward k-means (it is computed in Euclidean space and is blind to the hypersphere geometry where spherical k-means wins). Unsupervised clustering selection remains unsolved.

### Stage 6 — PCA+Leiden classical baseline (Oct 15)
Purpose: check the neural net adds anything over classical PCA + Leiden. It does, everywhere:
| Dataset | PCA+Leiden | Neural best | Gain |
|---------|-----------|-------------|------|
| A1 | 0.178 | 0.250 | +0.072 |
| D1 | 0.158 | 0.250 | +0.092 |
| E11 | 0.215 | 0.445 | +0.230 |
| E13 | 0.240 | 0.436 | +0.196 |
| E15 | 0.339 | 0.506 | +0.167 |
| E18 | 0.106 | 0.323 | +0.217 |
Results: `runs/s6-baseline/baseline_results.json`. (Note: the baseline's Leiden resolution was later force-fit to hit declared k exactly, keeping ARI reporting-only.)

### Stage 6 — Concatenation baselines (Oct 15)
Purpose: literature reports naive concatenation beating trained models on lymph node. Tested naive concat, variance-scaled concat, RNA-only, aux-only, PCA-32 versions × kmeans/GMM/spherical at declared k.
| Dataset | Best concat | Config | Neural best | Verdict |
|---------|------------|--------|-------------|---------|
| A1 | 0.240 | naive_concat_pca32 / gmm | 0.250 | TIE |
| D1 | 0.243 | **aux_only (ADT) / kmeans** | 0.250 | TIE |
| E11 | 0.286 | naive_concat / spherical | 0.445 | Neural wins |
| E13 | 0.400 | naive_concat_pca32 / kmeans | 0.436 | Neural wins |
| E15 | 0.426 | naive_concat / kmeans | 0.506 | Neural wins |
| E18 | 0.225 | naive_concat / kmeans | 0.323 | Neural wins |
**Bombshell:** on D1, ADT protein ALONE + k-means (0.243) nearly matches the full neural model. On human data the architecture is dead weight — the signal is in the features, especially protein. On brain, the model earns its keep. Results: `runs/s6-concat/`.

### Literature deep-dive — human lymph node (Oct 15)
Full report (owner's copy): `research_notes/lymph-node-adt-clustering-research-20261015-0732/` (Phoenix workspace, not in repo). Key facts:
- Our 0.25 ≈ **SpatialGlue's published 0.241** on human lymph node A1 (Nature Methods 2024) — we are at 2024 SOTA level, on a *different* annotation (ours: manual-anno k=10/11, ~200 proteins; the published benchmark: H&E labels k=6, 31 proteins — absolute numbers are NOT directly comparable, deltas/rankings are).
- **MultiSpaceNet (2025) reaches 0.570** on the benchmark — via per-cell attention + a **protein-similarity graph** (kNN on features, correlation distance) with weak/no spatial edges. Its ablation: deleting spatial edges (0.474) ≥ keeping them (0.468). Code public: github.com/yongzhuangliulab/MultiSpaceNet.
- **ADTnorm (Nature Communications 2025)** beat 14 ADT normalization methods on 13 CITE-seq datasets; needs no empty droplets (DSB does — inapplicable to Visium co-profiling). CLR's flaw: panel library size does not reflect depth. Expected gain +0.02–0.05.
- BaySC weight sweep: optimal fusion slightly RNA-leaning; MultiSpaceNet attention converged to RNA:protein ≈ 0.48/0.52 → a single global weight is enough; per-cell weights buy nothing.
- Lymph node A1 is the highest-variance benchmark in the literature (MultiSpaceNet σ = 0.214, bimodal across seeds) → ensembling is warranted.

---

## 7. CURRENT SCOREBOARD

**Best per dataset (ceiling, any method):** A1 **0.250** · D1 **0.250** · E11 **0.445** · E13 **0.436** · E15 **0.506** · E18 **0.323**
**Best true one-pipeline:** S5N-BASE + standard/kmeans, mean **0.320** (cherry-picked mean 0.368).
**Status vs goal:** mouse brain is respectable; human lymph node is the wall. Nothing — graph, no-graph, gating, concat — has broken 0.25 on A1/D1.

---

## 8. PROVEN THESES & DEAD ENDS (do not re-litigate)

1. **Silhouette actively misleads under over-smoothing.** Proven four ways (Stages 1, 3/SCOT, 4, selector test). Banned from selection.
2. **Reconstruction ≠ clustering quality.** ρ = −0.011 across 234 runs. The models optimize the wrong objective — this is the deepest known problem.
3. **Seed-stability is necessary but not sufficient.** SCOT was stable (0.606) with ARI −0.009.
4. **The tissue divide is structural** and appears at BOTH training time (nograph ablation) and clustering time (Exp0 transform/head patterns).
5. **Per-spot gating cannot learn graph-reliance from unsupervised losses.** Three independent failures (S6-ONE, S6-ONE-v2, FRANK's MoE).
6. **Embeddings are better than their default clustering suggests** (Exp0: E11 +0.151 with zero retraining).
7. **Paper mechanisms don't plug-and-play** (FRANK failed on all six datasets).
8. **Unsupervised (transform, head) selection is unsolved** — stability 2/6, silhouette 3/6.
9. Moran's I does not predict graph usefulness. Cross-modal kNN agreement is ~0.01 everywhere and cannot drive adaptation.

---

## 9. OPEN THREADS & QUEUED NEXT STEPS (in priority order)

1. **RUN `notebooks/stage6_human_push.ipynb`** — built and pushed, never run. One CPU run (~30 min) testing five ideas on all six datasets: (a) ADT preprocessing ablation — CLR vs arcsinh vs CLR+per-protein z-score vs per-protein robust scaling; (b) UMAP-16/32 + clustering; (c) RNA:aux weight sweep (0.0/0.3/0.5/0.7/1.0); (d) ADT feature selection (top 25%/50%/100% variable); (e) 20× k-means consensus ensemble. Output: `runs/s6-human-push/`.
2. **MultiSpaceNet reproduction** — the highest-upside direction for human data (literature 0.25 → 0.45+): modality MLP encoders → per-cell attention fusion → **feature-similarity graphs** (RNA-kNN + protein-kNN, correlation distance) instead of the spatial graph → reconstruction + cross-modal protein-prediction losses → L2 → PCA≤20 → GMM. Run 10 seeds and ensemble (high variance expected).
3. **ADTnorm** — replace CLR with the real ADTnorm (R package / Python wrapper) if the arcsinh/robust approximations in (1) show preprocessing matters.
4. **Protein-first model** — the concat result (ADT-only ≈ full model on D1) motivates an ADT-primary architecture with RNA as auxiliary, at least as a diagnostic.
5. **Cross-modal contrastive (graph-free)** — InfoNCE/VICReg alignment of RNA↔ADT embeddings (scMDCF-style) as the fusion objective; addresses thesis #2 by optimizing something other than reconstruction.
6. **Housekeeping:** harden `tests/test_firewall.py` (currently shallow); try RAUS v3c (add R²_aux); eigengap k-estimation to replace declared-k; write the missing Stage 5 formal report (findings are in the master report §Stage 5).

**Decision the owner still owes:** whether the endgame is (a) one fixed pipeline accepting ~0.32 mean, (b) continued search for a genuinely unified architecture, or (c) a two-tier model with a *label-free* tissue switch (the previous agent's recommendation — the owner has so far rejected switches).

---

## 10. THE AGENT REVIEW — READ WITH THIS CAVEAT

`reports/AGENT-REVIEW.md` (2026-10-11) is a serious critique: it argues ONE model is unlikely (accept a two-tier graph/no-graph switch), calls declared-k label leakage (technically true; it is the disclosed exception), confirms the local-optimum diagnosis (reconstruction uncorrelated with ARI), and proposes VICReg contrastive learning, protein-first embeddings, and neighborhood contrastive learning for human data.

**Caveat:** several of its code-bug findings (SAGEConv L2-norm destroying ADT magnitude, missing LSI depth filter) were found in `pipeline/` — which the Stage 5/6 notebooks do not use. The inline notebook code already has the depth filter and no L2 norm. Verify any review claim against the notebook code before acting on it.

---

## 11. SECURITY NOTES

- A Stage-0 GitHub PAT was exposed in plaintext early on; a second PAT was exposed on 2026-10-09 when a notebook error handler printed the full git command. Both were **revoked and reissued**; current notebooks scrub the token from all output. Never print `git` commands containing the auth header.
- `scripts/generate_stage2_notebook.py` (line ~145) contains a **hardcoded Kaggle API key** flagged in audit. Status: owner to revoke/rotate; the string has NOT yet been stripped from the file. Handle with care; do not copy it anywhere.
- Kaggle Secrets changes require a kernel restart to take effect.

---

## 12. WORKING AGREEMENTS WITH THE OWNER

- **Never assume** paths, file layouts, column names, or environment details — read the existing working code (Stage 5/6 loaders) or ask. Two run cycles were wasted on guessed GT paths (`anno.csv`/`cluster` for mouse; `annotation.csv`/`manual-anno` for human — §2 has them).
- **Every Kaggle notebook ships with a verified install-first cell.** Kaggle lacks scanpy/leidenalg/etc. Two notebooks in a row failed on missing imports; the owner will not tolerate a third.
- Lead every build report with an **honest runtime estimate**; keep runs checkpointed and chunked.
- Terse messages from the owner ("check github", "done", "try") mean *execute*, not *discuss*. "Leave it" / "later" are absolute stops.
- Verify results server-side (fetch the pushed files) rather than trusting notebook output claims.
- Formal documents use the owner's full name: **Md Rifat Ahasan Pulock**. In conversation: PULOK.

---

## 13. WHERE EVERYTHING LIVES

| What | Where |
|------|-------|
| All Stage 5 runs | `runs/s5-notebook/registry.jsonl` (234 rows) + per-run folders with embeddings |
| Exp0 evals | `runs/s6-exp0/exp0_results.jsonl` (2,819 rows) |
| Selector test | `runs/s6-selector/` |
| Baselines | `runs/s6-baseline/baseline_results.json`, `runs/s6-concat/concat_summary.json` |
| Full experiment narrative | `reports/REPORT-master-experiments.md` |
| Codebase walkthrough | `reports/REPORT-codebase-understanding.md` |
| Independent critique | `reports/AGENT-REVIEW.md` (+ the prompt that produced it, `AGENT-REVIEW-PROMPT.md`) |
| Stage briefs | `briefs/BRIEF-*.md` |
| Firewall | `tests/test_firewall.py` |

*End of handoff. Good luck — the human lymph node wall (0.25) is the problem that matters. Everything else is context.*
