# AGENT REVIEW

**Reviewer:** Astra (agent review)
**Date:** 2026-10-11
**Files examined:** All three reports, `pipeline/models.py`, `pipeline/datasets.py`, `pipeline/training.py`, `tests/test_firewall.py`, `briefs/BRIEF-007-stage-4-brief.md`, `runs/s5-notebook/registry.jsonl` (234 rows), `runs/s6-exp0/exp0_results.jsonl` (2,819 rows), `runs/s6-selector/selector_summary.json`

---

## Part 1: Approach Critique

### 1. ONE model achievability: **Unlikely with current architecture family — reframe the goal**

The tissue divide is not a tuning problem. It is a structural incompatibility between the inductive biases of spatial-graph-based message passing and two tissue types with fundamentally different spatial organisation:

- **Mouse brain (E11–E18):** Spatially coherent domains → graph smoothing creates embedding regions that align with cell-type clusters → graph helps (E11: 0.065 without graph vs 0.294 with graph).
- **Human lymph node (A1/D1):** Cellular neighborhoods are spatially interleaved (T cells and B cells coexist at fine scale) → graph smoothing **blurs** cluster boundaries → graph actively hurts (A1: 0.240 without graph vs 0.205 with graph).

Three gating mechanisms tried to learn this automatically and all failed because the reconstruction loss rewards smoothing regardless of clustering quality. This is not fixable within the current loss design.

**Verdict:** Accept two-tier architecture. A single *codebase* with a dataset-level binary switch (graph / no-graph) is honest, simple, and still "one pipeline." The pretension that one set of weights will be best on both tissue types is contradicted by 234+36+18 runs of evidence. Stop trying.

The pragmatic decision: **S5N-BASE with graph for E11–E18, S5N-nograph (or S5N-adapt-modality) for A1/D1.** This is one codebase, two modes. Declare it as an architecture with conditional spatial encoding, not two separate models.

### 2. Declared-k: **Technically a violation, practically acceptable — but fix it**

Reading k from `DECLARED_K` (which mirrors ground-truth label counts) is unambiguously label information in the pipeline. The fact that it's applied identically doesn't matter — you still need the labels to define k. In a true deployment scenario, k would be unknown.

**However:** For a benchmarking study, this is standard practice and widely accepted in the literature (SpaGCN, STAGATE, GraphST all assume known k). The violation is **epistemic**, not methodological — it doesn't bias the model toward specific clusters, it only sets the partitioning resolution.

**Recommendation:** Estimate k with a simple, unsupervised heuristic and report both. Best candidate: **eigengap of the graph Laplacian** (you already compute Laplacians). Take the k where the gap between consecutive eigenvalues is maximised. This costs zero GPU time and eliminates the label dependency. If the eigengap-k matches declared-k ± 2 on most datasets, the declared-k exception was harmless. If it diverges, that's scientifically interesting.

### 3. Local optimum: **Yes — the architecture family is exhausted**

All 13 combos are SAGE/GAT + MLP fusion + reconstruction loss variants. The Exp0 results show that the *best possible* ARI from these embeddings (cherry-picked per dataset with optimal clustering) is 0.295 for A1, 0.268 for D1. Even with perfect clustering selection, the embeddings themselves cap out here for human data.

The 234-run registry confirms: `rho(R2_rna, ARI) = −0.011` across the full dataset — reconstruction quality is **uncorrelated** with clustering quality. The reconstruction objective is not teaching the model what you want it to learn. You are iterating on a model that optimises for the wrong thing.

**What hasn't been tried:**
- Contrastive cross-modal learning (CLIP-style) — no reconstruction, directly aligns RNA↔ADT/ATAC
- Deep graph infomax / mutual information maximisation — no reconstruction
- Variational approaches (scVI family) — different loss landscape entirely
- Simple baselines: PCA concat → UMAP → Leiden (no neural network at all)

The last point is important. You should know how far ahead of PCA+Leiden your model actually is, on every dataset. If the gap is small, the neural network isn't contributing much.

### 4. Post-hoc ARI: **Imperfect but unavoidable — add a second metric**

ARI against manual annotations is the standard benchmark in this field. Abandoning it would make results incomparable to published work. The concern about noisy labels is valid, especially for E11 (8 clusters in ~1,300 spots — some clusters have <100 spots).

**However:** The E11 spherical mystery (0.292→0.445 with just a clustering change) suggests the embedding geometry is non-Euclidean. When spherical k-means helps this dramatically, it means the embedding concentrates on a hypersphere — the information is *directional*, not distance-based. This is actually a strong signal about embedding quality, not annotation noise.

**Recommendation:** Add **NMI** as a co-primary metric (you already compute it in `evaluate_posthoc_quarantined`). ARI is sensitive to cluster size imbalance; NMI is not. Report both. If a method improves ARI but drops NMI, it's gaming the metric.

### 5. Abandoning gates: **Correct decision, wrong diagnosis**

The three gate failures share a root cause you correctly identified: *gates optimise reconstruction, not clustering*. But the conclusion "gating is wrong" is too strong. The correct conclusion is: **gating is wrong when the gate objective is reconstruction.**

A clustering-aware gate is theoretically possible: train the gate to maximise the IDEC clustering loss (Student-t KL divergence) rather than reconstruction. The gate learns to blend graph/no-graph in a way that sharpens clusters, not smoothing. But this creates a chicken-and-egg problem: good clusters require good gating, good gating requires good clusters.

**Verdict:** Abandoning gates was correct *given the current loss design*. But if you switch to a contrastive or information-maximising objective (Experiment 1 below), revisiting gates becomes worthwhile because the new objective doesn't reward smoothing.

---

## Part 2: Code Review

### `pipeline/models.py`

**Bug 1 — SAGEConv L2 normalisation destroys magnitude information (line 41):**
```python
if self.normalize:
    out = F.normalize(out, p=2, dim=-1)
```
`normalize=True` is the default. This forces all node embeddings onto the unit sphere *at every layer*. For the encoder output, this means the fusion module receives unit-norm vectors where magnitude carries no information. This is particularly damaging for ADT/protein data where abundance (magnitude) is biologically meaningful. The L2 norm projects everything to the same radius, destroying information about expression level.

**Impact:** This may be why the human datasets plateau. ADT has only ~200 features — magnitude matters more than direction. Setting `normalize=False` on the mod2 encoder (line 210) is the single cheapest experiment to try.

> [!WARNING]
> This is a high-priority finding. The `normalize=True` default in [SAGEConv](file:///Users/rifatahasan/Documents/spatial-omics-auto-starter/pipeline/models.py#L17-L42) means every embedding layer projects to the unit sphere. For protein (ADT) data with ~200 features, magnitude carries biological signal about protein abundance that is being discarded.

**Bug 2 — ClusterHead initialisation is random (line 156):**
```python
nn.init.xavier_uniform_(self.cluster_centers)
```
The IDEC cluster centres are initialised with Xavier uniform, not from a k-means pre-fit on the embeddings. Standard IDEC practice is to pre-train the autoencoder, run k-means on the embeddings, and initialise `cluster_centers` from the k-means centroids. Random initialisation means the Student-t soft assignments start from garbage and the KL loss may never converge to meaningful clusters. This could explain why IDEC underperforms plain k-means in many Exp0 rows.

**Bug 3 — `Model_H_HiRe` uses three separate graph edge sets but `Model5_FullARISE` shares `deconv1` across three decoder paths (line 414–416):**
```python
rec_sim = self.deconv_sim(self.deconv1(x_sim))
rec_dist = self.deconv_dist(self.deconv1(x_dist))
rec_pro = self.deconv_mod2(self.deconv1(pro))
```
The same `deconv1` layer is applied to `x_sim`, `x_dist`, and `pro`, which have different semantic meanings. This shared bottleneck forces the decoder to learn a compromise representation. Not necessarily a bug, but a design choice worth ablating.

**Missed opportunity — No dropout or layer norm in `Model_H_HiRe`:**
The main model has no regularisation beyond BatchNorm in the fusion layers. Adding dropout (0.1–0.2) after the encoder and before fusion could reduce over-smoothing, especially on small datasets like E11 (1,300 spots).

### `pipeline/datasets.py`

**Issue 1 — CLR normalisation is non-standard (line 82–89):**
```python
log_data = np.log1p(data)
geometric_mean = np.mean(log_data, axis=1, keepdims=True)
adata.X = log_data - geometric_mean
```
This computes `log1p(x) − mean(log1p(x))` per cell. Standard CLR is `log(x / geometric_mean(x))` which equals `log(x) − mean(log(x))`. The `log1p` instead of `log` introduces a bias for low-count proteins: `log1p(0) = 0` whereas `log(0)` would be handled by adding a pseudocount. This is *common* but there's a subtle issue: `np.mean(log_data)` averages over *all* proteins including zeros, pulling the geometric mean down. For ADT panels with many zeros (common in lymph node), this inflates the CLR-transformed values of abundant proteins while compressing rare ones.

**Impact on human data:** ADT panels for A1/D1 have ~200 proteins, many with sparse counts. The CLR bias toward abundant proteins may be destroying the signal from rare surface markers that distinguish lymph node subtypes (e.g., CD45RA vs CD45RO).

**Recommendation:** Try **DSB normalisation** (Denoised and Scaled by Background) which explicitly models background noise in ADT — designed specifically for CITE-seq protein data. Or at minimum, try CLR with `log(x + 1) − mean_nonzero(log(x + 1))` to exclude zeros from the geometric mean.

**Issue 2 — ATAC LSI component dropping is absent (line 172–177):**
The report states the pipeline should "drop components with |corr(LSI, log depth)| > 0.75". But `datasets.py` has no such filtering — it simply does TF-IDF → TruncatedSVD and takes the top components. The first LSI component is almost always correlated with sequencing depth and should be removed. The fact that it's kept means the ATAC embedding contains a library-size confounder.

```python
# CURRENT CODE (no depth correction):
tfidf_mat = tfidf(adata_mod2.X)
atac_comps = min(60, tfidf_mat.shape[1])
mod2_raw = run_pca(tfidf_mat, n_comps=atac_comps)
mod2_pca = mod2_raw[:, :min(n_comps_mod2, atac_comps)]
```

**Issue 3 — Spatial graph is always k=15 (hardcoded via `kneighbors_graph`):**
Not visible in `datasets.py` directly but mentioned in the report. k=15 is used for all datasets regardless of spot density. Human lymph node has ~3,500 spots; mouse brain E11 has ~1,300. At k=15, the E11 graph connects each spot to ~1.2% of all spots; the A1 graph connects each spot to ~0.4%. These are very different connectivity regimes. A density-adaptive k (e.g., k proportional to sqrt(N)) would be more principled.

### `tests/test_firewall.py`

**Gap 1 — The firewall only inspects `training.py` (line 8):**
```python
self.training_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pipeline", "training.py"))
```
It does NOT inspect `datasets.py` or `models.py`. If someone reads ground-truth labels inside a model's `forward()` method (unlikely but possible) or uses labels in preprocessing (e.g., to select HVGs per cell type), the firewall wouldn't catch it.

**Gap 2 — The AST walk only checks function names, not attribute access (line 21):**
```python
fname = getattr(sub.func, "id", getattr(sub.func, "attr", ""))
```
This catches `adjusted_rand_score(...)` but NOT `sklearn.metrics.adjusted_rand_score(...)` (where the call is an `ast.Attribute` on an `ast.Attribute`). A malicious or careless import like `from sklearn.metrics import adjusted_rand_score as ars` followed by `ars(y_true, y_pred)` would also evade detection.

**Gap 3 — Runtime firewall (line 24–27) only checks for `true_labels` and `ground_truth` attributes:**
```python
if hasattr(args, "true_labels") or hasattr(args, "ground_truth"):
    raise PermissionError(...)
```
Renaming the attribute to `labels`, `gt`, `annotations`, or `y_true` would bypass this check. The test should scan ALL attributes of `args` for arrays that match ground-truth shape/dtype.

**Gap 4 — The `DECLARED_K` dict in `datasets.py` is not tested:**
The firewall doesn't verify that `DECLARED_K` values are only used to set `num_clusters` and nothing else. If someone used `DECLARED_K["A1"]` as an input feature dimension or to weight a loss, the firewall wouldn't flag it.

> [!IMPORTANT]
> The firewall is better than nothing but has significant gaps. It only inspects one file, uses shallow AST patterns, and the runtime check is trivially bypassable. For a project where "no label leakage" is the core methodological claim, this needs hardening.

---

## Part 3: Next Experiments (prioritized by information gain)

### Experiment 1: **Remove SAGEConv L2 normalisation for mod2 encoder**

- **What:** In [pipeline/models.py line 210](file:///Users/rifatahasan/Documents/spatial-omics-auto-starter/pipeline/models.py#L210), change `self.enc_mod2 = make_conv(in_mod2_dim, hidden_dim)` to pass `normalize=False` to SAGEConv. Or simpler: run Exp0-style reclustering on existing embeddings but first extract the *pre-normalisation* mod2 embedding from a single forward pass.
- **Why:** The unit-sphere projection destroys ADT magnitude information. Human lymph node has ~200 ADT features where abundance level IS the signal. This is the single most plausible explanation for the human-data plateau: the model is throwing away the most informative part of the protein data.
- **Evaluate:** Post-hoc ARI on A1/D1 specifically. If A1 goes from 0.25 → 0.28+ with just this change, the diagnosis is confirmed.
- **Compute:** ~1 hour (6 datasets × 3 seeds = 18 runs). Cheapest experiment on this list.
- **Risk:** Low. If it doesn't help, it costs almost nothing and rules out a major hypothesis.

### Experiment 2: **PCA+Leiden baseline comparison**

- **What:** For each dataset, take the RNA PCA (64D) and mod2 PCA, concatenate, optionally standard-scale, run Leiden clustering (scanpy) at multiple resolutions. No neural network.
- **Why:** You have no baseline comparison. If PCA+Leiden achieves ARI 0.22 on A1, your best model (0.25) is only +0.03 ahead — that's within noise. If PCA+Leiden achieves 0.15, the model is contributing real value. **You cannot evaluate progress without knowing the floor.**
- **Evaluate:** ARI table for all 6 datasets, compared to current best.
- **Compute:** Near-zero (CPU only, minutes).
- **Risk:** The risk is psychological — you might discover the neural network adds little value. But that's essential to know.

### Experiment 3: **Contrastive cross-modal pretraining (VICReg-style)**

- **What:** New training objective. Instead of reconstruction, use VICReg (variance-invariance-covariance regularisation) between RNA and mod2 embeddings:
  - **Invariance:** MSE between `z_rna` and `z_mod2` for same spot
  - **Variance:** force std(z) > 1 across the batch dimension (prevents collapse)
  - **Covariance:** decorrelate embedding dimensions (prevents redundancy)
  No reconstruction decoder needed. Keep the SAGE encoders. Add a small projection head (2-layer MLP) between encoder and loss.
- **Why:** Reconstruction is provably uncorrelated with ARI (`rho = -0.011`). Contrastive/self-supervised objectives learn representations that are invariant across views (modalities) while retaining discriminative information. This directly addresses the core problem: the model should learn *what's shared* between RNA and protein/ATAC, not *how to reconstruct*.
- **Evaluate:** Post-hoc ARI on all 6 datasets. Compare to S5N-BASE. Particular interest: does it break the human-data plateau?
- **Compute:** ~3 hours (need to implement the loss + train 18 runs).
- **Risk:** Contrastive methods can collapse (all embeddings identical) despite the variance term. Monitor embedding std during training. If std drops below 0.1, the loss weights need adjustment.

### Experiment 4: **Fix ADT preprocessing — try DSB or remove CLR zero-bias**

- **What:** In [pipeline/datasets.py](file:///Users/rifatahasan/Documents/spatial-omics-auto-starter/pipeline/datasets.py#L82-L89), replace `clr_normalize_each_cell` with either:
  - (a) **Exclude zeros from geometric mean:** `geometric_mean = np.mean(log_data[:, nonzero_mask], axis=1)` — prevents zero-inflated proteins from pulling down the normalisation
  - (b) **Simple log1p + z-score** — minimal preprocessing, let the model learn
  - (c) **DSB (muon package)** — gold-standard for CITE-seq ADT, models background noise explicitly
- **Why:** The current CLR implementation averages over zeros, biasing the normalisation for sparse proteins. ADT panels in lymph node have many zero-count markers. Better preprocessing could unlock signal that's currently being normalised away. The human-data ceiling may be a preprocessing problem, not a model problem.
- **Evaluate:** ARI on A1/D1 with the existing S5N-BASE architecture (isolate the preprocessing effect).
- **Compute:** ~1 hour (18 runs with existing architecture).
- **Risk:** DSB requires estimating background distributions; if empty droplets aren't available in the data, it won't work. Options (a) and (b) are safe fallbacks.

### Experiment 5: **Implement LSI depth-correction for ATAC**

- **What:** In [pipeline/datasets.py lines 172–177](file:///Users/rifatahasan/Documents/spatial-omics-auto-starter/pipeline/datasets.py#L172-L177), after TruncatedSVD, compute correlation of each component with `log(total_counts)` and drop components with `|r| > 0.75` as stated in the report but never implemented.
- **Why:** The first LSI component is likely a depth confounder. Removing it changes the ATAC embedding for all brain datasets. If E11/E13 improve, the current brain scores are artificially depressed by the confounder. If they don't change, the architecture is robust to it (also useful to know).
- **Evaluate:** ARI on E11–E18 before/after filtering.
- **Compute:** Near-zero (recompute preprocessing, ~10 minutes CPU).
- **Risk:** Minimal. Could hurt slightly if a depth-correlated component carries real signal.

---

## Part 4: Breaking Through on Human Data

### Approach 1: **Graph-free multimodal transformer**

Drop the spatial graph entirely for human data. Replace SAGE encoders with a **multimodal transformer** where each spot's RNA features and ADT features are separate tokens:

```
Input: [CLS] [RNA_token] [ADT_token]
```

Each token is a linear projection of the raw features. Self-attention learns cross-modal interactions without spatial smoothing. The `[CLS]` token output is the fused embedding.

**Why this could work:** The lymph node's cellular organisation is defined by *molecular identity*, not *spatial proximity*. T cells, B cells, and macrophages intermingle spatially but are molecularly distinct. A transformer that attends to feature correlations (which ADT features co-occur with which RNA patterns) captures this better than a spatial graph that averages over neighbors.

**Why previous approaches failed here:** All previous architectures used spatial graph convolution, which assumes spatially adjacent spots are similar. For lymph node, this assumption is wrong — adjacent spots can be different cell types.

**Estimated impact:** If the bottleneck is truly the graph, removing it should get A1/D1 closer to ~0.30-0.35. The current nograph model (S5N-nograph) already reaches 0.240/0.250 with a simple MLP — a transformer should do better.

### Approach 2: **Protein-first embedding with RNA regularisation**

Current pipeline: RNA → primary encoder, ADT → secondary encoder, fuse. This treats RNA as primary and ADT as auxiliary. But for lymph node cell-type identification, **surface proteins (ADT) are MORE informative than RNA**. CITEseq studies routinely show that 20-30 well-chosen antibodies separate cell types better than 3,000 RNA HVGs.

**Flip the hierarchy:** Make ADT the primary input. Use a simple MLP encoder on the ~200 ADT features (no graph, no spatial). Use RNA only as a *regulariser*: add a reconstruction loss that predicts RNA from the ADT embedding. This forces the ADT embedding to capture information predictive of gene expression, but the primary clustering axis is protein-defined.

**Why this could work:** ADT features are already low-dimensional (~200), high signal-to-noise, and directly measure surface markers used in clinical cell-type classification (CD3 for T cells, CD19 for B cells, CD14 for monocytes). The current pipeline reduces these 200 features through CLR → PCA → SAGE → fusion, losing signal at each step.

**Risk:** This only works for ADT datasets (A1/D1). It's tissue-specific by design, which contradicts the "ONE model" goal. But if the goal is reframed as "one codebase, adaptive to tissue type," this is perfectly valid.

### Approach 3: **Neighborhood contrastive learning (no graph convolution)**

Instead of message-passing over the spatial graph, use the spatial graph as a **sampling strategy** for contrastive learning:

1. For each spot, define positive pairs as its k-nearest spatial neighbors
2. Define negative pairs as random non-neighbor spots
3. Train an MLP encoder to produce embeddings where positives are close and negatives are far (InfoNCE loss)

The key difference from SAGE: the spatial graph is used for *pair sampling*, not for *feature aggregation*. The encoder sees only its own features and learns to predict whether two spots are spatial neighbors. If two molecularly different cell types are spatial neighbors (as in lymph node), the contrastive loss will learn this as a *hard negative* — it will **not** smooth them together because the loss explicitly penalises mapping dissimilar spots to the same embedding.

**Why this could work:** In lymph node, the contrastive approach naturally handles interleaved cell types. A T cell and an adjacent B cell become a hard negative pair — the model learns they are spatially close but molecularly distinct. SAGE would average their features, destroying the distinction.

**Estimated compute:** ~2 hours for implementation + 18 runs.

---

## Summary Verdict

The project has done an impressive amount of systematic experimentation — 234 Stage 5 runs, 2,819 Exp0 evaluations, three distinct gating attempts, a selector validation. The methodology is rigorous, the firewall concept is sound (if under-tested), and the experimental discipline (briefs → runs → reports) is exemplary.

**But the project is stuck in a local optimum.** All architectures are SAGE+MLP+reconstruction variants. Reconstruction is provably uncorrelated with clustering quality (`ρ = −0.011`). The model is optimising for the wrong objective. Three adaptive architectures failed because they all inherited this same objective. The human-data plateau at ~0.25 is likely a combination of (a) the SAGEConv L2 normalisation destroying ADT magnitude signal, (b) CLR preprocessing biasing against sparse proteins, and (c) the spatial graph actively hurting on interleaved tissue.

**The single most important thing to do next:** Run Experiment 1 (remove L2 normalisation on mod2 encoder) and Experiment 2 (PCA+Leiden baseline). Together these cost <1 GPU-hour and will answer two critical questions: (1) is the model throwing away ADT signal, and (2) is the neural network adding value above a trivial baseline. Everything else should wait until these answers are in.
