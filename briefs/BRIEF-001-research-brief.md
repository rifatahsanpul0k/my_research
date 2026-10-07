# Fully Automated Unsupervised Spatial Multi-Omics Research — Local Agent Brief

**For:** the Antigravity IDE agent running on PULOK's MacBook Air (model: e.g. Gemini 3.8 Flash — the brief is model-agnostic)
**Author:** Phoenix (Muse), 2026-10-07 — prepared for PULOK's "fully automated research" project
**Status:** DRAFT — §9 lists the decisions PULOK must confirm before the agent starts

---

## 0. Mission

Build and run a **fully automated, strictly unsupervised** machine-learning research pipeline for **spatial multi-omics** data (RNA + ATAC or RNA + ADT, each spot/cell also carrying a 2-D spatial coordinate). The pipeline must:

1. ingest datasets,
2. preprocess each modality (including its spatial coordinates),
3. encode each modality with a modality-specific encoder,
4. fuse the modality embeddings,
5. cluster the fused embeddings,
6. **select the best configuration using only label-free criteria**,
7. report final quality with silhouette (label-free) and ARI/NMI (post-hoc, labels used only after selection).

The research goal is to **find the combination of preprocessing → encoder → fusion → clustering that works best**, discovered by systematic automated experimentation — not by hand-tuning against labels.

---

## 1. Non-negotiable methodological constraints

These are hard rules. The agent must encode them as runtime assertions, not comments. Violating any of them invalidates the "unsupervised" claim of the whole project.

1. **No ground-truth labels anywhere in the decision loop.** Labels (the `annotation` columns shipped with the datasets) may not influence: training, preprocessing choices, graph construction, modality weighting/gating, hyperparameter choice, early stopping, or **model selection**. They are computed **once, post-hoc, for reporting only**.
2. **Selection metric is label-free, always.** Allowed: silhouette score, seed-stability (ARI between clusterings from different random seeds — no ground truth involved), kNN-overlap, and the **RAUS** criterion (Borda count over {silhouette, seed-stability ARI, kNN-overlap}). **Forbidden for selection:** ARI / NMI / AMI against annotations. Any code path that passes annotation labels into a selection decision must `raise` — this "unsupervised-selection firewall" already exists in `smolab` (`validate_config` + `run_experiment`); keep it and extend it.
3. **One identical final architecture across all datasets** after modality-appropriate preprocessing. No `if dataset == ...` branches, no modality-identity priors inside the model. The model may not know whether the auxiliary modality is ADT or ATAC — it must adapt from the data statistics alone.
4. **Literature grounding starts in general ML venues**, not spatial-omics integration methods; no biology-specific architectures as starting points (per PULOK's standing boundary, 2026-09-30/10-01).
5. **Reproducibility:** every run logs a full CONFIG dict, git commit, package versions, and seeds to `registry.jsonl`. Re-runs with `skip_done=True` must reproduce bit-identical selection metrics.

### 1.1 A note on "evaluate with silhouette and ARI"

PULOK asked for both silhouette and ARI. Both are computed — but they play **different roles**, and the automation must keep them separated:

| Metric | Needs labels? | Role |
|---|---|---|
| Silhouette | No | **Selection** — picks the winning config |
| Seed-stability / kNN-overlap | No | **Selection** — part of RAUS |
| ARI / NMI vs annotations | Yes | **Reporting only** — final table, never touches selection |

If the agent ever ranks, filters, early-stops, or tunes on ARI-against-labels, the experiment is label-contaminated and must be discarded. Encode this as an assertion in the experiment runner.

---

## 2. What already exists — do not reinvent it

PULOK and Phoenix have already built most of the foundation (all paths relative to Phoenix's server workspace; see §9 for how they reach the Mac):

- **`smolab`** (`spatial-omics-lab/smolab/`): modular framework. Every experiment is a composition **M = (P, G, E, I, F, L, C)** — Preprocessing, Graph, Encoder, Integration/alignment, Fusion, Loss, Clustering — defined by one plain `CONFIG` dict. `run_experiment()` executes; `validate_config()` enforces the unsupervised firewall. Registry logging per §25 of the internal spec.
- **Phase 1 matrix** (`smolab/phase1.py`): 15 experiments × screening runs (50 runs), dataset registry with `n_clusters` per dataset. **AWAITS RUN APPROVAL — the single highest-leverage first job for the local agent.**
- **Four candidate architectures A–D** (`research/multiview-architectures-20261001.md`): A (deterministic shared/private AE + CCA-on-shared + structural fusion + k-means), B (dual VAE + MoE posterior aggregation + VCR), C (high-τ InfoNCE + MoE routing + IDEC head), D (GW structural alignment + consensus fusion + Leiden) — plus the **M0→M8 one-module-at-a-time ablation plan** and the **RAUS** selection criterion.
- **Literature sweeps**: `research/lit-sweep-20261001/` (26 reusable modules, 8 cluster reports, architecture proposal with 9 flagged judgment calls) and `research/lit-sweep-multiview-20261001/` (fusion techniques, 18 losses, 13 clustering methods, no-leakage evaluation tiers).
- **Topic analysis of IPE finals** — irrelevant here, ignore.
- **Colab notebook template** for heavy runs (the MacBook Air is CPU-only; GNN sweeps will be slow locally — see §9).

**Default instruction to the agent:** extend `smolab` (add modules via the registry pattern, no framework edits), do not start a fresh codebase. If PULOK chooses "fresh start" (§9), the agent must still reimplement the firewall (§1) before any experiment runs.

---

## 3. Data

### 3.1 Datasets in hand

| ID | Tissue | Modalities | Location on Phoenix's server |
|---|---|---|---|
| A1 | human lymph node | RNA + ADT | `spatial-omics-lab/data/10x_human_lymph_node_A1/` |
| D1 | human lymph node | RNA + ADT | `spatial-omics-lab/data/10x_human_lymph_node_D1/` |
| E11 | mouse brain | RNA + ATAC | `spatial-omics-lab/data/Mouse_Brain_E11_S1/` |
| E13 | mouse brain | RNA + ATAC | `spatial-omics-lab/data/Mouse_Brain_E13_S1/` |
| E15 | mouse brain | RNA + ATAC | `spatial-omics-lab/data/Mouse_Brain_E15_S1/` |
| E18 | mouse brain | RNA + ATAC | `spatial-omics-lab/data/Mouse_Brain_E18_S1/` — **incomplete** (source link was truncated; PULOK to re-supply) |

Each ships with `.h5ad` matrices, spatial coordinates in `obsm["spatial"]`, and annotation CSVs. **The annotation files are post-hoc-only** — the agent's data loader must expose them through a separate, clearly-marked channel that the selection code cannot import.

### 3.2 Data locality problem (must be solved before automation starts)

The datasets above live on **Phoenix's server VM, not on the Mac**. The local Antigravity agent cannot see them. Options:

- **(a) Copy to the Mac** (~600 MB total): Phoenix can package `smolab/` + `data/` for download; PULOK copies to e.g. `~/research/spatial-omics-lab/`. Simplest, recommended.
- **(b) Re-download from original sources** on the Mac via the agent (Drive links need re-supply; E18 link is broken anyway).
- **(c) New datasets from Kaggle** via the Kaggle MCP server (§8) to expand the pool — in *addition* to (a)/(b), not instead of them, unless PULOK explicitly wants a Kaggle-only study.

### 3.3 Preprocessing recipes (established 2026-09-30, keep as defaults)

- **RNA:** counts → SCTransform v2 → PCA 64 → scaling
- **ADT:** counts → CLR → scaling
- **ATAC:** counts → TF-IDF → LSI 50 → drop components with |corr(LSI, log sequencing depth)| > 0.75 → scaling
- **Spatial:** kNN graph, k = 15 (from coordinates in `obsm["spatial"]`)

The agent may propose alternative preprocessing **as registered P-module variants** — never by editing the default in place.

---

## 4. Pipeline stages (what the automation runs)

```
S0  Ingest & validate      AnnData per modality, obs aligned, obsm["spatial"] present, counts in X
S1  Preprocess (P)         per-modality recipe (§3.3); record versions + seeds
S2  Encode (E)             modality-specific encoder → RNA 32-D / AUX 30-D (baseline: Spatial GAT per modality)
S3  Integrate (I)          optional alignment (CCA-on-shared, GW structural, …)
S4  Fuse (F)               concat / attention / MoE routing / consensus → joint embedding
S5  Cluster (C)            k-means / Leiden / DEC / IDEC → cluster labels
S6  Select                 RAUS over {silhouette, seed-stability ARI, kNN-overlap} — LABEL-FREE
S7  Report                 post-hoc ARI/NMI vs annotations + all label-free metrics → registry + summary tables
```

Encoder candidates already surveyed (from the lit sweeps): MLP, GCN, GAT, VGAE/GAE, and the per-architecture encoders A–D. Fusion candidates: protected concatenation, between-modality attention (use with care — 2026-09-30 finding: naive attention can suppress informative ATAC signal), MoE routing, GW structural alignment, consensus fusion. The agent expands the matrix systematically along these axes; it does not sample configs at random.

---

## 5. The automation loop (the "fully automated" part)

The agent implements a closed loop — this is the core deliverable, not any single model:

```
1. PROPOSE   next CONFIG from the experiment matrix / ablation plan (M0→M8), or a new
             registered module variant with a one-paragraph justification citing the lit sweep.
2. VALIDATE  validate_config(cfg) — firewall: raises on annotation-based selection,
             dataset-identity branches, missing seeds.
3. RUN       run_experiment(...) with ≥3 seeds; skip_done=True; per-seed metrics logged.
3b. SYNC     download/copy ALL result folders from the run — wherever it executed (local
             disk, Kaggle output, Colab) — into the local repository, e.g.
             `runs/<stage>/<exp_id>/` (metrics, embeddings, cluster assignments, logs,
             plots). The agent then ANALYZES these results (metric distributions,
             failure modes, embedding sanity checks) and only this analysis — not
             intuition — drives steps 6 and 8. Result folders are never left on remote
             machines; the local repo is the single source of truth.
4. CHECK     assertion: true_labels never entered the selection path (static check on the
             call graph + runtime flag). Leakage → discard run, log incident, continue.
5. LOG       append full entry to registry.jsonl (config, metrics, post-hoc ARI quarantined
             in a separate field marked "reporting only").
6. SELECT    RAUS Borda ranking per dataset; promote winners to the next ablation stage.
7. REPORT    after each stage: markdown summary (ranking table, metric distributions,
             what was tried, what failed) committed to the repo.
8. REPEAT    until the matrix / ablation plan is exhausted or the run budget is spent.
```

**Guardrails the agent must implement:**
- A run budget (max GPU/CPU-hours and max configs) confirmed by PULOK — the loop must terminate on budget exhaustion, not on "vibes".
- Failure handling: a failed config is logged with its traceback and skipped; 3 consecutive infra failures pause the loop and notify PULOK.
- No silent config edits: every change is a new CONFIG + registry entry, never a mutation.
- The agent **may not** "fix" poor silhouette by peeking at annotations. If it does, §1 is violated.
- **Docs escalation:** when the agent needs documentation it cannot find or verify itself (API references, dataset provenance, credential setup), it stops and asks PULOK instead of guessing. PULOK is the documentation oracle; the agent never invents docs.
- **Kaggle GPUs:** the free tier attaches ONE GPU per notebook session (typically a T4 or P100, ~30 h/week). The agent plans for one accelerator per run and budgets hours accordingly — it does not assume two GPUs are available to a single run. (Phoenix's own server has no GPU at all: 2 CPUs, 7 GB RAM — CPU-only orchestration and light configs only.)

---

## 6. Staged work plan for the agent

| Stage | Job | Entry criterion |
|---|---|---|
| 0 | Environment + data setup: clone `smolab`, install `requirements.txt`, verify data dirs, run smoke test on synthetic data | PULOK confirms §9 decisions |
| 1 | **Run the Phase 1 matrix** (15 exps, screening seeds) — the pending, already-designed sweep | Stage 0 green |
| 2 | RAUS ranking of Phase 1; pick per-dataset-type winners | Registry complete |
| 3 | Implement architectures **A–D** as smolab module compositions | PULOK picks order (or A→D) |
| 4 | **M0→M8 ablations**, one module at a time | A–D implemented |
| 5 | Final report: winning combination per dataset type, post-hoc ARI table, limitations, next questions | Stages 1–4 complete |

Heavy stages (2–4) on the MacBook Air's CPU will be slow — the existing Colab template exists for exactly this; the agent should structure runs so they can be dispatched to Colab/Kaggle GPUs (§9).

---

## 7. Definition of done

- `registry.jsonl` complete for all planned configs; every entry reproducible from its CONFIG.
- RAUS ranking table per dataset; a named winning combination per dataset type (lymph node RNA+ADT; mouse brain RNA+ATAC).
- Post-hoc ARI/NMI table (clearly marked reporting-only) for the winners.
- A `REPORT.md`: what was tried, what won, by how much, failure modes, and the open questions for the next research cycle.
- No label-leakage incidents unresolved; the firewall assertions all passing in CI (the agent adds a `test_firewall.py`).

---

## 8. Kaggle MCP setup (runs on the Mac — Phoenix cannot install this)

Antigravity supports MCP natively. On **PULOK's Mac**, in Antigravity:

1. Open the **Agent** panel → `...` → **MCP Servers** → **Manage MCP Servers** → **View raw config** (opens `mcp_config.json`, also reachable at `~/.gemini/config/mcp_config.json`).
2. Add a Kaggle server entry (community implementations exist — e.g. `theprathpatel/kaggle-mcp`; verify the repo is the one you intend before installing, these are third-party):
```json
{
  "mcpServers": {
    "kaggle": {
      "command": "python",
      "args": ["-m", "kaggle_mcp.server"],
      "cwd": "/Users/<you>/tools/kaggle-mcp",
      "env": {
        "KAGGLE_USERNAME": "<kaggle username>",
        "KAGGLE_KEY": "<kaggle api key>"
      }
    }
  }
}
```
3. Get the API key at kaggle.com → account → **Create New API Token** (`kaggle.json` → `~/.kaggle/kaggle.json`, `chmod 600`), or use the env vars above.
4. Save, hit **Refresh** in the MCP panel, confirm the Kaggle tools appear.

**Intended uses in this project:** searching/downloading public spatial-omics datasets to expand the pool; dispatching GPU experiment runs to Kaggle notebooks/kernels (one GPU per session — see guardrails); pulling competition-style metadata. It is *not* required for the core 6 datasets (§3.2). Keep API keys out of the repo — env vars or `~/.kaggle/kaggle.json` only.

---

## 9. Decisions PULOK must confirm (agent does not start before these)

1. **Codebase:** extend `smolab` (recommended — firewall, registry, and Phase 1 matrix already exist) vs. fresh repo (agent reimplements the firewall first).
2. **Data locality:** (a) Phoenix packages `smolab/` + `data/` (~600 MB) for download to the Mac, (b) agent re-downloads from sources, (c) Kaggle-only datasets.
3. **E18:** re-supply the download link (the Drive folder ID was truncated — 32 vs 33 chars), or drop E18 from the study.
4. **Compute budget:** how many CPU-hours on the MacBook Air before spilling to Colab/Kaggle GPUs? (Recommend: Stage 1 screening on Kaggle/Colab GPUs; Mac for orchestration and analysis.)
5. **Run budget for the loop:** max configs and max wall-clock per stage — the loop's termination condition.
6. **Model:** confirm the exact model name configured in Antigravity (brief is model-agnostic).

---

## 10. Standing context the agent must read first

- `spatial-omics-lab/README.md` — framework quickstart
- `research/multiview-architectures-20261001.md` — architectures A–D, ablations, RAUS
- `research/lit-sweep-20261001/SWEEP_SYNTHESIS.md` — reusable modules
- `research/spatial-multiomics-ul-report-PART6.md` — revised E3/E7/E9 experiments
- This brief (§1 especially) — the firewall is the law
