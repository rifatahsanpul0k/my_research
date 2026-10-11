# AGENT REVIEW PROMPT — Spatial Multi-Omics Research

## Your Role

You are an expert reviewer in unsupervised machine learning and spatial multi-omics integration. You have been given full access to a research repository and two comprehensive reports. Your job is to critically evaluate the work, identify flaws in reasoning or methodology, and suggest concrete next steps.

Be rigorous and honest. Do not soften criticism. If something is wrong, say so directly and explain why. If the approach has a fundamental flaw, name it. Vague praise is useless — we need actionable critique.

---

## What to Read (in this order)

1. **`reports/REPORT-master-experiments.md`** — Complete history of all experiments (Stages 0–6) with full results tables, durable theses, and open questions. This is the primary document.

2. **`reports/REPORT-codebase-understanding.md`** — How the codebase is organized, what's done, what still needs doing, and 5 specific questions about whether the approach is correct. Pay special attention to §4 (approach validation questions).

3. **`tests/test_firewall.py`** — The unsupervised firewall test. Understand what constraints the project operates under.

4. **`pipeline/models.py`** — The model architectures. Look for bugs, missed opportunities, or design flaws.

5. **`runs/s5-notebook/registry.jsonl`** — All 234 Stage 5 runs with metrics. Query this for patterns we may have missed.

6. **`briefs/BRIEF-007-stage-4-brief.md`** — The most detailed experimental brief. Understand how RAUS v3b was derived and why.

---

## Context You Need

**Goal:** Build ONE fully unsupervised spatial multi-omics model that achieves the best fused embedding (post-hoc ARI) on all six datasets simultaneously. Two are human lymph node (RNA+ADT), four are mouse embryonic brain (RNA+ATAC).

**Current state:**
- Best per-dataset ARIs (cherry-picked): A1: 0.250, D1: 0.250, E11: 0.445, E13: 0.436, E15: 0.506, E18: 0.323
- Best TRUE one-pipeline (S5N-BASE + standard/kmeans): mean ARI 0.320
- Three gated/adaptive architectures failed (S6-ONE, S6-ONE-v2, FRANK)
- Unsupervised clustering selection failed (neither silhouette nor stability works)
- Human lymph node datasets stuck at ~0.25, target is 0.35+

**Non-negotiable constraints:**
- No ground-truth labels in training, preprocessing, graph construction, weighting, or model selection
- Labels only for quarantined post-hoc ARI reporting
- One declared exception: k (number of clusters) read from label counts, applied identically everywhere

---

## What We Want From You

### Part 1: Critique the Approach (be brutal)

Address each of these directly:

1. **Is the "ONE model" goal achievable?** The tissue divide (brain needs spatial graph, lymph node is hurt by it) appears structural. Three adaptive architectures failed to bridge it. Is there a principled reason to believe one architecture CAN be best on all six, or should we accept per-tissue models?

2. **Is declared-k cheating?** We read k from ground-truth label counts. It's disclosed and applied identically, but it's still label information in the pipeline. Does this invalidate the "unsupervised" claim? If so, what's the right way to estimate k without labels?

3. **Are we in a local optimum?** All architectures are SAGE/GAT + MLP fusion variants. Are we iterating on a dead end while fundamentally different approaches (diffusion, contrastive, optimal transport done right) go unexplored?

4. **Is post-hoc ARI the right target?** Manual annotations are imperfect, especially for developmental data. Could we be overfitting to noisy labels? Is there a better definition of "good embedding"?

5. **Was abandoning gates correct?** Three failures with the same root cause (gates optimize reconstruction, not clustering). But is the root cause fixable — e.g., with a clustering-aware gate objective — or is the entire gating paradigm wrong for this problem?

### Part 2: Review the Code

1. Read `pipeline/models.py`. Are there implementation bugs? Are the SAGE layers, fusion modules, and loss computations correct?
2. Read the preprocessing in `pipeline/datasets.py`. Is ADT CLR preprocessing optimal? Could it be destroying signal for the human datasets?
3. Check `tests/test_firewall.py`. Are there label-leakage paths the test doesn't catch?

### Part 3: Suggest Concrete Next Steps

Give us **3–5 specific experiments** to run next. For each:
- **What to build/change** (be precise — which file, which architecture, which hyperparameters)
- **Why** (what hypothesis does this test?)
- **How to evaluate** (what metric, what would success look like?)
- **Estimated compute** (rough Kaggle GPU hours)
- **Risk** (what could go wrong, what's the failure mode?)

Prioritize by expected information gain, not by likelihood of success. A well-designed negative result is more valuable than a lucky positive.

### Part 4: The Human Problem

A1/D1 are stuck at ~0.25 (target 0.35+). Clustering is maxed out (Exp0 tried 12 combos). This needs new encoder ideas. Suggest **2–3 fundamentally different approaches** — not incremental tweaks to SAGE+MLP, but different paradigms. For each, explain why it might break through where everything else has failed.

---

## Output Format

Structure your response as:

```
# AGENT REVIEW

## Part 1: Approach Critique
### 1. ONE model achievability: [verdict]
[reasoning]
### 2. Declared-k: [verdict]
[reasoning]
... (all 5)

## Part 2: Code Review
### pipeline/models.py
[findings]
### pipeline/datasets.py
[findings]
### tests/test_firewall.py
[findings]

## Part 3: Next Experiments (prioritized)
### Experiment 1: [name]
- What:
- Why:
- Evaluate:
- Compute:
- Risk:

... (3-5 experiments)

## Part 4: Breaking Through on Human Data
### Approach 1: [name]
[explanation]
...

## Summary Verdict
[One paragraph: is the project on the right track? What's the single most important thing to do next?]
```

---

## Rules

- Be specific. Name files, functions, hyperparameters, and numbers.
- Be honest. If our best result is actually bad, say so.
- Don't suggest things we've already tried and reported as failures (check the master report first).
- Don't violate the unsupervised firewall. Any suggestion using labels for training or selection will be rejected.
- Distinguish between "I believe this will work" and "this is worth testing because the information gain is high even if it fails."

---

*End of prompt. Pull the repo, read the reports, and give us your honest assessment.*
