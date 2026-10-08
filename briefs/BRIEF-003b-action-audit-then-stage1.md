# BRIEF-003b — ACTION: do the train_model audit, then start Stage 1

**This is a direct order. Read BRIEF-003 first if you haven't.**

## 1. Do pre-flight fix #2 NOW

Audit the full pipeline code (`train_model` and everything it calls):

- `ground_truth` may enter **only** the post-hoc logging path — the code that writes
  `posthoc.json` (ARI/NMI vs annotations, reporting-only).
- Refactor so the training loop, loss computation, early stopping, and selection code
  **cannot import or receive** `ground_truth`. If it currently takes it as an argument,
  remove the argument or split the function.
- Add `test_firewall.py`: an automated assertion that fails if labels ever reach a
  selection-path call. It must pass before any Stage 1 run.
- Commit the fix.

The k question is settled (see BRIEF-003 §Pre-flight fix #1 — declared exception):
read k once from the dataset registry as a declared constant. Do not relitigate it.

## 2. Then start Stage 1 immediately

No further brief is needed — BRIEF-003 **is** the Stage 1 approval. Execute it:
mount `pulokpulok/spatial-multiomics-6datasets-private`, run the 15-experiment /
50-run screening matrix (≥3 seeds, `skip_done=True`), push after every run,
resume-only-reruns on disconnect, RAUS selection, post-hoc ARI quarantined.

## 3. Report per the loop

`reports/REPORT-s1.md` when the matrix or the weekly GPU quota is exhausted,
whichever comes first. Check the accelerator quota before each batch — 30h/week
is a hard cap. Do not start Stage 2 without BRIEF-004.
