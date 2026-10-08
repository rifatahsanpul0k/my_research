# Agent ↔ Phoenix Handoff Protocol (GitHub bridge)

How PULOK's local Antigravity agent and Phoenix (Muse) collaborate through one shared GitHub repo — async, token-cheap, no permission popups.

## Repo layout

```
spatial-omics-auto/
├── briefs/                  # Phoenix → agent (specs, decisions, feedback)
│   └── BRIEF-<n>-<topic>.md
├── runs/<stage>/<exp_id>/   # agent → Phoenix (result folders, synced after every run)
│   ├── registry_entry.json  # the CONFIG + metrics for this run
│   ├── metrics.json         # label-free selection metrics
│   ├── posthoc.json        # ARI/NMI vs annotations — REPORTING ONLY, quarantined
│   ├── plots/               # embedding/cluster visualizations
│   └── run.log
├── reports/
│   └── REPORT-<stage>.md    # agent's analysis per stage (see below)
└── HANDOFF.md               # this file; agent reads it first, every session
```

## The agent's obligations

1. **After every run:** push the full result folder under `runs/<stage>/<exp_id>/`. Never leave results only on Kaggle/Colab.
2. **After every stage:** write `reports/REPORT-<stage>.md` containing:
   - RAUS ranking table (config → silhouette, seed-stability, kNN-overlap, Borda score)
   - metric distributions across seeds (mean ± std, not just the best seed)
   - what failed and why (with tracebacks linked, not pasted in full)
   - the proposed next stage (configs to try, with one-line justifications)
   - unresolved questions for PULOK (docs it couldn't verify — it asks, never invents)
3. **Before every stage:** read the newest file in `briefs/` and acknowledge it in the report.
4. **Firewall, always:** annotation labels never enter selection, tuning, or early stopping. `posthoc.json` is written by a separate code path that the selection code cannot import. Any violation → discard the run, log the incident in the report, continue.
5. **Resume from GitHub, never from scratch:** push every result folder and the updated registry immediately after each run — never batch them. If the Kaggle runtime disconnects, pull the repo, find the last completed experiment in the registry, and rerun only what was interrupted.

## Phoenix's obligations

1. Read **only** `reports/REPORT-<stage>.md` plus `git diff` since the last review — never whole result folders. Raw embeddings and logs stay in the repo; summaries travel.
2. Reply with a new brief in `briefs/` (feedback, next-stage approval, or a stop signal) — the agent does not proceed to a new stage without one, unless PULOK granted a multi-stage run budget in writing.
3. Keep every reply token-lean: tables and decisions, not prose.

## Token discipline (both sides)

- Reports are capped: ranking tables + 5-line analysis per finding. No pasted tracebacks, no raw configs in chat — they live in `registry_entry.json`.
- Plots are PNG files in the repo, referenced by name — never base64'd into messages.
- If a message needs a number, it cites the file and line (`runs/s1/exp03/metrics.json:12`), not the value alone.

## Why this beats a live channel

- Works while the MacBook sleeps; no permission dialogs; full history via git.
- Phoenix can diff instead of re-reading: less tokens, less time, zero repeated work.
