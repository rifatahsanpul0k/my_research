# Kaggle → GitHub result push — setup & run pattern

The Kaggle notebook pushes every run's results to GitHub **itself**. No download step,
no manual copying. The local agent on the Mac just `git pull`s. This file contains no
secrets and is safe to commit.

## One-time setup

1. **GitHub token.** github.com → Settings → Developer settings → Personal access tokens →
   **Fine-grained token**: select repo `rifatahsanpul0k/my_research`, permission
   **Contents: Read and write**. Copy the token (shown once).
2. **Kaggle secret.** Open the notebook → right sidebar **Add-ons → Secrets** → add secret
   named `GITHUB_TOKEN`, value = the token. Attach it to every notebook that pushes.
3. **Notebook settings:** Internet must be ON.

The token lives **only** in Kaggle Secrets. It is never printed, never committed.

## Notebook setup cell (paste once)

```python
from kaggle_secrets import UserSecretsClient
import os, subprocess

TOKEN = UserSecretsClient().get_secret("GITHUB_TOKEN")
REPO = "/kaggle/working/my_research"
GIT = ["git", "-c", "http.extraHeader=Authorization: bearer " + TOKEN]

def sh(*args):
    subprocess.run(list(args), check=True, cwd=REPO)

if not os.path.isdir(os.path.join(REPO, ".git")):
    subprocess.run(GIT + ["clone", "https://github.com/rifatahsanpul0k/my_research.git", REPO],
                   check=True)
    sh("git", "config", "user.name", "kaggle-runner")
    sh("git", "config", "user.email", "kaggle-runner@local")
else:
    sh(*GIT, "pull", "--rebase", "origin", "main")
```

## End of every run (not end of notebook — every run)

```python
# results written to f"{REPO}/runs/<stage>/<exp_id>/" and registry.jsonl appended
sh(*GIT, "pull", "--rebase", "origin", "main")  # pick up the agent's reports first
sh("git", "add", "runs/<stage>/<exp_id>", "registry.jsonl")
sh("git", "commit", "-m", f"results: <stage>/<exp_id>")
sh(*GIT, "push", "origin", "main")
print("pushed <stage>/<exp_id>")
```

## Rules

- **Push after EVERY run**, never batched at the end of the notebook. A disconnect
  mid-notebook then loses nothing completed — this is the resume guarantee
  (BRIEF-001 §5: rerun only the interrupted experiment).
- Never `print(TOKEN)`. Never write it to a file. If a traceback might contain it,
  scrub before logging.
- Result folders stay lean: `metrics.json`, `posthoc.json`, `plots/`, `run.log`.
  Raw `.h5ad`/datasets are never committed (`.gitignore`).
- Push conflict (`non-fast-forward`)? `pull --rebase` and retry once. Still failing →
  leave results in `/kaggle/working`, report it, **never force-push**.

## Smoke test (Stage 0)

Before any real run: write `runs/_smoke/ping.json` (`{"ok": true}`), push with the
pattern above, confirm it appears on github.com, then remove it. If the smoke test
fails, stop — do not start Stage 1.

## The local agent's side

`git pull` on the Mac → results appear under `runs/` → analyze → write
`reports/REPORT-<stage>.md` → commit + push → next stage per `briefs/`.
