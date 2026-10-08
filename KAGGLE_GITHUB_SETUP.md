# Kaggle → GitHub result push — setup & run pattern

The Kaggle notebook pushes every run's results to GitHub **itself**. No download step,
no manual copying. The local agent on the Mac just `git pull`s. This file contains no
secrets and is safe to commit.

## One-time setup

1. **GitHub token.** github.com → Settings → Developer settings → Personal access tokens →
   **Fine-grained token**: select repo `rifatahsanpul0k/my_research`, permission
   **Contents: Read and write**. Copy the token (shown once).
2. **Kaggle API credentials.** kaggle.com → account icon → **Settings** → scroll to the
   **API** section → **Create New Token**. This downloads `kaggle.json` containing
   `{"username": "<your kaggle username>", "key": "<the key>"}` — copy those two values
   (you do not upload the file anywhere).
3. **Kaggle secrets.** Open the notebook → right sidebar **Add-ons → Secrets** → add three
   secrets, each attached to every notebook that pushes results or saves the dataset:
   - `GITHUB_TOKEN` = the GitHub token from step 1
   - `KAGGLE_USERNAME` = the username from step 2
   - `KAGGLE_KEY` = the key from step 2
4. **Notebook settings:** Internet must be ON.

The secrets live **only** in Kaggle Secrets. They are never printed, never committed.
Note: this is separate from the agent's own Kaggle MCP credentials on the Mac
(`mcp_config.json` env vars) — those authenticate the *agent* to Kaggle; the secrets
above authenticate the *notebook's own code* to Kaggle and GitHub.

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

## One-time data ingestion (Stage 0): Drive → private Kaggle dataset

Run once, in `research_notebook`, before any experiment. Downloads the six dataset
folders from Drive inside Kaggle, then saves them as a private Kaggle dataset that all
later runs mount.

```python
import json, os, subprocess
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()

# 1. Authenticate the Kaggle API inside this notebook
kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)
with open(os.path.join(kaggle_dir, "kaggle.json"), "w") as f:
    json.dump({"username": secrets.get_secret("KAGGLE_USERNAME"),
               "key": secrets.get_secret("KAGGLE_KEY")}, f)
os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)

# 2. Download the Drive folders (links injected by the agent from its local DATASETS.md)
#    pip install gdown  (once, first cell)
#    !gdown --folder <drive-folder-url> -O /kaggle/working/data/<DATASET_ID>
#    ... repeat for A1, D1, E11, E13, E15, E18

# 3. Create the private dataset from the download
#    !kaggle datasets create -p /kaggle/working/data \
#        --dir-mode zip -t "spatial-multiomics-6datasets-private"
#    then set it to private if prompted, or create via the Kaggle UI.
```

After this succeeds, later runs only add the dataset as a **data source** to the
notebook (right sidebar → Add data → your private dataset) — no re-download, ever.

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
