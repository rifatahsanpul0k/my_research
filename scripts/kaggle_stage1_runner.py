"""
Standalone Stage 1 Kaggle Runner Script.
Designed to run inside Kaggle Notebook with GPU and Internet enabled.
Pushes results to GitHub after EVERY run.
"""

import os
import sys
import json
import time
import base64
import random
import subprocess
from pathlib import Path
from typing import Dict, Any, List

# Kaggle secrets
try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    TOKEN = secrets.get_secret("GITHUB_TOKEN")
except Exception:
    TOKEN = os.environ.get("GITHUB_TOKEN", "")

REPO_DIR = "/kaggle/working/my_research"
auth_b64 = base64.b64encode(f"x-access-token:{TOKEN}".encode()).decode() if TOKEN else ""
GIT_CMD = ["git", "-c", f"http.extraHeader=Authorization: Basic {auth_b64}"] if auth_b64 else ["git"]


def sh(*args):
    return subprocess.run(list(args), check=True, cwd=REPO_DIR, capture_output=True, text=True)


def git_push_with_retry(commit_msg: str, files_to_add: List[str] = None):
    if not TOKEN:
        print("Warning: No GITHUB_TOKEN available, skipping git push.")
        return
    for attempt in range(3):
        try:
            sh(*GIT_CMD, "pull", "--rebase", "origin", "main")
            if files_to_add:
                sh("git", "add", *files_to_add)
            else:
                sh("git", "add", "-A")
            
            # Check if there are changes to commit
            status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_DIR, capture_output=True, text=True)
            if not status.stdout.strip():
                print("No git changes to commit.")
                return
            
            sh("git", "commit", "-m", commit_msg)
            sh(*GIT_CMD, "push", "origin", "main")
            print(f"Git push successful: {commit_msg}")
            return
        except Exception as e:
            print(f"Git push attempt {attempt + 1} failed: {e}. Retrying in 5s...")
            time.sleep(5)
    print("Warning: Git push failed after 3 attempts.")


def setup_repository():
    print("=== Step 1: Setting up GitHub repository ===")
    if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
        cmd = GIT_CMD + ["clone", "https://github.com/rifatahsanpul0k/my_research.git", REPO_DIR]
        subprocess.run(cmd, check=True)
        sh("git", "config", "user.name", "kaggle-runner")
        sh("git", "config", "user.email", "kaggle-runner@local")
    else:
        sh(*GIT_CMD, "pull", "--rebase", "origin", "main")

    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)


def run_firewall_tests():
    print("=== Step 2: Running Unsupervised Firewall Test Suite ===")
    test_file = os.path.join(REPO_DIR, "tests", "test_firewall.py")
    res = subprocess.run([sys.executable, test_file], cwd=REPO_DIR, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        raise RuntimeError("UNSUPERVISED FIREWALL TEST FAILED! ABORTING RUN.")
    print("UNSUPERVISED FIREWALL TESTS PASSED (100% compliant).")


def locate_dataset_root() -> str:
    print("=== Step 3: Locating Mounted Kaggle Dataset ===")
    candidates = [
        "/kaggle/input/spatial-multiomics-6datasets-private",
        "/kaggle/input/datasets/pulokpulok/spatial-multiomics-6datasets-private",
        "/kaggle/input/spatial-multiomics-6datasets-private/data",
        os.path.join(REPO_DIR, "data"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            # check subfolders
            sub = os.listdir(c)
            if any("10x_human_lymph_node_A1" in s for s in sub):
                print(f"Found dataset root at: {c}")
                return c
            # check nested
            for s in sub:
                nested = os.path.join(c, s)
                if os.path.isdir(nested) and any("10x_human_lymph_node_A1" in x for x in os.listdir(nested)):
                    print(f"Found dataset root at: {nested}")
                    return nested
    raise FileNotFoundError(f"Could not locate spatial-multiomics-6datasets-private among {candidates}")


if __name__ == "__main__":
    setup_repository()
    run_firewall_tests()
    dataset_root = locate_dataset_root()
    print("Ready to run Stage 1 matrix.")
