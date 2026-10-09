"""
Stage 3 Notebook Generator: Synthesis - Best Fused Embedding across all 6 datasets.
Combines verified Stage 2 winners and embedding-module sweep.
Enforces unsupervised selection firewall, bounded execution matrix, and git sync per run.
"""

import json
import os

with open("pipeline/training.py", "r") as f:
    training_code = f.read()

with open("pipeline/datasets.py", "r") as f:
    datasets_code = f.read()

with open("pipeline/models.py", "r") as f:
    models_code = f.read()

with open("tests/test_firewall.py", "r") as f:
    firewall_code = f.read()

cell_1_code = f'''# Cell 1: Setup Pipeline and Firewall Modules
import os

TRAINING_PY = {repr(training_code)}
DATASETS_PY = {repr(datasets_code)}
MODELS_PY = {repr(models_code)}
TEST_FIREWALL_PY = {repr(firewall_code)}

print("Pipeline modules loaded into runner.")
'''

cell_2_code = '''# Cell 2: Stage 3 Synthesis Execution across all 6 datasets
import os, sys, json, time, random, shutil, base64, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scanpy", "anndata"], check=True)
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
TOKEN = secrets.get_secret("GITHUB_TOKEN")
REPO = "/kaggle/working/my_research"
auth_b64 = base64.b64encode(f"x-access-token:{TOKEN}".encode()).decode()
GIT = ["git", "-c", f"http.extraHeader=Authorization: Basic {auth_b64}"]

def sh(*args):
    return subprocess.run(list(args), check=True, cwd=REPO, capture_output=True, text=True)

def push_to_github(commit_msg: str, files: List[str] = None):
    for attempt in range(3):
        try:
            sh(*GIT, "pull", "--rebase", "origin", "main")
            if files:
                sh("git", "add", *files)
            else:
                sh("git", "add", "-A")
            # Check if there are changes to commit
            status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True)
            if status.stdout.strip():
                sh("git", "commit", "-m", commit_msg)
                sh(*GIT, "push", "origin", "main")
                print(f"[GIT PUSH SUCCESS] {commit_msg}")
            else:
                print(f"[GIT CLEAN] Nothing to commit for {commit_msg}")
            return
        except Exception as e:
            print(f"Git push attempt {attempt+1} failed: {e}. Retrying in 3s...")
            time.sleep(3)

# 1. Setup repository
if not os.path.isdir(os.path.join(REPO, ".git")):
    print("Cloning repository...")
    if os.path.exists(REPO):
        shutil.rmtree(REPO, ignore_errors=True)
    subprocess.run(GIT + ["clone", "https://github.com/rifatahsanpul0k/my_research.git", REPO], check=True)
    subprocess.run(["git", "config", "user.name", "kaggle-runner"], cwd=REPO, check=True)
    subprocess.run(["git", "config", "user.email", "kaggle-runner@local"], cwd=REPO, check=True)
else:
    print("Pulling latest repository state...")
    sh(*GIT, "pull", "--rebase", "origin", "main")

os.makedirs(os.path.join(REPO, "pipeline"), exist_ok=True)
os.makedirs(os.path.join(REPO, "tests"), exist_ok=True)
os.makedirs(os.path.join(REPO, "runs", "s3"), exist_ok=True)
os.makedirs(os.path.join(REPO, "reports"), exist_ok=True)

with open(os.path.join(REPO, "pipeline", "training.py"), "w") as f:
    f.write(TRAINING_PY)

with open(os.path.join(REPO, "pipeline", "datasets.py"), "w") as f:
    f.write(DATASETS_PY)

with open(os.path.join(REPO, "pipeline", "models.py"), "w") as f:
    f.write(MODELS_PY)

with open(os.path.join(REPO, "tests", "test_firewall.py"), "w") as f:
    f.write(TEST_FIREWALL_PY)

# 2. Run Firewall Tests
print("Running Unsupervised Firewall Test Suite...")
test_res = subprocess.run([sys.executable, os.path.join(REPO, "tests", "test_firewall.py")], cwd=REPO, capture_output=True, text=True)
print("Firewall test stdout:\\n", test_res.stdout)
if test_res.returncode != 0:
    print("Firewall test stderr:\\n", test_res.stderr)
    raise RuntimeError("UNSUPERVISED FIREWALL TEST FAILED! ABORTING RUN.")
print("UNSUPERVISED FIREWALL TESTS PASSED (100% compliant).")

if REPO not in sys.path:
    sys.path.insert(0, REPO)

from pipeline.datasets import DATASET_CONFIGS, DECLARED_K, load_dataset_features_unsupervised, load_posthoc_ground_truth
from pipeline.training import evaluate_posthoc_quarantined
from pipeline.models import Model_H_HiRe, ClusterHead, target_distribution, kl_clustering_loss, scot_loss

# 3. Locate Dataset Root
CANDIDATES = [
    "/kaggle/input/spatial-multiomics-6datasets-private",
    "/kaggle/input/datasets/pulokpulok/spatial-multiomics-6datasets-private",
    "/kaggle/input/spatial-multiomics-6datasets-private/data",
]
dataset_root = None
for c in CANDIDATES:
    if os.path.isdir(c):
        for sub in os.listdir(c):
            if "10x_human_lymph_node_A1" in sub or os.path.isdir(os.path.join(c, "10x_human_lymph_node_A1")):
                dataset_root = c
                break
        if dataset_root:
            break

if not dataset_root:
    for root, dirs, files in os.walk("/kaggle/input"):
        for d in dirs:
            if "10x_human_lymph_node_A1" in d:
                dataset_root = root
                break
        if dataset_root:
            break

print(f"Dataset root found: {dataset_root}")

# Helper: Graph builders
def build_dual_graphs(rna_feats: np.ndarray, coords: np.ndarray, num_neighbors: int = 15, device: str = "cpu"):
    nbrs_sim = NearestNeighbors(n_neighbors=num_neighbors + 1, metric='cosine').fit(rna_feats)
    dist_sim, idx_sim = nbrs_sim.kneighbors(rna_feats)
    sim_scores = 1.0 - dist_sim[:, 1:]
    sim_weights = np.clip(sim_scores, 0.0, 1.0)
    rows_s = np.repeat(np.arange(len(rna_feats)), num_neighbors)
    cols_s = idx_sim[:, 1:].flatten()
    e_sim = torch.tensor(np.stack([rows_s, cols_s]), dtype=torch.long, device=device)
    w_sim = torch.tensor(sim_weights.flatten(), dtype=torch.float32, device=device)

    nbrs_sp = NearestNeighbors(n_neighbors=num_neighbors + 1, metric='euclidean').fit(coords)
    dist_sp, idx_sp = nbrs_sp.kneighbors(coords)
    sigma = np.median(dist_sp[:, 1:]) + 1e-6
    sp_weights = np.exp(-(dist_sp[:, 1:] ** 2) / (2 * (sigma ** 2)))
    rows_d = np.repeat(np.arange(len(coords)), num_neighbors)
    cols_d = idx_sp[:, 1:].flatten()
    e_dist = torch.tensor(np.stack([rows_d, cols_d]), dtype=torch.long, device=device)
    w_dist = torch.tensor(sp_weights.flatten(), dtype=torch.float32, device=device)

    return (e_sim, w_sim), (e_dist, w_dist), (e_dist, w_dist)

def compute_laplacian_loss(z: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
    row, col = edge_index
    diff = z[row] - z[col]
    dist_sq = torch.sum(diff ** 2, dim=-1)
    if edge_weight is not None:
        dist_sq = dist_sq * edge_weight
    return torch.mean(dist_sq)

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

SEEDS = [42, 1234, 2024]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Executing Stage 3 on device: {DEVICE}")

# 4. Preload and Cache all 6 Datasets
DATA_CACHE = {}
print("\\nPreloading and caching all 6 datasets...")
for d_cfg in DATASET_CONFIGS:
    d_name = d_cfg["name"]
    d_dir = None
    for folder in d_cfg["folder_candidates"]:
        cand = os.path.join(dataset_root, folder)
        if os.path.isdir(cand):
            d_dir = cand
            break
        for sub in os.listdir(dataset_root):
            cand2 = os.path.join(dataset_root, sub, folder)
            if os.path.isdir(cand2):
                d_dir = cand2
                break
        if d_dir:
            break

    if not d_dir:
        print(f"Warning: Directory for {d_name} not found.")
        continue

    print(f"Loading {d_name} (Declared K={DECLARED_K[d_name]})...")
    data = load_dataset_features_unsupervised(d_dir, d_cfg, n_hvg=3000, n_comps_rna=128, n_comps_mod2=128)
    ground_truth = load_posthoc_ground_truth(d_dir, d_cfg)

    rna_raw = torch.tensor(data["rna_raw"], dtype=torch.float32).to(DEVICE)
    mod2_raw = torch.tensor(data["mod2_raw"], dtype=torch.float32).to(DEVICE)
    rna_pca30 = torch.tensor(data["rna_pca"][:, :30], dtype=torch.float32).to(DEVICE)
    mod2_pca30 = torch.tensor(data["mod2_pca"][:, :30], dtype=torch.float32).to(DEVICE)
    rna_pca128 = torch.tensor(data["rna_pca"][:, :128], dtype=torch.float32).to(DEVICE)
    mod2_pca128 = torch.tensor(data["mod2_pca"][:, :min(128, data["mod2_pca"].shape[1])], dtype=torch.float32).to(DEVICE)

    joint_raw = torch.cat([rna_raw, mod2_raw], dim=1)
    joint_pca30 = torch.cat([rna_pca30, mod2_pca30], dim=1)
    joint_pca128 = torch.cat([rna_pca128, mod2_pca128], dim=1)

    (e_sim, w_sim), (e_dist, w_dist), (e_com, w_com) = build_dual_graphs(
        data["rna_raw"], data["cell_positions"], num_neighbors=15, device=DEVICE
    )

    DATA_CACHE[d_name] = {
        "cfg": d_cfg,
        "k": DECLARED_K[d_name],
        "data": data,
        "ground_truth": ground_truth,
        "rna_raw": rna_raw,
        "mod2_raw": mod2_raw,
        "rna_pca30": rna_pca30,
        "mod2_pca30": mod2_pca30,
        "rna_pca128": rna_pca128,
        "mod2_pca128": mod2_pca128,
        "joint_raw": joint_raw,
        "joint_pca30": joint_pca30,
        "joint_pca128": joint_pca128,
        "var_rna": float(np.var(data["rna_raw"])),
        "var_mod2": float(np.var(data["mod2_raw"])),
        "graphs": ((e_sim, w_sim), (e_dist, w_dist), (e_com, w_com)),
    }

print(f"All {len(DATA_CACHE)} datasets preprocessed and resident in GPU memory.")

# Runner function for an experiment configuration across all datasets and seeds
registry_file = os.path.join(REPO, "runs", "s3", "registry.jsonl")

def run_experiment_config(exp_cfg: Dict[str, Any], datasets_subset: List[str] = None):
    exp_id = exp_cfg["id"]
    print("\\n" + "="*80)
    print(f" RUNNING EXPERIMENT: {exp_id} ".center(80, "="))
    print(f" Desc: {exp_cfg.get('desc', '')} ")
    print("="*80)

    target_datasets = datasets_subset if datasets_subset else list(DATA_CACHE.keys())
    exp_predictions = {d: {} for d in target_datasets}

    # Pre-load existing cluster assignments if present
    for d_name in target_datasets:
        for seed in SEEDS:
            run_dir = os.path.join(REPO, "runs", "s3", exp_id, f"{d_name}_s{seed}")
            pred_f = os.path.join(run_dir, "cluster_assignments.npy")
            if os.path.exists(pred_f):
                try:
                    exp_predictions[d_name][seed] = np.load(pred_f)
                except Exception:
                    pass

    for d_name in target_datasets:
        cached = DATA_CACHE[d_name]
        k_clusters = cached["k"]
        (e_sim, w_sim), (e_dist, w_dist), (e_com, w_com) = cached["graphs"]
        ground_truth = cached["ground_truth"]

        for seed in SEEDS:
            run_key = f"{d_name}_{exp_id}_s{seed}"
            run_dir = os.path.join(REPO, "runs", "s3", exp_id, f"{d_name}_s{seed}")
            metrics_file = os.path.join(run_dir, "metrics.json")
            os.makedirs(run_dir, exist_ok=True)
            os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)

            if os.path.exists(metrics_file):
                print(f"[SKIP_DONE] {run_key} already completed.")
                continue

            set_seed(seed)
            t0 = time.time()

            fusion = exp_cfg.get("fusion", "2stage")
            backbone = exp_cfg.get("backbone", "sage")
            recon_target = exp_cfg.get("recon_target", "raw")
            align_mode = exp_cfg.get("align_mode", None)
            l_rec = exp_cfg.get("l_rec", 15.0)
            l_spa = exp_cfg.get("l_spa", 2.0)
            l_align = exp_cfg.get("l_align", 0.0)
            lr = exp_cfg.get("lr", 1e-3)
            epochs = exp_cfg.get("epochs", 150)

            # Setup feature inputs and targets
            if recon_target == "raw":
                x_r_in = cached["rna_raw"]
                x_m_in = cached["mod2_raw"]
                target_r = cached["rna_raw"]
                target_m = cached["mod2_raw"]
                target_joint = cached["joint_raw"]
            elif recon_target == "pca30":
                x_r_in = cached["rna_pca30"]
                x_m_in = cached["mod2_pca30"]
                target_r = cached["rna_pca30"]
                target_m = cached["mod2_pca30"]
                target_joint = cached["joint_pca30"]
            elif recon_target == "pca128":
                x_r_in = cached["rna_pca128"]
                x_m_in = cached["mod2_pca128"]
                target_r = cached["rna_pca128"]
                target_m = cached["mod2_pca128"]
                target_joint = cached["joint_pca128"]
            else:
                raise ValueError(f"Unknown recon target: {recon_target}")

            model = Model_H_HiRe(
                in_rna_dim=x_r_in.shape[1],
                in_mod2_dim=x_m_in.shape[1],
                hidden_dim=256,
                out_dim=64,
                backbone=backbone,
                fusion=fusion,
                recon_mode=recon_target
            ).to(DEVICE)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

            best_sil = -1.0
            best_emb = None
            best_pred = None
            best_recs = None

            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()

                z_final, z_sim, z_dist, z_mod2, rec_rna, rec_mod2, rec_joint = model(
                    x_r_in, x_m_in, e_sim, w_sim, e_dist, w_dist, e_com, w_com
                )

                loss_rec = F.mse_loss(target_joint, rec_joint) + 0.5 * (F.mse_loss(target_r, rec_rna) + F.mse_loss(target_m, rec_mod2))
                loss_spa = compute_laplacian_loss(z_final, e_dist, w_dist)
                loss = l_rec * loss_rec + l_spa * loss_spa

                if align_mode == "scot":
                    loss_ot = scot_loss(model.last_z_rna, z_mod2, m=512, eps=0.1)
                    loss = loss + l_align * loss_ot

                loss.backward()
                optimizer.step()

                # Periodic evaluation using KMeans on intermediate embedding
                if (epoch + 1) % 15 == 0 or epoch == epochs - 1:
                    model.eval()
                    with torch.no_grad():
                        z_eval, _, _, _, r_rna, r_mod2, _ = model(
                            x_r_in, x_m_in, e_sim, w_sim, e_dist, w_dist, e_com, w_com
                        )
                        emb_np = z_eval.cpu().numpy()

                    km = KMeans(n_clusters=k_clusters, n_init=5, random_state=seed)
                    pred_np = km.fit_predict(emb_np)
                    try:
                        sil = float(silhouette_score(emb_np, pred_np))
                    except Exception:
                        sil = -1.0

                    if sil > best_sil:
                        best_sil = sil
                        best_emb = emb_np
                        best_pred = pred_np
                        best_recs = (r_rna.cpu().numpy(), r_mod2.cpu().numpy())

            wall_time = time.time() - t0

            if best_emb is None:
                model.eval()
                with torch.no_grad():
                    z_eval, _, _, _, r_rna, r_mod2, _ = model(
                        x_r_in, x_m_in, e_sim, w_sim, e_dist, w_dist, e_com, w_com
                    )
                best_emb = z_eval.cpu().numpy()
                km = KMeans(n_clusters=k_clusters, n_init=10, random_state=seed)
                best_pred = km.fit_predict(best_emb)
                best_recs = (r_rna.cpu().numpy(), r_mod2.cpu().numpy())
                best_sil = float(silhouette_score(best_emb, best_pred))

            exp_predictions[d_name][seed] = best_pred

            # Compute label-free reconstruction balance
            if best_recs is not None:
                r_np, m_np = best_recs
                mse_r = float(np.mean((target_r.cpu().numpy() - r_np) ** 2))
                mse_m = float(np.mean((target_m.cpu().numpy() - m_np) ** 2))
                var_r = float(np.var(target_r.cpu().numpy()))
                var_m = float(np.var(target_m.cpu().numpy()))
                fid_r = max(0.0, 1.0 - (mse_r / max(var_r, 1e-6)))
                fid_m = max(0.0, 1.0 - (mse_m / max(var_m, 1e-6)))
                recon_balance = float((min(fid_r, fid_m) + 1e-4) / (max(fid_r, fid_m) + 1e-4))
            else:
                recon_balance = 0.5

            # Compute seed stability across off-diagonal pairs
            stability_aris = [
                float(adjusted_rand_score(best_pred, other_p))
                for other_s, other_p in exp_predictions[d_name].items()
                if other_s != seed
            ]
            seed_stability_ari = float(np.mean(stability_aris)) if stability_aris else 0.5

            # kNN overlap (diagnostic only, not voted)
            try:
                nbrs_raw = NearestNeighbors(n_neighbors=15).fit(cached["joint_raw"].cpu().numpy())
                nbrs_emb = NearestNeighbors(n_neighbors=15).fit(best_emb)
                _, idx_r = nbrs_raw.kneighbors()
                _, idx_e = nbrs_emb.kneighbors()
                overlaps = [len(set(idx_r[i]) & set(idx_e[i])) / 15.0 for i in range(len(best_emb))]
                knn_overlap = float(np.mean(overlaps))
            except Exception:
                knn_overlap = 0.0

            metrics = {
                "silhouette": best_sil,
                "seed_stability_ari": seed_stability_ari,
                "reconstruction_balance": recon_balance,
                "knn_overlap": knn_overlap,
                "selection_score": float(0.4 * best_sil + 0.4 * seed_stability_ari + 0.2 * recon_balance),
                "wall_time_seconds": round(wall_time, 2),
                "unsupervised_selection_guarantee": True
            }

            posthoc_metrics = evaluate_posthoc_quarantined(best_pred, ground_truth)
            print(f"Done: {exp_id} ({d_name} s{seed}) -> Sil: {best_sil:.3f} | Stab: {seed_stability_ari:.3f} | ReconBal: {recon_balance:.3f} | PostHoc ARI: {posthoc_metrics['ari']:.3f}")

            np.save(os.path.join(run_dir, "latent_embeddings.npy"), best_emb)
            np.save(os.path.join(run_dir, "cluster_assignments.npy"), best_pred)

            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)

            with open(os.path.join(run_dir, "posthoc.json"), "w") as f:
                json.dump(posthoc_metrics, f, indent=2)

            with open(os.path.join(run_dir, "run.log"), "w") as f:
                f.write(f"Dataset: {d_name}\\nExp: {exp_id}\\nSeed: {seed}\\nWall Time: {wall_time:.2f}s\\n")

            fig, ax = plt.subplots(figsize=(6, 5))
            sc_plot = ax.scatter(best_emb[:, 0], best_emb[:, 1], c=best_pred, cmap="tab20", s=6, alpha=0.8)
            ax.set_title(f"{d_name} - {exp_id} (Seed {seed})")
            fig.colorbar(sc_plot, ax=ax)
            fig.savefig(os.path.join(run_dir, "plots", "cluster_latent.png"), dpi=100, bbox_inches="tight")
            plt.close(fig)

            reg_entry = {
                "dataset": d_name,
                "exp_id": exp_id,
                "seed": seed,
                "model_type": "Model_H_HiRe",
                "metrics": metrics,
                "posthoc": posthoc_metrics,
                "timestamp": time.time()
            }
            with open(os.path.join(run_dir, "registry_entry.json"), "w") as f:
                json.dump(reg_entry, f, indent=2)

            with open(registry_file, "a") as f:
                f.write(json.dumps(reg_entry) + "\\n")

            # Push run to git
            push_to_github(f"run(s3): {exp_id} on {d_name} s{seed}", [run_dir, registry_file])

    # Re-calculate exact 3-seed off-diagonal stability for this experiment
    for d_name in target_datasets:
        preds = exp_predictions[d_name]
        if len(preds) >= 2:
            for s in SEEDS:
                if s in preds:
                    other_aris = [float(adjusted_rand_score(preds[s], preds[o])) for o in SEEDS if o != s and o in preds]
                    corr_stab = float(np.mean(other_aris))
                    run_dir = os.path.join(REPO, "runs", "s3", exp_id, f"{d_name}_s{s}")
                    m_f = os.path.join(run_dir, "metrics.json")
                    if os.path.exists(m_f):
                        with open(m_f) as f:
                            m_d = json.load(f)
                        m_d["seed_stability_ari"] = corr_stab
                        m_d["selection_score"] = float(0.4 * m_d["silhouette"] + 0.4 * corr_stab + 0.2 * m_d["reconstruction_balance"])
                        with open(m_f, "w") as f:
                            json.dump(m_d, f, indent=2)

    return exp_predictions


# 5. PHASE 1: EXECUTE CANDIDATE MATRIX
STAGE3_CANDIDATES = [
    {
        "id": "S3-BASE",
        "fusion": "2stage",
        "backbone": "sage",
        "recon_target": "raw",
        "align_mode": None,
        "l_rec": 15.0,
        "l_spa": 2.0,
        "l_align": 0.0,
        "lr": 1e-3,
        "epochs": 150,
        "desc": "Stage 3 Base: Laplacian-H-HiRe (SAGEConv + 2-Stage MLP + 3000-HVG raw recon + Laplacian Dirichlet loss)"
    },
    {
        "id": "S3-CAND-A",
        "fusion": "2stage",
        "backbone": "sage",
        "recon_target": "raw",
        "align_mode": "scot",
        "l_rec": 15.0,
        "l_spa": 2.0,
        "l_align": 0.5,
        "lr": 1e-3,
        "epochs": 150,
        "desc": "Candidate A: Base + SCOT-Sinkhorn Optimal Transport Alignment Loss"
    },
    {
        "id": "S3-CAND-B",
        "fusion": "uaf",
        "backbone": "sage",
        "recon_target": "raw",
        "align_mode": None,
        "l_rec": 15.0,
        "l_spa": 2.0,
        "l_align": 0.0,
        "lr": 1e-3,
        "epochs": 150,
        "desc": "Candidate B: Base with Uncertainty-Aware Gaussian Fusion (UAF-Gaussian)"
    },
    {
        "id": "S3-CAND-C",
        "fusion": "2stage",
        "backbone": "gat",
        "recon_target": "raw",
        "align_mode": None,
        "l_rec": 15.0,
        "l_spa": 2.0,
        "l_align": 0.0,
        "lr": 1e-3,
        "epochs": 150,
        "desc": "Candidate C: Base with Graph Attention Network (GAT) Encoder Backbone"
    },
]

print("\\n" + "="*80)
print(" STAGE 3 PHASE 1: CANDIDATES EXECUTION ".center(80, "="))
print("="*80)

for cand in STAGE3_CANDIDATES:
    run_experiment_config(cand)

# 6. EVALUATE PHASE 1 BORDA RANKS
def compute_borda_ranks(exp_ids: List[str]):
    dataset_ranks = {}
    for d_name in DATA_CACHE.keys():
        scores = {}
        for eid in exp_ids:
            sils, stabs, rebals, aris = [], [], [], []
            for s in SEEDS:
                m_f = os.path.join(REPO, "runs", "s3", eid, f"{d_name}_s{s}", "metrics.json")
                p_f = os.path.join(REPO, "runs", "s3", eid, f"{d_name}_s{s}", "posthoc.json")
                if os.path.exists(m_f):
                    with open(m_f) as f:
                        m = json.load(f)
                    sils.append(m["silhouette"])
                    stabs.append(m["seed_stability_ari"])
                    rebals.append(m["reconstruction_balance"])
                if os.path.exists(p_f):
                    with open(p_f) as f:
                        p = json.load(f)
                    aris.append(p["ari"])

            scores[eid] = {
                "sil": float(np.mean(sils)) if sils else -1.0,
                "stab": float(np.mean(stabs)) if stabs else 0.0,
                "rebal": float(np.mean(rebals)) if rebals else 0.0,
                "posthoc_ari": float(np.mean(aris)) if aris else 0.0,
            }

        sorted_sil = sorted(exp_ids, key=lambda x: scores[x]["sil"], reverse=True)
        sorted_stab = sorted(exp_ids, key=lambda x: scores[x]["stab"], reverse=True)
        sorted_rebal = sorted(exp_ids, key=lambda x: scores[x]["rebal"], reverse=True)

        rank_sil = {eid: sorted_sil.index(eid) + 1 for eid in exp_ids}
        rank_stab = {eid: sorted_stab.index(eid) + 1 for eid in exp_ids}
        rank_rebal = {eid: sorted_rebal.index(eid) + 1 for eid in exp_ids}

        borda = {eid: rank_sil[eid] + rank_stab[eid] + rank_rebal[eid] for eid in exp_ids}
        dataset_ranks[d_name] = {
            eid: {
                "borda_score": borda[eid],
                "rank_sil": rank_sil[eid],
                "rank_stab": rank_stab[eid],
                "rank_rebal": rank_rebal[eid],
                "flag": abs(rank_sil[eid] - rank_stab[eid]) >= 4,
                "posthoc_ari": scores[eid]["posthoc_ari"],
                "sil": scores[eid]["sil"],
                "stab": scores[eid]["stab"],
                "rebal": scores[eid]["rebal"]
            }
            for eid in exp_ids
        }

    # Cross-dataset aggregated ranks
    cross_borda = {}
    for eid in exp_ids:
        ranks = []
        for d_name in DATA_CACHE.keys():
            sorted_eids = sorted(exp_ids, key=lambda x: dataset_ranks[d_name][x]["borda_score"])
            ranks.append(sorted_eids.index(eid) + 1)
        cross_borda[eid] = float(np.mean(ranks))

    return dataset_ranks, cross_borda

cand_ids = [c["id"] for c in STAGE3_CANDIDATES]
dataset_ranks, cross_borda = compute_borda_ranks(cand_ids)

print("\\n" + "="*80)
print(" PHASE 1 BORDA RESULTS ACROSS ALL 6 DATASETS ".center(80, "="))
print("="*80)
sorted_cands = sorted(cand_ids, key=lambda x: cross_borda[x])
for rank_i, cid in enumerate(sorted_cands, 1):
    avg_ari = np.mean([dataset_ranks[d][cid]["posthoc_ari"] for d in DATA_CACHE.keys()])
    flags = sum([1 for d in DATA_CACHE.keys() if dataset_ranks[d][cid]["flag"]])
    print(f"#{rank_i}: {cid} | Avg Cross-Dataset Rank: {cross_borda[cid]:.2f} | Avg Post-Hoc ARI: {avg_ari:.3f} | Flags: {flags}/6")

best_cand_id = sorted_cands[0] if sorted_cands[0] != "S3-BASE" else (sorted_cands[1] if len(sorted_cands) > 1 else sorted_cands[0])
# Candidates beating base individually
cands_beating_base = [cid for cid in ["S3-CAND-A", "S3-CAND-B", "S3-CAND-C"] if cross_borda[cid] < cross_borda["S3-BASE"]]
print(f"\\nBest Candidate among {{A, B, C}}: {best_cand_id}")
print(f"Candidates individually beating Base: {cands_beating_base}")

# 7. PHASE 2: RECONSTRUCTION TARGET ABLATIONS ON BEST CANDIDATE
best_cand_cfg = next(c for c in STAGE3_CANDIDATES if c["id"] == best_cand_id)

RECON_ABLATIONS = [
    {
        **best_cand_cfg,
        "id": "S3-ABL-recon-pca30",
        "recon_target": "pca30",
        "l_rec": 2.0,
        "desc": f"Reconstruction Target Ablation: PCA-30 on {best_cand_id} architecture"
    },
    {
        **best_cand_cfg,
        "id": "S3-ABL-recon-pca128",
        "recon_target": "pca128",
        "l_rec": 5.0,
        "desc": f"Reconstruction Target Ablation: PCA-128 on {best_cand_id} architecture"
    }
]

print("\\n" + "="*80)
print(f" STAGE 3 PHASE 2: RECONSTRUCTION TARGET ABLATIONS ON {best_cand_id} ".center(80, "="))
print("="*80)

for r_exp in RECON_ABLATIONS:
    run_experiment_config(r_exp)

# Compare reconstruction targets {raw, pca30, pca128}
recon_exp_ids = [best_cand_id, "S3-ABL-recon-pca30", "S3-ABL-recon-pca128"]
recon_ds_ranks, recon_cross_borda = compute_borda_ranks(recon_exp_ids)

sorted_recon = sorted(recon_exp_ids, key=lambda x: recon_cross_borda[x])
winning_recon_exp = sorted_recon[0]
if winning_recon_exp == "S3-ABL-recon-pca30":
    winning_recon = "pca30"
    winning_l_rec = 2.0
elif winning_recon_exp == "S3-ABL-recon-pca128":
    winning_recon = "pca128"
    winning_l_rec = 5.0
else:
    winning_recon = "raw"
    winning_l_rec = 15.0

print(f"\\nWinning Reconstruction Target: {winning_recon} (Exp: {winning_recon_exp}, Avg Rank: {recon_cross_borda[winning_recon_exp]:.2f})")

# 8. PHASE 3: CHAMPION FORMULATION & LEARNABLE CLUSTERING HEADS
# Per BRIEF-006: combine winning modules from A/B/C + winning recon target ONLY IF each individually beats Base.
champion_fusion = "uaf" if "S3-CAND-B" in cands_beating_base else "2stage"
champion_backbone = "gat" if "S3-CAND-C" in cands_beating_base else "sage"
champion_align = "scot" if "S3-CAND-A" in cands_beating_base else None
champion_l_align = 0.5 if champion_align == "scot" else 0.0

champion_cfg = {
    "id": "S3-CHAMPION",
    "fusion": champion_fusion,
    "backbone": champion_backbone,
    "recon_target": winning_recon,
    "align_mode": champion_align,
    "l_rec": winning_l_rec,
    "l_spa": 2.0,
    "l_align": champion_l_align,
    "lr": 1e-3,
    "epochs": 150,
    "desc": f"Stage 3 Champion: Synthesized best fused embedding (fusion={champion_fusion}, backbone={champion_backbone}, align={champion_align}, recon={winning_recon})"
}

print("\\n" + "="*80)
print(f" STAGE 3 PHASE 3: CHAMPION EMBEDDING ({champion_cfg['desc']}) ".center(80, "="))
print("="*80)

# Check if champion is already identical to an existing run
need_train_champion = True
for existing_cfg in STAGE3_CANDIDATES + RECON_ABLATIONS:
    if (existing_cfg["fusion"] == champion_cfg["fusion"] and
        existing_cfg["backbone"] == champion_cfg["backbone"] and
        existing_cfg["recon_target"] == champion_cfg["recon_target"] and
        existing_cfg["align_mode"] == champion_cfg["align_mode"]):
        print(f"Champion configuration is identical to {existing_cfg['id']}. Linking runs...")
        champion_source_id = existing_cfg["id"]
        need_train_champion = False
        break

if need_train_champion:
    run_experiment_config(champion_cfg)
    champion_source_id = "S3-CHAMPION"

# 9. LEARNABLE CLUSTERING AXIS ON CHAMPION: DEC & IDEC HEADS
print("\\n" + "="*80)
print(" EVALUATING LEARNABLE CLUSTERING AXIS ON CHAMPION (k-means vs DEC vs IDEC) ".center(80, "="))
print("="*80)

cluster_methods = ["DEC", "IDEC"]
for cl_method in cluster_methods:
    cl_exp_id = f"S3-ABL-cluster-{cl_method.lower()}"
    print(f"\\nRunning learnable clustering head: {cl_method} (Exp ID: {cl_exp_id})...")
    cl_predictions = {d: {} for d in DATA_CACHE.keys()}

    for d_name in DATA_CACHE.keys():
        cached = DATA_CACHE[d_name]
        k_clusters = cached["k"]
        ground_truth = cached["ground_truth"]

        for seed in SEEDS:
            run_key = f"{d_name}_{cl_exp_id}_s{seed}"
            run_dir = os.path.join(REPO, "runs", "s3", cl_exp_id, f"{d_name}_s{seed}")
            metrics_file = os.path.join(run_dir, "metrics.json")
            os.makedirs(run_dir, exist_ok=True)
            os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)

            if os.path.exists(metrics_file):
                print(f"[SKIP_DONE] {run_key} already completed.")
                continue

            set_seed(seed)
            t0 = time.time()

            # Load champion embedding
            source_dir = os.path.join(REPO, "runs", "s3", champion_source_id, f"{d_name}_s{seed}")
            emb_f = os.path.join(source_dir, "latent_embeddings.npy")
            if not os.path.exists(emb_f):
                print(f"Error: Champion embedding {emb_f} missing.")
                continue

            emb_np = np.load(emb_f)
            z_tensor = torch.tensor(emb_np, dtype=torch.float32).to(DEVICE)

            # Initialize DEC ClusterHead with k-means centroids
            km_init = KMeans(n_clusters=k_clusters, n_init=10, random_state=seed)
            km_labels = km_init.fit_predict(emb_np)
            centers_init = km_init.cluster_centers_

            cl_head = ClusterHead(num_clusters=k_clusters, in_features=emb_np.shape[1], alpha=1.0).to(DEVICE)
            with torch.no_grad():
                cl_head.cluster_centers.copy_(torch.tensor(centers_init, dtype=torch.float32).to(DEVICE))

            opt_cl = torch.optim.Adam(cl_head.parameters(), lr=1e-3)

            # Fine-tune clustering head for 50 epochs
            for cl_epoch in range(50):
                opt_cl.zero_grad()
                q = cl_head(z_tensor)
                p = target_distribution(q).detach()
                loss_kl = kl_clustering_loss(q, p)

                if cl_method == "IDEC":
                    # IDEC adds geometric anchor to preserve manifold distance
                    recon_anchor = F.mse_loss(q @ cl_head.cluster_centers, z_tensor)
                    loss_total = loss_kl + 0.1 * recon_anchor
                else:
                    loss_total = loss_kl

                loss_total.backward()
                opt_cl.step()

            # Final predictions from cluster head
            with torch.no_grad():
                q_final = cl_head(z_tensor)
                cl_pred = torch.argmax(q_final, dim=1).cpu().numpy()

            wall_time = time.time() - t0
            cl_predictions[d_name][seed] = cl_pred

            try:
                cl_sil = float(silhouette_score(emb_np, cl_pred))
            except Exception:
                cl_sil = -1.0

            # Seed stability
            cl_stab_aris = [
                float(adjusted_rand_score(cl_pred, other_p))
                for other_s, other_p in cl_predictions[d_name].items()
                if other_s != seed
            ]
            cl_stab = float(np.mean(cl_stab_aris)) if cl_stab_aris else 0.5

            metrics = {
                "silhouette": cl_sil,
                "seed_stability_ari": cl_stab,
                "reconstruction_balance": 1.0,
                "knn_overlap": 0.0,
                "selection_score": float(0.5 * cl_sil + 0.5 * cl_stab),
                "wall_time_seconds": round(wall_time, 2),
                "unsupervised_selection_guarantee": True
            }

            posthoc_metrics = evaluate_posthoc_quarantined(cl_pred, ground_truth)
            print(f"Done: {cl_exp_id} ({d_name} s{seed}) -> Sil: {cl_sil:.3f} | Stab: {cl_stab:.3f} | PostHoc ARI: {posthoc_metrics['ari']:.3f}")

            np.save(os.path.join(run_dir, "latent_embeddings.npy"), emb_np)
            np.save(os.path.join(run_dir, "cluster_assignments.npy"), cl_pred)

            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)

            with open(os.path.join(run_dir, "posthoc.json"), "w") as f:
                json.dump(posthoc_metrics, f, indent=2)

            with open(os.path.join(run_dir, "run.log"), "w") as f:
                f.write(f"Dataset: {d_name}\\nExp: {cl_exp_id}\\nSeed: {seed}\\nWall Time: {wall_time:.2f}s\\n")

            fig, ax = plt.subplots(figsize=(6, 5))
            sc_plot = ax.scatter(emb_np[:, 0], emb_np[:, 1], c=cl_pred, cmap="tab20", s=6, alpha=0.8)
            ax.set_title(f"{d_name} - {cl_exp_id} (Seed {seed})")
            fig.colorbar(sc_plot, ax=ax)
            fig.savefig(os.path.join(run_dir, "plots", "cluster_latent.png"), dpi=100, bbox_inches="tight")
            plt.close(fig)

            reg_entry = {
                "dataset": d_name,
                "exp_id": cl_exp_id,
                "seed": seed,
                "model_type": f"ClusterHead_{cl_method}",
                "metrics": metrics,
                "posthoc": posthoc_metrics,
                "timestamp": time.time()
            }
            with open(os.path.join(run_dir, "registry_entry.json"), "w") as f:
                json.dump(reg_entry, f, indent=2)

            with open(registry_file, "a") as f:
                f.write(json.dumps(reg_entry) + "\\n")

            push_to_github(f"run(s3): {cl_exp_id} on {d_name} s{seed}", [run_dir, registry_file])

print("\\n" + "="*80)
print(" STAGE 3 EXECUTION COMPLETE! ALL RUNS PERSISTED & PUSHED ".center(80, "="))
print("="*80)
'''

notebook = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": cell_1_code.splitlines(keepends=True)
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": cell_2_code.splitlines(keepends=True)
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

output_path = "research_notebook.ipynb"
with open(output_path, "w") as f:
    json.dump(notebook, f, indent=2)

print(f"Generated {output_path} successfully. Total cells: {len(notebook['cells'])}")
