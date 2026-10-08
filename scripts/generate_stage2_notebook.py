"""
Stage 2 Notebook Generator for Robust Architecture Search and Ablations across all 6 datasets.
Executes on Kaggle GPU, enforces label-free selection firewall, and syncs directly to GitHub.
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

cell_2_code = '''# Cell 2: Stage 2 Robust Architecture Search Execution across all 6 datasets
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
                sh("git", "add", "runs/s2", "reports")
            status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True)
            if not status.stdout.strip():
                return
            sh("git", "commit", "-m", commit_msg)
            sh(*GIT, "push", "origin", "main")
            print(f"Pushed to GitHub: {commit_msg}")
            return
        except Exception as e:
            print(f"Git push attempt {attempt+1} failed: {e}. Retrying in 3s...")
            time.sleep(3)

# 1. Setup repository
if not os.path.isdir(os.path.join(REPO, ".git")):
    print("Cloning repository...")
    subprocess.run(GIT + ["clone", "https://github.com/rifatahsanpul0k/my_research.git", REPO], check=True)
    subprocess.run(["git", "config", "user.name", "kaggle-runner"], cwd=REPO, check=True)
    subprocess.run(["git", "config", "user.email", "kaggle-runner@local"], cwd=REPO, check=True)
else:
    print("Pulling latest repository state...")
    sh(*GIT, "pull", "--rebase", "origin", "main")

os.makedirs(os.path.join(REPO, "pipeline"), exist_ok=True)
os.makedirs(os.path.join(REPO, "tests"), exist_ok=True)
os.makedirs(os.path.join(REPO, "runs", "s2"), exist_ok=True)
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
from pipeline.models import Model_H_HiRe, Model_Gated_ARISE

# 3. Locate Dataset Root (with Kaggle API download fallback if not pre-mounted)
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

if not dataset_root and os.path.exists("/kaggle/input"):
    for r, dirs, files in os.walk("/kaggle/input"):
        if "adata_RNA.h5ad" in files and "10x_human_lymph_node_A1" in r:
            dataset_root = os.path.dirname(r) if os.path.basename(r) == "10x_human_lymph_node_A1" else r
            break

if not dataset_root:
    print("Dataset not pre-mounted in /kaggle/input. Downloading private dataset via Kaggle API...")
    k_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(k_dir, exist_ok=True)
    with open(os.path.join(k_dir, "kaggle.json"), "w") as f:
        json.dump({"username": "pulokpulok", "key": "5e8ca54718aa6e55d747a93fef820b41"}, f)
    os.chmod(os.path.join(k_dir, "kaggle.json"), 0o600)
    dl_target = "/kaggle/working/data"
    os.makedirs(dl_target, exist_ok=True)
    subprocess.run([
        "kaggle", "datasets", "download", "-d", "pulokpulok/spatial-multiomics-6datasets-private",
        "-p", dl_target, "--unzip"
    ], check=True)
    dataset_root = dl_target

print("Dataset root located at:", dataset_root)

# Helper functions
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_dual_graphs(rna_raw, cell_positions, num_neighbors=15, device="cuda"):
    sim_mat = cosine_similarity(rna_raw)
    nbrs = NearestNeighbors(n_neighbors=num_neighbors + 1, metric="cosine").fit(rna_raw)
    _, indices = nbrs.kneighbors(rna_raw)
    adj_sim = np.zeros_like(sim_mat, dtype=int)
    for i in range(len(rna_raw)):
        for j in indices[i][1:]:
            adj_sim[i, j] = 1
            adj_sim[j, i] = 1

    nbrs_sp = NearestNeighbors(n_neighbors=num_neighbors + 1).fit(cell_positions)
    _, ind_sp = nbrs_sp.kneighbors(cell_positions)
    adj_dist = np.zeros((len(cell_positions), len(cell_positions)), dtype=int)
    for i in range(len(cell_positions)):
        for j in ind_sp[i][1:]:
            adj_dist[i, j] = 1
            adj_dist[j, i] = 1

    adj_com = adj_sim * adj_dist
    def to_edge_index_and_weight(adj, sim_vals):
        rows, cols = np.where(adj > 0)
        edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long, device=device)
        weights = torch.tensor(sim_vals[rows, cols], dtype=torch.float32, device=device)
        return edge_index, weights

    return (
        to_edge_index_and_weight(adj_sim, sim_mat),
        to_edge_index_and_weight(adj_dist, np.ones_like(sim_mat)),
        to_edge_index_and_weight(adj_com, sim_mat)
    )

def compute_knn_overlap(input_features, latent_embeddings, k=15):
    nbrs_in = NearestNeighbors(n_neighbors=k + 1).fit(input_features)
    _, idx_in = nbrs_in.kneighbors(input_features)
    nbrs_lat = NearestNeighbors(n_neighbors=k + 1).fit(latent_embeddings)
    _, idx_lat = nbrs_lat.kneighbors(latent_embeddings)
    jaccards = []
    for i in range(len(input_features)):
        set_in = set(idx_in[i, 1:])
        set_lat = set(idx_lat[i, 1:])
        jaccards.append(len(set_in.intersection(set_lat)) / max(1, len(set_in.union(set_lat))))
    return float(np.mean(jaccards))

def compute_laplacian_loss(emb, edge_index, edge_weight=None):
    r, c = edge_index
    diff = emb[r] - emb[c]
    dist_sq = (diff ** 2).sum(dim=-1)
    if edge_weight is not None:
        dist_sq = dist_sq * edge_weight
    return dist_sq.mean()

def compute_dense_bce(emb, edge_index, edge_weight):
    emb_norm = F.normalize(emb, p=2, dim=1, eps=1e-8)
    sim_mat = torch.matmul(emb_norm, emb_norm.T)
    sim_mat = sim_mat - torch.diag_embed(torch.diag(sim_mat))
    sim_prob = torch.sigmoid(sim_mat)
    
    num_nodes = emb.size(0)
    graph_nei = torch.sparse_coo_tensor(
        edge_index, edge_weight, torch.Size([num_nodes, num_nodes])
    ).coalesce().to_dense()
    graph_nei = torch.clamp(graph_nei, max=1.0)
    graph_neg = 1.0 - graph_nei
    
    neigh_loss = torch.mul(graph_nei, torch.log(sim_prob + 1e-10)).mean()
    neg_loss = torch.mul(graph_neg, torch.log(1.0 - sim_prob + 1e-10)).mean()
    return -(neigh_loss + neg_loss) / 2.0

# 4. Stage 2 Bounded Matrix Definitions (10 Experiments)
STAGE2_EXPERIMENTS = [
    # Core Candidates
    {
        "id": "S2-EXP01-CAND-H-HiRe",
        "model_type": "H-HiRe",
        "fusion": "2stage",
        "backbone": "sage",
        "recon_target": "raw",
        "spa_mode": "calibrated",
        "l_rec": 15.0,
        "l_spa": 2.0,
        "lr": 1e-3,
        "epochs": 150,
        "desc": "Candidate 1: H-HiRe (Hierarchical 2-stage MLP, 3000 HVG, SAGEConv, Calibrated Spatial)"
    },
    {
        "id": "S2-EXP02-CAND-Attn-SAGE",
        "model_type": "H-HiRe",
        "fusion": "cross_attn",
        "backbone": "sage",
        "recon_target": "raw",
        "spa_mode": "calibrated",
        "l_rec": 15.0,
        "l_spa": 2.0,
        "lr": 1e-3,
        "epochs": 150,
        "desc": "Candidate 2: Attn-SAGE (Cross-Attention, 3000 HVG, SAGEConv, Calibrated Spatial)"
    },
    {
        "id": "S2-EXP03-CAND-Gated-ARISE",
        "model_type": "Gated-ARISE",
        "fusion": "2stage",
        "backbone": "gcn",
        "recon_target": "raw",
        "spa_mode": "gated",
        "l_rec": 15.0,
        "l_spa": 2.0,
        "lr": 1e-3,
        "epochs": 150,
        "desc": "Candidate 3: Gated-ARISE (Dynamic Edge-Gated GCNConv, 3000 HVG recon)"
    },
    # Ablations
    {
        "id": "S2-EXP04-ABL-fusion-concat",
        "model_type": "H-HiRe",
        "fusion": "concat",
        "backbone": "sage",
        "recon_target": "raw",
        "spa_mode": "calibrated",
        "l_rec": 15.0,
        "l_spa": 2.0,
        "lr": 1e-3,
        "epochs": 150,
        "desc": "Ablation: Simple Concat Fusion Baseline"
    },
    {
        "id": "S2-EXP05-ABL-recon-pca30",
        "model_type": "H-HiRe",
        "fusion": "2stage",
        "backbone": "sage",
        "recon_target": "pca",
        "spa_mode": "calibrated",
        "l_rec": 2.0,
        "l_spa": 1.0,
        "lr": 1e-3,
        "epochs": 150,
        "desc": "Ablation: PCA-30 Feature Reconstruction"
    },
    {
        "id": "S2-EXP06-ABL-recon-decoupled",
        "model_type": "H-HiRe",
        "fusion": "2stage",
        "backbone": "sage",
        "recon_target": "decoupled",
        "spa_mode": "calibrated",
        "l_rec": 15.0,
        "l_spa": 2.0,
        "lr": 1e-3,
        "epochs": 150,
        "desc": "Ablation: Decoupled Modality Reconstruction"
    },
    {
        "id": "S2-EXP07-ABL-spa-zero",
        "model_type": "H-HiRe",
        "fusion": "2stage",
        "backbone": "sage",
        "recon_target": "raw",
        "spa_mode": "zero",
        "l_rec": 15.0,
        "l_spa": 0.0,
        "lr": 1e-3,
        "epochs": 150,
        "desc": "Ablation: Zero Spatial Regularization (Pure Feature AE)"
    },
    {
        "id": "S2-EXP08-ABL-spa-dense-bce",
        "model_type": "H-HiRe",
        "fusion": "2stage",
        "backbone": "sage",
        "recon_target": "raw",
        "spa_mode": "dense_bce",
        "l_rec": 15.0,
        "l_spa": 5.0,
        "lr": 1e-3,
        "epochs": 150,
        "desc": "Ablation: Dense Spatial Contrastive BCE (Over-clustering Diagnostic)"
    },
    {
        "id": "S2-EXP09-ABL-spa-laplacian",
        "model_type": "H-HiRe",
        "fusion": "2stage",
        "backbone": "sage",
        "recon_target": "raw",
        "spa_mode": "laplacian",
        "l_rec": 15.0,
        "l_spa": 2.0,
        "lr": 1e-3,
        "epochs": 150,
        "desc": "Ablation: Graph Laplacian Smoothness Regularization"
    },
    {
        "id": "S2-EXP10-ABL-backbone-gat",
        "model_type": "H-HiRe",
        "fusion": "cross_attn",
        "backbone": "gat",
        "recon_target": "raw",
        "spa_mode": "calibrated",
        "l_rec": 15.0,
        "l_spa": 2.0,
        "lr": 1e-3,
        "epochs": 150,
        "desc": "Ablation: Graph Attention Network (GAT) Backbone"
    },
]

SEEDS = [42, 1234, 2024]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Executing Stage 2 on device: {DEVICE}")

# 5. Execute Stage 2 Matrix Across All 6 Datasets
registry_entries = []
registry_file = os.path.join(REPO, "runs", "s2", "registry.jsonl")

# Load existing registry if present
if os.path.exists(registry_file):
    with open(registry_file) as f:
        for line in f:
            if line.strip():
                registry_entries.append(json.loads(line))

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
        print(f"Dataset directory for {d_name} not found. Skipping.")
        continue

    print("\\n" + "="*80)
    print(f" STAGE 2 DATASET: {d_name} (Declared K={DECLARED_K[d_name]}) ".center(80, "="))
    print("="*80)

    data = load_dataset_features_unsupervised(d_dir, d_cfg)
    k_clusters = data["num_clusters"]

    rna_raw = torch.tensor(data["rna_raw"], dtype=torch.float32).to(DEVICE)
    mod2_raw = torch.tensor(data["mod2_raw"], dtype=torch.float32).to(DEVICE)
    rna_pca = torch.tensor(data["rna_pca"][:, :30], dtype=torch.float32).to(DEVICE)
    mod2_pca = torch.tensor(data["mod2_pca"][:, :30], dtype=torch.float32).to(DEVICE)
    joint_raw = torch.cat([rna_raw, mod2_raw], dim=1)
    joint_pca = torch.cat([rna_pca, mod2_pca], dim=1)

    var_rna = float(np.var(data["rna_raw"]))
    var_mod2 = float(np.var(data["mod2_raw"]))

    (e_sim, w_sim), (e_dist, w_dist), (e_com, w_com) = build_dual_graphs(
        data["rna_raw"], data["cell_positions"], num_neighbors=15, device=DEVICE
    )

    ground_truth = load_posthoc_ground_truth(d_dir, d_cfg)
    config_predictions = {}

    for exp in STAGE2_EXPERIMENTS:
        exp_id = exp["id"]
        config_predictions[exp_id] = {}

        # First load existing runs for seed stability
        for seed in SEEDS:
            run_dir = os.path.join(REPO, "runs", "s2", exp_id, f"{d_name}_s{seed}")
            pred_f = os.path.join(run_dir, "cluster_assignments.npy")
            if os.path.exists(pred_f):
                try:
                    config_predictions[exp_id][seed] = np.load(pred_f)
                except Exception:
                    pass

        for seed in SEEDS:
            run_key = f"{d_name}_{exp_id}_s{seed}"
            run_dir = os.path.join(REPO, "runs", "s2", exp_id, f"{d_name}_s{seed}")
            metrics_file = os.path.join(run_dir, "metrics.json")
            os.makedirs(run_dir, exist_ok=True)
            os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)

            if os.path.exists(metrics_file):
                print(f"[SKIP_DONE] {run_key} already completed.")
                continue

            set_seed(seed)
            t0 = time.time()

            m_type = exp["model_type"]
            lr = exp["lr"]
            epochs = exp["epochs"]
            l_rec = exp["l_rec"]
            l_spa = exp["l_spa"]
            recon_target = exp["recon_target"]
            spa_mode = exp["spa_mode"]

            in_rna = rna_pca.shape[1] if recon_target == "pca" else rna_raw.shape[1]
            in_mod2 = mod2_pca.shape[1] if recon_target == "pca" else mod2_raw.shape[1]
            x_r_in = rna_pca if recon_target == "pca" else rna_raw
            x_m_in = mod2_pca if recon_target == "pca" else mod2_raw
            target_r = rna_pca if recon_target == "pca" else rna_raw
            target_m = mod2_pca if recon_target == "pca" else mod2_raw
            target_joint = joint_pca if recon_target == "pca" else joint_raw

            if m_type == "H-HiRe":
                model = Model_H_HiRe(
                    in_rna, in_mod2, hidden_dim=256, out_dim=64,
                    backbone=exp["backbone"], fusion=exp["fusion"], recon_mode=recon_target
                ).to(DEVICE)
            elif m_type == "Gated-ARISE":
                model = Model_Gated_ARISE(in_rna, in_mod2, hidden_dim=256, out_dim=64).to(DEVICE)
            else:
                raise ValueError(f"Unknown model type: {m_type}")

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

            best_sil = -1.0
            best_emb = None
            best_pred = None
            best_recs = None

            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()

                if m_type == "H-HiRe":
                    z_final, z_sim, z_dist, z_mod2, rec_rna, rec_mod2, rec_joint = model(
                        x_r_in, x_m_in, e_sim, w_sim, e_dist, w_dist, e_com, w_com
                    )
                    curr_emb = z_final

                    if recon_target == "decoupled":
                        loss_rec = F.mse_loss(target_r, rec_rna) + F.mse_loss(target_m, rec_mod2)
                    else:
                        loss_rec = F.mse_loss(target_joint, rec_joint) + 0.5 * (F.mse_loss(target_r, rec_rna) + F.mse_loss(target_m, rec_mod2))

                    if spa_mode == "zero":
                        loss_spa = torch.tensor(0.0, device=DEVICE)
                    elif spa_mode == "dense_bce":
                        loss_spa = compute_dense_bce(z_dist, e_dist, w_dist)
                    elif spa_mode == "laplacian":
                        loss_spa = compute_laplacian_loss(z_final, e_dist, w_dist)
                    else: # calibrated
                        loss_spa = compute_laplacian_loss(z_dist, e_dist, w_dist)

                    loss = l_rec * loss_rec + l_spa * loss_spa

                elif m_type == "Gated-ARISE":
                    z_final, x_sim, x_dist, pro, rec_rna, rec_mod2, rec_joint, gate_w_dist = model(
                        x_r_in, x_m_in, e_sim, w_sim, e_dist, w_dist, e_com, w_com
                    )
                    curr_emb = z_final
                    loss_rec = F.mse_loss(target_joint, rec_joint) + 0.5 * (F.mse_loss(target_r, rec_rna) + F.mse_loss(target_m, rec_mod2))
                    loss_spa = compute_laplacian_loss(x_dist, e_dist, gate_w_dist)
                    loss = l_rec * loss_rec + l_spa * loss_spa

                loss.backward()
                optimizer.step()

                if (epoch + 1) % 25 == 0 or epoch == epochs - 1:
                    model.eval()
                    with torch.no_grad():
                        e_np = curr_emb.detach().cpu().numpy()
                    km = KMeans(n_clusters=k_clusters, random_state=seed, n_init=3)
                    pred_c = km.fit_predict(e_np)
                    sil = float(silhouette_score(e_np, pred_c))
                    if sil > best_sil:
                        best_sil = sil
                        best_emb = e_np.copy()
                        best_pred = pred_c.copy()
                        best_recs = (rec_rna.detach().cpu().numpy(), rec_mod2.detach().cpu().numpy())

            wall_time = time.time() - t0
            final_emb = best_emb if best_emb is not None else curr_emb.detach().cpu().numpy()
            km_final = KMeans(n_clusters=k_clusters, random_state=seed, n_init=10)
            final_pred = km_final.fit_predict(final_emb)
            config_predictions[exp_id][seed] = final_pred

            final_sil = float(silhouette_score(final_emb, final_pred))
            knn_overlap = compute_knn_overlap(data["rna_raw"], final_emb, k=15)

            # Compute label-free reconstruction balance
            if best_recs is not None:
                r_np, m_np = best_recs
                mse_r = float(np.mean((data["rna_raw"] - r_np) ** 2)) if recon_target != "pca" else float(np.mean((data["rna_pca"][:, :30] - r_np) ** 2))
                mse_m = float(np.mean((data["mod2_raw"] - m_np) ** 2)) if recon_target != "pca" else float(np.mean((data["mod2_pca"][:, :30] - m_np) ** 2))
                fid_r = max(0.0, 1.0 - (mse_r / max(var_rna, 1e-6)))
                fid_m = max(0.0, 1.0 - (mse_m / max(var_mod2, 1e-6)))
                recon_balance = float((min(fid_r, fid_m) + 1e-4) / (max(fid_r, fid_m) + 1e-4))
            else:
                recon_balance = 0.5

            # Compute seed stability over off-diagonal pairs
            stability_aris = [
                float(adjusted_rand_score(final_pred, other_p))
                for other_s, other_p in config_predictions[exp_id].items()
                if other_s != seed
            ]
            seed_stability_ari = float(np.mean(stability_aris)) if stability_aris else 0.5

            metrics = {
                "silhouette": final_sil,
                "seed_stability_ari": seed_stability_ari,
                "reconstruction_balance": recon_balance,
                "knn_overlap": knn_overlap,
                "selection_score": float(0.4 * final_sil + 0.4 * seed_stability_ari + 0.2 * recon_balance),
                "wall_time_seconds": round(wall_time, 2),
                "unsupervised_selection_guarantee": True
            }

            posthoc_metrics = evaluate_posthoc_quarantined(final_pred, ground_truth)
            print(f"Done: {exp_id} ({d_name} s{seed}) -> Sil: {final_sil:.3f} | Stab: {seed_stability_ari:.3f} | ReconBal: {recon_balance:.3f} | PostHoc ARI: {posthoc_metrics['ari']:.3f}")

            np.save(os.path.join(run_dir, "latent_embeddings.npy"), final_emb)
            np.save(os.path.join(run_dir, "cluster_assignments.npy"), final_pred)

            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)

            with open(os.path.join(run_dir, "posthoc.json"), "w") as f:
                json.dump(posthoc_metrics, f, indent=2)

            with open(os.path.join(run_dir, "run.log"), "w") as f:
                f.write(f"Dataset: {d_name}\\nExp: {exp_id}\\nSeed: {seed}\\nWall Time: {wall_time:.2f}s\\n")

            fig, ax = plt.subplots(figsize=(6, 5))
            sc_plot = ax.scatter(final_emb[:, 0], final_emb[:, 1], c=final_pred, cmap="tab20", s=6, alpha=0.8)
            ax.set_title(f"{d_name} - {exp_id} (Seed {seed})")
            fig.colorbar(sc_plot, ax=ax)
            fig.savefig(os.path.join(run_dir, "plots", "cluster_latent.png"), dpi=100, bbox_inches="tight")
            plt.close(fig)

            reg_entry = {
                "dataset": d_name,
                "exp_id": exp_id,
                "seed": seed,
                "model_type": m_type,
                "metrics": metrics,
                "posthoc": posthoc_metrics,
                "timestamp": time.time()
            }
            with open(os.path.join(run_dir, "registry_entry.json"), "w") as f:
                json.dump(reg_entry, f, indent=2)

            registry_entries.append(reg_entry)
            with open(registry_file, "a") as f:
                f.write(json.dumps(reg_entry) + "\\n")

    # Re-calculate exact off-diagonal stability across the 3 seeds for this dataset
    for exp in STAGE2_EXPERIMENTS:
        exp_id = exp["id"]
        if exp_id in config_predictions and len(config_predictions[exp_id]) >= 2:
            preds = config_predictions[exp_id]
            for s in SEEDS:
                if s in preds:
                    other_aris = [float(adjusted_rand_score(preds[s], preds[other])) for other in SEEDS if other != s and other in preds]
                    corr_stab = float(np.mean(other_aris))
                    run_dir = os.path.join(REPO, "runs", "s2", exp_id, f"{d_name}_s{s}")
                    m_f = os.path.join(run_dir, "metrics.json")
                    if os.path.exists(m_f):
                        with open(m_f) as f:
                            m_d = json.load(f)
                        m_d["seed_stability_ari"] = corr_stab
                        m_d["selection_score"] = float(0.4 * m_d["silhouette"] + 0.4 * corr_stab + 0.2 * m_d["reconstruction_balance"])
                        with open(m_f, "w") as f:
                            json.dump(m_d, f, indent=2)

print("\\n" + "="*80)
print(" STAGE 2 MATRIX EXECUTION COMPLETE ".center(80, "="))
print("="*80)

# 6. Rebuild registry.jsonl from all finalized runs
all_s2_entries = []
for exp in STAGE2_EXPERIMENTS:
    exp_id = exp["id"]
    for d_cfg in DATASET_CONFIGS:
        d_name = d_cfg["name"]
        for s in SEEDS:
            r_entry_file = os.path.join(REPO, "runs", "s2", exp_id, f"{d_name}_s{s}", "registry_entry.json")
            if os.path.exists(r_entry_file):
                with open(r_entry_file) as f:
                    entry = json.load(f)
                # Ensure seed_stability_ari is updated from metrics.json
                m_file = os.path.join(REPO, "runs", "s2", exp_id, f"{d_name}_s{s}", "metrics.json")
                if os.path.exists(m_file):
                    with open(m_file) as f:
                        entry["metrics"] = json.load(f)
                all_s2_entries.append(entry)

with open(registry_file, "w") as f:
    for e in all_s2_entries:
        f.write(json.dumps(e) + "\\n")
print(f"Saved {len(all_s2_entries)} finalized runs to {registry_file}")

# Push results to GitHub
push_to_github("results(stage2): complete Stage 2 robust architecture search and ablation runs", ["runs/s2/"])
'''

# Assemble Jupyter Notebook
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
