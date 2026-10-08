import os
import json
import numpy as np
from sklearn.metrics import adjusted_rand_score

RUNS_DIR = "runs/s1"
REGISTRY_PATH = os.path.join(RUNS_DIR, "registry.jsonl")
DATASETS = ["10x_human_lymph_node_A1", "10x_human_lymph_node_D1", "Mouse_Brain_E11_S1"]
SEEDS = [42, 1234, 2024]

experiments = sorted([
    d for d in os.listdir(RUNS_DIR)
    if os.path.isdir(os.path.join(RUNS_DIR, d)) and d.startswith("EXP")
])

# 1. Update individual metrics.json and registry_entry.json
for exp_id in experiments:
    for d_name in DATASETS:
        preds = {}
        for s in SEEDS:
            run_dir = os.path.join(RUNS_DIR, exp_id, f"{d_name}_s{s}")
            c_file = os.path.join(run_dir, "cluster_assignments.npy")
            if os.path.exists(c_file):
                preds[s] = np.load(c_file)
                
        # For each seed, calculate stability vs other seeds only (off-diagonal)
        for s in SEEDS:
            run_dir = os.path.join(RUNS_DIR, exp_id, f"{d_name}_s{s}")
            m_file = os.path.join(run_dir, "metrics.json")
            r_file = os.path.join(run_dir, "registry_entry.json")
            
            if not os.path.exists(m_file) or s not in preds:
                continue
                
            other_aris = [
                float(adjusted_rand_score(preds[s], preds[other]))
                for other in SEEDS if other != s and other in preds
            ]
            corrected_stab = float(np.mean(other_aris)) if other_aris else 0.0
            
            with open(m_file) as f:
                m_data = json.load(f)
                
            m_data["seed_stability_ari"] = corrected_stab
            m_data["selection_score"] = float(
                0.5 * m_data["silhouette"] + 0.3 * corrected_stab + 0.2 * m_data["knn_overlap"]
            )
            
            with open(m_file, "w") as f:
                json.dump(m_data, f, indent=2)
                
            if os.path.exists(r_file):
                with open(r_file) as f:
                    r_data = json.load(f)
                r_data["metrics"]["seed_stability_ari"] = corrected_stab
                r_data["metrics"]["selection_score"] = m_data["selection_score"]
                with open(r_file, "w") as f:
                    json.dump(r_data, f, indent=2)

print("Updated individual metrics.json and registry_entry.json files.")

# 2. Re-write registry.jsonl from corrected registry_entry.json files
updated_entries = []
for exp_id in experiments:
    for d_name in DATASETS:
        for s in SEEDS:
            run_dir = os.path.join(RUNS_DIR, exp_id, f"{d_name}_s{s}")
            r_file = os.path.join(run_dir, "registry_entry.json")
            if os.path.exists(r_file):
                with open(r_file) as f:
                    updated_entries.append(json.load(f))

with open(REGISTRY_PATH, "w") as f:
    for entry in updated_entries:
        f.write(json.dumps(entry) + "\n")

print(f"Rewrote {REGISTRY_PATH} with {len(updated_entries)} updated entries.")
