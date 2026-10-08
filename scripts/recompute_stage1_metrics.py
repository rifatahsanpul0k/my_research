import os
import json
import numpy as np
from sklearn.metrics import adjusted_rand_score

RUNS_DIR = "runs/s1"
DATASETS = ["10x_human_lymph_node_A1", "10x_human_lymph_node_D1", "Mouse_Brain_E11_S1"]
SEEDS = ["s42", "s1234", "s2024"]

experiments = sorted([
    d for d in os.listdir(RUNS_DIR)
    if os.path.isdir(os.path.join(RUNS_DIR, d)) and d.startswith("EXP")
])

print(f"Total experiments found: {len(experiments)}")

dataset_results = {d: {} for d in DATASETS}

for exp_id in experiments:
    for d_name in DATASETS:
        preds = {}
        sils = []
        knns = []
        aris = []
        
        for s in SEEDS:
            run_dir = os.path.join(RUNS_DIR, exp_id, f"{d_name}_{s}")
            c_file = os.path.join(run_dir, "cluster_assignments.npy")
            m_file = os.path.join(run_dir, "metrics.json")
            p_file = os.path.join(run_dir, "posthoc.json")
            
            if os.path.exists(c_file):
                preds[s] = np.load(c_file)
            if os.path.exists(m_file):
                with open(m_file) as f:
                    m = json.load(f)
                    sils.append(m["silhouette"])
                    knns.append(m["knn_overlap"])
            if os.path.exists(p_file):
                with open(p_file) as f:
                    p = json.load(f)
                    aris.append(p["ari"])
                    
        # Compute true off-diagonal seed stability ARI
        pair_aris = []
        pairs = [("s42", "s1234"), ("s42", "s2024"), ("s1234", "s2024")]
        for s1, s2 in pairs:
            if s1 in preds and s2 in preds:
                pair_aris.append(float(adjusted_rand_score(preds[s1], preds[s2])))
                
        mean_stability = float(np.mean(pair_aris)) if pair_aris else 0.0
        
        dataset_results[d_name][exp_id] = {
            "sil_mean": float(np.mean(sils)),
            "sil_std": float(np.std(sils)),
            "knn_mean": float(np.mean(knns)),
            "knn_std": float(np.std(knns)),
            "seed_stability_mean": mean_stability,
            "seed_stability_pairs": pair_aris,
            "posthoc_ari_mean": float(np.mean(aris)),
            "posthoc_ari_std": float(np.std(aris)),
        }

# Now compute rankings per dataset
for d_name in DATASETS:
    print("\n" + "=" * 90)
    print(f"DATASET: {d_name}")
    print("=" * 90)
    
    res = dataset_results[d_name]
    
    # Ranks for silhouette (higher is better rank 1..15)
    sorted_by_sil = sorted(experiments, key=lambda e: res[e]["sil_mean"], reverse=True)
    sil_ranks = {exp: r + 1 for r, exp in enumerate(sorted_by_sil)}
    
    # Ranks for seed_stability (higher is better rank 1..15)
    sorted_by_stab = sorted(experiments, key=lambda e: res[e]["seed_stability_mean"], reverse=True)
    stab_ranks = {exp: r + 1 for r, exp in enumerate(sorted_by_stab)}
    
    # Ranks for knn_overlap (higher is better rank 1..15)
    sorted_by_knn = sorted(experiments, key=lambda e: res[e]["knn_mean"], reverse=True)
    knn_ranks = {exp: r + 1 for r, exp in enumerate(sorted_by_knn)}
    
    # Borda rank (sum of ranks across 3 label-free metrics: sil, stab, knn)
    borda_scores = {}
    for exp in experiments:
        borda_scores[exp] = sil_ranks[exp] + stab_ranks[exp] + knn_ranks[exp]
        
    sorted_borda = sorted(experiments, key=lambda e: borda_scores[e])
    
    # Also check Borda with only (sil, stab) to see how knn affects it
    borda2_scores = {exp: sil_ranks[exp] + stab_ranks[exp] for exp in experiments}
    sorted_borda2 = sorted(experiments, key=lambda e: borda2_scores[e])

    print(f"{'Rank':<4} | {'Exp ID':<28} | {'Borda':<5} | {'Sil (Rank)':<14} | {'Stab (Rank)':<14} | {'kNN (Rank)':<14} | {'Post-Hoc ARI':<15}")
    print("-" * 105)
    for r, exp in enumerate(sorted_borda, 1):
        d = res[exp]
        print(f"#{r:<3} | {exp:<28} | {borda_scores[exp]:<5} | "
              f"{d['sil_mean']:.3f} (#{sil_ranks[exp]:<2}) | "
              f"{d['seed_stability_mean']:.3f} (#{stab_ranks[exp]:<2}) | "
              f"{d['knn_mean']:.4f} (#{knn_ranks[exp]:<2}) | "
              f"{d['posthoc_ari_mean']:.3f} ± {d['posthoc_ari_std']:.3f}")
        
    print("\nTop 3 by Borda (Sil+Stab+kNN):", sorted_borda[:3])
    print("Top 3 by Borda (Sil+Stab only):", sorted_borda2[:3])

# Save raw computed metrics to JSON for reference
with open("runs/s1/recomputed_metrics.json", "w") as f:
    json.dump(dataset_results, f, indent=2)
print("\nWrote runs/s1/recomputed_metrics.json")
