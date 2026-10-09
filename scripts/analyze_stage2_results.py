import os
import json
import numpy as np
from sklearn.metrics import adjusted_rand_score

RUNS_DIR = "runs/s2"
REGISTRY_PATH = os.path.join(RUNS_DIR, "registry.jsonl")

DATASETS = [
    "10x_human_lymph_node_A1",
    "10x_human_lymph_node_D1",
    "Mouse_Brain_E11_S1",
    "Mouse_Brain_E13_S1",
    "Mouse_Brain_E15_S1",
    "Mouse_Brain_E18_S1"
]
SEEDS = ["s42", "s1234", "s2024"]

experiments = sorted([
    d for d in os.listdir(RUNS_DIR)
    if os.path.isdir(os.path.join(RUNS_DIR, d)) and d.startswith("S2-EXP")
])

print(f"Total Stage 2 experiments found: {len(experiments)}")

results = {d: {} for d in DATASETS}

for exp_id in experiments:
    for d_name in DATASETS:
        preds = {}
        sils = []
        stabs = []
        recons = []
        knns = []
        post_aris = []
        post_nmis = []
        
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
                    recons.append(m.get("reconstruction_balance", 0.5))
                    knns.append(m.get("knn_overlap", 0.0))
            if os.path.exists(p_file):
                with open(p_file) as f:
                    p = json.load(f)
                    post_aris.append(p["ari"])
                    post_nmis.append(p["nmi"])
                    
        # Strict off-diagonal stability ARI across the 3 pairs
        pair_aris = []
        pairs = [("s42", "s1234"), ("s42", "s2024"), ("s1234", "s2024")]
        for s1, s2 in pairs:
            if s1 in preds and s2 in preds:
                pair_aris.append(float(adjusted_rand_score(preds[s1], preds[s2])))
                
        mean_stability = float(np.mean(pair_aris)) if pair_aris else 0.0
        
        results[d_name][exp_id] = {
            "sil_mean": float(np.mean(sils)),
            "sil_std": float(np.std(sils)),
            "seed_stability_mean": mean_stability,
            "seed_stability_pairs": pair_aris,
            "recon_balance_mean": float(np.mean(recons)),
            "recon_balance_std": float(np.std(recons)),
            "knn_mean": float(np.mean(knns)),
            "knn_std": float(np.std(knns)),
            "posthoc_ari_mean": float(np.mean(post_aris)),
            "posthoc_ari_std": float(np.std(post_aris)),
            "posthoc_nmi_mean": float(np.mean(post_nmis)),
            "posthoc_nmi_std": float(np.std(post_nmis)),
        }

# Save computed summary
with open("runs/s2/stage2_analyzed_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Compute rankings per dataset
borda_ranks_per_dataset = {}

for d_name in DATASETS:
    print("\n" + "=" * 105)
    print(f"DATASET: {d_name}")
    print("=" * 105)
    
    d_res = results[d_name]
    
    # Rank by silhouette
    sorted_by_sil = sorted(experiments, key=lambda e: d_res[e]["sil_mean"], reverse=True)
    sil_ranks = {e: r + 1 for r, e in enumerate(sorted_by_sil)}
    
    # Rank by seed stability
    sorted_by_stab = sorted(experiments, key=lambda e: d_res[e]["seed_stability_mean"], reverse=True)
    stab_ranks = {e: r + 1 for r, e in enumerate(sorted_by_stab)}
    
    # Rank by reconstruction balance
    sorted_by_recon = sorted(experiments, key=lambda e: d_res[e]["recon_balance_mean"], reverse=True)
    recon_ranks = {e: r + 1 for r, e in enumerate(sorted_by_recon)}
    
    # RAUS v2 Borda: sil + stab + recon_bal
    borda_scores = {e: sil_ranks[e] + stab_ranks[e] + recon_ranks[e] for e in experiments}
    sorted_borda = sorted(experiments, key=lambda e: (borda_scores[e], -d_res[e]["seed_stability_mean"]))
    
    borda_ranks_per_dataset[d_name] = {e: r + 1 for r, e in enumerate(sorted_borda)}
    
    print(f"{'Rank':<4} | {'Exp ID':<28} | {'Borda':<5} | {'Sil [Rk]':<13} | {'Stab [Rk]':<13} | {'ReconBal [Rk]':<14} | {'|Sil-Stab|':<10} | {'Post-Hoc ARI':<15}")
    print("-" * 115)
    for r, exp in enumerate(sorted_borda, 1):
        m = d_res[exp]
        divergence = abs(sil_ranks[exp] - stab_ranks[exp])
        flag = " ⚠️ FLAG" if divergence >= 4 else ""
        print(f"#{r:<3} | {exp:<28} | {borda_scores[exp]:<5} | "
              f"{m['sil_mean']:.3f} (#{sil_ranks[exp]:<2}) | "
              f"{m['seed_stability_mean']:.3f} (#{stab_ranks[exp]:<2}) | "
              f"{m['recon_balance_mean']:.3f} (#{recon_ranks[exp]:<2}) | "
              f"{divergence:<2}{flag:<8} | "
              f"{m['posthoc_ari_mean']:.3f} ± {m['posthoc_ari_std']:.3f}")

# Cross-dataset aggregated ranking
print("\n" + "=" * 105)
print("STAGE 2 CROSS-DATASET AGGREGATE SUMMARY (ALL 6 DATASETS)")
print("=" * 105)

avg_ranks = {}
for exp in experiments:
    avg_ranks[exp] = sum(borda_ranks_per_dataset[d][exp] for d in DATASETS) / len(DATASETS)

sorted_overall = sorted(experiments, key=lambda e: avg_ranks[e])

print(f"{'Rank':<4} | {'Exp ID':<28} | {'Avg Rank':<8} | {'A1':<4} | {'D1':<4} | {'E11':<4} | {'E13':<4} | {'E15':<4} | {'E18':<4} | {'Avg Post-Hoc ARI':<16}")
print("-" * 105)
for r, exp in enumerate(sorted_overall, 1):
    avg_ari = np.mean([results[d][exp]["posthoc_ari_mean"] for d in DATASETS])
    print(f"#{r:<3} | {exp:<28} | {avg_ranks[exp]:<8.2f} | "
          f"#{borda_ranks_per_dataset['10x_human_lymph_node_A1'][exp]:<3} | "
          f"#{borda_ranks_per_dataset['10x_human_lymph_node_D1'][exp]:<3} | "
          f"#{borda_ranks_per_dataset['Mouse_Brain_E11_S1'][exp]:<3} | "
          f"#{borda_ranks_per_dataset['Mouse_Brain_E13_S1'][exp]:<3} | "
          f"#{borda_ranks_per_dataset['Mouse_Brain_E15_S1'][exp]:<3} | "
          f"#{borda_ranks_per_dataset['Mouse_Brain_E18_S1'][exp]:<3} | "
          f"{avg_ari:.3f}")
