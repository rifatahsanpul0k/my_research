import json

with open("runs/s1/recomputed_metrics.json") as f:
    res = json.load(f)

exp_names = {
    "EXP01-M0-base-smart": "M0 Base SMART (Spatial graph, SAGEConv, Concat)",
    "EXP02-M1-dual-graph": "M1 DualGraph (+ Dual/Common graphs)",
    "EXP03-M2-hierarchical": "M2 Hierarchical SMART (+ 2-Stage MLP fusion)",
    "EXP04-M3-contrastive": "M3 Contrastive SMART (+ Dense Spatial BCE)",
    "EXP05-M4-highdim": "M4 High-Dim SMART (+ Raw 3000 HVG recon)",
    "EXP06-M5-arise-gcn": "M5 Full ARISE (+ Spectral GCNConv)",
    "EXP07-encoder-gat-like": "GCN with dropout regularization",
    "EXP08-encoder-mlp-baseline": "Single graph baseline with L2 penalty",
    "EXP09-fusion-cross-attention": "Cross-Modality Multihead Attention Fusion",
    "EXP10-loss-recon-only": "ARISE GCN (Zero contrastive loss)",
    "EXP11-loss-contrast-heavy": "ARISE GCN (Heavy spatial contrastive)",
    "EXP12-graph-spatial-heavy": "High-dim recon (Equal spatial & recon)",
    "EXP13-fast-convergence": "Higher learning rate (lr=2e-3) ARISE GCN",
    "EXP14-dim-bottleneck32": "32-D bottleneck latent embedding",
    "EXP15-dim-bottleneck128": "128-D latent embedding",
}

datasets = list(res.keys())
exps = list(res[datasets[0]].keys())

tables_md = []

for d in datasets:
    data = res[d]
    sil_ranks = {e: r+1 for r, e in enumerate(sorted(exps, key=lambda x: data[x]['sil_mean'], reverse=True))}
    stab_ranks = {e: r+1 for r, e in enumerate(sorted(exps, key=lambda x: data[x]['seed_stability_mean'], reverse=True))}
    knn_ranks = {e: r+1 for r, e in enumerate(sorted(exps, key=lambda x: data[x]['knn_mean'], reverse=True))}
    
    borda3 = {e: sil_ranks[e] + stab_ranks[e] + knn_ranks[e] for e in exps}
    borda2 = {e: sil_ranks[e] + stab_ranks[e] for e in exps}
    sorted_b3 = sorted(exps, key=lambda e: borda3[e])
    
    header = f"### Dataset: `{d}`\n\n"
    header += "| Borda Rank | Exp ID | Architecture | Borda (3-M) | Borda (2-M) | Silhouette [Rank] | Seed Stability [Rank] | kNN Overlap [Rank] | Post-Hoc ARI |\n"
    header += "|---|---|---|---|---|---|---|---|---|\n"
    
    rows = []
    for r, e in enumerate(sorted_b3, 1):
        m = data[e]
        desc = exp_names.get(e, "")
        row = (
            f"| #{r} | `{e}` | {desc} | {borda3[e]} | {borda2[e]} | "
            f"{m['sil_mean']:.3f} ± {m['sil_std']:.3f} [#{sil_ranks[e]}] | "
            f"{m['seed_stability_mean']:.3f} [#{stab_ranks[e]}] | "
            f"{m['knn_mean']:.4f} [#{knn_ranks[e]}] | "
            f"**{m['posthoc_ari_mean']:.3f} ± {m['posthoc_ari_std']:.3f}** |"
        )
        rows.append(row)
    tables_md.append(header + "\n".join(rows))

with open("runs/s1/tables.md", "w") as f:
    f.write("\n\n".join(tables_md))
print("Wrote runs/s1/tables.md")
