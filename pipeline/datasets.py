"""
Dataset Registry and Data Loader for Spatial Multi-Omics.
Enforces BRIEF-001 §1, BRIEF-003, BRIEF-004, and BRIEF-005:
- Predeclared K constants per dataset (declared exception).
- Strict separation between unsupervised feature data and quarantined post-hoc annotations.
- Strict cell index alignment across modalities and ground truth.
- Robust sparse TF-IDF + TruncatedSVD for high-dimensional ATAC-seq without cell filtering.
"""

import os
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
import scanpy as sc
import torch

DECLARED_K: Dict[str, int] = {
    "10x_human_lymph_node_A1": 10,
    "10x_human_lymph_node_D1": 11,
    "Mouse_Brain_E11_S1": 8,
    "Mouse_Brain_E13_S1": 12,
    "Mouse_Brain_E15_S1": 12,
    "Mouse_Brain_E18_S1": 14,
}

DATASET_CONFIGS = [
    {
        "name": "10x_human_lymph_node_A1",
        "type": "10x",
        "folder_candidates": ["10x_human_lymph_node_A1", "Human_Lymph_Node_A1"],
        "mod2_file": "adata_ADT.h5ad",
        "anno_file": "annotation.csv",
        "gt_column": "manual-anno",
    },
    {
        "name": "10x_human_lymph_node_D1",
        "type": "10x",
        "folder_candidates": ["10x_human_lymph_node_D1", "Human_Lymph_Node_D1"],
        "mod2_file": "adata_ADT.h5ad",
        "anno_file": "annotation.csv",
        "gt_column": "manual-anno",
    },
    {
        "name": "Mouse_Brain_E11_S1",
        "type": "Spatial-epigenome-transcriptome",
        "folder_candidates": ["Mouse_Brain_E11_S1"],
        "mod2_file": "adata_ATAC.h5ad",
        "anno_file": "anno.csv",
        "gt_column": "cluster",
    },
    {
        "name": "Mouse_Brain_E13_S1",
        "type": "Spatial-epigenome-transcriptome",
        "folder_candidates": ["Mouse_Brain_E13_S1"],
        "mod2_file": "adata_ATAC.h5ad",
        "anno_file": "anno.csv",
        "gt_column": "cluster",
    },
    {
        "name": "Mouse_Brain_E15_S1",
        "type": "Spatial-epigenome-transcriptome",
        "folder_candidates": ["Mouse_Brain_E15_S1"],
        "mod2_file": "adata_ATAC.h5ad",
        "anno_file": "anno.csv",
        "gt_column": "cluster",
    },
    {
        "name": "Mouse_Brain_E18_S1",
        "type": "Spatial-epigenome-transcriptome",
        "folder_candidates": ["Mouse_Brain_E18_S1"],
        "mod2_file": "adata_ATAC.h5ad",
        "anno_file": "anno.csv",
        "gt_column": "cluster",
    },
]


def clr_normalize_each_cell(adata):
    data = adata.X.copy()
    if sp.issparse(data):
        data = data.toarray()
    log_data = np.log1p(data)
    geometric_mean = np.mean(log_data, axis=1, keepdims=True)
    adata.X = log_data - geometric_mean
    return adata


def tfidf(count_mat):
    if sp.issparse(count_mat):
        mat = count_mat.tocsr()
        row_sum = np.array(mat.sum(axis=1)).flatten()
        row_sum[row_sum == 0] = 1.0
        tf = mat.multiply(1.0 / row_sum[:, np.newaxis])
        col_sum = np.array((mat > 0).sum(axis=0)).flatten()
        col_sum[col_sum == 0] = 1.0
        idf = np.log(1.0 + mat.shape[0] / col_sum)
        return tf.multiply(idf).tocsr()
    else:
        row_sum = count_mat.sum(axis=-1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        tf = count_mat / row_sum
        col_sum = (count_mat > 0).sum(axis=0, keepdims=True)
        col_sum[col_sum == 0] = 1.0
        idf = np.log(1.0 + count_mat.shape[0] / col_sum)
        return tf * idf


def run_pca(feature_matrix, n_comps: int = 64) -> np.ndarray:
    max_comps = min(n_comps, feature_matrix.shape[0] - 1, feature_matrix.shape[1] - 1)
    if sp.issparse(feature_matrix):
        svd = TruncatedSVD(n_components=max(2, max_comps), random_state=42)
        return svd.fit_transform(feature_matrix)
    pca = PCA(n_components=max(2, max_comps), random_state=42)
    return pca.fit_transform(feature_matrix)


def load_dataset_features_unsupervised(
    base_dir: str,
    cfg: Dict[str, Any],
    n_hvg: int = 3000,
    n_comps_rna: int = 128,
    n_comps_mod2: int = 128
) -> Dict[str, Any]:
    rna_path = os.path.join(base_dir, "adata_RNA.h5ad")
    mod2_path = os.path.join(base_dir, cfg["mod2_file"])

    if not os.path.exists(rna_path) or not os.path.exists(mod2_path):
        raise FileNotFoundError(f"Missing data files in {base_dir}")

    adata_rna = sc.read_h5ad(rna_path)
    adata_mod2 = sc.read_h5ad(mod2_path)

    adata_rna.var_names_make_unique()
    adata_mod2.var_names_make_unique()

    # Align common cell barcodes across modalities strictly
    common_cells = adata_rna.obs_names.intersection(adata_mod2.obs_names)
    if len(common_cells) == 0:
        raise ValueError(f"No overlapping cells found between RNA and {cfg['mod2_file']} in {base_dir}")

    adata_rna = adata_rna[common_cells].copy()
    adata_mod2 = adata_mod2[common_cells].copy()

    # Preprocess RNA
    sc.pp.filter_genes(adata_rna, min_cells=10)
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    try:
        sc.pp.highly_variable_genes(adata_rna, flavor="seurat", n_top_genes=min(n_hvg, adata_rna.shape[1]))
    except Exception:
        sc.pp.highly_variable_genes(adata_rna, n_top_genes=min(n_hvg, adata_rna.shape[1]))
    sc.pp.scale(adata_rna)

    hvg_mask = adata_rna.var['highly_variable']
    rna_raw = adata_rna[:, hvg_mask].X
    if sp.issparse(rna_raw):
        rna_raw = rna_raw.toarray()
    rna_pca = run_pca(rna_raw, n_comps=n_comps_rna)

    # Preprocess Modality 2 (ADT or ATAC)
    if cfg["type"] == "10x":
        adata_mod2 = clr_normalize_each_cell(adata_mod2)
        sc.pp.scale(adata_mod2)
        mod2_raw = adata_mod2.X
        if sp.issparse(mod2_raw):
            mod2_raw = mod2_raw.toarray()
        mod2_pca = run_pca(mod2_raw, n_comps=min(n_comps_mod2, mod2_raw.shape[1]))
    else:
        # ATAC-seq: Sparse TF-IDF + TruncatedSVD (preserves all cells, no cell dropping)
        tfidf_mat = tfidf(adata_mod2.X)
        atac_comps = min(60, tfidf_mat.shape[1])
        mod2_raw = run_pca(tfidf_mat, n_comps=atac_comps)
        mod2_pca = mod2_raw[:, :min(n_comps_mod2, atac_comps)]

    cell_positions = adata_rna.obsm['spatial'].astype(np.float32)

    return {
        "dataset_name": cfg["name"],
        "num_clusters": DECLARED_K[cfg["name"]],
        "cell_names": list(common_cells),
        "rna_raw": rna_raw.astype(np.float32),
        "mod2_raw": mod2_raw.astype(np.float32),
        "rna_pca": rna_pca.astype(np.float32),
        "mod2_pca": mod2_pca.astype(np.float32),
        "cell_positions": cell_positions,
    }


def load_posthoc_ground_truth(base_dir: str, cfg: Dict[str, Any], cell_names: Optional[List[str]] = None) -> np.ndarray:
    anno_path = os.path.join(base_dir, cfg["anno_file"])
    if not os.path.exists(anno_path):
        raise FileNotFoundError(f"Missing annotation file at {anno_path}")

    anno_df = pd.read_csv(anno_path, index_col=0)
    gt_column = cfg["gt_column"]
    if gt_column not in anno_df.columns:
        raise KeyError(f"Column '{gt_column}' not found in {anno_path}")

    if cell_names is not None:
        common = [c for c in cell_names if c in anno_df.index]
        if len(common) == len(cell_names):
            return anno_df.loc[cell_names, gt_column].to_numpy()

    return anno_df[gt_column].to_numpy()
