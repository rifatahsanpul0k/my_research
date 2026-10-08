from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

def evaluate_model(model: torch.nn.Module, data: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        sim_z, dist_z, fused_z, fused_pro, pro = model(
            data.x_RNA, data.x_ADT,
            data.sim_edge_index, data.sim_edge_weight,
            data.dist_edge_index, data.dist_edge_weight,
            data.common_edge_index, data.common_edge_weight
        )
    return fused_pro.cpu().numpy(), sim_z.cpu().numpy(), dist_z.cpu().numpy()

def cluster_embeddings(embeddings: np.ndarray, num_clusters: int, seed: int = 42) -> np.ndarray:
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=seed)
    return kmeans.fit_predict(embeddings)

def train_model(model: torch.nn.Module, data: Any, args: Any, selection_callback: Optional[Any] = None) -> Tuple[torch.nn.Module, np.ndarray, np.ndarray]:
    if hasattr(args, "true_labels") or hasattr(args, "ground_truth"):
        raise PermissionError("FIREWALL VIOLATION: ground_truth found on args passed to train_model!")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    best_sil = -1.0
    best_embeddings = None
    best_labels = None
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        sim_z, dist_z, fused_z, fused_pro, pro = model(
            data.x_RNA, data.x_ADT,
            data.sim_edge_index, data.sim_edge_weight,
            data.dist_edge_index, data.dist_edge_weight,
            data.common_edge_index, data.common_edge_weight
        )
        loss, l_rec = model.compute_losses(
            data.x_RNA, data.x_ADT, sim_z, dist_z, fused_z, fused_pro, torch.cat([data.x_RNA, data.x_ADT], dim=1), pro
        )
        loss.backward()
        optimizer.step()
        embeddings, _, _ = evaluate_model(model, data)
        predicted_labels = cluster_embeddings(embeddings, args.num_clusters)
        sil = silhouette_score(embeddings, predicted_labels)
        if sil > best_sil:
            best_sil = sil
            best_embeddings = embeddings.copy()
            best_labels = predicted_labels.copy()
    return model, best_embeddings, best_labels

def evaluate_posthoc_quarantined(predicted_labels: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(ground_truth).astype(str)
    y_pred = np.asarray(predicted_labels).astype(str)
    return {
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "quarantined_reporting_only": True
    }
