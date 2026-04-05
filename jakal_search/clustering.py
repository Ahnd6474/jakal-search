from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN

from .types import ClusterResult
from .utils import mean_embedding


class DensityClusterer:
    def __init__(self, min_cluster_size: int, eps: float = 0.28) -> None:
        self.min_cluster_size = min_cluster_size
        self.eps = eps
        self._eps_candidates = [eps, 0.34, 0.40, 0.46]
        self._hdbscan = None
        try:
            import hdbscan  # type: ignore

            self._hdbscan = hdbscan
        except ImportError:
            self._hdbscan = None

    def cluster(self, embeddings: np.ndarray) -> list[ClusterResult]:
        if embeddings.size == 0 or len(embeddings) < self.min_cluster_size:
            return []

        if self._hdbscan is not None and len(embeddings) >= self.min_cluster_size + 1:
            model = self._hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=max(2, self.min_cluster_size // 2),
                metric="euclidean",
            )
            labels = model.fit_predict(embeddings)
        else:
            for candidate_eps in self._eps_candidates:
                model = DBSCAN(
                    eps=candidate_eps,
                    metric="cosine",
                    min_samples=self.min_cluster_size,
                )
                labels = model.fit_predict(embeddings)
                clusters = self._collect_clusters(labels, embeddings)
                if clusters:
                    return clusters
            return []

        return self._collect_clusters(labels, embeddings)

    def _collect_clusters(self, labels: np.ndarray, embeddings: np.ndarray) -> list[ClusterResult]:
        clusters: list[ClusterResult] = []
        for cluster_id in sorted(label for label in set(labels) if label != -1):
            indexes = np.where(labels == cluster_id)[0].tolist()
            centroid = mean_embedding(embeddings[indexes])
            if centroid is None:
                continue
            clusters.append(ClusterResult(cluster_id=cluster_id, member_indexes=indexes, centroid=centroid))
        return clusters
