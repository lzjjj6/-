"""
Privacy-preserving DBSCAN聚类算法实现
基于论文：PPA-DBSCAN: Privacy-Preserving ρ-Approximate Density-Based Clustering
"""

import numpy as np
from typing import List, Set, Tuple
from geometry import Point2D
from sklearn.neighbors import NearestNeighbors


class PrivacyPreservingDBSCAN:

    def __init__(self, eps: float, min_samples: int, 
                 epsilon: float = 1.0, rho: float = 0.1):
        self.eps = eps
        self.min_samples = min_samples
        self.epsilon = epsilon
        self.rho = rho
        
    def fit(self, points: List[Point2D]) -> Tuple[List[int], List[List[int]]]:
        if len(points) == 0:
            return [], []
        X = np.array([[p.x, p.y] for p in points])
        n_points = len(points)
        noisy_distances = self._compute_noisy_distances(X)
        core_points = self._identify_core_points_with_privacy(
            X, noisy_distances
        )
        labels = np.full(n_points, -1)  # -1表示噪声点
        cluster_id = 0
        visited = set()
        clusters = []
        for i in range(n_points):
            if i in visited or i not in core_points:
                continue
            cluster = []
            self._expand_cluster(
                i, X, noisy_distances, core_points, 
                visited, labels, cluster_id, cluster
            )
            if len(cluster) > 0:
                clusters.append(cluster)
                cluster_id += 1       
        return labels.tolist(), clusters

    def get_private_points(self, points: List[Point2D]) -> List[Point2D]:
        if len(points) == 0:
            return []

        X = np.array([[p.x, p.y] for p in points], dtype=float)
        x_min, y_min = X.min(axis=0)
        x_max, y_max = X.max(axis=0)
        diag = float(np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2))
        sensitivity = max(diag, 1e-6)
        scale = sensitivity / (self.epsilon * 0.5)
        noisy = []
        for p in points:
            nx = p.x + np.random.laplace(0.0, scale)
            ny = p.y + np.random.laplace(0.0, scale)
            noisy.append(Point2D(nx, ny))
        return noisy
    
    def _compute_noisy_distances(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(X[i] - X[j])
                distances[i, j] = dist
                distances[j, i] = dist
        sensitivity = np.max(distances) if n > 1 else 1.0
        noise_scale = sensitivity / (self.epsilon * 0.5)
        noisy_distances = distances.copy()
        for i in range(n):
            for j in range(i + 1, n):
                noise = np.random.laplace(0, noise_scale)
                noisy_distances[i, j] = distances[i, j] + noise
                noisy_distances[j, i] = noisy_distances[i, j]
        adjusted_eps = self.eps * (1 + self.rho)
        return noisy_distances
    
    def _identify_core_points_with_privacy(
        self, X: np.ndarray, noisy_distances: np.ndarray
    ) -> Set[int]:
        n = len(X)
        core_points = set()
        neighbor_counts = []
        for i in range(n):
            adjusted_eps = self.eps * (1 + self.rho)
            neighbors = np.sum(noisy_distances[i] <= adjusted_eps) - 1
            neighbor_counts.append(neighbors)
        noise_scale = 1.0 / (self.epsilon * 0.5)
        noisy_counts = []
        for count in neighbor_counts:
            noise = np.random.laplace(0, noise_scale)
            noisy_count = max(0, int(count + noise))  # 确保非负
            noisy_counts.append(noisy_count)
        for i, noisy_count in enumerate(noisy_counts):
            if noisy_count >= self.min_samples:
                core_points.add(i)
        return core_points
    
    def _expand_cluster(
        self, point_idx: int, X: np.ndarray, 
        noisy_distances: np.ndarray, core_points: Set[int],
        visited: Set[int], labels: np.ndarray, 
        cluster_id: int, cluster: List[int]
    ):
        cluster.append(point_idx)
        labels[point_idx] = cluster_id
        visited.add(point_idx)
        adjusted_eps = self.eps * (1 + self.rho)
        neighbors = [
            j for j in range(len(X))
            if j != point_idx and noisy_distances[point_idx, j] <= adjusted_eps
        ]
        for neighbor_idx in neighbors:
            if neighbor_idx not in visited:
                visited.add(neighbor_idx)
                if neighbor_idx in core_points:
                    self._expand_cluster(
                        neighbor_idx, X, noisy_distances, core_points,
                        visited, labels, cluster_id, cluster
                    )
                else:
                    labels[neighbor_idx] = cluster_id
                    cluster.append(neighbor_idx)


class DBSCANRANSAC:
    def __init__(self, ransac_params: dict, dbscan_params: dict):
        from ransac import RANSAC
        self.ransac = RANSAC(**ransac_params)
        self.dbscan = PrivacyPreservingDBSCAN(**dbscan_params)
    
    def fit_line(self, points: List[Point2D]):
        from geometry import FittingResult
        labels, clusters = self.dbscan.fit(points)
        if len(clusters) > 0:
            largest_cluster = max(clusters, key=len)
            candidate_points = [points[i] for i in largest_cluster]
        else:
            candidate_points = points
        result = self.ransac.fit_line(candidate_points)
        if len(clusters) > 0:
            original_inliers = []
            for i, point in enumerate(points):
                if result.model.distance(point) <= self.ransac.threshold:
                    original_inliers.append(i)
            result.inliers = original_inliers
            result.score = len(original_inliers) / len(points)
        return result
    
    def fit_circle(self, points: List[Point2D]):
        from geometry import FittingResult
        labels, clusters = self.dbscan.fit(points)
        if len(clusters) > 0:
            largest_cluster = max(clusters, key=len)
            candidate_points = [points[i] for i in largest_cluster]
        else:
            candidate_points = points
        result = self.ransac.fit_circle(candidate_points)
        if len(clusters) > 0:
            original_inliers = []
            for i, point in enumerate(points):
                if result.model.distance(point) <= self.ransac.threshold:
                    original_inliers.append(i)
            result.inliers = original_inliers
            result.score = len(original_inliers) / len(points)
        
        return result


class DBSCANLSC:
    def __init__(self, lsc_params: dict, dbscan_params: dict):
        from lsc import LSC
        self.lsc = LSC(**lsc_params)
        self.dbscan = PrivacyPreservingDBSCAN(**dbscan_params)
    
    def fit_line(self, points: List[Point2D]):
        from geometry import FittingResult
        labels, clusters = self.dbscan.fit(points)
        best_result = FittingResult()
        
        if len(clusters) > 0:
            for cluster_indices in clusters:
                cluster_points = [points[i] for i in cluster_indices]
                if len(cluster_points) < 2:
                    continue
                result = self.lsc.fit_line(cluster_points)
                all_inliers = []
                for i, point in enumerate(points):
                    if result.model.distance(point) <= self.lsc.threshold:
                        all_inliers.append(i)                
                result.inliers = all_inliers
                result.score = len(all_inliers) / len(points)
                
                if result.score > best_result.score:
                    best_result = result
        else:
            best_result = self.lsc.fit_line(points)
        
        return best_result
    
    def fit_circle(self, points: List[Point2D]):
        from geometry import FittingResult
        labels, clusters = self.dbscan.fit(points)
        best_result = FittingResult()
        if len(clusters) > 0:
            for cluster_indices in clusters:
                cluster_points = [points[i] for i in cluster_indices]
                if len(cluster_points) < 3:
                    continue
                result = self.lsc.fit_circle(cluster_points)
                all_inliers = []
                for i, point in enumerate(points):
                    if result.model.distance(point) <= self.lsc.threshold:
                        all_inliers.append(i)               
                result.inliers = all_inliers
                result.score = len(all_inliers) / len(points)                
                if result.score > best_result.score:
                    best_result = result
        else:
            best_result = self.lsc.fit_circle(points)       
        return best_result