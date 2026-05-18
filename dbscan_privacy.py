"""
Privacy-preserving DBSCAN聚类算法实现
基于论文：PPA-DBSCAN: Privacy-Preserving ρ-Approximate Density-Based Clustering
"""

import numpy as np
from typing import List, Set, Tuple
from geometry import Point2D
from sklearn.neighbors import NearestNeighbors


class PrivacyPreservingDBSCAN:
    """
    隐私保护DBSCAN聚类算法
    
    改进点1：使用ρ-近似DBSCAN，在保证隐私的同时提高聚类质量
    改进点2：在距离计算和密度估计中应用差分隐私，保护聚类过程
    """
    
    def __init__(self, eps: float, min_samples: int, 
                 epsilon: float = 1.0, rho: float = 0.1):
        """
        初始化隐私保护DBSCAN参数
        
        Args:
            eps: 邻域半径
            min_samples: 最小样本数（形成核心点）
            epsilon: 隐私预算（差分隐私）
            rho: 近似参数（ρ-近似DBSCAN）
        """
        self.eps = eps
        self.min_samples = min_samples
        self.epsilon = epsilon
        self.rho = rho
        
    def fit(self, points: List[Point2D]) -> Tuple[List[int], List[List[int]]]:
        """
        执行隐私保护DBSCAN聚类
        
        Returns:
            labels: 每个点的标签（-1表示噪声点）
            clusters: 聚类列表，每个聚类包含点的索引
        """
        if len(points) == 0:
            return [], []
        
        # 转换为numpy数组
        X = np.array([[p.x, p.y] for p in points])
        n_points = len(points)
        
        # 改进1：使用ρ-近似距离计算，减少隐私泄露
        # 在距离计算中添加噪声，实现ρ-近似
        noisy_distances = self._compute_noisy_distances(X)
        
        # 改进2：在密度估计中应用差分隐私
        # 计算每个点的噪声邻居数
        core_points = self._identify_core_points_with_privacy(
            X, noisy_distances
        )
        
        # 构建聚类
        labels = np.full(n_points, -1)  # -1表示噪声点
        cluster_id = 0
        
        visited = set()
        clusters = []
        
        for i in range(n_points):
            if i in visited or i not in core_points:
                continue
            
            # 开始新聚类
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
        """
        生成 PP-DBSCAN 对应的“隐私化点集”（用于可视化展示）。

        说明：
        - PPA-DBSCAN 主要隐私化的是距离/密度统计；这里额外输出一个坐标扰动版本，
          便于在报告中直观看到“PP-DBSCAN 隐私化后的点集效果”。
        - 使用与 DBSCAN 相同的 epsilon，并用数据范围估计敏感度后添加拉普拉斯噪声。
        """
        if len(points) == 0:
            return []

        X = np.array([[p.x, p.y] for p in points], dtype=float)
        # 用数据对角线长度作为粗略敏感度估计（便于展示差异）
        x_min, y_min = X.min(axis=0)
        x_max, y_max = X.max(axis=0)
        diag = float(np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2))
        sensitivity = max(diag, 1e-6)

        # 噪声尺度：与 fit 中距离噪声同样采用 0.5 的预算比例（保持一致性）
        scale = sensitivity / (self.epsilon * 0.5)

        noisy = []
        for p in points:
            nx = p.x + np.random.laplace(0.0, scale)
            ny = p.y + np.random.laplace(0.0, scale)
            noisy.append(Point2D(nx, ny))
        return noisy
    
    def _compute_noisy_distances(self, X: np.ndarray) -> np.ndarray:
        """
        改进1：计算ρ-近似噪声距离矩阵
        
        使用差分隐私保护距离信息，实现ρ-近似DBSCAN
        """
        n = len(X)
        distances = np.zeros((n, n))
        
        # 计算真实距离
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(X[i] - X[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        # 添加拉普拉斯噪声实现ρ-近似
        # 敏感度：最大可能距离（这里简化为数据范围）
        sensitivity = np.max(distances) if n > 1 else 1.0
        
        # 计算噪声尺度（使用部分隐私预算）
        noise_scale = sensitivity / (self.epsilon * 0.5)
        
        # 添加噪声（只对非对角线元素）
        noisy_distances = distances.copy()
        for i in range(n):
            for j in range(i + 1, n):
                noise = np.random.laplace(0, noise_scale)
                noisy_distances[i, j] = distances[i, j] + noise
                noisy_distances[j, i] = noisy_distances[i, j]
        
        # ρ-近似：调整eps以适应噪声
        adjusted_eps = self.eps * (1 + self.rho)
        
        return noisy_distances
    
    def _identify_core_points_with_privacy(
        self, X: np.ndarray, noisy_distances: np.ndarray
    ) -> Set[int]:
        """
        改进2：使用隐私保护的密度估计识别核心点
        
        在计算邻居数时应用差分隐私，保护密度信息
        """
        n = len(X)
        core_points = set()
        
        # 计算每个点的邻居数（使用噪声距离）
        neighbor_counts = []
        for i in range(n):
            # 计算在调整后的eps范围内的邻居数
            adjusted_eps = self.eps * (1 + self.rho)
            neighbors = np.sum(noisy_distances[i] <= adjusted_eps) - 1  # 排除自己
            neighbor_counts.append(neighbors)
        
        # 对邻居数添加拉普拉斯噪声（差分隐私）
        # 敏感度为1（添加或删除一个点最多改变1个邻居）
        noise_scale = 1.0 / (self.epsilon * 0.5)
        
        noisy_counts = []
        for count in neighbor_counts:
            noise = np.random.laplace(0, noise_scale)
            noisy_count = max(0, int(count + noise))  # 确保非负
            noisy_counts.append(noisy_count)
        
        # 识别核心点（噪声邻居数 >= min_samples）
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
        """扩展聚类"""
        cluster.append(point_idx)
        labels[point_idx] = cluster_id
        visited.add(point_idx)
        
        # 获取邻居
        adjusted_eps = self.eps * (1 + self.rho)
        neighbors = [
            j for j in range(len(X))
            if j != point_idx and noisy_distances[point_idx, j] <= adjusted_eps
        ]
        
        # 递归处理邻居
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
    """
    改进的RANSAC：使用DBSCAN预聚类提高内点识别
    
    改进点1：使用DBSCAN进行预聚类，识别潜在的内点区域
    改进点2：在DBSCAN聚类过程中应用隐私保护
    """
    
    def __init__(self, ransac_params: dict, dbscan_params: dict):
        """
        Args:
            ransac_params: RANSAC参数字典
            dbscan_params: DBSCAN参数字典（包含eps, min_samples, epsilon等）
        """
        from ransac import RANSAC
        self.ransac = RANSAC(**ransac_params)
        self.dbscan = PrivacyPreservingDBSCAN(**dbscan_params)
    
    def fit_line(self, points: List[Point2D]):
        """使用DBSCAN增强的RANSAC拟合直线"""
        from geometry import FittingResult
        
        # 改进1：使用DBSCAN预聚类
        labels, clusters = self.dbscan.fit(points)
        
        # 选择最大的聚类作为潜在内点区域
        if len(clusters) > 0:
            largest_cluster = max(clusters, key=len)
            candidate_points = [points[i] for i in largest_cluster]
        else:
            # 如果没有聚类，使用所有点
            candidate_points = points
        
        # 改进2：在候选点上应用RANSAC
        result = self.ransac.fit_line(candidate_points)
        
        # 映射回原始点索引
        if len(clusters) > 0:
            # 重新评估所有点
            original_inliers = []
            for i, point in enumerate(points):
                if result.model.distance(point) <= self.ransac.threshold:
                    original_inliers.append(i)
            result.inliers = original_inliers
            result.score = len(original_inliers) / len(points)
        
        return result
    
    def fit_circle(self, points: List[Point2D]):
        """使用DBSCAN增强的RANSAC拟合圆"""
        from geometry import FittingResult
        
        # 使用DBSCAN预聚类
        labels, clusters = self.dbscan.fit(points)
        
        # 选择最大的聚类
        if len(clusters) > 0:
            largest_cluster = max(clusters, key=len)
            candidate_points = [points[i] for i in largest_cluster]
        else:
            candidate_points = points
        
        # 应用RANSAC
        result = self.ransac.fit_circle(candidate_points)
        
        # 映射回原始点索引
        if len(clusters) > 0:
            original_inliers = []
            for i, point in enumerate(points):
                if result.model.distance(point) <= self.ransac.threshold:
                    original_inliers.append(i)
            result.inliers = original_inliers
            result.score = len(original_inliers) / len(points)
        
        return result


class DBSCANLSC:
    """
    改进的LSC：使用DBSCAN预聚类提高模型选择
    
    改进点1：使用DBSCAN识别多个结构，提高多结构拟合能力
    改进点2：在聚类过程中保护数据隐私
    """
    
    def __init__(self, lsc_params: dict, dbscan_params: dict):
        """
        Args:
            lsc_params: LSC参数字典
            dbscan_params: DBSCAN参数字典
        """
        from lsc import LSC
        self.lsc = LSC(**lsc_params)
        self.dbscan = PrivacyPreservingDBSCAN(**dbscan_params)
    
    def fit_line(self, points: List[Point2D]):
        """使用DBSCAN增强的LSC拟合直线"""
        from geometry import FittingResult
        
        # 改进1：使用DBSCAN识别多个结构
        labels, clusters = self.dbscan.fit(points)
        
        # 对每个聚类分别拟合，选择最佳结果
        best_result = FittingResult()
        
        if len(clusters) > 0:
            for cluster_indices in clusters:
                cluster_points = [points[i] for i in cluster_indices]
                if len(cluster_points) < 2:
                    continue
                
                # 对聚类应用LSC
                result = self.lsc.fit_line(cluster_points)
                
                # 评估在整个数据集上的表现
                all_inliers = []
                for i, point in enumerate(points):
                    if result.model.distance(point) <= self.lsc.threshold:
                        all_inliers.append(i)
                
                result.inliers = all_inliers
                result.score = len(all_inliers) / len(points)
                
                if result.score > best_result.score:
                    best_result = result
        else:
            # 如果没有聚类，直接使用LSC
            best_result = self.lsc.fit_line(points)
        
        return best_result
    
    def fit_circle(self, points: List[Point2D]):
        """使用DBSCAN增强的LSC拟合圆"""
        from geometry import FittingResult
        
        # 使用DBSCAN识别多个结构
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
