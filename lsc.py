"""
LSC算法实现
Latent Semantic Consensus - 潜在语义一致性算法
"""

import numpy as np
from typing import List
from geometry import Point2D, LineModel, CircleModel, FittingResult
from sklearn.cluster import KMeans
import random


class LSC:
    """LSC算法实现（Latent Semantic Consensus）"""
    
    def __init__(self, num_samples=1000, threshold=1.0, min_samples=2,
                 k_nearest=5, max_iterations=100, convergence_threshold=1e-6):
        """
        初始化LSC参数
        
        Args:
            num_samples: 采样模型数量
            threshold: 内点阈值
            min_samples: 最小样本数
            k_nearest: K近邻数量（用于SVD降维）
            max_iterations: 最大迭代次数（用于K-means）
            convergence_threshold: 收敛阈值
        """
        self.num_samples = num_samples
        self.threshold = threshold
        self.min_samples = min_samples
        self.k_nearest = k_nearest
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
    
    def fit_line(self, points: List[Point2D]) -> FittingResult:
        """拟合直线"""
        self.min_samples = 2
        
        # 生成候选模型
        models = self._generate_line_models(points)
        
        # 计算残差矩阵
        residuals = self._compute_line_residual_matrix(points, models)
        
        # 计算潜在语义
        latent_semantics = self._compute_latent_semantics(residuals)
        
        # 聚类模型
        clusters = self._cluster_models(latent_semantics)
        
        # 选择最佳模型
        best_model = self._select_best_line_model(models, points, clusters)
        
        # 评估结果
        result = FittingResult()
        result.model = best_model
        result.inliers = self._evaluate_line_model(best_model, points)
        result.score = len(result.inliers) / len(points)
        result.iterations = self.max_iterations
        
        return result
    
    def fit_circle(self, points: List[Point2D]) -> FittingResult:
        """拟合圆"""
        self.min_samples = 3
        
        # 生成候选模型
        models = self._generate_circle_models(points)
        
        # 计算残差矩阵
        residuals = self._compute_circle_residual_matrix(points, models)
        
        # 计算潜在语义
        latent_semantics = self._compute_latent_semantics(residuals)
        
        # 聚类模型
        clusters = self._cluster_models(latent_semantics)
        
        # 选择最佳模型
        best_model = self._select_best_circle_model(models, points, clusters)
        
        # 评估结果
        result = FittingResult()
        result.model = best_model
        result.inliers = self._evaluate_circle_model(best_model, points)
        result.score = len(result.inliers) / len(points)
        result.iterations = self.max_iterations
        
        return result
    
    def _generate_line_models(self, points: List[Point2D]) -> List[LineModel]:
        """生成候选直线模型"""
        models = []
        for _ in range(self.num_samples):
            indices = random.sample(range(len(points)), self.min_samples)
            try:
                model = LineModel.from_points(
                    points[indices[0]], points[indices[1]]
                )
                models.append(model)
            except:
                continue
        return models
    
    def _generate_circle_models(self, points: List[Point2D]) -> List[CircleModel]:
        """生成候选圆模型"""
        models = []
        for _ in range(self.num_samples):
            indices = random.sample(range(len(points)), self.min_samples)
            try:
                model = CircleModel.from_points(
                    points[indices[0]], 
                    points[indices[1]], 
                    points[indices[2]]
                )
                models.append(model)
            except:
                continue
        return models
    
    def _compute_line_residual_matrix(self, points: List[Point2D], 
                                     models: List[LineModel]) -> np.ndarray:
        """计算直线残差矩阵"""
        residuals = np.zeros((len(points), len(models)))
        for i, point in enumerate(points):
            for j, model in enumerate(models):
                residuals[i, j] = model.distance(point)
        return residuals
    
    def _compute_circle_residual_matrix(self, points: List[Point2D], 
                                        models: List[CircleModel]) -> np.ndarray:
        """计算圆残差矩阵"""
        residuals = np.zeros((len(points), len(models)))
        for i, point in enumerate(points):
            for j, model in enumerate(models):
                residuals[i, j] = model.distance(point)
        return residuals
    
    def _compute_latent_semantics(self, residuals: np.ndarray) -> np.ndarray:
        """计算潜在语义表示（使用SVD）"""
        # 使用SVD进行潜在语义分析
        U, s, Vt = np.linalg.svd(residuals, full_matrices=False)
        
        # 选择前k个主成分
        k = min(self.k_nearest, residuals.shape[1])
        U_k = U[:, :k]
        S_k = np.diag(s[:k])
        
        return U_k @ S_k
    
    def _cluster_models(self, latent_semantics: np.ndarray) -> List[List[int]]:
        """聚类模型（使用K-means）"""
        # 转置，使每列代表一个模型
        X = latent_semantics.T
        
        # 确定聚类数
        k = min(10, X.shape[0])
        
        if k <= 1:
            return [[i for i in range(X.shape[0])]]
        
        # K-means聚类
        kmeans = KMeans(n_clusters=k, max_iter=self.max_iterations, 
                       random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # 组织聚类结果
        clusters = [[] for _ in range(k)]
        for idx, label in enumerate(labels):
            clusters[label].append(idx)
        
        return clusters
    
    def _select_best_line_model(self, models: List[LineModel], 
                                points: List[Point2D],
                                clusters: List[List[int]]) -> LineModel:
        """选择最佳直线模型"""
        best_model = LineModel(0, 0, 0)
        max_inliers = 0
        
        for cluster in clusters:
            if not cluster:
                continue
            
            for model_idx in cluster:
                if model_idx >= len(models):
                    continue
                
                inliers = self._evaluate_line_model(models[model_idx], points)
                if len(inliers) > max_inliers:
                    max_inliers = len(inliers)
                    best_model = models[model_idx]
        
        return best_model
    
    def _select_best_circle_model(self, models: List[CircleModel], 
                                 points: List[Point2D],
                                 clusters: List[List[int]]) -> CircleModel:
        """选择最佳圆模型"""
        best_model = CircleModel(0, 0, 1)
        max_inliers = 0
        
        for cluster in clusters:
            if not cluster:
                continue
            
            for model_idx in cluster:
                if model_idx >= len(models):
                    continue
                
                inliers = self._evaluate_circle_model(models[model_idx], points)
                if len(inliers) > max_inliers:
                    max_inliers = len(inliers)
                    best_model = models[model_idx]
        
        return best_model
    
    def _evaluate_line_model(self, model: LineModel, 
                            points: List[Point2D]) -> List[int]:
        """评估直线模型"""
        inliers = []
        for i, point in enumerate(points):
            dist = model.distance(point)
            if dist <= self.threshold:
                inliers.append(i)
        return inliers
    
    def _evaluate_circle_model(self, model: CircleModel, 
                              points: List[Point2D]) -> List[int]:
        """评估圆模型"""
        inliers = []
        for i, point in enumerate(points):
            dist = model.distance(point)
            if dist <= self.threshold:
                inliers.append(i)
        return inliers
