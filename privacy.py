"""
隐私保护机制实现
包含差分隐私（拉普拉斯机制和高斯机制）
"""

import numpy as np
from typing import List
from geometry import Point2D, LineModel, CircleModel
from ransac import RANSAC
from lsc import LSC


class PrivacyProtection:
    """隐私保护机制"""
    
    def __init__(self, epsilon=1.0, delta=1e-5, sensitivity=1.0):
        """
        初始化隐私保护参数
        
        Args:
            epsilon: 隐私预算
            delta: 松弛参数（用于高斯机制）
            sensitivity: 敏感度
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
    
    def add_laplace_noise(self, points: List[Point2D], scale: float) -> List[Point2D]:
        """添加拉普拉斯噪声（差分隐私）"""
        noisy_points = []
        for point in points:
            noise_x = np.random.laplace(0, scale)
            noise_y = np.random.laplace(0, scale)
            noisy_points.append(Point2D(point.x + noise_x, point.y + noise_y))
        return noisy_points
    
    def add_gaussian_noise(self, points: List[Point2D], sigma: float) -> List[Point2D]:
        """添加高斯噪声（近似差分隐私）"""
        noisy_points = []
        for point in points:
            noise_x = np.random.normal(0, sigma)
            noise_y = np.random.normal(0, sigma)
            noisy_points.append(Point2D(point.x + noise_x, point.y + noise_y))
        return noisy_points
    
    def compute_laplace_scale(self, sensitivity: float = None) -> float:
        """计算拉普拉斯机制所需的噪声尺度"""
        if sensitivity is None:
            sensitivity = self.sensitivity
        return sensitivity / self.epsilon
    
    def compute_gaussian_sigma(self, sensitivity: float = None) -> float:
        """计算高斯机制所需的噪声标准差"""
        if sensitivity is None:
            sensitivity = self.sensitivity
        # 对于(epsilon, delta)-差分隐私
        c = np.sqrt(2.0 * np.log(1.25 / self.delta))
        return c * sensitivity / self.epsilon
    
    def perturb_line_model(self, model: LineModel, noise_scale: float) -> LineModel:
        """对直线模型参数添加噪声"""
        noise_a = np.random.laplace(0, noise_scale)
        noise_b = np.random.laplace(0, noise_scale)
        noise_c = np.random.laplace(0, noise_scale)
        return LineModel(
            model.a + noise_a,
            model.b + noise_b,
            model.c + noise_c
        )
    
    def perturb_circle_model(self, model: CircleModel, noise_scale: float) -> CircleModel:
        """对圆模型参数添加噪声"""
        noise_cx = np.random.laplace(0, noise_scale)
        noise_cy = np.random.laplace(0, noise_scale)
        noise_r = np.random.laplace(0, noise_scale)
        return CircleModel(
            model.cx + noise_cx,
            model.cy + noise_cy,
            max(0.0, model.r + noise_r)
        )


class PrivateRANSAC:
    """隐私保护的RANSAC包装器"""
    
    def __init__(self, ransac_params: dict, privacy_params: dict):
        """
        初始化隐私保护的RANSAC
        
        Args:
            ransac_params: RANSAC参数字典
            privacy_params: 隐私参数字典
        """
        self.ransac = RANSAC(**ransac_params)
        self.privacy = PrivacyProtection(**privacy_params)
        self.noise_multiplier = privacy_params.get("noise_multiplier", 1.0)

    def get_noisy_points(self, points: List[Point2D], mechanism: str = "laplace") -> List[Point2D]:
        """返回隐私处理后的点集，便于可视化对比"""
        if mechanism == "gaussian":
            sigma = self.privacy.compute_gaussian_sigma() * self.noise_multiplier
            return self.privacy.add_gaussian_noise(points, sigma)
        scale = self.privacy.compute_laplace_scale() * self.noise_multiplier
        return self.privacy.add_laplace_noise(points, scale)
    
    def fit_line(self, points: List[Point2D]):
        """隐私保护的直线拟合"""
        noisy_points = self.get_noisy_points(points, mechanism="laplace")
        # 使用RANSAC拟合
        return self.ransac.fit_line(noisy_points)
    
    def fit_circle(self, points: List[Point2D]):
        """隐私保护的圆拟合"""
        noisy_points = self.get_noisy_points(points, mechanism="laplace")
        # 使用RANSAC拟合
        return self.ransac.fit_circle(noisy_points)


class PrivateLSC:
    """隐私保护的LSC包装器"""
    
    def __init__(self, lsc_params: dict, privacy_params: dict):
        """
        初始化隐私保护的LSC
        
        Args:
            lsc_params: LSC参数字典
            privacy_params: 隐私参数字典
        """
        self.lsc = LSC(**lsc_params)
        self.privacy = PrivacyProtection(**privacy_params)
        self.noise_multiplier = privacy_params.get("noise_multiplier", 1.0)

    def get_noisy_points(self, points: List[Point2D], mechanism: str = "laplace") -> List[Point2D]:
        """返回隐私处理后的点集，便于可视化对比"""
        if mechanism == "gaussian":
            sigma = self.privacy.compute_gaussian_sigma() * self.noise_multiplier
            return self.privacy.add_gaussian_noise(points, sigma)
        scale = self.privacy.compute_laplace_scale() * self.noise_multiplier
        return self.privacy.add_laplace_noise(points, scale)
    
    def fit_line(self, points: List[Point2D]):
        """隐私保护的直线拟合"""
        noisy_points = self.get_noisy_points(points, mechanism="laplace")
        # 使用LSC拟合
        return self.lsc.fit_line(noisy_points)
    
    def fit_circle(self, points: List[Point2D]):
        """隐私保护的圆拟合"""
        noisy_points = self.get_noisy_points(points, mechanism="laplace")
        # 使用LSC拟合
        return self.lsc.fit_circle(noisy_points)
