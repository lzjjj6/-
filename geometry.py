"""
几何模型定义模块
包含点、直线、圆等基本几何结构
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class Point2D:
    """2D点"""
    x: float
    y: float
    
    def to_array(self) -> np.ndarray:
        """转换为numpy数组"""
        return np.array([self.x, self.y])
    
    def distance(self, other: 'Point2D') -> float:
        """计算到另一点的距离"""
        dx = self.x - other.x
        dy = self.y - other.y
        return np.sqrt(dx * dx + dy * dy)


@dataclass
class LineModel:
    """直线模型: ax + by + c = 0"""
    a: float
    b: float
    c: float
    
    def __post_init__(self):
        """归一化参数"""
        self.normalize()
    
    def normalize(self):
        """归一化直线参数"""
        norm = np.sqrt(self.a * self.a + self.b * self.b)
        if norm > 1e-10:
            self.a /= norm
            self.b /= norm
            self.c /= norm
    
    def distance(self, point: Point2D) -> float:
        """计算点到直线的距离"""
        return abs(self.a * point.x + self.b * point.y + self.c)
    
    @staticmethod
    def from_points(p1: Point2D, p2: Point2D) -> 'LineModel':
        """从两个点拟合直线"""
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        a = -dy
        b = dx
        c = -(a * p1.x + b * p1.y)
        return LineModel(a, b, c)


@dataclass
class CircleModel:
    """圆模型: (x - cx)^2 + (y - cy)^2 = r^2"""
    cx: float
    cy: float
    r: float
    
    def distance(self, point: Point2D) -> float:
        """计算点到圆的距离（径向距离）"""
        dx = point.x - self.cx
        dy = point.y - self.cy
        dist = np.sqrt(dx * dx + dy * dy)
        return abs(dist - self.r)
    
    @staticmethod
    def from_points(p1: Point2D, p2: Point2D, p3: Point2D) -> 'CircleModel':
        """从三个点拟合圆"""
        # 使用三点确定圆的几何方法
        # 解线性方程组: 2*cx*x + 2*cy*y + (r^2 - cx^2 - cy^2) = x^2 + y^2
        
        A = np.array([
            [2 * p1.x, 2 * p1.y, 1],
            [2 * p2.x, 2 * p2.y, 1],
            [2 * p3.x, 2 * p3.y, 1]
        ])
        
        b = np.array([
            p1.x * p1.x + p1.y * p1.y,
            p2.x * p2.x + p2.y * p2.y,
            p3.x * p3.x + p3.y * p3.y
        ])
        
        try:
            solution = np.linalg.solve(A, b)
            cx = solution[0]
            cy = solution[1]
            r = np.sqrt(solution[2] + cx * cx + cy * cy)
            return CircleModel(cx, cy, r)
        except np.linalg.LinAlgError:
            # 三点共线，返回默认圆
            return CircleModel(0, 0, 1)


@dataclass
class FittingResult:
    """模型拟合结果"""
    model: object
    inliers: List[int]
    score: float
    iterations: int
    
    def __init__(self, model=None, inliers=None, score=0.0, iterations=0):
        self.model = model
        self.inliers = inliers if inliers is not None else []
        self.score = score
        self.iterations = iterations
