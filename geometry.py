import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class Point2D:
    x: float
    y: float
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y])
    
    def distance(self, other: 'Point2D') -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return np.sqrt(dx * dx + dy * dy)

@dataclass
class LineModel:
    a: float
    b: float
    c: float
    def __post_init__(self):
        self.normalize()
    
    def normalize(self):
        norm = np.sqrt(self.a * self.a + self.b * self.b)
        if norm > 1e-10:
            self.a /= norm
            self.b /= norm
            self.c /= norm
    
    def distance(self, point: Point2D) -> float:
        return abs(self.a * point.x + self.b * point.y + self.c)
    
    @staticmethod
    def from_points(p1: Point2D, p2: Point2D) -> 'LineModel':
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        a = -dy
        b = dx
        c = -(a * p1.x + b * p1.y)
        return LineModel(a, b, c)


@dataclass
class CircleModel:
    cx: float
    cy: float
    r: float
    
    def distance(self, point: Point2D) -> float:
        dx = point.x - self.cx
        dy = point.y - self.cy
        dist = np.sqrt(dx * dx + dy * dy)
        return abs(dist - self.r)
    
    @staticmethod
    def from_points(p1: Point2D, p2: Point2D, p3: Point2D) -> 'CircleModel':
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
            return CircleModel(0, 0, 1)


@dataclass
class FittingResult:
    model: object
    inliers: List[int]
    score: float
    iterations: int
    
    def __init__(self, model=None, inliers=None, score=0.0, iterations=0):
        self.model = model
        self.inliers = inliers if inliers is not None else []
        self.score = score
        self.iterations = iterations