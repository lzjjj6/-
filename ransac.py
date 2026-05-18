import numpy as np
from typing import List, TypeVar, Callable
from geometry import Point2D, LineModel, CircleModel, FittingResult
import random

ModelType = TypeVar('ModelType', LineModel, CircleModel)

class RANSAC:
    def __init__(self, max_iterations=1000, threshold=1.0, min_samples=2, 
                 confidence=0.99, min_inliers=10):
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.min_samples = min_samples
        self.confidence = confidence
        self.min_inliers = min_inliers
    
    def fit_line(self, points: List[Point2D]) -> FittingResult:
        self.min_samples = 2
        return self._fit(points, self._generate_line_model, self._evaluate_line)
    
    def fit_circle(self, points: List[Point2D]) -> FittingResult:
        self.min_samples = 3
        return self._fit(points, self._generate_circle_model, self._evaluate_circle)
    
    def _fit(self, points: List[Point2D], 
             generate_model: Callable,
             evaluate_model: Callable) -> FittingResult:
        best_result = FittingResult()
        if len(points) < self.min_samples:
            return best_result  
        iterations = 0
        max_iter = self.max_iterations      
        while iterations < max_iter:
            sample_indices = random.sample(range(len(points)), self.min_samples)
            try:
                model = generate_model(points, sample_indices)
            except:
                iterations += 1
                continue
            inliers = evaluate_model(model, points)
            if len(inliers) >= self.min_inliers:
                score = len(inliers) / len(points)
                if score > best_result.score:
                    best_result.model = model
                    best_result.inliers = inliers
                    best_result.score = score
                    estimated_iterations = self._compute_iterations(
                        len(inliers), len(points)
                    )
                    if estimated_iterations < max_iter:
                        max_iter = estimated_iterations
            iterations += 1
        best_result.iterations = iterations
        return best_result
    
    def _generate_line_model(self, points: List[Point2D], 
                             indices: List[int]) -> LineModel:
        return LineModel.from_points(points[indices[0]], points[indices[1]])
    
    def _generate_circle_model(self, points: List[Point2D], 
                               indices: List[int]) -> CircleModel:
        return CircleModel.from_points(
            points[indices[0]], 
            points[indices[1]], 
            points[indices[2]]
        )
    
    def _evaluate_line(self, model: LineModel, 
                       points: List[Point2D]) -> List[int]:
        inliers = []
        for i, point in enumerate(points):
            dist = model.distance(point)
            if dist <= self.threshold:
                inliers.append(i)
        return inliers
    
    def _evaluate_circle(self, model: CircleModel, 
                        points: List[Point2D]) -> List[int]:
        inliers = []
        for i, point in enumerate(points):
            dist = model.distance(point)
            if dist <= self.threshold:
                inliers.append(i)
        return inliers
    
    def _compute_iterations(self, inlier_count: int, 
                           total_points: int) -> int:
        if inlier_count == 0:
            return self.max_iterations
        
        inlier_ratio = inlier_count / total_points
        if inlier_ratio >= 1.0:
            return 1
        
        prob_all_outliers = (1.0 - inlier_ratio) ** self.min_samples
        if prob_all_outliers < 1e-10:
            return 1
        
        iterations = int(
            np.log(1.0 - self.confidence) / np.log(prob_all_outliers)
        )
        return min(iterations, self.max_iterations)