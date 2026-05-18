"""
工具函数模块
包含数据生成、性能评估等辅助函数
"""

import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from geometry import Point2D, FittingResult


def generate_line_data(num_inliers: int, num_outliers: int, 
                       noise_level: float = 0.1) -> List[Point2D]:
    """
    生成直线测试数据
    
    Args:
        num_inliers: 内点数量
        num_outliers: 外点数量
        noise_level: 噪声水平
    
    Returns:
        点列表
    """
    points = []
    np.random.seed(42)
    
    # 生成内点（在直线 y = x 上）
    for _ in range(num_inliers):
        x = np.random.uniform(-10, 10)
        y = x + np.random.normal(0, noise_level)
        points.append(Point2D(x, y))
    
    # 生成外点
    for _ in range(num_outliers):
        x = np.random.uniform(-10, 10)
        y = np.random.uniform(-10, 10)
        points.append(Point2D(x, y))
    
    return points


def generate_circle_data(num_inliers: int, num_outliers: int,
                        cx: float = 0, cy: float = 0, r: float = 5.0,
                        noise_level: float = 0.1) -> List[Point2D]:
    """
    生成圆测试数据
    
    Args:
        num_inliers: 内点数量
        num_outliers: 外点数量
        cx, cy: 圆心坐标
        r: 半径
        noise_level: 噪声水平
    
    Returns:
        点列表
    """
    points = []
    np.random.seed(42)
    
    # 生成内点（在圆上）
    for _ in range(num_inliers):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = r + np.random.normal(0, noise_level)
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        points.append(Point2D(x, y))
    
    # 生成外点
    for _ in range(num_outliers):
        x = np.random.uniform(-15, 15)
        y = np.random.uniform(-15, 15)
        points.append(Point2D(x, y))
    
    return points


class PerformanceMetrics:
    """性能指标类"""
    
    def __init__(self):
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0
        self.execution_time_ms = 0.0
        self.iterations = 0
    
    def __str__(self):
        return (f"准确率: {self.accuracy:.4f}, "
                f"精确率: {self.precision:.4f}, "
                f"召回率: {self.recall:.4f}, "
                f"F1分数: {self.f1_score:.4f}, "
                f"时间: {self.execution_time_ms:.2f}ms, "
                f"迭代: {self.iterations}")


def evaluate_algorithm(points: List[Point2D], 
                      ground_truth_inliers: List[int],
                      fit_func) -> PerformanceMetrics:
    """
    评估算法性能
    
    Args:
        points: 数据点
        ground_truth_inliers: 真实内点索引
        fit_func: 拟合函数
    
    Returns:
        性能指标
    """
    import time
    
    metrics = PerformanceMetrics()
    
    # 执行拟合
    start_time = time.time()
    result: FittingResult = fit_func(points)
    end_time = time.time()
    
    metrics.execution_time_ms = (end_time - start_time) * 1000
    metrics.iterations = result.iterations
    
    # 计算准确率、精确率、召回率
    predicted_inliers = set(result.inliers)
    true_inliers = set(ground_truth_inliers)
    
    true_positives = len(predicted_inliers & true_inliers)
    false_positives = len(predicted_inliers - true_inliers)
    false_negatives = len(true_inliers - predicted_inliers)
    
    total = true_positives + false_positives + false_negatives
    if total > 0:
        metrics.accuracy = true_positives / total
    
    if true_positives + false_positives > 0:
        metrics.precision = true_positives / (true_positives + false_positives)
    
    if true_positives + false_negatives > 0:
        metrics.recall = true_positives / (true_positives + false_negatives)
    
    if metrics.precision + metrics.recall > 0:
        metrics.f1_score = (2.0 * metrics.precision * metrics.recall / 
                           (metrics.precision + metrics.recall))
    
    return metrics


def compute_displacement_statistics(
    original: List[Point2D], privatized: List[Point2D]
) -> Dict[str, float]:
    """
    计算差分隐私加噪后，每个点相对原始点的欧氏位移统计量。
    用于量化“隐私扰动强度”，与理论拉普拉斯尺度对照。
    """
    if len(original) != len(privatized) or len(original) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    d = []
    for p0, p1 in zip(original, privatized):
        dx = p1.x - p0.x
        dy = p1.y - p0.y
        d.append(np.hypot(dx, dy))
    arr = np.array(d, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


@dataclass
class DPRobustFittingEvaluation:
    """鲁棒拟合 + 差分隐私：效用与扰动综合评估（便于写报告）"""
    task_name: str
    algorithm: str
    epsilon: float
    sensitivity: float
    theoretical_laplace_scale: float
    mean_displacement: float
    median_displacement: float
    p95_displacement: float
    max_displacement: float
    f1_baseline: float
    f1_private: float
    f1_retention_ratio: float
    utility_loss_ratio: float
    precision_baseline: float
    precision_private: float
    recall_baseline: float
    recall_private: float
    precision_delta: float
    recall_delta: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task_name,
            "algorithm": self.algorithm,
            "epsilon": self.epsilon,
            "sensitivity": self.sensitivity,
            "theoretical_laplace_scale": self.theoretical_laplace_scale,
            "mean_displacement": self.mean_displacement,
            "median_displacement": self.median_displacement,
            "p95_displacement": self.p95_displacement,
            "max_displacement": self.max_displacement,
            "f1_baseline": self.f1_baseline,
            "f1_private": self.f1_private,
            "f1_retention_ratio": self.f1_retention_ratio,
            "utility_loss_ratio": self.utility_loss_ratio,
            "precision_delta": self.precision_delta,
            "recall_delta": self.recall_delta,
        }


def evaluate_dp_robust_fitting(
    task_name: str,
    algorithm: str,
    epsilon: float,
    sensitivity: float,
    baseline_metrics: PerformanceMetrics,
    private_metrics: PerformanceMetrics,
    original_points: List[Point2D],
    privatized_points: List[Point2D],
) -> DPRobustFittingEvaluation:
    """
    综合评估：在相同 ground-truth 下，差分隐私对鲁棒拟合效用的影响。

    - theoretical_laplace_scale: 拉普拉斯机制理论噪声尺度 b = Δf/ε
    - f1_retention_ratio: F1_private / F1_baseline（越接近 1 说明隐私下仍保持拟合效用）
    - utility_loss_ratio: 1 - retention（越大说明隐私代价越高）
    - displacement*: 实际加噪位移，可与理论尺度对照（拉普拉斯期望约为 b）
    """
    disp = compute_displacement_statistics(original_points, privatized_points)
    b = sensitivity / epsilon if epsilon > 0 else float("inf")
    f1_b = baseline_metrics.f1_score
    f1_p = private_metrics.f1_score
    retention = (f1_p / f1_b) if f1_b > 1e-12 else 0.0
    return DPRobustFittingEvaluation(
        task_name=task_name,
        algorithm=algorithm,
        epsilon=epsilon,
        sensitivity=sensitivity,
        theoretical_laplace_scale=float(b),
        mean_displacement=disp["mean"],
        median_displacement=disp["median"],
        p95_displacement=disp["p95"],
        max_displacement=disp["max"],
        f1_baseline=f1_b,
        f1_private=f1_p,
        f1_retention_ratio=float(retention),
        utility_loss_ratio=float(1.0 - retention),
        precision_baseline=baseline_metrics.precision,
        precision_private=private_metrics.precision,
        recall_baseline=baseline_metrics.recall,
        recall_private=private_metrics.recall,
        precision_delta=float(private_metrics.precision - baseline_metrics.precision),
        recall_delta=float(private_metrics.recall - baseline_metrics.recall),
    )


def print_dp_robust_evaluations(evaluations: List[DPRobustFittingEvaluation]) -> None:
    """控制台打印评估表，便于实验记录"""
    print("\n" + "=" * 70)
    print("鲁棒拟合 + 差分隐私：综合评估指标")
    print("=" * 70)
    for ev in evaluations:
        print(f"\n[{ev.task_name}] {ev.algorithm}  (ε={ev.epsilon}, Δf={ev.sensitivity})")
        print(f"  理论拉普拉斯尺度 b=Δf/ε: {ev.theoretical_laplace_scale:.4f}")
        print(f"  实际位移: mean={ev.mean_displacement:.4f}, median={ev.median_displacement:.4f}, "
              f"p95={ev.p95_displacement:.4f}, max={ev.max_displacement:.4f}")
        print(f"  F1: 基线={ev.f1_baseline:.4f} → 隐私={ev.f1_private:.4f}  "
              f"| 保留率={ev.f1_retention_ratio*100:.1f}% | 效用损失={ev.utility_loss_ratio*100:.1f}%")
        print(f"  Precision: {ev.precision_baseline:.4f} → {ev.precision_private:.4f}  "
              f"(Δ={ev.precision_delta:+.4f})")
        print(f"  Recall:    {ev.recall_baseline:.4f} → {ev.recall_private:.4f}  "
              f"(Δ={ev.recall_delta:+.4f})")
