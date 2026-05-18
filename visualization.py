"""
可视化模块：对比不同算法的拟合效果
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Any
from geometry import Point2D, LineModel, CircleModel, FittingResult

# 设置中文字体（如果可用）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def _points_to_xy(points: List[Point2D]):
    x = np.array([p.x for p in points], dtype=float)
    y = np.array([p.y for p in points], dtype=float)
    return x, y


def _noise_displacements(original_points: List[Point2D], noisy_points: List[Point2D]):
    ox, oy = _points_to_xy(original_points)
    nx, ny = _points_to_xy(noisy_points)
    dx = nx - ox
    dy = ny - oy
    dist = np.sqrt(dx * dx + dy * dy)
    return dx, dy, dist


def plot_original_vs_privatized_points(
    original_points: List[Point2D],
    privatized_points: List[Point2D],
    epsilon: float,
    task_title: str,
    save_path: str = None,
):
    """
    原始点集 vs 差分隐私加噪后的点集（并排、同坐标范围），突出数据层面的隐私化效果。
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ox, oy = _points_to_xy(original_points)
    px, py = _points_to_xy(privatized_points)

    # 为原始点集设置缩小的Y轴范围
    ox_min, ox_max = ox.min(), ox.max()
    oy_min, oy_max = oy.min(), oy.max()
    ox_pad = max(0.1, 0.05 * (ox_max - ox_min))
    oy_pad = max(0.1, 0.03 * (oy_max - oy_min))  # 缩小Y轴padding
    original_xlim = (ox_min - ox_pad, ox_max + ox_pad)
    original_ylim = (oy_min - oy_pad, oy_max + oy_pad)

    # 为隐私化点集设置正常范围
    px_min, px_max = px.min(), px.max()
    py_min, py_max = py.min(), py.max()
    px_pad = max(0.1, 0.05 * (px_max - px_min))
    py_pad = max(0.1, 0.05 * (py_max - py_min))
    privatized_xlim = (px_min - px_pad, px_max + px_pad)
    privatized_ylim = (py_min - py_pad, py_max + py_pad)

    # 原始点集子图
    axes[0].scatter(ox, oy, c="tab:blue", s=15, alpha=0.6, edgecolors="none")
    axes[0].set_xlim(original_xlim)
    axes[0].set_ylim(original_ylim)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_title("原始点集", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("X", fontsize=12)
    axes[0].set_ylabel("Y", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='both', which='major', labelsize=11)

    # 隐私化点集子图
    axes[1].scatter(px, py, c="tab:red", s=15, alpha=0.6, edgecolors="none")
    axes[1].set_xlim(privatized_xlim)
    axes[1].set_ylim(privatized_ylim)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_title(f"隐私化点集 (DP, ε={epsilon})", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("X", fontsize=12)
    axes[1].set_ylabel("Y", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='both', which='major', labelsize=11)

    fig.suptitle(task_title, fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        # 保存为SVG矢量图格式
        svg_path = save_path.replace('.png', '.svg')
        plt.savefig(svg_path, format='svg', bbox_inches="tight")
        print(f"原始 vs 隐私化点集对比图已保存到: {svg_path}")
    plt.show()


def plot_dp_robust_fitting_evaluation(
    evaluations: List[Any],
    save_path: str = None,
    suptitle: str = "",
):
    """
    将「鲁棒拟合 + 差分隐私」评估指标可视化：
    F1 基线 vs 隐私、F1 保留率、平均位移 vs 理论拉普拉斯尺度。
    """
    from utils import DPRobustFittingEvaluation  # 运行时导入，避免循环

    if not evaluations:
        return
    if not isinstance(evaluations[0], DPRobustFittingEvaluation):
        raise TypeError("evaluations 应为 DPRobustFittingEvaluation 列表")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    names = [e.algorithm for e in evaluations]
    n = len(names)
    x = np.arange(n)
    w = 0.35

    ax0 = axes[0]
    ax0.bar(x - w / 2, [e.f1_baseline for e in evaluations], w, label="F1 (无DP)", color="steelblue")
    ax0.bar(x + w / 2, [e.f1_private for e in evaluations], w, label="F1 (鲁棒+DP)", color="coral")
    ax0.set_xticks(x)
    ax0.set_xticklabels(names, fontsize=12)
    ax0.set_ylabel("F1", fontsize=12)
    ax0.set_title("内点识别效用 (F1)", fontweight="bold", fontsize=14)
    ax0.legend(fontsize=10)
    ax0.grid(True, axis="y", alpha=0.3)
    ax0.set_ylim(0, 1.05)
    ax0.tick_params(axis='both', which='major', labelsize=11)

    ax1 = axes[1]
    ret = [e.f1_retention_ratio * 100 for e in evaluations]
    bars = ax1.bar(names, ret, color="seagreen", alpha=0.85)
    ax1.axhline(100.0, color="gray", linestyle="--", linewidth=1, label="100% 保留")
    ax1.set_ylabel("F1 保留率 (%)", fontsize=12)
    ax1.set_title("隐私下的 F1 相对基线保留率", fontweight="bold", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=11)
    for b, r in zip(bars, ret):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{r:.1f}%", ha="center", fontsize=10)

    ax2 = axes[2]
    mean_d = [e.mean_displacement for e in evaluations]
    theory = evaluations[0].theoretical_laplace_scale
    x2 = np.arange(n)
    ax2.bar(x2 - w / 2, mean_d, w, label="实测平均位移", color="mediumpurple")
    ax2.bar(x2 + w / 2, [theory] * n, w, label=f"理论尺度 b=Δf/ε ({theory:.3f})", color="gold", alpha=0.7)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(names, fontsize=12)
    ax2.set_ylabel("距离", fontsize=12)
    ax2.set_title("扰动强度：实测 vs 理论", fontweight="bold", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=11)
    # 设置纵坐标范围以改善显示效果
    max_y = max(max(mean_d) if mean_d else 0, theory) * 1.1
    ax2.set_ylim(0, max_y)

    eps = evaluations[0].epsilon
    fig.suptitle(suptitle or f"鲁棒拟合+差分隐私评估 (ε={eps})", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    if save_path:
        svg_path = save_path.replace('.png', '.svg')
        plt.savefig(svg_path, format='svg', bbox_inches="tight")
        print(f"DP鲁棒拟合评估图已保存到: {svg_path}")
    plt.show()


def plot_original_vs_dp_vs_ppdbscan_fit(
    original_points: List[Point2D],
    dp_noisy_points: List[Point2D],
    ppdbscan_private_points: List[Point2D],
    original_result: FittingResult,
    dp_result: FittingResult,
    ppdbscan_result: FittingResult,
    ppdbscan_labels: List[int],
    title: str,
    save_path: str = None,
):
    """
    专用图：同一算法在 3 种设置下对比
    - 原始拟合
    - 差分隐私(DP)拟合：点坐标加噪后拟合
    - PP-DBSCAN 隐私聚类增强拟合：聚类过程隐私化，拟合在簇上/簇间选择

    布局：2×3
    上排：三种方法的拟合效果并排（原始 / DP / PP-DBSCAN）
    下排：DP 隐私化点集可视化；PP-DBSCAN 隐私化点集可视化；PP-DBSCAN 聚类标签可视化
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ox, oy = _points_to_xy(original_points)
    nx, ny = _points_to_xy(dp_noisy_points)

    def _draw_fit(ax, bg_points, overlay_outline_points, result: FittingResult, bg_color, inlier_color, prefix: str):
        bx, by = _points_to_xy(bg_points)
        if overlay_outline_points is not None:
            sx, sy = _points_to_xy(overlay_outline_points)
            ax.scatter(sx, sy, facecolors="none", edgecolors="gray", s=22, alpha=0.3, label="原始点(轮廓)")
        ax.scatter(bx, by, c=bg_color, s=14, alpha=0.45, label=f"{prefix}点")

        if result.inliers:
            ix = [bg_points[i].x for i in result.inliers]
            iy = [bg_points[i].y for i in result.inliers]
            ax.scatter(ix, iy, c=inlier_color, s=18, alpha=0.75, label=f"内点({prefix})")

        if isinstance(result.model, LineModel):
            x_range = np.array([bx.min(), bx.max()])
            if abs(result.model.b) > 1e-10:
                y_range = -(result.model.a * x_range + result.model.c) / result.model.b
                ax.plot(x_range, y_range, "g-", linewidth=2, label=f"模型({prefix})")
            else:
                x_val = -result.model.c / result.model.a if abs(result.model.a) > 1e-10 else 0
                ax.axvline(x=x_val, color="g", linewidth=2, label=f"模型({prefix})")
        elif isinstance(result.model, CircleModel):
            circle = plt.Circle(
                (result.model.cx, result.model.cy),
                result.model.r,
                fill=False,
                color="g",
                linewidth=2,
                label=f"模型({prefix})",
            )
            ax.add_patch(circle)
            ax.scatter([result.model.cx], [result.model.cy], c="g", s=60, marker="+", linewidths=2)

        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=11)

    # --- 上排：三种拟合 ---
    axes[0, 0].set_title("原始拟合", fontsize=14, fontweight="bold")
    _draw_fit(
        axes[0, 0],
        bg_points=original_points,
        overlay_outline_points=None,
        result=original_result,
        bg_color="lightgray",
        inlier_color="tab:blue",
        prefix="原始",
    )

    axes[0, 1].set_title("差分隐私拟合（DP）", fontsize=14, fontweight="bold")
    _draw_fit(
        axes[0, 1],
        bg_points=dp_noisy_points,
        overlay_outline_points=original_points,
        result=dp_result,
        bg_color="lightcoral",
        inlier_color="tab:purple",
        prefix="DP",
    )

    axes[0, 2].set_title("PP-DBSCAN 隐私聚类增强拟合", fontsize=14, fontweight="bold")
    # PP-DBSCAN 的拟合结果通常是对原始点重新评估过的，因此背景用 original_points
    _draw_fit(
        axes[0, 2],
        bg_points=original_points,
        overlay_outline_points=None,
        result=ppdbscan_result,
        bg_color="lightgray",
        inlier_color="tab:green",
        prefix="PPDBSCAN",
    )

    # --- 下排：DP 隐私化点集 / PP-DBSCAN 隐私化点集 / PP-DBSCAN 聚类标签 ---
    # DP 隐私化点集（用于体现“隐私化后的图”）
    ax = axes[1, 0]
    ax.scatter(ox, oy, facecolors="none", edgecolors="gray", s=22, alpha=0.35, label="原始点(轮廓)")
    ax.scatter(nx, ny, c="lightcoral", s=14, alpha=0.55, label="DP 加噪点")
    ax.set_title("DP 隐私化点集", fontsize=14, fontweight="bold")
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=11)

    # PP-DBSCAN 隐私化点集（用于体现“PP-DBSCAN 隐私化后的点集效果”）
    ax = axes[1, 1]
    px, py = _points_to_xy(ppdbscan_private_points)
    ax.scatter(ox, oy, facecolors="none", edgecolors="gray", s=22, alpha=0.35, label="原始点(轮廓)")
    ax.scatter(px, py, c="tab:green", s=14, alpha=0.55, label="PP-DBSCAN 隐私化点")
    ax.set_title("PP-DBSCAN 隐私化点集", fontsize=14, fontweight="bold")
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=11)

    # PP-DBSCAN 聚类标签图
    ax = axes[1, 2]
    labels = np.array(ppdbscan_labels, dtype=int)
    unique_labels = np.unique(labels)
    # 噪声点(-1)用灰色，其余簇用tab20
    cmap = plt.cm.get_cmap("tab20", max(1, len(unique_labels)))
    for k_idx, k in enumerate(unique_labels):
        mask = labels == k
        color = "lightgray" if k == -1 else cmap(k_idx)
        ax.scatter(ox[mask], oy[mask], s=12, alpha=0.75, c=[color], label=f"簇{k}")
    ax.set_title("PP-DBSCAN 聚类结果（隐私化）", fontsize=14, fontweight="bold")
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8, ncols=2)
    ax.tick_params(axis='both', which='major', labelsize=11)

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        svg_path = save_path.replace('.png', '.svg')
        plt.savefig(svg_path, format='svg', bbox_inches="tight")
        print(f"原始 vs DP vs PP-DBSCAN 对比图已保存到: {svg_path}")
    plt.show()


def plot_line_fitting_comparison(
    points: List[Point2D],
    results: dict,
    ground_truth_inliers: List[int],
    metrics_dict: dict = None,
    save_path: str = None
):
    """
    绘制直线拟合对比图
    
    Args:
        points: 数据点
        results: 字典，键为算法名，值为FittingResult
        ground_truth_inliers: 真实内点索引
        metrics_dict: 性能指标字典
        save_path: 保存路径（可选）
    """
    n_algorithms = len(results)
    # 为原始数据 + n_algorithms个算法分配子图
    total_plots = 1 + n_algorithms  # 1个原始数据 + n个算法
    n_cols = 4  # 固定4列
    n_rows = (total_plots + n_cols - 1) // n_cols  # 计算需要的行数
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()
    
    # 绘制原始数据
    ax = axes[0]
    x_coords = [p.x for p in points]
    y_coords = [p.y for p in points]
    
    # 绘制所有点
    ax.scatter(x_coords, y_coords, c='lightgray', s=20, alpha=0.5, label='所有点')
    
    # 绘制真实内点
    true_inlier_x = [points[i].x for i in ground_truth_inliers]
    true_inlier_y = [points[i].y for i in ground_truth_inliers]
    ax.scatter(true_inlier_x, true_inlier_y, c='green', s=30, 
              marker='o', alpha=0.6, label='真实内点')
    
    # 绘制外点
    outlier_indices = [i for i in range(len(points)) if i not in ground_truth_inliers]
    outlier_x = [points[i].x for i in outlier_indices]
    outlier_y = [points[i].y for i in outlier_indices]
    ax.scatter(outlier_x, outlier_y, c='red', s=30, 
              marker='x', alpha=0.6, label='外点')
    
    ax.set_title('原始数据', fontsize=14, fontweight='bold')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # 绘制每个算法的结果
    for idx, (alg_name, result) in enumerate(results.items(), 1):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # 绘制所有点
        ax.scatter(x_coords, y_coords, c='lightgray', s=20, alpha=0.3)
        
        # 绘制预测内点
        if result.inliers:
            inlier_x = [points[i].x for i in result.inliers]
            inlier_y = [points[i].y for i in result.inliers]
            ax.scatter(inlier_x, inlier_y, c='blue', s=30, 
                      marker='o', alpha=0.7, label='预测内点')
        
        # 绘制外点
        outlier_indices = [i for i in range(len(points)) 
                          if i not in result.inliers]
        if outlier_indices:
            outlier_x = [points[i].x for i in outlier_indices]
            outlier_y = [points[i].y for i in outlier_indices]
            ax.scatter(outlier_x, outlier_y, c='red', s=30, 
                      marker='x', alpha=0.6, label='外点')
        
        # 绘制拟合的直线
        if result.model and isinstance(result.model, LineModel):
            x_range = np.array([min(x_coords), max(x_coords)])
            # 直线方程: ax + by + c = 0 => y = -(ax + c) / b
            if abs(result.model.b) > 1e-10:
                y_range = -(result.model.a * x_range + result.model.c) / result.model.b
                f1_score = metrics_dict[alg_name].f1_score if metrics_dict and alg_name in metrics_dict else result.score
                ax.plot(x_range, y_range, 'g-', linewidth=2, 
                       label=f'拟合直线 (F1={f1_score:.3f})')
            else:
                # 垂直线
                x_val = -result.model.c / result.model.a if abs(result.model.a) > 1e-10 else 0
                f1_score = metrics_dict[alg_name].f1_score if metrics_dict and alg_name in metrics_dict else result.score
                ax.axvline(x=x_val, color='g', linewidth=2, 
                          label=f'拟合直线 (F1={f1_score:.3f})')
        
        f1_score = metrics_dict[alg_name].f1_score if metrics_dict and alg_name in metrics_dict else result.score
        ax.set_title(f'{alg_name}\nF1={f1_score:.3f}, 内点数={len(result.inliers)}', 
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(axis='both', which='major', labelsize=11)
    
    # 隐藏多余的子图
    total_plots = 1 + len(results)  # 1个原始数据 + n个算法
    for idx in range(total_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        svg_path = save_path.replace('.png', '.svg')
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
        print(f"直线拟合对比图已保存到: {svg_path}")
    
    plt.show()


def plot_circle_fitting_comparison(
    points: List[Point2D],
    results: dict,
    ground_truth_inliers: List[int],
    metrics_dict: dict = None,
    save_path: str = None
):
    """
    绘制圆拟合对比图
    """
    n_algorithms = len(results)
    # 为原始数据 + n_algorithms个算法分配子图
    total_plots = 1 + n_algorithms  # 1个原始数据 + n个算法
    n_cols = 4  # 固定4列
    n_rows = (total_plots + n_cols - 1) // n_cols  # 计算需要的行数
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()
    
    # 绘制原始数据
    ax = axes[0]
    x_coords = [p.x for p in points]
    y_coords = [p.y for p in points]
    
    ax.scatter(x_coords, y_coords, c='lightgray', s=20, alpha=0.5, label='所有点')
    
    true_inlier_x = [points[i].x for i in ground_truth_inliers]
    true_inlier_y = [points[i].y for i in ground_truth_inliers]
    ax.scatter(true_inlier_x, true_inlier_y, c='green', s=30, 
              marker='o', alpha=0.6, label='真实内点')
    
    outlier_indices = [i for i in range(len(points)) if i not in ground_truth_inliers]
    outlier_x = [points[i].x for i in outlier_indices]
    outlier_y = [points[i].y for i in outlier_indices]
    ax.scatter(outlier_x, outlier_y, c='red', s=30, 
              marker='x', alpha=0.6, label='外点')
    
    ax.set_title('原始数据', fontsize=14, fontweight='bold')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # 绘制每个算法的结果
    for idx, (alg_name, result) in enumerate(results.items(), 1):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        ax.scatter(x_coords, y_coords, c='lightgray', s=20, alpha=0.3)
        
        if result.inliers:
            inlier_x = [points[i].x for i in result.inliers]
            inlier_y = [points[i].y for i in result.inliers]
            ax.scatter(inlier_x, inlier_y, c='blue', s=30, 
                      marker='o', alpha=0.7, label='预测内点')
        
        outlier_indices = [i for i in range(len(points)) 
                          if i not in result.inliers]
        if outlier_indices:
            outlier_x = [points[i].x for i in outlier_indices]
            outlier_y = [points[i].y for i in outlier_indices]
            ax.scatter(outlier_x, outlier_y, c='red', s=30, 
                      marker='x', alpha=0.6, label='外点')
        
        # 绘制拟合的圆
        if result.model and isinstance(result.model, CircleModel):
            circle = plt.Circle(
                (result.model.cx, result.model.cy), 
                result.model.r,
                fill=False, color='g', linewidth=2,
                label=f'拟合圆 (F1={metrics_dict[alg_name].f1_score:.3f})'
            )
            ax.add_patch(circle)
            ax.scatter([result.model.cx], [result.model.cy], 
                      c='green', s=50, marker='+', linewidths=2)
        
        f1_score = metrics_dict[alg_name].f1_score if metrics_dict and alg_name in metrics_dict else result.score
        ax.set_title(f'{alg_name}\nF1={f1_score:.3f}, 内点数={len(result.inliers)}', 
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(axis='both', which='major', labelsize=11)
    
    # 隐藏多余的子图
    total_plots = 1 + len(results)  # 1个原始数据 + n个算法
    for idx in range(total_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        svg_path = save_path.replace('.png', '.svg')
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
        print(f"圆拟合对比图已保存到: {svg_path}")
    
    plt.show()


def plot_performance_comparison(metrics_dict: dict, save_path: str = None):
    """
    绘制性能指标对比图
    
    Args:
        metrics_dict: 字典，键为算法名，值为PerformanceMetrics对象
        save_path: 保存路径（可选）
    """
    from utils import PerformanceMetrics
    
    algorithms = list(metrics_dict.keys())
    f1_scores = [metrics_dict[alg].f1_score for alg in algorithms]
    times = [metrics_dict[alg].execution_time_ms for alg in algorithms]
    precisions = [metrics_dict[alg].precision for alg in algorithms]
    recalls = [metrics_dict[alg].recall for alg in algorithms]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # F1分数对比
    ax = axes[0, 0]
    bars = ax.bar(algorithms, f1_scores, color='skyblue', alpha=0.7)
    ax.set_ylabel('F1分数', fontsize=12)
    ax.set_title('F1分数对比', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    for i, (bar, score) in enumerate(zip(bars, f1_scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # 执行时间对比
    ax = axes[0, 1]
    bars = ax.bar(algorithms, times, color='lightcoral', alpha=0.7)
    ax.set_ylabel('执行时间 (ms)', fontsize=12)
    ax.set_title('执行时间对比', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, (bar, time) in enumerate(zip(bars, times)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{time:.2f}', ha='center', va='bottom', fontsize=10)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # 精确率对比
    ax = axes[1, 0]
    bars = ax.bar(algorithms, precisions, color='lightgreen', alpha=0.7)
    ax.set_ylabel('精确率', fontsize=12)
    ax.set_title('精确率对比', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    for i, (bar, prec) in enumerate(zip(bars, precisions)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{prec:.3f}', ha='center', va='bottom', fontsize=10)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # 召回率对比
    ax = axes[1, 1]
    bars = ax.bar(algorithms, recalls, color='plum', alpha=0.7)
    ax.set_ylabel('召回率', fontsize=12)
    ax.set_title('召回率对比', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    for i, (bar, rec) in enumerate(zip(bars, recalls)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{rec:.3f}', ha='center', va='bottom', fontsize=10)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    plt.tight_layout()
    
    if save_path:
        svg_path = save_path.replace('.png', '.svg')
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
        print(f"性能对比图已保存到: {svg_path}")
    
    plt.show()


def plot_privacy_tradeoff(epsilon_values: List[float], 
                         f1_scores: List[float],
                         save_path: str = None):
    """
    绘制隐私-精度权衡曲线
    
    Args:
        epsilon_values: 隐私预算值列表
        f1_scores: 对应的F1分数列表
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_values, f1_scores, 'o-', linewidth=2, markersize=8, 
            color='coral', label='隐私-精度权衡')
    plt.xlabel('隐私预算 ε', fontsize=12)
    plt.ylabel('F1分数', fontsize=12)
    plt.title('隐私保护与拟合精度的权衡', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tick_params(axis='both', which='major', labelsize=11)
    
    # 标注点
    for eps, f1 in zip(epsilon_values, f1_scores):
        plt.annotate(f'ε={eps:.2f}\nF1={f1:.3f}', 
                    xy=(eps, f1), 
                    xytext=(10, 10), 
                    textcoords='offset points',
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        svg_path = save_path.replace('.png', '.svg')
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
        print(f"隐私权衡图已保存到: {svg_path}")
    
    plt.show()


def plot_privacy_data_comparison(
    original_points: List[Point2D],
    weak_private_points: List[Point2D],
    strong_private_points: List[Point2D],
    save_path: str = None
):
    """
    可视化原始点与差分隐私加噪点的差异。
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    datasets = [
        ("原始数据", original_points, "tab:blue"),
        ("弱隐私(较大ε)加噪数据", weak_private_points, "tab:green"),
        ("强隐私(较小ε)加噪数据", strong_private_points, "tab:red"),
    ]

    for ax, (title, pts, color) in zip(axes, datasets):
        x = [p.x for p in pts]
        y = [p.y for p in pts]
        ax.scatter(x, y, c=color, s=12, alpha=0.65)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(axis='both', which='major', labelsize=11)

    plt.tight_layout()
    if save_path:
        svg_path = save_path.replace('.png', '.svg')
        plt.savefig(svg_path, format='svg', bbox_inches="tight")
        print(f"隐私数据对比图已保存到: {svg_path}")
    plt.show()
