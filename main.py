import numpy as np
from geometry import Point2D
from ransac import RANSAC
from lsc import LSC
from privacy import PrivateRANSAC, PrivateLSC
from dbscan_privacy import DBSCANRANSAC, DBSCANLSC
from utils import (generate_line_data, generate_circle_data,
                  evaluate_algorithm, PerformanceMetrics,
                  evaluate_dp_robust_fitting, print_dp_robust_evaluations)
from visualization import (plot_line_fitting_comparison,
                          plot_circle_fitting_comparison,
                          plot_performance_comparison,
                          plot_privacy_tradeoff,
                          plot_privacy_data_comparison,
                          plot_original_vs_privatized_points,
                          plot_dp_robust_fitting_evaluation,
                          plot_original_vs_dp_vs_ppdbscan_fit)
from dbscan_privacy import PrivacyPreservingDBSCAN

def main():
    print("=" * 70)
    print()
    num_inliers = 200
    num_outliers = 100
    noise_level = 0.5
    weak_privacy_epsilon = 2.0
    strong_privacy_epsilon = 0.2
    privacy_sensitivity = 1.5
    compare_epsilon = strong_privacy_epsilon
    print("生成测试数据...")
    print(f"内点数: {num_inliers}")
    print(f"外点数: {num_outliers}")
    print(f"噪声水平: {noise_level}")
    print()
    print("=" * 70)
    print("直线拟合测试与可视化对比")
    print("=" * 70)
    line_points = generate_line_data(num_inliers, num_outliers, noise_level)
    line_ground_truth = list(range(num_inliers))
    line_results = {}
    line_metrics_dict = {}
    print("\n[1/5] 运行原始RANSAC...")
    ransac = RANSAC(max_iterations=1000, threshold=1.0)
    ransac_result = ransac.fit_line(line_points)
    ransac_metrics = evaluate_algorithm(
        line_points, line_ground_truth, 
        lambda pts: ransac.fit_line(pts)
    )
    line_results["原始RANSAC"] = ransac_result
    line_metrics_dict["原始RANSAC"] = ransac_metrics
    print(f"  F1分数: {ransac_metrics.f1_score:.4f}, "
          f"时间: {ransac_metrics.execution_time_ms:.2f}ms")
    print("\n[2/5] 运行原始LSC...")
    lsc = LSC(num_samples=1000, threshold=1.0)
    lsc_result = lsc.fit_line(line_points)
    lsc_metrics = evaluate_algorithm(
        line_points, line_ground_truth,
        lambda pts: lsc.fit_line(pts)
    )
    line_results["原始LSC"] = lsc_result
    line_metrics_dict["原始LSC"] = lsc_metrics
    print(f"  F1分数: {lsc_metrics.f1_score:.4f}, "
          f"时间: {lsc_metrics.execution_time_ms:.2f}ms")
    print("\n[3/5] 运行DBSCAN增强RANSAC...")
    dbscan_ransac = DBSCANRANSAC(
        ransac_params={'max_iterations': 1000, 'threshold': 1.0},
        dbscan_params={'eps': 2.0, 'min_samples': 5, 'epsilon': compare_epsilon, 'rho': 0.1}
    )
    dbscan_ransac_result = dbscan_ransac.fit_line(line_points)
    dbscan_ransac_metrics = evaluate_algorithm(
        line_points, line_ground_truth,
        lambda pts: dbscan_ransac.fit_line(pts)
    )
    line_results["PPDBSCAN+RANSAC"] = dbscan_ransac_result
    line_metrics_dict["PPDBSCAN+RANSAC"] = dbscan_ransac_metrics
    print(f"  F1分数: {dbscan_ransac_metrics.f1_score:.4f}, "
          f"时间: {dbscan_ransac_metrics.execution_time_ms:.2f}ms")
    print("\n[4/5] 运行DBSCAN增强LSC...")
    dbscan_lsc = DBSCANLSC(
        lsc_params={'num_samples': 1000, 'threshold': 1.0},
        dbscan_params={'eps': 2.0, 'min_samples': 5, 'epsilon': compare_epsilon, 'rho': 0.1}
    )
    dbscan_lsc_result = dbscan_lsc.fit_line(line_points)
    dbscan_lsc_metrics = evaluate_algorithm(
        line_points, line_ground_truth,
        lambda pts: dbscan_lsc.fit_line(pts)
    )
    line_results["PPDBSCAN+LSC"] = dbscan_lsc_result
    line_metrics_dict["PPDBSCAN+LSC"] = dbscan_lsc_metrics
    print(f"  F1分数: {dbscan_lsc_metrics.f1_score:.4f}, "
          f"时间: {dbscan_lsc_metrics.execution_time_ms:.2f}ms")
    print("\n[5/7] 运行隐私保护RANSAC（弱隐私）...")
    private_ransac = PrivateRANSAC(
        {'max_iterations': 1000, 'threshold': 1.0},
        {'epsilon': weak_privacy_epsilon, 'sensitivity': privacy_sensitivity}
    )
    private_ransac_result = private_ransac.fit_line(line_points)
    private_ransac_metrics = evaluate_algorithm(
        line_points, line_ground_truth,
        lambda pts: private_ransac.fit_line(pts)
    )
    line_results[f"隐私RANSAC(ε={weak_privacy_epsilon})"] = private_ransac_result
    line_metrics_dict[f"隐私RANSAC(ε={weak_privacy_epsilon})"] = private_ransac_metrics
    print(f"  F1分数: {private_ransac_metrics.f1_score:.4f}, "
          f"时间: {private_ransac_metrics.execution_time_ms:.2f}ms")
    print("\n[6/7] 运行隐私保护RANSAC（强隐私）...")
    private_ransac_strong = PrivateRANSAC(
        {'max_iterations': 1000, 'threshold': 1.0},
        {'epsilon': compare_epsilon, 'sensitivity': privacy_sensitivity}
    )
    private_ransac_strong_result = private_ransac_strong.fit_line(line_points)
    private_ransac_strong_metrics = evaluate_algorithm(
        line_points, line_ground_truth,
        lambda pts: private_ransac_strong.fit_line(pts)
    )
    line_results[f"隐私RANSAC(ε={strong_privacy_epsilon})"] = private_ransac_strong_result
    line_metrics_dict[f"隐私RANSAC(ε={strong_privacy_epsilon})"] = private_ransac_strong_metrics
    print(f"  F1分数: {private_ransac_strong_metrics.f1_score:.4f}, "
          f"时间: {private_ransac_strong_metrics.execution_time_ms:.2f}ms")
    print("\n[7/7] 运行隐私保护LSC（强隐私）...")
    private_lsc_strong = PrivateLSC(
        {'num_samples': 1000, 'threshold': 1.0},
        {'epsilon': compare_epsilon, 'sensitivity': privacy_sensitivity}
    )
    private_lsc_strong_result = private_lsc_strong.fit_line(line_points)
    private_lsc_strong_metrics = evaluate_algorithm(
        line_points, line_ground_truth,
        lambda pts: private_lsc_strong.fit_line(pts)
    )
    line_results[f"隐私LSC(ε={strong_privacy_epsilon})"] = private_lsc_strong_result
    line_metrics_dict[f"隐私LSC(ε={strong_privacy_epsilon})"] = private_lsc_strong_metrics
    print(f"  F1分数: {private_lsc_strong_metrics.f1_score:.4f}, "
          f"时间: {private_lsc_strong_metrics.execution_time_ms:.2f}ms")
    weak_noisy_points = private_ransac.get_noisy_points(line_points)
    strong_noisy_points = private_ransac_strong.get_noisy_points(line_points)
    print("\n生成差分隐私加噪数据对比图...")
    plot_privacy_data_comparison(
        line_points,
        weak_noisy_points,
        strong_noisy_points,
        save_path='privacy_data_line.png'
    )
    strong_noisy_points_lsc = private_lsc_strong.get_noisy_points(line_points)
    print("\n生成「原始点集 vs 隐私化点集」并排对比图（直线，DP）...")
    plot_original_vs_privatized_points(
        line_points,
        strong_noisy_points,
        epsilon=compare_epsilon,
        task_title=f"直线：原始点集 vs 差分隐私加噪点集 (ε={compare_epsilon})",
        save_path="orig vs privacy_points_line.png",
    )
    line_dp_evals = [
        evaluate_dp_robust_fitting(
            "直线", "RANSAC", compare_epsilon, privacy_sensitivity,
            line_metrics_dict["原始RANSAC"],
            line_metrics_dict[f"隐私RANSAC(ε={strong_privacy_epsilon})"],
            line_points, strong_noisy_points,
        ),
        evaluate_dp_robust_fitting(
            "直线", "LSC", compare_epsilon, privacy_sensitivity,
            line_metrics_dict["原始LSC"],
            line_metrics_dict[f"隐私LSC(ε={strong_privacy_epsilon})"],
            line_points, strong_noisy_points_lsc,
        ),
    ]
    print_dp_robust_evaluations(line_dp_evals)
    plot_dp_robust_fitting_evaluation(
        line_dp_evals,
        save_path="dp_robust_evaluation_line.png",
        suptitle=f"直线：鲁棒拟合 + 差分隐私 (ε={compare_epsilon})",
    )
    print("\n生成“原始 vs DP vs PP-DBSCAN”对比图（RANSAC，直线）...")
    ppdbscan_engine_line = PrivacyPreservingDBSCAN(
        eps=2.0, min_samples=5, epsilon=compare_epsilon, rho=0.1
    )
    ppdbscan_labels_line, _ = ppdbscan_engine_line.fit(line_points)
    ppdbscan_private_points_line = ppdbscan_engine_line.get_private_points(line_points)
    plot_original_vs_dp_vs_ppdbscan_fit(
        original_points=line_points,
        dp_noisy_points=strong_noisy_points,
        ppdbscan_private_points=ppdbscan_private_points_line,
        original_result=ransac_result,
        dp_result=private_ransac_strong_result,
        ppdbscan_result=dbscan_ransac_result,
        ppdbscan_labels=ppdbscan_labels_line,
        title=f"RANSAC：原始 vs DP(ε={compare_epsilon}) vs PP-DBSCAN(ε={compare_epsilon}) - 直线",
        save_path="orig vs dp vs ppdbscan_ransac_line.png",
    )
    print("\n生成“原始 vs DP vs PP-DBSCAN”对比图（LSC，直线）...")
    plot_original_vs_dp_vs_ppdbscan_fit(
        original_points=line_points,
        dp_noisy_points=strong_noisy_points_lsc,
        ppdbscan_private_points=ppdbscan_private_points_line,
        original_result=lsc_result,
        dp_result=private_lsc_strong_result,
        ppdbscan_result=dbscan_lsc_result,
        ppdbscan_labels=ppdbscan_labels_line,
        title=f"LSC：原始 vs DP(ε={compare_epsilon}) vs PP-DBSCAN(ε={compare_epsilon}) - 直线",
        save_path="orig vs dp vs ppdbscan_lsc_line.png",
    )
    print("\n生成直线拟合可视化对比图...")
    plot_line_fitting_comparison(
        line_points, line_results, line_ground_truth, line_metrics_dict,
        save_path='line_fitting.png'
    )
    print("\n生成性能指标对比图...")
    plot_performance_comparison(
        line_metrics_dict,
        save_path='performance_line.png'
    )
    print("\n" + "=" * 70)
    print("圆拟合测试与可视化对比")
    print("=" * 70)
    circle_points = generate_circle_data(
        num_inliers, num_outliers, 0, 0, 5.0, noise_level
    )
    circle_ground_truth = list(range(num_inliers))
    circle_results = {}
    circle_metrics_dict = {}
    print("\n[1/6] 运行原始RANSAC（圆）...")
    ransac_circle = RANSAC(max_iterations=1000, threshold=1.0, min_samples=3)
    ransac_circle_result = ransac_circle.fit_circle(circle_points)
    ransac_circle_metrics = evaluate_algorithm(
        circle_points, circle_ground_truth,
        lambda pts: ransac_circle.fit_circle(pts)
    )
    circle_results["原始RANSAC"] = ransac_circle_result
    circle_metrics_dict["原始RANSAC"] = ransac_circle_metrics
    print("\n[2/6] 运行DBSCAN增强RANSAC（圆）...")
    dbscan_ransac_circle = DBSCANRANSAC(
        ransac_params={'max_iterations': 1000, 'threshold': 1.0, 'min_samples': 3},
        dbscan_params={'eps': 2.5, 'min_samples': 5, 'epsilon': compare_epsilon, 'rho': 0.1}
    )
    dbscan_ransac_circle_result = dbscan_ransac_circle.fit_circle(circle_points)
    dbscan_ransac_circle_metrics = evaluate_algorithm(
        circle_points, circle_ground_truth,
        lambda pts: dbscan_ransac_circle.fit_circle(pts)
    )
    circle_results["PPDBSCAN+RANSAC"] = dbscan_ransac_circle_result
    circle_metrics_dict["PPDBSCAN+RANSAC"] = dbscan_ransac_circle_metrics
    print("\n[3/6] 运行DBSCAN增强LSC（圆）...")
    dbscan_lsc_circle = DBSCANLSC(
        lsc_params={'num_samples': 1000, 'threshold': 1.0, 'min_samples': 3},
        dbscan_params={'eps': 2.5, 'min_samples': 5, 'epsilon': compare_epsilon, 'rho': 0.1}
    )
    dbscan_lsc_circle_result = dbscan_lsc_circle.fit_circle(circle_points)
    dbscan_lsc_circle_metrics = evaluate_algorithm(
        circle_points, circle_ground_truth,
        lambda pts: dbscan_lsc_circle.fit_circle(pts)
    )
    circle_results["PPDBSCAN+LSC"] = dbscan_lsc_circle_result
    circle_metrics_dict["PPDBSCAN+LSC"] = dbscan_lsc_circle_metrics
    print("\n[4/6] 运行原始LSC（圆）...")
    lsc_circle = LSC(num_samples=1000, threshold=1.0, min_samples=3)
    lsc_circle_result = lsc_circle.fit_circle(circle_points)
    lsc_circle_metrics = evaluate_algorithm(
        circle_points, circle_ground_truth,
        lambda pts: lsc_circle.fit_circle(pts)
    )
    circle_results["原始LSC"] = lsc_circle_result
    circle_metrics_dict["原始LSC"] = lsc_circle_metrics
    print("\n[5/6] 运行隐私保护RANSAC（圆，强隐私）...")
    private_ransac_circle = PrivateRANSAC(
        {'max_iterations': 1000, 'threshold': 1.0, 'min_samples': 3},
        {'epsilon': compare_epsilon, 'sensitivity': privacy_sensitivity}
    )
    private_ransac_circle_result = private_ransac_circle.fit_circle(circle_points)
    private_ransac_circle_metrics = evaluate_algorithm(
        circle_points, circle_ground_truth,
        lambda pts: private_ransac_circle.fit_circle(pts)
    )
    circle_results[f"隐私RANSAC(ε={strong_privacy_epsilon})"] = private_ransac_circle_result
    circle_metrics_dict[f"隐私RANSAC(ε={strong_privacy_epsilon})"] = private_ransac_circle_metrics
    print("\n[6/6] 运行隐私保护LSC（圆，强隐私）...")
    private_lsc_circle = PrivateLSC(
        {'num_samples': 1000, 'threshold': 1.0, 'min_samples': 3},
        {'epsilon': compare_epsilon, 'sensitivity': privacy_sensitivity}
    )
    private_lsc_circle_result = private_lsc_circle.fit_circle(circle_points)
    private_lsc_circle_metrics = evaluate_algorithm(
        circle_points, circle_ground_truth,
        lambda pts: private_lsc_circle.fit_circle(pts)
    )
    circle_results[f"隐私LSC(ε={strong_privacy_epsilon})"] = private_lsc_circle_result
    circle_metrics_dict[f"隐私LSC(ε={strong_privacy_epsilon})"] = private_lsc_circle_metrics
    print("\n生成差分隐私加噪数据对比图（圆）...")
    circle_noisy_weak = PrivateRANSAC(
        {'max_iterations': 1000, 'threshold': 1.0, 'min_samples': 3},
        {'epsilon': weak_privacy_epsilon, 'sensitivity': privacy_sensitivity}
    ).get_noisy_points(circle_points)
    circle_noisy_strong = private_ransac_circle.get_noisy_points(circle_points)
    plot_privacy_data_comparison(
        circle_points,
        circle_noisy_weak,
        circle_noisy_strong,
        save_path='privacy_data_circle.png'
    )
    circle_noisy_strong_lsc = private_lsc_circle.get_noisy_points(circle_points)
    print("\n生成「原始点集 vs 隐私化点集」并排对比图（圆，DP）...")
    plot_original_vs_privatized_points(
        circle_points,
        circle_noisy_strong,
        epsilon=compare_epsilon,
        task_title=f"圆：原始点集 vs 差分隐私加噪点集 (ε={compare_epsilon})",
        save_path="orig vs privacy_points_circle.png",
    )
    circle_dp_evals = [
        evaluate_dp_robust_fitting(
            "圆", "RANSAC", compare_epsilon, privacy_sensitivity,
            circle_metrics_dict["原始RANSAC"],
            circle_metrics_dict[f"隐私RANSAC(ε={strong_privacy_epsilon})"],
            circle_points, circle_noisy_strong,
        ),
        evaluate_dp_robust_fitting(
            "圆", "LSC", compare_epsilon, privacy_sensitivity,
            circle_metrics_dict["原始LSC"],
            circle_metrics_dict[f"隐私LSC(ε={strong_privacy_epsilon})"],
            circle_points, circle_noisy_strong_lsc,
        ),
    ]
    print_dp_robust_evaluations(circle_dp_evals)
    plot_dp_robust_fitting_evaluation(
        circle_dp_evals,
        save_path="dp_robust_evaluation_circle.png",
        suptitle=f"圆：鲁棒拟合 + 差分隐私 (ε={compare_epsilon})",
    )
    print("\n生成“原始 vs DP vs PP-DBSCAN”对比图（RANSAC，圆）...")
    ppdbscan_engine_circle = PrivacyPreservingDBSCAN(
        eps=2.5, min_samples=5, epsilon=compare_epsilon, rho=0.1
    )
    ppdbscan_labels_circle, _ = ppdbscan_engine_circle.fit(circle_points)
    ppdbscan_private_points_circle = ppdbscan_engine_circle.get_private_points(circle_points)
    plot_original_vs_dp_vs_ppdbscan_fit(
        original_points=circle_points,
        dp_noisy_points=circle_noisy_strong,
        ppdbscan_private_points=ppdbscan_private_points_circle,
        original_result=ransac_circle_result,
        dp_result=private_ransac_circle_result,
        ppdbscan_result=dbscan_ransac_circle_result,
        ppdbscan_labels=ppdbscan_labels_circle,
        title=f"RANSAC：原始 vs DP(ε={compare_epsilon}) vs PP-DBSCAN(ε={compare_epsilon}) - 圆",
        save_path="orig_vs_dp_vs_ppdbscan_ransac_circle.png",
    )
    print("\n生成“原始 vs DP vs PP-DBSCAN”对比图（LSC，圆）...")
    plot_original_vs_dp_vs_ppdbscan_fit(
        original_points=circle_points,
        dp_noisy_points=circle_noisy_strong_lsc,
        ppdbscan_private_points=ppdbscan_private_points_circle,
        original_result=lsc_circle_result,
        dp_result=private_lsc_circle_result,
        ppdbscan_result=dbscan_lsc_circle_result,
        ppdbscan_labels=ppdbscan_labels_circle,
        title=f"LSC：原始 vs DP(ε={compare_epsilon}) vs PP-DBSCAN(ε={compare_epsilon}) - 圆",
        save_path="orig_vs_dp_vs_ppdbscan_lsc_circle.png",
    )
    print("\n生成圆拟合可视化对比图...")
    plot_circle_fitting_comparison(
        circle_points, circle_results, circle_ground_truth, circle_metrics_dict,
        save_path='circle_fitting.png'
    )
    print("\n生成性能指标对比图...")
    plot_performance_comparison(
        circle_metrics_dict,
        save_path='performance_circle.png'
    )
    print("\n" + "=" * 70)
    print("隐私-精度权衡分析")
    print("=" * 70)
    epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    f1_scores = []
    print("\n测试不同隐私预算下的性能...")
    for eps in epsilon_values:
        dbscan_ransac_test = DBSCANRANSAC(
            ransac_params={'max_iterations': 1000, 'threshold': 1.0},
            dbscan_params={'eps': 2.0, 'min_samples': 5, 
                          'epsilon': eps, 'rho': 0.1}
        )
        result = dbscan_ransac_test.fit_line(line_points)
        metrics = evaluate_algorithm(
            line_points, line_ground_truth,
            lambda pts: dbscan_ransac_test.fit_line(pts)
        )
        f1_scores.append(metrics.f1_score)
        print(f"  ε={eps:.1f}: F1={metrics.f1_score:.4f}")
    print("\n生成隐私-精度权衡曲线...")
    plot_privacy_tradeoff(
        epsilon_values, f1_scores,
        save_path='privacy_tradeoff.png'
    )
    print("\n" + "=" * 70)
    print("\n所有可视化图表已生成并保存！")
    print("=" * 70)

if __name__ == "__main__":
    main()