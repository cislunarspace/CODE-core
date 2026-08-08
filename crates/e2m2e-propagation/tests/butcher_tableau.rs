//! Butcher 表系数检验（ADR 0013：按定义验证，不用 golden file）。
//!
//! 对 e2m2e-propagation 的所有嵌入 RK 方法（PD45、PD78、RK89）
//! 做一致性校验：行和下三角、c = Σa₍ᵢⱼ₎、Σb = Σb* = 1、阶数元数据。
//!
//! 依据：这些是 Runge-Kutta 方法"按定义"必须满足的基本性质；
//! 不满足则表系数有误。
//!
//! # 参考
//! - Hairner & Wanner, *Solving Ordinary Differential Equations I* (2nd ed.)
//! - Butcher, *Numerical Methods for Ordinary Differential Equations* (3rd ed.)
//! - ADR 0013 "验证策略——按定义完成任务"

use e2m2e_propagation::butcher::ButcherTable;
use e2m2e_propagation::rk_methods::RkMethod;

/// 收集所有方法的 Butcher 表。
fn all_tables() -> Vec<(&'static str, RkMethod, &'static ButcherTable)> {
    let methods = [RkMethod::Pd45, RkMethod::Pd78, RkMethod::Rk89];
    methods
        .iter()
        .map(|&m| {
            let name = format!("{m:?}");
            (name, m, m.table())
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|(name, m, t)| {
            // 移动 name 为 &'static str — 用泄露，仅测试使用。
            let leaked: &'static str = Box::leak(name.into_boxed_str());
            (leaked, m, t)
        })
        .collect()
}

// ── 测试 1：a 行长度为 i（严格下三角） ───────────────────────────────────

/// RK 矩阵 a[i] 的长度必须等于 i（严格下三角）。
///
/// 依据（ADR 0013 / Butcher 表定义）：显式 RK 方法中，第 i 行非零元素
/// 只涉及 k₀..k_{i-1}，因此行长度必须为 i。
#[test]
fn test_a_row_lengths_are_lower_triangular() {
    for (_name, method, table) in all_tables() {
        for i in 0..table.stages {
            assert_eq!(
                table.a[i].len(),
                i,
                "{method:?} row {i} length {} != {i}",
                table.a[i].len()
            );
        }
    }
}

// ── 测试 2：c 节点 = Σⱼ aᵢⱼ（行和条件） ──────────────────────────────────

/// 每行的 RK 系数之和应等于对应的时间节点：c[i] = Σⱼ a[i][j]。
///
/// 依据（ADR 0013 / Butcher 简化假设 C(1)）：这是 Runge-Kutta 方法
/// 的基本一致性条件，确保每个阶段的截断误差至少 O(h)。
#[test]
fn test_butcher_rows_sum_to_c_nodes() {
    for (_name, method, table) in all_tables() {
        for i in 1..table.stages {
            let row_sum: f64 = table.a[i].iter().sum();
            let diff = (row_sum - table.c[i]).abs();
            assert!(
                diff < 1e-10,
                "{method:?} row {i}: Σa = {row_sum}, c = {}, diff = {diff:.2e}",
                table.c[i]
            );
        }
    }
}

// ── 测试 3：权重之和为 1（一致性条件） ───────────────────────────────────

/// 主解权重 b 与嵌入解权重 b* 之和均为 1。
///
/// 依据（ADR 0013 / Butcher 表定义）：显式 RK 方法的 b 与 b* 必须各自
/// 满足 Σbᵢ = 1、Σb*ᵢ = 1，否则方法不具有相应阶数。
#[test]
fn test_weights_sum_to_one() {
    for (_name, method, table) in all_tables() {
        let b_sum: f64 = table.b.iter().sum();
        let b_star_sum: f64 = table.b_star.iter().sum();
        assert!(
            (b_sum - 1.0).abs() < 1e-12,
            "{method:?} b sum = {b_sum} ≠ 1"
        );
        assert!(
            (b_star_sum - 1.0).abs() < 1e-12,
            "{method:?} b* sum = {b_star_sum} ≠ 1"
        );
    }
}

// ── 测试 4：阶数元数据合理 ───────────────────────────────────────────────

/// 嵌入阶 < 主阶（误差估计基于低阶嵌入解）。
///
/// 依据（ADR 0013 / 嵌入 RK 定义）：嵌入方法阶数必须严格小于主方法阶数，
/// 否则误差估计恒为零。
#[test]
fn test_embedded_order_less_than_main_order() {
    for (_name, method, table) in all_tables() {
        assert!(
            table.embedded_order < table.order,
            "{method:?} embedded_order {} >= order {}",
            table.embedded_order,
            table.order
        );
    }
}

// ── 测试 5：误差估计权重 (b - b*) 非平凡 ─────────────────────────────────

/// b 与 b* 不可完全相同，否则误差估计恒为零。
#[test]
fn test_error_weights_nontrivial() {
    for (_name, method, table) in all_tables() {
        let max_diff = table
            .b
            .iter()
            .zip(table.b_star.iter())
            .map(|(bi, bsi)| (bi - bsi).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff > 1e-15,
            "{method:?} b and b* appear identical (max diff = {max_diff:e})"
        );
    }
}
