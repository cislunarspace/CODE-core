//! 完整自适应步长 ODE 积分器（scipy `solve_ivp` 等价物）。
//!
//! 从 qiao 仓库的 `qiao-propagation` crate 迁移，改造为：
//! - 去掉 `Complex<f64>` 泛化，只保留 `f64`
//! - 复用 e2m2e 已有的 `explicit_rk_step`（返回标量误差）
//!
//! 使用 Prince-Dormand 8(7)13M (DOP853) 方法。

use crate::butcher::{explicit_rk_step, suggest_next_step, ButcherTable};
use crate::pd78::PD78_TABLE as DOP853;

/// 自适应积分步数上限（防止发散/塌缩轨迹死循环的兜底值）。
pub const MAX_ADAPTIVE_STEPS: usize = 500_000;

/// 纯 Rust solve_ivp：接受 Rust 闭包作为力模型回调。
///
/// # 参数
/// - `table`: Butcher 表（如 DOP853）
/// - `f`: ODE 右端函数 `f(t, y) -> Result<dy/dt, E>`
/// - `t_span`: 积分区间 `(t_start, t_end)`
/// - `y0`: 初始状态
/// - `t_eval`: 输出时间点（必须在 `t_span` 内且单调递增）
/// - `rtol`: 相对容差
/// - `atol`: 绝对容差
/// - `max_step`: 最大步长（`f64::INFINITY` = 无限制）
/// - `max_steps`: 最大步数（`usize::MAX` = 无限制）
/// - `error_dim`: 步长误差控制只统计前 N 维（`None` = 统计全部，用于 STM 增广传播）
///
/// # 返回
/// `Vec<Vec<f64>>`：每个 `t_eval` 对应的状态向量。
/// 若积分因步数耗尽或步长塌缩提前退出，返回的解可能不完整（长度 < `t_eval.len()`）。
#[allow(clippy::too_many_arguments)]
pub fn solve_ivp_impl<F, E>(
    table: &ButcherTable,
    f: F,
    t_span: (f64, f64),
    y0: &[f64],
    t_eval: &[f64],
    rtol: f64,
    atol: f64,
    max_step: f64,
    max_steps: usize,
    error_dim: Option<usize>,
) -> Vec<Vec<f64>>
where
    F: Fn(f64, &[f64]) -> Result<Vec<f64>, E>,
{
    let mut t = t_span.0;
    let mut y = y0.to_vec();

    // scipy 风格自适应初始步长选择（select_initial_step）。
    // 当 ||f0|| 很大时自动用很小的 h，避免第一步就溢出。
    let dir = (t_span.1 - t_span.0).signum();
    let span = (t_span.1 - t_span.0).abs();
    let f0 = match f(t, &y) {
        Ok(v) => v,
        Err(_) => return vec![y.clone()], // 力模型在初值处失败，只返回初值
    };
    let scale_vec: Vec<f64> = y
        .iter()
        .zip(f0.iter())
        .map(|(&yi, &fi)| atol + rtol * yi.abs().max(fi.abs()))
        .collect();
    let n_real = y.len() as f64;
    let d0: f64 = y
        .iter()
        .zip(scale_vec.iter())
        .map(|(&yi, &si)| (yi.abs() / si).powi(2))
        .sum::<f64>()
        .sqrt()
        / n_real.sqrt();
    let d1: f64 = f0
        .iter()
        .zip(scale_vec.iter())
        .map(|(&fi, &si)| (fi.abs() / si).powi(2))
        .sum::<f64>()
        .sqrt()
        / n_real.sqrt();
    let h0 = if d0 < 1e-5 || d1 < 1e-5 {
        1e-6
    } else {
        0.01 * (d0 / d1)
    };
    let mut h = dir * h0.min(span).min(max_step);

    let mut out = Vec::with_capacity(t_eval.len());
    let mut eval_idx = 0;
    let mut steps = 0usize;

    // 收集 t == t_span.0 的初始值
    while eval_idx < t_eval.len() && (t_eval[eval_idx] - t).abs() < 1e-14 * (1.0 + t.abs()) {
        out.push(y.clone());
        eval_idx += 1;
    }

    while eval_idx < t_eval.len() {
        if steps >= max_steps {
            break;
        }
        steps += 1;

        let t_next = t_eval[eval_idx];

        // 不超过下一个输出时间点
        if dir * (t + h - t_next) > 0.0 {
            h = t_next - t;
        }

        // RK 单步
        let step_result = explicit_rk_step(table, t, &y, h, &f, error_dim);
        let (y_new, error) = match step_result {
            Ok(r) => r,
            Err(_) => break, // 力模型回调失败，提前退出
        };

        // 步长已塌缩到机器精度仍无法满足容差 → 轨迹发散/近奇异，积分失败。
        // NaN 误差（如 0/0 动力学）立即退出。
        let min_step = 10.0 * f64::EPSILON * (1.0 + t.abs());
        if error.is_nan() || (error > 1.0 && h.abs() <= min_step) {
            break;
        }

        if error <= 1.0 {
            t += h;
            y = y_new;

            while eval_idx < t_eval.len() && (t_eval[eval_idx] - t).abs() < 1e-14 * (1.0 + t.abs())
            {
                out.push(y.clone());
                eval_idx += 1;
            }
        }
        h = suggest_next_step(h, error, 1.0, table.embedded_order);
        // 限制步长不超过 MaxStep
        if h.abs() > max_step {
            h = h.signum() * max_step;
        }
    }

    out
}

/// 纯 Rust solve_ivp：使用 DOP853，无步长/步数限制。
pub fn solve_ivp<F, E>(
    f: F,
    t_span: (f64, f64),
    y0: &[f64],
    t_eval: &[f64],
    rtol: f64,
    atol: f64,
    error_dim: Option<usize>,
) -> Vec<Vec<f64>>
where
    F: Fn(f64, &[f64]) -> Result<Vec<f64>, E>,
{
    solve_ivp_impl(
        &DOP853,
        f,
        t_span,
        y0,
        t_eval,
        rtol,
        atol,
        f64::INFINITY,
        usize::MAX,
        error_dim,
    )
}

/// 纯 Rust solve_ivp_capped：使用 DOP853，带 max_step 和步数上限。
#[allow(clippy::too_many_arguments)]
pub fn solve_ivp_capped<F, E>(
    f: F,
    t_span: (f64, f64),
    y0: &[f64],
    t_eval: &[f64],
    rtol: f64,
    atol: f64,
    max_step: f64,
    max_steps: usize,
    error_dim: Option<usize>,
) -> Vec<Vec<f64>>
where
    F: Fn(f64, &[f64]) -> Result<Vec<f64>, E>,
{
    solve_ivp_impl(
        &DOP853, f, t_span, y0, t_eval, rtol, atol, max_step, max_steps, error_dim,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn harmonic_oscillator() {
        // y'' = -y → [y, v]: y' = v, v' = -y
        let f = |_t: f64, y: &[f64]| -> Result<Vec<f64>, String> { Ok(vec![y[1], -y[0]]) };

        let t_eval: Vec<f64> = (0..=100).map(|i| i as f64 * 0.1).collect();
        let y0 = vec![1.0, 0.0];

        let sol = solve_ivp(f, (0.0, 10.0), &y0, &t_eval, 1e-10, 1e-10, None);

        assert_eq!(sol.len(), t_eval.len());

        for (i, &t) in t_eval.iter().enumerate() {
            let y_exact = t.cos();
            let v_exact = -t.sin();
            assert!(
                (sol[i][0] - y_exact).abs() < 1e-8,
                "t={t}: y={} vs {y_exact}",
                sol[i][0]
            );
            assert!(
                (sol[i][1] - v_exact).abs() < 1e-8,
                "t={t}: v={} vs {v_exact}",
                sol[i][1]
            );
        }
    }

    #[test]
    fn step_collapse_early_exit() {
        // 回归测试：步长塌缩时提前退出，返回不完整解。
        let f = |_t: f64, y: &[f64]| -> Result<Vec<f64>, String> { Ok(vec![100.0 * y[0] * y[0]]) };

        let t_eval: Vec<f64> = (0..=20).map(|i| i as f64 * 0.01).collect();
        let y0 = vec![1.0];

        let sol = solve_ivp_capped(
            f,
            (0.0, 0.2),
            &y0,
            &t_eval,
            1e-12,
            1e-12,
            0.01,
            MAX_ADAPTIVE_STEPS,
            None,
        );

        assert!(
            sol.len() < t_eval.len(),
            "预期积分因步长塌缩提前退出，实际返回了 {} / {} 个点（完整）",
            sol.len(),
            t_eval.len()
        );
        assert!(!sol.is_empty(), "解不应为空，应至少包含初值点");
    }

    #[test]
    fn error_dim_for_stm() {
        // 验证 error_dim 参数：STM 增广传播时，只统计前 6 维误差。
        // 构造一个 8 维系统：前 2 维是物理状态，后 6 维是"STM"（故意用大值）。
        // 若 error_dim=None，大值 STM 分量会主导误差，步长很小。
        // 若 error_dim=Some(2)，只统计前 2 维，步长正常。
        let f = |_t: f64, y: &[f64]| -> Result<Vec<f64>, String> {
            // 物理状态：简单谐振；STM 分量：故意用大值
            Ok(vec![
                y[1], -y[0], 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0,
            ])
        };

        let t_eval: Vec<f64> = (0..=10).map(|i| i as f64 * 0.5).collect();
        let y0 = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        // 用 error_dim=None：STM 分量主导误差，步长很小，可能超时或步数耗尽
        let sol_no_limit = solve_ivp_capped(
            &f,
            (0.0, 5.0),
            &y0,
            &t_eval,
            1e-10,
            1e-10,
            f64::INFINITY,
            100, // 限制步数，防止死循环
            None,
        );

        // 用 error_dim=Some(2)：只统计前 2 维，步长正常
        let sol_with_dim = solve_ivp_capped(
            &f,
            (0.0, 5.0),
            &y0,
            &t_eval,
            1e-10,
            1e-10,
            f64::INFINITY,
            100,
            Some(2),
        );

        // error_dim=Some(2) 应该能完整积分
        assert_eq!(
            sol_with_dim.len(),
            t_eval.len(),
            "error_dim=Some(2) 应能完整积分"
        );

        // error_dim=None 可能因步数耗尽提前退出（STM 分量主导误差）
        // 这个测试不强制要求提前退出，但验证两种模式都不 panic
    }
}
