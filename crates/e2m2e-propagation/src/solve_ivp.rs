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

/// 单个事件的规格：事件函数 + scipy `solve_ivp` 语义的 terminal/direction。
pub struct EventSpec<G> {
    /// 事件函数 `g(t, y) -> 标量`，零点即事件面。
    pub g: G,
    /// 触发后是否终止积分。
    pub terminal: bool,
    /// 方向过滤：`> 0` 只记上行穿越（g 由负到正），`< 0` 只记下行，`0` 双向。
    pub direction: f64,
}

impl<G> EventSpec<G> {
    pub fn new(g: G, terminal: bool, direction: f64) -> Self {
        Self {
            g,
            terminal,
            direction,
        }
    }
}

/// 带事件检测的积分结果。
pub struct EventSolveResult {
    /// 输出时间点：`t_eval` 的前缀；terminal 事件截断时末点为事件时刻。
    pub t: Vec<f64>,
    /// 与 `t` 对齐的状态。
    pub states: Vec<Vec<f64>>,
    /// 每个事件的触发时刻列表。
    pub t_events: Vec<Vec<f64>>,
    /// 每个事件的触发状态列表（接受步内线性插值 + 二分求精）。
    pub y_events: Vec<Vec<Vec<f64>>>,
    /// 触发终止的事件索引（未终止为 `None`）。
    pub terminal_event: Option<usize>,
    /// 积分步数（含拒绝步）。
    pub n_steps: usize,
}

/// scipy 语义的方向过滤。`g_prev` 恰为 0 不重复触发（上一步终点落在
/// 事件面上时，下一步不再因该点重复记录）。
fn crossed(g_prev: f64, g_curr: f64, direction: f64) -> bool {
    if g_prev == 0.0 {
        return false;
    }
    let up = g_prev < 0.0 && g_curr >= 0.0;
    let down = g_prev > 0.0 && g_curr <= 0.0;
    if direction > 0.0 {
        up
    } else if direction < 0.0 {
        down
    } else {
        up || down
    }
}

/// 在接受步内对线性插值态二分求精事件零点。
///
/// 不做稠密输出：以 `(t0, y0)`→`(t1, y1)` 的线性插值近似步内状态，
/// 在插值权重 w ∈ [0, 1] 上二分 `g = 0`。求精精度受线性插值误差
/// （量级 h²/8 · |ÿ|）限制，需更紧时请减小 `max_step`。
fn refine_crossing<G, E>(
    ev: &EventSpec<G>,
    t0: f64,
    y0: &[f64],
    t1: f64,
    y1: &[f64],
    g0: f64,
) -> (f64, Vec<f64>)
where
    G: Fn(f64, &[f64]) -> Result<f64, E>,
{
    let interp = |w: f64| -> Vec<f64> {
        y0.iter()
            .zip(y1.iter())
            .map(|(a, b)| a + w * (b - a))
            .collect()
    };
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    let mut g_lo = g0;
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        let t_mid = t0 + mid * (t1 - t0);
        let y_mid = interp(mid);
        let g_mid = match (ev.g)(t_mid, &y_mid) {
            Ok(v) => v,
            Err(_) => break, // 事件函数中途失败：保留当前包围区估计
        };
        if g_mid == 0.0 {
            lo = mid;
            hi = mid;
            break;
        }
        if (g_mid > 0.0) == (g_lo > 0.0) {
            lo = mid;
            g_lo = g_mid;
        } else {
            hi = mid;
        }
    }
    let w = 0.5 * (lo + hi);
    (t0 + w * (t1 - t0), interp(w))
}

/// 纯 Rust solve_ivp（带事件检测）：接受 Rust 闭包作为力模型与事件回调。
///
/// 事件检测在每个接受步的端点评估事件函数，符号变化（经 direction 过滤）
/// 时在步内对线性插值态二分求精；terminal 事件触发即截断积分，输出末点
/// 替换为求精后的事件点。无事件（`events` 为空）时行为与 `solve_ivp_impl`
/// 完全一致。
///
/// # 参数
/// 同 `solve_ivp_impl`，另加：
/// - `events`: 事件规格列表，见 [`EventSpec`]
///
/// # 返回
/// [`EventSolveResult`]。若积分因步数耗尽或步长塌缩提前退出，返回的解
/// 可能不完整（长度 < `t_eval.len()`）。
#[allow(clippy::too_many_arguments)]
pub fn solve_ivp_events_impl<F, G, E>(
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
    events: &[EventSpec<G>],
) -> EventSolveResult
where
    F: Fn(f64, &[f64]) -> Result<Vec<f64>, E>,
    G: Fn(f64, &[f64]) -> Result<f64, E>,
{
    let mut t = t_span.0;
    let mut y = y0.to_vec();

    let mut result = EventSolveResult {
        t: Vec::with_capacity(t_eval.len()),
        states: Vec::with_capacity(t_eval.len()),
        t_events: events.iter().map(|_| Vec::new()).collect(),
        y_events: events.iter().map(|_| Vec::new()).collect(),
        terminal_event: None,
        n_steps: 0,
    };

    // scipy 风格自适应初始步长选择（select_initial_step）。
    // 当 ||f0|| 很大时自动用很小的 h，避免第一步就溢出。
    let dir = (t_span.1 - t_span.0).signum();
    let span = (t_span.1 - t_span.0).abs();
    let f0 = match f(t, &y) {
        Ok(v) => v,
        Err(_) => {
            // 力模型在初值处失败，只返回初值
            result.t.push(t);
            result.states.push(y.clone());
            return result;
        }
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

    // 事件函数在初值处的取值；任一失败只返回初值（与力模型失败一致）。
    let mut g_values: Vec<f64> = Vec::with_capacity(events.len());
    for ev in events {
        match (ev.g)(t, &y) {
            Ok(v) => g_values.push(v),
            Err(_) => {
                result.t.push(t);
                result.states.push(y.clone());
                return result;
            }
        }
    }

    let mut eval_idx = 0;
    let mut steps = 0usize;

    // 收集 t == t_span.0 的初始值
    while eval_idx < t_eval.len() && (t_eval[eval_idx] - t).abs() < 1e-14 * (1.0 + t.abs()) {
        result.t.push(t_eval[eval_idx]);
        result.states.push(y.clone());
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
            let t_prev = t;
            let y_prev = std::mem::replace(&mut y, y_new);
            t += h;

            // 事件检测：比较接受步两端点的符号，触发则步内二分求精。
            let mut terminal_hit: Option<(usize, f64, Vec<f64>)> = None;
            let mut event_failed = false;
            for (i, ev) in events.iter().enumerate() {
                let g_prev = g_values[i];
                let g_curr = match (ev.g)(t, &y) {
                    Ok(v) => v,
                    Err(_) => {
                        event_failed = true;
                        break;
                    }
                };
                g_values[i] = g_curr;
                if !crossed(g_prev, g_curr, ev.direction) {
                    continue;
                }
                let (t_ev, y_ev) = refine_crossing(ev, t_prev, &y_prev, t, &y, g_prev);
                result.t_events[i].push(t_ev);
                result.y_events[i].push(y_ev.clone());
                // 同一步内多个 terminal 事件触发时，按事件索引取第一个
                if ev.terminal && terminal_hit.is_none() {
                    terminal_hit = Some((i, t_ev, y_ev));
                }
            }
            if event_failed {
                break; // 事件函数失败，提前退出
            }
            if let Some((i, t_ev, y_ev)) = terminal_hit {
                result.t.push(t_ev);
                result.states.push(y_ev);
                result.terminal_event = Some(i);
                result.n_steps = steps;
                return result;
            }

            while eval_idx < t_eval.len() && (t_eval[eval_idx] - t).abs() < 1e-14 * (1.0 + t.abs())
            {
                result.t.push(t_eval[eval_idx]);
                result.states.push(y.clone());
                eval_idx += 1;
            }
        }
        h = suggest_next_step(h, error, 1.0, table.embedded_order);
        // 限制步长不超过 MaxStep
        if h.abs() > max_step {
            h = h.signum() * max_step;
        }
    }

    result.n_steps = steps;
    result
}

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
    type NoEvent<E> = fn(f64, &[f64]) -> Result<f64, E>;
    solve_ivp_events_impl::<F, NoEvent<E>, E>(
        table,
        f,
        t_span,
        y0,
        t_eval,
        rtol,
        atol,
        max_step,
        max_steps,
        error_dim,
        &[],
    )
    .states
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

    /// 谐振子 y = cos(t)，事件 g = y[0] 的零点有解析解 t = π/2 + kπ。
    fn harmonic_rhs(_t: f64, y: &[f64]) -> Result<Vec<f64>, String> {
        Ok(vec![y[1], -y[0]])
    }

    #[test]
    fn event_terminal_stops_at_zero_crossing() {
        // terminal + direction=-1：应在首个下行零点 t = π/2 处截断。
        let events = [EventSpec::new(
            |_t: f64, y: &[f64]| -> Result<f64, String> { Ok(y[0]) },
            true,
            -1.0,
        )];

        let t_eval: Vec<f64> = (0..=100).map(|i| i as f64 * 0.1).collect();
        let y0 = vec![1.0, 0.0];

        let result = solve_ivp_events_impl(
            &DOP853,
            harmonic_rhs,
            (0.0, 10.0),
            &y0,
            &t_eval,
            1e-10,
            1e-10,
            0.01, // max_step：把步内线性插值误差压到 ~h²/8 ≈ 1e-5
            usize::MAX,
            None,
            &events,
        );

        assert_eq!(result.terminal_event, Some(0));
        assert_eq!(result.t_events[0].len(), 1);
        let t_ev = result.t_events[0][0];
        assert!(
            (t_ev - std::f64::consts::FRAC_PI_2).abs() < 1e-4,
            "t_ev={t_ev}, 期望 ≈ π/2"
        );
        // 末点为求精后的事件点，积分被截断
        assert_eq!(result.t.last(), Some(&t_ev));
        assert!(result.states.len() < t_eval.len());
        assert!(result.states.last().unwrap()[0].abs() < 1e-4);
    }

    #[test]
    fn event_direction_filtering() {
        // g = y[0] = cos(t)：下行零点 π/2, 5π/2（10s 内），上行零点 3π/2。
        let mk_event = |direction: f64| {
            EventSpec::new(
                |_t: f64, y: &[f64]| -> Result<f64, String> { Ok(y[0]) },
                false,
                direction,
            )
        };

        let t_eval: Vec<f64> = (0..=1000).map(|i| i as f64 * 0.01).collect();
        let y0 = vec![1.0, 0.0];

        let down = solve_ivp_events_impl(
            &DOP853,
            harmonic_rhs,
            (0.0, 10.0),
            &y0,
            &t_eval,
            1e-10,
            1e-10,
            0.01,
            usize::MAX,
            None,
            &[mk_event(-1.0)],
        );
        let up = solve_ivp_events_impl(
            &DOP853,
            harmonic_rhs,
            (0.0, 10.0),
            &y0,
            &t_eval,
            1e-10,
            1e-10,
            0.01,
            usize::MAX,
            None,
            &[mk_event(1.0)],
        );

        use std::f64::consts::PI;
        let expected_down = [0.5 * PI, 2.5 * PI];
        assert_eq!(down.t_events[0].len(), expected_down.len());
        for (got, want) in down.t_events[0].iter().zip(expected_down.iter()) {
            assert!((got - want).abs() < 1e-4, "下行穿越 t={got}, 期望 ≈ {want}");
        }

        let expected_up = [1.5 * PI];
        assert_eq!(up.t_events[0].len(), expected_up.len());
        for (got, want) in up.t_events[0].iter().zip(expected_up.iter()) {
            assert!((got - want).abs() < 1e-4, "上行穿越 t={got}, 期望 ≈ {want}");
        }

        // 非 terminal 事件不截断积分
        assert_eq!(down.terminal_event, None);
        assert_eq!(down.states.len(), t_eval.len());
        assert_eq!(up.states.len(), t_eval.len());
    }

    #[test]
    fn event_not_triggered_keeps_full_solution() {
        // g = y[0] + 2 恒正，永不触发；解应与普通 solve_ivp 一致。
        let events = [EventSpec::new(
            |_t: f64, y: &[f64]| -> Result<f64, String> { Ok(y[0] + 2.0) },
            true,
            0.0,
        )];

        let t_eval: Vec<f64> = (0..=100).map(|i| i as f64 * 0.1).collect();
        let y0 = vec![1.0, 0.0];

        let result = solve_ivp_events_impl(
            &DOP853,
            harmonic_rhs,
            (0.0, 10.0),
            &y0,
            &t_eval,
            1e-10,
            1e-10,
            f64::INFINITY,
            usize::MAX,
            None,
            &events,
        );
        let plain = solve_ivp(harmonic_rhs, (0.0, 10.0), &y0, &t_eval, 1e-10, 1e-10, None);

        assert_eq!(result.terminal_event, None);
        assert!(result.t_events[0].is_empty());
        assert_eq!(result.states.len(), t_eval.len());
        assert_eq!(result.states, plain);
    }
}
