//! 平面三角平动点族的全周期 PAL 适配器。
//!
//! SPO/LPO 没有 Halo 的半周期镜面对称，这里以完整平面闭合、相位条件与
//! 伪弧长条件连续追踪周期解（ADR 0028）。数值机器（传播、STM、SVD 与
//! Newton 循环）已抽取到共享内核 `pal_core`（#820，ADR 0060），本文件
//! 只保留 Python 可见的结果类型与入口：平面问题形状（四分量选择、缩放、
//! 闭合圈数）交给共享内核实例化，Rust 侧不维护第二份迭代实现。

use pyo3::prelude::*;

use crate::pal_core::{run_full_period_pal, Options, PalShape, PalTraceResult};

/// Python 可见的完整周期 PAL 结果。成员始终包含初始种子；软失败保留已收敛成员。
#[pyclass]
#[derive(Clone)]
pub struct PlanarPalRustResult {
    #[pyo3(get)]
    pub states: Vec<[f64; 6]>,
    #[pyo3(get)]
    pub periods: Vec<f64>,
    #[pyo3(get)]
    pub closure_errors: Vec<f64>,
    #[pyo3(get)]
    pub newton_iterations: Vec<usize>,
    #[pyo3(get)]
    pub tangent_system_ranks: Vec<usize>,
    #[pyo3(get)]
    pub tangent_system_conditions: Vec<f64>,
    #[pyo3(get)]
    pub augmented_system_ranks: Vec<usize>,
    #[pyo3(get)]
    pub augmented_system_conditions: Vec<f64>,
    #[pyo3(get)]
    pub step_sizes: Vec<f64>,
    #[pyo3(get)]
    pub jacobi_drifts: Vec<f64>,
    #[pyo3(get)]
    pub status: String,
    #[pyo3(get)]
    pub cause: String,
    #[pyo3(get)]
    pub message: String,
    #[pyo3(get)]
    pub steps: usize,
    #[pyo3(get)]
    pub step_size: f64,
}

impl From<PalTraceResult> for PlanarPalRustResult {
    fn from(raw: PalTraceResult) -> Self {
        Self {
            states: raw.states,
            periods: raw.periods,
            closure_errors: raw.closure_errors,
            newton_iterations: raw.newton_iterations,
            tangent_system_ranks: raw.tangent_system_ranks,
            tangent_system_conditions: raw.tangent_system_conditions,
            augmented_system_ranks: raw.augmented_system_ranks,
            augmented_system_conditions: raw.augmented_system_conditions,
            step_sizes: raw.step_sizes,
            jacobi_drifts: raw.jacobi_drifts,
            status: raw.status,
            cause: raw.cause,
            message: raw.message,
            steps: raw.steps,
            step_size: raw.step_size,
        }
    }
}

/// Python 入口：平面 SPO/LPO 的完整周期伪弧长延拓。
///
/// `n_revolutions` 为闭合圈数 n（默认 1 即 ADR 0028 的单周期闭合形式）：
/// 闭合残差在 n·T 上求值，返回的 `periods` 为成员的闭合周期 n·T。
/// `seed_period` 的语义同为种子的闭合周期。
#[pyfunction]
#[pyo3(signature = (mu, seed_state, seed_period, n_orbits, step_size, initial_direction, tolerance=1e-9, max_iterations=16, rtol=1e-12, atol=1e-12, max_step=None, n_revolutions=1))]
#[allow(clippy::too_many_arguments)]
pub fn planar_full_period_pal_py(
    mu: f64,
    seed_state: Vec<f64>,
    seed_period: f64,
    n_orbits: usize,
    step_size: f64,
    initial_direction: &str,
    tolerance: f64,
    max_iterations: usize,
    rtol: f64,
    atol: f64,
    max_step: Option<f64>,
    n_revolutions: usize,
    py: Python<'_>,
) -> PyResult<PlanarPalRustResult> {
    if seed_state.len() != 6 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "seed_state 必须含 6 个分量",
        ));
    }
    if seed_period <= 0.0 || step_size <= 0.0 || tolerance <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "seed_period、step_size、tolerance 必须为正数",
        ));
    }
    if n_revolutions == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_revolutions 必须为正整数",
        ));
    }
    if !matches!(initial_direction, "increase-x0" | "decrease-x0") {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "initial_direction 必须为 increase-x0 或 decrease-x0",
        ));
    }
    let mut seed = [0.0; 6];
    seed.copy_from_slice(&seed_state);
    let options = Options {
        tolerance,
        max_iterations,
        rtol,
        atol,
        max_step,
    };
    let shape = PalShape::planar(n_revolutions);
    Ok(py.allow_threads(|| {
        PlanarPalRustResult::from(run_full_period_pal(
            mu,
            seed,
            seed_period,
            n_orbits,
            step_size,
            initial_direction,
            options,
            &shape,
        ))
    }))
}
