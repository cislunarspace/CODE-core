//! 三维非对称全周期 PAL 适配器。
//!
//! 承载既无 XZ 镜面对称、也非平面的周期族（如从 L2 NRHO 分岔出的
//! butterfly 支，#820）。闭合残差覆盖全部六分量，相位条件与弧长条件
//! 与平面形态同口径（ADR 0060）。数值机器在共享内核 `pal_core`，本文件
//! 与 `planar_pal` 是同一内核的两个形状实例化。

use pyo3::prelude::*;

use crate::pal_core::{run_full_period_pal, Options, PalShape, PalTraceResult};

/// Python 可见的三维全周期 PAL 结果。成员始终包含初始种子；软失败保留已收敛成员。
#[pyclass]
#[derive(Clone)]
pub struct SpatialPalRustResult {
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

impl From<PalTraceResult> for SpatialPalRustResult {
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

/// Python 入口：三维非对称族的完整周期伪弧长延拓。
///
/// 不施加对称性或平面约束：种子状态取全部六分量，闭合残差在 n·T 上
/// 求值（`n_revolutions` 默认 1）。`seed_period` 与返回的 `periods`
/// 语义均为闭合周期。
#[pyfunction]
#[pyo3(signature = (mu, seed_state, seed_period, n_orbits, step_size, initial_direction, tolerance=1e-9, max_iterations=16, rtol=1e-12, atol=1e-12, max_step=None, n_revolutions=1))]
#[allow(clippy::too_many_arguments)]
pub fn spatial_full_period_pal_py(
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
) -> PyResult<SpatialPalRustResult> {
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
    let shape = PalShape::spatial(n_revolutions);
    Ok(py.allow_threads(|| {
        SpatialPalRustResult::from(run_full_period_pal(
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
