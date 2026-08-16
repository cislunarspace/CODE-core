//! CR3BP 单段微分修正数值内核。
//!
//! Python 侧保留对称性策略、问题构造和 Orbit 编排；本模块负责 STM 传播后的
//! 残差、雅可比、Newton 修正以及收敛状态机。传播与 `CR3BP_Dynamics` 共用
//! `e2m2e-forces::cr3bp::propagate_cr3bp_stm`，保证两条 Rust 路径使用同一数值内核。

use e2m2e_forces::cr3bp::{cr3bp_eom, propagate_cr3bp_stm};
use nalgebra::{DMatrix, DVector};
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub(crate) struct CorrectionRawResult {
    pub(crate) solution_state: Option<[f64; 6]>,
    pub(crate) solution_time: Option<f64>,
    pub(crate) status: &'static str,
    pub(crate) cause: &'static str,
    pub(crate) message: String,
    pub(crate) iterations: usize,
    pub(crate) residual: Option<f64>,
    error_history: Vec<f64>,
    correction_history: Vec<f64>,
    state_history: Vec<Vec<f64>>,
    time_history: Vec<f64>,
    final_state_history: Vec<Vec<f64>>,
}

#[allow(clippy::too_many_arguments)]
fn result_with(
    solution_state: Option<[f64; 6]>,
    solution_time: Option<f64>,
    status: &'static str,
    cause: &'static str,
    message: String,
    iterations: usize,
    residual: Option<f64>,
    error_history: Vec<f64>,
    correction_history: Vec<f64>,
    state_history: Vec<Vec<f64>>,
    time_history: Vec<f64>,
    final_state_history: Vec<Vec<f64>>,
) -> CorrectionRawResult {
    CorrectionRawResult {
        solution_state,
        solution_time,
        status,
        cause,
        message,
        iterations,
        residual,
        error_history,
        correction_history,
        state_history,
        time_history,
        final_state_history,
    }
}

fn solve_correction(jacobian: &[Vec<f64>], error: &[f64]) -> Result<Vec<f64>, String> {
    let rows = jacobian.len();
    let cols = jacobian.first().map_or(0, Vec::len);
    if rows == 0 || cols == 0 || error.len() != rows {
        return Err("雅可比矩阵维度无效".to_string());
    }
    if jacobian.iter().any(|row| row.len() != cols) {
        return Err("雅可比矩阵不是规则矩阵".to_string());
    }

    let values: Vec<f64> = jacobian
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();
    let matrix = DMatrix::from_row_slice(rows, cols, &values);
    let rhs = DVector::from_column_slice(error);

    if rows == cols {
        matrix
            .lu()
            .solve(&rhs)
            .map(|delta| delta.iter().copied().collect())
            .ok_or_else(|| "雅可比矩阵奇异".to_string())
    } else {
        // 与 numpy.linalg.lstsq 对齐：欠定/超定系统取最小范数解，不使用正规方程。
        matrix
            .svd(true, true)
            .solve(&rhs, f64::EPSILON * rows.max(cols) as f64 * 10.0)
            .map(|delta| delta.iter().copied().collect())
            .map_err(|_| "雅可比矩阵奇异".to_string())
    }
}

fn linspace(start: f64, end: f64, count: usize) -> Vec<f64> {
    if count <= 1 {
        return vec![start];
    }
    let step = (end - start) / (count - 1) as f64;
    let mut values: Vec<f64> = (0..count).map(|i| start + i as f64 * step).collect();
    // NumPy linspace 的 endpoint 合同：最后一个采样点必须是 stop 本身。
    *values.last_mut().expect("count >= 2") = end;
    values
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_correction(
    mu: f64,
    initial_state: [f64; 6],
    initial_time: f64,
    constraint_indices: &[usize],
    target_values: &[f64],
    free_variable_indices: &[usize],
    full_period: bool,
    recover_halo_time: bool,
    max_iterations: usize,
    tolerance: f64,
    stagnation_limit: f64,
    divergence_limit: f64,
    rtol: f64,
    atol: f64,
    max_step: Option<f64>,
    sample_count: usize,
) -> CorrectionRawResult {
    let mut current_state = initial_state;
    let mut current_time = initial_time;
    let mut error_history = Vec::new();
    let mut correction_history = Vec::new();
    let mut state_history = Vec::new();
    let mut time_history = Vec::new();
    let mut final_state_history = Vec::new();
    let mut last_error = None;

    for iteration in 0..max_iterations {
        let t_eval = linspace(0.0, current_time, sample_count.max(2));
        let propagation = propagate_cr3bp_stm(
            mu,
            (0.0, current_time),
            &t_eval,
            &current_state,
            rtol,
            atol,
            max_step,
            Some(500_000),
        );
        let result = match propagation {
            Ok(result) => result,
            Err(error) => {
                return result_with(
                    None,
                    None,
                    "failed",
                    "integration_failed",
                    format!("积分失败: {error}"),
                    iteration + 1,
                    last_error,
                    error_history,
                    correction_history,
                    state_history,
                    time_history,
                    final_state_history,
                );
            }
        };

        let final_state = match result.states.last() {
            Some(state) => *state,
            None => {
                return result_with(
                    None,
                    None,
                    "failed",
                    "integration_failed",
                    "积分失败: 空传播结果".to_string(),
                    iteration + 1,
                    last_error,
                    error_history,
                    correction_history,
                    state_history,
                    time_history,
                    final_state_history,
                );
            }
        };
        let final_stm = match result.stms.last() {
            Some(stm) => stm,
            None => {
                return result_with(
                    None,
                    None,
                    "failed",
                    "integration_failed",
                    "积分失败: 空 STM 结果".to_string(),
                    iteration + 1,
                    last_error,
                    error_history,
                    correction_history,
                    state_history,
                    time_history,
                    final_state_history,
                );
            }
        };

        let mut error_vector = Vec::with_capacity(constraint_indices.len());
        for (i, &constraint_index) in constraint_indices.iter().enumerate() {
            if constraint_index >= 6 {
                return result_with(
                    None,
                    None,
                    "failed",
                    "invalid_input",
                    format!("约束索引越界: {constraint_index}"),
                    iteration + 1,
                    last_error,
                    error_history,
                    correction_history,
                    state_history,
                    time_history,
                    final_state_history,
                );
            }
            let residual = if full_period {
                final_state[constraint_index] - current_state[constraint_index]
            } else {
                final_state[constraint_index] - target_values[i]
            };
            error_vector.push(residual);
        }

        let current_error = error_vector
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        last_error = Some(current_error);
        error_history.push(current_error);
        state_history.push(current_state.to_vec());
        time_history.push(current_time);
        final_state_history.push(final_state.to_vec());

        if current_error < tolerance {
            return result_with(
                Some(current_state),
                Some(current_time),
                "converged",
                "none",
                if full_period {
                    "收敛成功：闭合残差小于容差".to_string()
                } else {
                    "收敛成功：误差小于容差".to_string()
                },
                iteration + 1,
                Some(current_error),
                error_history,
                correction_history,
                state_history,
                time_history,
                final_state_history,
            );
        }

        if current_error > divergence_limit {
            return result_with(
                None,
                None,
                "diverged",
                "divergence_detected",
                "发散：误差超过限制".to_string(),
                iteration + 1,
                Some(current_error),
                error_history,
                correction_history,
                state_history,
                time_history,
                final_state_history,
            );
        }

        let state_derivative = cr3bp_eom(mu, &final_state);
        let mut jacobian = vec![vec![0.0; free_variable_indices.len()]; constraint_indices.len()];
        for (j, &variable_index) in free_variable_indices.iter().enumerate() {
            for (i, &constraint_index) in constraint_indices.iter().enumerate() {
                if variable_index < 6 {
                    jacobian[i][j] = final_stm[constraint_index * 6 + variable_index];
                    if full_period && variable_index == constraint_index {
                        jacobian[i][j] -= 1.0;
                    }
                } else if variable_index == 6 {
                    jacobian[i][j] = state_derivative[constraint_index];
                }
            }
        }

        let delta = match solve_correction(&jacobian, &error_vector) {
            Ok(delta) => delta,
            Err(message) => {
                return result_with(
                    None,
                    None,
                    "failed",
                    "singular_jacobian",
                    message,
                    iteration + 1,
                    Some(current_error),
                    error_history,
                    correction_history,
                    state_history,
                    time_history,
                    final_state_history,
                );
            }
        };
        let correction_norm = delta.iter().map(|value| value * value).sum::<f64>().sqrt();
        correction_history.push(correction_norm);

        for (j, &variable_index) in free_variable_indices.iter().enumerate() {
            if variable_index < 6 {
                current_state[variable_index] -= delta[j];
            } else if variable_index == 6 {
                current_time -= delta[j];
            }
        }

        if recover_halo_time && current_time < 0.02 {
            current_time = 0.25;
        } else if current_time <= 0.0 {
            current_time = if full_period { 0.1 } else { 1e-6 };
        }

        if correction_norm < stagnation_limit {
            return result_with(
                None,
                None,
                "stagnated",
                "stagnation_detected",
                "停滞：修正量过小".to_string(),
                iteration + 1,
                Some(current_error),
                error_history,
                correction_history,
                state_history,
                time_history,
                final_state_history,
            );
        }
    }

    result_with(
        None,
        None,
        "max_iterations",
        "max_iterations_reached",
        "达到最大迭代次数".to_string(),
        max_iterations,
        last_error,
        error_history,
        correction_history,
        state_history,
        time_history,
        final_state_history,
    )
}

/// Python 接口：CR3BP 单段微分修正。
#[pyfunction]
#[pyo3(signature = (mu, initial_state, initial_time, constraint_indices, target_values, free_variable_indices, full_period, recover_halo_time, max_iterations, tolerance, stagnation_limit, divergence_limit, rtol, atol, max_step=None, sample_count=1000))]
#[allow(clippy::too_many_arguments)]
pub fn differential_correction_cr3bp_py(
    mu: f64,
    initial_state: Vec<f64>,
    initial_time: f64,
    constraint_indices: Vec<usize>,
    target_values: Vec<f64>,
    free_variable_indices: Vec<usize>,
    full_period: bool,
    recover_halo_time: bool,
    max_iterations: usize,
    tolerance: f64,
    stagnation_limit: f64,
    divergence_limit: f64,
    rtol: f64,
    atol: f64,
    max_step: Option<f64>,
    sample_count: usize,
    py: Python<'_>,
) -> PyResult<PyObject> {
    if initial_state.len() != 6 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "initial_state must have length 6",
        ));
    }
    if !mu.is_finite() || initial_state.iter().any(|value| !value.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "mu and initial_state must be finite",
        ));
    }
    if !initial_time.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "initial_time must be finite",
        ));
    }
    if constraint_indices.iter().any(|&index| index >= 6) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "constraint_indices must be less than 6",
        ));
    }
    if free_variable_indices.iter().any(|&index| index > 6) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "free_variable_indices must be at most 6",
        ));
    }
    if target_values.iter().any(|value| !value.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "target_values must be finite",
        ));
    }
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "tolerance must be finite and positive",
        ));
    }
    if !stagnation_limit.is_finite() || stagnation_limit < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "stagnation_limit must be finite and non-negative",
        ));
    }
    if !divergence_limit.is_finite() || divergence_limit <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "divergence_limit must be finite and positive",
        ));
    }
    if !rtol.is_finite() || rtol <= 0.0 || !atol.is_finite() || atol <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rtol and atol must be finite and positive",
        ));
    }
    if max_step.is_some_and(|step| !step.is_finite() || step <= 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "max_step must be finite and positive when provided",
        ));
    }
    if constraint_indices.len() != target_values.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "constraint_indices and target_values must have equal length",
        ));
    }
    if constraint_indices.is_empty() || free_variable_indices.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "constraints and free variables must not be empty",
        ));
    }
    if max_iterations == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "max_iterations must be positive",
        ));
    }

    let mut state = [0.0; 6];
    state.copy_from_slice(&initial_state);
    let result = py.allow_threads(|| {
        run_correction(
            mu,
            state,
            initial_time,
            &constraint_indices,
            &target_values,
            &free_variable_indices,
            full_period,
            recover_halo_time,
            max_iterations,
            tolerance,
            stagnation_limit,
            divergence_limit,
            rtol,
            atol,
            max_step,
            sample_count,
        )
    });

    let dict = PyDict::new(py);
    dict.set_item(
        "solution_state",
        result.solution_state.map(|state| state.to_vec()),
    )?;
    dict.set_item("solution_time", result.solution_time)?;
    dict.set_item("status", result.status)?;
    dict.set_item("cause", result.cause)?;
    dict.set_item("message", result.message)?;
    dict.set_item("iterations", result.iterations)?;
    dict.set_item("residual", result.residual)?;
    dict.set_item("error_history", result.error_history)?;
    dict.set_item("correction_history", result.correction_history)?;
    dict.set_item("state_history", result.state_history)?;
    dict.set_item("time_history", result.time_history)?;
    dict.set_item("final_state_history", result.final_state_history)?;
    Ok(dict.into())
}

#[cfg(test)]
mod tests {
    use super::solve_correction;

    #[test]
    fn square_system_uses_direct_solution() {
        let delta = solve_correction(&[vec![2.0, 0.0], vec![0.0, 4.0]], &[2.0, 8.0]).unwrap();
        assert_eq!(delta, vec![1.0, 2.0]);
    }

    #[test]
    fn underdetermined_system_returns_minimum_norm_solution() {
        let delta = solve_correction(&[vec![1.0, 1.0]], &[2.0]).unwrap();
        assert!((delta[0] - 1.0).abs() < 1e-12);
        assert!((delta[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn correction_reports_stagnation_before_next_iteration() {
        let result = super::run_correction(
            0.012_150_585_6,
            [0.8, 0.0, 0.0, 0.0, 0.2, 0.0],
            0.1,
            &[1, 3],
            &[0.0, 0.0],
            &[4, 6],
            false,
            false,
            2,
            1e-12,
            f64::MAX,
            1e10,
            1e-12,
            1e-12,
            None,
            2,
        );

        assert_eq!(result.status, "stagnated");
        assert_eq!(result.cause, "stagnation_detected");
        assert_eq!(result.iterations, 1);
        assert_eq!(result.error_history.len(), 1);
        assert_eq!(result.correction_history.len(), 1);
    }

    #[test]
    fn full_period_equilibrium_converges_on_closure_residual() {
        let result = super::run_correction(
            0.012_150_585_6,
            [0.487_849_414_4, 0.866_025_403_784_438_6, 0.0, 0.0, 0.0, 0.0],
            0.1,
            &[0, 1],
            &[0.0, 0.0],
            &[0],
            true,
            false,
            1,
            1e-10,
            1e-14,
            1e10,
            1e-12,
            1e-12,
            None,
            2,
        );

        assert_eq!(result.status, "converged");
        assert_eq!(result.cause, "none");
        assert_eq!(result.iterations, 1);
        assert!(result.error_history[0] < 1e-10);
        assert_eq!(result.correction_history, Vec::<f64>::new());
    }
}
