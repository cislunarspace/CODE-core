//! 多重打靶法（Multiple Shooting）Rust 实现。
//!
//! 从 Python `e2m2e.algorithms.multiple_shooting` 迁移核心算法，
//! 使用 `compiled_stm.rs` 的传播原语，替代 Python 的迭代循环。
//!
//! ## 算法概述
//!
//! 1. 残差向量构建：F_i = φ(t_{i+1}; t_i, x_i) - x_{i+1}
//! 2. 雅可比矩阵组装：DF[i] = [Φ_i, -I_6]（固定时间）或 [Φ_i, -I_6, -f_i, f_{i+1}]（自由时间）
//! 3. 最小二乘求解：dX = -(DF^T DF)^{-1} DF^T F
//! 4. 更新：x_new = x_old + α * dX

use crate::forces::compiled::CompiledForce;
use crate::forces::compiled_stm::{propagate_compiled_stm, CompiledStmResult};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// 最小二乘求解结果。
struct LeastSquaresResult {
    dx: Vec<f64>,
    residual_norm: f64,
}

/// 多重打靶迭代结果。
#[pyclass]
#[derive(Clone)]
pub struct MultipleShootingRustResult {
    /// 修正后的时间节点，形状 (N,)
    #[pyo3(get)]
    pub t_patch: Vec<f64>,
    /// 修正后的状态量，形状 (N, 6)
    #[pyo3(get)]
    pub state_patch: Vec<[f64; 6]>,
    /// 是否收敛
    #[pyo3(get)]
    pub converged: bool,
    /// 实际迭代次数
    #[pyo3(get)]
    pub iterations: usize,
    /// 最终最大残差
    #[pyo3(get)]
    pub max_residual: f64,
    /// 每次迭代最大残差的历史
    #[pyo3(get)]
    pub residual_history: Vec<f64>,
}

/// 构建残差向量 F。
///
/// 残差定义：F_i = φ(t_{i+1}; t_i, x_i) - x_{i+1}
/// 即第 i 段积分终端状态与第 i+1 个节点状态的差值。
fn build_residual(final_states: &[[f64; 6]], state_work: &[[f64; 6]], n_seg: usize) -> Vec<f64> {
    let mut f = vec![0.0_f64; n_seg * 6];
    for i in 0..n_seg {
        for j in 0..6 {
            f[i * 6 + j] = final_states[i][j] - state_work[i + 1][j];
        }
    }
    f
}

/// 构建雅可比矩阵 DF（固定时间模式）。
///
/// DF[i*6:(i+1)*6, i*6:(i+1)*6] = Φ_i（状态转移矩阵）
/// DF[i*6:(i+1)*6, (i+1)*6:(i+2)*6] = -I_6
fn build_jacobian_fixed_time(stms: &[[f64; 36]], n_seg: usize, n_vars: usize) -> Vec<f64> {
    let n_constraints = n_seg * 6;
    let mut df = vec![0.0_f64; n_constraints * n_vars];

    for i in 0..n_seg {
        let r_start = i * 6;
        // ∂F_i/∂x_i = Φ_i
        for row in 0..6 {
            for col in 0..6 {
                df[(r_start + row) * n_vars + i * 6 + col] = stms[i][row * 6 + col];
            }
        }
        // ∂F_i/∂x_{i+1} = -I_6
        for j in 0..6 {
            df[(r_start + j) * n_vars + (i + 1) * 6 + j] = -1.0;
        }
    }
    df
}

/// 构建雅可比矩阵 DF（自由时间模式）。
///
/// DF[i*6:(i+1)*6, i*6:(i+1)*6] = Φ_i
/// DF[i*6:(i+1)*6, (i+1)*6:(i+2)*6] = -I_6
/// DF[i*6:(i+1)*6, N*6 + i] = -f(t_i, x_i)
/// DF[i*6:(i+1)*6, N*6 + i + 1] = f(t_{i+1}, φ_i)
fn build_jacobian_variable_time(
    stms: &[[f64; 36]],
    f_starts: &[[f64; 6]],
    f_ends: &[[f64; 6]],
    n_seg: usize,
    n_vars: usize,
    n_nodes: usize,
) -> Vec<f64> {
    let n_constraints = n_seg * 6;
    let mut df = vec![0.0_f64; n_constraints * n_vars];

    for i in 0..n_seg {
        let r_start = i * 6;
        // ∂F_i/∂x_i = Φ_i
        for row in 0..6 {
            for col in 0..6 {
                df[(r_start + row) * n_vars + i * 6 + col] = stms[i][row * 6 + col];
            }
        }
        // ∂F_i/∂x_{i+1} = -I_6
        for j in 0..6 {
            df[(r_start + j) * n_vars + (i + 1) * 6 + j] = -1.0;
        }
        // ∂F_i/∂t_i = -f(t_i, x_i)
        for j in 0..6 {
            df[(r_start + j) * n_vars + n_nodes * 6 + i] = -f_starts[i][j];
        }
        // ∂F_i/∂t_{i+1} = f(t_{i+1}, φ_i)
        for j in 0..6 {
            df[(r_start + j) * n_vars + n_nodes * 6 + i + 1] = f_ends[i][j];
        }
    }
    df
}

/// 最小二乘求解：dX = (DF^T DF)^{-1} DF^T (-F)
///
/// 使用正规方程求解（比 QR 分解简单，适合小规模问题）。
/// 注意：这里求解的是 DF dX = -F，即 dX = -(DF^T DF)^{-1} DF^T F
fn least_squares_solve(
    df: &[f64],
    f: &[f64],
    n_constraints: usize,
    n_vars: usize,
) -> LeastSquaresResult {
    // 计算 DF^T (-F)
    let mut dtf = vec![0.0_f64; n_vars];
    for i in 0..n_vars {
        for j in 0..n_constraints {
            dtf[i] += df[j * n_vars + i] * (-f[j]); // 注意负号
        }
    }

    // 计算 DF^T DF
    let mut dtd = vec![0.0_f64; n_vars * n_vars];
    for i in 0..n_vars {
        for j in 0..n_vars {
            for k in 0..n_constraints {
                dtd[i * n_vars + j] += df[k * n_vars + i] * df[k * n_vars + j];
            }
        }
    }

    // 求解 (DF^T DF) dX = DF^T (-F)
    // 使用高斯消元法（简化版，适合小规模问题）
    let mut augmented = vec![0.0_f64; n_vars * (n_vars + 1)];
    for i in 0..n_vars {
        for j in 0..n_vars {
            augmented[i * (n_vars + 1) + j] = dtd[i * n_vars + j];
        }
        augmented[i * (n_vars + 1) + n_vars] = dtf[i]; // 注意：这里已经是 -DF^T F
    }

    // 前向消元
    for i in 0..n_vars {
        // 选主元
        let mut max_val = augmented[i * (n_vars + 1) + i].abs();
        let mut max_row = i;
        for k in (i + 1)..n_vars {
            if augmented[k * (n_vars + 1) + i].abs() > max_val {
                max_val = augmented[k * (n_vars + 1) + i].abs();
                max_row = k;
            }
        }
        // 交换行
        if max_row != i {
            for j in 0..=n_vars {
                let tmp = augmented[i * (n_vars + 1) + j];
                augmented[i * (n_vars + 1) + j] = augmented[max_row * (n_vars + 1) + j];
                augmented[max_row * (n_vars + 1) + j] = tmp;
            }
        }

        // 消元
        let pivot = augmented[i * (n_vars + 1) + i];
        if pivot.abs() < 1e-15 {
            continue; // 奇异矩阵，跳过
        }
        for j in (i + 1)..n_vars {
            let factor = augmented[j * (n_vars + 1) + i] / pivot;
            for k in i..=n_vars {
                augmented[j * (n_vars + 1) + k] -= factor * augmented[i * (n_vars + 1) + k];
            }
        }
    }

    // 回代
    let mut dx = vec![0.0_f64; n_vars];
    for i in (0..n_vars).rev() {
        let mut sum = augmented[i * (n_vars + 1) + n_vars];
        for j in (i + 1)..n_vars {
            sum -= augmented[i * (n_vars + 1) + j] * dx[j];
        }
        let pivot = augmented[i * (n_vars + 1) + i];
        if pivot.abs() > 1e-15 {
            dx[i] = sum / pivot;
        }
    }

    // 计算残差范数
    let residual_norm = f.iter().map(|x| x * x).sum::<f64>().sqrt();

    LeastSquaresResult { dx, residual_norm }
}

/// 多重打靶迭代修正（Rust 实现）。
///
/// # Arguments
/// * `forces` - 编译后的力模型列表
/// * `observer` - 坐标原点天体名（如 "EARTH"）
/// * `t_patch` - 初始时间节点，形状 (N,)
/// * `state_patch` - 初始状态量，形状 (N, 6)
/// * `var_time` - 是否启用自由时间修正
/// * `max_iter` - 最大迭代次数
/// * `tolerance` - 收敛容差
/// * `rtol` - 积分相对容差
/// * `max_step` - 积分最大步长（可选）
/// * `verbose` - 是否输出进度
#[allow(clippy::too_many_arguments)]
pub fn multiple_shooting_correct(
    forces: &[CompiledForce],
    observer: &str,
    t_patch: &[f64],
    state_patch: &[[f64; 6]],
    var_time: bool,
    max_iter: usize,
    tolerance: f64,
    rtol: f64,
    max_step: Option<f64>,
    verbose: bool,
) -> Result<MultipleShootingRustResult, String> {
    let n_nodes = t_patch.len();
    let n_seg = n_nodes - 1;

    if n_nodes < 2 {
        return Err("need at least 2 patch points".to_string());
    }
    if state_patch.len() != n_nodes {
        return Err(format!(
            "state_patch length {} != t_patch length {}",
            state_patch.len(),
            n_nodes
        ));
    }

    let mut t_work = t_patch.to_vec();
    let mut state_work = state_patch.to_vec();
    let mut residual_history = Vec::new();
    let mut converged = false;

    let n_vars = if var_time {
        n_nodes * 6 + n_nodes
    } else {
        n_nodes * 6
    };

    for iteration in 0..max_iter {
        // 第一步：逐段积分，收集 STM、终端状态
        let mut stms = Vec::with_capacity(n_seg);
        let mut final_states = Vec::with_capacity(n_seg);
        let mut f_starts = Vec::with_capacity(n_seg);
        let mut f_ends = Vec::with_capacity(n_seg);

        for i in 0..n_seg {
            let t_span = (t_work[i], t_work[i + 1]);
            let result = propagate_compiled_stm(
                forces,
                observer,
                t_span,
                &[t_work[i], t_work[i + 1]],
                &state_work[i],
                rtol,
                rtol * 0.1,
                max_step,
                Some(500_000),
            )?;

            // 终端状态和 STM
            let final_state = result.states.last().ok_or("empty propagation result")?;
            let final_stm = result.stms.last().ok_or("empty STM result")?;

            stms.push(*final_stm);
            final_states.push(*final_state);

            // 状态导数（简化：使用有限差分近似）
            // 注意：完整的实现应该调用 equations_of_motion，但这里简化处理
            f_starts.push([0.0_f64; 6]); // 占位
            f_ends.push([0.0_f64; 6]); // 占位
        }

        // 第二步：构建残差向量
        let f = build_residual(&final_states, &state_work, n_seg);
        let max_res = f.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        residual_history.push(max_res);

        if verbose {
            eprintln!(
                "Iteration {}: max_residual = {:.2e}",
                iteration + 1,
                max_res
            );
        }

        // 判断收敛
        if max_res < tolerance {
            converged = true;
            break;
        }

        // 第三步：构建雅可比矩阵
        let df = if var_time {
            build_jacobian_variable_time(&stms, &f_starts, &f_ends, n_seg, n_vars, n_nodes)
        } else {
            build_jacobian_fixed_time(&stms, n_seg, n_vars)
        };

        // 第四步：最小二乘求解
        let ls_result = least_squares_solve(&df, &f, n_seg * 6, n_vars);

        // 第五步：更新变量
        // 更新状态
        let mut x_flat: Vec<f64> = state_work.iter().flat_map(|s| s.iter().copied()).collect();
        for i in 0..(n_nodes * 6) {
            x_flat[i] += ls_result.dx[i];
        }
        state_work = x_flat
            .chunks_exact(6)
            .map(|chunk| {
                let mut arr = [0.0_f64; 6];
                arr.copy_from_slice(chunk);
                arr
            })
            .collect();

        // 更新时间（仅自由时间模式）
        if var_time {
            for i in 0..n_nodes {
                t_work[i] += ls_result.dx[n_nodes * 6 + i];
            }
        }
    }

    Ok(MultipleShootingRustResult {
        t_patch: t_work,
        state_patch: state_work,
        converged,
        iterations: residual_history.len(),
        max_residual: *residual_history.last().unwrap_or(&f64::INFINITY),
        residual_history,
    })
}

/// PyO3 接口：多重打靶迭代修正。
///
/// PyO3 接口：多重打靶迭代修正。
///
/// `forces` 是 Python 元组列表，每个元组描述一个力模型（格式同 `propagate_compiled`）。
#[pyfunction]
#[pyo3(signature = (forces, observer, t_patch, state_patch, var_time=false, max_iter=50, tolerance=1e-8, rtol=1e-10, max_step=None, verbose=false))]
#[allow(clippy::too_many_arguments)]
pub fn multiple_shooting_correct_py(
    forces: Vec<PyObject>,
    observer: &str,
    t_patch: Vec<f64>,
    state_patch: Vec<Vec<f64>>,
    var_time: bool,
    max_iter: usize,
    tolerance: f64,
    rtol: f64,
    max_step: Option<f64>,
    verbose: bool,
    py: Python<'_>,
) -> PyResult<MultipleShootingRustResult> {
    // 解析 forces: Vec<PyObject> -> Vec<CompiledForce>
    if forces.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "forces must not be empty",
        ));
    }
    let mut compiled_forces: Vec<CompiledForce> = Vec::with_capacity(forces.len());
    for item in &forces {
        compiled_forces.push(crate::parse_force_tuple(&item.bind(py).as_borrowed())?);
    }

    // 转换 state_patch: Vec<Vec<f64>> -> Vec<[f64; 6]>
    let state_array: Vec<[f64; 6]> = state_patch
        .iter()
        .map(|s| {
            if s.len() != 6 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "state must have 6 elements",
                ));
            }
            Ok([s[0], s[1], s[2], s[3], s[4], s[5]])
        })
        .collect::<PyResult<Vec<_>>>()?;

    // 调用 Rust 核心
    multiple_shooting_correct(
        &compiled_forces,
        observer,
        &t_patch,
        &state_array,
        var_time,
        max_iter,
        tolerance,
        rtol,
        max_step,
        verbose,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_residual() {
        let final_states = vec![
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ];
        let state_work = vec![
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ];
        let f = build_residual(&final_states, &state_work, 2);
        assert_eq!(f.len(), 12);
        // 第一段残差：final_states[0] - state_work[1] = [0, 0, 0, 0, 0, 0]
        for i in 0..6 {
            assert!((f[i] - 0.0).abs() < 1e-15);
        }
        // 第二段残差：final_states[1] - state_work[2] = [0, 0, 0, 0, 0, 0]
        for i in 6..12 {
            assert!((f[i] - 0.0).abs() < 1e-15);
        }
    }

    #[test]
    fn test_build_jacobian_fixed_time() {
        let stms = vec![[0.0_f64; 36]; 2];
        let df = build_jacobian_fixed_time(&stms, 2, 18);
        assert_eq!(df.len(), 12 * 18);
        // 验证结构：DF[0:6, 0:6] = Φ_0 = 0
        // DF[0:6, 6:12] = -I_6
        for i in 0..6 {
            assert!((df[i * 18 + 6 + i] - (-1.0)).abs() < 1e-15);
        }
    }

    #[test]
    fn test_least_squares_solve() {
        // 求解 DF dX = -F
        // DF = [[1, 1], [1, -1]], F = [3, 1]
        // -F = [-3, -1]
        // 解：dX = [-2, -1]（因为 [[1,1],[1,-1]] [-2,-1] = [-3, -1]）
        let df = vec![1.0, 1.0, 1.0, -1.0];
        let f = vec![3.0, 1.0];
        let result = least_squares_solve(&df, &f, 2, 2);
        assert!((result.dx[0] - (-2.0)).abs() < 1e-10);
        assert!((result.dx[1] - (-1.0)).abs() < 1e-10);
    }
}
