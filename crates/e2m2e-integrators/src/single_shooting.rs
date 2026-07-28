//! 单打靶法（Single Shooting）Rust 实现。
//!
//! 参考 GMAT 的 DifferentialCorrector 类，实现单打靶迭代算法。
//! 与多重打靶不同，单打靶从初始状态传播到终端，没有中间 patch point。
//!
//! ## 算法概述
//!
//! 1. 从初始状态 x0 传播到终端状态 xf
//! 2. 计算残差 F = xf - x_target
//! 3. 通过有限差分计算雅可比 DF = ∂xf/∂x0
//! 4. 求解修正量 Δx0 = -DF^{-1} F
//! 5. 更新 x0_new = x0 + α * Δx0（α为阻尼因子）
//! 6. 重复直到收敛

use crate::forces::compiled::CompiledForce;
use crate::forces::compiled_stm::propagate_compiled_stm;
use pyo3::prelude::*;

/// 单打靶迭代结果。
#[pyclass]
#[derive(Clone)]
pub struct SingleShootingResult {
    /// 修正后的初始状态
    #[pyo3(get)]
    pub initial_state: Vec<f64>,
    /// 修正后的终端状态
    #[pyo3(get)]
    pub final_state: Vec<f64>,
    /// 是否收敛
    #[pyo3(get)]
    pub converged: bool,
    /// 实际迭代次数
    #[pyo3(get)]
    pub iterations: usize,
    /// 最终残差
    #[pyo3(get)]
    pub max_residual: f64,
    /// 每次迭代残差历史
    #[pyo3(get)]
    pub residual_history: Vec<f64>,
}

/// 有限差分类型。
#[derive(Clone, Copy)]
pub enum DifferenceType {
    /// 前向差分
    Forward,
    /// 中心差分
    Central,
    /// 后向差分
    Backward,
}

/// 计算雅可比矩阵（有限差分）。
///
/// 通过对初始状态的每个分量进行扰动，计算终端状态的变化。
fn compute_jacobian(
    forces: &[CompiledForce],
    observer: &str,
    t_span: (f64, f64),
    initial_state: &[f64; 6],
    perturbation: f64,
    diff_type: DifferenceType,
    rtol: f64,
) -> Result<[[f64; 6]; 6], String> {
    let mut jacobian = [[0.0_f64; 6]; 6];

    // 名义传播
    let nominal = propagate_compiled_stm(
        forces,
        observer,
        t_span,
        &[t_span.0, t_span.1],
        initial_state,
        rtol,
        rtol * 0.1,
        None,
        None,
    )?;
    let nominal_final = nominal.states.last().ok_or("empty propagation")?;

    // 逐分量扰动
    for j in 0..6 {
        let mut state_plus = *initial_state;
        let mut state_minus = *initial_state;

        // 计算扰动量（相对或绝对）
        let h = if initial_state[j].abs() > 1e-10 {
            perturbation * initial_state[j].abs()
        } else {
            perturbation
        };

        match diff_type {
            DifferenceType::Forward => {
                state_plus[j] += h;
                let result_plus = propagate_compiled_stm(
                    forces,
                    observer,
                    t_span,
                    &[t_span.0, t_span.1],
                    &state_plus,
                    rtol,
                    rtol * 0.1,
                    None,
                    None,
                )?;
                let final_plus = result_plus.states.last().ok_or("empty propagation")?;

                for i in 0..6 {
                    jacobian[i][j] = (final_plus[i] - nominal_final[i]) / h;
                }
            }
            DifferenceType::Central => {
                state_plus[j] += h;
                state_minus[j] -= h;

                let result_plus = propagate_compiled_stm(
                    forces,
                    observer,
                    t_span,
                    &[t_span.0, t_span.1],
                    &state_plus,
                    rtol,
                    rtol * 0.1,
                    None,
                    None,
                )?;
                let result_minus = propagate_compiled_stm(
                    forces,
                    observer,
                    t_span,
                    &[t_span.0, t_span.1],
                    &state_minus,
                    rtol,
                    rtol * 0.1,
                    None,
                    None,
                )?;

                let final_plus = result_plus.states.last().ok_or("empty propagation")?;
                let final_minus = result_minus.states.last().ok_or("empty propagation")?;

                for i in 0..6 {
                    jacobian[i][j] = (final_plus[i] - final_minus[i]) / (2.0 * h);
                }
            }
            DifferenceType::Backward => {
                state_minus[j] -= h;
                let result_minus = propagate_compiled_stm(
                    forces,
                    observer,
                    t_span,
                    &[t_span.0, t_span.1],
                    &state_minus,
                    rtol,
                    rtol * 0.1,
                    None,
                    None,
                )?;
                let final_minus = result_minus.states.last().ok_or("empty propagation")?;

                for i in 0..6 {
                    jacobian[i][j] = (nominal_final[i] - final_minus[i]) / h;
                }
            }
        }
    }

    Ok(jacobian)
}

/// 求解线性方程组 Ax = b（高斯消元法）。
fn solve_linear_system(a: &[[f64; 6]; 6], b: &[f64; 6]) -> Result<[f64; 6], String> {
    let n = 6;
    let mut augmented = [[0.0_f64; 7]; 6];

    // 构造增广矩阵
    for i in 0..n {
        for j in 0..n {
            augmented[i][j] = a[i][j];
        }
        augmented[i][n] = b[i];
    }

    // 前向消元
    for i in 0..n {
        // 选主元
        let mut max_val = augmented[i][i].abs();
        let mut max_row = i;
        for k in (i + 1)..n {
            if augmented[k][i].abs() > max_val {
                max_val = augmented[k][i].abs();
                max_row = k;
            }
        }
        // 交换行
        if max_row != i {
            for j in 0..=n {
                let tmp = augmented[i][j];
                augmented[i][j] = augmented[max_row][j];
                augmented[max_row][j] = tmp;
            }
        }

        // 消元
        let pivot = augmented[i][i];
        if pivot.abs() < 1e-15 {
            return Err("singular matrix".to_string());
        }
        for j in (i + 1)..n {
            let factor = augmented[j][i] / pivot;
            for k in i..=n {
                augmented[j][k] -= factor * augmented[i][k];
            }
        }
    }

    // 回代
    let mut x = [0.0_f64; 6];
    for i in (0..n).rev() {
        let mut sum = augmented[i][n];
        for j in (i + 1)..n {
            sum -= augmented[i][j] * x[j];
        }
        x[i] = sum / augmented[i][i];
    }

    Ok(x)
}

/// 单打靶迭代修正。
#[allow(clippy::too_many_arguments)]
pub fn single_shooting_correct(
    forces: &[CompiledForce],
    observer: &str,
    t_span: (f64, f64),
    initial_state: &[f64; 6],
    target_state: &[f64; 6],
    max_iter: usize,
    tolerance: f64,
    rtol: f64,
    perturbation: f64,
    max_step_size: Option<f64>,
    diff_type: DifferenceType,
    verbose: bool,
) -> Result<SingleShootingResult, String> {
    let mut state = *initial_state;
    let mut residual_history = Vec::new();
    let mut converged = false;

    for iteration in 0..max_iter {
        // 传播到终端
        let result = propagate_compiled_stm(
            forces,
            observer,
            t_span,
            &[t_span.0, t_span.1],
            &state,
            rtol,
            rtol * 0.1,
            None,
            None,
        )?;
        let final_state = result.states.last().ok_or("empty propagation")?;

        // 计算残差
        let mut residual = [0.0_f64; 6];
        for i in 0..6 {
            residual[i] = final_state[i] - target_state[i];
        }
        let max_res = residual.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        residual_history.push(max_res);

        if verbose {
            eprintln!("Iteration {}: max_residual = {:.2e}", iteration + 1, max_res);
        }

        // 判断收敛
        if max_res < tolerance {
            converged = true;
            break;
        }

        // 计算雅可比
        let jacobian = compute_jacobian(
            forces,
            observer,
            t_span,
            &state,
            perturbation,
            diff_type,
            rtol,
        )?;

        // 求解修正量 Δx = -J^{-1} * residual
        let neg_residual: [f64; 6] = [
            -residual[0], -residual[1], -residual[2],
            -residual[3], -residual[4], -residual[5],
        ];
        let delta = solve_linear_system(&jacobian, &neg_residual)?;

        // 步长限制
        let step_limit = max_step_size.unwrap_or(f64::INFINITY);
        let mut alpha = 1.0_f64;
        for i in 0..6 {
            if delta[i].abs() > step_limit {
                let factor = step_limit / delta[i].abs();
                if factor < alpha {
                    alpha = factor;
                }
            }
        }

        // 更新状态
        for i in 0..6 {
            state[i] += alpha * delta[i];
        }
    }

    // 最终传播获取终端状态
    let final_result = propagate_compiled_stm(
        forces,
        observer,
        t_span,
        &[t_span.0, t_span.1],
        &state,
        rtol,
        rtol * 0.1,
        None,
        None,
    )?;
    let final_state = final_result.states.last().ok_or("empty propagation")?.to_vec();

    Ok(SingleShootingResult {
        initial_state: state.to_vec(),
        final_state,
        converged,
        iterations: residual_history.len(),
        max_residual: *residual_history.last().unwrap_or(&f64::INFINITY),
        residual_history,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_linear_system() {
        // 简单的 6x6 系统
        let a = [
            [2.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ];
        let b = [5.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let x = solve_linear_system(&a, &b).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-10);
        assert!((x[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_linear_system_singular() {
        // 奇异矩阵
        let a = [
            [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 4.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ];
        let b = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0];
        let result = solve_linear_system(&a, &b);
        assert!(result.is_err());
    }
}
