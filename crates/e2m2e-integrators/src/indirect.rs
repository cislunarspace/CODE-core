//! 间接低推力单打靶的状态、协态与变分传播。
//!
//! 六维机械状态为 `(r, v)`，自然加速度及其对位置、速度的雅可比由调用方注入。
//! 当前实现固定质量、固定时间与推力加速度上限，使用 sigmoid 平滑燃料目标。
//! 末端残差由 Python 编排层负责，本模块不绑定目标轨道或坐标表示。

use e2m2e_propagation::solve_ivp::solve_ivp_capped;
use nalgebra::{Matrix3, Matrix6, Vector3};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// 平滑有界控制与积分容差。时间单位与状态单位由调用方保持一致。
#[derive(Clone, Copy, Debug)]
pub struct IndirectControl {
    pub acceleration_max: f64,
    pub epsilon: f64,
    pub rtol: f64,
    pub atol: f64,
    pub max_step: f64,
    pub max_steps: usize,
}

impl IndirectControl {
    /// 输入协态已按最大加速度归一化，切换函数为 `|mu_v|-1`。
    pub fn throttle(&self, lambda_v_norm: f64) -> f64 {
        let switching = (lambda_v_norm - 1.0) / self.epsilon;
        if switching > 40.0 {
            1.0
        } else if switching < -40.0 {
            0.0
        } else {
            1.0 / (1.0 + (-switching).exp())
        }
    }
}

/// 传播结果：末态、末端协态及末态对初始协态的雅可比。
pub struct IndirectPropagation {
    pub state: [f64; 6],
    pub costate: [f64; 6],
    pub jacobian: Option<Matrix6<f64>>,
}

fn costate_derivative(
    lambda_r: &Vector3<f64>,
    lambda_v: &Vector3<f64>,
    ar: &Matrix3<f64>,
    av: &Matrix3<f64>,
) -> (Vector3<f64>, Vector3<f64>) {
    (
        -ar.transpose() * lambda_v,
        -lambda_r - av.transpose() * lambda_v,
    )
}

/// 传播 `[r, v, lambda_r, lambda_v]`，可选积分 `12 x 12` 变分矩阵。
///
/// `natural(t, state)` 返回 `(acceleration, da/dr, da/dv)`，其中矩阵均为行优先
/// `3 x 3` 展平。推力沿 `-lambda_v` 方向。积分失败返回 `Err`，不把部分轨迹当成功。
type NaturalResult = Result<([f64; 3], [f64; 9], [f64; 9]), String>;

pub fn propagate_indirect<F>(
    t0: f64,
    tf: f64,
    state0: &[f64; 6],
    lambda0: &[f64; 6],
    control: &IndirectControl,
    with_jacobian: bool,
    natural: F,
) -> Result<IndirectPropagation, String>
where
    F: Fn(f64, &[f64; 6]) -> NaturalResult,
{
    if !t0.is_finite() || !tf.is_finite() || tf <= t0 {
        return Err("tf must be finite and greater than t0".into());
    }
    if control.acceleration_max <= 0.0
        || control.epsilon <= 0.0
        || control.rtol <= 0.0
        || control.atol <= 0.0
        || control.max_step <= 0.0
        || control.max_steps == 0
    {
        return Err(
            "acceleration, epsilon, tolerances, max_step and max_steps must be positive".into(),
        );
    }
    let dimension = if with_jacobian { 156 } else { 12 };
    let mut initial = vec![0.0; dimension];
    initial[..6].copy_from_slice(state0);
    initial[6..12].copy_from_slice(lambda0);
    if with_jacobian {
        for i in 0..12 {
            initial[12 + 12 * i + i] = 1.0;
        }
    }
    let rhs = |t: f64, y: &[f64]| -> Result<Vec<f64>, String> {
        let state: [f64; 6] = y[..6]
            .try_into()
            .map_err(|_| "six state entries required")?;
        let (acceleration, ar_flat, av_flat) = natural(t, &state)?;
        let ar = Matrix3::from_row_slice(&ar_flat);
        let av = Matrix3::from_row_slice(&av_flat);
        let lambda_r = Vector3::new(y[6], y[7], y[8]);
        let lambda_v = Vector3::new(y[9], y[10], y[11]);
        let lambda_norm = lambda_v.norm();
        let throttle = control.throttle(lambda_norm);
        let primer_direction = if lambda_norm > 1e-14 {
            -lambda_v / lambda_norm
        } else {
            Vector3::zeros()
        };
        let thrust_acceleration = control.acceleration_max * throttle * primer_direction;
        let (dlr, dlv) = costate_derivative(&lambda_r, &lambda_v, &ar, &av);
        let mut dy = vec![0.0; dimension];
        dy[..3].copy_from_slice(&state[3..6]);
        for i in 0..3 {
            dy[3 + i] = acceleration[i] + thrust_acceleration[i];
        }
        dy[6..9].copy_from_slice(dlr.as_slice());
        dy[9..12].copy_from_slice(dlv.as_slice());
        if with_jacobian {
            let mut dcontrol = Matrix3::zeros();
            if lambda_norm > 1e-14 {
                let normal = lambda_v / lambda_norm;
                let projection = Matrix3::identity() - normal * normal.transpose();
                dcontrol = -control.acceleration_max
                    * ((throttle / lambda_norm) * projection
                        + (throttle * (1.0 - throttle) / control.epsilon)
                            * normal
                            * normal.transpose());
            }

            // 协态变分还含 -(∂A/∂x)^T λ 对 δx 的直接贡献。对一般自然
            // 动力学回调，此处对 A 做中心差分，不要求调用方推导三阶张量。
            let mut da_dx = [[[0.0_f64; 6]; 6]; 3];
            for j in 0..6 {
                let step = 1e-6 * state[j].abs().max(1.0);
                let mut plus = state;
                let mut minus = state;
                plus[j] += step;
                minus[j] -= step;
                let (_, ar_p, av_p) = natural(t, &plus)?;
                let (_, ar_m, av_m) = natural(t, &minus)?;
                for k in 0..3 {
                    for i in 0..6 {
                        let gp = if i < 3 {
                            ar_p[3 * k + i]
                        } else {
                            av_p[3 * k + i - 3]
                        };
                        let gm = if i < 3 {
                            ar_m[3 * k + i]
                        } else {
                            av_m[3 * k + i - 3]
                        };
                        da_dx[k][i][j] = (gp - gm) / (2.0 * step);
                    }
                }
            }
            for column in 0..12 {
                let position_variation = Vector3::new(
                    y[12 + 12 * column],
                    y[12 + 1 + 12 * column],
                    y[12 + 2 + 12 * column],
                );
                let velocity_variation = Vector3::new(
                    y[12 + 3 + 12 * column],
                    y[12 + 4 + 12 * column],
                    y[12 + 5 + 12 * column],
                );
                let lr_variation = Vector3::new(
                    y[12 + 6 + 12 * column],
                    y[12 + 7 + 12 * column],
                    y[12 + 8 + 12 * column],
                );
                let lv_variation = Vector3::new(
                    y[12 + 9 + 12 * column],
                    y[12 + 10 + 12 * column],
                    y[12 + 11 + 12 * column],
                );
                let dposition = velocity_variation;
                // 状态变分包含 A·δx 和控制的 B·(∂a/∂λv)·δλv。
                let dvelocity =
                    ar * position_variation + av * velocity_variation + dcontrol * lv_variation;
                let (dc_lr, dc_lv) = costate_derivative(&lr_variation, &lv_variation, &ar, &av);
                let dx = [
                    position_variation[0],
                    position_variation[1],
                    position_variation[2],
                    velocity_variation[0],
                    velocity_variation[1],
                    velocity_variation[2],
                ];
                let mut direct_costate = [0.0_f64; 6];
                for c in 0..6 {
                    for k in 0..3 {
                        for j in 0..6 {
                            direct_costate[c] -= da_dx[k][c][j] * lambda_v[k] * dx[j];
                        }
                    }
                }
                let dc_lr =
                    dc_lr + Vector3::new(direct_costate[0], direct_costate[1], direct_costate[2]);
                let dc_lv =
                    dc_lv + Vector3::new(direct_costate[3], direct_costate[4], direct_costate[5]);
                dy[12 + 12 * column..12 + 3 + 12 * column].copy_from_slice(dposition.as_slice());
                dy[12 + 3 + 12 * column..12 + 6 + 12 * column]
                    .copy_from_slice(dvelocity.as_slice());
                dy[12 + 6 + 12 * column..12 + 9 + 12 * column].copy_from_slice(dc_lr.as_slice());
                dy[12 + 9 + 12 * column..12 + 12 + 12 * column].copy_from_slice(dc_lv.as_slice());
            }
        }
        Ok(dy)
    };
    let output = solve_ivp_capped(
        rhs,
        (t0, tf),
        &initial,
        &[t0, tf],
        control.rtol,
        control.atol,
        control.max_step,
        control.max_steps,
        Some(12),
    );
    if output.len() != 2 {
        return Err("indirect state-costate propagation failed before tf".into());
    }
    let terminal = output.last().expect("two output points checked");
    let final_state: [f64; 6] = terminal[..6]
        .try_into()
        .map_err(|_| "propagated state must have six entries")?;
    if !final_state.iter().all(|value| value.is_finite()) {
        return Err("indirect propagation produced non-finite state".into());
    }
    let jacobian = with_jacobian
        .then(|| Matrix6::from_fn(|row, column| terminal[12 + row + 12 * (column + 6)]));
    Ok(IndirectPropagation {
        state: final_state,
        costate: terminal[6..12].try_into().expect("six terminal costates"),
        jacobian,
    })
}

fn call_natural(
    callback: &Bound<'_, PyAny>,
    t: f64,
    state: &[f64; 6],
) -> PyResult<([f64; 3], [f64; 9], [f64; 9])> {
    let value = callback.call1((t, state.to_vec()))?;
    let py = value.py();
    let numpy = py.import("numpy")?;
    let flattened = |object: &Bound<'_, PyAny>| -> PyResult<Vec<f64>> {
        numpy
            .call_method1("asarray", (object,))?
            .call_method0("ravel")?
            .call_method0("tolist")?
            .extract()
    };
    let acceleration = flattened(&value.get_item(0)?)?;
    let ar = flattened(&value.get_item(1)?)?;
    let av = flattened(&value.get_item(2)?)?;
    if acceleration.len() != 3 || ar.len() != 9 || av.len() != 9 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "natural callback must return acceleration (3,), da/dr (9,) and da/dv (9,)",
        ));
    }
    Ok((
        acceleration.try_into().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("acceleration must have three entries")
        })?,
        ar.try_into()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("da/dr must have nine entries"))?,
        av.try_into()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("da/dv must have nine entries"))?,
    ))
}

/// Python 间接传播接口。回调必须返回 `(a, da_dr, da_dv)`，矩阵行优先。
#[pyfunction]
#[pyo3(signature = (t0, tf, state0, lambda0, acceleration_max, epsilon, natural, rtol=1e-9, atol=1e-9, max_step=0.05, max_steps=30000, with_jacobian=true))]
#[allow(clippy::too_many_arguments)]
pub fn propagate_indirect_py(
    t0: f64,
    tf: f64,
    state0: [f64; 6],
    lambda0: [f64; 6],
    acceleration_max: f64,
    epsilon: f64,
    natural: &Bound<'_, PyAny>,
    rtol: f64,
    atol: f64,
    max_step: f64,
    max_steps: usize,
    with_jacobian: bool,
    py: Python<'_>,
) -> PyResult<PyObject> {
    let control = IndirectControl {
        acceleration_max,
        epsilon,
        rtol,
        atol,
        max_step,
        max_steps,
    };
    // 在进入积分器前检查回调形状，保留 Python 输入错误的原始类型。
    call_natural(natural, t0, &state0)?;
    let result = propagate_indirect(
        t0,
        tf,
        &state0,
        &lambda0,
        &control,
        with_jacobian,
        |t, state| call_natural(natural, t, state).map_err(|error| error.to_string()),
    )
    .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    let dict = PyDict::new(py);
    dict.set_item("state", result.state.to_vec())?;
    dict.set_item("costate", result.costate.to_vec())?;
    dict.set_item(
        "jacobian",
        result.jacobian.map(|matrix| {
            (0..6)
                .map(|row| {
                    (0..6)
                        .map(|column| matrix[(row, column)])
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        }),
    )?;
    Ok(dict.into())
}

/// CR3BP 的 Rust 自然动力学与雅可比，不经过 Python 回调。
#[pyfunction]
#[pyo3(signature = (mu, t0, tf, state0, lambda0, acceleration_max, epsilon, rtol=1e-9, atol=1e-9, max_step=0.05, max_steps=30000, with_jacobian=true))]
#[allow(clippy::too_many_arguments)]
pub fn propagate_cr3bp_indirect_py(
    mu: f64,
    t0: f64,
    tf: f64,
    state0: [f64; 6],
    lambda0: [f64; 6],
    acceleration_max: f64,
    epsilon: f64,
    rtol: f64,
    atol: f64,
    max_step: f64,
    max_steps: usize,
    with_jacobian: bool,
    py: Python<'_>,
) -> PyResult<PyObject> {
    let control = IndirectControl {
        acceleration_max,
        epsilon,
        rtol,
        atol,
        max_step,
        max_steps,
    };
    let result = propagate_indirect(
        t0,
        tf,
        &state0,
        &lambda0,
        &control,
        with_jacobian,
        |_t, state| {
            let derivative = e2m2e_forces::cr3bp::cr3bp_eom(mu, state);
            let matrix = e2m2e_forces::cr3bp::cr3bp_jacobian_6x6(mu, state);
            let mut ar = [0.0; 9];
            let mut av = [0.0; 9];
            for i in 0..3 {
                for j in 0..3 {
                    ar[3 * i + j] = matrix[3 + i][j];
                    av[3 * i + j] = matrix[3 + i][3 + j];
                }
            }
            Ok((
                derivative[3..6].try_into().expect("three accelerations"),
                ar,
                av,
            ))
        },
    )
    .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    let dict = PyDict::new(py);
    dict.set_item("state", result.state.to_vec())?;
    dict.set_item("costate", result.costate.to_vec())?;
    dict.set_item(
        "jacobian",
        result.jacobian.map(|matrix| {
            (0..6)
                .map(|row| {
                    (0..6)
                        .map(|column| matrix[(row, column)])
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        }),
    )?;
    Ok(dict.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn free_natural(_t: f64, _state: &[f64; 6]) -> NaturalResult {
        Ok(([0.0; 3], [0.0; 9], [0.0; 9]))
    }

    fn control() -> IndirectControl {
        IndirectControl {
            acceleration_max: 0.1,
            epsilon: 0.1,
            rtol: 1e-11,
            atol: 1e-12,
            max_step: 0.2,
            max_steps: 10_000,
        }
    }

    #[test]
    fn position_costate_is_driven_by_velocity_costate() {
        let ar = Matrix3::identity();
        let av = Matrix3::zeros();
        let lambda_r = Vector3::new(1.0, 0.0, 0.0);
        let lambda_v = Vector3::zeros();
        let (dlr, dlv) = costate_derivative(&lambda_r, &lambda_v, &ar, &av);
        assert_eq!(dlr, Vector3::zeros());
        assert_eq!(dlv, -lambda_r);
    }

    #[test]
    fn zero_control_has_free_state_transition() {
        let mut config = control();
        config.acceleration_max = 1e-300;
        let state = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3];
        let lambda = [0.2, -0.1, 0.3, -0.5, 0.4, 0.1];
        let output =
            propagate_indirect(0.0, 1.0, &state, &lambda, &config, true, free_natural).unwrap();
        for i in 0..3 {
            assert!((output.state[i] - (state[i] + state[i + 3])).abs() < 2e-9);
            assert!((output.state[i + 3] - state[i + 3]).abs() < 2e-9);
        }
        let jac = output.jacobian.unwrap();
        for i in 0..6 {
            for j in 0..6 {
                assert!(jac[(i, j)].abs() < 1e-20);
            }
        }
    }

    #[test]
    fn free_variational_jacobian_matches_independent_perturbation() {
        let state = [0.0; 6];
        let lambda = [0.05, -0.04, 0.03, 1.2, 0.0, 0.0];
        let config = control();
        let nominal = propagate_indirect(0.0, 1.0, &state, &lambda, &config, true, free_natural)
            .expect("nominal propagation");
        let h = 1e-6;
        let mut shifted = lambda;
        shifted[3] += h;
        let changed = propagate_indirect(0.0, 1.0, &state, &shifted, &config, false, free_natural)
            .expect("perturbed propagation");
        let jac = nominal.jacobian.expect("Jacobian");
        for i in 0..6 {
            let finite_difference = (changed.state[i] - nominal.state[i]) / h;
            assert!(
                (jac[(i, 3)] - finite_difference).abs() < 2e-6,
                "row {i}: variational {} vs finite difference {finite_difference}",
                jac[(i, 3)],
            );
        }
    }

    #[test]
    fn zero_primer_has_smooth_zero_thrust_limit() {
        let config = control();
        assert!(config.throttle(0.0) > 0.0);
        let output =
            propagate_indirect(0.0, 0.1, &[0.0; 6], &[0.0; 6], &config, true, free_natural)
                .unwrap();
        assert!(output.state.iter().all(|value| value.abs() < 1e-12));
        assert!(output
            .jacobian
            .unwrap()
            .iter()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn cr3bp_variational_jacobian_matches_independent_perturbations() {
        let state = [0.85, 0.02, 0.01, 0.002, 0.18, 0.01];
        let lambda = [0.03, -0.02, 0.01, 0.4, 0.2, -0.1];
        let config = control();
        let nominal = propagate_cr3bp_indirect_for_test(&state, &lambda, &config, true);
        let jacobian = nominal.jacobian.as_ref().expect("Jacobian");
        // 带变分与不带变分的自适应步长序列不同。以纯传播为中心差分基准，
        // 避免把这一末态差异当成初始协态导数。
        let h = 1e-4;
        for column in 0..6 {
            let mut plus = lambda;
            let mut minus = lambda;
            plus[column] += h;
            minus[column] -= h;
            let upper = propagate_cr3bp_indirect_for_test(&state, &plus, &config, false);
            let lower = propagate_cr3bp_indirect_for_test(&state, &minus, &config, false);
            for row in 0..6 {
                let finite_difference = (upper.state[row] - lower.state[row]) / (2.0 * h);
                let tolerance = 2e-5 * finite_difference.abs().max(1.0);
                assert!(
                    (jacobian[(row, column)] - finite_difference).abs() < tolerance,
                    "row {row}, column {column}: variational {} vs finite difference {finite_difference}",
                    jacobian[(row, column)],
                );
            }
        }
    }

    fn propagate_cr3bp_indirect_for_test(
        state: &[f64; 6],
        lambda: &[f64; 6],
        config: &IndirectControl,
        with_jacobian: bool,
    ) -> IndirectPropagation {
        propagate_indirect(
            0.0,
            0.5,
            state,
            lambda,
            config,
            with_jacobian,
            |_t, state| {
                let derivative = e2m2e_forces::cr3bp::cr3bp_eom(0.012_150_585_609_624, state);
                let matrix = e2m2e_forces::cr3bp::cr3bp_jacobian_6x6(0.012_150_585_609_624, state);
                let mut ar = [0.0; 9];
                let mut av = [0.0; 9];
                for i in 0..3 {
                    for j in 0..3 {
                        ar[3 * i + j] = matrix[3 + i][j];
                        av[3 * i + j] = matrix[3 + i][3 + j];
                    }
                }
                Ok((derivative[3..6].try_into().expect("acceleration"), ar, av))
            },
        )
        .expect("CR3BP propagation")
    }

    #[test]
    fn invalid_time_interval_is_rejected() {
        let error = propagate_indirect(
            1.0,
            1.0,
            &[0.0; 6],
            &[0.0; 6],
            &control(),
            false,
            free_natural,
        );
        assert!(error.is_err());
    }
}
