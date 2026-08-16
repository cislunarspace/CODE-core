//! 轨道族生成的纯 Rust 数值原子。
//!
//! Python 只构造族参数并解释结果；本模块承担：
//! - 共线平动点求根、线性中心模态与 Lissajous 有界轨迹采样；
//! - 周期轨道传播后的近月距、L4/L5 径向振幅与面外振幅测量。

use e2m2e_forces::cr3bp::propagate_cr3bp;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const DEFAULT_MAX_STEP: f64 = 0.01;

fn equilibrium_x(mu: f64, x: f64) -> f64 {
    let dx1 = x + mu;
    let dx2 = x - 1.0 + mu;
    x - (1.0 - mu) * dx1 / dx1.abs().powi(3) - mu * dx2 / dx2.abs().powi(3)
}

pub(crate) fn collinear_libration_x(mu: f64, point: u8) -> Result<f64, String> {
    if !mu.is_finite() || !(0.0..0.5).contains(&mu) {
        return Err(format!("质量参数 mu 必须在 (0, 0.5) 内，当前 {mu}"));
    }
    let eps = 1e-8;
    let (mut lo, mut hi) = match point {
        1 => (-mu + eps, 1.0 - mu - eps),
        2 => (1.0 - mu + eps, 2.0),
        3 => (-2.0, -mu - eps),
        _ => return Err(format!("collinear_point 必须为 1、2 或 3，当前 {point}")),
    };
    let mut f_lo = equilibrium_x(mu, lo);
    let f_hi = equilibrium_x(mu, hi);
    if !f_lo.is_finite() || !f_hi.is_finite() || f_lo * f_hi > 0.0 {
        return Err(format!("L{point} 平动点求根区间未包围根"));
    }
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        let f_mid = equilibrium_x(mu, mid);
        if f_mid.abs() < 1e-14 || (hi - lo).abs() < 1e-14 {
            return Ok(mid);
        }
        if f_lo * f_mid <= 0.0 {
            hi = mid;
        } else {
            lo = mid;
            f_lo = f_mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

pub(crate) fn collinear_center_modes(mu: f64, point: u8) -> Result<(f64, f64, f64, f64), String> {
    let x_l = collinear_libration_x(mu, point)?;
    let r1 = (x_l + mu).abs();
    let r2 = (x_l - 1.0 + mu).abs();
    let c2 = (1.0 - mu) / r1.powi(3) + mu / r2.powi(3);
    let a = 2.0 - c2;
    let b = (1.0 + 2.0 * c2) * (1.0 - c2);
    let discriminant = a * a - 4.0 * b;
    if discriminant <= 0.0 || c2 <= 0.0 {
        return Err(format!("L{point} 线性中心模态不存在"));
    }
    let lambda_squared = 0.5 * (-a - discriminant.sqrt());
    if lambda_squared >= 0.0 {
        return Err(format!("L{point} 面内中心频率无效"));
    }
    let omega_xy = (-lambda_squared).sqrt();
    let omega_z = c2.sqrt();
    let y_ratio = (omega_xy * omega_xy + 1.0 + 2.0 * c2) / (2.0 * omega_xy);
    Ok((x_l, omega_xy, omega_z, y_ratio))
}

/// 返回共线点和线性中心模态参数 `(x_l, omega_xy, omega_z, y_ratio)`。
#[pyfunction]
pub fn collinear_center_modes_py(mu: f64, collinear_point: u8) -> PyResult<(f64, f64, f64, f64)> {
    collinear_center_modes(mu, collinear_point).map_err(PyValueError::new_err)
}

/// Rust Lissajous 线性中心流多点轨迹。
///
/// 面内与面外中心模态分别按独立频率解析推进；不把状态重新送入含双曲方向的
/// 完整 CR3BP，因此轨迹按构造保持有界。该结果是参数采样轨迹，不宣称周期闭合。
#[pyfunction]
#[pyo3(signature = (mu, collinear_point, char_length_km, amplitude_in_km, amplitude_out_km, phase_in, phase_out, n_periods=3, points_per_period=60))]
#[allow(clippy::too_many_arguments)]
pub fn lissajous_bounded_trajectory_py(
    mu: f64,
    collinear_point: u8,
    char_length_km: f64,
    amplitude_in_km: f64,
    amplitude_out_km: f64,
    phase_in: f64,
    phase_out: f64,
    n_periods: usize,
    points_per_period: usize,
) -> PyResult<(Vec<[f64; 6]>, Vec<f64>, f64)> {
    if !char_length_km.is_finite() || char_length_km <= 0.0 {
        return Err(PyValueError::new_err("char_length_km 必须为正数"));
    }
    if !amplitude_in_km.is_finite()
        || amplitude_in_km <= 0.0
        || !amplitude_out_km.is_finite()
        || amplitude_out_km <= 0.0
    {
        return Err(PyValueError::new_err("Lissajous 振幅必须为正数"));
    }
    if !(0.0..=1.0).contains(&phase_in) || !(0.0..=1.0).contains(&phase_out) {
        return Err(PyValueError::new_err("Lissajous 相位必须在 [0, 1] 内"));
    }
    if n_periods == 0 || points_per_period < 2 {
        return Err(PyValueError::new_err(
            "n_periods 必须大于 0，points_per_period 必须至少为 2",
        ));
    }

    let (x_l, omega_xy, omega_z, y_ratio) =
        collinear_center_modes(mu, collinear_point).map_err(PyValueError::new_err)?;
    let planar_norm = (1.0 + y_ratio * y_ratio).sqrt();
    let alpha = (amplitude_in_km / char_length_km) / planar_norm;
    let beta = amplitude_out_km / char_length_km;
    let phi = phase_in * std::f64::consts::TAU;
    let psi = phase_out * std::f64::consts::TAU;
    let nominal_period = std::f64::consts::TAU / omega_xy;
    let count = (points_per_period * n_periods).max(30);
    let final_time = n_periods as f64 * nominal_period;
    let dt = final_time / (count - 1) as f64;

    let mut states = Vec::with_capacity(count);
    let mut times = Vec::with_capacity(count);
    for index in 0..count {
        let time = if index + 1 == count {
            final_time
        } else {
            index as f64 * dt
        };
        let theta_xy = omega_xy * time + phi;
        let theta_z = omega_z * time + psi;
        states.push([
            x_l + alpha * theta_xy.cos(),
            -alpha * y_ratio * theta_xy.sin(),
            beta * theta_z.cos(),
            -alpha * omega_xy * theta_xy.sin(),
            -alpha * omega_xy * y_ratio * theta_xy.cos(),
            -beta * omega_z * theta_z.sin(),
        ]);
        times.push(time);
    }
    Ok((states, times, nominal_period))
}

fn linspace(end: f64, count: usize) -> Vec<f64> {
    let step = end / (count - 1) as f64;
    let mut values: Vec<f64> = (0..count).map(|i| i as f64 * step).collect();
    values[count - 1] = end;
    values
}

/// 传播周期轨道并在 Rust 内测量族几何量。
///
/// metric 可取 moon-distance、l45-distance 或 z-amplitude；返回对应
/// 绝对距离或振幅的 (minimum, maximum)（无量纲）。
#[pyfunction]
#[pyo3(signature = (mu, metric, point, initial_state, period, rtol=1e-12, atol=1e-12, max_step=None, sample_count=1000))]
#[allow(clippy::too_many_arguments)]
pub fn orbit_family_metric_py(
    mu: f64,
    metric: &str,
    point: u8,
    initial_state: [f64; 6],
    period: f64,
    rtol: f64,
    atol: f64,
    max_step: Option<f64>,
    sample_count: usize,
) -> PyResult<(f64, f64)> {
    if !period.is_finite() || period <= 0.0 {
        return Err(PyValueError::new_err("period 必须为正数"));
    }
    if !rtol.is_finite() || rtol <= 0.0 || !atol.is_finite() || atol <= 0.0 {
        return Err(PyValueError::new_err("rtol/atol 必须为正数"));
    }
    if sample_count < 2 {
        return Err(PyValueError::new_err("sample_count 必须至少为 2"));
    }
    let t_eval = linspace(period, sample_count);
    let propagation = propagate_cr3bp(
        mu,
        (0.0, period),
        &t_eval,
        &initial_state,
        rtol,
        atol,
        max_step.or(Some(DEFAULT_MAX_STEP)),
        Some(500_000),
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;

    let mut minimum = f64::INFINITY;
    let mut maximum = 0.0_f64;
    for state in &propagation.states {
        let value = match metric {
            "moon-distance" => {
                let dx = state[0] - (1.0 - mu);
                (dx * dx + state[1] * state[1] + state[2] * state[2]).sqrt()
            }
            "l45-distance" => {
                if point != 4 && point != 5 {
                    return Err(PyValueError::new_err("l45-distance 的 point 必须为 4 或 5"));
                }
                let x_l = 0.5 - mu;
                let y_l = if point == 4 {
                    3.0_f64.sqrt() / 2.0
                } else {
                    -3.0_f64.sqrt() / 2.0
                };
                let dx = state[0] - x_l;
                let dy = state[1] - y_l;
                (dx * dx + dy * dy).sqrt()
            }
            "z-amplitude" => state[2].abs(),
            _ => return Err(PyValueError::new_err(format!("未知族度量 {metric:?}"))),
        };
        minimum = minimum.min(value);
        maximum = maximum.max(value);
    }
    Ok((minimum, maximum))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MU: f64 = 0.012_150_668_3;
    const DU: f64 = 384_400.0;

    #[test]
    fn collinear_points_satisfy_equilibrium() {
        for point in [1, 2, 3] {
            let x = collinear_libration_x(MU, point).unwrap();
            assert!(equilibrium_x(MU, x).abs() < 1e-11);
        }
    }

    #[test]
    fn lissajous_linear_center_flow_is_bounded() {
        let (states, times, period) =
            lissajous_bounded_trajectory_py(MU, 1, DU, 500.0, 2000.0, 0.01, 0.55, 3, 60).unwrap();
        assert_eq!(states.len(), 180);
        assert_eq!(times.len(), states.len());
        assert!(period > 1.5 && period < 5.0);
        let x_l = collinear_libration_x(MU, 1).unwrap();
        let max_distance = states
            .iter()
            .map(|state| {
                ((state[0] - x_l).powi(2) + state[1].powi(2) + state[2].powi(2)).sqrt() * DU
            })
            .fold(0.0, f64::max);
        assert!(max_distance < 2.5 * (500.0_f64.hypot(2000.0)));
    }

    #[test]
    fn z_metric_measures_propagated_amplitude() {
        let result = orbit_family_metric_py(
            MU,
            "z-amplitude",
            0,
            [0.8, 0.0, 0.01, 0.0, 0.2, 0.0],
            0.1,
            1e-12,
            1e-12,
            Some(0.01),
            20,
        )
        .unwrap();
        assert!(result.1 >= result.0);
        assert!(result.1.is_finite());
    }
}
