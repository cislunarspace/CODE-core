//! Q-law 低推力初猜的数值内核（纯 Rust，无 pyo3）。
//!
//! 反馈律、Q 函数评估和前向自适应积分都在这里执行；Python 侧只负责
//! 参数解析、后端选择和把轨迹重采样为求解器初猜。

use e2m2e_propagation::butcher::{explicit_rk_step, suggest_next_step};
use e2m2e_propagation::rk_methods::RkMethod;

const G0: f64 = 9.81;
const MAX_REJECTED_STEPS: usize = 100;
const MIN_STEP: f64 = 1e-6;
const NU_GRID: usize = 18;

#[derive(Debug)]
pub enum QlawError {
    Diverged(String),
    StepCollapsed(String),
    BudgetExhausted(String),
}

impl std::fmt::Display for QlawError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Diverged(message)
            | Self::StepCollapsed(message)
            | Self::BudgetExhausted(message) => {
                write!(f, "{message}")
            }
        }
    }
}

impl std::error::Error for QlawError {}

#[derive(Clone, Copy, Debug)]
pub struct SegmentEvaluation {
    pub a: f64,
    pub e: f64,
    pub inclination: f64,
    pub q_value: f64,
    pub inertial_direction: [f64; 3],
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(v: [f64; 3]) -> f64 {
    dot(v, v).sqrt()
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// 笛卡尔状态转换为经典开普勒根数 `(a, e, i, Ω, ω, ν)`。
pub fn rv_to_keplerian(r: [f64; 3], v: [f64; 3], mu: f64) -> [f64; 6] {
    let r_norm = norm(r);
    let v_norm = norm(v);
    let energy = v_norm * v_norm / 2.0 - mu / r_norm;
    let a = -mu / (2.0 * energy);

    let h_vec = cross(r, v);
    let h_norm = norm(h_vec);
    let n_vec = [-h_vec[1], h_vec[0], 0.0];
    let n_norm = norm(n_vec);
    let inclination = (h_vec[2] / h_norm).clamp(-1.0, 1.0).acos();

    let rv = dot(r, v);
    let e_vec = [
        ((v_norm * v_norm - mu / r_norm) * r[0] - rv * v[0]) / mu,
        ((v_norm * v_norm - mu / r_norm) * r[1] - rv * v[1]) / mu,
        ((v_norm * v_norm - mu / r_norm) * r[2] - rv * v[2]) / mu,
    ];
    let eccentricity = norm(e_vec);

    let raan = if n_norm > 1e-12 {
        let mut value = (n_vec[0] / n_norm).clamp(-1.0, 1.0).acos();
        if n_vec[1] < 0.0 {
            value = 2.0 * std::f64::consts::PI - value;
        }
        value
    } else {
        0.0
    };

    let argp = if eccentricity > 1e-10 && n_norm > 1e-12 {
        let mut value = (dot(n_vec, e_vec) / (n_norm * eccentricity))
            .clamp(-1.0, 1.0)
            .acos();
        if e_vec[2] < 0.0 {
            value = 2.0 * std::f64::consts::PI - value;
        }
        value
    } else if eccentricity > 1e-10 {
        let mut value = (e_vec[0] / eccentricity).clamp(-1.0, 1.0).acos();
        if e_vec[1] < 0.0 {
            value = 2.0 * std::f64::consts::PI - value;
        }
        value
    } else {
        0.0
    };

    let true_anomaly = if eccentricity > 1e-10 {
        let mut value = (dot(e_vec, r) / (eccentricity * r_norm))
            .clamp(-1.0, 1.0)
            .acos();
        if rv < 0.0 {
            value = 2.0 * std::f64::consts::PI - value;
        }
        value
    } else {
        if n_norm > 1e-12 {
            let mut value = (dot(n_vec, r) / (n_norm * r_norm)).clamp(-1.0, 1.0).acos();
            if r[2] < 0.0 {
                value = 2.0 * std::f64::consts::PI - value;
            }
            value
        } else {
            let mut value = (r[0] / r_norm).clamp(-1.0, 1.0).acos();
            if r[1] < 0.0 {
                value = 2.0 * std::f64::consts::PI - value;
            }
            value
        }
    };

    [a, eccentricity, inclination, raan, argp, true_anomaly]
}

fn gauss_b_aei(a: f64, e: f64, omega: f64, nu: f64, p: f64, h: f64, r: f64) -> [[f64; 3]; 3] {
    let cos_nu = nu.cos();
    let sin_nu = nu.sin();
    let cos_wv = (omega + nu).cos();
    [
        [2.0 * a * a / h * e * sin_nu, 2.0 * a * a / h * p / r, 0.0],
        [p / h * sin_nu, ((p + r) * cos_nu + r * e) / h, 0.0],
        [0.0, 0.0, r * cos_wv / h],
    ]
}

fn max_rates_aei(a: f64, e: f64, _inclination: f64, omega: f64, f: f64, mu: f64) -> [f64; 3] {
    if f <= 0.0 {
        return [1.0; 3];
    }
    let p = a * (1.0 - e * e);
    let h = (mu * p).sqrt();
    let mut maxima = [0.0_f64; 3];
    for index in 0..NU_GRID {
        let nu = 2.0 * std::f64::consts::PI * index as f64 / NU_GRID as f64;
        let radius = p / (1.0 + e * nu.cos());
        let b = gauss_b_aei(a, e, omega, nu, p, h, radius);
        for row in 0..3 {
            let row_norm =
                (b[row][0] * b[row][0] + b[row][1] * b[row][1] + b[row][2] * b[row][2]).sqrt();
            maxima[row] = maxima[row].max(row_norm);
        }
    }
    [maxima[0] * f, maxima[1] * f, maxima[2] * f]
}

fn q_value(oe: [f64; 3], target: [f64; 3], rates: [f64; 3]) -> f64 {
    let mut value = 0.0;
    for index in 0..3 {
        let delta = if rates[index] > 0.0 {
            (oe[index] - target[index]) / rates[index]
        } else {
            0.0
        };
        value += delta * delta;
    }
    value
}

fn q_gradient(oe: [f64; 4], target: [f64; 3], f: f64, mu: f64) -> [f64; 3] {
    let rates = max_rates_aei(oe[0], oe[1], oe[2], oe[3], f, mu);
    let eps = [(1e-3 * oe[0]).max(1.0), 1e-6, 1e-8];
    let mut gradient = [0.0; 3];
    for index in 0..3 {
        let mut plus = [oe[0], oe[1], oe[2]];
        let mut minus = plus;
        plus[index] += eps[index];
        minus[index] -= eps[index];
        gradient[index] =
            (q_value(plus, target, rates) - q_value(minus, target, rates)) / (2.0 * eps[index]);
    }
    gradient
}

fn direction_rtn_from_state(
    state: &[f64],
    target: [f64; 3],
    mu: f64,
    t_max: f64,
) -> ([f64; 3], f64) {
    let r = [state[0], state[1], state[2]];
    let v = [state[3], state[4], state[5]];
    let mass = state[6];
    let oe = rv_to_keplerian(r, v, mu);
    let f_km = t_max / mass / 1000.0;
    let gradient = q_gradient([oe[0], oe[1], oe[2], oe[4]], target, f_km, mu);
    let p = oe[0] * (1.0 - oe[1] * oe[1]);
    let h = (mu * p).sqrt();
    let radius = norm(r);
    let b = gauss_b_aei(oe[0], oe[1], oe[4], oe[5], p, h, radius);
    let mb = [
        gradient[0] * b[0][0] + gradient[1] * b[1][0] + gradient[2] * b[2][0],
        gradient[0] * b[0][1] + gradient[1] * b[1][1] + gradient[2] * b[2][1],
        gradient[0] * b[0][2] + gradient[1] * b[1][2] + gradient[2] * b[2][2],
    ];
    let norm_mb = norm(mb);
    if norm_mb < 1e-15 {
        return ([0.0, 1.0, 0.0], f_km);
    }
    let mut direction = [
        -(b[0][0] * gradient[0] + b[1][0] * gradient[1] + b[2][0] * gradient[2]) / norm_mb,
        -(b[0][1] * gradient[0] + b[1][1] * gradient[1] + b[2][1] * gradient[2]) / norm_mb,
        -(b[0][2] * gradient[0] + b[1][2] * gradient[1] + b[2][2] * gradient[2]) / norm_mb,
    ];
    let direction_norm = norm(direction);
    if direction_norm < 1e-15 {
        direction = [0.0, 1.0, 0.0];
    } else {
        for component in &mut direction {
            *component /= direction_norm;
        }
    }
    (direction, f_km)
}

fn rtn_to_inertial(u_rtn: [f64; 3], r: [f64; 3], v: [f64; 3]) -> [f64; 3] {
    let r_norm = norm(r);
    let r_hat = [r[0] / r_norm, r[1] / r_norm, r[2] / r_norm];
    let h_vec = cross(r, v);
    let h_norm = norm(h_vec);
    if h_norm < 1e-15 {
        return [
            u_rtn[0] * r_hat[0],
            u_rtn[0] * r_hat[1],
            u_rtn[0] * r_hat[2],
        ];
    }
    let n_hat = [h_vec[0] / h_norm, h_vec[1] / h_norm, h_vec[2] / h_norm];
    let t_hat = cross(n_hat, r_hat);
    [
        u_rtn[0] * r_hat[0] + u_rtn[1] * t_hat[0] + u_rtn[2] * n_hat[0],
        u_rtn[0] * r_hat[1] + u_rtn[1] * t_hat[1] + u_rtn[2] * n_hat[1],
        u_rtn[0] * r_hat[2] + u_rtn[1] * t_hat[2] + u_rtn[2] * n_hat[2],
    ]
}

fn eom_7d(y: &[f64], target: [f64; 3], mu: f64, t_max: f64, isp: f64) -> Vec<f64> {
    let r = [y[0], y[1], y[2]];
    let v = [y[3], y[4], y[5]];
    let radius = norm(r);
    let acceleration_gravity = [
        -mu / radius.powi(3) * r[0],
        -mu / radius.powi(3) * r[1],
        -mu / radius.powi(3) * r[2],
    ];
    let (u_rtn, _) = direction_rtn_from_state(y, target, mu, t_max);
    let u_inertial = rtn_to_inertial(u_rtn, r, v);
    let thrust_acceleration = t_max / y[6] / 1000.0;
    let mdot = -t_max / (isp * G0);
    vec![
        v[0],
        v[1],
        v[2],
        acceleration_gravity[0] + thrust_acceleration * u_inertial[0],
        acceleration_gravity[1] + thrust_acceleration * u_inertial[1],
        acceleration_gravity[2] + thrust_acceleration * u_inertial[2],
        mdot,
    ]
}

/// 对一个状态执行 Q-law 的段中点评估。
pub fn evaluate_segment(
    state: &[f64; 7],
    target: [f64; 3],
    mu: f64,
    t_max: f64,
) -> SegmentEvaluation {
    let r = [state[0], state[1], state[2]];
    let v = [state[3], state[4], state[5]];
    let oe = rv_to_keplerian(r, v, mu);
    let f_km = t_max / state[6] / 1000.0;
    let rates = max_rates_aei(oe[0], oe[1], oe[2], oe[4], f_km, mu);
    let q = q_value([oe[0], oe[1], oe[2]], target, rates);
    let (u_rtn, _) = direction_rtn_from_state(state, target, mu, t_max);
    let inertial_direction = rtn_to_inertial(u_rtn, r, v);
    SegmentEvaluation {
        a: oe[0],
        e: oe[1],
        inclination: oe[2],
        q_value: q,
        inertial_direction,
    }
}

/// 执行完整的 Q-law 自适应反馈积分，并返回所有已验收的状态。
#[allow(clippy::too_many_arguments)]
pub fn propagate(
    t0: f64,
    tf: f64,
    y0: [f64; 7],
    target: [f64; 3],
    mu: f64,
    t_max: f64,
    isp: f64,
    h_init: f64,
    tol: f64,
    max_steps: usize,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), QlawError> {
    let table = RkMethod::Pd45.table();
    let mut state = y0.to_vec();
    let mut times = vec![t0];
    let mut states = vec![state.clone()];
    let mut t = t0;
    let mut h = h_init;
    let mut n_steps = 0usize;
    let mut rejected_streak = 0usize;

    while t < tf && n_steps < max_steps {
        n_steps += 1;
        if t + h > tf {
            h = tf - t;
        }
        let (new_state, error) = explicit_rk_step(
            table,
            t,
            &state,
            h,
            |_, y| Ok::<Vec<f64>, String>(eom_7d(y, target, mu, t_max, isp)),
            Some(6),
        )
        .map_err(QlawError::Diverged)?;
        if new_state.iter().any(|value| !value.is_finite()) || !error.is_finite() {
            return Err(QlawError::StepCollapsed(
                "Q-law propagation produced non-finite values".to_owned(),
            ));
        }
        if error <= tol {
            t += h;
            state = new_state;
            times.push(t);
            states.push(state.clone());
            h = suggest_next_step(h, error, tol, RkMethod::Pd45.embedded_order());
            rejected_streak = 0;
        } else {
            h = suggest_next_step(h, error, tol, RkMethod::Pd45.embedded_order());
            rejected_streak += 1;
            if rejected_streak >= MAX_REJECTED_STEPS {
                return Err(QlawError::Diverged(format!(
                    "{} consecutive rejected steps at t={t:.6e} (h={h:.3e}); trajectory diverges",
                    rejected_streak
                )));
            }
        }
        if h < MIN_STEP && tf - t > MIN_STEP {
            return Err(QlawError::StepCollapsed(format!(
                "step size collapsed below 1e-6 at t={t:.6e}"
            )));
        }
    }

    if t < tf - 1e-9 {
        return Err(QlawError::BudgetExhausted(format!(
            "step budget ({max_steps}) exhausted at t={t:.6e} < tf={tf:.6e}"
        )));
    }
    Ok((times, states))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circular_orbit_has_expected_elements() {
        let radius = 7000.0_f64;
        let mu = 398600.435507;
        let speed = (mu / radius).sqrt();
        let oe = rv_to_keplerian([radius, 0.0, 0.0], [0.0, speed, 0.0], mu);
        assert!((oe[0] - radius).abs() / radius < 1e-12);
        assert!(oe[1] < 1e-10);
        assert!(oe[2].abs() < 1e-12);
    }

    #[test]
    fn segment_direction_is_unit_length() {
        let state = [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0, 1000.0];
        let result = evaluate_segment(&state, [7020.0, 0.0, 0.0], 398600.435507, 0.5);
        let length = norm(result.inertial_direction);
        assert!((length - 1.0).abs() < 1e-12);
    }
}
