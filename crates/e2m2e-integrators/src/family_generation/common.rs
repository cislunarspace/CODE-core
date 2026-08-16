//! 七类轨道族共享的传播、修正、模态和度量实现。

use crate::differential_correction::{run_correction, CorrectionRawResult};
use crate::family::collinear_center_modes;
use e2m2e_forces::cr3bp::propagate_cr3bp;

use super::types::{Context, PeriodicOrbit};

const SAMPLE_COUNT: usize = 1000;

#[derive(Debug)]
pub(crate) struct Failure {
    pub(crate) status: &'static str,
    pub(crate) cause: &'static str,
    pub(crate) message: String,
}

impl From<CorrectionRawResult> for Failure {
    fn from(result: CorrectionRawResult) -> Self {
        Self {
            status: result.status,
            cause: result.cause,
            message: result.message,
        }
    }
}

fn finish_correction(
    context: Context,
    result: CorrectionRawResult,
    full_period_multiplier: f64,
) -> Result<PeriodicOrbit, Failure> {
    let state = match result.solution_state {
        Some(state) => state,
        None => return Err(result.into()),
    };
    let solution_time = match result.solution_time {
        Some(time) => time,
        None => return Err(result.into()),
    };
    let period = solution_time * full_period_multiplier;
    let closure_error = closure_error(context, state, period).map_err(|message| Failure {
        status: "failed",
        cause: "integration_failed",
        message,
    })?;
    Ok(PeriodicOrbit {
        state,
        period,
        closure_error,
    })
}

#[allow(clippy::too_many_arguments)]
fn correct(
    context: Context,
    state: [f64; 6],
    initial_time: f64,
    constraint_indices: &[usize],
    free_variable_indices: &[usize],
    full_period: bool,
    recover_halo_time: bool,
    tolerance: f64,
    max_iterations: usize,
) -> Result<PeriodicOrbit, Failure> {
    let targets = vec![0.0; constraint_indices.len()];
    let result = run_correction(
        context.mu,
        state,
        initial_time,
        constraint_indices,
        &targets,
        free_variable_indices,
        full_period,
        recover_halo_time,
        max_iterations,
        tolerance,
        1e-14,
        1e10,
        context.rtol,
        context.atol,
        context.max_step,
        SAMPLE_COUNT,
    );
    finish_correction(context, result, if full_period { 1.0 } else { 2.0 })
}

pub(crate) fn correct_halo_fixed_z(
    context: Context,
    z0: f64,
    point: u8,
    guess: Option<&PeriodicOrbit>,
) -> Result<PeriodicOrbit, Failure> {
    let (mut state, period) = if let Some(orbit) = guess {
        (orbit.state, orbit.period)
    } else {
        halo_initial_guess(context.mu, point, z0)?
    };
    state[1] = 0.0;
    state[2] = z0;
    state[3] = 0.0;
    state[5] = 0.0;
    let orbit = correct(
        context,
        state,
        period / 2.0,
        &[1, 3, 5],
        &[0, 4, 6],
        false,
        true,
        1e-12,
        150,
    )?;
    if let Some(previous) = guess {
        if orbit.period > 1.2 * previous.period {
            return Err(Failure {
                status: "diverged",
                cause: "divergence_detected",
                message: "Halo 修正跳到长周期伪解".to_string(),
            });
        }
    }
    Ok(orbit)
}

pub(crate) fn correct_halo_fixed_x(
    context: Context,
    x0: f64,
    guess: &PeriodicOrbit,
) -> Result<PeriodicOrbit, Failure> {
    let mut state = guess.state;
    state[0] = x0;
    state[1] = 0.0;
    state[3] = 0.0;
    state[5] = 0.0;
    correct(
        context,
        state,
        guess.period / 2.0,
        &[1, 3, 5],
        &[2, 4, 6],
        false,
        true,
        1e-12,
        150,
    )
}

pub(crate) fn correct_axial_fixed_vz(
    context: Context,
    vz0: f64,
    guess: &PeriodicOrbit,
) -> Result<PeriodicOrbit, Failure> {
    let mut state = guess.state;
    state[1] = 0.0;
    state[2] = 0.0;
    state[3] = 0.0;
    state[5] = vz0;
    let orbit = correct(
        context,
        state,
        guess.period / 2.0,
        &[1, 2, 3],
        &[0, 4, 6],
        false,
        false,
        1e-12,
        150,
    )?;
    if orbit.period > 1.2 * guess.period {
        return Err(Failure {
            status: "diverged",
            cause: "divergence_detected",
            message: "Axial 修正跳到长周期伪解".to_string(),
        });
    }
    Ok(orbit)
}

pub(crate) fn correct_planar_fixed_x(
    context: Context,
    state: [f64; 6],
    period: f64,
) -> Result<PeriodicOrbit, Failure> {
    correct(
        context,
        state,
        period,
        &[1, 3, 4],
        &[1, 3, 4, 6],
        true,
        false,
        1e-12,
        150,
    )
}

fn halo_initial_guess(mu: f64, point: u8, z0: f64) -> Result<([f64; 6], f64), Failure> {
    let (x_l, omega_xy, _, _) = collinear_center_modes(mu, point).map_err(invalid_failure)?;
    let (k, delta) = match point {
        1 => (1.0, -1.0),
        2 => (-1.0, 1.0),
        _ => return Err(invalid_failure("Halo 平动点必须为 L1 或 L2".to_string())),
    };
    let amplitude = z0.abs();
    let x0 = x_l + delta * amplitude * 0.05;
    let vy0 = k * amplitude.sqrt() * 0.5 * omega_xy;
    Ok((
        [x0, 0.0, z0, 0.0, vy0, 0.0],
        std::f64::consts::TAU / omega_xy,
    ))
}

pub(crate) fn triangular_seed(
    context: Context,
    point: u8,
    short_period: bool,
    amplitude_km: f64,
) -> Result<([f64; 6], f64), Failure> {
    if point != 4 && point != 5 {
        return Err(invalid_failure("三角族平动点必须为 L4 或 L5".to_string()));
    }
    let discriminant = 1.0 - 27.0 * context.mu * (1.0 - context.mu);
    if discriminant <= 0.0 {
        return Err(invalid_failure(
            "质量参数超过 L4/L5 线性稳定范围".to_string(),
        ));
    }
    let omega_squared = if short_period {
        0.5 * (1.0 + discriminant.sqrt())
    } else {
        0.5 * (1.0 - discriminant.sqrt())
    };
    let omega = omega_squared.sqrt();
    let omega_xy =
        if point == 4 { 1.0 } else { -1.0 } * 3.0 * 3.0_f64.sqrt() / 4.0 * (1.0 - 2.0 * context.mu);
    let numerator = -omega_squared - 0.75;
    let denominator = omega_xy * omega_xy + 4.0 * omega_squared;
    // 取 X=-1 的相位约定，与既有 L4/L5 种子方向一致。
    let y_real = -numerator * omega_xy / denominator;
    let y_imag = numerator * 2.0 * omega / denominator;
    let position_norm = (1.0 + y_real * y_real + y_imag * y_imag).sqrt();
    let alpha = (amplitude_km / context.characteristic_length_km) / position_norm;
    let x_l = 0.5 - context.mu;
    let y_l = if point == 4 {
        3.0_f64.sqrt() / 2.0
    } else {
        -3.0_f64.sqrt() / 2.0
    };
    let state = [
        x_l - alpha,
        y_l + alpha * y_real,
        0.0,
        0.0,
        -alpha * omega * y_imag,
        0.0,
    ];
    Ok((state, std::f64::consts::TAU / omega))
}

pub(crate) fn closure_error(context: Context, state: [f64; 6], period: f64) -> Result<f64, String> {
    let result = propagate_cr3bp(
        context.mu,
        (0.0, period),
        &[period],
        &state,
        context.rtol,
        context.atol,
        context.max_step,
        Some(500_000),
    )
    .map_err(|error| error.to_string())?;
    let final_state = result.states.last().ok_or("周期传播未返回末态")?;
    Ok(final_state
        .iter()
        .zip(state)
        .map(|(final_value, initial_value)| (final_value - initial_value).abs())
        .fold(0.0, f64::max))
}

pub(crate) fn metric_minmax(
    context: Context,
    state: [f64; 6],
    period: f64,
    metric: &str,
    point: u8,
    sample_count: usize,
) -> Result<(f64, f64), Failure> {
    let count = sample_count.max(2);
    let dt = period / (count - 1) as f64;
    let mut times: Vec<f64> = (0..count).map(|index| index as f64 * dt).collect();
    times[count - 1] = period;
    let propagation = propagate_cr3bp(
        context.mu,
        (0.0, period),
        &times,
        &state,
        context.rtol,
        context.atol,
        context.max_step,
        Some(500_000),
    )
    .map_err(|error| Failure {
        status: "failed",
        cause: "integration_failed",
        message: error.to_string(),
    })?;
    let mut minimum = f64::INFINITY;
    let mut maximum = 0.0_f64;
    for sample in &propagation.states {
        let value = match metric {
            "moon-distance" => {
                let dx = sample[0] - (1.0 - context.mu);
                (dx * dx + sample[1] * sample[1] + sample[2] * sample[2]).sqrt()
            }
            "l45-distance" => {
                let x_l = 0.5 - context.mu;
                let y_l = if point == 4 {
                    3.0_f64.sqrt() / 2.0
                } else {
                    -3.0_f64.sqrt() / 2.0
                };
                ((sample[0] - x_l).powi(2) + (sample[1] - y_l).powi(2)).sqrt()
            }
            "z-amplitude" => sample[2].abs(),
            _ => return Err(invalid_failure(format!("未知轨道族度量 {metric}"))),
        };
        minimum = minimum.min(value);
        maximum = maximum.max(value);
    }
    Ok((minimum, maximum))
}

pub(crate) fn invalid_failure(message: String) -> Failure {
    Failure {
        status: "failed",
        cause: "invalid_input",
        message,
    }
}
