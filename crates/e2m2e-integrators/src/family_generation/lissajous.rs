//! Lissajous 振幅采样与非线性中心约化流轨迹。

use crate::family::collinear_center_modes;

use super::types::{Context, Member, Outcome};

#[allow(clippy::too_many_arguments)]
pub(crate) fn generate(
    context: Context,
    point: u8,
    amplitude_in_km: f64,
    amplitude_out_km: f64,
    phase_in: f64,
    phase_out: f64,
    member_limit: usize,
    n_periods: usize,
) -> Result<Outcome, String> {
    if point != 1 && point != 2 && point != 3 {
        return Err("Lissajous 平动点必须为 L1、L2 或 L3".to_string());
    }
    if member_limit == 0 {
        return Err("n_orbits 必须大于 0".to_string());
    }
    if n_periods == 0 {
        return Err("n_periods 必须大于 0".to_string());
    }
    if amplitude_in_km <= 0.0 || amplitude_out_km <= 0.0 {
        return Err("Lissajous 振幅必须为正数".to_string());
    }
    if !(0.0..=1.0).contains(&phase_in) || !(0.0..=1.0).contains(&phase_out) {
        return Err("Lissajous 相位必须在 [0, 1] 内".to_string());
    }
    let (x_l, omega_xy, omega_z, y_ratio) = match collinear_center_modes(context.mu, point) {
        Ok(modes) => modes,
        Err(message) => {
            return Ok(Outcome::soft_failure(
                "lissajous",
                "quasi-periodic",
                member_limit,
                Vec::new(),
                "failed",
                "backend_failure",
                message,
            ));
        }
    };
    let planar_norm = (1.0 + y_ratio * y_ratio).sqrt();
    let nominal_period = std::f64::consts::TAU / omega_xy;
    let point_count = (60 * n_periods).max(30);
    let final_time = n_periods as f64 * nominal_period;
    let dt = final_time / (point_count - 1) as f64;
    let phi = phase_in * std::f64::consts::TAU;
    let psi = phase_out * std::f64::consts::TAU;
    let mut members = Vec::with_capacity(member_limit);

    for index in 1..=member_limit {
        let fraction = index as f64 / member_limit as f64;
        let alpha = (fraction * amplitude_in_km / context.characteristic_length_km) / planar_norm;
        let beta = fraction * amplitude_out_km / context.characteristic_length_km;
        let mut states = Vec::with_capacity(point_count);
        let mut times = Vec::with_capacity(point_count);
        let mut reduced = [
            alpha * phi.cos(),
            -alpha * omega_xy * phi.sin(),
            beta * psi.cos(),
            -beta * omega_z * psi.sin(),
        ];
        for sample_index in 0..point_count {
            let time = if sample_index + 1 == point_count {
                final_time
            } else {
                sample_index as f64 * dt
            };
            states.push(reconstruct_state(x_l, omega_xy, y_ratio, reduced));
            times.push(time);
            if sample_index + 1 < point_count {
                let next_time = if sample_index + 2 == point_count {
                    final_time
                } else {
                    (sample_index + 1) as f64 * dt
                };
                reduced =
                    advance_reduced(context, x_l, omega_xy, y_ratio, reduced, next_time - time);
            }
        }
        members.push(Member {
            states,
            times,
            period: Some(nominal_period),
            closure_error: None,
            amplitude_km: None,
            perilune_height_km: None,
            sampling_fraction: Some(fraction),
            jacobi_drift: None,
            newton_iterations: None,
            tangent_system_rank: None,
            tangent_system_condition: None,
            augmented_system_rank: None,
            augmented_system_condition: None,
            step_size: None,
        });
    }
    Ok(Outcome::converged(
        "lissajous",
        "quasi-periodic",
        member_limit,
        members,
    ))
}

fn reconstruct_state(x_l: f64, omega_xy: f64, y_ratio: f64, reduced: [f64; 4]) -> [f64; 6] {
    let [x, vx, z, vz] = reduced;
    [
        x_l + x,
        y_ratio / omega_xy * vx,
        z,
        vx,
        -omega_xy * y_ratio * x,
        vz,
    ]
}

fn reduced_derivative(
    context: Context,
    x_l: f64,
    omega_xy: f64,
    y_ratio: f64,
    reduced: [f64; 4],
) -> [f64; 4] {
    let state = reconstruct_state(x_l, omega_xy, y_ratio, reduced);
    let dx1 = state[0] + context.mu;
    let dx2 = state[0] - 1.0 + context.mu;
    let r1_squared = dx1 * dx1 + state[1] * state[1] + state[2] * state[2];
    let r2_squared = dx2 * dx2 + state[1] * state[1] + state[2] * state[2];
    let r1_cubed = r1_squared * r1_squared.sqrt();
    let r2_cubed = r2_squared * r2_squared.sqrt();
    let potential_x = state[0] - (1.0 - context.mu) * dx1 / r1_cubed - context.mu * dx2 / r2_cubed;
    let potential_z = -(1.0 - context.mu) * state[2] / r1_cubed - context.mu * state[2] / r2_cubed;
    [
        reduced[1],
        2.0 * state[4] + potential_x,
        reduced[3],
        potential_z,
    ]
}

fn advance_reduced(
    context: Context,
    x_l: f64,
    omega_xy: f64,
    y_ratio: f64,
    mut state: [f64; 4],
    interval: f64,
) -> [f64; 4] {
    const SUBSTEPS: usize = 8;
    let step = interval / SUBSTEPS as f64;
    for _ in 0..SUBSTEPS {
        let k1 = reduced_derivative(context, x_l, omega_xy, y_ratio, state);
        let k2 = reduced_derivative(
            context,
            x_l,
            omega_xy,
            y_ratio,
            add_scaled(state, k1, 0.5 * step),
        );
        let k3 = reduced_derivative(
            context,
            x_l,
            omega_xy,
            y_ratio,
            add_scaled(state, k2, 0.5 * step),
        );
        let k4 = reduced_derivative(context, x_l, omega_xy, y_ratio, add_scaled(state, k3, step));
        for index in 0..4 {
            state[index] +=
                step / 6.0 * (k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index]);
        }
    }
    state
}

fn add_scaled(state: [f64; 4], derivative: [f64; 4], scale: f64) -> [f64; 4] {
    std::array::from_fn(|index| state[index] + scale * derivative[index])
}
