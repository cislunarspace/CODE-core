//! WSB 三维网格候选评估的纯 Rust 数值核。
//!
//! 每个 `(sun_phase, tof)` 任务内部遍历出发相位角，直接调用 BCR4BP Rust
//! 传播器，再复刻 Python 的密采样截面检测与候选筛选。任务结果先按网格
//! 字典序收集，再稳定按总 Δv 排序，保持现有 Python 搜索的确定性。

use std::cmp::Ordering;

use crate::bcr4bp::propagate_bcr4bp;

const MIN_BISECT_XTOL: f64 = 1e-14;
const MAX_BISECT_ITER: usize = 50;
const MIN_VELOCITY_NORM: f64 = 1e-12;
const MIN_RADIAL_STEP: f64 = 1e-12;

/// WSB 网格搜索的标量配置。
#[derive(Clone, Debug)]
pub struct WsbSearchParams {
    pub sun_phase_min: f64,
    pub sun_phase_max: f64,
    pub n_sun_phase: usize,
    pub departure_phase_min: f64,
    pub departure_phase_max: f64,
    pub n_departure_phase: usize,
    pub tof_min_sec: f64,
    pub tof_max_sec: f64,
    pub n_tof: usize,
    pub perilune_alt_min: f64,
    pub perilune_alt_max: f64,
    pub max_total_dv: f64,
    pub h2_energy_threshold: f64,
    pub n_propagation_samples: usize,
    pub rtol: f64,
    pub atol: f64,
    pub max_step: f64,
    pub secondary_radius_km: f64,
    pub characteristic_length_km: f64,
    pub characteristic_time_sec: f64,
}

/// 单个 WSB 候选。
#[derive(Clone, Debug)]
pub struct WsbCandidate {
    pub sun_phase0: f64,
    pub departure_phase: f64,
    pub tof_sec: f64,
    pub departure_state: [f64; 6],
    pub perilune_state: [f64; 6],
    pub perilune_alt_km: f64,
    pub perilune_time_dim: f64,
    pub arrival_state: [f64; 6],
    pub h2_kepler: f64,
    pub dv_departure: f64,
    pub dv_arrival: f64,
    pub total_dv: f64,
    pub arrival_time_dim: f64,
}

/// WSB 网格搜索结果及传播失败计数。
#[derive(Clone, Debug)]
pub struct WsbSearchResult {
    pub candidates: Vec<WsbCandidate>,
    pub n_propagation_failures: usize,
}

/// 单个 `(sun_phase, tof)` 任务共享的只读数值输入。
struct WsbTaskContext<'a> {
    target_state: &'a [f64; 6],
    mu: f64,
    mu_sun: f64,
    sun_distance: f64,
    sun_angular_rate: f64,
    params: &'a WsbSearchParams,
    r0: &'a [f64; 3],
    v_park: &'a [f64; 3],
    v_hat: &'a [f64; 3],
    r_hat: &'a [f64; 3],
    v_tli: f64,
    r_target: f64,
}

fn norm3(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn distance3(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn linspace_inclusive(start: f64, end: f64, n: usize) -> Vec<f64> {
    match n {
        0 => Vec::new(),
        1 => vec![start],
        _ => (0..n)
            .map(|i| start + (end - start) * i as f64 / (n - 1) as f64)
            .collect(),
    }
}

fn linspace_exclusive(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    (0..n)
        .map(|i| start + (end - start) * i as f64 / n as f64)
        .collect()
}

fn periapsis_value(state: &[f64; 6], mu: f64) -> f64 {
    let moon_x = 1.0 - mu;
    (state[0] - moon_x) * state[3] + state[1] * state[4] + state[2] * state[5]
}

fn interpolate_state(t0: f64, t1: f64, y0: &[f64; 6], y1: &[f64; 6], t: f64) -> [f64; 6] {
    let weight = (t - t0) / (t1 - t0);
    let mut state = [0.0; 6];
    for i in 0..6 {
        state[i] = y0[i] + weight * (y1[i] - y0[i]);
    }
    state
}

fn refine_periapsis_crossing(
    t0: f64,
    t1: f64,
    y0: &[f64; 6],
    y1: &[f64; 6],
    mu: f64,
) -> (f64, [f64; 6]) {
    let mut a = t0;
    let mut b = t1;
    let mut fa = periapsis_value(y0, mu);
    let fb = periapsis_value(y1, mu);
    if fa == 0.0 {
        return (t0, *y0);
    }
    if fb == 0.0 {
        return (t1, *y1);
    }
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }

    let mut root = 0.5 * (a + b);
    for _ in 0..MAX_BISECT_ITER {
        root = 0.5 * (a + b);
        let state = interpolate_state(t0, t1, y0, y1, root);
        let fc = periapsis_value(&state, mu);
        if fc == 0.0 || (b - a) * 0.5 < MIN_BISECT_XTOL {
            return (root, state);
        }
        if fa * fc < 0.0 {
            b = root;
        } else {
            a = root;
            fa = fc;
        }
    }
    let state = interpolate_state(t0, t1, y0, y1, root);
    (root, state)
}

fn first_periapsis_crossing(
    times: &[f64],
    states: &[[f64; 6]],
    mu: f64,
) -> Option<(f64, [f64; 6], usize)> {
    let values: Vec<f64> = states
        .iter()
        .map(|state| periapsis_value(state, mu))
        .collect();
    for i in 0..values.len().saturating_sub(1) {
        let s0 = values[i];
        let s1 = values[i + 1];
        if s0 == 0.0 || s0 * s1 > 0.0 {
            continue;
        }
        let (time, state) =
            refine_periapsis_crossing(times[i], times[i + 1], &states[i], &states[i + 1], mu);
        return Some((time, state, i));
    }
    None
}

fn compute_kepler_energy_moon(state: &[f64; 6], mu: f64) -> f64 {
    let moon_x = 1.0 - mu;
    let rx = state[0] - moon_x;
    let ry = state[1];
    let rz = state[2];
    let r_moon = (rx * rx + ry * ry + rz * rz).sqrt();
    let rel_vx = state[3] - state[1];
    let rel_vy = state[4] + state[0] - moon_x;
    let rel_vz = state[5];
    0.5 * (rel_vx * rel_vx + rel_vy * rel_vy + rel_vz * rel_vz) - mu / r_moon
}

fn evaluate_task(
    context: &WsbTaskContext<'_>,
    sun_phase0: f64,
    tof_sec: f64,
) -> Result<(Vec<WsbCandidate>, usize), String> {
    let target_state = context.target_state;
    let mu = context.mu;
    let mu_sun = context.mu_sun;
    let sun_distance = context.sun_distance;
    let sun_angular_rate = context.sun_angular_rate;
    let params = context.params;
    let r0 = context.r0;
    let v_park = context.v_park;
    let v_hat = context.v_hat;
    let r_hat = context.r_hat;
    let v_tli = context.v_tli;
    let r_target = context.r_target;
    let angle_grid = linspace_exclusive(
        params.departure_phase_min,
        params.departure_phase_max,
        params.n_departure_phase,
    );
    let tof_dim = tof_sec / params.characteristic_time_sec;
    let t_eval = linspace_inclusive(0.0, tof_dim, params.n_propagation_samples);
    let moon_x = 1.0 - mu;
    let mut candidates = Vec::new();
    let mut n_propagation_failures = 0;

    for angle in angle_grid {
        let (cos_angle, sin_angle) = (angle.cos(), angle.sin());
        let v_dir = [
            cos_angle * v_hat[0] + sin_angle * r_hat[0],
            cos_angle * v_hat[1] + sin_angle * r_hat[1],
            cos_angle * v_hat[2] + sin_angle * r_hat[2],
        ];
        let v_dep = [v_dir[0] * v_tli, v_dir[1] * v_tli, v_dir[2] * v_tli];
        let initial_state = [r0[0], r0[1], r0[2], v_dep[0], v_dep[1], v_dep[2]];
        let dv_departure = distance3(&v_dep, v_park);

        let propagated = propagate_bcr4bp(
            mu,
            mu_sun,
            sun_distance,
            sun_angular_rate,
            sun_phase0,
            (0.0, tof_dim),
            &t_eval,
            &initial_state,
            params.rtol,
            params.atol,
            Some(params.max_step),
            None,
        );
        let propagated = match propagated {
            Ok(result) => result,
            Err(error) if error.contains("step size collapsed") => {
                n_propagation_failures += 1;
                continue;
            }
            Err(error) => return Err(error),
        };

        let Some((perilune_time_dim, perilune_state, peri_idx)) =
            first_periapsis_crossing(&propagated.times, &propagated.states, mu)
        else {
            continue;
        };

        let r_peri_rel = distance3(&perilune_state[..3], &[moon_x, 0.0, 0.0]);
        let perilune_alt_km =
            r_peri_rel * params.characteristic_length_km - params.secondary_radius_km;
        if perilune_alt_km < params.perilune_alt_min || perilune_alt_km > params.perilune_alt_max {
            continue;
        }

        let h2_kepler = compute_kepler_energy_moon(&perilune_state, mu);
        if h2_kepler >= params.h2_energy_threshold {
            continue;
        }

        let r_traj: Vec<f64> = propagated
            .states
            .iter()
            .map(|state| norm3(&state[..3]))
            .collect();
        let mut arrival_state = *propagated.states.last().expect("t_eval 不能为空");
        let mut arrival_time_dim = tof_dim;
        for k in peri_idx..r_traj.len().saturating_sub(1) {
            let r1 = r_traj[k];
            let r2 = r_traj[k + 1];
            if (r1 <= r_target && r_target <= r2) || (r2 <= r_target && r_target <= r1) {
                let fraction = if (r2 - r1).abs() > MIN_RADIAL_STEP {
                    (r_target - r1) / (r2 - r1)
                } else {
                    0.5
                };
                arrival_time_dim = propagated.times[k]
                    + fraction * (propagated.times[k + 1] - propagated.times[k]);
                arrival_state = interpolate_state(
                    propagated.times[k],
                    propagated.times[k + 1],
                    &propagated.states[k],
                    &propagated.states[k + 1],
                    arrival_time_dim,
                );
                break;
            }
        }

        let tof_sec_actual = arrival_time_dim * params.characteristic_time_sec;
        let dv_arrival = distance3(&arrival_state[3..], &target_state[3..]);
        let total_dv = dv_departure + dv_arrival;
        if total_dv > params.max_total_dv {
            continue;
        }

        candidates.push(WsbCandidate {
            sun_phase0,
            departure_phase: angle,
            tof_sec: tof_sec_actual,
            departure_state: initial_state,
            perilune_state,
            perilune_alt_km,
            perilune_time_dim,
            arrival_state,
            h2_kepler,
            dv_departure,
            dv_arrival,
            total_dv,
            arrival_time_dim,
        });
    }

    Ok((candidates, n_propagation_failures))
}

fn prepare_inputs(
    departure_state: &[f64; 6],
    target_state: &[f64; 6],
    mu: f64,
) -> ([f64; 3], [f64; 3], [f64; 3], [f64; 3], f64, f64) {
    let r0 = [departure_state[0], departure_state[1], departure_state[2]];
    let v_park = [departure_state[3], departure_state[4], departure_state[5]];
    let r0_norm = norm3(&r0);
    let v_esc = (2.0 * (1.0 - mu) / r0_norm).sqrt();
    let v_tli = v_esc * 1.01;
    let v_park_norm = norm3(&v_park);
    let v_hat = if v_park_norm < MIN_VELOCITY_NORM {
        [0.0, 1.0, 0.0]
    } else {
        [
            v_park[0] / v_park_norm,
            v_park[1] / v_park_norm,
            v_park[2] / v_park_norm,
        ]
    };
    let r_hat = if r0_norm > MIN_VELOCITY_NORM {
        [r0[0] / r0_norm, r0[1] / r0_norm, r0[2] / r0_norm]
    } else {
        [1.0, 0.0, 0.0]
    };
    let r_target = norm3(&target_state[..3]);
    (r0, v_park, v_hat, r_hat, v_tli, r_target)
}

fn validate_params(params: &WsbSearchParams) {
    assert!(params.n_sun_phase > 0, "n_sun_phase 必须大于 0");
    assert!(params.n_departure_phase > 0, "n_departure_phase 必须大于 0");
    assert!(params.n_tof > 0, "n_tof 必须大于 0");
    assert!(
        params.n_propagation_samples > 0,
        "n_propagation_samples 必须大于 0"
    );
}

fn sort_candidates(candidates: &mut [WsbCandidate]) {
    candidates.sort_by(|left, right| {
        left.total_dv
            .partial_cmp(&right.total_dv)
            .unwrap_or(Ordering::Equal)
    });
}

/// 串行 WSB 网格搜索，结果按现有网格顺序收集后稳定按总 Δv 排序。
#[allow(clippy::too_many_arguments)]
pub fn wsb_search_serial(
    departure_state: &[f64; 6],
    target_state: &[f64; 6],
    mu: f64,
    mu_sun: f64,
    sun_distance: f64,
    sun_angular_rate: f64,
    params: &WsbSearchParams,
    progress_tx: Option<&crossbeam_channel::Sender<usize>>,
) -> Result<WsbSearchResult, String> {
    validate_params(params);
    let (r0, v_park, v_hat, r_hat, v_tli, r_target) =
        prepare_inputs(departure_state, target_state, mu);
    let sun_phase_grid = linspace_exclusive(
        params.sun_phase_min,
        params.sun_phase_max,
        params.n_sun_phase,
    );
    let tof_grid = linspace_inclusive(params.tof_min_sec, params.tof_max_sec, params.n_tof);

    let context = WsbTaskContext {
        target_state,
        mu,
        mu_sun,
        sun_distance,
        sun_angular_rate,
        params,
        r0: &r0,
        v_park: &v_park,
        v_hat: &v_hat,
        r_hat: &r_hat,
        v_tli,
        r_target,
    };
    let mut candidates = Vec::new();
    let mut n_propagation_failures = 0;
    for sun_phase0 in sun_phase_grid {
        for &tof_sec in &tof_grid {
            let (mut task_candidates, task_failures) =
                evaluate_task(&context, sun_phase0, tof_sec)?;
            candidates.append(&mut task_candidates);
            n_propagation_failures += task_failures;
            if let Some(tx) = progress_tx {
                let _ = tx.send(1);
            }
        }
    }
    sort_candidates(&mut candidates);
    Ok(WsbSearchResult {
        candidates,
        n_propagation_failures,
    })
}

/// Rayon 并行 WSB 网格搜索；`collect` 保持网格任务的确定性顺序。
#[allow(clippy::too_many_arguments)]
pub fn wsb_search_parallel(
    departure_state: &[f64; 6],
    target_state: &[f64; 6],
    mu: f64,
    mu_sun: f64,
    sun_distance: f64,
    sun_angular_rate: f64,
    params: &WsbSearchParams,
    progress_tx: Option<&crossbeam_channel::Sender<usize>>,
) -> Result<WsbSearchResult, String> {
    use rayon::prelude::*;

    validate_params(params);
    let (r0, v_park, v_hat, r_hat, v_tli, r_target) =
        prepare_inputs(departure_state, target_state, mu);
    let sun_phase_grid = linspace_exclusive(
        params.sun_phase_min,
        params.sun_phase_max,
        params.n_sun_phase,
    );
    let tof_grid = linspace_inclusive(params.tof_min_sec, params.tof_max_sec, params.n_tof);
    let tasks: Vec<_> = sun_phase_grid
        .into_iter()
        .flat_map(|sun_phase0| tof_grid.iter().map(move |&tof_sec| (sun_phase0, tof_sec)))
        .collect();
    let context = WsbTaskContext {
        target_state,
        mu,
        mu_sun,
        sun_distance,
        sun_angular_rate,
        params,
        r0: &r0,
        v_park: &v_park,
        v_hat: &v_hat,
        r_hat: &r_hat,
        v_tli,
        r_target,
    };

    let task_results: Result<Vec<_>, String> = tasks
        .into_par_iter()
        .map(|(sun_phase0, tof_sec)| {
            let result = evaluate_task(&context, sun_phase0, tof_sec);
            if let Some(tx) = progress_tx {
                let _ = tx.send(1);
            }
            result
        })
        .collect();

    let mut candidates = Vec::new();
    let mut n_propagation_failures = 0;
    for (mut task_candidates, task_failures) in task_results? {
        candidates.append(&mut task_candidates);
        n_propagation_failures += task_failures;
    }
    sort_candidates(&mut candidates);
    Ok(WsbSearchResult {
        candidates,
        n_propagation_failures,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_helpers_match_numpy_linspace_shapes() {
        assert_eq!(linspace_exclusive(0.0, 2.0, 2), vec![0.0, 1.0]);
        assert_eq!(linspace_inclusive(0.0, 2.0, 3), vec![0.0, 1.0, 2.0]);
        assert_eq!(linspace_inclusive(4.0, 8.0, 1), vec![4.0]);
    }

    #[test]
    fn periapsis_interpolation_returns_first_crossing() {
        let y0 = [1.0, 0.0, 0.0, -1.0, 0.0, 0.0];
        let y1 = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let result = first_periapsis_crossing(&[0.0, 1.0], &[y0, y1], 0.1);
        let (time, state, index) = result.expect("应找到近拱点穿越");
        assert_eq!(index, 0);
        assert_eq!(time, 0.5);
        assert_eq!(state[3], 0.0);
    }
}
