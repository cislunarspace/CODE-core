//! 八类轨道族（Axial/DRO/Halo/NRHO/RO/SPO/LPO/Horseshoe）共享的传播、修正、
//! 模态和度量实现。

use crate::differential_correction::{run_correction, CorrectionRawResult};
use crate::family::collinear_center_modes;
use e2m2e_forces::cr3bp::propagate_cr3bp;

use super::types::{Context, PeriodicOrbit};

const SAMPLE_COUNT: usize = 1000;

/// DRO 族标准种子（与 Python ``data/templates/seed.py`` 的 ``_DRO_SEED_*``
/// 同源）：近侧 x 轴穿越点 x0、vy0 与周期，振幅约 90,786 km。
pub(crate) const SEED_DRO_X0: f64 = 0.791_885_566_197_42;
pub(crate) const SEED_DRO_VY0: f64 = 0.536_819_842_572_739;
pub(crate) const SEED_DRO_PERIOD: f64 = 3.472_535_773_770_595;

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
    let mut final_state = state;
    let mut closure = closure_error(context, final_state, period).map_err(|message| Failure {
        status: "failed",
        cause: "integration_failed",
        message,
    })?;
    // 与 Python `_create_corrected_orbit` 同款 vy 半步微调（仅在更好时
    // 采用）：高偏心 RO 成员的闭合误差由积分噪声主导，该抛光可压到
    // 1e-9 量级；否则 3:1 等成员的原始闭合约 1.9e-6，越过公共契约 1e-6。
    if closure > 1e-10 {
        if let Ok(diff) = closure_diff(context, final_state, period) {
            if diff[..3].iter().any(|value| value.abs() > 1e-14)
                && diff[3..].iter().any(|value| value.abs() > 1e-14)
            {
                let mut polished = final_state;
                polished[4] -= 0.5 * diff[4];
                if let Ok(polished_closure) = closure_error(context, polished, period) {
                    if polished_closure < closure {
                        final_state = polished;
                        closure = polished_closure;
                    }
                }
            }
        }
    }
    Ok(PeriodicOrbit {
        state: final_state,
        period,
        closure_error: closure,
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

pub(crate) fn correct_dro_fixed_x(
    context: Context,
    x0: f64,
    guess: Option<&PeriodicOrbit>,
) -> Result<PeriodicOrbit, Failure> {
    if x0 >= 1.0 - context.mu {
        // 穿越点越过月球（对侧），已不在近侧 DRO 族参数域内
        return Err(invalid_failure(format!(
            "DRO x0={x0:.6} 越过月心位置，超出族参数域"
        )));
    }
    let (mut state, period) = match guess {
        Some(orbit) => (orbit.state, orbit.period),
        None => ([x0, 0.0, 0.0, 0.0, SEED_DRO_VY0, 0.0], SEED_DRO_PERIOD),
    };
    state[0] = x0;
    state[1] = 0.0;
    state[3] = 0.0;
    state[5] = 0.0;
    let orbit = correct(
        context,
        state,
        period / 2.0,
        &[1, 3],
        &[4, 6],
        false,
        false,
        1e-12,
        50,
    )?;
    if orbit.period > 1.2 * period {
        // 大步长行走时修正器会跳到长周期伪解（多圈对称周期轨道），
        // 周期相对初猜显著变长即判为伪解，交由族行走退半步重试
        return Err(Failure {
            status: "diverged",
            cause: "divergence_detected",
            message: format!("DRO(x0={x0:.6}) 修正跳到长周期伪解"),
        });
    }
    Ok(orbit)
}

/// DRO 振幅（km）：一个周期内距月心距离最小/最大值的均值，与 Python
/// ``design_dro`` 同一定义；4000 点采样保证与单轨入口的测量一致。
pub(crate) fn dro_amplitude_km(context: Context, orbit: &PeriodicOrbit) -> Result<f64, Failure> {
    let (minimum, maximum) =
        metric_minmax(context, orbit.state, orbit.period, "moon-distance", 0, 4000)?;
    Ok(0.5 * (minimum + maximum) * context.characteristic_length_km)
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

/// 支持生成的共振比 (p, q)：顺行内共振五档（与 Python 侧
/// ``RO_SUPPORTED_RESONANCES`` 同源）；其余比值的初猜不可靠。
pub(crate) const RO_SUPPORTED_RESONANCES: &[(u32, u32)] = &[(2, 1), (3, 1), (3, 2), (4, 1), (4, 3)];

/// 恒星 p:q 共振的会合系闭合周期：q 个恒星月（``T = 2πq``）。
fn ro_period(_resonance_p: u32, resonance_q: u32) -> f64 {
    std::f64::consts::TAU * resonance_q as f64
}

fn ro_kepler_a(mu: f64, resonance_p: u32, resonance_q: u32) -> f64 {
    ((1.0 - mu) * (resonance_q as f64 / resonance_p as f64).powi(2)).cbrt()
}

fn ro_eccentric_seed(
    context: Context,
    resonance_p: u32,
    resonance_q: u32,
    eccentricity: f64,
    apoapsis: bool,
) -> ([f64; 6], f64, f64) {
    let a = ro_kepler_a(context.mu, resonance_p, resonance_q);
    let radius = a * if apoapsis {
        1.0 + eccentricity
    } else {
        1.0 - eccentricity
    };
    let x0 = radius - context.mu;
    let inertial_speed = ((1.0 - context.mu) * (2.0 / radius - 1.0 / a)).sqrt();
    let vy0 = if apoapsis {
        x0 - inertial_speed
    } else {
        inertial_speed - x0
    };
    (
        [x0, 0.0, 0.0, 0.0, vy0, 0.0],
        ro_period(resonance_p, resonance_q),
        a,
    )
}

/// 一次传播测出 RO 形状特征：净卷绕数（绕质心，单位 2π）与地心距包络。
fn ro_shape(context: Context, orbit: &PeriodicOrbit) -> Result<(f64, f64, f64), Failure> {
    let count = 2000usize;
    let dt = orbit.period / (count - 1) as f64;
    let mut times: Vec<f64> = (0..count).map(|index| index as f64 * dt).collect();
    times[count - 1] = orbit.period;
    let propagation = propagate_cr3bp(
        context.mu,
        (0.0, orbit.period),
        &times,
        &orbit.state,
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
    let mut r_min = f64::INFINITY;
    let mut r_max = 0.0_f64;
    let mut winding = 0.0_f64;
    let mut previous_angle: Option<f64> = None;
    for sample in &propagation.states {
        let dx = sample[0] + context.mu;
        let radius = (dx * dx + sample[1] * sample[1] + sample[2] * sample[2]).sqrt();
        r_min = r_min.min(radius);
        r_max = r_max.max(radius);
        let angle = sample[1].atan2(sample[0]);
        if let Some(previous) = previous_angle {
            let mut delta = angle - previous;
            while delta > std::f64::consts::PI {
                delta -= std::f64::consts::TAU;
            }
            while delta < -std::f64::consts::PI {
                delta += std::f64::consts::TAU;
            }
            winding += delta;
        }
        previous_angle = Some(angle);
    }
    Ok((winding / std::f64::consts::TAU, r_min, r_max))
}

/// RO 伪支守卫：Kepler 半长轴量级、偏心形态与卷绕数。闭合契约只在
/// 最终成员处验收（陡峭支的中间试探含不可抛光的积分噪声，逐点验收
/// 会把合法族成员拒之门外；与 Python `_guard_ro_branch` 同口径）。
fn check_ro_member(
    context: Context,
    resonance_p: u32,
    resonance_q: u32,
    orbit: PeriodicOrbit,
    a_kepler: f64,
) -> Result<PeriodicOrbit, Failure> {
    let (winding, r_min, r_max) = ro_shape(context, &orbit)?;
    let amplitude = 0.5 * (r_min + r_max);
    let relative_error = (amplitude - a_kepler).abs() / a_kepler;
    if relative_error > 0.1 {
        return Err(Failure {
            status: "failed",
            cause: "constraint_violation",
            message: format!(
                "RO({resonance_p}:{resonance_q}) 偏离 Kepler 半长轴过大：均值={amplitude:.6} DU、a={a_kepler:.6} DU、相对偏差={relative_error:.2}%"
            ),
        });
    }
    if r_max / r_min < 1.2 {
        return Err(Failure {
            status: "failed",
            cause: "constraint_violation",
            message: format!(
                "RO({resonance_p}:{resonance_q}) 跳到近圆伪支：r_max/r_min={:.3} < 1.2",
                r_max / r_min
            ),
        });
    }
    let target_winding = (resonance_p - resonance_q) as f64;
    if (winding - target_winding).abs() > 0.01 {
        return Err(Failure {
            status: "failed",
            cause: "constraint_violation",
            message: format!(
                "RO({resonance_p}:{resonance_q}) 卷绕数 {winding:+.4} 不等于目标 {target_winding:+}"
            ),
        });
    }
    Ok(orbit)
}

/// 修正族参数 x_try 处的 RO 成员（族行走与割线钉定共用）。
/// 3:1/4:1 的 x0–vy0 映射在目标邻域很陡，邻点续猜会跳支；这两档按
/// x_try 重建近心点 Kepler 种子（逐点收敛已实证），其余三档沿用邻点
/// 续猜。
pub(crate) fn correct_ro_trial(
    context: Context,
    resonance_p: u32,
    resonance_q: u32,
    x_try: f64,
    guess: &PeriodicOrbit,
) -> Result<PeriodicOrbit, Failure> {
    if !matches!((resonance_p, resonance_q), (3, 1) | (4, 1)) {
        return correct_ro_fixed_x(context, x_try, guess);
    }
    let a_kepler = ro_kepler_a(context.mu, resonance_p, resonance_q);
    let eccentricity = 1.0 - (x_try + context.mu) / a_kepler;
    if !(0.0..1.0).contains(&eccentricity) {
        return Err(invalid_failure(format!(
            "RO({resonance_p}:{resonance_q}) 试探点超出偏心种子域：x={x_try:.6}"
        )));
    }
    let (state, period, _) =
        ro_eccentric_seed(context, resonance_p, resonance_q, eccentricity, false);
    let fresh = PeriodicOrbit {
        state,
        period,
        closure_error: f64::INFINITY,
    };
    correct_ro_fixed_x(context, x_try, &fresh)
}

fn pin_ro_period_secant(
    context: Context,
    resonance_p: u32,
    resonance_q: u32,
    seed: PeriodicOrbit,
    a_kepler: f64,
    dx: f64,
) -> Result<PeriodicOrbit, Failure> {
    let target = ro_period(resonance_p, resonance_q);
    let tolerance = 1e-9 * target;
    let mut left = check_ro_member(context, resonance_p, resonance_q, seed, a_kepler)?;
    let mut x_left = left.state[0];
    let mut right = check_ro_member(
        context,
        resonance_p,
        resonance_q,
        correct_ro_trial(context, resonance_p, resonance_q, x_left + dx, &left)?,
        a_kepler,
    )?;
    let mut x_right = x_left + dx;
    for _ in 0..8 {
        let denominator = right.period - left.period;
        if denominator == 0.0 {
            break;
        }
        let x_try = x_left + (target - left.period) / denominator * (x_right - x_left);
        let guess = if (x_try - x_left).abs() <= (x_try - x_right).abs() {
            &left
        } else {
            &right
        };
        let candidate = check_ro_member(
            context,
            resonance_p,
            resonance_q,
            correct_ro_trial(context, resonance_p, resonance_q, x_try, guess)?,
            a_kepler,
        )?;
        if (candidate.period - target).abs() <= tolerance {
            return Ok(candidate);
        }
        if (candidate.period - target) * (left.period - target) <= 0.0 {
            right = candidate;
            x_right = x_try;
        } else {
            left = candidate;
            x_left = x_try;
        }
    }
    Err(invalid_failure(format!(
        "RO({resonance_p}:{resonance_q}) 割线周期钉定未命中目标"
    )))
}

fn correct_ro_fast_21_seed(
    context: Context,
    resonance_p: u32,
    resonance_q: u32,
) -> Result<PeriodicOrbit, Failure> {
    let (mut state, period, a_kepler) = ro_eccentric_seed(context, 2, 1, 0.75, false);
    state[0] += 0.0008;
    state[4] -= 0.025;
    let initial = PeriodicOrbit {
        state,
        period,
        closure_error: f64::INFINITY,
    };
    let seed = check_ro_member(
        context,
        resonance_p,
        resonance_q,
        correct_ro_fixed_x(context, state[0], &initial)?,
        a_kepler,
    )?;
    pin_ro_period_secant(context, resonance_p, resonance_q, seed, a_kepler, 0.0002)
}

fn correct_ro_eccentric_seed(
    context: Context,
    resonance_p: u32,
    resonance_q: u32,
    eccentricity: f64,
    apoapsis: bool,
) -> Result<PeriodicOrbit, Failure> {
    let (state, period, a_kepler) =
        ro_eccentric_seed(context, resonance_p, resonance_q, eccentricity, apoapsis);
    let initial = PeriodicOrbit {
        state,
        period,
        closure_error: f64::INFINITY,
    };
    let upper = correct_ro_fixed_x(context, state[0], &initial)?;
    walk_ro_period(context, resonance_p, resonance_q, upper, a_kepler)
}

fn walk_ro_period(
    context: Context,
    resonance_p: u32,
    resonance_q: u32,
    seed: PeriodicOrbit,
    a_kepler: f64,
) -> Result<PeriodicOrbit, Failure> {
    let target = ro_period(resonance_p, resonance_q);
    let tolerance = 1e-9 * target;
    let current = check_ro_member(context, resonance_p, resonance_q, seed, a_kepler)?;
    let p = current.state[0];
    let measure = current.period;
    if (measure - target).abs() <= tolerance {
        return Ok(current);
    }

    let dp_init = 0.0025;
    let probe = correct_ro_fixed_x(context, p + dp_init, &current)
        .and_then(|orbit| check_ro_member(context, resonance_p, resonance_q, orbit, a_kepler))?;
    let slope = (probe.period - measure) / dp_init;
    if slope == 0.0 {
        return Err(invalid_failure(
            "RO 族行走停滞：周期不随参数变化".to_string(),
        ));
    }
    let direction = if (target - measure) / slope > 0.0 {
        1.0
    } else {
        -1.0
    };
    let mut step = dp_init;
    let mut p_prev = p;
    let mut m_prev = measure;
    let mut orbit_prev = current;
    let mut bracket: Option<(f64, f64, PeriodicOrbit, PeriodicOrbit)> = None;
    for _ in 0..600 {
        let p_try = p_prev + direction * step;
        let orbit_new = match correct_ro_fixed_x(context, p_try, &orbit_prev)
            .and_then(|orbit| check_ro_member(context, resonance_p, resonance_q, orbit, a_kepler))
        {
            Ok(orbit) => orbit,
            Err(_) => {
                step *= 0.5;
                if step < 1e-5 {
                    return Err(invalid_failure(
                        "RO 族行走步长已减至最小仍未跨过目标周期".to_string(),
                    ));
                }
                continue;
            }
        };
        let m_new = orbit_new.period;
        if (m_new - target).abs() <= tolerance {
            return Ok(orbit_new);
        }
        if (m_new - target) * (m_prev - target) <= 0.0 {
            bracket = Some((p_prev, p_try, orbit_prev, orbit_new));
            break;
        }
        p_prev = p_try;
        m_prev = m_new;
        orbit_prev = orbit_new;
    }
    let (mut p_lo, mut p_hi, mut orbit_lo, mut orbit_hi) = bracket
        .ok_or_else(|| invalid_failure(format!("RO 族行走未在预算内跨过目标周期 {target:.12}")))?;
    let mut m_lo = orbit_lo.period;
    for _ in 0..600 {
        let p_mid = 0.5 * (p_lo + p_hi);
        let orbit_mid = correct_ro_fixed_x(
            context,
            p_mid,
            if (p_mid - p_lo).abs() <= (p_mid - p_hi).abs() {
                &orbit_lo
            } else {
                &orbit_hi
            },
        )
        .and_then(|orbit| check_ro_member(context, resonance_p, resonance_q, orbit, a_kepler))?;
        let m_mid = orbit_mid.period;
        if (m_mid - target).abs() <= tolerance {
            return Ok(orbit_mid);
        }
        if (m_mid - target) * (m_lo - target) > 0.0 {
            p_lo = p_mid;
            m_lo = m_mid;
            orbit_lo = orbit_mid;
        } else {
            p_hi = p_mid;
            orbit_hi = orbit_mid;
        }
    }
    Err(invalid_failure(format!(
        "RO 族行走二分未命中目标周期 {target:.12}"
    )))
}

/// 3:1/4:1 的偏心 Kepler 种子入口：偏心率取值使近心点 x0 落在目录
/// 精确成员穿越点邻域（3:1 x0≈0.021、4:1 x0≈0.280），固定 x0 修正
/// 直接落在目标偏心支，再割线钉定 ``T = 2πq``。
fn correct_ro_plain_seed(
    context: Context,
    resonance_p: u32,
    resonance_q: u32,
    eccentricity: f64,
) -> Result<PeriodicOrbit, Failure> {
    let (state, period, a_kepler) =
        ro_eccentric_seed(context, resonance_p, resonance_q, eccentricity, false);
    let initial = PeriodicOrbit {
        state,
        period,
        closure_error: f64::INFINITY,
    };
    let seed = check_ro_member(
        context,
        resonance_p,
        resonance_q,
        correct_ro_fixed_x(context, state[0], &initial)?,
        a_kepler,
    )?;
    pin_ro_period_secant(context, resonance_p, resonance_q, seed, a_kepler, -0.0025)
}

/// RO 精确共振种子：p:q 为航天器惯性圈数:月球圈数，闭合条件为 q 个
/// 恒星月内绕地 p 圈（``T = 2πq``、净卷绕 ``w = p−q``）。五档均从
/// 偏心族入口上族并把周期钉到目标值；2:1 的入口在数值折返区需局部
/// 扫描，3:2/4:3 沿固定 x0 行走加周期二分。
pub(crate) fn correct_ro_seed(
    context: Context,
    resonance_p: u32,
    resonance_q: u32,
) -> Result<PeriodicOrbit, Failure> {
    if !RO_SUPPORTED_RESONANCES.contains(&(resonance_p, resonance_q)) {
        return Err(invalid_failure(format!(
            "不支持的共振比 {resonance_p}:{resonance_q}（RO 支持 2:1/3:1/3:2/4:1/4:3）"
        )));
    }
    let orbit = match (resonance_p, resonance_q) {
        (3, 1) => correct_ro_plain_seed(context, resonance_p, resonance_q, 0.93)?,
        (4, 1) => correct_ro_plain_seed(context, resonance_p, resonance_q, 0.26)?,
        (2, 1) => correct_ro_fast_21_seed(context, resonance_p, resonance_q)?,
        (3, 2) => correct_ro_eccentric_seed(context, resonance_p, resonance_q, 0.5, false)?,
        (4, 3) => correct_ro_eccentric_seed(context, resonance_p, resonance_q, 0.6, true)?,
        _ => unreachable!(),
    };
    let member = check_ro_member(
        context,
        resonance_p,
        resonance_q,
        orbit,
        ro_kepler_a(context.mu, resonance_p, resonance_q),
    )?;
    // 最终成员验收闭合契约（中间试探不设此门，见 check_ro_member）。
    if member.closure_error >= 1e-6 {
        return Err(Failure {
            status: "failed",
            cause: "closure_error",
            message: format!(
                "RO({resonance_p}:{resonance_q}) 精确成员闭合误差超限：{:.3e}",
                member.closure_error
            ),
        });
    }
    Ok(member)
}

/// RO 族成员修正：固定 +x 穿越点 x0，自由 vy0 与半周期。周期跳变超
/// ±20% 判为异周期伪解，交由族行走退半步重试（RO 族周期随 x0 双向
/// 变化，双侧判定，与 DRO 单侧不同）。
pub(crate) fn correct_ro_fixed_x(
    context: Context,
    x0: f64,
    guess: &PeriodicOrbit,
) -> Result<PeriodicOrbit, Failure> {
    let mut state = guess.state;
    state[0] = x0;
    state[1] = 0.0;
    state[3] = 0.0;
    state[5] = 0.0;
    let orbit = correct(
        context,
        state,
        guess.period / 2.0,
        &[1, 3],
        &[4, 6],
        false,
        false,
        1e-12,
        50,
    )?;
    if (orbit.period - guess.period).abs() > 0.2 * guess.period {
        return Err(Failure {
            status: "diverged",
            cause: "divergence_detected",
            message: format!("RO(x0={x0:.6}) 修正跳到异周期伪解"),
        });
    }
    Ok(orbit)
}

/// RO 振幅（km）：一个周期内距地心距离最小/最大值的均值，与 Python
/// ``design_ro`` 同一定义；4000 点采样保证与单轨入口的测量一致。
pub(crate) fn ro_amplitude_km(context: Context, orbit: &PeriodicOrbit) -> Result<f64, Failure> {
    let (minimum, maximum) = metric_minmax(
        context,
        orbit.state,
        orbit.period,
        "earth-distance",
        0,
        4000,
    )?;
    Ok(0.5 * (minimum + maximum) * context.characteristic_length_km)
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

pub(crate) fn closure_diff(
    context: Context,
    state: [f64; 6],
    period: f64,
) -> Result<[f64; 6], String> {
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
    let mut diff = [0.0; 6];
    for index in 0..6 {
        diff[index] = final_state[index] - state[index];
    }
    Ok(diff)
}

pub(crate) fn closure_error(context: Context, state: [f64; 6], period: f64) -> Result<f64, String> {
    let diff = closure_diff(context, state, period)?;
    Ok(diff.iter().map(|value| value.abs()).fold(0.0, f64::max))
}

/// CR3BP Jacobi 常数（Parker 约定，不含 ½μ(1−μ) 常数项）：C = 2U − v²。
/// 与 Python 侧 ``CR3BPSystem.get_jacobi_constant`` 同一定义，保证能量
/// 窗口筛选与库内 jacobi 包络逐位一致。
pub(crate) fn jacobi_constant(mu: f64, state: [f64; 6]) -> f64 {
    let (x, y, z, vx, vy, vz) = (state[0], state[1], state[2], state[3], state[4], state[5]);
    let r1 = ((x + mu).powi(2) + y * y + z * z).sqrt();
    let r2 = ((x - 1.0 + mu).powi(2) + y * y + z * z).sqrt();
    let potential = (x * x + y * y) / 2.0 + (1.0 - mu) / r1 + mu / r2;
    2.0 * potential - (vx * vx + vy * vy + vz * vz)
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
            "earth-distance" => {
                let dx = sample[0] + context.mu;
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
            "y-amplitude" => sample[1].abs(),
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
