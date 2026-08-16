//! Axial Type B 族的单次 Rust 生成。

use super::common::{correct_axial_fixed_vz, metric_minmax};
use super::types::{Context, Member, Outcome, PeriodicOrbit};

// DE421 地月模型下的垂直临界 Lyapunov 轨道。它们是数值算法标定种子，
// 进入族前仍由 Rust 微分修正收紧，不作为最终成员直接返回。
const L1_BIFURCATION_STATE: [f64; 6] = [0.78157319, 0.0, 0.0, 0.0, 0.44320030, 0.0];
const L1_BIFURCATION_PERIOD: f64 = 3.9500220877036183;
const L2_BIFURCATION_STATE: [f64; 6] = [1.02950000, 0.0, 0.0, 0.0, 0.72543427, 0.0];
const L2_BIFURCATION_PERIOD: f64 = 4.310490051967385;
const DE421_EARTH_MOON_MU: f64 = 0.012_150_585_350_562_453;

pub(crate) fn generate(
    context: Context,
    point: u8,
    max_amplitude_km: f64,
    member_limit: usize,
) -> Result<Outcome, String> {
    if point != 1 && point != 2 {
        return Err("Axial 平动点必须为 L1 或 L2".to_string());
    }
    if (context.mu - DE421_EARTH_MOON_MU).abs() > 1e-12 {
        return Err("Axial 族当前只支持 DE421 地月 CR3BP 标定上下文".to_string());
    }
    if member_limit == 0 {
        return Err("n_orbits 必须大于 0".to_string());
    }
    if max_amplitude_km == 0.0 || !max_amplitude_km.is_finite() {
        return Err("max_amplitude_km 必须为有限非零数".to_string());
    }
    let (state, period) = if point == 1 {
        (L1_BIFURCATION_STATE, L1_BIFURCATION_PERIOD)
    } else {
        (L2_BIFURCATION_STATE, L2_BIFURCATION_PERIOD)
    };
    let bifurcation = PeriodicOrbit {
        state,
        period,
        closure_error: 0.0,
    };
    let sign = max_amplitude_km.signum();
    let target_du = max_amplitude_km.abs() / context.characteristic_length_km;
    let mut vz0 = sign * 0.001;
    let mut step = sign * 0.005;
    let mut previous = bifurcation;
    let mut members = Vec::new();

    while members.len() < member_limit {
        let orbit = match correct_axial_fixed_vz(context, vz0, &previous) {
            Ok(orbit) => orbit,
            Err(failure) if members.is_empty() => {
                return Ok(Outcome::soft_failure(
                    "axial",
                    "periodic",
                    member_limit,
                    members,
                    failure.status,
                    failure.cause,
                    failure.message,
                ));
            }
            Err(_) => {
                step *= 0.5;
                if step.abs() < 1e-4 {
                    return Ok(Outcome::soft_failure(
                        "axial",
                        "periodic",
                        member_limit,
                        members,
                        "stagnated",
                        "stagnation_detected",
                        "Axial 固定 vz0 行走步长已降至下限",
                    ));
                }
                vz0 = previous.state[5] + step;
                continue;
            }
        };
        let (_, amplitude_du) =
            match metric_minmax(context, orbit.state, orbit.period, "z-amplitude", 0, 300) {
                Ok(range) => range,
                Err(failure) => {
                    return Ok(Outcome::soft_failure(
                        "axial",
                        "periodic",
                        member_limit,
                        members,
                        failure.status,
                        failure.cause,
                        failure.message,
                    ));
                }
            };
        if amplitude_du > target_du {
            break;
        }
        let mut member = Member::periodic(orbit.state, orbit.period, orbit.closure_error, None);
        member.amplitude_km = Some(amplitude_du * context.characteristic_length_km);
        members.push(member);
        previous = orbit;
        vz0 = previous.state[5] + step;
    }
    if members.is_empty() {
        return Ok(Outcome::soft_failure(
            "axial",
            "periodic",
            member_limit,
            members,
            "infeasible",
            "constraint_violation",
            format!(
                "Axial(L{point}) 未生成振幅不超过 {:.0} km 的成员",
                max_amplitude_km.abs()
            ),
        ));
    }
    Ok(Outcome::converged(
        "axial",
        "periodic",
        member_limit,
        members,
    ))
}
