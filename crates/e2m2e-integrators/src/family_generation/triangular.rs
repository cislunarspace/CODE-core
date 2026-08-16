//! SPO、LPO 与 Horseshoe 的单次 Rust 族生成。

use crate::planar_pal::{run_planar_pal, Options};

use super::common::{correct_planar_fixed_x, metric_minmax, triangular_seed, Failure};
use super::types::{Context, Member, Outcome};

const STEP_BUDGET: usize = 400;

#[allow(clippy::too_many_arguments)]
pub(crate) fn generate(
    context: Context,
    family_type: &'static str,
    point: u8,
    min_amplitude_km: f64,
    max_amplitude_km: f64,
    member_limit: usize,
    direction: &'static str,
    match_tolerance_km: f64,
) -> Result<Outcome, String> {
    if !matches!(family_type, "spo" | "lpo" | "horseshoe") {
        return Err(format!("未知三角轨道族 {family_type}"));
    }
    if point != 4 && point != 5 {
        return Err("SPO/LPO/Horseshoe 平动点必须为 L4 或 L5".to_string());
    }
    if member_limit == 0 {
        return Err("n_orbits 必须大于 0".to_string());
    }
    if min_amplitude_km <= 0.0 || min_amplitude_km >= max_amplitude_km {
        return Err("振幅范围必须满足 0 < min_amplitude_km < max_amplitude_km".to_string());
    }
    if !matches!(direction, "increase-x0" | "decrease-x0") {
        return Err("continuation_direction 必须为 increase-x0 或 decrease-x0".to_string());
    }
    if match_tolerance_km <= 0.0 {
        return Err("match_tolerance_km 必须为正数".to_string());
    }

    let short_period = family_type == "spo";
    let (initial_state, initial_period) =
        match triangular_seed(context, point, short_period, 1000.0) {
            Ok(seed) => seed,
            Err(failure) => {
                return Ok(failure_outcome(
                    family_type,
                    member_limit,
                    Vec::new(),
                    failure,
                ));
            }
        };
    let seed = match correct_planar_fixed_x(context, initial_state, initial_period) {
        Ok(seed) => seed,
        Err(failure) => {
            return Ok(failure_outcome(
                family_type,
                member_limit,
                Vec::new(),
                failure,
            ));
        }
    };
    let family_step_amplitude = if short_period { 2000.0 } else { 3000.0 };
    let estimated_steps =
        ((max_amplitude_km + match_tolerance_km) / family_step_amplitude).ceil() as usize + 8;
    let walk_steps = STEP_BUDGET.min((2 * member_limit).max(estimated_steps).max(10));
    let pal = run_planar_pal(
        context.mu,
        seed.state,
        seed.period,
        walk_steps,
        0.01,
        direction,
        Options {
            tolerance: 1e-9,
            max_iterations: 16,
            rtol: context.rtol,
            atol: context.atol,
            max_step: context.max_step,
        },
    );

    let lower_bound = min_amplitude_km;
    let upper_bound = max_amplitude_km;
    let mut covered_upper = false;
    let mut members = Vec::new();
    for index in 0..pal.states.len() {
        let state = pal.states[index];
        let period = pal.periods[index];
        let closure_error = pal.closure_errors[index];
        let jacobi_drift = pal.jacobi_drifts[index];
        let (minimum, maximum) =
            match metric_minmax(context, state, period, "l45-distance", point, 400) {
                Ok(range) => range,
                Err(failure) => {
                    return Ok(failure_outcome(family_type, member_limit, members, failure));
                }
            };
        let amplitude_km = 0.5 * (minimum + maximum) * context.characteristic_length_km;
        covered_upper |= amplitude_km >= max_amplitude_km - match_tolerance_km;
        if lower_bound <= amplitude_km
            && amplitude_km <= upper_bound
            && members.len() < member_limit
        {
            let mut member = Member::periodic(state, period, closure_error, Some(jacobi_drift));
            member.amplitude_km = Some(amplitude_km);
            member.newton_iterations = pal.newton_iterations.get(index).copied();
            member.tangent_system_rank = pal.tangent_system_ranks.get(index).copied();
            member.tangent_system_condition = pal.tangent_system_conditions.get(index).copied();
            member.augmented_system_rank = pal.augmented_system_ranks.get(index).copied();
            member.augmented_system_condition = pal.augmented_system_conditions.get(index).copied();
            member.step_size = pal.step_sizes.get(index).copied();
            members.push(member);
        }
    }

    if !members.is_empty() && (members.len() >= member_limit || covered_upper) {
        return Ok(Outcome::converged(
            family_type,
            "periodic",
            member_limit,
            members,
        ));
    }
    if pal.status != "converged" {
        let (status, cause) = status_contract(&pal.status, &pal.cause);
        return Ok(Outcome::soft_failure(
            family_type,
            "periodic",
            member_limit,
            members,
            status,
            cause,
            format!("PAL 链提前终止：{}", pal.message),
        ));
    }
    Ok(Outcome::soft_failure(
        family_type,
        "periodic",
        member_limit,
        members,
        "stagnated",
        "stagnation_detected",
        format!("单次 PAL 的 {walk_steps} 步预算未覆盖完整请求范围"),
    ))
}

fn status_contract(status: &str, cause: &str) -> (&'static str, &'static str) {
    let normalized_status = match status {
        "converged" => "converged",
        "max_iterations" => "max_iterations",
        "stagnated" => "stagnated",
        "diverged" => "diverged",
        "infeasible" => "infeasible",
        _ => "failed",
    };
    let normalized_cause = match cause {
        "none" => "none",
        "max_iterations_reached" => "max_iterations_reached",
        "stagnation_detected" => "stagnation_detected",
        "divergence_detected" => "divergence_detected",
        "integration_failed" => "integration_failed",
        "singular_jacobian" => "singular_jacobian",
        "invalid_period" => "invalid_period",
        _ => "unknown",
    };
    (normalized_status, normalized_cause)
}

fn failure_outcome(
    family_type: &'static str,
    member_limit: usize,
    members: Vec<Member>,
    failure: Failure,
) -> Outcome {
    Outcome::soft_failure(
        family_type,
        "periodic",
        member_limit,
        members,
        failure.status,
        failure.cause,
        failure.message,
    )
}
