//! NRHO 族的单次 Rust 生成。

use e2m2e_forces::pal_continuation::{f_df_tangent, pal_newton_step};

use super::common::{
    closure_error, correct_halo_fixed_x, correct_halo_fixed_z, metric_minmax, Failure,
};
use super::types::{Context, Member, Outcome, PeriodicOrbit};

// 基准种子是 **北族**（vy<0 穿越点 z0>0，ADR 0042 几何约定）的折叠后
// L2 Halo 成员；`state[2] *= z_sign` 在此语义下成立——z_sign=+1 保持
// 北族，−1 镜像到南族。种子曾以南族入库导致南北双向反号（issue #586）。
const L2_SEED_STATE: [f64; 6] = [
    1.128103754424342,
    0.0,
    0.17883236940616654,
    0.0,
    -0.22553424464298827,
    0.0,
];
const L2_SEED_PERIOD: f64 = 2.9899387540796956;
const DE421_EARTH_MOON_MU: f64 = 0.012_150_585_350_562_453;

pub(crate) fn generate(
    context: Context,
    point: u8,
    north_south: u8,
    perilune_height_max_km: f64,
    member_limit: usize,
) -> Result<Outcome, String> {
    if point != 1 && point != 2 {
        return Err("NRHO 平动点必须为 L1 或 L2".to_string());
    }
    if (context.mu - DE421_EARTH_MOON_MU).abs() > 1e-12 {
        return Err("NRHO 族当前只支持 DE421 地月 CR3BP 标定上下文".to_string());
    }
    if north_south != 1 && north_south != 2 {
        return Err("NRHO north_south 必须为 1 或 2".to_string());
    }
    if perilune_height_max_km <= 0.0 || member_limit == 0 {
        return Err("NRHO 高度必须为正数且 n_orbits 必须大于 0".to_string());
    }
    let z_sign = if north_south == 1 { 1.0 } else { -1.0 };
    let threshold =
        (perilune_height_max_km + context.secondary_radius_km) / context.characteristic_length_km;
    if point == 1 {
        return generate_l1_pal(context, z_sign, threshold, member_limit);
    }

    let mut state = L2_SEED_STATE;
    state[2] *= z_sign;
    let period = L2_SEED_PERIOD;
    let initial_closure = match closure_error(context, state, period) {
        Ok(error) => error,
        Err(message) => {
            return Ok(Outcome::soft_failure(
                "nrho",
                "periodic",
                member_limit,
                Vec::new(),
                "failed",
                "integration_failed",
                message,
            ));
        }
    };
    let mut previous = PeriodicOrbit {
        state,
        period,
        closure_error: initial_closure,
    };
    let mut members = Vec::new();
    let mut dmin = match perilune(context, &previous) {
        Ok(distance) => distance,
        Err(failure) => {
            return Ok(Outcome::soft_failure(
                "nrho",
                "periodic",
                member_limit,
                members,
                failure.status,
                failure.cause,
                failure.message,
            ));
        }
    };
    if dmin <= threshold {
        members.push(member(context, previous, dmin));
    }

    let mut attempts = 0usize;
    while members.len() < member_limit && attempts < 600 {
        attempts += 1;
        let next_x = previous.state[0] - 0.005;
        let candidate = correct_halo_fixed_x(context, next_x, &previous);
        let orbit = match candidate {
            Ok(orbit) => orbit,
            Err(failure) => {
                return Ok(Outcome::soft_failure(
                    "nrho",
                    "periodic",
                    member_limit,
                    members,
                    failure.status,
                    failure.cause,
                    failure.message,
                ));
            }
        };
        dmin = match perilune(context, &orbit) {
            Ok(distance) => distance,
            Err(failure) => {
                return Ok(Outcome::soft_failure(
                    "nrho",
                    "periodic",
                    member_limit,
                    members,
                    failure.status,
                    failure.cause,
                    failure.message,
                ));
            }
        };
        previous = orbit;
        if dmin <= threshold {
            members.push(member(context, previous, dmin));
        }
        // 低于月面附近的高度没有继续行走的物理意义。
        if dmin * context.characteristic_length_km <= context.secondary_radius_km + 500.0 {
            break;
        }
    }
    if members.is_empty() {
        return Ok(Outcome::soft_failure(
            "nrho",
            "periodic",
            member_limit,
            members,
            "infeasible",
            "constraint_violation",
            format!(
                "NRHO(L{point}) 未到达 {:.0} km 近月点高度上限内的成员",
                perilune_height_max_km
            ),
        ));
    }
    Ok(Outcome::converged(
        "nrho",
        "periodic",
        member_limit,
        members,
    ))
}

fn generate_l1_pal(
    context: Context,
    z_sign: f64,
    threshold: f64,
    member_limit: usize,
) -> Result<Outcome, String> {
    let mut current = match correct_halo_fixed_z(context, 0.001, 1, None) {
        Ok(seed) => seed,
        Err(failure) => {
            return Ok(Outcome::soft_failure(
                "nrho",
                "periodic",
                member_limit,
                Vec::new(),
                failure.status,
                failure.cause,
                failure.message,
            ));
        }
    };
    let max_step = context.max_step.unwrap_or(0.01);
    let mut x = [
        current.state[0],
        current.state[2],
        current.state[4],
        current.period / 2.0,
    ];
    let mut tangent = match f_df_tangent(
        context.mu,
        x,
        current.state,
        context.rtol,
        context.atol,
        max_step,
    ) {
        Ok(result) => result.tangent,
        Err(message) => {
            return Ok(Outcome::soft_failure(
                "nrho",
                "periodic",
                member_limit,
                Vec::new(),
                "failed",
                "integration_failed",
                message,
            ));
        }
    };
    if tangent[1] < 0.0 {
        tangent.iter_mut().for_each(|value| *value = -*value);
    }

    let mut ds = 0.0045;
    let mut members = Vec::new();
    for _ in 0..600 {
        let x_start = std::array::from_fn(|index| x[index] + ds * tangent[index]);
        let pal = match pal_newton_step(
            context.mu,
            x_start,
            x,
            current.state,
            tangent,
            ds,
            1e-6,
            100,
            context.rtol,
            context.atol,
            max_step,
        ) {
            Ok(outcome) => outcome,
            Err(message) => {
                return Ok(Outcome::soft_failure(
                    "nrho",
                    "periodic",
                    member_limit,
                    members,
                    "failed",
                    "integration_failed",
                    message,
                ));
            }
        };
        if pal.singular {
            ds *= 0.5;
            if ds < 1e-5 {
                break;
            }
            continue;
        }
        let q = pal.x_new;
        if !(0.75 < q[0]
            && q[0] < 1.05
            && 1e-3 < q[1].abs()
            && q[1].abs() < 0.55
            && 0.35 < q[3]
            && q[3] < std::f64::consts::FRAC_PI_2)
        {
            ds *= 0.5;
            if ds < 1e-5 {
                break;
            }
            continue;
        }
        let guess = PeriodicOrbit {
            state: [q[0], 0.0, q[1], 0.0, q[2], 0.0],
            period: 2.0 * q[3],
            closure_error: f64::INFINITY,
        };
        let next = match correct_halo_fixed_z(context, q[1], 1, Some(&guess)) {
            Ok(orbit) => orbit,
            Err(_) => {
                ds *= 0.5;
                if ds < 1e-5 {
                    break;
                }
                continue;
            }
        };
        let previous_tangent = tangent;
        current = next;
        x = [
            current.state[0],
            current.state[2],
            current.state[4],
            current.period / 2.0,
        ];
        tangent = match f_df_tangent(
            context.mu,
            x,
            current.state,
            context.rtol,
            context.atol,
            max_step,
        ) {
            Ok(result) => result.tangent,
            Err(message) => {
                return Ok(Outcome::soft_failure(
                    "nrho",
                    "periodic",
                    member_limit,
                    members,
                    "failed",
                    "integration_failed",
                    message,
                ));
            }
        };
        let alignment: f64 = tangent
            .iter()
            .zip(previous_tangent)
            .map(|(left, right)| left * right)
            .sum();
        if alignment < 0.0 {
            tangent.iter_mut().for_each(|value| *value = -*value);
        }

        let dmin = match perilune(context, &current) {
            Ok(distance) => distance,
            Err(failure) => {
                return Ok(Outcome::soft_failure(
                    "nrho",
                    "periodic",
                    member_limit,
                    members,
                    failure.status,
                    failure.cause,
                    failure.message,
                ));
            }
        };
        if dmin <= threshold {
            let mut output = current;
            if z_sign < 0.0 {
                output.state[2] = -output.state[2];
                output.state[5] = -output.state[5];
            }
            members.push(member(context, output, dmin));
            if members.len() >= member_limit {
                return Ok(Outcome::converged(
                    "nrho",
                    "periodic",
                    member_limit,
                    members,
                ));
            }
        }
        if dmin * context.characteristic_length_km <= context.secondary_radius_km + 500.0 {
            break;
        }
    }
    Ok(Outcome::soft_failure(
        "nrho",
        "periodic",
        member_limit,
        members,
        "stagnated",
        "stagnation_detected",
        "L1 NRHO 的 Rust PAL 在步数或步长预算内未生成全部请求成员",
    ))
}

fn perilune(context: Context, orbit: &PeriodicOrbit) -> Result<f64, Failure> {
    metric_minmax(context, orbit.state, orbit.period, "moon-distance", 0, 1000).map(|range| range.0)
}

fn member(context: Context, orbit: PeriodicOrbit, dmin: f64) -> Member {
    let mut result = Member::periodic(orbit.state, orbit.period, orbit.closure_error, None);
    result.perilune_height_km =
        Some(dmin * context.characteristic_length_km - context.secondary_radius_km);
    result
}
