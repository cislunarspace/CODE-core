//! Halo 自然参数族的单次 Rust 生成。

use super::common::correct_halo_fixed_z;
use super::types::{Context, Member, Outcome};

const SEED_Z0: f64 = 0.001;

pub(crate) fn generate(
    context: Context,
    point: u8,
    max_amplitude_km: f64,
    member_limit: usize,
) -> Result<Outcome, String> {
    if point != 1 && point != 2 {
        return Err("Halo 平动点必须为 L1 或 L2".to_string());
    }
    if member_limit == 0 {
        return Err("n_orbits 必须大于 0".to_string());
    }
    if max_amplitude_km == 0.0 || !max_amplitude_km.is_finite() {
        return Err("max_amplitude_km 必须为有限非零数".to_string());
    }
    let branch_sign = max_amplitude_km.signum();
    let target_z = max_amplitude_km / context.characteristic_length_km;
    let seed_z = branch_sign * SEED_Z0;
    let seed = match correct_halo_fixed_z(context, seed_z, point, None) {
        Ok(seed) => seed,
        Err(failure) => {
            return Ok(Outcome::soft_failure(
                "halo",
                "periodic",
                member_limit,
                Vec::new(),
                failure.status,
                failure.cause,
                failure.message,
            ));
        }
    };
    let mut members = vec![Member::periodic(
        seed.state,
        seed.period,
        seed.closure_error,
        None,
    )];
    if member_limit == 1 || (target_z - seed_z).abs() <= 1e-12 {
        return Ok(Outcome::converged(
            "halo",
            "periodic",
            member_limit,
            members,
        ));
    }

    let direction = (target_z - seed_z).signum();
    let mut current = seed;
    let mut current_z = seed_z;
    let step: f64 = 0.001;
    while members.len() < member_limit {
        let remaining = (target_z - current_z) * direction;
        if remaining <= 1e-12 {
            break;
        }
        let next_z = current_z + direction * step.min(remaining);
        match correct_halo_fixed_z(context, next_z, point, Some(&current)) {
            Ok(next) => {
                members.push(Member::periodic(
                    next.state,
                    next.period,
                    next.closure_error,
                    None,
                ));
                current = next;
                current_z = next_z;
            }
            Err(failure) => {
                return Ok(Outcome::soft_failure(
                    "halo",
                    "periodic",
                    member_limit,
                    members,
                    failure.status,
                    failure.cause,
                    failure.message,
                ));
            }
        }
    }
    Ok(Outcome::converged(
        "halo",
        "periodic",
        member_limit,
        members,
    ))
}
