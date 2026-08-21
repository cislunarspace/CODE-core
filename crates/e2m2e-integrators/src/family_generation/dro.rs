//! DRO（远距逆行轨道族）的单次 Rust 生成。

use super::common::{correct_dro_fixed_x, dro_amplitude_km, Failure, SEED_DRO_X0};
use super::types::{Context, Member, Outcome, PeriodicOrbit};

/// 名义延拓步长（x0 方向）。月球附近轨道小、族参数对振幅敏感，固定步长
/// 会让修正发散：失败时步长减半重试，成功后恢复名义步长。
const STEP_X0: f64 = 0.005;
const MIN_STEP_X0: f64 = 1e-4;
/// 单方向行走的步数预算（兜底；正常在成员满额或越过窗口边界处停止）。
const STEP_BUDGET: usize = 600;

pub(crate) fn generate(
    context: Context,
    min_amplitude_km: f64,
    max_amplitude_km: f64,
    member_limit: usize,
) -> Result<Outcome, String> {
    if member_limit == 0 {
        return Err("n_orbits 必须大于 0".to_string());
    }
    if min_amplitude_km <= 0.0 || min_amplitude_km >= max_amplitude_km {
        return Err("振幅范围必须满足 0 < min_amplitude_km < max_amplitude_km".to_string());
    }

    let seed = match correct_dro_fixed_x(context, SEED_DRO_X0, None) {
        Ok(seed) => seed,
        Err(failure) => return Ok(failure_outcome(member_limit, Vec::new(), failure)),
    };
    let seed_amplitude = match dro_amplitude_km(context, &seed) {
        Ok(amplitude) => amplitude,
        Err(failure) => return Ok(failure_outcome(member_limit, Vec::new(), failure)),
    };

    // 窗口在种子振幅下方：x0 向月侧行走（振幅递减）；上方：向地侧行走
    // （振幅递增）；跨种子窗口双向各走一段，仍是一次族生成调用。
    let directions: &[f64] = if max_amplitude_km <= seed_amplitude {
        &[1.0]
    } else if min_amplitude_km >= seed_amplitude {
        &[-1.0]
    } else {
        &[1.0, -1.0]
    };

    let mut members: Vec<Member> = Vec::new();
    // 覆盖判定：族链跨过窗口边界（含种子自身落在边界外）即视为该侧覆盖。
    let mut covered_lower = seed_amplitude <= min_amplitude_km;
    let mut covered_upper = seed_amplitude >= max_amplitude_km;
    if (min_amplitude_km..=max_amplitude_km).contains(&seed_amplitude) {
        members.push(member(&seed, seed_amplitude));
    }
    let mut failure: Option<Failure> = None;
    'walk: for &direction in directions {
        let mut current = seed;
        for _ in 0..STEP_BUDGET {
            if members.len() >= member_limit {
                break 'walk;
            }
            match step_once(context, direction, current) {
                Step::Orbit(orbit, amplitude) => {
                    covered_lower |= amplitude <= min_amplitude_km;
                    covered_upper |= amplitude >= max_amplitude_km;
                    if (min_amplitude_km..=max_amplitude_km).contains(&amplitude) {
                        members.push(member(&orbit, amplitude));
                    }
                    // 越过该方向上的窗口边界后不再有命中成员
                    if direction > 0.0 && amplitude < min_amplitude_km {
                        break;
                    }
                    if direction < 0.0 && amplitude > max_amplitude_km {
                        break;
                    }
                    current = orbit;
                }
                Step::Failed(walk_failure) => {
                    failure = Some(walk_failure);
                    break;
                }
            }
        }
    }
    members.sort_by(|left, right| {
        left.amplitude_km
            .expect("DRO 成员振幅已测量")
            .total_cmp(&right.amplitude_km.expect("DRO 成员振幅已测量"))
    });

    if !members.is_empty() && (members.len() >= member_limit || (covered_lower && covered_upper)) {
        return Ok(Outcome::converged("dro", "periodic", member_limit, members));
    }
    if let Some(failure) = failure {
        return Ok(failure_outcome(member_limit, members, failure));
    }
    if members.is_empty() {
        return Ok(Outcome::soft_failure(
            "dro",
            "periodic",
            member_limit,
            members,
            "infeasible",
            "constraint_violation",
            format!(
                "DRO 族延拓未覆盖振幅窗口 [{min_amplitude_km:.0}, {max_amplitude_km:.0}] km 的成员"
            ),
        ));
    }
    Ok(Outcome::soft_failure(
        "dro",
        "periodic",
        member_limit,
        members,
        "stagnated",
        "stagnation_detected",
        format!("DRO 族延拓在 {STEP_BUDGET} 步预算内未覆盖完整请求范围"),
    ))
}

enum Step {
    /// 修正成功：周期轨道及其振幅（km）。
    Orbit(PeriodicOrbit, f64),
    /// 修正或测量失败（步长耗尽时已合成停滞失败）。
    Failed(Failure),
}

fn step_once(context: Context, direction: f64, current: PeriodicOrbit) -> Step {
    let mut step = STEP_X0;
    loop {
        let x_try = current.state[0] + direction * step;
        match correct_dro_fixed_x(context, x_try, Some(&current)) {
            Ok(orbit) => match dro_amplitude_km(context, &orbit) {
                Ok(amplitude) => return Step::Orbit(orbit, amplitude),
                Err(failure) => return Step::Failed(failure),
            },
            Err(_) => {
                step *= 0.5;
                if step < MIN_STEP_X0 {
                    return Step::Failed(Failure {
                        status: "stagnated",
                        cause: "stagnation_detected",
                        message: format!("DRO 族行走步长已减至 {MIN_STEP_X0} 仍未跨过振幅窗口"),
                    });
                }
            }
        }
    }
}

fn member(orbit: &PeriodicOrbit, amplitude: f64) -> Member {
    let mut result = Member::periodic(orbit.state, orbit.period, orbit.closure_error, None);
    result.amplitude_km = Some(amplitude);
    result
}

fn failure_outcome(member_limit: usize, members: Vec<Member>, failure: Failure) -> Outcome {
    Outcome::soft_failure(
        "dro",
        "periodic",
        member_limit,
        members,
        failure.status,
        failure.cause,
        failure.message,
    )
}
