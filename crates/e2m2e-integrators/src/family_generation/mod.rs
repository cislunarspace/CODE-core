//! 八类 CR3BP 轨道族的统一 Rust 生成模块。
//!
//! 该模块的接口是一次请求、一次返回；种子、修正、延拓、步长、筛选和
//! 结构化终止全部藏在接口后。PyO3 适配器只把已校验参数翻成内部 `Spec`。

mod axial;
mod common;
mod dro;
mod halo;
mod lissajous;
mod nrho;
mod triangular;
mod types;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use common::jacobi_constant;
use types::{Context, Member, Outcome, Spec};

fn required<T>(value: Option<T>, name: &str, family_type: &str) -> Result<T, String> {
    value.ok_or_else(|| format!("{family_type} 缺少必需字段 {name}"))
}

/// 能量窗口扫描中共线族（Halo/NRHO/Axial）延拓 walk 的最小成员预算。
/// 这些族的行走以振幅上限为终点、以成员数为步数上限；预算低于到达
/// 族默认振幅上限所需步数时 trace 会在中途截断，窗口失去高能量段的
/// 命中机会。三角族的 walk 步数公式已含按请求范围的步数估计，无需
/// 下限。
const MIN_TRACE_MEMBERS: usize = 200;

#[allow(clippy::too_many_arguments)]
fn build_spec(
    family_type: &str,
    point: u8,
    n_orbits: usize,
    max_amplitude_km: Option<f64>,
    min_amplitude_km: Option<f64>,
    perilune_height_max_km: Option<f64>,
    north_south: Option<u8>,
    amplitude_in_km: Option<f64>,
    amplitude_out_km: Option<f64>,
    phase_in: Option<f64>,
    phase_out: Option<f64>,
    continuation_direction: Option<&str>,
    match_tolerance_km: Option<f64>,
    n_periods: usize,
) -> Result<Spec, String> {
    match family_type {
        "halo" => Ok(Spec::Halo {
            point,
            max_amplitude_km: required(max_amplitude_km, "max_amplitude_km", family_type)?,
            member_limit: n_orbits,
        }),
        "nrho" => Ok(Spec::Nrho {
            point,
            north_south: required(north_south, "north_south", family_type)?,
            perilune_height_max_km: required(
                perilune_height_max_km,
                "perilune_height_max_km",
                family_type,
            )?,
            member_limit: n_orbits,
        }),
        "axial" => Ok(Spec::Axial {
            point,
            max_amplitude_km: required(max_amplitude_km, "max_amplitude_km", family_type)?,
            member_limit: n_orbits,
        }),
        "lissajous" => Ok(Spec::Lissajous {
            point,
            amplitude_in_km: required(amplitude_in_km, "amplitude_in_km", family_type)?,
            amplitude_out_km: required(amplitude_out_km, "amplitude_out_km", family_type)?,
            phase_in: required(phase_in, "phase_in", family_type)?,
            phase_out: required(phase_out, "phase_out", family_type)?,
            member_limit: n_orbits,
            n_periods,
        }),
        "spo" | "lpo" | "horseshoe" => {
            let direction = match continuation_direction.unwrap_or("decrease-x0") {
                "increase-x0" => "increase-x0",
                "decrease-x0" => "decrease-x0",
                value => return Err(format!("未知 continuation_direction {value}")),
            };
            let kind = match family_type {
                "spo" => "spo",
                "lpo" => "lpo",
                _ => "horseshoe",
            };
            Ok(Spec::Triangular {
                family_type: kind,
                point,
                min_amplitude_km: required(min_amplitude_km, "min_amplitude_km", family_type)?,
                max_amplitude_km: required(max_amplitude_km, "max_amplitude_km", family_type)?,
                member_limit: n_orbits,
                direction,
                match_tolerance_km: required(
                    match_tolerance_km,
                    "match_tolerance_km",
                    family_type,
                )?,
            })
        }
        "dro" => Ok(Spec::Dro {
            min_amplitude_km: required(min_amplitude_km, "min_amplitude_km", family_type)?,
            max_amplitude_km: required(max_amplitude_km, "max_amplitude_km", family_type)?,
            member_limit: n_orbits,
        }),
        _ => Err(format!("未知 orbit family {family_type}")),
    }
}

fn generate(context: Context, spec: Spec) -> Result<Outcome, String> {
    match spec {
        Spec::Halo {
            point,
            max_amplitude_km,
            member_limit,
        } => halo::generate(context, point, max_amplitude_km, member_limit),
        Spec::Nrho {
            point,
            north_south,
            perilune_height_max_km,
            member_limit,
        } => nrho::generate(
            context,
            point,
            north_south,
            perilune_height_max_km,
            member_limit,
        ),
        Spec::Axial {
            point,
            max_amplitude_km,
            member_limit,
        } => axial::generate(context, point, max_amplitude_km, member_limit),
        Spec::Lissajous {
            point,
            amplitude_in_km,
            amplitude_out_km,
            phase_in,
            phase_out,
            member_limit,
            n_periods,
        } => lissajous::generate(
            context,
            point,
            amplitude_in_km,
            amplitude_out_km,
            phase_in,
            phase_out,
            member_limit,
            n_periods,
        ),
        Spec::Triangular {
            family_type,
            point,
            min_amplitude_km,
            max_amplitude_km,
            member_limit,
            direction,
            match_tolerance_km,
        } => triangular::generate(
            context,
            family_type,
            point,
            min_amplitude_km,
            max_amplitude_km,
            member_limit,
            direction,
            match_tolerance_km,
        ),
        Spec::Dro {
            min_amplitude_km,
            max_amplitude_km,
            member_limit,
        } => dro::generate(context, min_amplitude_km, max_amplitude_km, member_limit),
    }
}

/// 把 Python 传入的窗口列表翻成 (min, max) 对；空列表、长度不为 2、
/// 非有限值或 min ≥ max 均为非法输入。
fn parse_windows(raw: Vec<Vec<f64>>) -> Result<Vec<(f64, f64)>, String> {
    if raw.is_empty() {
        return Err("jacobi_windows 不能为空".to_string());
    }
    raw.into_iter()
        .map(|window| {
            if window.len() != 2 {
                return Err(format!(
                    "jacobi_windows 每项须为 [min, max]，当前长度 {}",
                    window.len()
                ));
            }
            let (lower, upper) = (window[0], window[1]);
            if !lower.is_finite() || !upper.is_finite() || lower >= upper {
                return Err(format!(
                    "jacobi_windows 每项须为有限数且 min < max，当前 [{lower}, {upper}]"
                ));
            }
            Ok((lower, upper))
        })
        .collect()
}

/// 能量窗口扫描的延拓 walk 成员预算：同组窗口共享一条 trace，预算须
/// 同时覆盖成员需求（每窗口上限 × 窗口数）与（共线族）到达族默认振幅
/// 上限所需步数。Lissajous 是参数采样而非延拓 trace，不参与能量窗口。
fn walk_limit_for_windows(
    family_type: &str,
    per_window_limit: usize,
    window_count: usize,
) -> Result<usize, String> {
    let demand = per_window_limit * window_count;
    match family_type {
        "spo" | "lpo" | "horseshoe" => Ok(demand),
        "halo" | "nrho" | "axial" => Ok(demand.max(MIN_TRACE_MEMBERS)),
        "lissajous" => {
            Err("Lissajous 族是参数采样而非延拓 trace，不参与 Jacobi 能量窗口".to_string())
        }
        other => Err(format!("未知 orbit family {other}")),
    }
}

/// 能量窗口批量生成：延拓 trace 只走一次（`spec` 的 member_limit 已按
/// `walk_limit_for_windows` 放大），各窗口在 trace 上分别筛选成员
/// （窗口边界包含），每窗口至多 `per_window_limit` 条、一条结果。
/// 与 ADR 0029 决策 4 的公开振幅窗口同层：筛选留在单次调用内。
fn generate_windowed(
    context: Context,
    spec: Spec,
    windows: Vec<(f64, f64)>,
    per_window_limit: usize,
) -> Result<Vec<Outcome>, String> {
    let full = generate(context, spec)?;
    let jacobis: Vec<f64> = full
        .members
        .iter()
        .map(|member| jacobi_constant(context.mu, member.states[0]))
        .collect();
    windows
        .iter()
        .map(|&(lower, upper)| {
            let members: Vec<Member> = full
                .members
                .iter()
                .zip(&jacobis)
                .filter(|(_, jacobi)| lower <= **jacobi && **jacobi <= upper)
                .map(|(member, _)| member.clone())
                .take(per_window_limit)
                .collect();
            if !members.is_empty() || full.members.is_empty() {
                // 命中成员：继承基线结局（含软失败但已有成员的 trace）；
                // 基线 trace 零成员：窗口无从筛选，逐窗口保留基线原因。
                return Ok(Outcome {
                    family_type: full.family_type,
                    periodicity: full.periodicity,
                    status: full.status,
                    cause: full.cause,
                    message: full.message.clone(),
                    requested_members: per_window_limit,
                    members,
                });
            }
            Ok(Outcome::soft_failure(
                full.family_type,
                full.periodicity,
                per_window_limit,
                Vec::new(),
                "infeasible",
                "constraint_violation",
                format!("Jacobi 窗口 [{lower}, {upper}] 内零成员命中：族能量包络未覆盖该区间"),
            ))
        })
        .collect()
}

/// 一次调用完成八类 CR3BP 轨道族生成。
fn validate_context(
    mu: f64,
    characteristic_length_km: f64,
    secondary_radius_km: f64,
    rtol: f64,
    atol: f64,
    max_step: Option<f64>,
) -> Result<(), String> {
    if !(0.0..0.5).contains(&mu)
        || characteristic_length_km <= 0.0
        || secondary_radius_km <= 0.0
        || rtol <= 0.0
        || atol <= 0.0
        || max_step.is_some_and(|step| step <= 0.0)
    {
        return Err("CR3BP 族生成上下文无效".to_string());
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (
    family_type,
    mu,
    characteristic_length_km,
    secondary_radius_km,
    point,
    n_orbits,
    max_amplitude_km=None,
    min_amplitude_km=None,
    perilune_height_max_km=None,
    north_south=None,
    amplitude_in_km=None,
    amplitude_out_km=None,
    phase_in=None,
    phase_out=None,
    continuation_direction=None,
    match_tolerance_km=None,
    n_periods=3,
    rtol=1e-12,
    atol=1e-12,
    max_step=None
))]
#[allow(clippy::too_many_arguments)]
pub fn generate_cr3bp_family_py(
    family_type: &str,
    mu: f64,
    characteristic_length_km: f64,
    secondary_radius_km: f64,
    point: u8,
    n_orbits: usize,
    max_amplitude_km: Option<f64>,
    min_amplitude_km: Option<f64>,
    perilune_height_max_km: Option<f64>,
    north_south: Option<u8>,
    amplitude_in_km: Option<f64>,
    amplitude_out_km: Option<f64>,
    phase_in: Option<f64>,
    phase_out: Option<f64>,
    continuation_direction: Option<&str>,
    match_tolerance_km: Option<f64>,
    n_periods: usize,
    rtol: f64,
    atol: f64,
    max_step: Option<f64>,
    py: Python<'_>,
) -> PyResult<PyObject> {
    validate_context(
        mu,
        characteristic_length_km,
        secondary_radius_km,
        rtol,
        atol,
        max_step,
    )
    .map_err(PyValueError::new_err)?;
    let spec = build_spec(
        family_type,
        point,
        n_orbits,
        max_amplitude_km,
        min_amplitude_km,
        perilune_height_max_km,
        north_south,
        amplitude_in_km,
        amplitude_out_km,
        phase_in,
        phase_out,
        continuation_direction,
        match_tolerance_km,
        n_periods,
    )
    .map_err(PyValueError::new_err)?;
    let context = Context {
        mu,
        characteristic_length_km,
        secondary_radius_km,
        rtol,
        atol,
        max_step,
    };
    let outcome = py
        .allow_threads(|| generate(context, spec))
        .map_err(PyValueError::new_err)?;
    outcome_to_python(py, outcome)
}

/// 按 Jacobi 能量窗口批量生成轨道族：同一组生成参数下延拓 trace 只走
/// 一次，返回与 ``jacobi_windows`` 同序的结果列表（每窗口一条，成员
/// Jacobi 均落在窗口内，边界包含）。窗口成员超上限时按 trace 顺序取
/// 前 ``n_orbits`` 条（与一维扫描的成员上限语义一致）。窗口零成员时
/// 该窗口结果为零成员的结构化软失败；族生成参数与
/// ``generate_cr3bp_family_py`` 同集（走能量窗口时族延拓范围取各族
/// 默认振幅/近月点上限，由调用方给定）。
#[pyfunction]
#[pyo3(signature = (
    family_type,
    mu,
    characteristic_length_km,
    secondary_radius_km,
    point,
    n_orbits,
    jacobi_windows,
    max_amplitude_km=None,
    min_amplitude_km=None,
    perilune_height_max_km=None,
    north_south=None,
    amplitude_in_km=None,
    amplitude_out_km=None,
    phase_in=None,
    phase_out=None,
    continuation_direction=None,
    match_tolerance_km=None,
    n_periods=3,
    rtol=1e-12,
    atol=1e-12,
    max_step=None
))]
#[allow(clippy::too_many_arguments)]
pub fn generate_cr3bp_family_windows_py(
    family_type: &str,
    mu: f64,
    characteristic_length_km: f64,
    secondary_radius_km: f64,
    point: u8,
    n_orbits: usize,
    jacobi_windows: Vec<Vec<f64>>,
    max_amplitude_km: Option<f64>,
    min_amplitude_km: Option<f64>,
    perilune_height_max_km: Option<f64>,
    north_south: Option<u8>,
    amplitude_in_km: Option<f64>,
    amplitude_out_km: Option<f64>,
    phase_in: Option<f64>,
    phase_out: Option<f64>,
    continuation_direction: Option<&str>,
    match_tolerance_km: Option<f64>,
    n_periods: usize,
    rtol: f64,
    atol: f64,
    max_step: Option<f64>,
    py: Python<'_>,
) -> PyResult<PyObject> {
    validate_context(
        mu,
        characteristic_length_km,
        secondary_radius_km,
        rtol,
        atol,
        max_step,
    )
    .map_err(PyValueError::new_err)?;
    if n_orbits == 0 {
        return Err(PyValueError::new_err("n_orbits 必须大于 0"));
    }
    let windows = parse_windows(jacobi_windows).map_err(PyValueError::new_err)?;
    let walk_limit = walk_limit_for_windows(family_type, n_orbits, windows.len())
        .map_err(PyValueError::new_err)?;
    let spec = build_spec(
        family_type,
        point,
        walk_limit,
        max_amplitude_km,
        min_amplitude_km,
        perilune_height_max_km,
        north_south,
        amplitude_in_km,
        amplitude_out_km,
        phase_in,
        phase_out,
        continuation_direction,
        match_tolerance_km,
        n_periods,
    )
    .map_err(PyValueError::new_err)?;
    let context = Context {
        mu,
        characteristic_length_km,
        secondary_radius_km,
        rtol,
        atol,
        max_step,
    };
    let outcomes = py
        .allow_threads(|| generate_windowed(context, spec, windows, n_orbits))
        .map_err(PyValueError::new_err)?;
    let list = PyList::empty(py);
    for outcome in outcomes {
        list.append(outcome_to_python(py, outcome)?)?;
    }
    Ok(list.into())
}

fn outcome_to_python(py: Python<'_>, outcome: Outcome) -> PyResult<PyObject> {
    let result = PyDict::new(py);
    result.set_item("family_type", outcome.family_type)?;
    result.set_item("periodicity", outcome.periodicity)?;
    result.set_item("status", outcome.status)?;
    result.set_item("cause", outcome.cause)?;
    result.set_item("message", outcome.message)?;
    result.set_item("requested_members", outcome.requested_members)?;
    result.set_item("generated_members", outcome.members.len())?;
    let members = PyList::empty(py);
    for member in outcome.members {
        let item = PyDict::new(py);
        item.set_item("states", member.states)?;
        item.set_item("times", member.times)?;
        item.set_item("period", member.period)?;
        item.set_item("closure_error", member.closure_error)?;
        item.set_item("amplitude_km", member.amplitude_km)?;
        item.set_item("perilune_height_km", member.perilune_height_km)?;
        item.set_item("sampling_fraction", member.sampling_fraction)?;
        item.set_item("jacobi_drift", member.jacobi_drift)?;
        item.set_item("newton_iterations", member.newton_iterations)?;
        item.set_item("tangent_system_rank", member.tangent_system_rank)?;
        item.set_item("tangent_system_condition", member.tangent_system_condition)?;
        item.set_item("augmented_system_rank", member.augmented_system_rank)?;
        item.set_item(
            "augmented_system_condition",
            member.augmented_system_condition,
        )?;
        item.set_item("step_size", member.step_size)?;
        members.append(item)?;
    }
    result.set_item("members", members)?;
    Ok(result.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> Context {
        Context {
            mu: 0.012_150_585_350_562_453,
            characteristic_length_km: 384_400.0,
            secondary_radius_km: 1737.4,
            // 测试套件用筛选级容差（ADR 0021 #536 精神），不用研究级 1e-12：
            // 契约断言只要求 closure ≤ 1e-8，1e-9 下全部满足且套件快约 25%。
            rtol: 1e-9,
            atol: 1e-9,
            // 0.05 与 0.01 的契约断言结果一致，整组测试约快 2 倍（max_step
            // 是本测试的主导成本，而非 rtol）。
            max_step: Some(0.05),
        }
    }

    fn halo_spec(max_amplitude_km: f64, member_limit: usize) -> Spec {
        Spec::Halo {
            point: 1,
            max_amplitude_km,
            member_limit,
        }
    }

    fn windowed_halo(
        max_amplitude_km: f64,
        windows: &[(f64, f64)],
        per_window_limit: usize,
    ) -> Vec<Outcome> {
        let walk_limit = walk_limit_for_windows("halo", per_window_limit, windows.len()).unwrap();
        generate_windowed(
            context(),
            halo_spec(max_amplitude_km, walk_limit),
            windows.to_vec(),
            per_window_limit,
        )
        .unwrap()
    }

    fn jacobis_of(members: &[Member]) -> Vec<f64> {
        members
            .iter()
            .map(|member| jacobi_constant(context().mu, member.states[0]))
            .collect()
    }

    fn jacobi_bounds(members: &[Member]) -> (f64, f64) {
        let jacobis = jacobis_of(members);
        (
            jacobis.iter().cloned().fold(f64::INFINITY, f64::min),
            jacobis.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        )
    }

    #[test]
    fn parse_windows_rejects_degenerate_entries() {
        assert!(parse_windows(vec![]).is_err());
        assert!(parse_windows(vec![vec![3.0]]).is_err());
        assert!(parse_windows(vec![vec![3.0, 3.1, 3.2]]).is_err());
        assert!(parse_windows(vec![vec![3.2, 3.0]]).is_err());
        assert!(parse_windows(vec![vec![f64::NAN, 3.0]]).is_err());
        assert_eq!(
            parse_windows(vec![vec![3.0, 3.1]]).unwrap(),
            vec![(3.0, 3.1)]
        );
    }

    #[test]
    fn walk_limit_for_windows_rejects_lissajous_and_floors_collinear_walks() {
        assert!(walk_limit_for_windows("lissajous", 4, 2).is_err());
        assert!(walk_limit_for_windows("dro", 4, 2).is_err());
        assert_eq!(
            walk_limit_for_windows("halo", 4, 2).unwrap(),
            MIN_TRACE_MEMBERS
        );
        assert_eq!(
            walk_limit_for_windows("nrho", 4, 2).unwrap(),
            MIN_TRACE_MEMBERS
        );
        assert_eq!(
            walk_limit_for_windows("axial", 4, 2).unwrap(),
            MIN_TRACE_MEMBERS
        );
        // 三角族 walk 步数公式已含范围估计，按成员需求给预算即可
        assert_eq!(walk_limit_for_windows("spo", 4, 2).unwrap(), 8);
        assert_eq!(walk_limit_for_windows("horseshoe", 30, 3).unwrap(), 90);
    }

    #[test]
    fn jacobi_windows_partition_members_into_bounds() {
        let full = generate(context(), halo_spec(3000.0, 200)).unwrap();
        assert_eq!(full.status, "converged");
        let (c_min, c_max) = jacobi_bounds(&full.members);
        let middle = 0.5 * (c_min + c_max);
        let windows = [(c_min, middle), (middle, c_max)];
        let outcomes = windowed_halo(3000.0, &windows, 50);

        assert_eq!(outcomes.len(), 2);
        let mut total = 0usize;
        for (outcome, (lower, upper)) in outcomes.iter().zip(windows) {
            assert_eq!(outcome.status, "converged");
            assert!(!outcome.members.is_empty());
            total += outcome.members.len();
            for jacobi in jacobis_of(&outcome.members) {
                assert!(
                    (lower..=upper).contains(&jacobi),
                    "成员 Jacobi {jacobi} 越出窗口 [{lower}, {upper}]"
                );
            }
        }
        // 窗口二分覆盖族能量包络时不丢成员（成员上限不生效）
        assert_eq!(total, full.members.len());
    }

    #[test]
    fn jacobi_window_bounds_are_inclusive() {
        let full = generate(context(), halo_spec(3000.0, 200)).unwrap();
        let first_jacobi = jacobis_of(&full.members)[0];
        // 下界恰取首个成员的 Jacobi：边界包含意味着该成员必须命中
        let outcomes = windowed_halo(3000.0, &[(first_jacobi, first_jacobi + 0.01)], 10);
        assert_eq!(outcomes.len(), 1);
        assert_eq!(outcomes[0].status, "converged");
        assert!(jacobis_of(&outcomes[0].members)
            .iter()
            .any(|jacobi| (*jacobi - first_jacobi).abs() < 1e-12));
    }

    #[test]
    fn empty_jacobi_window_is_structured_soft_failure() {
        let outcomes = windowed_halo(3000.0, &[(9.9, 9.95)], 10);
        assert_eq!(outcomes.len(), 1);
        assert_eq!(outcomes[0].status, "infeasible");
        assert_eq!(outcomes[0].cause, "constraint_violation");
        assert!(outcomes[0].message.contains("零成员"));
        assert!(outcomes[0].members.is_empty());
        assert_eq!(outcomes[0].requested_members, 10);
    }

    #[test]
    fn per_window_limit_caps_members_inside_window() {
        let full = generate(context(), halo_spec(3000.0, 200)).unwrap();
        let (c_min, c_max) = jacobi_bounds(&full.members);
        let outcomes = windowed_halo(3000.0, &[(c_min, c_max)], 2);
        assert_eq!(outcomes.len(), 1);
        assert_eq!(outcomes[0].members.len(), 2);
        assert_eq!(outcomes[0].requested_members, 2);
    }

    #[test]
    fn triangular_windows_filter_runs_parallel_to_amplitude_window() {
        let make_spec = |member_limit: usize| Spec::Triangular {
            family_type: "spo",
            point: 4,
            min_amplitude_km: 5000.0,
            max_amplitude_km: 20_000.0,
            member_limit,
            direction: "decrease-x0",
            match_tolerance_km: 20.0,
        };
        let full = generate(context(), make_spec(8)).unwrap();
        assert_eq!(full.status, "converged");
        let (c_min, c_max) = jacobi_bounds(&full.members);
        let middle = 0.5 * (c_min + c_max);
        let windows = [(c_min, middle), (middle, c_max)];
        let walk_limit = walk_limit_for_windows("spo", 8, windows.len()).unwrap();
        let outcomes =
            generate_windowed(context(), make_spec(walk_limit), windows.to_vec(), 8).unwrap();

        for (outcome, (lower, upper)) in outcomes.iter().zip(windows) {
            assert!(!outcome.members.is_empty());
            for jacobi in jacobis_of(&outcome.members) {
                assert!((lower..=upper).contains(&jacobi));
            }
            for member in &outcome.members {
                // 振幅窗口筛选不受能量窗口影响（两者并列生效）
                assert!(member
                    .amplitude_km
                    .is_some_and(|amp| (5000.0..=20_000.0).contains(&amp)));
            }
        }
    }

    #[test]
    fn lissajous_single_call_generates_requested_samples() {
        let result = generate(
            context(),
            Spec::Lissajous {
                point: 2,
                amplitude_in_km: 2400.0,
                amplitude_out_km: 7200.0,
                phase_in: 0.01,
                phase_out: 0.55,
                member_limit: 3,
                n_periods: 3,
            },
        )
        .unwrap();
        assert_eq!(result.status, "converged");
        assert_eq!(result.periodicity, "quasi-periodic");
        assert_eq!(result.members.len(), 3);
        assert_eq!(result.members[0].states.len(), 180);
        let x_l = crate::family::collinear_libration_x(context().mu, 2).unwrap();
        for (index, member) in result.members.iter().enumerate() {
            let requested_fraction = (index + 1) as f64 / 3.0;
            let requested_scale_km = requested_fraction * 2400.0_f64.hypot(7200.0);
            let max_distance_km = member
                .states
                .iter()
                .map(|state| {
                    ((state[0] - x_l).powi(2) + state[1].powi(2) + state[2].powi(2)).sqrt()
                        * context().characteristic_length_km
                })
                .fold(0.0, f64::max);
            assert!(max_distance_km < 2.5 * requested_scale_km);
        }
    }

    #[test]
    fn dro_members_stay_in_window_and_sorted_by_amplitude() {
        // 窗口跨种子振幅（约 90,786 km）：双向行走后按振幅升序输出，
        // 两侧成员都存在
        let result = generate(
            context(),
            Spec::Dro {
                min_amplitude_km: 85_000.0,
                max_amplitude_km: 100_000.0,
                member_limit: 4,
            },
        )
        .unwrap();
        assert_eq!(result.status, "converged", "{}", result.message);
        assert_eq!(result.members.len(), 4);
        let amplitudes: Vec<f64> = result
            .members
            .iter()
            .map(|member| member.amplitude_km.unwrap())
            .collect();
        assert!(amplitudes.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(amplitudes
            .iter()
            .all(|amp| (85_000.0..=100_000.0).contains(amp)));
        // 双向覆盖：种子（≈90,786 km）两侧都有成员
        assert!(amplitudes.first().unwrap() < &90_000.0);
        assert!(amplitudes.last().unwrap() > &91_000.0);
    }

    #[test]
    fn dro_rejects_degenerate_amplitude_window() {
        assert!(generate(
            context(),
            Spec::Dro {
                min_amplitude_km: 20_000.0,
                max_amplitude_km: 20_000.0,
                member_limit: 1,
            }
        )
        .is_err());
        assert!(generate(
            context(),
            Spec::Dro {
                min_amplitude_km: 30_000.0,
                max_amplitude_km: 20_000.0,
                member_limit: 1,
            }
        )
        .is_err());
        assert!(generate(
            context(),
            Spec::Dro {
                min_amplitude_km: 1000.0,
                max_amplitude_km: 20_000.0,
                member_limit: 0,
            }
        )
        .is_err());
    }

    #[test]
    fn periodic_families_satisfy_unified_result_contract() {
        let specs = [
            Spec::Halo {
                point: 1,
                max_amplitude_km: 3000.0,
                member_limit: 2,
            },
            Spec::Nrho {
                point: 1,
                north_south: 1,
                perilune_height_max_km: 30_000.0,
                member_limit: 1,
            },
            Spec::Nrho {
                point: 2,
                north_south: 2,
                perilune_height_max_km: 30_000.0,
                member_limit: 2,
            },
            Spec::Axial {
                point: 2,
                max_amplitude_km: 1500.0,
                member_limit: 2,
            },
            Spec::Triangular {
                family_type: "spo",
                point: 4,
                min_amplitude_km: 5000.0,
                max_amplitude_km: 20_000.0,
                member_limit: 2,
                direction: "decrease-x0",
                match_tolerance_km: 20.0,
            },
            Spec::Triangular {
                family_type: "lpo",
                point: 5,
                min_amplitude_km: 5000.0,
                max_amplitude_km: 30_000.0,
                member_limit: 2,
                direction: "decrease-x0",
                match_tolerance_km: 20.0,
            },
            Spec::Triangular {
                family_type: "horseshoe",
                point: 4,
                min_amplitude_km: 50_000.0,
                max_amplitude_km: 110_000.0,
                member_limit: 1,
                direction: "decrease-x0",
                match_tolerance_km: 50.0,
            },
            Spec::Dro {
                min_amplitude_km: 5000.0,
                max_amplitude_km: 20_000.0,
                member_limit: 1,
            },
        ];

        for spec in specs {
            let result = generate(context(), spec).unwrap();
            assert_eq!(
                result.status, "converged",
                "{}: {}",
                result.cause, result.message
            );
            assert_eq!(result.periodicity, "periodic");
            assert!(!result.members.is_empty());
            for member in result.members {
                assert!(member.period.is_some_and(|period| period > 0.0));
                assert!(member.closure_error.is_some_and(|closure| closure <= 1e-8));
            }
        }
    }
}
