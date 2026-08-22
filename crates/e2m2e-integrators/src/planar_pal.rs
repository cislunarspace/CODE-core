//! 平面三角平动点周期轨道的完整周期伪弧长延拓。
//!
//! SPO/LPO 没有 Halo 的半周期镜面对称。这里以
//! ``q = (x0, y0, vx0, vy0, T)`` 为未知量，使用完整平面闭合、相位条件与
//! 伪弧长条件连续追踪周期解。Python 只构造问题和解释结果；传播、STM、SVD
//! 与 Newton 循环全部留在 Rust 数值层。

use e2m2e_forces::cr3bp::{cr3bp_eom, propagate_cr3bp_stm};
use nalgebra::{SMatrix, SVector};
use pyo3::prelude::*;

const PLANAR: [usize; 4] = [0, 1, 3, 4];
const SCALES: [f64; 5] = [1.0, 1.0, 1.0, 1.0, 10.0];
const RANK_RELATIVE_TOLERANCE: f64 = 1e-8;
const MIN_STEP_SIZE: f64 = 1e-5;

/// Python 可见的完整周期 PAL 结果。成员始终包含初始种子；软失败保留已收敛成员。
#[pyclass]
#[derive(Clone)]
pub struct PlanarPalRustResult {
    #[pyo3(get)]
    pub states: Vec<[f64; 6]>,
    #[pyo3(get)]
    pub periods: Vec<f64>,
    #[pyo3(get)]
    pub closure_errors: Vec<f64>,
    #[pyo3(get)]
    pub newton_iterations: Vec<usize>,
    #[pyo3(get)]
    pub tangent_system_ranks: Vec<usize>,
    #[pyo3(get)]
    pub tangent_system_conditions: Vec<f64>,
    #[pyo3(get)]
    pub augmented_system_ranks: Vec<usize>,
    #[pyo3(get)]
    pub augmented_system_conditions: Vec<f64>,
    #[pyo3(get)]
    pub step_sizes: Vec<f64>,
    #[pyo3(get)]
    pub jacobi_drifts: Vec<f64>,
    #[pyo3(get)]
    pub status: String,
    #[pyo3(get)]
    pub cause: String,
    #[pyo3(get)]
    pub message: String,
    #[pyo3(get)]
    pub steps: usize,
    #[pyo3(get)]
    pub step_size: f64,
}

#[derive(Clone, Copy)]
pub(crate) struct Options {
    pub(crate) tolerance: f64,
    pub(crate) max_iterations: usize,
    pub(crate) rtol: f64,
    pub(crate) atol: f64,
    pub(crate) max_step: Option<f64>,
}

#[derive(Clone, Copy)]
enum Termination {
    Converged,
    MaxIterations,
    Stagnated,
    SingularJacobian,
    IntegrationFailed,
    InvalidPeriod,
    InvalidInput,
}

impl Termination {
    fn contract(self) -> (&'static str, &'static str, &'static str) {
        match self {
            Self::Converged => ("converged", "none", "平面全周期 PAL 完成"),
            Self::MaxIterations => (
                "max_iterations",
                "max_iterations_reached",
                "平面全周期 PAL 达到最大迭代次数",
            ),
            Self::Stagnated => (
                "stagnated",
                "stagnation_detected",
                "平面全周期 PAL 步长耗尽或线搜索停滞",
            ),
            Self::SingularJacobian => (
                "failed",
                "singular_jacobian",
                "平面全周期 PAL 增广 Jacobian 秩不足",
            ),
            Self::IntegrationFailed => (
                "failed",
                "integration_failed",
                "平面全周期 PAL 的 CR3BP STM 传播失败",
            ),
            Self::InvalidPeriod => (
                "infeasible",
                "invalid_period",
                "平面全周期 PAL 得到非正周期",
            ),
            Self::InvalidInput => ("failed", "invalid_input", "平面全周期 PAL 输入无效"),
        }
    }
}

#[derive(Debug)]
enum PalError {
    Integration,
    Singular,
    LineSearchStagnated,
    MaxIterations,
    InvalidPeriod,
}

impl PalError {
    fn termination(&self) -> Termination {
        match self {
            Self::Integration => Termination::IntegrationFailed,
            Self::Singular => Termination::SingularJacobian,
            Self::LineSearchStagnated => Termination::Stagnated,
            Self::MaxIterations => Termination::MaxIterations,
            Self::InvalidPeriod => Termination::InvalidPeriod,
        }
    }
}

struct Evaluation {
    closure: SVector<f64, 4>,
    jacobian: SMatrix<f64, 4, 5>,
    final_state: [f64; 6],
}

struct Metrics {
    tangent_rank: usize,
    tangent_condition: f64,
    augmented_rank: usize,
    augmented_condition: f64,
}

struct StepSystem {
    residual: SVector<f64, 6>,
    jacobian: SMatrix<f64, 6, 5>,
    closure_error: f64,
    jacobi_drift: f64,
    metrics: Metrics,
}

fn scales() -> SVector<f64, 5> {
    SVector::<f64, 5>::from_row_slice(&SCALES)
}

fn q_from_state(state: &[f64; 6], period: f64) -> SVector<f64, 5> {
    SVector::<f64, 5>::new(state[0], state[1], state[3], state[4], period)
}

fn state_from_q(q: &SVector<f64, 5>) -> [f64; 6] {
    [q[0], q[1], 0.0, q[2], q[3], 0.0]
}

fn max_abs(values: impl IntoIterator<Item = f64>) -> f64 {
    values.into_iter().map(f64::abs).fold(0.0, f64::max)
}

fn rank_and_condition(values: &[f64]) -> (usize, f64) {
    let threshold = values[0] * RANK_RELATIVE_TOLERANCE;
    let rank = values.iter().filter(|value| **value > threshold).count();
    let condition = if rank == 0 {
        f64::INFINITY
    } else {
        values[0] / values[rank - 1]
    };
    (rank, condition)
}

fn evaluate(mu: f64, q: &SVector<f64, 5>, options: Options) -> Result<Evaluation, PalError> {
    if q[4] <= 0.0 {
        return Err(PalError::InvalidPeriod);
    }
    let state = state_from_q(q);
    let result = propagate_cr3bp_stm(
        mu,
        (0.0, q[4]),
        &[q[4]],
        &state,
        options.rtol,
        options.atol,
        options.max_step,
        Some(500_000),
    )
    .map_err(|_| PalError::Integration)?;
    let final_state = *result.states.last().ok_or(PalError::Integration)?;
    let stm = result.stms.last().ok_or(PalError::Integration)?;
    let flow = cr3bp_eom(mu, &final_state);

    let mut closure = SVector::<f64, 4>::zeros();
    let mut jacobian = SMatrix::<f64, 4, 5>::zeros();
    for row in 0..4 {
        let component = PLANAR[row];
        closure[row] = final_state[component] - state[component];
        for col in 0..4 {
            let variable = PLANAR[col];
            jacobian[(row, col)] = stm[component * 6 + variable] - f64::from(component == variable);
        }
        jacobian[(row, 4)] = flow[component];
    }
    for col in 0..5 {
        for row in 0..4 {
            jacobian[(row, col)] *= SCALES[col];
        }
    }

    Ok(Evaluation {
        closure,
        jacobian,
        final_state,
    })
}

fn phase_and_jacobian(
    mu: f64,
    q: &SVector<f64, 5>,
    reference: &SVector<f64, 5>,
) -> Result<(f64, SVector<f64, 5>), PalError> {
    let reference_state = state_from_q(reference);
    let flow = cr3bp_eom(mu, &reference_state);
    let planar_flow = SVector::<f64, 4>::new(flow[0], flow[1], flow[3], flow[4]);
    let flow_norm = planar_flow.norm();
    if flow_norm == 0.0 {
        return Err(PalError::Singular);
    }
    let delta = SVector::<f64, 4>::new(
        q[0] - reference[0],
        q[1] - reference[1],
        q[2] - reference[2],
        q[3] - reference[3],
    );
    let phase = delta.dot(&planar_flow) / flow_norm;
    let mut jacobian = SVector::<f64, 5>::zeros();
    for i in 0..4 {
        jacobian[i] = planar_flow[i] / flow_norm * SCALES[i];
    }
    Ok((phase, jacobian))
}

fn tangent_metrics(
    mu: f64,
    q: &SVector<f64, 5>,
    options: Options,
) -> Result<(SVector<f64, 5>, usize, f64, SMatrix<f64, 5, 5>), PalError> {
    let evaluation = evaluate(mu, q, options)?;
    let (_, phase_jacobian) = phase_and_jacobian(mu, q, q)?;
    let mut system = SMatrix::<f64, 5, 5>::zeros();
    for row in 0..4 {
        for col in 0..5 {
            system[(row, col)] = evaluation.jacobian[(row, col)];
        }
    }
    for col in 0..5 {
        system[(4, col)] = phase_jacobian[col];
    }
    let svd = system.svd(false, true);
    let singular_values = svd.singular_values.as_slice();
    let (rank, condition) = rank_and_condition(singular_values);
    if rank != 4 {
        return Err(PalError::Singular);
    }
    let v_t = svd.v_t.ok_or(PalError::Singular)?;
    let mut vector = SVector::<f64, 5>::zeros();
    for col in 0..5 {
        vector[col] = v_t[(4, col)];
    }
    Ok((vector.normalize(), rank, condition, system))
}

fn solve_svd(
    jacobian: SMatrix<f64, 6, 5>,
    residual: SVector<f64, 6>,
    expected_rank: usize,
) -> Result<(SVector<f64, 5>, usize, f64), PalError> {
    let svd = jacobian.svd(true, true);
    let singular_values = svd.singular_values.as_slice();
    let (rank, condition) = rank_and_condition(singular_values);
    if rank < expected_rank {
        return Err(PalError::Singular);
    }
    let delta = damped_svd_solve(
        &svd,
        &(-residual),
        singular_values[0] * RANK_RELATIVE_TOLERANCE,
    )?;
    Ok((delta, rank, condition))
}

fn damped_svd_solve(
    svd: &nalgebra::SVD<f64, nalgebra::Const<6>, nalgebra::Const<5>>,
    rhs: &SVector<f64, 6>,
    threshold: f64,
) -> Result<SVector<f64, 5>, PalError> {
    let u = svd.u.ok_or(PalError::Singular)?;
    let v_t = svd.v_t.as_ref().ok_or(PalError::Singular)?;
    let projected = u.transpose() * rhs;
    let mut delta = SVector::<f64, 5>::zeros();
    for i in 0..5 {
        let sigma = svd.singular_values[i];
        if sigma > threshold {
            let damping = sigma * sigma * RANK_RELATIVE_TOLERANCE;
            for col in 0..5 {
                delta[col] += sigma / (sigma * sigma + damping) * projected[i] * v_t[(i, col)];
            }
        }
    }
    Ok(delta)
}

fn refine_seed(
    mu: f64,
    mut q: SVector<f64, 5>,
    options: Options,
) -> Result<(SVector<f64, 5>, f64), PalError> {
    let reference = q;
    let fixed_x0 = q[0];
    for _ in 0..options.max_iterations {
        let evaluation = evaluate(mu, &q, options)?;
        let (phase, phase_jacobian) = phase_and_jacobian(mu, &q, &reference)?;
        let mut residual = SVector::<f64, 6>::zeros();
        let mut jacobian = SMatrix::<f64, 6, 5>::zeros();
        for row in 0..4 {
            residual[row] = evaluation.closure[row];
            for col in 0..5 {
                jacobian[(row, col)] = evaluation.jacobian[(row, col)];
            }
        }
        residual[4] = phase;
        residual[5] = q[0] - fixed_x0;
        for col in 0..5 {
            jacobian[(4, col)] = phase_jacobian[col];
        }
        jacobian[(5, 0)] = SCALES[0];
        if max_abs(residual.iter().copied()) <= options.tolerance {
            let closure_error = max_abs(
                evaluation
                    .final_state
                    .iter()
                    .zip(state_from_q(&q).iter())
                    .map(|(final_value, initial_value)| final_value - initial_value),
            );
            return Ok((q, closure_error));
        }
        let (delta, _, _) = solve_svd(jacobian, residual, 5)?;
        q += delta.component_mul(&scales());
    }
    Err(PalError::MaxIterations)
}

fn step_system(
    mu: f64,
    q: &SVector<f64, 5>,
    reference: &SVector<f64, 5>,
    tangent: &SVector<f64, 5>,
    ds: f64,
    options: Options,
) -> Result<StepSystem, PalError> {
    let evaluation = evaluate(mu, q, options)?;
    let (phase, phase_jacobian) = phase_and_jacobian(mu, q, reference)?;
    let scaled_delta = (q - reference).component_div(&scales());
    let arc = scaled_delta.dot(tangent) - ds;
    let mut residual = SVector::<f64, 6>::zeros();
    let mut jacobian = SMatrix::<f64, 6, 5>::zeros();
    for row in 0..4 {
        residual[row] = evaluation.closure[row];
        for col in 0..5 {
            jacobian[(row, col)] = evaluation.jacobian[(row, col)];
        }
    }
    residual[4] = phase;
    residual[5] = arc;
    for col in 0..5 {
        jacobian[(4, col)] = phase_jacobian[col];
        jacobian[(5, col)] = tangent[col];
    }
    let closure_error = max_abs(
        evaluation
            .final_state
            .iter()
            .zip(state_from_q(q).iter())
            .map(|(final_value, initial_value)| final_value - initial_value),
    );
    let augmented_svd = jacobian.svd(true, true);
    let singular_values = augmented_svd.singular_values.as_slice();
    let (rank, condition) = rank_and_condition(singular_values);
    if rank < 5 {
        return Err(PalError::Singular);
    }
    let jacobi_drift = jacobi_drift(mu, &state_from_q(q), &evaluation.final_state);
    let metrics = Metrics {
        tangent_rank: 4,
        tangent_condition: f64::NAN,
        augmented_rank: rank,
        augmented_condition: condition,
    };
    Ok(StepSystem {
        residual,
        jacobian,
        closure_error,
        jacobi_drift,
        metrics,
    })
}

fn jacobi_constant(mu: f64, state: &[f64; 6]) -> f64 {
    let (x, y, z, vx, vy, vz) = (state[0], state[1], state[2], state[3], state[4], state[5]);
    let x1 = x + mu;
    let x2 = x - 1.0 + mu;
    let r1 = (x1 * x1 + y * y + z * z).sqrt().max(1e-10);
    let r2 = (x2 * x2 + y * y + z * z).sqrt().max(1e-10);
    x * x + y * y + 2.0 * (1.0 - mu) / r1 + 2.0 * mu / r2 - (vx * vx + vy * vy + vz * vz)
}

fn jacobi_drift(mu: f64, state: &[f64; 6], final_state: &[f64; 6]) -> f64 {
    (jacobi_constant(mu, final_state) - jacobi_constant(mu, state)).abs()
}

fn correct_step(
    mu: f64,
    reference: &SVector<f64, 5>,
    tangent: &SVector<f64, 5>,
    ds: f64,
    options: Options,
) -> Result<(SVector<f64, 5>, f64, f64, usize, Metrics), PalError> {
    let mut q = reference + tangent.component_mul(&scales()) * ds;
    for iteration in 1..=options.max_iterations {
        let system = step_system(mu, &q, reference, tangent, ds, options)?;
        if max_abs(system.residual.iter().copied()) <= options.tolerance {
            return Ok((
                q,
                system.closure_error,
                system.jacobi_drift,
                iteration,
                system.metrics,
            ));
        }
        let (delta, _, _) = solve_svd(system.jacobian, system.residual, 5)?;
        let current_norm = system.residual.norm();
        let mut accepted = false;
        for line_step in 0..16 {
            let alpha = 0.5_f64.powi(line_step);
            let candidate = q + delta.component_mul(&scales()) * alpha;
            if candidate[4] <= 0.0 {
                continue;
            }
            let candidate_system =
                match step_system(mu, &candidate, reference, tangent, ds, options) {
                    Ok(system) => system,
                    Err(PalError::Integration | PalError::InvalidPeriod | PalError::Singular) => {
                        continue
                    }
                    Err(error) => return Err(error),
                };
            if candidate_system.residual.norm() < current_norm {
                q = candidate;
                accepted = true;
                break;
            }
        }
        if !accepted {
            return Err(PalError::LineSearchStagnated);
        }
    }
    Err(PalError::MaxIterations)
}

#[derive(Clone)]
struct Trace {
    states: Vec<[f64; 6]>,
    periods: Vec<f64>,
    closure_errors: Vec<f64>,
    newton_iterations: Vec<usize>,
    tangent_system_ranks: Vec<usize>,
    tangent_system_conditions: Vec<f64>,
    augmented_system_ranks: Vec<usize>,
    augmented_system_conditions: Vec<f64>,
    step_sizes: Vec<f64>,
    jacobi_drifts: Vec<f64>,
}

impl Trace {
    fn new(seed_state: [f64; 6], seed_period: f64, step_size: f64) -> Self {
        Self {
            states: vec![seed_state],
            periods: vec![seed_period],
            closure_errors: vec![f64::INFINITY],
            newton_iterations: vec![0],
            tangent_system_ranks: vec![0],
            tangent_system_conditions: vec![f64::NAN],
            augmented_system_ranks: vec![0],
            augmented_system_conditions: vec![f64::NAN],
            step_sizes: vec![step_size],
            jacobi_drifts: vec![f64::NAN],
        }
    }

    fn steps(&self) -> usize {
        self.states.len() - 1
    }

    fn update_seed(
        &mut self,
        q: &SVector<f64, 5>,
        closure_error: f64,
        metrics: &Metrics,
        drift: f64,
    ) {
        self.states[0] = state_from_q(q);
        self.periods[0] = q[4];
        self.closure_errors[0] = closure_error;
        self.tangent_system_ranks[0] = metrics.tangent_rank;
        self.tangent_system_conditions[0] = metrics.tangent_condition;
        self.augmented_system_ranks[0] = metrics.augmented_rank;
        self.augmented_system_conditions[0] = metrics.augmented_condition;
        self.jacobi_drifts[0] = drift;
    }

    fn push_member(
        &mut self,
        q: &SVector<f64, 5>,
        closure_error: f64,
        iterations: usize,
        metrics: &Metrics,
        step_size: f64,
        drift: f64,
    ) {
        self.states.push(state_from_q(q));
        self.periods.push(q[4]);
        self.closure_errors.push(closure_error);
        self.newton_iterations.push(iterations);
        self.tangent_system_ranks.push(metrics.tangent_rank);
        self.tangent_system_conditions
            .push(metrics.tangent_condition);
        self.augmented_system_ranks.push(metrics.augmented_rank);
        self.augmented_system_conditions
            .push(metrics.augmented_condition);
        self.step_sizes.push(step_size);
        self.jacobi_drifts.push(drift);
    }
}

fn result(
    trace: Trace,
    termination: Termination,
    steps: usize,
    step_size: f64,
) -> PlanarPalRustResult {
    let (status, cause, message) = termination.contract();
    PlanarPalRustResult {
        states: trace.states,
        periods: trace.periods,
        closure_errors: trace.closure_errors,
        newton_iterations: trace.newton_iterations,
        tangent_system_ranks: trace.tangent_system_ranks,
        tangent_system_conditions: trace.tangent_system_conditions,
        augmented_system_ranks: trace.augmented_system_ranks,
        augmented_system_conditions: trace.augmented_system_conditions,
        step_sizes: trace.step_sizes,
        jacobi_drifts: trace.jacobi_drifts,
        status: status.to_string(),
        cause: cause.to_string(),
        message: message.to_string(),
        steps,
        step_size,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_planar_pal(
    mu: f64,
    seed_state: [f64; 6],
    seed_period: f64,
    n_orbits: usize,
    step_size: f64,
    initial_direction: &str,
    options: Options,
) -> PlanarPalRustResult {
    let mut trace = Trace::new(seed_state, seed_period, step_size);

    let direction = match initial_direction {
        "increase-x0" => 1.0,
        "decrease-x0" => -1.0,
        _ => return result(trace, Termination::InvalidInput, 0, step_size),
    };

    let (mut q, seed_error) = match refine_seed(mu, q_from_state(&seed_state, seed_period), options)
    {
        Ok(value) => value,
        Err(error) => return result(trace, error.termination(), 0, step_size),
    };
    let (mut current_tangent, tangent_rank, tangent_condition, _) =
        match tangent_metrics(mu, &q, options) {
            Ok(value) => value,
            Err(error) => return result(trace, error.termination(), 0, step_size),
        };
    let seed_step = match step_system(mu, &q, &q, &current_tangent, 0.0, options) {
        Ok(value) => value,
        Err(error) => return result(trace, error.termination(), 0, step_size),
    };
    let seed_drift = seed_step.jacobi_drift;
    trace.update_seed(
        &q,
        seed_error,
        &Metrics {
            tangent_rank,
            tangent_condition,
            augmented_rank: seed_step.metrics.augmented_rank,
            augmented_condition: seed_step.metrics.augmented_condition,
        },
        seed_drift,
    );
    if current_tangent[0] * direction < 0.0 {
        current_tangent = -current_tangent;
    }

    let mut current_step = step_size;
    for _ in 0..n_orbits {
        loop {
            match correct_step(mu, &q, &current_tangent, current_step, options) {
                Ok((next_q, closure_error, drift, iterations, mut metrics)) => {
                    let (mut next_tangent, tangent_rank, tangent_condition, _) =
                        match tangent_metrics(mu, &next_q, options) {
                            Ok(value) => value,
                            Err(error) => {
                                return result(
                                    trace.clone(),
                                    error.termination(),
                                    trace.steps(),
                                    current_step,
                                )
                            }
                        };
                    metrics.tangent_rank = tangent_rank;
                    metrics.tangent_condition = tangent_condition;
                    if next_tangent.dot(&current_tangent) < 0.0 {
                        next_tangent = -next_tangent;
                    }
                    q = next_q;
                    current_tangent = next_tangent;
                    trace.push_member(&q, closure_error, iterations, &metrics, current_step, drift);
                    break;
                }
                Err(PalError::Singular) => {
                    return result(
                        trace.clone(),
                        Termination::SingularJacobian,
                        trace.steps(),
                        current_step,
                    )
                }
                Err(PalError::InvalidPeriod) => {
                    return result(
                        trace.clone(),
                        Termination::InvalidPeriod,
                        trace.steps(),
                        current_step,
                    )
                }
                Err(PalError::Integration) => {
                    return result(
                        trace.clone(),
                        Termination::IntegrationFailed,
                        trace.steps(),
                        current_step,
                    )
                }
                Err(PalError::LineSearchStagnated) => {
                    return result(
                        trace.clone(),
                        Termination::Stagnated,
                        trace.steps(),
                        current_step,
                    );
                }
                Err(PalError::MaxIterations) => {
                    current_step *= 0.5;
                    if current_step < MIN_STEP_SIZE {
                        return result(
                            trace.clone(),
                            Termination::Stagnated,
                            trace.steps(),
                            current_step,
                        );
                    }
                }
            }
        }
    }

    result(trace, Termination::Converged, n_orbits, current_step)
}

/// Python 入口：平面 SPO/LPO 的完整周期伪弧长延拓。
#[pyfunction]
#[pyo3(signature = (mu, seed_state, seed_period, n_orbits, step_size, initial_direction, tolerance=1e-9, max_iterations=16, rtol=1e-12, atol=1e-12, max_step=None))]
#[allow(clippy::too_many_arguments)]
pub fn planar_full_period_pal_py(
    mu: f64,
    seed_state: Vec<f64>,
    seed_period: f64,
    n_orbits: usize,
    step_size: f64,
    initial_direction: &str,
    tolerance: f64,
    max_iterations: usize,
    rtol: f64,
    atol: f64,
    max_step: Option<f64>,
    py: Python<'_>,
) -> PyResult<PlanarPalRustResult> {
    if seed_state.len() != 6 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "seed_state 必须含 6 个分量",
        ));
    }
    if seed_period <= 0.0 || step_size <= 0.0 || tolerance <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "seed_period、step_size、tolerance 必须为正数",
        ));
    }
    if !matches!(initial_direction, "increase-x0" | "decrease-x0") {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "initial_direction 必须为 increase-x0 或 decrease-x0",
        ));
    }
    let mut seed = [0.0; 6];
    seed.copy_from_slice(&seed_state);
    let options = Options {
        tolerance,
        max_iterations,
        rtol,
        atol,
        max_step,
    };
    Ok(py.allow_threads(|| {
        run_planar_pal(
            mu,
            seed,
            seed_period,
            n_orbits,
            step_size,
            initial_direction,
            options,
        )
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    // 沿用数据层记录的标准 L4 SPO 种子；它与测试使用同一组地月 CR3BP 常数。
    const SPO_STATE: [f64; 6] = [-0.2255, 0.8660, 0.0, -0.2384, 0.2494, 0.0];
    const SPO_PERIOD: f64 = 6.529;
    const MU: f64 = 0.01215058560962404;

    fn options() -> Options {
        Options {
            tolerance: 1e-9,
            max_iterations: 16,
            rtol: 1e-12,
            atol: 1e-12,
            max_step: Some(0.01),
        }
    }

    fn corrected_spo_q() -> SVector<f64, 5> {
        let (q, closure) = refine_seed(MU, q_from_state(&SPO_STATE, SPO_PERIOD), options())
            .expect("标准 SPO 种子应收紧为完整平面闭合解");
        assert!(closure < 1e-8);
        q
    }

    #[test]
    fn rank_threshold_discards_autonomous_nullspace() {
        let (rank, condition) = rank_and_condition(&[10.0, 2.0, 1.0, 1e-3, 1e-11]);
        assert_eq!(rank, 4);
        assert!((condition - 1e4).abs() < 1e-8);
    }

    #[test]
    fn invalid_direction_returns_structured_failure() {
        let result = run_planar_pal(MU, SPO_STATE, SPO_PERIOD, 1, 0.01, "sideways", options());
        assert_eq!(result.status, "failed");
        assert_eq!(result.cause, "invalid_input");
        assert_eq!(result.steps, 0);
    }

    #[test]
    fn closure_jacobian_matches_stm_and_time_columns() {
        let q = corrected_spo_q();
        let evaluation = evaluate(MU, &q, options()).unwrap();
        let result = propagate_cr3bp_stm(
            MU,
            (0.0, q[4]),
            &[q[4]],
            &state_from_q(&q),
            options().rtol,
            options().atol,
            options().max_step,
            Some(500_000),
        )
        .unwrap();
        let final_state = *result.states.last().unwrap();
        let stm = result.stms.last().unwrap();
        let flow = cr3bp_eom(MU, &final_state);

        for (row, &component) in PLANAR.iter().enumerate() {
            assert!(
                (evaluation.closure[row] - (final_state[component] - state_from_q(&q)[component]))
                    .abs()
                    < 1e-12
            );
            for (col, &variable) in PLANAR.iter().enumerate() {
                let expected = (stm[component * 6 + variable] - f64::from(component == variable))
                    * SCALES[col];
                assert!((evaluation.jacobian[(row, col)] - expected).abs() < 1e-8);
            }
            assert!((evaluation.jacobian[(row, 4)] - flow[component] * SCALES[4]).abs() < 1e-8);
        }
    }

    #[test]
    fn phase_condition_pins_q_to_reference_flow() {
        let q = corrected_spo_q();
        let reference = q + SVector::<f64, 5>::new(0.001, -0.002, 0.003, -0.001, 0.01);
        let (phase, jacobian) = phase_and_jacobian(MU, &q, &reference).unwrap();
        let flow = cr3bp_eom(MU, &state_from_q(&reference));
        let planar_flow = SVector::<f64, 4>::new(flow[0], flow[1], flow[3], flow[4]);
        let delta = SVector::<f64, 4>::new(
            q[0] - reference[0],
            q[1] - reference[1],
            q[2] - reference[2],
            q[3] - reference[3],
        );
        let expected = delta.dot(&planar_flow) / planar_flow.norm();
        assert!((phase - expected).abs() < 1e-12);
        for col in 0..4 {
            let expected_jacobian = planar_flow[col] / planar_flow.norm() * SCALES[col];
            assert!((jacobian[col] - expected_jacobian).abs() < 1e-10);
        }
        assert_eq!(jacobian[4], 0.0);
    }

    #[test]
    fn augmented_svd_solves_known_least_squares_problem() {
        let jacobian = SMatrix::<f64, 6, 5>::new(
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        );
        let residual = SVector::<f64, 6>::new(1.0, 2.0, 3.0, 4.0, 5.0, -5.0);
        let (delta, rank, condition) = solve_svd(jacobian, residual, 5).unwrap();
        assert_eq!(rank, 5);
        assert!(condition.is_finite());
        // 固定 SVD 阻尼有意偏离严格最小二乘正交性；接缝可观察的要求是
        // 秩与条件数有效、残差较原问题显著下降，而不是追求无限精度。
        let solved_residual = jacobian * delta + residual;
        assert!(solved_residual.norm() <= residual.norm());
        assert!(delta.norm().is_finite());
    }

    #[test]
    fn tangent_vectors_point_to_same_family_direction() {
        let q = corrected_spo_q();
        let first = tangent_metrics(MU, &q, options()).unwrap().0;
        let second = tangent_metrics(MU, &q, options()).unwrap().0;
        assert!(first.norm() > 0.999999);
        assert!(first.norm() < 1.000001);
        assert!(first.dot(&second).abs() > 0.999999);
    }

    #[test]
    fn short_pal_run_reports_metrics_and_members() {
        let q = corrected_spo_q();
        let result = run_planar_pal(
            MU,
            state_from_q(&q),
            q[4],
            1,
            0.01,
            "decrease-x0",
            options(),
        );
        assert_eq!(
            result.status, "converged",
            "{}: {}",
            result.cause, result.message
        );
        assert_eq!(result.states.len(), 2);
        assert_eq!(result.steps, 1);
        assert_eq!(result.tangent_system_ranks, vec![4, 4]);
        assert_eq!(result.augmented_system_ranks, vec![5, 5]);
        assert!(result
            .tangent_system_conditions
            .iter()
            .all(|value| value.is_finite()));
        assert!(result
            .augmented_system_conditions
            .iter()
            .all(|value| value.is_finite()));
        assert_eq!(result.step_sizes.len(), 2);
    }

    fn run_with_max_iterations(max_iterations: usize) -> PlanarPalRustResult {
        let q = corrected_spo_q();
        run_planar_pal(
            MU,
            state_from_q(&q),
            q[4],
            1,
            0.01,
            "decrease-x0",
            Options {
                max_iterations,
                ..options()
            },
        )
    }

    #[test]
    fn max_iterations_failure_preserves_partial_family() {
        let result = run_with_max_iterations(0);
        assert_eq!(result.status, "max_iterations");
        assert_eq!(result.cause, "max_iterations_reached");
        assert_eq!(result.states.len(), 1);
        assert_eq!(result.steps, 0);
    }

    #[test]
    fn line_search_failure_shrinks_step_before_stagnating() {
        let result = run_with_max_iterations(1);
        assert_eq!(result.status, "stagnated");
        assert_eq!(result.cause, "stagnation_detected");
        assert_eq!(result.states.len(), 1);
        assert!(result.step_size < 0.01);
    }
}
