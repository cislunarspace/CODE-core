//! 共享全周期伪弧长延拓（PAL）数值内核。
//!
//! 从平面三角平动点适配器（ADR 0028）抽取的维度无关机器：传播与闭合残差
//! 求值、相位规范、SVD 切向量、阻尼最小二乘牛顿、线搜索、步长收缩与秩
//! 诊断。问题形状由 [`PalShape`] 描述（状态分量选择、缩放、闭合圈数），
//! 平面与三维非对称全周期适配器共用本内核（#820，ADR 0060）。XZ 对称
//! 半周期内核（e2m2e-dyn 的 pal_continuation）与三处延拓编排的分工不因
//! 本内核改变。
//!
//! 维度约定：设状态分量数为 m（平面 4，三维 6），未知量 q 维数为 m+1，
//! 前 m 个分量是被选中的状态分量，末位是周期尺度 T。闭合残差在 n·T 上
//! 求值（n 为闭合圈数，1 即 ADR 0028 的原始形式），成员的闭合周期为
//! n·T。

use e2m2e_forces::cr3bp::{cr3bp_eom, propagate_cr3bp_stm};
use nalgebra::{DMatrix, DVector, Dyn, SVD};

/// 奇异值相对阈值：秩判定与阻尼都以其为参照（沿用 ADR 0028 实验标定）。
pub(crate) const RANK_RELATIVE_TOLERANCE: f64 = 1e-8;
/// 步长收缩下限；低于此值仍不收敛即按停滞终止。
pub(crate) const MIN_STEP_SIZE: f64 = 1e-5;

/// 传播与牛顿迭代的数值配置。
#[derive(Clone, Copy)]
pub(crate) struct Options {
    pub(crate) tolerance: f64,
    pub(crate) max_iterations: usize,
    pub(crate) rtol: f64,
    pub(crate) atol: f64,
    pub(crate) max_step: Option<f64>,
}

/// 问题形状：分量选择、缩放、闭合圈数与消息前缀。
pub(crate) struct PalShape {
    /// 参与闭合与相位条件的 6 维状态分量下标（平面为 [0, 1, 3, 4]，
    /// 三维为 [0, 1, 2, 3, 4, 5]）。
    components: Vec<usize>,
    /// q 各分量的缩放（长度为分量数加一，末位对应周期）。
    scales: DVector<f64>,
    /// 闭合圈数 n（正整数）。闭合残差在 n·T 上求值。
    revolutions: usize,
    /// 状态消息前缀，用于区分平面与三维适配器的报错文案。
    label: &'static str,
}

impl PalShape {
    /// 平面形状：状态取 x、y、vx、vy，缩放 (1, 1, 1, 1, 10)（ADR 0028）。
    pub(crate) fn planar(revolutions: usize) -> Self {
        Self {
            components: vec![0, 1, 3, 4],
            scales: DVector::from_vec(vec![1.0, 1.0, 1.0, 1.0, 10.0]),
            revolutions,
            label: "平面",
        }
    }

    /// 三维形状：状态取全部六分量，缩放 (1, 1, 1, 1, 1, 1, 10)。
    pub(crate) fn spatial(revolutions: usize) -> Self {
        Self {
            components: vec![0, 1, 2, 3, 4, 5],
            scales: DVector::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0]),
            revolutions,
            label: "三维",
        }
    }

    fn m(&self) -> usize {
        self.components.len()
    }

    fn q_dim(&self) -> usize {
        self.components.len() + 1
    }

    fn span(&self) -> f64 {
        self.revolutions as f64
    }

    /// 成员闭合周期（n·T）。
    fn closure_period(&self, q: &DVector<f64>) -> f64 {
        self.span() * q[self.m()]
    }
}

/// 终止状态与状态三元组文案（label 区分适配器，既有平面文案逐字保留）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Termination {
    Converged,
    MaxIterations,
    Stagnated,
    SingularJacobian,
    IntegrationFailed,
    InvalidPeriod,
    InvalidInput,
}

impl Termination {
    pub(crate) fn contract(&self, label: &str) -> (&'static str, &'static str, String) {
        match self {
            Self::Converged => ("converged", "none", format!("{label}全周期 PAL 完成")),
            Self::MaxIterations => (
                "max_iterations",
                "max_iterations_reached",
                format!("{label}全周期 PAL 达到最大迭代次数"),
            ),
            Self::Stagnated => (
                "stagnated",
                "stagnation_detected",
                format!("{label}全周期 PAL 步长耗尽或线搜索停滞"),
            ),
            Self::SingularJacobian => (
                "failed",
                "singular_jacobian",
                format!("{label}全周期 PAL 增广 Jacobian 秩不足"),
            ),
            Self::IntegrationFailed => (
                "failed",
                "integration_failed",
                format!("{label}全周期 PAL 的 CR3BP STM 传播失败"),
            ),
            Self::InvalidPeriod => (
                "infeasible",
                "invalid_period",
                format!("{label}全周期 PAL 得到非正周期"),
            ),
            Self::InvalidInput => (
                "failed",
                "invalid_input",
                format!("{label}全周期 PAL 输入无效"),
            ),
        }
    }
}

#[derive(Debug)]
pub(crate) enum PalError {
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
    closure: DVector<f64>,
    jacobian: DMatrix<f64>,
    final_state: [f64; 6],
}

pub(crate) struct Metrics {
    pub(crate) tangent_rank: usize,
    pub(crate) tangent_condition: f64,
    pub(crate) augmented_rank: usize,
    pub(crate) augmented_condition: f64,
}

struct StepSystem {
    residual: DVector<f64>,
    jacobian: DMatrix<f64>,
    closure_error: f64,
    jacobi_drift: f64,
    metrics: Metrics,
}

pub(crate) fn q_from_state(
    state: &[f64; 6],
    closure_period: f64,
    shape: &PalShape,
) -> DVector<f64> {
    let m = shape.m();
    let mut q = DVector::zeros(m + 1);
    for i in 0..m {
        q[i] = state[shape.components[i]];
    }
    q[m] = closure_period / shape.span();
    q
}

pub(crate) fn state_from_q(q: &DVector<f64>, shape: &PalShape) -> [f64; 6] {
    let mut state = [0.0; 6];
    for i in 0..shape.m() {
        state[shape.components[i]] = q[i];
    }
    state
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

fn evaluate(
    mu: f64,
    q: &DVector<f64>,
    shape: &PalShape,
    options: Options,
) -> Result<Evaluation, PalError> {
    let m = shape.m();
    if q[m] <= 0.0 {
        return Err(PalError::InvalidPeriod);
    }
    let span = shape.span() * q[m];
    let state = state_from_q(q, shape);
    let result = propagate_cr3bp_stm(
        mu,
        (0.0, span),
        &[span],
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

    let mut closure = DVector::zeros(m);
    let mut jacobian = DMatrix::zeros(m, m + 1);
    for row in 0..m {
        let component = shape.components[row];
        closure[row] = final_state[component] - state[component];
        for col in 0..m {
            let variable = shape.components[col];
            jacobian[(row, col)] = stm[component * 6 + variable] - f64::from(component == variable);
        }
        jacobian[(row, m)] = shape.span() * flow[component];
    }
    for col in 0..=m {
        for row in 0..m {
            jacobian[(row, col)] *= shape.scales[col];
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
    q: &DVector<f64>,
    reference: &DVector<f64>,
    shape: &PalShape,
) -> Result<(f64, DVector<f64>), PalError> {
    let m = shape.m();
    let reference_state = state_from_q(reference, shape);
    let flow = cr3bp_eom(mu, &reference_state);
    let mut projected_flow = DVector::zeros(m);
    for i in 0..m {
        projected_flow[i] = flow[shape.components[i]];
    }
    let flow_norm = projected_flow.norm();
    if flow_norm == 0.0 {
        return Err(PalError::Singular);
    }
    let mut delta = DVector::zeros(m);
    for i in 0..m {
        delta[i] = q[i] - reference[i];
    }
    let phase = delta.dot(&projected_flow) / flow_norm;
    let mut jacobian = DVector::zeros(m + 1);
    for i in 0..m {
        jacobian[i] = projected_flow[i] / flow_norm * shape.scales[i];
    }
    Ok((phase, jacobian))
}

fn tangent_metrics(
    mu: f64,
    q: &DVector<f64>,
    shape: &PalShape,
    options: Options,
) -> Result<(DVector<f64>, usize, f64, DMatrix<f64>), PalError> {
    let m = shape.m();
    let evaluation = evaluate(mu, q, shape, options)?;
    let (_, phase_jacobian) = phase_and_jacobian(mu, q, q, shape)?;
    let mut system = DMatrix::zeros(m + 1, m + 1);
    for row in 0..m {
        for col in 0..=m {
            system[(row, col)] = evaluation.jacobian[(row, col)];
        }
    }
    for col in 0..=m {
        system[(m, col)] = phase_jacobian[col];
    }
    let svd = system.clone().svd(false, true);
    let singular_values = svd.singular_values.as_slice();
    let (rank, condition) = rank_and_condition(singular_values);
    if rank != m {
        return Err(PalError::Singular);
    }
    let v_t = svd.v_t.ok_or(PalError::Singular)?;
    let mut vector = DVector::zeros(m + 1);
    for col in 0..=m {
        vector[col] = v_t[(m, col)];
    }
    Ok((vector.normalize(), rank, condition, system))
}

fn solve_svd(
    jacobian: DMatrix<f64>,
    residual: &DVector<f64>,
    expected_rank: usize,
) -> Result<(DVector<f64>, usize, f64), PalError> {
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
    svd: &SVD<f64, Dyn, Dyn>,
    rhs: &DVector<f64>,
    threshold: f64,
) -> Result<DVector<f64>, PalError> {
    let u = svd.u.clone().ok_or(PalError::Singular)?;
    let v_t = svd.v_t.clone().ok_or(PalError::Singular)?;
    let count = svd.singular_values.len();
    let cols = v_t.ncols();
    let projected = u.tr_mul(rhs);
    let mut delta = DVector::zeros(cols);
    for i in 0..count {
        let sigma = svd.singular_values[i];
        if sigma > threshold {
            let damping = sigma * sigma * RANK_RELATIVE_TOLERANCE;
            for col in 0..cols {
                delta[col] += sigma / (sigma * sigma + damping) * projected[i] * v_t[(i, col)];
            }
        }
    }
    Ok(delta)
}

fn refine_seed(
    mu: f64,
    mut q: DVector<f64>,
    shape: &PalShape,
    options: Options,
) -> Result<(DVector<f64>, f64), PalError> {
    let m = shape.m();
    let reference = q.clone();
    let fixed_x0 = q[0];
    for _ in 0..options.max_iterations {
        let evaluation = evaluate(mu, &q, shape, options)?;
        let (phase, phase_jacobian) = phase_and_jacobian(mu, &q, &reference, shape)?;
        let mut residual = DVector::zeros(m + 2);
        let mut jacobian = DMatrix::zeros(m + 2, m + 1);
        for row in 0..m {
            residual[row] = evaluation.closure[row];
            for col in 0..=m {
                jacobian[(row, col)] = evaluation.jacobian[(row, col)];
            }
        }
        residual[m] = phase;
        residual[m + 1] = q[0] - fixed_x0;
        for col in 0..=m {
            jacobian[(m, col)] = phase_jacobian[col];
        }
        jacobian[(m + 1, 0)] = shape.scales[0];
        if max_abs(residual.iter().copied()) <= options.tolerance {
            let closure_error = max_abs(
                evaluation
                    .final_state
                    .iter()
                    .zip(state_from_q(&q, shape).iter())
                    .map(|(final_value, initial_value)| final_value - initial_value),
            );
            return Ok((q, closure_error));
        }
        let (delta, _, _) = solve_svd(jacobian, &residual, m + 1)?;
        q += delta.component_mul(&shape.scales);
    }
    Err(PalError::MaxIterations)
}

fn step_system(
    mu: f64,
    q: &DVector<f64>,
    reference: &DVector<f64>,
    tangent: &DVector<f64>,
    ds: f64,
    shape: &PalShape,
    options: Options,
) -> Result<StepSystem, PalError> {
    let m = shape.m();
    let evaluation = evaluate(mu, q, shape, options)?;
    let (phase, phase_jacobian) = phase_and_jacobian(mu, q, reference, shape)?;
    let scaled_delta = (q - reference).component_div(&shape.scales);
    let arc = scaled_delta.dot(tangent) - ds;
    let mut residual = DVector::zeros(m + 2);
    let mut jacobian = DMatrix::zeros(m + 2, m + 1);
    for row in 0..m {
        residual[row] = evaluation.closure[row];
        for col in 0..=m {
            jacobian[(row, col)] = evaluation.jacobian[(row, col)];
        }
    }
    residual[m] = phase;
    residual[m + 1] = arc;
    for col in 0..=m {
        jacobian[(m, col)] = phase_jacobian[col];
        jacobian[(m + 1, col)] = tangent[col];
    }
    let closure_error = max_abs(
        evaluation
            .final_state
            .iter()
            .zip(state_from_q(q, shape).iter())
            .map(|(final_value, initial_value)| final_value - initial_value),
    );
    let augmented_svd = jacobian.clone().svd(true, true);
    let singular_values = augmented_svd.singular_values.as_slice();
    let (rank, condition) = rank_and_condition(singular_values);
    if rank < m + 1 {
        return Err(PalError::Singular);
    }
    let jacobi_drift = jacobi_drift(mu, &state_from_q(q, shape), &evaluation.final_state);
    let metrics = Metrics {
        tangent_rank: m,
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
    reference: &DVector<f64>,
    tangent: &DVector<f64>,
    ds: f64,
    shape: &PalShape,
    options: Options,
) -> Result<(DVector<f64>, f64, f64, usize, Metrics), PalError> {
    let period_index = shape.m();
    let mut q = reference + tangent.component_mul(&shape.scales) * ds;
    for iteration in 1..=options.max_iterations {
        let system = step_system(mu, &q, reference, tangent, ds, shape, options)?;
        if max_abs(system.residual.iter().copied()) <= options.tolerance {
            return Ok((
                q,
                system.closure_error,
                system.jacobi_drift,
                iteration,
                system.metrics,
            ));
        }
        let (delta, _, _) = solve_svd(system.jacobian.clone(), &system.residual, shape.q_dim())?;
        let current_norm = system.residual.norm();
        let mut accepted = false;
        for line_step in 0..16 {
            let alpha = 0.5_f64.powi(line_step);
            let candidate = &q + delta.component_mul(&shape.scales) * alpha;
            if candidate[period_index] <= 0.0 {
                continue;
            }
            let candidate_system =
                match step_system(mu, &candidate, reference, tangent, ds, shape, options) {
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

    #[allow(clippy::too_many_arguments)]
    fn update_seed(
        &mut self,
        q: &DVector<f64>,
        shape: &PalShape,
        closure_error: f64,
        metrics: &Metrics,
        drift: f64,
    ) {
        self.states[0] = state_from_q(q, shape);
        self.periods[0] = shape.closure_period(q);
        self.closure_errors[0] = closure_error;
        self.tangent_system_ranks[0] = metrics.tangent_rank;
        self.tangent_system_conditions[0] = metrics.tangent_condition;
        self.augmented_system_ranks[0] = metrics.augmented_rank;
        self.augmented_system_conditions[0] = metrics.augmented_condition;
        self.jacobi_drifts[0] = drift;
    }

    #[allow(clippy::too_many_arguments)]
    fn push_member(
        &mut self,
        q: &DVector<f64>,
        shape: &PalShape,
        closure_error: f64,
        iterations: usize,
        metrics: &Metrics,
        step_size: f64,
        drift: f64,
    ) {
        self.states.push(state_from_q(q, shape));
        self.periods.push(shape.closure_period(q));
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

/// 共享内核的原始结果（适配器负责包装为各自的 Python 结果类型）。
pub(crate) struct PalTraceResult {
    pub(crate) states: Vec<[f64; 6]>,
    pub(crate) periods: Vec<f64>,
    pub(crate) closure_errors: Vec<f64>,
    pub(crate) newton_iterations: Vec<usize>,
    pub(crate) tangent_system_ranks: Vec<usize>,
    pub(crate) tangent_system_conditions: Vec<f64>,
    pub(crate) augmented_system_ranks: Vec<usize>,
    pub(crate) augmented_system_conditions: Vec<f64>,
    pub(crate) step_sizes: Vec<f64>,
    pub(crate) jacobi_drifts: Vec<f64>,
    pub(crate) status: String,
    pub(crate) cause: String,
    pub(crate) message: String,
    pub(crate) steps: usize,
    pub(crate) step_size: f64,
}

fn result(
    trace: Trace,
    termination: Termination,
    label: &str,
    steps: usize,
    step_size: f64,
) -> PalTraceResult {
    let (status, cause, message) = termination.contract(label);
    PalTraceResult {
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
        message,
        steps,
        step_size,
    }
}

/// 共享全周期 PAL 主循环。`seed_period` 语义为种子轨道的闭合周期
/// （n 圈的总周期），内部换算为周期尺度 T 后进入迭代。
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_full_period_pal(
    mu: f64,
    seed_state: [f64; 6],
    seed_period: f64,
    n_orbits: usize,
    step_size: f64,
    initial_direction: &str,
    options: Options,
    shape: &PalShape,
) -> PalTraceResult {
    let label = shape.label;
    let mut trace = Trace::new(seed_state, seed_period, step_size);

    let direction = match initial_direction {
        "increase-x0" => 1.0,
        "decrease-x0" => -1.0,
        _ => return result(trace, Termination::InvalidInput, label, 0, step_size),
    };

    let (mut q, seed_error) = match refine_seed(
        mu,
        q_from_state(&seed_state, seed_period, shape),
        shape,
        options,
    ) {
        Ok(value) => value,
        Err(error) => return result(trace, error.termination(), label, 0, step_size),
    };
    let (mut current_tangent, tangent_rank, tangent_condition, _) =
        match tangent_metrics(mu, &q, shape, options) {
            Ok(value) => value,
            Err(error) => return result(trace, error.termination(), label, 0, step_size),
        };
    let seed_step = match step_system(mu, &q, &q, &current_tangent, 0.0, shape, options) {
        Ok(value) => value,
        Err(error) => return result(trace, error.termination(), label, 0, step_size),
    };
    let seed_drift = seed_step.jacobi_drift;
    trace.update_seed(
        &q,
        shape,
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
            match correct_step(mu, &q, &current_tangent, current_step, shape, options) {
                Ok((next_q, closure_error, drift, iterations, mut metrics)) => {
                    let (mut next_tangent, tangent_rank, tangent_condition, _) =
                        match tangent_metrics(mu, &next_q, shape, options) {
                            Ok(value) => value,
                            Err(error) => {
                                return result(
                                    trace.clone(),
                                    error.termination(),
                                    label,
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
                    trace.push_member(
                        &q,
                        shape,
                        closure_error,
                        iterations,
                        &metrics,
                        current_step,
                        drift,
                    );
                    break;
                }
                Err(PalError::Singular) => {
                    return result(
                        trace.clone(),
                        Termination::SingularJacobian,
                        label,
                        trace.steps(),
                        current_step,
                    )
                }
                Err(PalError::InvalidPeriod) => {
                    return result(
                        trace.clone(),
                        Termination::InvalidPeriod,
                        label,
                        trace.steps(),
                        current_step,
                    )
                }
                Err(PalError::Integration) => {
                    return result(
                        trace.clone(),
                        Termination::IntegrationFailed,
                        label,
                        trace.steps(),
                        current_step,
                    )
                }
                Err(PalError::LineSearchStagnated) => {
                    return result(
                        trace.clone(),
                        Termination::Stagnated,
                        label,
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
                            label,
                            trace.steps(),
                            current_step,
                        );
                    }
                }
            }
        }
    }

    result(trace, Termination::Converged, label, n_orbits, current_step)
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

    fn corrected_spo_q() -> DVector<f64> {
        let shape = PalShape::planar(1);
        let (q, closure) = refine_seed(
            MU,
            q_from_state(&SPO_STATE, SPO_PERIOD, &shape),
            &shape,
            options(),
        )
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
        let shape = PalShape::planar(1);
        let result = run_full_period_pal(
            MU,
            SPO_STATE,
            SPO_PERIOD,
            1,
            0.01,
            "sideways",
            options(),
            &shape,
        );
        assert_eq!(result.status, "failed");
        assert_eq!(result.cause, "invalid_input");
        assert_eq!(result.steps, 0);
    }

    #[test]
    fn closure_jacobian_matches_stm_and_time_columns() {
        let shape = PalShape::planar(1);
        let q = corrected_spo_q();
        let evaluation = evaluate(MU, &q, &shape, options()).unwrap();
        let span = q[4];
        let result = propagate_cr3bp_stm(
            MU,
            (0.0, span),
            &[span],
            &state_from_q(&q, &shape),
            options().rtol,
            options().atol,
            options().max_step,
            Some(500_000),
        )
        .unwrap();
        let final_state = *result.states.last().unwrap();
        let stm = result.stms.last().unwrap();
        let flow = cr3bp_eom(MU, &final_state);

        for (row, &component) in shape.components.iter().enumerate() {
            assert!(
                (evaluation.closure[row]
                    - (final_state[component] - state_from_q(&q, &shape)[component]))
                    .abs()
                    < 1e-12
            );
            for (col, &variable) in shape.components.iter().enumerate() {
                let expected = (stm[component * 6 + variable] - f64::from(component == variable))
                    * shape.scales[col];
                assert!((evaluation.jacobian[(row, col)] - expected).abs() < 1e-8);
            }
            assert!(
                (evaluation.jacobian[(row, 4)] - flow[component] * shape.scales[4]).abs() < 1e-8
            );
        }
    }

    #[test]
    fn spatial_closure_jacobian_covers_six_components() {
        // 三维形状的闭合与雅可比必须覆盖全部六分量，周期列按闭合圈数缩放：
        // 用 n 等于 2 的形状独立重传播对拍（nT 闭合与 2 倍流列）。
        let shape = PalShape::spatial(2);
        let state = [-0.2255, 0.8660, 0.01, -0.2384, 0.2494, 0.002];
        let q = q_from_state(&state, 2.0 * SPO_PERIOD, &shape);
        let evaluation = evaluate(MU, &q, &shape, options()).unwrap();
        assert_eq!(evaluation.closure.len(), 6);
        assert_eq!(evaluation.jacobian.nrows(), 6);
        assert_eq!(evaluation.jacobian.ncols(), 7);
        let span = 2.0 * q[6];
        let rebuilt = state_from_q(&q, &shape);
        let result = propagate_cr3bp_stm(
            MU,
            (0.0, span),
            &[span],
            &rebuilt,
            options().rtol,
            options().atol,
            options().max_step,
            Some(500_000),
        )
        .unwrap();
        let final_state = *result.states.last().unwrap();
        let stm = result.stms.last().unwrap();
        let flow = cr3bp_eom(MU, &final_state);
        for (row, &component) in shape.components.iter().enumerate() {
            assert!(
                (evaluation.closure[row] - (final_state[component] - rebuilt[component])).abs()
                    < 1e-12
            );
            for (col, &variable) in shape.components.iter().enumerate() {
                let expected = (stm[component * 6 + variable] - f64::from(component == variable))
                    * shape.scales[col];
                assert!((evaluation.jacobian[(row, col)] - expected).abs() < 1e-8);
            }
            assert!(
                (evaluation.jacobian[(row, 6)] - 2.0 * flow[component] * shape.scales[6]).abs()
                    < 1e-8
            );
        }
    }

    #[test]
    fn phase_condition_pins_q_to_reference_flow() {
        let shape = PalShape::planar(1);
        let q = corrected_spo_q();
        let reference = q.clone() + DVector::from_vec(vec![0.001, -0.002, 0.003, -0.001, 0.01]);
        let (phase, jacobian) = phase_and_jacobian(MU, &q, &reference, &shape).unwrap();
        let flow = cr3bp_eom(MU, &state_from_q(&reference, &shape));
        let projected_flow = DVector::from_vec(vec![flow[0], flow[1], flow[3], flow[4]]);
        let mut delta = DVector::zeros(4);
        for i in 0..4 {
            delta[i] = q[i] - reference[i];
        }
        let expected = delta.dot(&projected_flow) / projected_flow.norm();
        assert!((phase - expected).abs() < 1e-12);
        for col in 0..4 {
            let expected_jacobian = projected_flow[col] / projected_flow.norm() * shape.scales[col];
            assert!((jacobian[col] - expected_jacobian).abs() < 1e-10);
        }
        assert_eq!(jacobian[4], 0.0);
    }

    #[test]
    fn augmented_svd_solves_known_least_squares_problem() {
        let jacobian = DMatrix::from_row_slice(
            6,
            5,
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 1.0, //
                1.0, 1.0, 1.0, 1.0, 1.0,
            ],
        );
        let residual = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, -5.0]);
        let (delta, rank, condition) = solve_svd(jacobian.clone(), &residual, 5).unwrap();
        assert_eq!(rank, 5);
        assert!(condition.is_finite());
        // 固定 SVD 阻尼有意偏离严格最小二乘正交性；接缝可观察的要求是
        // 秩与条件数有效、残差较原问题显著下降，而不是追求无限精度。
        let solved_residual = &jacobian * &delta + &residual;
        assert!(solved_residual.norm() <= residual.norm());
        assert!(delta.norm().is_finite());
    }

    #[test]
    fn tangent_vectors_point_to_same_family_direction() {
        let shape = PalShape::planar(1);
        let q = corrected_spo_q();
        let first = tangent_metrics(MU, &q, &shape, options()).unwrap().0;
        let second = tangent_metrics(MU, &q, &shape, options()).unwrap().0;
        assert!(first.norm() > 0.999999);
        assert!(first.norm() < 1.000001);
        assert!(first.dot(&second).abs() > 0.999999);
    }

    #[test]
    fn short_pal_run_reports_metrics_and_members() {
        let shape = PalShape::planar(1);
        let q = corrected_spo_q();
        let result = run_full_period_pal(
            MU,
            state_from_q(&q, &shape),
            shape.closure_period(&q),
            1,
            0.01,
            "decrease-x0",
            options(),
            &shape,
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

    fn run_with_max_iterations(max_iterations: usize) -> PalTraceResult {
        let shape = PalShape::planar(1);
        let q = corrected_spo_q();
        run_full_period_pal(
            MU,
            state_from_q(&q, &shape),
            shape.closure_period(&q),
            1,
            0.01,
            "decrease-x0",
            Options {
                max_iterations,
                ..options()
            },
            &shape,
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
