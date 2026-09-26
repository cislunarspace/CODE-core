//! J2000 惯性系 N 体力模型 + STM 变分方程的 Rust 实现。
//!
//! 从 Python `EphemerisDynamics` 迁移，实现：
//! 1. `compute_nbody_acceleration_and_jacobian`：单次遍历所有天体，同时计算
//!    加速度 a 和雅可比 ∂a/∂r（从 `_compute_acc_and_jacobian` 迁移）
//! 2. `propagate_with_stm`：42 维增广状态传播（6 状态 + 36 STM 展平），
//!    使用 `solve_ivp_capped` + STM 变分方程（dΦ/dt = A·Φ）
//! 3. `propagate_with_state`：6 维纯状态传播，与 `propagate_with_stm` 同用
//!    `solve_ivp_capped`，保证两条路径 states 前 6 维逐位相等（parity）。
//!
//! ## 物理模型
//! 以 `origin` 为坐标原点的受限 N 体问题：
//! ```text
//! a = -μ₀·r/|r|³ - Σᵢ μᵢ·[(r-rᵢ)/|r-rᵢ|³ + rᵢ/|rᵢ|³]
//! ```
//!
//! ## STM 变分方程
//! 增广状态：`[r(3), v(3), Φ(36)]`，共 42 维。
//! ```text
//! dΦ/dt = A · Φ,  A = [0₃ₓ₃  I₃ₓ₃; ∂a/∂r  0₃ₓ₃]
//! ```

use std::cell::RefCell;

use super::compiled::AccelJacobiResult;
use e2m2e_spice::spk_accel;

/// 最小距离钳位（km），防止除零。
pub const MIN_DISTANCE: f64 = 1e-6;

/// N 体力模型配置：描述天体列表和原点天体。
pub struct NBodyConfig {
    /// 天体名称列表（如 `["EARTH", "MOON", "SUN"]`）。
    pub bodies: Vec<String>,
    /// 原点天体名称（如 `"EARTH"`）。
    pub origin: String,
    /// 各天体的 GM（km³/s²），与 `bodies` 一一对应。
    pub gm_values: Vec<f64>,
}

/// 单次遍历所有天体，同时计算加速度和雅可比矩阵。
///
/// 从 Python `EphemerisDynamics._compute_acc_and_jacobian` 迁移。
///
/// # 参数
/// - `config`: N 体力模型配置（天体列表 + 原点 + GM）
/// - `et`: SPICE et 秒
/// - `r_sc`: 航天器位置 [x, y, z] km（相对原点天体）
///
/// # 返回
/// `(acc, jacobian, dadv)`：加速度 [f64; 3]、雅可比 ∂a/∂r [[f64; 3]; 3]、
/// 以及 ∂a/∂v [[f64; 3]; 3]（纯 N 体模型恒为零矩阵）。
pub fn compute_nbody_acceleration_and_jacobian(
    config: &NBodyConfig,
    et: f64,
    r_sc: &[f64; 3],
) -> AccelJacobiResult {
    let mut acc = [0.0_f64; 3];
    let mut jac = [[0.0_f64; 3]; 3];
    let dadv = [[0.0_f64; 3]; 3]; // N 体加速度不依赖速度

    for (body, gm) in config.bodies.iter().zip(config.gm_values.iter()) {
        if body == &config.origin {
            // 中心天体：a = -μ·r/|r|³
            let r_norm_sq = r_sc[0] * r_sc[0] + r_sc[1] * r_sc[1] + r_sc[2] * r_sc[2];
            let r_norm = r_norm_sq.sqrt();
            let r_safe = if r_norm < MIN_DISTANCE {
                MIN_DISTANCE
            } else {
                r_norm
            };
            let inv_r3 = 1.0 / (r_safe * r_safe * r_safe);
            let inv_r5 = inv_r3 / (r_safe * r_safe);

            for i in 0..3 {
                acc[i] -= gm * r_sc[i] * inv_r3;
                for j in 0..3 {
                    let delta = if i == j { 1.0 } else { 0.0 };
                    jac[i][j] -= gm * (delta * inv_r3 - 3.0 * r_sc[i] * r_sc[j] * inv_r5);
                }
            }
        } else {
            // 摄动天体：用 spk_accel 计算加速度 + 雅可比
            let (a_body, jac_body) = spk_accel::third_body_acceleration_and_jacobian(
                et,
                body,
                &config.origin,
                r_sc,
                *gm,
                MIN_DISTANCE,
            )
            .map_err(|e| format!("SPICE query failed for {}: {:?}", body, e))?;

            for i in 0..3 {
                acc[i] += a_body[i];
                for j in 0..3 {
                    jac[i][j] += jac_body[i][j];
                }
            }
        }
    }

    Ok((acc, jac, dadv))
}

/// 只计算 N 体加速度（不算雅可比）。
///
/// 加速度公式与 `compute_nbody_acceleration_and_jacobian` 逐位相同：中心
/// 天体用相同的 `r_safe` 钳位与 `inv_r3`，摄动天体改用
/// `spk_accel::third_body_acceleration`（只查一次 SPICE，不算雅可比——其
/// 加速度公式与 `third_body_acceleration_and_jacobian` 逐字一致，见
/// `spk_accel.rs`）。
///
/// 供纯状态传播 `propagate_with_state` 使用。逐位相同的加速度是纯状态
/// 路径与 STM 路径 states 前 6 维逐位一致（parity）的前提：两条路径同用
/// `solve_ivp_capped`（相同积分循环），当初始步长被 `max_step` 钳位时
/// （星历传播 km 量级状态下的常态），步长序列完全一致。
pub fn compute_nbody_acceleration(
    config: &NBodyConfig,
    et: f64,
    r_sc: &[f64; 3],
) -> Result<[f64; 3], String> {
    let mut acc = [0.0_f64; 3];

    for (body, gm) in config.bodies.iter().zip(config.gm_values.iter()) {
        if body == &config.origin {
            // 中心天体：a = -μ·r/|r|³（与 compute_nbody_acceleration_and_jacobian 逐位相同）
            let r_norm_sq = r_sc[0] * r_sc[0] + r_sc[1] * r_sc[1] + r_sc[2] * r_sc[2];
            let r_norm = r_norm_sq.sqrt();
            let r_safe = if r_norm < MIN_DISTANCE {
                MIN_DISTANCE
            } else {
                r_norm
            };
            let inv_r3 = 1.0 / (r_safe * r_safe * r_safe);
            for i in 0..3 {
                acc[i] -= gm * r_sc[i] * inv_r3;
            }
        } else {
            // 摄动天体：只算加速度，不算雅可比（避免雅可比开销）
            let a_body = spk_accel::third_body_acceleration(
                et,
                body,
                &config.origin,
                r_sc,
                *gm,
                MIN_DISTANCE,
            )
            .map_err(|e| format!("SPICE query failed for {}: {:?}", body, e))?;
            for i in 0..3 {
                acc[i] += a_body[i];
            }
        }
    }

    Ok(acc)
}

/// 6 维纯状态右端项 `[v(3), a(3)]`。
///
/// 加速度由 `compute_nbody_acceleration` 计算，与 `augmented_eom` 中的
/// 加速度逐位相同。供 `propagate_with_state` 使用。
fn state_eom(config: &NBodyConfig, et: f64, state: &[f64; 6]) -> Result<[f64; 6], String> {
    let r_sc = [state[0], state[1], state[2]];
    let v = [state[3], state[4], state[5]];
    let acc = compute_nbody_acceleration(config, et, &r_sc)?;
    Ok([v[0], v[1], v[2], acc[0], acc[1], acc[2]])
}

/// STM 变分方程的右端项：dΦ/dt = A · Φ。
///
/// # 参数
/// - `stm`: 6×6 状态转移矩阵（展平为 36 维）
/// - `jac_da_dr`: 加速度对位置的偏导数（3×3）
///
/// # 返回
/// dΦ/dt（展平为 36 维）
pub fn stm_derivative(
    stm: &[f64; 36],
    jac_da_dr: &[[f64; 3]; 3],
    dadv: &[[f64; 3]; 3],
) -> [f64; 36] {
    let mut dstm = [0.0_f64; 36];
    // dΦ/dt = A · Φ
    // A = [0, I; ∂a/∂r, ∂a/∂v]
    //
    // 对于 Φ 的第 j 列（col_j），有：
    //   d(col_j)/dt = A · col_j
    //   前3行：= col_j[3:6]（来自 I 块）
    //   后3行：= (∂a/∂r) · col_j[0:3] + (∂a/∂v) · col_j[3:6]
    for col in 0..6 {
        // 前 3 行：dstm[row][col] = stm[row+3][col]（即 I·Φ 的上半部分）
        for row in 0..3 {
            dstm[row * 6 + col] = stm[(row + 3) * 6 + col];
        }
        // 后 3 行：dstm[row+3][col] = Σ_k ∂a/∂r[row][k]*stm[k][col] + Σ_k ∂a/∂v[row][k]*stm[(k+3)][col]
        for row in 0..3 {
            let mut sum = 0.0;
            for k in 0..3 {
                sum += jac_da_dr[row][k] * stm[k * 6 + col];
                sum += dadv[row][k] * stm[(k + 3) * 6 + col];
            }
            dstm[(row + 3) * 6 + col] = sum;
        }
    }
    dstm
}

/// 42 维增广状态（6 状态 + 36 STM 展平）的右端项。
///
/// 供 `solve_ivp_capped` 使用的回调函数。
///
/// # 增广状态布局
/// - `[0:3]`：位置 r（km）
/// - `[3:6]`：速度 v（km/s）
/// - `[6:42]`：STM Φ（6×6 展平，行优先）
///
/// # 返回
/// `[v(3), a(3), dΦ/dt(36)]`，共 42 维。
pub fn augmented_eom(
    config: &NBodyConfig,
    et: f64,
    augmented_state: &[f64; 42],
) -> Result<[f64; 42], String> {
    let r_sc = [augmented_state[0], augmented_state[1], augmented_state[2]];
    let v = [augmented_state[3], augmented_state[4], augmented_state[5]];
    let mut stm = [0.0_f64; 36];
    stm.copy_from_slice(&augmented_state[6..42]);

    // 计算加速度和雅可比
    let (acc, jac_da_dr, dadv) = compute_nbody_acceleration_and_jacobian(config, et, &r_sc)?;

    // STM 变分方程
    let dstm = stm_derivative(&stm, &jac_da_dr, &dadv);

    // 组装增广状态导数
    let mut result = [0.0_f64; 42];
    // dr/dt = v
    result[0] = v[0];
    result[1] = v[1];
    result[2] = v[2];
    // dv/dt = a
    result[3] = acc[0];
    result[4] = acc[1];
    result[5] = acc[2];
    // dΦ/dt
    result[6..42].copy_from_slice(&dstm);

    Ok(result)
}

/// 传播结果：状态轨迹 + STM 序列。
pub struct PropagationResult {
    /// 各 `t_eval` 时刻的状态向量 `[x, y, z, vx, vy, vz]`（km, km/s）。
    pub states: Vec<[f64; 6]>,
    /// 各 `t_eval` 时刻的 STM 矩阵（6×6 展平，行优先）。
    pub stms: Vec<[f64; 36]>,
    /// 实际输出的时间点。
    pub times: Vec<f64>,
}

/// 42 维增广状态传播（状态 + STM）。
///
/// 初始 STM 设为单位矩阵，拼接为 42 维增广状态后用 DOP853 积分。
/// 步长误差控制只统计前 6 维（`error_dim = 6`），避免 STM 分量主导步长选择。
///
/// # 参数
/// - `config`: N 体力模型配置
/// - `t_span`: 积分区间 `(t_start, t_end)`（SPICE et 秒）
/// - `t_eval`: 输出时间点（必须在 `t_span` 内且单调递增）
/// - `initial_state`: 初始状态 `[x, y, z, vx, vy, vz]`（km, km/s）
/// - `rtol`, `atol`: 积分容差
/// - `max_step`: 最大步长（秒），`None` 则不限制
/// - `max_steps`: 最大步数，`None` 则用默认上限
///
/// # 返回
/// `PropagationResult`：每个 `t_eval` 对应的状态和 STM。
///
/// # 错误
/// - 初值处右端项求值失败（如 SPICE 内核缺失导致第三体位置查询失败）；
/// - 积分提前退出导致输出点数少于 `t_eval.len()`（如步长塌缩、中途力模型失败）。
///
/// 两类错误的消息都带 `cause:` 段，按状态查询分类（内核未加载 / 内核覆盖
/// 不足 / 星历缓存窗口外 / 步长塌缩或步数上限），覆盖与缓存类附带底层定位
/// 信息（见 `truncation_cause`）。
#[allow(clippy::too_many_arguments)]
pub fn propagate_with_stm(
    config: &NBodyConfig,
    t_span: (f64, f64),
    t_eval: &[f64],
    initial_state: &[f64; 6],
    rtol: f64,
    atol: f64,
    max_step: Option<f64>,
    max_steps: Option<usize>,
) -> Result<PropagationResult, String> {
    use e2m2e_propagation::solve_ivp::solve_ivp_capped;

    // 构造 42 维增广状态：[r(3), v(3), Φ(36)]
    let mut augmented0 = [0.0_f64; 42];
    augmented0[..6].copy_from_slice(initial_state);
    // 单位 STM
    for i in 0..6 {
        augmented0[6 + i * 6 + i] = 1.0;
    }

    // 预检：初值处右端项必须可求值。SPICE 内核缺失时第三体位置查询
    // 在此处确定性失败，避免积分静默返回截断结果。
    augmented_eom(config, t_span.0, &augmented0)
        .map_err(|e| initial_failure_message(t_span.0, e))?;

    let h_max = max_step.unwrap_or(f64::INFINITY);
    let s_max = max_steps.unwrap_or(500_000);

    // 力模型回调错误经 `RefCell` 捕获：`solve_ivp_capped` 在回调 `Err` 时
    // `break` 并丢弃原始错误（见 `solve_ivp.rs`），只有在此处留底才能在
    // 完整性校验失败时给出真实 cause，而不是笼统的"likely cause"。
    let eom_error: RefCell<Option<EomFailure>> = RefCell::new(None);

    // 积分
    let sol = solve_ivp_capped(
        |t: f64, y: &[f64]| -> Result<Vec<f64>, String> {
            let mut arr = [0.0_f64; 42];
            arr.copy_from_slice(y);
            match augmented_eom(config, t, &arr) {
                Ok(result) => Ok(result.to_vec()),
                Err(e) => {
                    *eom_error.borrow_mut() = Some(EomFailure {
                        et: t,
                        message: e.clone(),
                    });
                    Err(e)
                }
            }
        },
        t_span,
        &augmented0,
        t_eval,
        rtol,
        atol,
        h_max,
        s_max,
        Some(6), // 只统计前 6 维误差
    );

    // 完整性校验：solve_ivp_capped 在力模型失败/步长塌缩时会提前退出，
    // 输出点数不足即视为传播失败，不允许静默截断。cause 由回调留底的
    // 力模型错误与当前进程状态共同分类（见 `truncation_cause`）。
    if sol.len() != t_eval.len() {
        return Err(truncation_failure_message(
            sol.len(),
            t_eval.len(),
            t_span,
            eom_error.borrow().as_ref(),
        ));
    }

    // 分离状态和 STM
    let mut states = Vec::with_capacity(sol.len());
    let mut stms = Vec::with_capacity(sol.len());
    let mut times = Vec::with_capacity(sol.len());

    for (i, y) in sol.iter().enumerate() {
        let mut s = [0.0_f64; 6];
        s.copy_from_slice(&y[..6]);
        states.push(s);

        let mut stm = [0.0_f64; 36];
        stm.copy_from_slice(&y[6..42]);
        stms.push(stm);

        times.push(t_eval[i]);
    }

    Ok(PropagationResult {
        states,
        stms,
        times,
    })
}

/// 纯状态传播结果（不含 STM）。
pub struct StatePropagationResult {
    /// 各 `t_eval` 时刻的状态向量 `[x, y, z, vx, vy, vz]`（km, km/s）。
    pub states: Vec<[f64; 6]>,
    /// 实际输出的时间点（与 `t_eval` 一一对应）。
    pub times: Vec<f64>,
}

/// 6 维纯状态传播（不含 STM）。
///
/// 与 `propagate_with_stm` 同用 `solve_ivp_capped`（DOP853），因此两条路径
/// 走同一积分循环：相同的 PD78 Butcher 表、相同的 `error_dim = Some(6)`
/// 误差统计、相同的 `suggest_next_step` 步长建议。星历传播的状态为 km
/// 量级，自适应初始步长远大于 `max_step`，故两条路径的初始步长都被
/// `max_step` 钳位为同一值，后续步长序列完全一致——states 前 6 维与
/// `propagate_with_stm` 逐位相等（parity）。
///
/// 力模型回调 `state_eom` 只算加速度（`compute_nbody_acceleration`），
/// 不算雅可比，比 42 维 STM 路径省去雅可比与 STM 变分方程的开销。
/// `solve_ivp_capped` 内部按 `t_span` 方向带符号步进，支持双向积分
/// （`t_span.1 < t_span.0`）。
///
/// # 参数
/// - `config`: N 体力模型配置
/// - `t_span`: 积分区间 `(t_start, t_end)`（SPICE et 秒）
/// - `t_eval`: 输出时间点（须在 `t_span` 内，方向与 `t_span` 一致）
/// - `initial_state`: 初始状态 `[x, y, z, vx, vy, vz]`（km, km/s）
/// - `rtol`, `atol`: 积分容差
/// - `max_step`: 最大步长（秒），`None` 则不限制
/// - `max_steps`: 最大步数，`None` 则用默认上限
///
/// # 返回
/// `StatePropagationResult`：每个 `t_eval` 对应的状态和时间。
///
/// # 错误
/// - 初值处右端项求值失败（如 SPICE 内核缺失）；
/// - 积分提前退出导致输出点数少于 `t_eval.len()`（不允许静默截断）。
///
/// 两类错误的消息都带 `cause:` 段，按状态查询分类（内核未加载 / 内核覆盖
/// 不足 / 星历缓存窗口外 / 步长塌缩或步数上限），覆盖与缓存类附带底层定位
/// 信息（见 `truncation_cause`）。
#[allow(clippy::too_many_arguments)]
pub fn propagate_with_state(
    config: &NBodyConfig,
    t_span: (f64, f64),
    t_eval: &[f64],
    initial_state: &[f64; 6],
    rtol: f64,
    atol: f64,
    max_step: Option<f64>,
    max_steps: Option<usize>,
) -> Result<StatePropagationResult, String> {
    use e2m2e_propagation::solve_ivp::solve_ivp_capped;

    // 预检：初值处右端项必须可求值（与 propagate_with_stm 一致）。
    state_eom(config, t_span.0, initial_state).map_err(|e| initial_failure_message(t_span.0, e))?;

    let h_max = max_step.unwrap_or(f64::INFINITY);
    let s_max = max_steps.unwrap_or(500_000);

    // 力模型回调错误经 `RefCell` 捕获（与 propagate_with_stm 一致）。
    let eom_error: RefCell<Option<EomFailure>> = RefCell::new(None);

    let sol = solve_ivp_capped(
        |t: f64, y: &[f64]| -> Result<Vec<f64>, String> {
            let mut s = [0.0_f64; 6];
            s.copy_from_slice(&y[..6]);
            match state_eom(config, t, &s) {
                Ok(result) => Ok(result.to_vec()),
                Err(e) => {
                    *eom_error.borrow_mut() = Some(EomFailure {
                        et: t,
                        message: e.clone(),
                    });
                    Err(e)
                }
            }
        },
        t_span,
        initial_state,
        t_eval,
        rtol,
        atol,
        h_max,
        s_max,
        Some(6), // 全 6 维统计误差，与 propagate_with_stm 的前 6 维语义一致
    );

    // 完整性校验（与 propagate_with_stm 一致）。
    if sol.len() != t_eval.len() {
        return Err(truncation_failure_message(
            sol.len(),
            t_eval.len(),
            t_span,
            eom_error.borrow().as_ref(),
        ));
    }

    let mut states = Vec::with_capacity(sol.len());
    let mut times = Vec::with_capacity(sol.len());
    for (i, y) in sol.iter().enumerate() {
        let mut s = [0.0_f64; 6];
        s.copy_from_slice(&y[..6]);
        states.push(s);
        times.push(t_eval[i]);
    }

    Ok(StatePropagationResult { states, times })
}

/// 判断 SPK 内核池是否为空：区分"内核未加载"与"内核覆盖不足"。
///
/// `ktotal` 查询失败时按"已加载"处理——查询本身异常不该被翻译成
/// "内核未加载"的误导性 cause。
fn spk_kernels_loaded() -> bool {
    e2m2e_spice::spice_ffi::ktotal("SPK")
        .map(|n| n > 0)
        .unwrap_or(true)
}

/// 一次力模型回调失败：失败时刻 + 底层错误文本。
///
/// `et` 用于按状态判定失败是否落在星历缓存窗口外；`message` 只是证据文本，
/// 分类不读它（见 `truncation_cause`）。
struct EomFailure {
    et: f64,
    message: String,
}

/// 把一次传播提前退出分类成带定位信息的 cause 文本。
///
/// 分类**只看进程状态查询**，不匹配错误文本（ADR 0020：不用错误码字符串
/// 做翻译；同型的 `ktotal` 预检即该决策的先例）：缓存窗口取自
/// `ephem_cache::enabled_span()`，内核池状态取自 `ktotal`。底层错误文本仅
/// 作为证据拼在末尾。`spk_loaded` / `cache_window` 参数化以保证可纯测试。
///
/// 四类：缓存窗口外（附查询时刻与窗口区间）/ 缓存键未注册 / 内核未加载 /
/// 内核覆盖不足；`None` 表示积分器提前退出时力模型从未报错（步长塌缩或
/// 步数上限）。
fn truncation_cause(
    failure: Option<&EomFailure>,
    spk_loaded: bool,
    cache_window: Option<(f64, f64)>,
) -> String {
    let Some(failure) = failure else {
        return "step size collapsed or max steps reached \
                (integrator exited early, force model reported no error)"
            .to_string();
    };
    let cause = match cache_window {
        // 缓存已启用：查询失败只可能是键未注册（窗口内）或越界（窗口外）。
        Some((start, end)) if !(start..=end).contains(&failure.et) => {
            return format!(
                "ephem cache query outside cached window \
                 (et {:.3}, window [{:.3}, {:.3}]): {}",
                failure.et, start, end, failure.message
            );
        }
        Some(_) => "ephem cache lookup failed (key not registered)",
        None if spk_loaded => {
            "SPICE ephemeris query failed (insufficient kernel coverage or missing data)"
        }
        None => "SPICE kernels not loaded (SPK kernel pool is empty)",
    };
    format!("{cause}: {}", failure.message)
}

/// 取当前进程状态（SPK 内核池、星历缓存窗口）后分类。
fn classify_failure(failure: Option<&EomFailure>) -> String {
    truncation_cause(
        failure,
        spk_kernels_loaded(),
        e2m2e_spice::ephem_cache::enabled_span(),
    )
}

/// 初值预检失败的完整消息（两个 propagate 函数共用，保证口径一致）。
fn initial_failure_message(et: f64, message: String) -> String {
    format!(
        "initial RHS evaluation failed at t={et}; cause: {}",
        classify_failure(Some(&EomFailure { et, message }))
    )
}

/// 完整性校验失败的完整消息（两个 propagate 函数共用，保证口径一致）。
fn truncation_failure_message(
    got: usize,
    expected: usize,
    t_span: (f64, f64),
    failure: Option<&EomFailure>,
) -> String {
    format!(
        "propagation truncated: got {got} of {expected} time points \
         (t_span=({:.3}, {:.3})); cause: {}",
        t_span.0,
        t_span.1,
        classify_failure(failure)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 加载仓库 kernels/ 下可用的内核。返回是否加载到了 SPK 星历（.bsp）。
    ///
    /// de430/de440s.bsp 被 .gitignore 排除，干净 clone（如 CI runner）上没有；
    /// 依赖 MOON/SUN 星历的测试须据此跳过。
    fn load_kernels() -> bool {
        let kernel_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .unwrap()
            .join("kernels");
        let mut has_spk = false;
        for name in [
            "naif0012.tls",
            "pck00010.tpc",
            "de430.bsp",
            "de440s.bsp",
            "earth_latest_high_prec.bpc",
            "SPICEEarthPredictedKernel.bpc",
            "SPICELunaFrameKernel.tf",
            "SPICELunaCurrentKernel.bpc",
        ] {
            let path = kernel_dir.join(name);
            if path.exists() {
                // 按文件存在判断跳过；furnish 失败保持原有的容忍（let _），
                // 但 has_spk 仍置真——星历损坏时测试会照常运行并大声失败，
                // 不会被守卫掩盖。
                if name.ends_with(".bsp") {
                    has_spk = true;
                }
                let _ = cspice::data::furnish(path.to_string_lossy().to_string());
            }
        }
        has_spk
    }

    /// 地月日配置。
    fn earth_moon_sun_config() -> NBodyConfig {
        NBodyConfig {
            bodies: vec!["EARTH".to_string(), "MOON".to_string(), "SUN".to_string()],
            origin: "EARTH".to_string(),
            gm_values: vec![398600.435436, 4902.800066, 132712440041.9394],
        }
    }

    // =========================================================
    // 测试 1：中心天体引力 + 雅可比数值验证
    // =========================================================

    /// 纯中心引力测试：只用 EARTH，验证对称性。
    #[test]
    fn central_body_acceleration_basic() {
        load_kernels();
        // 只用 EARTH 作为中心天体（无摄动体），验证纯中心引力的对称性
        let config = NBodyConfig {
            bodies: vec!["EARTH".to_string()],
            origin: "EARTH".to_string(),
            gm_values: vec![398600.435436],
        };
        let et = 0.0;
        let r = [7000.0, 0.0, 0.0]; // LEO

        let (acc, jac, _dadv) = compute_nbody_acceleration_and_jacobian(&config, et, &r).unwrap();

        // a_x ≈ -μ/r² = -398600/7000² ≈ -8.13e-3 km/s²
        assert!(acc[0] < 0.0, "中心引力应指向 -x");
        assert!(acc[0].abs() > 1e-3, "LEO 处引力应有量级 1e-3 km/s²");
        assert!(acc[1].abs() < 1e-14, "对称性：y 方向应为 0");
        assert!(acc[2].abs() < 1e-14, "对称性：z 方向应为 0");

        // 雅可比数值验证：用有限差分
        let h = 1e-5; // km
        for dim in 0..3 {
            let mut r_plus = r;
            let mut r_minus = r;
            r_plus[dim] += h;
            r_minus[dim] -= h;

            let (acc_plus, _, _) =
                compute_nbody_acceleration_and_jacobian(&config, et, &r_plus).unwrap();
            let (acc_minus, _, _) =
                compute_nbody_acceleration_and_jacobian(&config, et, &r_minus).unwrap();

            for i in 0..3 {
                let fd = (acc_plus[i] - acc_minus[i]) / (2.0 * h);
                let analytical = jac[i][dim];
                let err = (fd - analytical).abs();
                let scale = fd.abs().max(analytical.abs()).max(1e-20);
                assert!(
                    err / scale < 1e-6,
                    "jac[{}][{}] 数值={}, 解析={}, 相对误差={}",
                    i,
                    dim,
                    fd,
                    analytical,
                    err / scale
                );
            }
        }
    }

    // =========================================================
    // 测试 2：第三体摄动 + 雅可比数值验证
    // =========================================================

    /// 第三体摄动雅可比数值验证（含 EARTH+MOON+SUN）。
    #[test]
    fn third_body_jacobian_numerical() {
        if !load_kernels() {
            eprintln!("跳过：无 SPK 星历内核（de430/de440s.bsp 未被 git 跟踪）");
            return;
        }
        let config = earth_moon_sun_config();
        let et = 0.0; // J2000
        let r = [7000.0, 0.0, 0.0]; // LEO

        let (_, jac, _dadv) = compute_nbody_acceleration_and_jacobian(&config, et, &r).unwrap();

        // 有限差分验证（中心天体 + 第三体一起测）
        let h = 1e-3; // km
        for dim in 0..3 {
            let mut r_plus = r;
            let mut r_minus = r;
            r_plus[dim] += h;
            r_minus[dim] -= h;

            let (acc_plus, _, _) =
                compute_nbody_acceleration_and_jacobian(&config, et, &r_plus).unwrap();
            let (acc_minus, _, _) =
                compute_nbody_acceleration_and_jacobian(&config, et, &r_minus).unwrap();

            for i in 0..3 {
                let fd = (acc_plus[i] - acc_minus[i]) / (2.0 * h);
                let analytical = jac[i][dim];
                let err = (fd - analytical).abs();
                let scale = fd.abs().max(analytical.abs()).max(1e-20);
                // 容差放宽到 5e-2（有限差分精度 + SPICE 插值误差）
                assert!(
                    err / scale < 5e-2,
                    "jac[{}][{}] 数值={}, 解析={}, 相对误差={}",
                    i,
                    dim,
                    fd,
                    analytical,
                    err / scale
                );
            }
        }
    }

    // =========================================================
    // 测试 3：STM 导数矩阵维度和对称性
    // =========================================================

    #[test]
    fn stm_derivative_dimensions() {
        // 构造单位 STM
        let mut stm = [0.0_f64; 36];
        for i in 0..6 {
            stm[i * 6 + i] = 1.0;
        }

        // 构造一个简单的雅可比
        let jac_da_dr = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let dadv = [[0.0_f64; 3]; 3];

        let dstm = stm_derivative(&stm, &jac_da_dr, &dadv);

        // 验证：dΦ/dt = A·Φ = A·I = A
        // A = [0₃ₓ₃  I₃ₓ₃]
        //     [U₃ₓ₃  0₃ₓ₃]
        //
        // 前 3 行：dstm[row][col] = δ(col, row+3)（来自 I 块）
        for row in 0..3 {
            for col in 0..6 {
                let expected = if col == row + 3 { 1.0 } else { 0.0 };
                assert!(
                    (dstm[row * 6 + col] - expected).abs() < 1e-14,
                    "dstm[{}][{}] = {} vs {}",
                    row,
                    col,
                    dstm[row * 6 + col],
                    expected
                );
            }
        }

        // 后 3 行：dstm[row+3][col] = U[row][col]（因为 Φ=I，U·I = U，dadv=0）
        // 注意：A 的后 3 行前 3 列是 U，后 3 列是 0
        for row in 0..3 {
            for col in 0..6 {
                let expected = if col < 3 {
                    jac_da_dr[row][col] // U 块
                } else {
                    0.0 // 0 块
                };
                assert!(
                    (dstm[(row + 3) * 6 + col] - expected).abs() < 1e-14,
                    "dstm[{}][{}] = {} vs {}",
                    row + 3,
                    col,
                    dstm[(row + 3) * 6 + col],
                    expected
                );
            }
        }
    }

    // =========================================================
    // 测试 4：42 维增广右端项基本正确性
    // =========================================================

    #[test]
    fn augmented_eom_basic() {
        if !load_kernels() {
            eprintln!("跳过：无 SPK 星历内核（de430/de440s.bsp 未被 git 跟踪）");
            return;
        }
        let config = earth_moon_sun_config();
        let et = 0.0;

        // 构造增广状态：LEO 位置 + 圆轨道速度 + 单位 STM
        let mut state = [0.0_f64; 42];
        state[0] = 7000.0; // x = 7000 km
        state[4] = 7.5; // vy ≈ 圆轨道速度 km/s
                        // 单位 STM
        for i in 0..6 {
            state[6 + i * 6 + i] = 1.0;
        }

        let result = augmented_eom(&config, et, &state).unwrap();

        // dr/dt = v: vx=0, vy=7.5, vz=0
        assert!((result[0] - 0.0).abs() < 1e-10, "dx/dt = vx = 0");
        assert!((result[1] - 7.5).abs() < 1e-10, "dy/dt = vy = 7.5");
        assert!((result[2] - 0.0).abs() < 1e-10, "dz/dt = vz = 0");

        // dv/dt = a
        assert!(result[3] < 0.0, "ax 应为负（指向地心）");
        assert!(result[3].abs() > 1e-3, "LEO 加速度量级");

        // dΦ/dt 应非零（因为 A ≠ 0）
        let dstm_norm: f64 = result[6..42].iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(dstm_norm > 0.0, "dΦ/dt 不应全为零");
    }

    // =========================================================
    // 测试 5：STM 传播与有限差分对比（端到端）
    // =========================================================

    /// STM 传播与有限差分对比（端到端）。
    ///
    /// 传播 1 个轨道周期（~5400 秒），让 STM 明显偏离单位矩阵，
    /// 然后用有限差分验证 STM 的 ∂r(T)/∂r(0) 和 ∂r(T)/∂v(0)。
    #[test]
    fn stm_propagation_vs_finite_difference() {
        if !load_kernels() {
            eprintln!("跳过：无 SPK 星历内核（de430/de440s.bsp 未被 git 跟踪）");
            return;
        }
        let config = earth_moon_sun_config();

        let et0 = 0.0;
        let r0 = [7000.0, 0.0, 0.0];
        let v0 = [0.0, 7.5, 0.0];
        let state0 = [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]];
        let dt = 5400.0;
        let et1 = et0 + dt;
        let t_eval = vec![et0, et1];

        // 正常传播（带 STM）
        let result = propagate_with_stm(
            &config,
            (et0, et1),
            &t_eval,
            &state0,
            1e-12,
            1e-12,
            None,
            None,
        )
        .unwrap();
        assert_eq!(result.states.len(), 2, "应输出 2 个时间点");
        let r1 = result.states[1];
        let stm_flat = &result.stms[1];

        // 有限差分验证
        let h = 1.0; // km（位置扰动）
        let h_vel = 0.001; // km/s（速度扰动）

        // 验证 ∂r(T)/∂r(0)
        for dim in 0..2 {
            let mut state0_pert = state0;
            state0_pert[dim] += h;

            let result_pert = propagate_with_stm(
                &config,
                (et0, et1),
                &t_eval,
                &state0_pert,
                1e-12,
                1e-12,
                None,
                None,
            )
            .unwrap();
            let r1_pert = result_pert.states[1];

            for row in 0..2 {
                let stm_val = stm_flat[row * 6 + dim];
                let fd_val = (r1_pert[row] - r1[row]) / h;
                let err = (stm_val - fd_val).abs();
                let scale = fd_val.abs().max(1e-10);
                assert!(
                    err / scale < 5e-2,
                    "STM[{}][{}] 解析={}, 有限差分={}, 相对误差={}",
                    row,
                    dim,
                    stm_val,
                    fd_val,
                    err / scale
                );
            }
        }

        // 验证 ∂r(T)/∂v(0)
        for dim in 0..2 {
            let mut state0_pert = state0;
            state0_pert[3 + dim] += h_vel;

            let result_pert = propagate_with_stm(
                &config,
                (et0, et1),
                &t_eval,
                &state0_pert,
                1e-12,
                1e-12,
                None,
                None,
            )
            .unwrap();
            let r1_pert = result_pert.states[1];

            for row in 0..2 {
                let stm_val = stm_flat[row * 6 + (dim + 3)];
                let fd_val = (r1_pert[row] - r1[row]) / h_vel;
                let err = (stm_val - fd_val).abs();
                let scale = fd_val.abs().max(1e-10);
                assert!(
                    err / scale < 0.5,
                    "STM[{}][{}] 解析={}, 有限差分={}, 相对误差={}",
                    row,
                    dim + 3,
                    stm_val,
                    fd_val,
                    err / scale
                );
            }
        }
    }

    // =========================================================
    // 测试 6：propagate_with_stm 高层接口
    // =========================================================

    /// propagate_with_stm 端到端测试：传播 LEO 一个周期，验证 STM 与有限差分一致。
    #[test]
    fn propagate_with_stm_leo_one_period() {
        if !load_kernels() {
            eprintln!("跳过：无 SPK 星历内核（de430/de440s.bsp 未被 git 跟踪）");
            return;
        }
        let config = earth_moon_sun_config();

        let et0 = 0.0;
        let r0 = [7000.0, 0.0, 0.0];
        let v0 = [0.0, 7.5, 0.0];
        let state0 = [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]];

        let dt = 5400.0; // 一个轨道周期
        let et1 = et0 + dt;
        let t_eval = vec![et0, et1];

        let result = propagate_with_stm(
            &config,
            (et0, et1),
            &t_eval,
            &state0,
            1e-12,
            1e-12,
            None,
            None,
        )
        .unwrap();

        assert_eq!(result.states.len(), 2, "应输出 2 个时间点");
        assert_eq!(result.stms.len(), 2);

        // 初始 STM 应为单位矩阵
        let stm0 = &result.stms[0];
        for i in 0..6 {
            for j in 0..6 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (stm0[i * 6 + j] - expected).abs() < 1e-14,
                    "初始 STM[{}][{}] = {} vs {}",
                    i,
                    j,
                    stm0[i * 6 + j],
                    expected
                );
            }
        }

        // STM 传播后应明显偏离单位矩阵
        let stm1 = &result.stms[1];
        let stm_deviation: f64 = stm1
            .iter()
            .enumerate()
            .map(|(k, &v)| {
                let i = k / 6;
                let j = k % 6;
                let expected = if i == j { 1.0 } else { 0.0 };
                (v - expected).powi(2)
            })
            .sum::<f64>()
            .sqrt();
        assert!(
            stm_deviation > 0.01,
            "STM 传播后应明显偏离单位矩阵，偏差={}",
            stm_deviation
        );

        // 用 propagate_with_stm 做有限差分验证 STM
        let r1 = result.states[1];
        let h = 1.0; // km

        for dim in 0..2 {
            let mut state0_pert = state0;
            state0_pert[dim] += h;

            let result_pert = propagate_with_stm(
                &config,
                (et0, et1),
                &t_eval,
                &state0_pert,
                1e-12,
                1e-12,
                None,
                None,
            )
            .unwrap();

            let r1_pert = result_pert.states[1];

            for row in 0..2 {
                let stm_val = stm1[row * 6 + dim];
                let fd_val = (r1_pert[row] - r1[row]) / h;
                let err = (stm_val - fd_val).abs();
                let scale = fd_val.abs().max(1e-10);
                assert!(
                    err / scale < 5e-2,
                    "propagate_with_stm STM[{}][{}] 解析={}, 有限差分={}, 相对误差={}",
                    row,
                    dim,
                    stm_val,
                    fd_val,
                    err / scale
                );
            }
        }
    }

    // =========================================================
    // 测试 7：无星历天体必须返回 Err，不允许静默截断
    // =========================================================

    /// 第三体无星历数据时，propagate_with_stm 必须返回带上下文的 Err，
    /// 而不是静默返回只有 t0 的截断结果。
    #[test]
    fn propagate_with_stm_unknown_body_returns_err() {
        load_kernels();
        let config = NBodyConfig {
            bodies: vec!["EARTH".to_string(), "FAKEBODY".to_string()],
            origin: "EARTH".to_string(),
            gm_values: vec![398600.435436, 1.0],
        };

        let state0 = [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0];
        let t_eval = vec![0.0, 100.0];

        let result = propagate_with_stm(
            &config,
            (0.0, 100.0),
            &t_eval,
            &state0,
            1e-12,
            1e-12,
            None,
            None,
        );

        let err = match result {
            Ok(_) => panic!("无星历天体必须返回 Err"),
            Err(e) => e,
        };
        assert!(
            err.contains("FAKEBODY") || err.contains("SPICE"),
            "错误信息应指名失败的天体或 SPICE 查询，实际: {}",
            err
        );
    }

    // =========================================================
    // 测试 8：truncation_cause 分类（纯逻辑，不碰内核池/缓存）
    // =========================================================

    /// 构造一次失败的力模型回调留底（`message` 只作证据文本）。
    fn failure_at(et: f64) -> EomFailure {
        EomFailure {
            et,
            message: "SPICE query failed for MOON: Error { short_message: \"EPHEM_CACHE_MISS\", \
                      explanation: \"ephem cache et 7200 out of range [-3000, 6600]\" }"
                .to_string(),
        }
    }

    /// 缓存已启用且失败时刻越界：cause 带查询时刻与缓存窗口。
    #[test]
    fn truncation_cause_cache_out_of_range() {
        let cause = truncation_cause(Some(&failure_at(7200.0)), true, Some((-3000.0, 6600.0)));
        assert!(cause.contains("outside cached window"), "实际: {cause}");
        assert!(cause.contains("7200.000"), "应带查询时刻，实际: {cause}");
        assert!(cause.contains("-3000.000"), "应带窗口下界，实际: {cause}");
        assert!(cause.contains("6600.000"), "应带窗口上界，实际: {cause}");
        assert!(!cause.contains("step size collapsed"), "实际: {cause}");
    }

    /// 缓存已启用且失败时刻在窗口内：归为键未注册，与越界区分开。
    #[test]
    fn truncation_cause_cache_key_miss() {
        let cause = truncation_cause(Some(&failure_at(0.0)), true, Some((-3000.0, 6600.0)));
        assert!(cause.contains("lookup failed"), "实际: {cause}");
        assert!(!cause.contains("outside cached window"), "实际: {cause}");
    }

    /// 无缓存 + 内核池为空：报"未加载"，不报覆盖不足。
    #[test]
    fn truncation_cause_kernels_not_loaded() {
        let cause = truncation_cause(Some(&failure_at(0.0)), false, None);
        assert!(cause.contains("kernels not loaded"), "实际: {cause}");
        assert!(
            !cause.contains("insufficient kernel coverage"),
            "实际: {cause}"
        );
    }

    /// 无缓存 + 内核已加载但覆盖不足：报星历查询失败并保留底层证据文本。
    #[test]
    fn truncation_cause_insufficient_coverage() {
        let cause = truncation_cause(Some(&failure_at(0.0)), true, None);
        assert!(
            cause.contains("insufficient kernel coverage"),
            "实际: {cause}"
        );
        assert!(
            cause.contains("EPHEM_CACHE_MISS"),
            "应保留底层证据文本，实际: {cause}"
        );
    }

    /// 力模型未报错：归类为步长塌缩/步数上限。
    #[test]
    fn truncation_cause_no_eom_error() {
        let cause = truncation_cause(None, true, None);
        assert!(cause.contains("step size collapsed"), "实际: {cause}");
    }

    // =========================================================
    // 测试 9：覆盖外传播给出真实 cause（#677）
    // =========================================================

    /// 二分定位 (MOON, EARTH) 的 SPK 覆盖末端 et。
    ///
    /// 不硬编码日历日期：本地与 CI 的 DE 内核（de430 / de440 / de440s）覆盖
    /// 区间不同，硬编码要么让用例退化成初值预检失败，要么根本不触发越界。
    /// J2000 必在覆盖内，1e11 秒（约 5138 年）必在任何 DE 内核覆盖外。
    fn kernel_coverage_end_et() -> f64 {
        let ok = |et: f64| {
            e2m2e_spice::spk_accel::third_body_acceleration(
                et,
                "MOON",
                "EARTH",
                &[6678.0, 0.0, 0.0],
                4902.800066,
                1e-6,
            )
            .is_ok()
        };
        let mut lo = 0.0;
        let mut hi = 1.0e11;
        assert!(ok(lo), "J2000 处必须能查到 MOON 星历（内核未加载？）");
        assert!(!ok(hi), "1e11 秒处不应有 MOON 星历（内核覆盖超出预期？）");
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            if mid <= lo || mid >= hi {
                break;
            }
            if ok(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// 越出内核覆盖时，错误消息须指名"覆盖不足"且携带底层 SPICE 文本，
    /// 不再是笼统的 "likely cause: SPICE kernels not loaded or step size collapsed"。
    #[test]
    fn propagate_beyond_kernel_coverage_reports_real_cause() {
        if !load_kernels() {
            eprintln!("跳过：无 SPK 星历内核（de430/de440s.bsp 未被 git 跟踪）");
            return;
        }
        let config = earth_moon_sun_config();
        let wall = kernel_coverage_end_et();
        // 覆盖墙两侧各 1e5 秒：t0 在覆盖内，t_end 越界（步长被自适应控制器
        // 定在百秒量级，约千步即撞墙，不依赖 50 万步上限）。
        let t0 = wall - 1.0e5;
        let t_end = wall + 1.0e5;
        let t_eval = vec![t0, t_end];
        let state0 = [6678.0, 0.0, 0.0, 0.0, 7.725, 0.0];

        let err = match propagate_with_state(
            &config,
            (t0, t_end),
            &t_eval,
            &state0,
            1e-9,
            1e-12,
            None,
            None,
        ) {
            Ok(_) => panic!("越出内核覆盖必须硬失败（ADR 0020）"),
            Err(e) => e,
        };

        assert!(err.contains("cause:"), "错误应带 cause 段，实际: {err}");
        assert!(
            err.contains("insufficient kernel coverage"),
            "应归类为内核覆盖不足，实际: {err}"
        );
        assert!(
            !err.contains("likely cause"),
            "不应再出现通用截断文案，实际: {err}"
        );
        assert!(
            !err.contains("step size collapsed"),
            "不应误报步长塌缩，实际: {err}"
        );
    }

    // =========================================================
    // 测试 10：缓存窗口外传播给出 et 与窗口区间（#677）
    // =========================================================

    /// 启用窄窗星历缓存后，越出缓存窗口的查询须在 cause 中携带 et 与区间。
    #[test]
    fn propagate_outside_cache_window_reports_window() {
        if !load_kernels() {
            eprintln!("跳过：无 SPK 星历内核（de430/de440s.bsp 未被 git 跟踪）");
            return;
        }
        let config = earth_moon_sun_config();
        let bodies = vec![
            ("MOON".to_string(), "EARTH".to_string()),
            ("SUN".to_string(), "EARTH".to_string()),
        ];
        // build 两端各留 5·dt margin ⇒ 覆盖 [-3000, 6600]。
        let cache =
            e2m2e_spice::ephem_cache::EphemCache::build(&bodies, &[], &[], 0.0, 3600.0, 600.0)
                .expect("缓存构建应成功");
        e2m2e_spice::ephem_cache::enable(cache);

        let state0 = [6678.0, 0.0, 0.0, 0.0, 7.725, 0.0];
        let result = propagate_with_state(
            &config,
            (0.0, 7200.0),
            &[0.0, 7200.0],
            &state0,
            1e-9,
            1e-12,
            None,
            None,
        );
        // 先还原全局态再断言，避免污染同进程后续测试（串行执行）。
        e2m2e_spice::ephem_cache::disable();

        let err = match result {
            Ok(_) => panic!("缓存窗口外必须硬失败（ADR 0020 决策 4）"),
            Err(e) => e,
        };
        assert!(err.contains("outside cached window"), "实际: {err}");
        assert!(err.contains("window ["), "应带缓存窗口区间，实际: {err}");
        assert!(err.contains("EPHEM_CACHE_MISS"), "实际: {err}");
    }

    // =========================================================
    // 测试 11：初值预检出口同样带 cause
    // =========================================================

    /// 初值时刻已在覆盖外时走预检出口，消息须同样带 `cause:` 段与分类结果。
    #[test]
    fn initial_rhs_failure_reports_real_cause() {
        if !load_kernels() {
            eprintln!("跳过：无 SPK 星历内核（de430/de440s.bsp 未被 git 跟踪）");
            return;
        }
        let config = earth_moon_sun_config();
        let wall = kernel_coverage_end_et();
        let t0 = wall + 1.0e5; // 初值已在覆盖外，积分不启动
        let t_end = t0 + 1.0e3;
        let state0 = [6678.0, 0.0, 0.0, 0.0, 7.725, 0.0];

        let err = match propagate_with_state(
            &config,
            (t0, t_end),
            &[t0, t_end],
            &state0,
            1e-9,
            1e-12,
            None,
            None,
        ) {
            Ok(_) => panic!("初值覆盖外必须硬失败"),
            Err(e) => e,
        };

        assert!(
            err.contains("initial RHS evaluation failed at t="),
            "应走初值预检出口，实际: {err}"
        );
        assert!(
            err.contains("cause:"),
            "预检出口也应带 cause 段，实际: {err}"
        );
        assert!(
            err.contains("insufficient kernel coverage"),
            "应归类为内核覆盖不足，实际: {err}"
        );
        assert!(!err.contains("likely cause"), "实际: {err}");
    }
}
