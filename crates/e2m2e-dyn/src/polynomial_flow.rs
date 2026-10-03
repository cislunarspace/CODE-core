//! CR3BP 多项式流（微分代数传播）内核（issue #786；ADR 0059）。
//!
//! 多项式流：把初值六分量偏差 δ 作为 DA 变量（变量 1..=6 对应
//! δx, δy, δz, δvx, δvy, δvz），初值多项式取恒等映射
//! `yᵢ = constant(y₀ᵢ) + variable(i)`，在 DA 算术下用**固定步长经典 RK4**
//! 积分 CR3BP 运动方程；各求值时刻的状态是初值偏差的 k 阶截断
//! Taylor 多项式（多项式流映射），可经 `Da::eval` / `CompiledDa` 在任意
//! 偏差处快速求值。骨架同 Losacco 等人 2022 年的 DA 多项式流做法：
//! 一条名义轨迹加高阶偏差映射，不做步长自适应、不做域分裂。
//!
//! # 与 f64 版 CR3BP 的口径差异
//!
//! `e2m2e_forces::cr3bp::cr3bp_eom` 对 r1/r2 设 `MIN_DISTANCE` 钳制防近碰撞
//! 除零；本模块的 DA 版右端项**不设钳制**——多项式在某点求值出的 r 不可
//! 先验钳制，近碰撞轨迹在 DA 算术下表现为奇点（r1 的常数部分过零时报
//! 645/641），按 `DaFlowError::Runtime` 上抛，属口径差异而非缺陷。
//!
//! # 上下文约定
//!
//! DA 上下文是进程级全局状态，由调用方初始化（Python 侧 `da_init_py`）；
//! 本内核只校验（已初始化、变量数 ≥ 6、截断阶在 1..=上下文阶数内）并
//! 临时切换线程局部截断阶，返回前恢复旧值，绝不自行 re-init。

use e2m2e_da::{catch_da_panic, sqrt, Da};

/// 多项式流传播结果。
///
/// `flows[k][i]` 是第 k 个求值时刻第 i 个状态分量的多项式（初值偏差的
/// 截断 Taylor 映射）；`states[k]` 是其常数部分（名义状态）。
pub struct DaFlowResult {
    /// 求值时刻（= `t_eval`）。
    pub times: Vec<f64>,
    /// 各时刻名义状态（多项式常数项）。
    pub states: Vec<[f64; 6]>,
    /// 各时刻 6 个多项式：初值偏差 δ → 状态偏差。
    pub flows: Vec<Vec<Da>>,
    /// RK4 步数（含对齐事件时刻的截短步）。
    pub n_steps: usize,
}

/// 多项式流错误。
///
/// `Context` 为前置条件/参数错误（绑定层映射为 Python `ValueError`）；
/// `Runtime` 为传播期 DA 运算失败（映射为 `RuntimeError`）。
#[derive(Debug)]
pub enum DaFlowError {
    /// 前置条件或参数错误。
    Context(String),
    /// 传播期 DA 运算失败。
    Runtime(String),
}

/// CR3BP 运动方程的 DA 版右端项，逐项转写自 `cr3bp_eom`。
///
/// 公式：x″ = 2vy + x − (1−μ)(x+μ)/r1³ − μ(x−1+μ)/r2³，y″ = −2vx + y −
/// (1−μ)y/r1³ − μy/r2³，z″ = −(1−μ)z/r1³ − μz/r2³；r1、r2 为到较大/较小
/// 天体的距离。**不设 MIN_DISTANCE 钳制**（见模块文档的口径差异说明）。
fn cr3bp_eom_da(mu: f64, y: [Da; 6]) -> [Da; 6] {
    let [x, py, pz, vx, vy, vz] = y;
    let xm = 1.0 - mu; // 较大天体质量
    let dx1 = x.clone() + mu; // 航天器 - 较大天体（x=-μ）
    let dx2 = x.clone() + (mu - 1.0); // 航天器 - 较小天体（x=1-μ）
    let r1 = sqrt(&(dx1.clone() * dx1.clone() + py.clone() * py.clone() + pz.clone() * pz.clone()));
    let r2 = sqrt(&(dx2.clone() * dx2.clone() + py.clone() * py.clone() + pz.clone() * pz.clone()));
    let inv_r1_3 = r1.powi(-3);
    let inv_r2_3 = r2.powi(-3);
    let gx1 = xm * dx1 * inv_r1_3.clone();
    let gx2 = mu * dx2 * inv_r2_3.clone();
    let gy1 = xm * py.clone() * inv_r1_3.clone();
    let gy2 = mu * py.clone() * inv_r2_3.clone();
    let gz1 = xm * pz.clone() * inv_r1_3;
    let gz2 = mu * pz * inv_r2_3;
    let ax = 2.0 * vy.clone() + x - gx1 - gx2;
    let ay = -2.0 * vx.clone() + py - gy1 - gy2;
    let az = -(gz1 + gz2);
    [vx, vy, vz, ax, ay, az]
}
/// `y + h/2·k1` 等中间量按 DA↔f64 双向算术逐分量构造（dace-rs 的二元
/// 运算按值消费 `Da`，中间量需克隆）。
fn rk4_step_da(mu: f64, h: f64, y: &[Da; 6]) -> [Da; 6] {
    let k1 = cr3bp_eom_da(mu, y.clone());
    let y2 = std::array::from_fn(|i| y[i].clone() + k1[i].clone() * (0.5 * h));
    let k2 = cr3bp_eom_da(mu, y2);
    let y3 = std::array::from_fn(|i| y[i].clone() + k2[i].clone() * (0.5 * h));
    let k3 = cr3bp_eom_da(mu, y3);
    let y4 = std::array::from_fn(|i| y[i].clone() + k3[i].clone() * h);
    let k4 = cr3bp_eom_da(mu, y4);
    std::array::from_fn(|i| {
        y[i].clone()
            + (k1[i].clone() + k2[i].clone() * 2.0 + k3[i].clone() * 2.0 + k4[i].clone())
                * (h / 6.0)
    })
}

/// CR3BP 多项式流传播（固定步长 RK4 + DA 算术）。
///
/// 初值多项式为恒等映射 `constant(y₀ᵢ) + variable(i+1)`；事件序列为
/// `t_eval` 逐点 + `t_end`：对每个事件按 `h_eff = min(step, |event−t|)·方向`
/// 步进到事件时刻（含对齐用的截短步），属 `t_eval` 的时刻记录名义状态与
/// 多项式映射。传播中临时把线程截断阶切到 `order`，返回前恢复旧值。
///
/// # 错误
/// - [`DaFlowError::Context`]：DA 上下文未初始化 / 变量数 < 6 / 截断阶越界
///   （`set_truncation_order` 会静默钳制、超界变量静默给零多项式，故必须
///   先挡）、mu 或 step 非正或非有限、t_span 两端相等、t_eval 为空、
///   非严格单调或落在 t_span 闭区间外。
/// - [`DaFlowError::Runtime`]：传播期 DA 运算 panic（如近碰撞除零）。
pub fn propagate_cr3bp_da(
    mu: f64,
    t_span: (f64, f64),
    t_eval: &[f64],
    initial_state: &[f64; 6],
    order: u32,
    step: f64,
) -> Result<DaFlowResult, DaFlowError> {
    // ---- 前置校验（须在触碰 max_order/max_variables 等会 panic 的内省前
    // ---- 先确认已初始化）----
    if !e2m2e_da::initialized() {
        return Err(DaFlowError::Context(
            "DA 上下文未初始化：请先 da_init_py(order, 6)".to_string(),
        ));
    }
    let ctx_vars = e2m2e_da::max_variables();
    if ctx_vars < 6 {
        return Err(DaFlowError::Context(format!(
            "DA 上下文变量数 {ctx_vars} < 6：dace 对超界变量静默给零多项式，\
             多项式流需要 6 个初值偏差变量，请 da_init_py(order, 6)"
        )));
    }
    let ctx_order = e2m2e_da::max_order();
    if order < 1 || order > ctx_order {
        return Err(DaFlowError::Context(format!(
            "截断阶 {order} 越界：当前 DA 上下文阶数上限 {ctx_order}\
             （set_truncation_order 会静默钳制），请传 1..={ctx_order} 或重新 da_init_py"
        )));
    }
    if !mu.is_finite() || mu <= 0.0 {
        return Err(DaFlowError::Context(format!(
            "mu 必须为正的有限值，得到 {mu}"
        )));
    }
    if !step.is_finite() || step <= 0.0 {
        return Err(DaFlowError::Context(format!(
            "step 必须为正的有限值（步长模长），得到 {step}"
        )));
    }
    let (t0, t_end) = t_span;
    if !t0.is_finite() || !t_end.is_finite() || t0 == t_end {
        return Err(DaFlowError::Context(format!(
            "t_span 两端必须为不同有限值，得到 ({t0}, {t_end})"
        )));
    }
    if t_eval.is_empty() {
        return Err(DaFlowError::Context("t_eval 不能为空".to_string()));
    }
    let direction = (t_end - t0).signum();
    let (lo, hi) = if t0 < t_end { (t0, t_end) } else { (t_end, t0) };
    for &te in t_eval {
        if !te.is_finite() || te < lo || te > hi {
            return Err(DaFlowError::Context(format!(
                "t_eval 含越界点 {te}：须全部落在 t_span [{t0}, {t_end}] 闭区间内"
            )));
        }
    }
    for w in t_eval.windows(2) {
        if direction * (w[1] - w[0]) <= 0.0 {
            return Err(DaFlowError::Context(
                "t_eval 必须沿 t_span 方向严格单调".to_string(),
            ));
        }
    }

    // ---- 截断阶临时切换 + 主循环（DA 运算 panic 归一为 Runtime）----
    let prev = e2m2e_da::truncation_order();
    e2m2e_da::set_truncation_order(order);
    let run = catch_da_panic(|| {
        let mut y: [Da; 6] =
            std::array::from_fn(|i| Da::constant(initial_state[i]) + Da::variable(i as u32 + 1));
        let mut times: Vec<f64> = Vec::with_capacity(t_eval.len());
        let mut states: Vec<[f64; 6]> = Vec::with_capacity(t_eval.len());
        let mut flows: Vec<Vec<Da>> = Vec::with_capacity(t_eval.len());
        let tol_t = 1e-12 * (1.0 + t_end.abs());
        let mut t = t0;
        let mut n_steps = 0usize;
        let mut eval_idx = 0usize;

        // t_eval 含 t0：记录恒等映射
        if (t_eval[0] - t0).abs() <= tol_t {
            times.push(t_eval[0]);
            states.push(std::array::from_fn(|i| y[i].cons()));
            flows.push(y.to_vec());
            eval_idx = 1;
        }

        // 事件序列 = 剩余 t_eval 逐点 + t_end（末尾补积分到区间端）
        let mut events: Vec<f64> = t_eval[eval_idx..].to_vec();
        events.push(t_end);
        for &event in &events {
            while (event - t).abs() > tol_t {
                let h_eff = step.min((event - t).abs()) * direction;
                y = rk4_step_da(mu, h_eff, &y);
                t += h_eff;
                n_steps += 1;
            }
            if eval_idx < t_eval.len() && (t_eval[eval_idx] - event).abs() <= tol_t {
                times.push(t_eval[eval_idx]);
                states.push(std::array::from_fn(|i| y[i].cons()));
                flows.push(y.to_vec());
                eval_idx += 1;
            }
        }

        if times.len() != t_eval.len() {
            return Err(DaFlowError::Context(format!(
                "输出点数不足：得到 {} 个，期望 {} 个",
                times.len(),
                t_eval.len()
            )));
        }
        Ok(DaFlowResult {
            times,
            states,
            flows,
            n_steps,
        })
    });
    e2m2e_da::set_truncation_order(prev);

    run.map_err(|e| DaFlowError::Runtime(format!("DA 运算失败: {e}")))?
}

#[cfg(test)]
mod tests {
    use super::*;

    /// DE421 校准地月质量比（与 family_generation/nrho.rs 的
    /// `DE421_EARTH_MOON_MU` 同值，种子周期在该 μ 下标定）。
    const MU: f64 = 0.012_150_585_350_562_453;
    /// 北族 L2 NRHO 折叠种子（`family_generation/nrho.rs` 的 `L2_SEED_STATE`）。
    const SEED: [f64; 6] = [
        1.128103754424342,
        0.0,
        0.17883236940616654,
        0.0,
        -0.22553424464298827,
        0.0,
    ];

    /// 进程级 DA 上下文按需重建到（阶 ≥ order、变量 ≥ 6）。
    fn ensure_context(order: u32) {
        if !e2m2e_da::initialized()
            || e2m2e_da::max_order() < order
            || e2m2e_da::max_variables() < 6
        {
            e2m2e_da::init(order, 6).unwrap();
        }
    }

    /// PD78 重积分终点状态（FD 对照用 oracle）。
    fn repropagate_final(mu: f64, arc: f64, seed: &[f64; 6]) -> [f64; 6] {
        let r = e2m2e_forces::cr3bp::propagate_cr3bp(
            mu,
            (0.0, arc),
            &[arc],
            seed,
            1e-13,
            1e-13,
            None,
            None,
        )
        .unwrap();
        *r.states.last().unwrap()
    }

    /// Jacobi 常数的 DA 展开C = x²+y² + 2(1−μ)/r1 + 2μ/r2 − (vx²+vy²+vz²)。
    fn jacobi_da(mu: f64, y: &[Da; 6]) -> Da {
        let r1 = sqrt(&((y[0].clone() + mu).powi(2) + y[1].clone().powi(2) + y[2].clone().powi(2)));
        let r2 = sqrt(
            &((y[0].clone() + (mu - 1.0)).powi(2) + y[1].clone().powi(2) + y[2].clone().powi(2)),
        );
        y[0].clone().powi(2) + y[1].clone().powi(2) + 2.0 * (1.0 - mu) / r1 + 2.0 * mu / r2
            - (y[3].clone().powi(2) + y[4].clone().powi(2) + y[5].clone().powi(2))
    }

    /// 验收 1（研究级）：一阶截断的线性映射与既有 36 维 STM 一致。
    #[test]
    fn test_linear_matches_stm() {
        ensure_context(1);
        let result = propagate_cr3bp_da(MU, (0.0, 1.0), &[1.0], &SEED, 1, 2.5e-4).unwrap();
        let stm = e2m2e_forces::cr3bp::propagate_cr3bp_stm(
            MU,
            (0.0, 1.0),
            &[1.0],
            &SEED,
            1e-13,
            1e-13,
            None,
            None,
        )
        .unwrap();
        for i in 0..6 {
            let lin = result.flows[0][i].linear();
            assert_eq!(lin.len(), 6, "线性系数应有 6 个");
            for j in 0..6 {
                let expect = stm.stms[0][i * 6 + j];
                let got = lin[j];
                let err = (got - expect).abs() / expect.abs().max(1.0);
                assert!(
                    err < 1e-10,
                    "L[{i}][{j}] = {got} 对 STM {expect}，相对误差 {err} 超过 1e-10"
                );
            }
        }
    }

    /// 验收 2（研究级）：二阶系数与中心差分 Hessian 抽查一致。
    ///
    /// 容差 1e-6 由 FD 噪声上限（rtol 1e-13 的重积分在 h_fd=1e-4 下的二阶
    /// 差分舍入 ~1e-8）决定，不是 DA 展开的精度极限。
    #[test]
    fn test_second_order_matches_finite_difference_hessian() {
        ensure_context(2);
        let arc = 0.5;
        let result = propagate_cr3bp_da(MU, (0.0, arc), &[arc], &SEED, 2, 5e-4).unwrap();
        let h = 1e-4_f64;
        let e = |j: usize, s: f64| {
            let mut seed = SEED;
            seed[j] += s * h;
            repropagate_final(MU, arc, &seed)
        };

        // x_f 对 δxδx：j=k 时多项式系数 = H_jj/2（Taylor 展开含 1/2!）
        let fpp = e(0, 2.0)[0];
        let f0 = repropagate_final(MU, arc, &SEED)[0];
        let fmm = e(0, -2.0)[0];
        let h_xx = (fpp - 2.0 * f0 + fmm) / (4.0 * h * h);
        let coef_xx = result.flows[0][0].get_coefficient(&[2, 0, 0, 0, 0, 0]);
        let err = (coef_xx - h_xx / 2.0).abs() / (h_xx / 2.0).abs().max(1.0);
        assert!(
            err < 1e-6,
            "H_xx/2 系数 {coef_xx} 对 FD {}/2，误差 {err}",
            h_xx
        );

        // y_f 对 δxδy：j≠k 时多项式系数 = H_jk（混合项展开无因子）
        let mixed = |sx: f64, sy: f64| {
            let mut seed = SEED;
            seed[0] += sx * h;
            seed[1] += sy * h;
            repropagate_final(MU, arc, &seed)[1]
        };
        let h_xy = (mixed(1.0, 1.0) - mixed(1.0, -1.0) - mixed(-1.0, 1.0) + mixed(-1.0, -1.0))
            / (4.0 * h * h);
        let coef_xy = result.flows[0][1].get_coefficient(&[1, 1, 0, 0, 0, 0]);
        let err = (coef_xy - h_xy).abs() / h_xy.abs().max(1.0);
        assert!(err < 1e-6, "H_xy 系数 {coef_xy} 对 FD {h_xy}，误差 {err}");
    }

    /// 验收 3（研究级）：Jacobi 常数 DA 展开余项——精确流下 C 沿轨迹守恒，
    /// 截断流在 1..=order 阶内的余项应只剩 RK4 积分误差。
    #[test]
    fn test_jacobi_da_drift() {
        ensure_context(3);
        let arc = 1.0;
        let result = propagate_cr3bp_da(MU, (0.0, arc), &[0.0, arc], &SEED, 3, 2.5e-4).unwrap();
        let c0: [Da; 6] = result.flows[0].clone().try_into().unwrap();
        let ce: [Da; 6] = result.flows[1].clone().try_into().unwrap();
        let d = jacobi_da(MU, &ce) - jacobi_da(MU, &c0);
        assert!(
            d.cons().abs() < 1e-10,
            "Jacobi 常数部分漂移 {} 超过 1e-10",
            d.cons()
        );
        for m in d.iter_monomials() {
            assert!(
                m.c.abs() < 1e-10,
                "Jacobi 展开系数 [{:?}] = {} 超过 1e-10",
                m.jj,
                m.c
            );
        }
    }

    /// 零偏差处求值与名义状态（常数项）相等。
    #[test]
    fn test_zero_deviation_eval_equals_nominal() {
        ensure_context(2);
        let result = propagate_cr3bp_da(MU, (0.0, 0.25), &[0.0, 0.25], &SEED, 2, 1e-3).unwrap();
        for k in 0..result.flows.len() {
            for i in 0..6 {
                assert_eq!(
                    result.flows[k][i].eval(&[0.0; 6]),
                    result.states[k][i],
                    "flows[{k}][{i}] 零偏差求值应等于名义状态"
                );
            }
        }
    }

    /// 前向再反向回积，末端名义状态回到种子（固定步长 RK4 自洽性）。
    #[test]
    fn test_backward_roundtrip() {
        ensure_context(2);
        let fwd = propagate_cr3bp_da(MU, (0.0, 1.0), &[1.0], &SEED, 2, 1e-3).unwrap();
        let back = propagate_cr3bp_da(MU, (1.0, 0.0), &[0.0], &fwd.states[0], 2, 1e-3).unwrap();
        for i in 0..6 {
            let diff = (back.states[0][i] - SEED[i]).abs();
            assert!(diff < 1e-9, "分量 {i} 回积差 {diff} 超过 1e-9");
        }
    }

    /// 前置校验错误路径（无条件 init 到 (2, 6)，钉住阶上限 = 2）。
    #[test]
    fn test_validation_errors() {
        e2m2e_da::init(2, 6).unwrap();
        match propagate_cr3bp_da(MU, (0.0, 1.0), &[], &SEED, 1, 1e-3) {
            Err(DaFlowError::Context(msg)) => assert!(msg.contains("t_eval"), "得到 {msg}"),
            Err(other) => panic!("t_eval 空应报 Context 错误，得到 {other:?}"),
            Ok(_) => panic!("t_eval 空应报 Context 错误，却成功返回"),
        }
        // 截断阶超过上下文阶数上限（当前上限 2）
        match propagate_cr3bp_da(MU, (0.0, 1.0), &[1.0], &SEED, 3, 1e-3) {
            Err(DaFlowError::Context(msg)) => assert!(msg.contains("截断阶"), "得到 {msg}"),
            Err(other) => panic!("order 越界应报 Context 错误，得到 {other:?}"),
            Ok(_) => panic!("order 越界应报 Context 错误，却成功返回"),
        }
        // 变量数不足（重新 init 到 3 变量）
        e2m2e_da::init(3, 3).unwrap();
        match propagate_cr3bp_da(MU, (0.0, 1.0), &[1.0], &SEED, 3, 1e-3) {
            Err(DaFlowError::Context(msg)) => assert!(msg.contains("变量数"), "得到 {msg}"),
            Err(other) => panic!("nvars<6 应报 Context 错误，得到 {other:?}"),
            Ok(_) => panic!("nvars<6 应报 Context 错误，却成功返回"),
        }
    }
}
