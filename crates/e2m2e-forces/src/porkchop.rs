//! porkchop 网格扫描评估单元（Rust 实现，纯数学，不依赖 SPICE）。
//!
//! 从 Python `e2m2e/algorithm/transfer/porkchop.py` 的网格循环下沉（#446，
//! ADR 0017 边界条目）：终端状态传播 + Lambert 求解 + ΔV 组装 + 网格分发
//! 全部在 Rust 完成，Python 只做问题构造与结果解释（架构设计文档 §2/§3）。
//!
//! 两条入口对应 Python 侧的两种问题构造方式：
//!
//! - [`porkchop_grid_serial`] / [`porkchop_grid_parallel`]：内置终端
//!   （[`TerminalSpec`]）规格直接传入，终端传播在 Rust 内逐点进行——
//!   语义逐行对照 `CR3BP_Dynamics.propagate_orbit_state_at_time`
//!   （周期取模 + 首点重积分，直接调纯 Rust [`propagate_cr3bp`]，
//!   不绕道持 GIL 的 `propagate_cr3bp_py`）。
//! - [`porkchop_grid_states_serial`] / [`porkchop_grid_states_parallel`]：
//!   终端状态网格由 Python 按 `get_arrival_state` 协议预提取（自定义
//!   `TerminalCondition` 子类场景），Rust 只做 Lambert + ΔV 组装。
//!
//! 并行照搬 `transfer_grid_search` 的 Rayon 范式：`par_iter` + `collect`
//! 保序，串/并逐位一致（`E2M2E_PORKCHOP_PARALLEL=0` 强制串行对照，
//! 对称 `E2M2E_SEARCH_PARALLEL`）。

use e2m2e_propagation::lambert::{lambert_izzo, TransferDirection};
use rayon::prelude::*;

use crate::cr3bp::propagate_cr3bp;

/// 终端条件规格（对照 Python `TerminalCondition` 的两个内置实现）。
#[derive(Clone, Copy, Debug)]
pub enum TerminalSpec {
    /// `OrbitTerminal`：周期轨道首点状态 + 时间原点 + 周期。
    Orbit {
        state0: [f64; 6],
        t0: f64,
        period: f64,
    },
    /// `StateTerminal`：固定状态（与时刻无关）。
    State { state: [f64; 6] },
}

/// 终端传播配置（CR3BP 纯数学路径）。
#[derive(Clone, Copy, Debug)]
pub struct PropagationParams {
    /// CR3BP 质量参数。
    pub mu: f64,
    pub rtol: f64,
    pub atol: f64,
    pub max_step: f64,
}

/// Lambert 求解配置。
#[derive(Clone, Copy, Debug)]
pub struct LambertParams {
    /// 中心天体 GM（km³/s²）。
    pub mu_central: f64,
    pub long_way: bool,
    pub revs: u32,
}

/// 单终端在时刻 `t` 的状态（移植 `propagate_orbit_state_at_time` 语义）。
///
/// 周期轨道：目标时刻对周期取模（`np.mod` ↔ `rem_euclid`，周期为正时一致），
/// `t_rel < 1e-14` 直接返回首点状态；否则从首点重积分到 `t0 + t_rel`。
///
/// `propagate_cr3bp` 会在每个 `t_eval` 点截断步长，采样点会影响之后的自适应
/// 步序列。因此此处逐式复现 Python `propagate_orbit_state_at_time` 的
/// `n_steps = max(ceil(t_rel / 0.01) + 1, 2)` 与 `np.linspace`，而不能只传
/// 首末两点；由此保证规格路径与协议路径的终端末态逐位一致。
pub fn terminal_state_at(
    terminal: &TerminalSpec,
    t: f64,
    propagation: Option<&PropagationParams>,
) -> Result<[f64; 6], String> {
    match terminal {
        TerminalSpec::State { state } => Ok(*state),
        TerminalSpec::Orbit { state0, t0, period } => {
            let params = propagation.ok_or_else(|| {
                "orbit 终端需要传播配置（CR3BP mu/容差/步长），但未提供".to_string()
            })?;
            let t_rel = (t - t0).rem_euclid(*period);
            if t_rel < 1e-14 {
                return Ok(*state0);
            }
            // 逐式对应 np.linspace(t0, t0 + t_rel, n_steps)：numpy 对末点
            // 单独赋 stop，避免浮点乘加误差，Rust 侧同样显式写入。
            let n_steps = ((t_rel / 0.01).ceil() as usize + 1).max(2);
            let step = t_rel / (n_steps - 1) as f64;
            let mut t_eval = Vec::with_capacity(n_steps);
            for i in 0..n_steps - 1 {
                t_eval.push(*t0 + step * i as f64);
            }
            t_eval.push(*t0 + t_rel);
            let result = propagate_cr3bp(
                params.mu,
                (*t0, t0 + t_rel),
                &t_eval,
                state0,
                params.rtol,
                params.atol,
                Some(params.max_step),
                None,
            )
            .map_err(|e| format!("porkchop 终端状态传播失败（t={t}）：{e}"))?;
            result
                .states
                .last()
                .copied()
                .ok_or_else(|| format!("porkchop 终端状态传播未返回状态（t={t}）"))
        }
    }
}

/// 单网格点评估：Lambert 求解 + 双脉冲 ΔV。
///
/// 无解组合（Lambert Err）返回 `(NaN, NaN)`，与 Python 侧
/// `solve_lambert_batch` 的 NaN 占位语义一致。
fn evaluate_cell(
    dep_state: &[f64; 6],
    arr_state: &[f64; 6],
    tof: f64,
    lambert: &LambertParams,
) -> (f64, f64) {
    let r0 = [dep_state[0], dep_state[1], dep_state[2]];
    let rf = [arr_state[0], arr_state[1], arr_state[2]];
    let direction = if lambert.long_way {
        TransferDirection::LongWay
    } else {
        TransferDirection::ShortWay
    };
    match lambert_izzo(&r0, &rf, tof, lambert.mu_central, direction, lambert.revs) {
        Ok((v0, vf, _n_iter)) => {
            let dv1 = ((v0[0] - dep_state[3]).powi(2)
                + (v0[1] - dep_state[4]).powi(2)
                + (v0[2] - dep_state[5]).powi(2))
            .sqrt();
            let dv2 = ((arr_state[3] - vf[0]).powi(2)
                + (arr_state[4] - vf[1]).powi(2)
                + (arr_state[5] - vf[2]).powi(2))
            .sqrt();
            (dv1, dv2)
        }
        Err(_) => (f64::NAN, f64::NAN),
    }
}

/// 状态网格评估（串行）：`dep_states[i]` 为 `t_dep[i]` 时刻出发状态，
/// `arr_states[i * m + j]` 为 `t_dep[i] + tof[j]` 时刻到达状态（行优先，
/// i 主序）。返回展平的 `(dv1, dv2)`，长度 `n * m`，同行优先约定。
pub fn porkchop_grid_states_serial(
    dep_states: &[[f64; 6]],
    arr_states: &[[f64; 6]],
    n_tof: usize,
    tofs: &[f64],
    lambert: &LambertParams,
) -> (Vec<f64>, Vec<f64>) {
    let n = dep_states.len();
    let mut dv1 = Vec::with_capacity(n * n_tof);
    let mut dv2 = Vec::with_capacity(n * n_tof);
    for i in 0..n {
        for j in 0..n_tof {
            let (d1, d2) =
                evaluate_cell(&dep_states[i], &arr_states[i * n_tof + j], tofs[j], lambert);
            dv1.push(d1);
            dv2.push(d2);
        }
    }
    (dv1, dv2)
}

/// 状态网格评估（Rayon 并行）：`par_iter` + `collect` 保序，与
/// [`porkchop_grid_states_serial`] 逐位一致。
pub fn porkchop_grid_states_parallel(
    dep_states: &[[f64; 6]],
    arr_states: &[[f64; 6]],
    n_tof: usize,
    tofs: &[f64],
    lambert: &LambertParams,
) -> (Vec<f64>, Vec<f64>) {
    let n = dep_states.len();
    let cells: Vec<(f64, f64)> = (0..n * n_tof)
        .into_par_iter()
        .map(|idx| {
            let (i, j) = (idx / n_tof, idx % n_tof);
            evaluate_cell(&dep_states[i], &arr_states[idx], tofs[j], lambert)
        })
        .collect();
    cells.into_iter().unzip()
}

/// 出发/到达状态网格对（`(dep_states, arr_states)`，行优先约定见
/// [`porkchop_grid_states_serial`]）。
type StateGrids = (Vec<[f64; 6]>, Vec<[f64; 6]>);

/// 从终端规格构造状态网格（串行）：出发状态按 `t_dep` 逐点、到达状态按
/// `(t_dep[i], tof[j])` 行优先逐点传播。
fn build_state_grids_serial(
    t_dep: &[f64],
    tofs: &[f64],
    dep: &TerminalSpec,
    arr: &TerminalSpec,
    propagation: Option<&PropagationParams>,
) -> Result<StateGrids, String> {
    let (n, m) = (t_dep.len(), tofs.len());
    let mut dep_states = Vec::with_capacity(n);
    for &td in t_dep {
        dep_states.push(terminal_state_at(dep, td, propagation)?);
    }
    let mut arr_states = Vec::with_capacity(n * m);
    for &td in t_dep {
        for &tf in tofs {
            arr_states.push(terminal_state_at(arr, td + tf, propagation)?);
        }
    }
    Ok((dep_states, arr_states))
}

/// 从终端规格构造状态网格（Rayon 并行，保序）。
fn build_state_grids_parallel(
    t_dep: &[f64],
    tofs: &[f64],
    dep: &TerminalSpec,
    arr: &TerminalSpec,
    propagation: Option<&PropagationParams>,
) -> Result<StateGrids, String> {
    let (n, m) = (t_dep.len(), tofs.len());
    let dep_states: Result<Vec<[f64; 6]>, String> = t_dep
        .par_iter()
        .map(|&td| terminal_state_at(dep, td, propagation))
        .collect();
    let dep_states = dep_states?;
    let arr_states: Result<Vec<[f64; 6]>, String> = (0..n * m)
        .into_par_iter()
        .map(|idx| {
            let (i, j) = (idx / m, idx % m);
            terminal_state_at(arr, t_dep[i] + tofs[j], propagation)
        })
        .collect();
    Ok((dep_states, arr_states?))
}

/// 规格路径网格扫描（串行）：终端传播 + Lambert + ΔV 组装。
///
/// 返回展平的 `(dv1, dv2)`，长度 `n * m`，行优先（i 主序）。
/// 传播失败（步长塌缩等）整调用报错——与 Python 协议路径中
/// `propagate_orbit_state_at_time` 直接上抛的行为对齐。
pub fn porkchop_grid_serial(
    t_dep: &[f64],
    tofs: &[f64],
    dep: &TerminalSpec,
    arr: &TerminalSpec,
    propagation: Option<&PropagationParams>,
    lambert: &LambertParams,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let (dep_states, arr_states) = build_state_grids_serial(t_dep, tofs, dep, arr, propagation)?;
    Ok(porkchop_grid_states_serial(
        &dep_states,
        &arr_states,
        tofs.len(),
        tofs,
        lambert,
    ))
}

/// 规格路径网格扫描（Rayon 并行）：与 [`porkchop_grid_serial`] 逐位一致。
pub fn porkchop_grid_parallel(
    t_dep: &[f64],
    tofs: &[f64],
    dep: &TerminalSpec,
    arr: &TerminalSpec,
    propagation: Option<&PropagationParams>,
    lambert: &LambertParams,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let (dep_states, arr_states) = build_state_grids_parallel(t_dep, tofs, dep, arr, propagation)?;
    Ok(porkchop_grid_states_parallel(
        &dep_states,
        &arr_states,
        tofs.len(),
        tofs,
        lambert,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 地月质量参数（DE421，与 Python 测试套件一致）。
    const MU_EM: f64 = 0.012150585609624;
    /// DRO 种子状态（Cui et al. 2025，与 tests/algorithm/conftest.py 一致）。
    const DRO_SEED: [f64; 6] = [0.79188556619742, 0.0, 0.0, 0.0, 0.573665890385585, 0.0];
    const DRO_PERIOD: f64 = 6.307498;

    fn prop_params() -> PropagationParams {
        PropagationParams {
            mu: MU_EM,
            rtol: 1e-9,
            atol: 1e-9,
            max_step: 0.05,
        }
    }

    #[test]
    fn state_terminal_returns_fixed_state() {
        let term = TerminalSpec::State {
            state: [1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
        };
        let s = terminal_state_at(&term, 123.0, None).unwrap();
        assert_eq!(s, [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]);
    }

    #[test]
    fn orbit_terminal_zero_phase_returns_seed() {
        let term = TerminalSpec::Orbit {
            state0: DRO_SEED,
            t0: 0.0,
            period: DRO_PERIOD,
        };
        // t_rel < 1e-14：不触发积分，直接返回首点。
        let s = terminal_state_at(&term, 1e-15, Some(&prop_params())).unwrap();
        assert_eq!(s, DRO_SEED);
    }

    #[test]
    fn orbit_terminal_period_wrap_is_deterministic() {
        // t 与 t + period 取模后 t_rel 相同 → 同一积分调用 → 逐位一致。
        let term = TerminalSpec::Orbit {
            state0: DRO_SEED,
            t0: 0.0,
            period: DRO_PERIOD,
        };
        let params = prop_params();
        let a = terminal_state_at(&term, 1.25, Some(&params)).unwrap();
        let b = terminal_state_at(&term, 1.25 + DRO_PERIOD, Some(&params)).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn orbit_terminal_requires_propagation_params() {
        let term = TerminalSpec::Orbit {
            state0: DRO_SEED,
            t0: 0.0,
            period: DRO_PERIOD,
        };
        assert!(terminal_state_at(&term, 1.0, None).is_err());
    }

    #[test]
    fn grid_states_assembles_dv_and_nan() {
        // 两个固定状态：Lambert 有解时 dv 为速度差范数，tof 过小无解时为 NaN。
        let dep = [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0];
        let arr = [0.0, 42164.0, 0.0, -3.0, 0.0, 0.0];
        let lambert = LambertParams {
            mu_central: 398600.4418,
            long_way: false,
            revs: 0,
        };
        let (dv1, dv2) =
            porkchop_grid_states_serial(&[dep], &[arr, arr], 2, &[3600.0, 1e-3], &lambert);
        assert!(dv1[0].is_finite() && dv2[0].is_finite());

        // 无解单元：revs=1 而 tof 远低于一圈最小转移时间 → NaN 占位。
        let lambert_multi_rev = LambertParams { revs: 1, ..lambert };
        let (dv1_bad, dv2_bad) =
            porkchop_grid_states_serial(&[dep], &[arr], 1, &[10.0], &lambert_multi_rev);
        assert!(dv1_bad[0].is_nan() && dv2_bad[0].is_nan());

        // 有解单元：与直接调 lambert_izzo 手工组装一致。
        let r0 = [dep[0], dep[1], dep[2]];
        let rf = [arr[0], arr[1], arr[2]];
        let (v0, vf, _) = lambert_izzo(
            &r0,
            &rf,
            3600.0,
            398600.4418,
            TransferDirection::ShortWay,
            0,
        )
        .unwrap();
        let exp_dv1 =
            ((v0[0] - dep[3]).powi(2) + (v0[1] - dep[4]).powi(2) + (v0[2] - dep[5]).powi(2)).sqrt();
        let exp_dv2 =
            ((arr[3] - vf[0]).powi(2) + (arr[4] - vf[1]).powi(2) + (arr[5] - vf[2]).powi(2)).sqrt();
        assert_eq!(dv1[0], exp_dv1);
        assert_eq!(dv2[0], exp_dv2);
    }

    #[test]
    fn grid_states_serial_parallel_bit_identical() {
        let dep = [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0];
        let arr = [0.0, 42164.0, 0.0, -3.0, 0.0, 0.0];
        let lambert = LambertParams {
            mu_central: 398600.4418,
            long_way: false,
            revs: 0,
        };
        let dep_states = vec![dep; 3];
        let arr_states = vec![arr; 3 * 4];
        let tofs = [1800.0, 3600.0, 5400.0, 1e-3];
        let (s1, s2) = porkchop_grid_states_serial(&dep_states, &arr_states, 4, &tofs, &lambert);
        let (p1, p2) = porkchop_grid_states_parallel(&dep_states, &arr_states, 4, &tofs, &lambert);
        for (a, b) in s1.iter().zip(p1.iter()).chain(s2.iter().zip(p2.iter())) {
            assert!(a == b || (a.is_nan() && b.is_nan()));
        }
    }

    #[test]
    fn grid_specs_serial_parallel_bit_identical() {
        let dep = TerminalSpec::Orbit {
            state0: DRO_SEED,
            t0: 0.0,
            period: DRO_PERIOD,
        };
        let arr = dep;
        let params = prop_params();
        let lambert = LambertParams {
            mu_central: 1.0,
            long_way: false,
            revs: 0,
        };
        let t_dep = [0.0, 0.7, 1.4];
        let tofs = [0.5, 1.5, 2.5];
        let (s1, s2) =
            porkchop_grid_serial(&t_dep, &tofs, &dep, &arr, Some(&params), &lambert).unwrap();
        let (p1, p2) =
            porkchop_grid_parallel(&t_dep, &tofs, &dep, &arr, Some(&params), &lambert).unwrap();
        for (a, b) in s1.iter().zip(p1.iter()).chain(s2.iter().zip(p2.iter())) {
            assert!(a == b || (a.is_nan() && b.is_nan()));
        }
    }
}
