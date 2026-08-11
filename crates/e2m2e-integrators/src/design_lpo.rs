//! LPO 族设计的分层网格搜索（纯 Rust + Rayon 并行）。
//!
//! 对齐 Python `design_lpo`（`family/cr3bp_orbits.py:1004-1140`）：以 x₀ 为族
//! 参数，粗网格（30 点）→ 细网格（15 点）→ 局部二分（10 步）三层搜索，
//! 逼近目标振幅（距 L4/L5 径向距离均值，无量纲）。
//!
//! 复用 M1/M2 数值原子（**不重写**）：
//! - [`crate::lpo_correction::correct_lpo_full_period`]：全周期 Newton 修正器
//! - [`crate::lpo_family::lpo_initial_guess`]：线性化长周期模态初猜
//! - [`crate::lpo_family::l45_distance`]：传播一周的振幅测量
//!
//! # 并行化（照 ADR 0017 / `multiple_shooting.rs` 范式）
//!
//! - **粗/细网格**：候选点彼此独立（每点一个 Newton 修正 + 振幅测量），用
//!   Rayon `par_iter` + `collect` 保序并行；评估完再按序扫描——保序下模拟
//!   Python 串行 `_grid_search` 的"首个命中提前返回"语义，与串行**位级一致**。
//! - **局部二分**：逐点依赖前一点（`seed_orbit` 更新、区间收窄），串行。
//! - **开关**：`E2M2E_LPO_PARALLEL=0` 强制串行（对照位级一致，沿用
//!   `E2M2E_MS_PARALLEL` / `E2M2E_SEARCH_PARALLEL` 命名）。
//!
//! 网格修正内部直接调纯 Rust `propagate_cr3bp_stm` / `propagate_cr3bp`
//! （CR3BP 纯数学，无 cspice FFI），天然可并行，无需 `multiple_shooting`
//! 的 StrictGuard / ephem_cache 前提。

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::lpo_correction::correct_lpo_full_period;
use crate::lpo_family::{l45_distance, lpo_initial_guess};

/// 粗网格点数（对齐 Python `design_lpo` 第 1 步）。
const COARSE_POINTS: usize = 30;
/// 细网格点数（对齐 Python 第 2 步）。
const FINE_POINTS: usize = 15;
/// 局部二分步数（对齐 Python 第 3 步）。
const BISECTION_STEPS: usize = 10;
/// 网格 x₀ 下界偏移：`x_lo = lp_x - 0.20`（大振幅/马蹄端）。
const X_LO_OFFSET: f64 = 0.20;
/// 网格 x₀ 上界偏移：`x_hi = lp_x + 0.05`（小振幅端）。
const X_HI_OFFSET: f64 = 0.05;
/// 初猜固定振幅（km，对齐 `_correct_lpo` 的
/// `compute_lpo_initial_guess(..., amplitude_km=1000.0)`）。
const INITIAL_GUESS_AMPLITUDE_KM: f64 = 1000.0;
/// 回退容差的大振幅兜底（km，`fallback_tol = max(10·tol_du, 1000/du)`）。
const FALLBACK_TOL_KM: f64 = 1000.0;

/// LPO 设计结果。
#[derive(Debug, Clone)]
pub struct DesignLpoResult {
    /// 修正后的初始状态（6 维，x₀ 固定、z₀=ż₀=0 平面）。
    pub state: [f64; 6],
    /// 修正后的全周期（无量纲时间）。
    pub period: f64,
    /// 是否收敛（`Ok` 结果恒为 `true`；失败走 `Err`）。
    pub converged: bool,
    /// 实际振幅（无量纲，`0.5*(d_min+d_max)`）。
    pub amplitude_du: f64,
    /// 最优轨道修正的迭代次数。
    pub iterations: usize,
}

/// 设计参数上下文（网格搜索共用，闭包只读捕获）。
struct DesignLpoCtx<'a> {
    point: u8,
    target_du: f64,
    tol_du: f64,
    mu: f64,
    char_length_km: f64,
    omega_l: f64,
    v_l: &'a [f64],
    x_l: &'a [f64; 3],
    rtol: f64,
    atol: f64,
}

/// 单个网格候选点的收敛轨道（含振幅与误差）。
#[derive(Debug, Clone, Copy)]
struct GridHit {
    x0: f64,
    state: [f64; 6],
    period: f64,
    iterations: usize,
    amplitude_du: f64,
    err: f64,
}

impl GridHit {
    fn to_result(self) -> DesignLpoResult {
        DesignLpoResult {
            state: self.state,
            period: self.period,
            converged: true,
            amplitude_du: self.amplitude_du,
            iterations: self.iterations,
        }
    }
}

/// 单个网格候选点的评估结果。
enum GridPointOutcome {
    /// 修正失败（发散 / 停滞 / 周期异常 / 传播失败）→ 跳过该点。对齐
    /// Python `_correct_lpo` 抛 `Cr3bpOrbitError` 被 `_grid_search` 捕获跳过。
    Skip,
    /// 收敛轨道。
    Hit(GridHit),
}

/// 在 x₀ 处修正 LPO 并测量振幅（对齐 Python `_correct_lpo` + `measure`）。
///
/// `seed = None`：用线性化长周期模态初猜（固定振幅 `INITIAL_GUESS_AMPLITUDE_KM`
/// = 1000 km，覆盖 x₀）；`seed = Some((state, period))`：用 seed 轨道状态作
/// 初猜（`correct_lpo_full_period` 内部覆盖 x₀、z₀=ż₀=0，与 Python
/// `_correct_lpo` 的 seed 分支一致）。
///
/// 返回 `Ok(Skip)` = 修正失败（对齐 `Cr3bpOrbitError` → 网格跳过）；
/// `Ok(Hit)` = 收敛轨道；`Err` = 硬错误（初猜参数非法 / 振幅测量传播失败，
/// 对应 Python 中不可被 `_grid_search` 捕获的异常）。
fn correct_lpo_grid_point(
    ctx: &DesignLpoCtx<'_>,
    x0: f64,
    seed: Option<(&[f64; 6], f64)>,
) -> Result<GridPointOutcome, String> {
    let (initial_state, period_guess) = match seed {
        Some((state, period)) => (*state, period),
        None => {
            let (state, period) = lpo_initial_guess(
                ctx.point,
                INITIAL_GUESS_AMPLITUDE_KM,
                ctx.char_length_km,
                ctx.omega_l,
                ctx.v_l,
                ctx.x_l,
            )?;
            (state, period)
        }
    };

    // 修正失败统一视为"该点跳过"：`Err`（传播失败 / 收敛后周期越界）与
    // `Ok(converged=false)`（发散 / 停滞 / 超迭代）都对齐 Python `_correct_lpo`
    // 抛 `Cr3bpOrbitError` → `_grid_search` `continue`。
    let result =
        match correct_lpo_full_period(ctx.mu, x0, &initial_state, period_guess, ctx.rtol, ctx.atol)
        {
            Ok(r) => r,
            Err(_) => return Ok(GridPointOutcome::Skip),
        };
    if !result.converged {
        return Ok(GridPointOutcome::Skip);
    }

    let (d_min, d_max) = l45_distance(
        ctx.mu,
        ctx.point,
        &result.state,
        result.period,
        ctx.rtol,
        ctx.atol,
    )?;
    let amplitude_du = 0.5 * (d_min + d_max);
    Ok(GridPointOutcome::Hit(GridHit {
        x0,
        state: result.state,
        period: result.period,
        iterations: result.iterations,
        amplitude_du,
        err: (amplitude_du - ctx.target_du).abs(),
    }))
}

/// 从候选点评估结果序列扫描全局最优，并模拟串行提前返回。
///
/// 候选点按 x₀ 升序（`np.linspace` 端点闭）评估完毕（并行 `collect` 保序）。
/// 按序扫描：全局最优只在 `err` **严格更小**时更新（对齐 Python
/// `if err < b_err`）；遇首个 `err <= tol_du` 即停（对齐 Python
/// `if err <= tol_du: return x0, orb, err`）。此前所有点的 err 均 > tol_du，
/// 故命中点 err 必为已见最小值 → 命中点即全局最优。
fn scan_grid(
    outcomes: Vec<Result<GridPointOutcome, String>>,
    tol_du: f64,
) -> Result<Option<GridHit>, String> {
    let mut best: Option<GridHit> = None;
    for outcome in outcomes {
        let hit = match outcome? {
            GridPointOutcome::Skip => continue,
            GridPointOutcome::Hit(h) => h,
        };
        let hit_err = hit.err;
        if best.as_ref().is_none_or(|b| hit_err < b.err) {
            best = Some(hit);
        }
        if hit_err <= tol_du {
            break;
        }
    }
    Ok(best)
}

/// 粗/细网格搜索（对齐 Python `_grid_search`）：在 `[x_lo, x_hi]` 均匀采样
/// `n_pts` 点，返回全局最优候选（有命中时即首个命中点，二者一致）。
///
/// 候选点独立 → Rayon `par_iter` 并行 + `collect` 保序；`parallel=false`
/// 走串行。扫描后结果与串行逐位一致。
fn grid_search(
    ctx: &DesignLpoCtx<'_>,
    x_lo: f64,
    x_hi: f64,
    n_pts: usize,
    seed: Option<(&[f64; 6], f64)>,
    parallel: bool,
) -> Result<Option<GridHit>, String> {
    debug_assert!(n_pts >= 2, "linspace 至少 2 点");
    let evaluate = |i: usize| -> Result<GridPointOutcome, String> {
        // np.linspace(x_lo, x_hi, n_pts) 端点闭
        let x0 = x_lo + i as f64 * (x_hi - x_lo) / (n_pts as f64 - 1.0);
        correct_lpo_grid_point(ctx, x0, seed)
    };
    let outcomes: Vec<Result<GridPointOutcome, String>> = if parallel {
        use rayon::prelude::*;
        (0..n_pts).into_par_iter().map(evaluate).collect()
    } else {
        (0..n_pts).map(evaluate).collect()
    };
    scan_grid(outcomes, ctx.tol_du)
}

/// 指定并行模式的 `design_lpo`（测试并行/串行位级一致用）。
#[allow(clippy::too_many_arguments)]
fn design_lpo_impl(
    point: u8,
    amplitude_km: f64,
    mu: f64,
    char_length_km: f64,
    omega_l: f64,
    v_l: &[f64],
    x_l: &[f64; 3],
    rtol: f64,
    atol: f64,
    tol_km: f64,
    parallel: bool,
) -> Result<DesignLpoResult, String> {
    // 入参校验（报错风格对齐 `lpo_initial_guess`）
    if point != 4 && point != 5 {
        return Err(format!("point 必须为 4（L4）或 5（L5），当前 {point}"));
    }
    if !char_length_km.is_finite() || char_length_km <= 0.0 {
        return Err(format!(
            "特征长度无效（char_length_km={char_length_km:.3}）"
        ));
    }
    if !amplitude_km.is_finite() {
        return Err(format!("振幅无效（amplitude_km={amplitude_km:.3}）"));
    }
    if !omega_l.is_finite() || omega_l <= 0.0 {
        return Err(format!("长周期频率无效（omega_l={omega_l:.6}）"));
    }
    if v_l.len() != 12 {
        return Err(format!(
            "v_l 须为 12 个 f64（6 维复向量实部/虚部交错展平），当前 {}",
            v_l.len()
        ));
    }
    if !mu.is_finite() || !(0.0..1.0).contains(&mu) {
        return Err(format!("质量参数无效（mu={mu:.6}）"));
    }
    if rtol <= 0.0 || atol <= 0.0 {
        return Err(format!("传播容差无效（rtol={rtol:.1e}, atol={atol:.1e}）"));
    }
    if !tol_km.is_finite() || tol_km <= 0.0 {
        return Err(format!("振幅容差无效（tol_km={tol_km:.3}）"));
    }

    let target_du = amplitude_km / char_length_km;
    let tol_du = tol_km / char_length_km;
    let lp_x = 0.5 - mu;
    let x_lo = lp_x - X_LO_OFFSET;
    let x_hi = lp_x + X_HI_OFFSET;

    let ctx = DesignLpoCtx {
        point,
        target_du,
        tol_du,
        mu,
        char_length_km,
        omega_l,
        v_l,
        x_l,
        rtol,
        atol,
    };

    // 第 1 步：粗网格（30 点，seed=None）。无任何收敛轨道 → 整体失败。
    let mut best = grid_search(&ctx, x_lo, x_hi, COARSE_POINTS, None, parallel)?
        .ok_or_else(|| format!("LPO(L{point}, amp={amplitude_km:.0} km) 网格搜索无收敛轨道"))?;
    if best.err <= tol_du {
        return Ok(best.to_result());
    }

    // 第 2 步：细网格精化（在最佳候选 ±2 步长内，15 点，seed=best_orbit）。
    let dx = (x_hi - x_lo) / COARSE_POINTS as f64;
    let refine_lo = x_lo.max(best.x0 - 2.0 * dx);
    let refine_hi = x_hi.min(best.x0 + 2.0 * dx);
    if let Some(fine_best) = grid_search(
        &ctx,
        refine_lo,
        refine_hi,
        FINE_POINTS,
        Some((&best.state, best.period)),
        parallel,
    )? {
        // 严格更小才更新（对齐 Python `if rerr < best_err`）
        if fine_best.err < best.err {
            best = fine_best;
        }
    }
    if best.err <= tol_du {
        return Ok(best.to_result());
    }

    // 第 3 步：局部二分精化（在最佳候选 ±1 步长内，10 步，串行）。
    // 逐点依赖（seed_orbit 随最优更新、区间随振幅符号收窄），不并行。
    let refine_lo = x_lo.max(best.x0 - dx);
    let refine_hi = x_hi.min(best.x0 + dx);
    let mut lo = refine_lo;
    let mut hi = refine_hi;
    let mut seed_orbit = (best.state, best.period);
    for _ in 0..BISECTION_STEPS {
        let x_mid = 0.5 * (lo + hi);
        let outcome = correct_lpo_grid_point(&ctx, x_mid, Some((&seed_orbit.0, seed_orbit.1)))?;
        let hit = match outcome {
            GridPointOutcome::Skip => break, // 修正失败 → 退出二分
            GridPointOutcome::Hit(h) => h,
        };
        if hit.err < best.err {
            seed_orbit = (hit.state, hit.period);
            best = hit;
        }
        if hit.err <= tol_du {
            return Ok(hit.to_result());
        }
        if hit.amplitude_du > target_du {
            lo = x_mid;
        } else {
            hi = x_mid;
        }
    }

    // 回退容差：max(10·tol_du, 1000 km / du)（大振幅区域允许合理误差）。
    let fallback_tol = (10.0 * tol_du).max(FALLBACK_TOL_KM / char_length_km);
    if best.err <= fallback_tol {
        return Ok(best.to_result());
    }

    Err(format!(
        "LPO(L{point}, amp={amplitude_km:.0} km) 未命中目标（最佳误差 {:.0} km）",
        best.err * char_length_km
    ))
}

/// 生成指定振幅的 L4/L5 LPO 周期轨道（纯 Rust 核心，M3）。
///
/// # 语义（对齐 Python `design_lpo`）
///
/// 以 x₀ 为族参数，粗网格（30 点）→ 细网格（15 点）→ 局部二分（10 步）
/// 逼近目标振幅（`0.5*(d_min+d_max)`，`l45_distance` 测量）。命中容差
/// `tol_km` 提前返回；否则回退容差 `max(10·tol_km, 1000 km)` 放行；都未命中
/// 返回 `Err`。
///
/// # 参数
/// - `point`：4（L4）或 5（L5）
/// - `amplitude_km`：目标振幅（km）
/// - `mu`：CR3BP 质量参数 μ = m₂/(m₁+m₂)
/// - `char_length_km`：`system.characteristic_length`（km）
/// - `omega_l` / `v_l` / `x_l`：L4/L5 长周期模态常量（Python 预计算传参，
///   见 `lpo_family::lpo_initial_guess`）
/// - `rtol` / `atol`：传播容差（传给修正器与振幅测量）
/// - `tol_km`：振幅匹配容差（km，默认 20.0）
///
/// # 并行
///
/// 粗/细网格用 Rayon `par_iter` 并行（候选点独立）；`E2M2E_LPO_PARALLEL=0`
/// 强制串行（对照位级一致）。局部二分串行。
///
/// # 返回
/// 收敛轨道：`Ok(DesignLpoResult)`；网格无收敛轨道 / 未命中回退容差 / 入参
/// 非法：`Err(String)`。
#[allow(clippy::too_many_arguments)]
pub fn design_lpo_rust(
    point: u8,
    amplitude_km: f64,
    mu: f64,
    char_length_km: f64,
    omega_l: f64,
    v_l: &[f64],
    x_l: &[f64; 3],
    rtol: f64,
    atol: f64,
    tol_km: f64,
) -> Result<DesignLpoResult, String> {
    // 并行开关：E2M2E_LPO_PARALLEL=0 强制串行（对照位级一致，沿用
    // E2M2E_MS_PARALLEL / E2M2E_SEARCH_PARALLEL 范式）。
    let parallel = std::env::var("E2M2E_LPO_PARALLEL").map_or(true, |v| v != "0");
    design_lpo_impl(
        point,
        amplitude_km,
        mu,
        char_length_km,
        omega_l,
        v_l,
        x_l,
        rtol,
        atol,
        tol_km,
        parallel,
    )
}

/// Python 接口：生成指定振幅的 L4/L5 LPO 周期轨道。
///
/// 参数与返回值对齐 Python `design_lpo`：
/// - `v_l` 为 6 维复特征向量展平成 12 个 f64（实部/虚部交错）。
/// - 内部释放 GIL，网格搜索走 Rayon 真并行。
#[pyfunction]
#[pyo3(signature = (point, amplitude_km, mu, char_length_km, omega_l, v_l, x_l, rtol, atol, tol_km))]
#[allow(clippy::too_many_arguments)]
pub fn design_lpo_py(
    point: u8,
    amplitude_km: f64,
    mu: f64,
    char_length_km: f64,
    omega_l: f64,
    v_l: Vec<f64>,
    x_l: [f64; 3],
    rtol: f64,
    atol: f64,
    tol_km: f64,
    py: Python<'_>,
) -> PyResult<PyObject> {
    let x_l_ref = x_l;
    let result = py
        .allow_threads(|| {
            design_lpo_rust(
                point,
                amplitude_km,
                mu,
                char_length_km,
                omega_l,
                &v_l,
                &x_l_ref,
                rtol,
                atol,
                tol_km,
            )
        })
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    let dict = PyDict::new(py);
    dict.set_item("state", result.state.to_vec())?;
    dict.set_item("period", result.period)?;
    dict.set_item("converged", result.converged)?;
    dict.set_item("amplitude_du", result.amplitude_du)?;
    dict.set_item("iterations", result.iterations)?;
    Ok(dict.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 地月系质量参数（对齐 `earth_moon_system()` 的 `EARTH_MOON_MU`）。
    const MU_EARTH_MOON: f64 = 0.0121506683;
    /// 特征长度（km，`CHAR_LENGTH_KM`）。
    const CHAR_LENGTH_KM: f64 = 384400.0;

    /// L4 长周期模态常量（Python `_triangular_modes(earth_moon_system, 4)` 预计算）。
    const OMEGA_L_L4: f64 = 0.2982092837049938;
    const V_L_L4: [f64; 12] = [
        -0.8221319912182276,
        0.0,
        0.4455148161239162,
        -0.20964075061607065,
        6.795037248874498e-17,
        -1.327803042151104e-17,
        8.062878806512508e-16,
        -0.24516739221214762,
        0.06251681807659477,
        0.13285665419627535,
        -7.726763846400462e-17,
        1.509869654199918e-17,
    ];
    const X_L_L4: [f64; 3] = [0.4878493317, 0.8660254037844386, 0.0];

    /// L5 长周期模态常量（xy 镜像，见 `lpo_family.rs` 测试）。
    const V_L_L5: [f64; 12] = [
        -0.8221319912182276,
        0.0,
        -0.4455148161239162,
        -0.20964075061607065,
        6.795037248874498e-17,
        1.327803042151104e-17,
        -8.062878806512508e-16,
        -0.24516739221214762,
        0.06251681807659477,
        -0.13285665419627535,
        7.726763846400462e-17,
        1.509869654199918e-17,
    ];
    const X_L_L5: [f64; 3] = [0.4878493317, -0.8660254037844386, 0.0];

    /// 端到端：`design_lpo(4, 50000 km)` 收敛，振幅接近目标（对齐 Python
    /// `test_design_lpo_l4_converges` / `test_amplitude_matches_target`）。
    #[test]
    fn design_lpo_l4_end_to_end() {
        let res = design_lpo_impl(
            4,
            50000.0,
            MU_EARTH_MOON,
            CHAR_LENGTH_KM,
            OMEGA_L_L4,
            &V_L_L4,
            &X_L_L4,
            1e-12,
            1e-12,
            20.0,
            true,
        )
        .expect("L4 50000km 设计应收敛");
        assert!(res.converged, "设计成功即已收敛");
        assert!(
            (10.0..=50.0).contains(&res.period),
            "周期 {} 应在 [10, 50]",
            res.period
        );
        // z₀=ż₀=0 平面约束（对齐 `_correct_lpo`）
        assert!(res.state[2].abs() < 1e-14, "z₀ = {} 应为 0", res.state[2]);
        assert!(res.state[5].abs() < 1e-14, "ż₀ = {} 应为 0", res.state[5]);
        // 振幅对齐目标（km）
        let amp_km = res.amplitude_du * CHAR_LENGTH_KM;
        assert!(
            (amp_km - 50000.0).abs() < 50.0,
            "振幅 {amp_km:.1} km 应接近 50000 km（±50）"
        );
        // y₀ 在 L4 侧为正
        assert!(res.state[1] > 0.0, "L4 的 y₀ = {} 应为正", res.state[1]);
    }

    /// 端到端：L5 镜像（y₀ 为负），标量特征（周期/振幅）与 L4 一致。
    #[test]
    fn design_lpo_l5_mirror() {
        let res4 = design_lpo_impl(
            4,
            50000.0,
            MU_EARTH_MOON,
            CHAR_LENGTH_KM,
            OMEGA_L_L4,
            &V_L_L4,
            &X_L_L4,
            1e-12,
            1e-12,
            20.0,
            true,
        )
        .expect("L4 设计应成功");
        let res5 = design_lpo_impl(
            5,
            50000.0,
            MU_EARTH_MOON,
            CHAR_LENGTH_KM,
            OMEGA_L_L4,
            &V_L_L5,
            &X_L_L5,
            1e-12,
            1e-12,
            20.0,
            true,
        )
        .expect("L5 设计应成功");
        assert!(res5.state[1] < 0.0, "L5 的 y₀ = {} 应为负", res5.state[1]);
        assert!(
            (res5.period - res4.period).abs() < 1e-3,
            "L4/L5 周期应一致（{} vs {}）",
            res5.period,
            res4.period
        );
        assert!(
            (res5.amplitude_du - res4.amplitude_du).abs() < 1e-4,
            "L4/L5 振幅应一致（{} vs {}）",
            res5.amplitude_du,
            res4.amplitude_du
        );
    }

    /// 并行与串行**位级一致**：同一输入，`parallel=true/false` 的
    /// state / period / amplitude_du / iterations 逐位相等。
    ///
    /// 网格候选点独立、`collect` 保序、扫描逻辑相同 → 两者结果逐位相同。
    #[test]
    fn design_lpo_parallel_matches_serial() {
        let par = design_lpo_impl(
            4,
            50000.0,
            MU_EARTH_MOON,
            CHAR_LENGTH_KM,
            OMEGA_L_L4,
            &V_L_L4,
            &X_L_L4,
            1e-12,
            1e-12,
            20.0,
            true,
        )
        .expect("并行设计应成功");
        let ser = design_lpo_impl(
            4,
            50000.0,
            MU_EARTH_MOON,
            CHAR_LENGTH_KM,
            OMEGA_L_L4,
            &V_L_L4,
            &X_L_L4,
            1e-12,
            1e-12,
            20.0,
            false,
        )
        .expect("串行设计应成功");

        for i in 0..6 {
            assert!(
                par.state[i] == ser.state[i],
                "并行/串行 state[{}] 应位级一致（{} vs {}）",
                i,
                par.state[i],
                ser.state[i]
            );
        }
        assert!(
            par.period == ser.period,
            "并行/串行 period 应位级一致（{} vs {}）",
            par.period,
            ser.period
        );
        assert!(
            par.amplitude_du == ser.amplitude_du,
            "并行/串行 amplitude_du 应位级一致（{} vs {}）",
            par.amplitude_du,
            ser.amplitude_du
        );
        assert!(
            par.iterations == ser.iterations,
            "并行/串行 iterations 应一致（{} vs {}）",
            par.iterations,
            ser.iterations
        );
    }

    /// 入参非法报错：point、特征长度、振幅、频率、v_l 长度、mu、容差、tol_km。
    #[test]
    fn design_lpo_rejects_invalid_input() {
        let good = (4u8, 50000.0, MU_EARTH_MOON, CHAR_LENGTH_KM, OMEGA_L_L4);
        assert!(design_lpo_impl(
            3, good.1, good.2, good.3, good.4, &V_L_L4, &X_L_L4, 1e-12, 1e-12, 20.0, true
        )
        .is_err());
        assert!(design_lpo_impl(
            6, good.1, good.2, good.3, good.4, &V_L_L4, &X_L_L4, 1e-12, 1e-12, 20.0, true
        )
        .is_err());
        assert!(design_lpo_impl(
            4,
            f64::NAN,
            good.2,
            good.3,
            good.4,
            &V_L_L4,
            &X_L_L4,
            1e-12,
            1e-12,
            20.0,
            true
        )
        .is_err());
        assert!(design_lpo_impl(
            4, good.1, good.2, 0.0, good.4, &V_L_L4, &X_L_L4, 1e-12, 1e-12, 20.0, true
        )
        .is_err());
        assert!(design_lpo_impl(
            4, good.1, good.2, good.3, 0.0, &V_L_L4, &X_L_L4, 1e-12, 1e-12, 20.0, true
        )
        .is_err());
        assert!(design_lpo_impl(
            4,
            good.1,
            good.2,
            good.3,
            good.4,
            &V_L_L4[..6],
            &X_L_L4,
            1e-12,
            1e-12,
            20.0,
            true
        )
        .is_err());
        assert!(design_lpo_impl(
            4, good.1, 0.0, good.3, good.4, &V_L_L4, &X_L_L4, 1e-12, 1e-12, 20.0, true
        )
        .is_err());
        assert!(design_lpo_impl(
            4, good.1, 1.5, good.3, good.4, &V_L_L4, &X_L_L4, 1e-12, 1e-12, 20.0, true
        )
        .is_err());
        assert!(design_lpo_impl(
            4, good.1, good.2, good.3, good.4, &V_L_L4, &X_L_L4, 0.0, 1e-12, 20.0, true
        )
        .is_err());
        assert!(design_lpo_impl(
            4, good.1, good.2, good.3, good.4, &V_L_L4, &X_L_L4, 1e-12, 1e-12, -5.0, true
        )
        .is_err());
    }

    /// 网格无收敛轨道 / 未命中目标 → `Err`（对齐 Python 抛 `Cr3bpOrbitError`）。
    ///
    /// 取远超 LPO 族振幅上界的目标（500 万 km），网格最优误差必然远超回退
    /// 容差（~1000 km），设计失败。
    #[test]
    fn design_lpo_out_of_family_returns_err() {
        let err = design_lpo_impl(
            4,
            5_000_000.0,
            MU_EARTH_MOON,
            CHAR_LENGTH_KM,
            OMEGA_L_L4,
            &V_L_L4,
            &X_L_L4,
            1e-12,
            1e-12,
            20.0,
            true,
        )
        .expect_err("超族振幅应设计失败");
        assert!(
            err.contains("未命中目标") || err.contains("无收敛轨道"),
            "错误信息应说明失败原因，当前 {err}"
        );
    }
}
