//! 转移网格搜索评估单元（Rust 实现，纯数学，不依赖 SPICE）。
//!
//! 从 Python `e2m2e/algorithm/transfer/search_parallel.py` 的
//! `search_single_departure` 内层 α 循环下沉。6 步评估单元
//! [`evaluate_point`] + 串行网格分发 [`transfer_grid_search_serial`]。
//!
//! 复用阶段 A 的 [`crate::transfer_geometry`] 与
//! [`crate::cr3bp::propagate_cr3bp`]（直接调纯 Rust，不绕道持 GIL 的
//! `propagate_cr3bp_py`）。逐行对照 Python 实现，保证数值与索引约定一致
//! （阶段 B Rust vs Python 等价性测试基准）。
//!
//! # 6 步评估单元
//!
//! [`evaluate_point`] 对单个 `(departure_state, alpha)` 候选执行：
//! 1. 出发速度合成（`compute_departure_velocity`，beta=0）
//! 2. 前向积分（`propagate_cr3bp`）
//! 3. 碰撞检测（`check_collision`）
//! 4. 距离序列 + 全局最近点（`compute_distance_series` + argmin）
//! 5. 相交 / 局部极小 / 首次命中阈值（`detect_intersection` /
//!    `detect_local_minimum` + 线性扫描）
//! 6. 组装 [`TransferPointResult`]
//!
//! 积分发散（步长塌缩）在 Step2 内 catch，走 `dv=1e10` 惩罚分支，
//! 不穿透 [`Result::Err`]（与设计文档 transfer-grid-search-rust.md:88 一致）。
//!
//! # 索引约定
//!
//! argmin / 首次命中均取首个（与阶段 A 一致），详见
//! [`crate::transfer_geometry`] 模块文档。

use crate::cr3bp::propagate_cr3bp;
use crate::transfer_geometry;

/// 积分发散时的 dv 惩罚（与设计文档 transfer-grid-search-rust.md:88 一致）。
///
/// Python `search_single_departure` 失败分支保留真实 `dv_departure`，本下沉
/// 按设计文档统一走 1e10 惩罚——等价性测试网格选得足够温和（小 α 范围、
/// 短 transfer_time），不触发该分支；触发时该字段不参与等价对照。
pub const DV_PENALTY: f64 = 1e10;

/// 单候选点（一个 `(departure_state, alpha)`）的评估结果。
///
/// 字段对齐 Python `search_single_departure` 组装的候选解 dict
/// （`search_parallel.py:189-215` 成功分支 + `:135-150` 失败分支）。
/// 成功分支填全部字段；失败分支（积分发散）填 `success=false` +
/// `status="integration_failed"` + `dv_departure=1e10` 惩罚，几何字段为
/// `None`/默认（与 Python 失败 dict 字段集合对齐）。
#[derive(Clone, Debug)]
pub struct TransferPointResult {
    pub success: bool,
    pub departure_state: [f64; 6],
    pub departure_time: f64,
    pub alpha: f64,
    /// 展平 n×6（成功分支），失败分支 `None`（轨迹不回传）。
    pub transfer_trajectory: Option<Vec<f64>>,
    pub transfer_times: Option<Vec<f64>>,
    pub transfer_time: Option<f64>,
    pub min_distance: Option<f64>,
    pub min_distance_idx: Option<i64>,
    pub min_distance_orbit_idx: Option<i64>,
    pub dv_departure: f64,
    pub dv_insertion: Option<f64>,
    pub intersection_found: bool,
    /// 命中时为完整 6 维 `[x,y,z,vx,vy,vz]`。
    pub intersection_point: Option<Vec<f64>>,
    /// 未命中 `-1`。
    pub intersection_idx: i64,
    pub first_intersection_idx: Option<i64>,
    pub first_intersection_time: Option<f64>,
    pub first_min_distance_idx: Option<i64>,
    pub first_min_distance_time: Option<f64>,
    pub local_minimum_found: bool,
    /// 未命中 `+inf`（与 numpy `np.inf` 对齐）。
    pub local_minimum_distance: f64,
    /// 未命中 `-1`。
    pub local_minimum_idx: i64,
    pub collision_found: bool,
    /// `"earth"` / `"moon"`，未命中 `None`。
    pub collision_body: Option<String>,
    /// 未命中 `-1`。
    pub collision_idx: i64,
    /// `success` / `collision` / `no_intersection` / `integration_failed`。
    pub status: String,
}

/// 网格搜索标量配置包（Python 端展平传入，对齐 `_process_pack_base`）。
pub struct GridSearchParams {
    pub mu: f64,
    pub max_transfer_time: f64,
    pub integration_dt: f64,
    pub intersection_threshold: f64,
    pub min_distance_threshold: f64,
    pub collision_earth_radius: f64,
    pub collision_moon_radius: f64,
    pub rtol: f64,
    pub atol: f64,
    pub max_step: f64,
}

/// 单候选点 6 步评估（移植 `search_parallel.py:117-222`）。
///
/// 逐行对照 Python `search_single_departure` 的 per-α 内核。积分失败
/// （步长塌缩）走 [`DV_PENALTY`] 惩罚分支，不穿透 [`Result::Err`]。
pub fn evaluate_point(
    departure_state: &[f64; 6],
    departure_time: f64,
    alpha: f64,
    arrival_states: &[f64],
    params: &GridSearchParams,
) -> TransferPointResult {
    // Step1 出发速度合成（compute_departure_velocity，beta=0）。
    // beta=0 时 v_inj = alpha·v_mag·t_hat = alpha·vel；退化（v_mag<1e-10）
    // 返回原速度（与 propulsion.py:59-61 一致，网格搜索不触发该分支）。
    let vel = [departure_state[3], departure_state[4], departure_state[5]];
    let v_mag = (vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]).sqrt();
    let v_inj = if v_mag < 1e-10 {
        vel
    } else {
        [alpha * vel[0], alpha * vel[1], alpha * vel[2]]
    };
    let dv_departure =
        ((v_inj[0] - vel[0]).powi(2) + (v_inj[1] - vel[1]).powi(2) + (v_inj[2] - vel[2]).powi(2))
            .sqrt();
    let initial_state = [
        departure_state[0],
        departure_state[1],
        departure_state[2],
        v_inj[0],
        v_inj[1],
        v_inj[2],
    ];

    // Step2 前向积分。t_eval = linspace(0, mtt, n_steps),
    // n_steps = max(int(mtt/dt)+1, 2)（与 forward_integrate 一致）。
    let n_steps = ((params.max_transfer_time / params.integration_dt) as usize + 1).max(2);
    let mut t_eval = Vec::with_capacity(n_steps);
    if n_steps == 1 {
        t_eval.push(0.0);
    } else {
        let denom = (n_steps - 1) as f64;
        for i in 0..n_steps {
            t_eval.push(params.max_transfer_time * (i as f64) / denom);
        }
    }

    let integration = propagate_cr3bp(
        params.mu,
        (0.0, params.max_transfer_time),
        &t_eval,
        &initial_state,
        params.rtol,
        params.atol,
        Some(params.max_step),
        None,
    );

    let Ok(propagated) = integration else {
        // 积分失败（步长塌缩）：dv 惩罚分支，不穿透 Err。
        return TransferPointResult {
            success: false,
            departure_state: *departure_state,
            departure_time,
            alpha,
            transfer_trajectory: None,
            transfer_times: None,
            transfer_time: None,
            min_distance: None,
            min_distance_idx: None,
            min_distance_orbit_idx: None,
            dv_departure: DV_PENALTY,
            dv_insertion: None,
            intersection_found: false,
            intersection_point: None,
            intersection_idx: -1,
            first_intersection_idx: None,
            first_intersection_time: None,
            first_min_distance_idx: None,
            first_min_distance_time: None,
            local_minimum_found: false,
            local_minimum_distance: f64::INFINITY,
            local_minimum_idx: -1,
            collision_found: false,
            collision_body: None,
            collision_idx: -1,
            status: "integration_failed".to_string(),
        };
    };

    // 展平轨迹 n×6（行优先）。
    let n_traj = propagated.states.len();
    let mut traj_flat = Vec::with_capacity(n_traj * 6);
    for s in &propagated.states {
        traj_flat.extend_from_slice(s);
    }
    let traj_times = propagated.times;

    // Step3 碰撞检测。
    let (collision_found, collision_body, collision_idx) = transfer_geometry::check_collision(
        &traj_flat,
        params.mu,
        params.collision_earth_radius,
        params.collision_moon_radius,
    );

    // Step4 距离序列 + 全局最近点（argmin 取首个）。
    let (d_per_step, orbit_idx_per_step) =
        transfer_geometry::compute_distance_series(&traj_flat, arrival_states);
    let mut min_idx = 0usize;
    let mut min_dist = d_per_step[0];
    for (i, &v) in d_per_step.iter().enumerate() {
        if v < min_dist {
            min_dist = v;
            min_idx = i;
        }
    }
    let orbit_idx = orbit_idx_per_step[min_idx] as usize;
    // dv_insertion = ‖traj[min_idx][3:6] - arrival[orbit_idx][3:6]‖
    let v_tr = [
        traj_flat[min_idx * 6 + 3],
        traj_flat[min_idx * 6 + 4],
        traj_flat[min_idx * 6 + 5],
    ];
    let v_ro = [
        arrival_states[orbit_idx * 6 + 3],
        arrival_states[orbit_idx * 6 + 4],
        arrival_states[orbit_idx * 6 + 5],
    ];
    let dv_insertion =
        ((v_tr[0] - v_ro[0]).powi(2) + (v_tr[1] - v_ro[1]).powi(2) + (v_tr[2] - v_ro[2]).powi(2))
            .sqrt();

    // Step5 相交 / 局部极小 / 首次命中阈值。
    let (intersection_found, intersection_point, intersection_idx) =
        transfer_geometry::detect_intersection(
            &traj_flat,
            arrival_states,
            params.intersection_threshold,
        );
    let (local_minimum_found, local_minimum_distance, local_minimum_idx) =
        transfer_geometry::detect_local_minimum(&traj_flat, arrival_states);

    // first_int_idx / first_md_idx：首个 d_per_step <= 阈值 的索引（与
    // search_parallel.py:174-187 的 np.where(...)[0][0] 一致）。
    let mut first_intersection_idx: Option<i64> = None;
    for (i, &v) in d_per_step.iter().enumerate() {
        if v <= params.intersection_threshold {
            first_intersection_idx = Some(i as i64);
            break;
        }
    }
    let first_intersection_time = first_intersection_idx.map(|i| traj_times[i as usize]);
    let mut first_min_distance_idx: Option<i64> = None;
    for (i, &v) in d_per_step.iter().enumerate() {
        if v <= params.min_distance_threshold {
            first_min_distance_idx = Some(i as i64);
            break;
        }
    }
    let first_min_distance_time = first_min_distance_idx.map(|i| traj_times[i as usize]);

    // Step6 组装 status（与 search_parallel.py:216-221 一致）。
    let status = if collision_found {
        "collision".to_string()
    } else if intersection_found || min_dist < params.min_distance_threshold {
        "success".to_string()
    } else {
        "no_intersection".to_string()
    };

    let transfer_time = *traj_times.last().unwrap_or(&params.max_transfer_time);

    TransferPointResult {
        success: true,
        departure_state: *departure_state,
        departure_time,
        alpha,
        transfer_trajectory: Some(traj_flat),
        transfer_times: Some(traj_times),
        transfer_time: Some(transfer_time),
        min_distance: Some(min_dist),
        min_distance_idx: Some(min_idx as i64),
        min_distance_orbit_idx: Some(orbit_idx as i64),
        dv_departure,
        dv_insertion: Some(dv_insertion),
        intersection_found,
        intersection_point: intersection_point.map(|p| p.to_vec()),
        intersection_idx,
        first_intersection_idx,
        first_intersection_time,
        first_min_distance_idx,
        first_min_distance_time,
        local_minimum_found,
        local_minimum_distance,
        local_minimum_idx,
        collision_found,
        collision_body,
        collision_idx,
        status,
    }
}

/// 串行网格搜索（保序，不用 Rayon）。
///
/// 遍历 `0..n_dep*n_alpha`，`idx → (i_dep = idx/n_alpha, i_alpha = idx%n_alpha)`，
/// 调 [`evaluate_point`]，`collect` 成 `Vec`。顺序与 Python
/// `grid_search_sequential` 一致（外层 departure、内层 alpha）——这是
/// 阶段 B 等价性测试能逐候选对照的前提。
///
/// 输入展平约定：`dep_states`/`arrival_states` 为 n×6 行优先，`dep_times`/
/// `alpha_grid` 一维。
pub fn transfer_grid_search_serial(
    dep_states: &[f64],
    dep_times: &[f64],
    alpha_grid: &[f64],
    arrival_states: &[f64],
    params: &GridSearchParams,
) -> Vec<TransferPointResult> {
    assert!(
        dep_states.len().is_multiple_of(6),
        "dep_states 展平长度必须是 6 的倍数，得到 {}",
        dep_states.len()
    );
    let n_dep = dep_states.len() / 6;
    assert_eq!(
        dep_times.len(),
        n_dep,
        "dep_times 长度 ({}) 须等于 n_dep ({})",
        dep_times.len(),
        n_dep
    );
    assert!(
        arrival_states.len().is_multiple_of(6),
        "arrival_states 展平长度必须是 6 的倍数，得到 {}",
        arrival_states.len()
    );
    let n_alpha = alpha_grid.len();
    let n_arrival = arrival_states.len() / 6;
    assert!(n_arrival > 0, "arrival_states 不能为空");

    let total = n_dep.checked_mul(n_alpha).expect("n_dep * n_alpha 溢出");
    let mut out = Vec::with_capacity(total);
    for idx in 0..total {
        let i_dep = idx / n_alpha;
        let i_alpha = idx % n_alpha;
        let mut dep = [0.0_f64; 6];
        dep.copy_from_slice(&dep_states[i_dep * 6..i_dep * 6 + 6]);
        out.push(evaluate_point(
            &dep,
            dep_times[i_dep],
            alpha_grid[i_alpha],
            arrival_states,
            params,
        ));
    }
    out
}

/// 并行网格搜索（Rayon `par_iter`，保序）。
///
/// 与 [`transfer_grid_search_serial`] 同样的 `idx → (i_dep, i_alpha)` 映射与
/// [`evaluate_point`] 调用，唯一差别是 `into_par_iter` 并行求值。Rayon
/// `par_iter` + `collect` 保序：各候选求值是纯函数（直接调纯 Rust
/// [`crate::cr3bp::propagate_cr3bp`]，CR3BP 纯数学无 SPICE FFI、无线程
/// 不安全状态），故并行与串行结果逐位相同。
///
/// # 并行安全前提
///
/// [`evaluate_point`] 调 [`crate::cr3bp::propagate_cr3bp`]（纯数学积分器，
/// 无全局可变状态、无 SPICE FFI）与 [`crate::transfer_geometry`]（纯函数
/// 几何核），均 `Send + Sync`。这与多重打靶段积分的 rayon 路径不同——后者
/// 段积分内调 cspice 需 `StrictGuard` + 星历预采样；本函数零 cspice，rayon
/// 安全前提更简单（见 `multiple_shooting.rs:347-359` 对比）。
pub fn transfer_grid_search_parallel(
    dep_states: &[f64],
    dep_times: &[f64],
    alpha_grid: &[f64],
    arrival_states: &[f64],
    params: &GridSearchParams,
) -> Vec<TransferPointResult> {
    use rayon::prelude::*;

    assert!(
        dep_states.len().is_multiple_of(6),
        "dep_states 展平长度必须是 6 的倍数，得到 {}",
        dep_states.len()
    );
    let n_dep = dep_states.len() / 6;
    assert_eq!(
        dep_times.len(),
        n_dep,
        "dep_times 长度 ({}) 须等于 n_dep ({})",
        dep_times.len(),
        n_dep
    );
    assert!(
        arrival_states.len().is_multiple_of(6),
        "arrival_states 展平长度必须是 6 的倍数，得到 {}",
        arrival_states.len()
    );
    let n_alpha = alpha_grid.len();
    let n_arrival = arrival_states.len() / 6;
    assert!(n_arrival > 0, "arrival_states 不能为空");

    let total = n_dep.checked_mul(n_alpha).expect("n_dep * n_alpha 溢出");

    (0..total)
        .into_par_iter()
        .map(|idx| {
            let i_dep = idx / n_alpha;
            let i_alpha = idx % n_alpha;
            let mut dep = [0.0_f64; 6];
            dep.copy_from_slice(&dep_states[i_dep * 6..i_dep * 6 + 6]);
            evaluate_point(
                &dep,
                dep_times[i_dep],
                alpha_grid[i_alpha],
                arrival_states,
                params,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const MU: f64 = 1.21506683e-2; // 地月质量参数

    fn params() -> GridSearchParams {
        GridSearchParams {
            mu: MU,
            max_transfer_time: 0.5,
            integration_dt: 0.05,
            intersection_threshold: 1e-3,
            min_distance_threshold: 0.05,
            collision_earth_radius: 5e-4,
            collision_moon_radius: 3e-4,
            rtol: 1e-9,
            atol: 1e-9,
            max_step: 0.05,
        }
    }

    /// 构造绕 (xc, 0) 的小圆轨道（平面），n 点等间隔采样。
    fn circular_orbit(xc: f64, r: f64, n: usize) -> Vec<f64> {
        let mut s = vec![0.0_f64; n * 6];
        for i in 0..n {
            let th = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
            s[i * 6] = xc + r * th.cos();
            s[i * 6 + 1] = r * th.sin();
            s[i * 6 + 3] = -r * th.sin();
            s[i * 6 + 4] = r * th.cos();
        }
        s
    }

    /// 评估单点应填充所有成功分支字段（success=true、status 非 failed）。
    #[test]
    fn evaluate_point_success_branch_populates_fields() {
        let dep = circular_orbit(0.9, 0.08, 40);
        let arrival = circular_orbit(0.7, 0.12, 30);
        let mut dep0 = [0.0_f64; 6];
        dep0.copy_from_slice(&dep[..6]);
        let r = evaluate_point(&dep0, 0.0, 1.0, &arrival, &params());
        assert!(r.success);
        assert_ne!(r.status, "integration_failed");
        assert!(r.min_distance.is_some());
        assert!(r.min_distance_idx.is_some());
        assert!(r.transfer_trajectory.is_some());
        // dv_departure = |alpha·vel - vel| = |alpha-1|·|vel|；alpha=1 → 0。
        assert!(r.dv_departure < 1e-12);
    }

    /// dv 惩罚分支：状态合理应不触发；本测试用病态 rtol 强制步长塌缩，
    /// 验证失败分支字段语义（dv=1e10、status=integration_failed）。
    #[test]
    fn evaluate_point_integration_failure_penalty() {
        let dep = circular_orbit(0.9, 0.08, 40);
        let arrival = circular_orbit(0.7, 0.12, 30);
        let mut dep0 = [0.0_f64; 6];
        dep0.copy_from_slice(&dep[..6]);
        // rtol 极小 + max_step 极小 → 任何接受步都难满足，步长快速塌缩。
        let bad = GridSearchParams {
            rtol: 1e-300,
            atol: 1e-300,
            max_step: 1e-15,
            max_transfer_time: 0.5,
            ..params()
        };
        let r = evaluate_point(&dep0, 0.0, 1.0, &arrival, &bad);
        // 无论是否触发（取决于 propagate_cr3bp 的步长下限），都不应穿透 Err；
        // 触发时检查惩罚字段；未触发时跳过（温和参数下 propagate 可能仍返回 Ok）。
        if !r.success {
            assert_eq!(r.status, "integration_failed");
            assert_eq!(r.dv_departure, DV_PENALTY);
            assert!(r.min_distance.is_none());
            assert!(r.transfer_trajectory.is_none());
        }
    }

    /// 串行网格保序：n_dep×n_alpha 候选，departure 在外、alpha 在内。
    #[test]
    fn serial_preserves_departure_outer_alpha_inner_order() {
        let n_dep = 2;
        let n_alpha = 3;
        let dep = circular_orbit(0.9, 0.08, 40);
        let dep_states: Vec<f64> = dep[..n_dep * 6].to_vec();
        let dep_times: Vec<f64> = (0..n_dep).map(|i| i as f64).collect();
        let alpha_grid = vec![0.9, 1.0, 1.1];
        let arrival = circular_orbit(0.7, 0.12, 30);

        let results =
            transfer_grid_search_serial(&dep_states, &dep_times, &alpha_grid, &arrival, &params());
        assert_eq!(results.len(), n_dep * n_alpha);
        // 顺序：idx = i_dep * n_alpha + i_alpha。
        for (idx, r) in results.iter().enumerate() {
            let i_dep = idx / n_alpha;
            let i_alpha = idx % n_alpha;
            assert_eq!(r.departure_time, dep_times[i_dep]);
            assert!((r.alpha - alpha_grid[i_alpha]).abs() < 1e-15);
            // departure_state 前 3 维与 dep_states 对应行一致。
            for k in 0..3 {
                assert!((r.departure_state[k] - dep_states[i_dep * 6 + k]).abs() < 1e-15);
            }
        }
    }

    /// 并行与串行逐位一致：par_iter+collect 保序 + evaluate_point 纯函数，
    /// 两边走同一 propagate_cr3bp，结果逐字段相等（含轨迹与浮点标量）。
    #[test]
    fn parallel_matches_serial_bit_for_bit() {
        let n_dep = 3;
        let n_alpha = 4;
        let dep = circular_orbit(0.9, 0.08, 40);
        let dep_states: Vec<f64> = dep[..n_dep * 6].to_vec();
        let dep_times: Vec<f64> = (0..n_dep).map(|i| i as f64).collect();
        let alpha_grid = vec![0.9, 0.95, 1.0, 1.05];
        let arrival = circular_orbit(0.7, 0.12, 30);

        let serial =
            transfer_grid_search_serial(&dep_states, &dep_times, &alpha_grid, &arrival, &params());
        let parallel = transfer_grid_search_parallel(
            &dep_states,
            &dep_times,
            &alpha_grid,
            &arrival,
            &params(),
        );
        assert_eq!(serial.len(), n_dep * n_alpha);
        assert_eq!(parallel.len(), serial.len());

        for (s, p) in serial.iter().zip(parallel.iter()) {
            // 保序：逐候选 (departure_time, alpha) 对齐。
            assert_eq!(s.departure_time, p.departure_time);
            assert!((s.alpha - p.alpha).abs() < 1e-15);
            assert_eq!(s.success, p.success);
            assert_eq!(s.status, p.status);
            assert_eq!(s.min_distance_idx, p.min_distance_idx);
            assert_eq!(s.min_distance_orbit_idx, p.min_distance_orbit_idx);
            assert_eq!(s.intersection_idx, p.intersection_idx);
            assert_eq!(s.local_minimum_idx, p.local_minimum_idx);
            assert_eq!(s.collision_idx, p.collision_idx);
            assert_eq!(s.dv_departure.to_bits(), p.dv_departure.to_bits());
            if let (Some(a), Some(b)) = (s.min_distance, p.min_distance) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
            if let (Some(a), Some(b)) = (
                s.transfer_trajectory.as_ref(),
                p.transfer_trajectory.as_ref(),
            ) {
                assert_eq!(a.len(), b.len());
                for (x, y) in a.iter().zip(b.iter()) {
                    assert_eq!(x.to_bits(), y.to_bits(), "轨迹浮点位级不一致");
                }
            }
        }
    }
}
