//! 转移搜索几何核（最近距离/相交/局部极小/碰撞）的 PyO3 绑定。

use pyo3::prelude::*;

/// 校验展平 states 数组长度是 6 的倍数，否则返回 ValueError。
///
/// 5 个 transfer_geometry pyfunction 共用：输入是 n×6 行优先展平。
fn require_states6(name: &str, states: &[f64]) -> PyResult<()> {
    if !states.len().is_multiple_of(6) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} 展平长度必须是 6 的倍数，得到 {}",
            states.len()
        )));
    }
    Ok(())
}

/// 转移搜索几何核：轨迹每步到目标轨道采样点集合的最近距离与索引。
///
/// 移植自 `search_geometry.compute_distance_series` （纯数学，非 SPICE 门控）。
/// `traj_states`/`orbit_states` 为 n×6 行优先展平，只用前 3 维位置；n_traj×n_orbit
/// 超过 1e7 时内部分块（与 numpy 同阈值）。argmin 取首个（numpy 约定）。
///
/// # 返回
/// `(d_per_step, orbit_idx_per_step)` ：两个长度 n_traj 的 list。
#[pyfunction]
pub fn compute_distance_series_py(
    traj_states: Vec<f64>,
    orbit_states: Vec<f64>,
) -> PyResult<(Vec<f64>, Vec<i64>)> {
    require_states6("traj_states", &traj_states)?;
    require_states6("orbit_states", &orbit_states)?;
    Ok(e2m2e_dyn::transfer_geometry::compute_distance_series(
        &traj_states,
        &orbit_states,
    ))
}

/// 转移搜索几何核：全局最近点（min_dist, step_idx, orbit_idx）。
///
/// 移植自 `search_geometry.compute_min_distance` 。`step_idx` 为 d_per_step 的
/// 首个最小值索引，`orbit_idx = orbit_idx_per_step[step_idx]` 。
#[pyfunction]
pub fn compute_min_distance_py(
    traj_states: Vec<f64>,
    orbit_states: Vec<f64>,
) -> PyResult<(f64, i64, i64)> {
    require_states6("traj_states", &traj_states)?;
    require_states6("orbit_states", &orbit_states)?;
    Ok(e2m2e_dyn::transfer_geometry::compute_min_distance(
        &traj_states,
        &orbit_states,
    ))
}

/// 转移搜索几何核：相交检测。
///
/// 移植自 `search_geometry.detect_intersection` 。全局最近点距离 < `threshold`
/// 时返回该点完整 6 维状态。返回 `(found, point|None, step_idx)` ，比较为严格 `<` 。
#[pyfunction]
pub fn detect_intersection_py(
    traj_states: Vec<f64>,
    orbit_states: Vec<f64>,
    threshold: f64,
) -> PyResult<(bool, Option<Vec<f64>>, i64)> {
    require_states6("traj_states", &traj_states)?;
    require_states6("orbit_states", &orbit_states)?;
    let (found, point, idx) =
        e2m2e_dyn::transfer_geometry::detect_intersection(&traj_states, &orbit_states, threshold);
    Ok((found, point.map(|p| p.to_vec()), idx))
}

/// 转移搜索几何核：局部极小检测。
///
/// 移植自 `search_geometry.detect_local_minimum` 。在每步最近距离序列上找严格
/// 局部极小（两侧严格大于），取所有极小中值最小者（首个并列）。返回
/// `(found, dist, idx)` ；无极小 `(false, inf, -1)` 。
#[pyfunction]
pub fn detect_local_minimum_py(
    traj_states: Vec<f64>,
    orbit_states: Vec<f64>,
) -> PyResult<(bool, f64, i64)> {
    require_states6("traj_states", &traj_states)?;
    require_states6("orbit_states", &orbit_states)?;
    Ok(e2m2e_dyn::transfer_geometry::detect_local_minimum(
        &traj_states,
        &orbit_states,
    ))
}

/// 转移搜索几何核：碰撞检测。
///
/// 移植自 `search_geometry.check_collision` 。earth 中心 `[-mu,0,0]` 、moon 中心
/// `[1-mu,0,0]` ；earth 优先（首个命中即返回），无 earth 再扫 moon，比较为严格 `<` 。
/// 返回 `(collision, body|None, idx)` ，body 为 `"earth"`/`"moon"` 。
#[pyfunction]
pub fn check_collision_py(
    traj_states: Vec<f64>,
    mu: f64,
    collision_earth_radius: f64,
    collision_moon_radius: f64,
) -> PyResult<(bool, Option<String>, i64)> {
    require_states6("traj_states", &traj_states)?;
    Ok(e2m2e_dyn::transfer_geometry::check_collision(
        &traj_states,
        mu,
        collision_earth_radius,
        collision_moon_radius,
    ))
}
