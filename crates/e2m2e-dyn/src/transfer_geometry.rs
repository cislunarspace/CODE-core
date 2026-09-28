//! 转移搜索几何核（Rust 实现，纯数学，不依赖 SPICE）。
//!
//! 从 Python `e2m2e/algorithm/transfer/search_geometry.py` 迁移，向量化
//! （原生循环，零 Python 解释器开销）。5 个公开函数 + 1 个 chunked 变体，
//! 逐行对照 numpy 实现，保证数值与索引约定一致。
//!
//! - [`compute_distance_series`]：n_traj×n_orbit 广播距离 + 行内 argmin
//! - [`compute_distance_series_chunked`]：n_traj×n_orbit > [`MAX_DISTANCE_PAIRS`]
//!   时分块（对应 Python 的内存控制意图）
//! - [`compute_min_distance`]：全局最近点
//! - [`detect_intersection`]：min_dist < threshold 返回全局最近点
//! - [`detect_local_minimum`]：相邻三点严格小于两侧找局部极小
//! - [`check_collision`]：到 earth/moon 中心距离过阈
//!
//! # argmin 约定
//!
//! 重复最小值取**首个**（与 numpy `argmin` 一致），统一用严格 `<` 比较：
//! 遇相等候选不更新，保留更早出现的索引。这是测试能断言"整数索引精确
//! 相等"的前提（见 transfer-grid-search-rust.md 第 2.3 节）。
//!
//! # 输入约定
//!
//! `*_states` 为 n×6 行优先展平（x,y,z,vx,vy,vz），距离计算只用前 3 维位置，
//! 与 numpy 版 `states[:, :3]` 对应。`detect_intersection` 返回点时取完整 6 维。

/// n_traj×n_orbit 超过此阈值时分块计算（对应 `search_geometry.py:12`）。
pub const MAX_DISTANCE_PAIRS: usize = 10_000_000;

/// 校验展平数组是 6 的倍数，返回点数。
fn checked_n_points(states: &[f64]) -> usize {
    assert!(
        states.len().is_multiple_of(6),
        "states 展平长度必须是 6 的倍数，得到 {}",
        states.len()
    );
    states.len() / 6
}

/// 对 `traj[start..end]` 子范围的每个点，在全部 orbit 点中求最近距离与索引。
///
/// 写入 `d_out[start..end]` 与 `idx_out[start..end]`。argmin 取首个（严格 `<`）。
/// 与 numpy `np.argmin(distances, axis=1)` 逐位一致：距离平方和按
/// `(dx²+dy²)+dz²` 顺序累加后开方，与 `np.sum(diff**2, axis=2)` 同序。
fn nearest_range(
    traj_states: &[f64],
    orbit_states: &[f64],
    n_orbit: usize,
    start: usize,
    end: usize,
    d_out: &mut [f64],
    idx_out: &mut [i64],
) {
    for i in start..end {
        let tx = traj_states[i * 6];
        let ty = traj_states[i * 6 + 1];
        let tz = traj_states[i * 6 + 2];
        let mut best_d = f64::INFINITY;
        let mut best_j: i64 = 0;
        for j in 0..n_orbit {
            let dx = tx - orbit_states[j * 6];
            let dy = ty - orbit_states[j * 6 + 1];
            let dz = tz - orbit_states[j * 6 + 2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            // 严格 <：相等不更新，保留首个最小（numpy argmin 约定）。
            if dist < best_d {
                best_d = dist;
                best_j = j as i64;
            }
        }
        d_out[i] = best_d;
        idx_out[i] = best_j;
    }
}

/// 计算轨迹每步到目标轨道采样点集合的最近距离与对应轨道索引。
///
/// 移植自 `compute_distance_series`（`search_geometry.py:15-28`）。当
/// n_traj×n_orbit 超过 [`MAX_DISTANCE_PAIRS`] 时走 [`compute_distance_series_chunked`]
/// （与 numpy 同阈值同分块）。返回 `(d_per_step[n_traj], orbit_idx_per_step[n_traj])`。
pub fn compute_distance_series(traj_states: &[f64], orbit_states: &[f64]) -> (Vec<f64>, Vec<i64>) {
    let n_traj = checked_n_points(traj_states);
    let n_orbit = checked_n_points(orbit_states);
    if n_traj * n_orbit > MAX_DISTANCE_PAIRS {
        return compute_distance_series_chunked(traj_states, orbit_states);
    }
    let mut d = vec![0.0_f64; n_traj];
    let mut idx = vec![0_i64; n_traj];
    nearest_range(
        traj_states,
        orbit_states,
        n_orbit,
        0,
        n_traj,
        &mut d,
        &mut idx,
    );
    (d, idx)
}

/// 分块版 [`compute_distance_series`]，输出与不分块等价。
///
/// 移植自 `compute_distance_series_chunked`（`search_geometry.py:31-48`）。
/// `chunk_size = max(1, MAX_DISTANCE_PAIRS // n_orbit)`，按 chunk 遍历 traj
/// 子范围。Python 版意在避免一次性物化 n_traj×n_orbit 距离矩阵；Rust 版逐点
/// 求解本就 O(1) 额外内存，分块在此仅保留语义对齐（数值与不分块逐位一致）。
pub fn compute_distance_series_chunked(
    traj_states: &[f64],
    orbit_states: &[f64],
) -> (Vec<f64>, Vec<i64>) {
    let n_traj = checked_n_points(traj_states);
    let n_orbit = checked_n_points(orbit_states);
    let chunk_size = MAX_DISTANCE_PAIRS / n_orbit.max(1);
    let chunk_size = chunk_size.max(1);
    let mut d = vec![0.0_f64; n_traj];
    let mut idx = vec![0_i64; n_traj];
    let mut start = 0usize;
    while start < n_traj {
        let end = (start + chunk_size).min(n_traj);
        nearest_range(
            traj_states,
            orbit_states,
            n_orbit,
            start,
            end,
            &mut d,
            &mut idx,
        );
        start = end;
    }
    (d, idx)
}

/// 全局最近点：在 [`compute_distance_series`] 的 `d_per_step` 上取 argmin。
///
/// 移植自 `compute_min_distance`（`search_geometry.py:51-56`）。返回
/// `(min_dist, step_idx, orbit_idx)`，其中 `step_idx` 为 `d_per_step` 的首个
/// 最小值索引，`orbit_idx = orbit_idx_per_step[step_idx]`。
pub fn compute_min_distance(traj_states: &[f64], orbit_states: &[f64]) -> (f64, i64, i64) {
    let (d_per_step, orbit_idx_per_step) = compute_distance_series(traj_states, orbit_states);
    let mut best_d = f64::INFINITY;
    let mut best_i: i64 = 0;
    for (i, &v) in d_per_step.iter().enumerate() {
        if v < best_d {
            best_d = v;
            best_i = i as i64;
        }
    }
    (best_d, best_i, orbit_idx_per_step[best_i as usize])
}

/// 相交检测：全局最近点距离 < threshold 时返回该点完整 6 维状态。
///
/// 移植自 `detect_intersection`（`search_geometry.py:59-65`）。返回
/// `(found, point, step_idx)`：命中时 `point = traj_states[step_idx]`（6 维），
/// 未命中 `(false, None, -1)`。比较为严格 `<`，与 numpy 一致。
pub fn detect_intersection(
    traj_states: &[f64],
    orbit_states: &[f64],
    threshold: f64,
) -> (bool, Option<[f64; 6]>, i64) {
    let (min_dist, step_idx, _) = compute_min_distance(traj_states, orbit_states);
    if min_dist < threshold {
        let mut point = [0.0_f64; 6];
        point.copy_from_slice(&traj_states[(step_idx as usize) * 6..(step_idx as usize) * 6 + 6]);
        (true, Some(point), step_idx)
    } else {
        (false, None, -1)
    }
}

/// 局部极小检测：在每步最近距离序列上找严格局部极小。
///
/// 移植自 `detect_local_minimum`（`search_geometry.py:68-83`）。先取
/// [`compute_distance_series`] 的 `d_per_step`（等价 numpy 的
/// `np.min(distances, axis=1)`），再扫 `i ∈ [1, n-1)`，满足
/// `d[i-1] > d[i] < d[i+1]`（两侧严格大于）即为局部极小。所有局部极小中取
/// 值最小者（严格 `<` 取首个并列），返回 `(found, dist, idx)`；无极小返回
/// `(false, +inf, -1)`。
///
/// `d_per_step` 长度 < 3 时无候选，直接返回未命中。
pub fn detect_local_minimum(traj_states: &[f64], orbit_states: &[f64]) -> (bool, f64, i64) {
    let (d_per_step, _) = compute_distance_series(traj_states, orbit_states);
    let n = d_per_step.len();
    let mut found = false;
    let mut best_dist = f64::INFINITY;
    let mut best_idx: i64 = -1;
    if n >= 3 {
        for i in 1..n - 1 {
            let center = d_per_step[i];
            if d_per_step[i - 1] > center && d_per_step[i + 1] > center && center < best_dist {
                best_dist = center;
                best_idx = i as i64;
                found = true;
            }
        }
    }
    if found {
        (true, best_dist, best_idx)
    } else {
        (false, f64::INFINITY, -1)
    }
}

/// 碰撞检测：到 earth/moon 中心距离过阈。
///
/// 移植自 `check_collision`（`search_geometry.py:86-103`）。earth 中心
/// `[-mu, 0, 0]`、moon 中心 `[1-mu, 0, 0]`；先完整扫 earth（首个命中即返回），
/// 无 earth 命中再扫 moon。比较为严格 `<`。返回 `(collision, body, idx)`：
/// 命中时 `body` 为 `"earth"`/`"moon"`、`idx` 为首个命中步；未命中
/// `(false, None, -1)`。
pub fn check_collision(
    traj_states: &[f64],
    mu: f64,
    collision_earth_radius: f64,
    collision_moon_radius: f64,
) -> (bool, Option<String>, i64) {
    let n = checked_n_points(traj_states);
    let earth_x = -mu;
    let moon_x = 1.0 - mu;
    for i in 0..n {
        let dx = traj_states[i * 6] - earth_x;
        let dy = traj_states[i * 6 + 1];
        let dz = traj_states[i * 6 + 2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist < collision_earth_radius {
            return (true, Some("earth".to_string()), i as i64);
        }
    }
    for i in 0..n {
        let dx = traj_states[i * 6] - moon_x;
        let dy = traj_states[i * 6 + 1];
        let dz = traj_states[i * 6 + 2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist < collision_moon_radius {
            return (true, Some("moon".to_string()), i as i64);
        }
    }
    (false, None, -1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造 n×6 展平状态：前 3 维由 (x,y,z) lambda 给出，后 3 维速度填 0。
    fn states_from_pos<F: Fn(usize) -> [f64; 3]>(n: usize, f: F) -> Vec<f64> {
        let mut s = vec![0.0_f64; n * 6];
        for i in 0..n {
            let [x, y, z] = f(i);
            s[i * 6] = x;
            s[i * 6 + 1] = y;
            s[i * 6 + 2] = z;
        }
        s
    }

    #[test]
    fn argmin_takes_first_on_ties() {
        // 两个相同 orbit 点（原点），单步轨迹距两者相等 → argmin 取首个（0）。
        let orbit = states_from_pos(2, |_| [0.0, 0.0, 0.0]);
        let traj = states_from_pos(1, |_| [1.0, 0.0, 0.0]);
        let (d, idx) = compute_distance_series(&traj, &orbit);
        assert_eq!(idx, vec![0]);
        assert!((d[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn min_distance_argmin_first() {
        // d_per_step = [0.3, 0.05, 0.1]，唯一最小在 idx=1。
        let orbit = states_from_pos(1, |_| [0.0, 0.0, 0.0]);
        let traj = states_from_pos(3, |i| [0.3 - 0.25 * (i as f64), 0.0, 0.0]);
        // i=0→0.3, i=1→0.05, i=2→0.1 —— 注意符号：i=1 时 0.3-0.25=0.05
        let (md, si, oi) = compute_min_distance(&traj, &orbit);
        assert_eq!(si, 1);
        assert!((md - 0.05).abs() < 1e-12);
        assert_eq!(oi, 0);
    }

    #[test]
    fn local_minimum_finds_strict_valley() {
        // 轨迹沿 x 先靠近再远离原点，d_per_step = [2, 0.5, 1]，谷在 idx=1。
        let orbit = states_from_pos(1, |_| [0.0, 0.0, 0.0]);
        let traj = states_from_pos(3, |i| match i {
            0 => [2.0, 0.0, 0.0],
            1 => [0.5, 0.0, 0.0],
            _ => [1.0, 0.0, 0.0],
        });
        let (found, dist, idx) = detect_local_minimum(&traj, &orbit);
        assert!(found);
        assert_eq!(idx, 1);
        assert!((dist - 0.5).abs() < 1e-12);
    }

    #[test]
    fn local_minimum_none_when_monotonic() {
        // d_per_step 严格递减，无局部极小。
        let orbit = states_from_pos(1, |_| [0.0, 0.0, 0.0]);
        let traj = states_from_pos(4, |i| [3.0 - i as f64, 0.0, 0.0]); // 3,2,1,0
        let (found, dist, idx) = detect_local_minimum(&traj, &orbit);
        assert!(!found);
        assert_eq!(idx, -1);
        assert!(dist.is_infinite());
    }

    #[test]
    fn detect_intersection_returns_full_state() {
        // 轨迹 idx=2 最近（距 0.5 < threshold 1.0），返回该点完整 6 维。
        let orbit = states_from_pos(1, |_| [0.0, 0.0, 0.0]);
        let mut traj = states_from_pos(3, |i| [2.0 - i as f64, 0.0, 0.0]); // 2,1,0 → 最近 idx=2 dist=0
                                                                           // 把 idx=2 的速度置为标志值，验证返回的是完整 6 维。
        traj[2 * 6 + 3] = 7.0;
        traj[2 * 6 + 4] = 8.0;
        traj[2 * 6 + 5] = 9.0;
        let (found, pt, idx) = detect_intersection(&traj, &orbit, 1.0);
        assert!(found);
        assert_eq!(idx, 2);
        let pt = pt.unwrap();
        assert!((pt[0]).abs() < 1e-12);
        assert!((pt[3] - 7.0).abs() < 1e-12);
        assert!((pt[4] - 8.0).abs() < 1e-12);
        assert!((pt[5] - 9.0).abs() < 1e-12);
    }

    #[test]
    fn check_collision_earth_priority_and_first_idx() {
        // mu=0.1：earth 在 x=-0.1，moon 在 x=0.9。轨迹 idx=1 距 earth 1e-5（命中 earth）。
        let mu = 0.1;
        let earth_r = 1e-4;
        let moon_r = 1e-4;
        let traj = states_from_pos(3, |i| match i {
            1 => [-mu + 1e-5, 0.0, 0.0],
            _ => [10.0, 0.0, 0.0],
        });
        let (hit, body, idx) = check_collision(&traj, mu, earth_r, moon_r);
        assert!(hit);
        assert_eq!(body.as_deref(), Some("earth"));
        assert_eq!(idx, 1);
    }

    #[test]
    fn check_collision_earth_priority_over_moon() {
        // 对抗用例：moon 在 idx=0 命中、earth 在 idx=1 命中——索引更早的是 moon，
        // 但 earth 优先级高于 moon（实现先完整扫 earth 命中即返回），故返回 earth@idx=1。
        // 钉死"earth 优先于 moon（索引无关）"这一非显然不变量。
        let mu = 0.1;
        let earth_r = 1e-4;
        let moon_r = 1e-4;
        let traj = states_from_pos(3, |i| match i {
            0 => [1.0 - mu + 1e-5, 0.0, 0.0], // moon 附近（距 moon 1e-5 < 1e-4）
            1 => [-mu + 1e-5, 0.0, 0.0],      // earth 附近（距 earth 1e-5 < 1e-4）
            _ => [10.0, 0.0, 0.0],            // 远离两天体
        });
        let (hit, body, idx) = check_collision(&traj, mu, earth_r, moon_r);
        assert!(hit);
        assert_eq!(body.as_deref(), Some("earth"));
        assert_eq!(idx, 1);
    }

    #[test]
    fn check_collision_moon_when_no_earth() {
        let mu = 0.1;
        let traj = states_from_pos(2, |i| match i {
            0 => [1.0 - mu + 1e-5, 0.0, 0.0], // moon 附近
            _ => [10.0, 0.0, 0.0],
        });
        let (hit, body, idx) = check_collision(&traj, mu, 1e-4, 1e-4);
        assert!(hit);
        assert_eq!(body.as_deref(), Some("moon"));
        assert_eq!(idx, 0);
    }

    #[test]
    fn check_collision_none_when_far() {
        let traj = states_from_pos(2, |i| [10.0 + i as f64, 0.0, 0.0]);
        let (hit, body, idx) = check_collision(&traj, 0.1, 1e-4, 1e-4);
        assert!(!hit);
        assert!(body.is_none());
        assert_eq!(idx, -1);
    }
}
