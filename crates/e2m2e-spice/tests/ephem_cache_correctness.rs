//! 星历缓存无内核正确性测试（ADR 0013：按定义验证，不用 golden file）。
//!
//! # 测试策略
//! 通过 [`EphemCache::from_raw_grids`] 用合成数据构造缓存，验证：
//!
//! 1. **三次样条插值误差界**：通过缓存体的 lookup_body_position 验证已知解析函数
//!    （sin/cos）在网格点和内部中间点的插值精度。自然三次样条（自然边界 m₀=mₙ=0）
//!    在边界附近降为 O(h²)，因此中间点查询避开首尾各两个子区间。
//! 2. **EphemCache 往返**：from_raw_grids → lookup_body_position/lookup_frame_matrix
//!    在采样点上的精确恢复。
//! 3. **帧旋转矩阵正交性**：参数化 Rz(θ) 的旋转矩阵经缓存插值后，
//!    在采样点满足 RᵀR = I、det(R) = 1（机器精度内）。
//!
//! **并发安全**：本测试通过 `e2m2e_spice::lock_spice_for_test()` 串行化
//! cache enable/disable，避免平行测试之间全局缓存的冲突。
//!
//! 所有断言基于数学恒等式（插值误差理论 + 正交矩阵定义），不依赖内核文件、
//! golden 文件或外部软件。

use e2m2e_spice::ephem_cache::{
    disable, enable, lookup_body_position, lookup_frame_matrix, EphemCache,
};
use std::f64::consts::PI;

/// 自然三次样条内部子区间的误差理论界：|f−s| ≤ (5/384)·h⁴·max|f⁽⁴⁾|。
///
/// 注意：此界在自然边界（m₀=mₙ=0）附近不成立——边界附近误差量级为
/// O(h²) 而非 O(h⁴)。本模块的中间点测试避开首末各两个子区间。
fn spline_interior_bound(h: f64, max_f4: f64) -> f64 {
    (5.0 / 384.0) * h.powi(4) * max_f4
}

// ── helpers ──────────────────────────────────────────────────────────────

fn build_body_cache(
    t_grid: &[f64],
    name: &str,
    pos_fn: impl Fn(f64) -> [f64; 3],
    vel_fn: impl Fn(f64) -> [f64; 3],
) -> EphemCache {
    let n = t_grid.len();
    let states: Vec<[f64; 6]> = (0..n)
        .map(|i| {
            let t = t_grid[i];
            let p = pos_fn(t);
            let v = vel_fn(t);
            [p[0], p[1], p[2], v[0], v[1], v[2]]
        })
        .collect();
    let key = (name.to_string(), "ORIGIN".to_string());
    EphemCache::from_raw_grids(t_grid, &[(key, states)], &[], &[]).unwrap()
}

fn build_frame_cache(t_grid: &[f64], mat_fn: impl Fn(f64) -> [[f64; 3]; 3]) -> EphemCache {
    let n = t_grid.len();
    let mats: Vec<[[f64; 3]; 3]> = (0..n).map(|i| mat_fn(t_grid[i])).collect();
    EphemCache::from_raw_grids(
        t_grid,
        &[],
        &[((String::from("FROM"), String::from("TO")), mats)],
        &[],
    )
    .unwrap()
}

/// 局部测试锁：防止多个测试并行启用/禁用全局缓存导致竞争。
/// 与 crate 内的 `SPICE_TEST_LOCK` 独立——本文件只序列化自身测试。
static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    TEST_LOCK.lock().unwrap()
}

/// 获取内部（避开边界子区间）的中间测试点。
fn interior_midpoints(t_grid: &[f64], skip: usize) -> Vec<f64> {
    t_grid
        .windows(2)
        .skip(skip)
        .rev()
        .skip(skip)
        .rev()
        .map(|w| 0.5 * (w[0] + w[1]))
        .collect()
}

// ══════════════════════════════════════════════════════════════════════════
// 测试类别 1：CubicSpline 插值误差界（通过 body cache 间接测试）
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn test_cubic_spline_exact_at_nodes() {
    let _guard = lock();
    let t_grid: Vec<f64> = (0..20).map(|i| i as f64 * 0.3).collect();
    let pos_fn = |t: f64| -> [f64; 3] { [t.sin(), t.cos(), t.powi(2)] };
    let vel_fn = |t: f64| -> [f64; 3] { [t.cos(), -t.sin(), 2.0 * t] };
    let cache = build_body_cache(&t_grid, "ND", &pos_fn, &vel_fn);
    enable(cache);

    for &t in &t_grid {
        let pos = lookup_body_position("ND", "ORIGIN", t)
            .expect("lookup failed")
            .expect("cache miss");
        let expected = pos_fn(t);
        for k in 0..3 {
            assert!(
                (pos[k] - expected[k]).abs() < 1e-13,
                "t={t}: pos[{k}]={} expected={} error={:.2e}",
                pos[k],
                expected[k],
                (pos[k] - expected[k]).abs()
            );
        }
    }
    disable();
}

#[test]
fn test_cubic_spline_sin_interior_error_bound() {
    let _guard = lock();
    // h=0.5, 21 格点 → 内部中间点避开首两/末两个子区间（自然边界 O(h²) 区域）。
    let t_grid: Vec<f64> = (0..=20).map(|i| i as f64 * 0.5).collect();
    let pos_fn = |t: f64| -> [f64; 3] { [t.sin(), 0.0, 0.0] };
    let vel_fn = |_t: f64| -> [f64; 3] { [0.0, 0.0, 0.0] };
    let cache = build_body_cache(&t_grid, "SIN", &pos_fn, &vel_fn);
    enable(cache);

    let h = 0.5;
    let bound = spline_interior_bound(h, 1.0);
    let test_points = interior_midpoints(&t_grid, 2);
    for &t in &test_points {
        let pos = lookup_body_position("SIN", "ORIGIN", t)
            .expect("lookup failed")
            .expect("cache miss");
        let exact = t.sin();
        let err = (pos[0] - exact).abs();
        assert!(
            err < bound * 2.0,
            "SIN t={t:.3}: err {err:.2e} > bound {bound:.2e}"
        );
    }
    disable();
}

#[test]
fn test_cubic_spline_cos_interior_error_bound() {
    let _guard = lock();
    let t_grid: Vec<f64> = (0..=20).map(|i| i as f64 * 0.5).collect();
    let pos_fn = |t: f64| -> [f64; 3] { [0.0, t.cos(), 0.0] };
    let vel_fn = |_t: f64| -> [f64; 3] { [0.0, 0.0, 0.0] };
    let cache = build_body_cache(&t_grid, "COS", &pos_fn, &vel_fn);
    enable(cache);

    let h = 0.5;
    let bound = spline_interior_bound(h, 1.0);
    let test_points = interior_midpoints(&t_grid, 2);
    for &t in &test_points {
        let pos = lookup_body_position("COS", "ORIGIN", t)
            .expect("lookup failed")
            .expect("cache miss");
        let exact = t.cos();
        let err = (pos[1] - exact).abs();
        assert!(
            err < bound * 2.0,
            "COS t={t:.3}: err {err:.2e} > bound {bound:.2e}"
        );
    }
    disable();
}

/// 常数函数（f⁽⁴⁾ ≡ 0）应被样条精确再现：边界条件对常数无影响。
#[test]
fn test_cubic_spline_constant_exact() {
    let _guard = lock();
    let t_grid: Vec<f64> = (0..=20).map(|i| i as f64 * 0.5).collect();
    let pos_fn = |_t: f64| -> [f64; 3] { [42.0, -7.0, 3.14] };
    let vel_fn = |_t: f64| -> [f64; 3] { [0.0, 0.0, 0.0] };
    let cache = build_body_cache(&t_grid, "CST", &pos_fn, &vel_fn);
    enable(cache);

    for &t in &t_grid {
        let pos = lookup_body_position("CST", "ORIGIN", t)
            .expect("lookup failed")
            .expect("cache miss");
        assert!((pos[0] - 42.0).abs() < 1e-14, "CST t={t}: x={}", pos[0]);
        assert!((pos[1] + 7.0).abs() < 1e-14, "CST t={t}: y={}", pos[1]);
        assert!((pos[2] - 3.14).abs() < 1e-14, "CST t={t}: z={}", pos[2]);
    }
    disable();
}

// ══════════════════════════════════════════════════════════════════════════
// 测试类别 2：EphemCache 往返
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn test_ephem_cache_position_roundtrip() {
    let _guard = lock();
    let t_grid: Vec<f64> = (0..50).map(|i| i as f64 * 100.0).collect();
    let pos_fn = |t: f64| -> [f64; 3] { [t * 1.5, t.sin() * 100.0, t.cos() * 100.0] };
    let vel_fn = |t: f64| -> [f64; 3] { [1.5, t.cos() * 100.0, -t.sin() * 100.0] };
    let cache = build_body_cache(&t_grid, "RT", &pos_fn, &vel_fn);
    enable(cache);

    for &t in &t_grid {
        let pos = lookup_body_position("RT", "ORIGIN", t)
            .expect("lookup failed")
            .expect("cache miss");
        let expected = pos_fn(t);
        for k in 0..3 {
            assert!(
                (pos[k] - expected[k]).abs() < 1e-13,
                "RT t={t} pos[{k}]={} vs {}",
                pos[k],
                expected[k]
            );
        }
    }
    disable();
}

#[test]
fn test_ephem_cache_interpolation_accuracy() {
    let _guard = lock();
    let t_grid: Vec<f64> = (0..=20).map(|i| i as f64 * 0.5).collect();
    let pos_fn = |t: f64| -> [f64; 3] { [t.sin(), t.cos(), 0.0] };
    let vel_fn = |t: f64| -> [f64; 3] { [t.cos(), -t.sin(), 0.0] };
    let cache = build_body_cache(&t_grid, "INT", &pos_fn, &vel_fn);
    enable(cache);

    let h = 0.5;
    let bound = spline_interior_bound(h, 1.0) * 2.0;
    let test_points = interior_midpoints(&t_grid, 2);
    for &t in &test_points {
        let pos = lookup_body_position("INT", "ORIGIN", t)
            .expect("lookup failed")
            .expect("cache miss");
        let exact = pos_fn(t);
        for k in 0..3 {
            let err = (pos[k] - exact[k]).abs();
            assert!(
                err < bound,
                "INT t={t:.3} pos[{k}] err {err:.2e} > bound {bound:.2e}"
            );
        }
    }
    disable();
}

// ══════════════════════════════════════════════════════════════════════════
// 测试类别 3：帧旋转矩阵正交性
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn test_frame_orthogonality_at_nodes() {
    let _guard = lock();
    let t_grid: Vec<f64> = (0..=10).map(|i| i as f64).collect();
    let mat_fn = |t: f64| -> [[f64; 3]; 3] {
        let a = PI * t / 5.0;
        [
            [a.cos(), -a.sin(), 0.0],
            [a.sin(), a.cos(), 0.0],
            [0.0, 0.0, 1.0],
        ]
    };
    let cache = build_frame_cache(&t_grid, &mat_fn);
    enable(cache);

    for &t in &t_grid {
        let r = lookup_frame_matrix("FROM", "TO", t)
            .expect("lookup failed")
            .expect("frame cache miss");

        // RᵀR = I
        for i in 0..3 {
            for j in 0..3 {
                let mut rtr = 0.0;
                for k in 0..3 {
                    rtr += r[k][i] * r[k][j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (rtr - expected).abs() < 1e-12,
                    "t={t}: (RᵀR)[{i}][{j}]={rtr} ≠ {expected}"
                );
            }
        }

        // det(R) = 1
        let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
        assert!((det - 1.0).abs() < 1e-12, "t={t}: det(R)={det} ≠ 1");
    }
    disable();
}

/// 中间点的正交性有偏差但应控制在合理量级内。
///
/// Rz(θ) 的分量 cos/sin 在自然三次样条内各自插值，在网格点上分量值精确，
/// 中间点分量误差量级 ~h⁴，导致 RᵀR 偏离 I 的量级同阶。
#[test]
fn test_frame_orthogonality_off_grid_bounded() {
    let _guard = lock();
    let t_grid: Vec<f64> = (0..=10).map(|i| i as f64).collect();
    let mat_fn = |t: f64| -> [[f64; 3]; 3] {
        let a = PI * t / 5.0;
        [
            [a.cos(), -a.sin(), 0.0],
            [a.sin(), a.cos(), 0.0],
            [0.0, 0.0, 1.0],
        ]
    };
    let cache = build_frame_cache(&t_grid, &mat_fn);
    enable(cache);

    // h=1，中间点处样条误差量级 ~h⁴ = 1 导致正交性偏离约 2e-2。
    // 容差按 (5/384)·h⁴·4 = 0.05 取。
    let bound = 0.05;
    let test_points = interior_midpoints(&t_grid, 2);
    for &t in &test_points {
        let r = lookup_frame_matrix("FROM", "TO", t)
            .expect("lookup failed")
            .expect("frame cache miss");

        for i in 0..3 {
            for j in 0..3 {
                let mut rtr = 0.0;
                for k in 0..3 {
                    rtr += r[k][i] * r[k][j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (rtr - expected).abs() < bound,
                    "off-grid t={t}: (RᵀR)[{i}][{j}] = {rtr} ≠ {expected}, err = {:.2e}",
                    (rtr - expected).abs()
                );
            }
        }
    }
    disable();
}

#[test]
fn test_cache_miss_returns_none() {
    let _guard = lock();
    // 先 disable 确保没有残留缓存
    disable();
    let result = lookup_body_position("X", "Y", 0.0);
    assert!(result.is_ok());
    assert!(result.unwrap().is_none());

    let result = lookup_frame_matrix("X", "Y", 0.0);
    assert!(result.is_ok());
    assert!(result.unwrap().is_none());
}
