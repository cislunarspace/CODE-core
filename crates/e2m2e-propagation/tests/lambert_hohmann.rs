//! Izzo Lambert 求解器 Hohmann 转移检验（ADR 0013：按定义验证）。
//!
//! # Hohmann 转移的解析解
//! Hohmann 转移是将两圆轨道之间最快的方法，具有封闭 Δv：
//! - 内圆轨道半径 R₁，圆轨道速度 v₁ = √(GM / R₁)
//! - 外圆轨道半径 R₂，圆轨道速度 v₂ = √(GM / R₂)
//! - 转移椭圆半长轴 aₜ = (R₁ + R₂) / 2
//! - 入库速度 v_p = √(2 GM / R₁ − GM / aₜ)（近地点）
//! - 出库速度 v_a = √(2 GM / R₂ − GM / aₜ)（远地点）
//! - Δv₁ = v_p − v₁, Δv₂ = v₂ − v_a
//!
//! 在轨道面内（二维问题），Lambert 的 v₀ v_f 应匹配 Hohmann 解析解；
//! 远地点目标的转移时间 tof = π √(aₜ³ / GM)（半周期）。
//!
//! # 其他解析边界
//! - Parabolic escape：轨道能量低至面摆脱
//! - 同一点往返（r₀ ≠ r_f）：Lambert 解得出的 v₀ 经二体传播应回到目标位置
//! - 多圈选项：长 TOF 下多圈右分支低能解应有限且逼近目标
//!
//! 所有断言基于二体理论公式（物理定义），不依赖黄金样本或外部软件。

use e2m2e_propagation::lambert::{lambert_izzo, TransferDirection};
use std::f64::consts::PI;

const MU: f64 = 398600.4418; // Earth GM (km³/s²)

// ── 二体解析传播（椭圆） ──────────────────────────────────────────────

/// 用 Kepler f/g 函数传播轨道：由 (r0, v0) 至时间 tof 的位置。
/// 只适用于椭圆轨道。
fn propagate_ellipse(r0: &[f64; 3], v0: &[f64; 3], tof: f64, mu: f64) -> [f64; 3] {
    let r0n = (r0[0].powi(2) + r0[1].powi(2) + r0[2].powi(2)).sqrt();
    let v2 = v0[0].powi(2) + v0[1].powi(2) + v0[2].powi(2);
    let a = 1.0 / (2.0 / r0n - v2 / mu);
    assert!(a > 0.0, "测试用例应为椭圆");
    let rdotv = r0[0] * v0[0] + r0[1] * v0[1] + r0[2] * v0[2];

    // 偏心率向量
    let ev = [
        (v2 - mu / r0n) * r0[0] / mu - rdotv * v0[0] / mu,
        (v2 - mu / r0n) * r0[1] / mu - rdotv * v0[1] / mu,
        (v2 - mu / r0n) * r0[2] / mu - rdotv * v0[2] / mu,
    ];
    let e = (ev[0].powi(2) + ev[1].powi(2) + ev[2].powi(2)).sqrt();
    let cos_e0 = (1.0 - r0n / a) / e;
    let sin_e0 = rdotv / (e * (mu * a).sqrt());
    let e0 = sin_e0.atan2(cos_e0);
    let n = (mu / (a * a * a)).sqrt();
    let m0 = e0 - e * e0.sin();
    let m = m0 + n * tof;

    // Newton 解 Kepler 方程
    let mut ecc = m;
    for _ in 0..50 {
        let f = ecc - e * ecc.sin() - m;
        let fp = 1.0 - e * ecc.cos();
        let step = f / fp;
        ecc -= step;
        if step.abs() < 1e-14 {
            break;
        }
    }
    let d_e = ecc - e0;
    let f = 1.0 + a * (d_e.cos() - 1.0) / r0n;
    let gg = tof - (d_e - d_e.sin()) / n;
    [
        f * r0[0] + gg * v0[0],
        f * r0[1] + gg * v0[1],
        f * r0[2] + gg * v0[2],
    ]
}

fn assert_close(a: &[f64; 3], b: &[f64; 3], tol: f64) {
    for i in 0..3 {
        assert!((a[i] - b[i]).abs() < tol, "分量 {i}: {} vs {}", a[i], b[i]);
    }
}

/// Lambert v₀ 经二体传播至 tof 回到 r_f。
fn assert_round_trip(r0: &[f64; 3], rf: &[f64; 3], tof: f64, v0: &[f64; 3]) {
    let r_end = propagate_ellipse(r0, v0, tof, MU);
    let rn = (rf[0].powi(2) + rf[1].powi(2) + rf[2].powi(2)).sqrt();
    for i in 0..3 {
        assert!(
            (r_end[i] - rf[i]).abs() < 1e-7 * rn,
            "Lambert 传播落点分量 {i}: {} vs {}",
            r_end[i],
            rf[i]
        );
    }
}

// ── 测试 1：Hohmann 解析 Δv 与 Lambert 一致 ────────────────────────────

/// 从 LEO (7000 km) 到 GEO (42164 km) 的 Hohmann 转移。
///
/// Hohmann 转移的 v0 有封闭解，Lambert 应返回相同结果（容许迭代误差）。
/// 这是已知物理解，不依赖任何外部软件。
#[test]
fn test_hohmann_transfer_matches_analytic() {
    let r1 = 7000.0;
    let r2 = 42164.0;
    let v1 = (MU / r1).sqrt();
    let v2 = (MU / r2).sqrt();
    let a_t = (r1 + r2) / 2.0;
    let vp = (2.0 * MU / r1 - MU / a_t).sqrt();
    let va = (2.0 * MU / r2 - MU / a_t).sqrt();
    let _dv1 = vp - v1;
    let _dv2 = v2 - va;
    let tof = PI * (a_t.powi(3) / MU).sqrt();

    // 二维轨道在 xy 面：内圆处 r0 在 +x，外圆处 rf 在 −x
    let r0 = [r1, 0.0, 0.0];
    let rf = [-r2, 0.0, 0.0];

    let (v0, vf, n_iter) = lambert_izzo(&r0, &rf, tof, MU, TransferDirection::ShortWay, 0).unwrap();

    // Hohmann v0 = [0, vp, 0]（纯切线方向）
    let v0_hohmann = [0.0, vp, 0.0];
    assert_close(&v0, &v0_hohmann, 1e-5);

    // Hohmann vf = [0, −va, 0]（纯切线方向，但 r 已反向）
    let vf_hohmann = [0.0, -va, 0.0];
    assert_close(&vf, &vf_hohmann, 1e-5);

    // 传播检验：经过 tof 后回到 r_f
    assert_round_trip(&r0, &rf, tof, &v0);

    assert!(n_iter <= 15, "Hohmann 迭代次数异常: {n_iter}");
}

// ── 测试 2：同一点往返（椭圆型快转移） ────────────────────────────────

/// Lambert 解出 v₀ 经二体传播后应回到 r_f。
/// 这是 Lambert 求解器"按定义"必须满足的性质。
#[test]
fn test_round_trip_for_various_geometries() {
    let cases = [
        ([7000.0, 0.0, 0.0], [7000.0, 7000.0, 0.0], 1800.0),
        ([7000.0, 0.0, 0.0], [-14600.0, 2500.0, 7000.0], 3600.0),
        ([15945.34, 0.0, 0.0], [12214.84, 10249.47, 0.0], 76.0 * 60.0),
    ];
    for &(r0, rf, tof) in &cases {
        let (v0, _, _) = lambert_izzo(&r0, &rf, tof, MU, TransferDirection::ShortWay, 0).unwrap();
        assert_round_trip(&r0, &rf, tof, &v0);
    }
}

// ── 测试 3：长程解方向与短程相反 ──────────────────────────────────────

/// 长程解与短程解有相反的角动量（r × v）方向。
/// 长程路线绕另一侧飞行，轨道面法向应反向。
#[test]
fn test_long_way_opposite_angular_momentum() {
    let r0 = [7000.0, 0.0, 0.0];
    let rf = [7000.0, 7000.0, 0.0];
    let tof = 3600.0 * 4.0;

    let (v_short, _, _) = lambert_izzo(&r0, &rf, tof, MU, TransferDirection::ShortWay, 0).unwrap();
    let (v_long, _, _) = lambert_izzo(&r0, &rf, tof, MU, TransferDirection::LongWay, 0).unwrap();

    let cross = |a: &[f64; 3], b: &[f64; 3]| -> f64 {
        a[0] * b[1] - a[1] * b[0] // z 分量（2D 平面）
    };
    let h_short = cross(&r0, &v_short);
    let h_long = cross(&r0, &v_long);
    assert!(h_short * h_long < 0.0, "长程/短程角动量应有相反符号");

    // 两项解均能满足往返传播
    assert_round_trip(&r0, &rf, tof, &v_short);
    assert_round_trip(&r0, &rf, tof, &v_long);
}

// ── 测试 4：多圈右分支低能解有限且正确 ─────────────────────────────────

/// 多圈（revs ≥ 1）Lambert 解返回右分支（低能耗），不产生 NaN/Inf。
#[test]
fn test_multi_rev_is_finite() {
    let r0 = [7000.0, 0.0, 0.0];
    let rf = [7000.0, 7000.0, 0.0];
    let tof = 3600.0 * 30.0;
    for revs in 1..=2 {
        let (v0, vf, n_iter) =
            lambert_izzo(&r0, &rf, tof, MU, TransferDirection::ShortWay, revs).unwrap();
        assert!(v0.iter().chain(vf.iter()).all(|x| x.is_finite()));
        assert!(n_iter > 0);
        assert_round_trip(&r0, &rf, tof, &v0);
    }
}

// ── 测试 5：bad-tof 应返回 Err ────────────────────────────────────────

/// tof 低于该圈数最小转移时间时，求解器应返回 Err 而非给出伪解。
#[test]
fn test_too_short_tof_errors() {
    let r0 = [7000.0, 0.0, 0.0];
    let rf = [7000.0, 7000.0, 0.0];
    let result = lambert_izzo(&r0, &rf, 3600.0, MU, TransferDirection::ShortWay, 1);
    assert!(result.is_err(), "多圈 tof 不足应返回 Err");
}
