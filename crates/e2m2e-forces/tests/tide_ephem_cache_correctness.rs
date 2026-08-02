#![cfg(feature = "spice")]

//! 潮汐缓存物理正确性测试（ADR 0013：按定义验证，不用 golden file）。
//!
//! PR #271 把 `gravity_field.rs` 的潮汐扰动体位置查询从裸调 spkezr
//! 改为走 EphemCache（`lookup_body_position` + `lookup_frame_matrix`）。
//! 本文件按物理定义补三项正确性测试：
//!
//! 1. `test_tide_delta_zero_when_k_love_zero`：k_love=0 时 ΔC/ΔS 全零
//!    （解析边界条件：Love 数为零等价于无潮汐）
//! 2. `test_perturber_position_cache_hit_consistency`：EphemCache 样条插值
//!    与 cspice spkezr 直接查的位置误差 < 1e-6 km（亚米级）
//! 3. `test_frame_matrix_cache_hit_consistency`：EphemCache 样条插值的帧
//!    旋转矩阵与 pxform 直接查结果逐元素误差 < 1e-10

use e2m2e_forces::forces::gravity_field::{effective_coefficients, TideConfig, TideMode};
use e2m2e_spice::ephem_cache;
use e2m2e_spice::spice_ffi;

// ── 加载 SPICE 内核 ─────────────────────────────────────────────────────

fn load_kernels() {
    let kernel_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .join("kernels");
    for name in [
        "naif0012.tls",
        "pck00010.tpc",
        "de430.bsp",
        "earth_latest_high_prec.bpc",
        "SPICEEarthPredictedKernel.bpc",
        "SPICELunaFrameKernel.tf",
        "SPICELunaCurrentKernel.bpc",
    ] {
        let path = kernel_dir.join(name);
        if path.exists() {
            let _ = cspice::data::furnish(path.to_string_lossy().to_string());
        }
    }
    e2m2e_spice::spice_ffi::register_bodies();
}

// ── 内核辅助函数 ────────────────────────────────────────────────────────
///
/// 注册 MOON→EARTH 位置对 + (MOON_PA, J2000) 帧对，与 moon tide 路径匹配。
fn enable_moon_tide_cache(et_start: f64, et_end: f64, dt: f64) {
    let cache = ephem_cache::EphemCache::build(
        &[
            ("MOON".to_string(), "EARTH".to_string()),
            ("EARTH".to_string(), "SOLAR SYSTEM BARYCENTER".to_string()),
        ],
        &[
            ("MOON_PA".to_string(), "J2000".to_string()),
            // effective_coefficients 内部查 (input_frame, "J2000")
            // input_frame_for_body("MOON") = "MOON_PA"
        ],
        &[],
        et_start,
        et_end,
        dt,
    )
    .expect("EphemCache build failed");
    ephem_cache::enable(cache);
}

// ── 测试 1：k_love=0 → 潮汐修正全零 ────────────────────────────────────

/// k_love 全零时，effective_coefficients 应返回与 base 系数完全相同的 C/S。
///
/// 依据：solid_tide_step1 是 k_love 的线性函数（`kn = k_love[n][m] / (2n+1)`），
/// k_love=0 ⇒ ΔC=ΔS=0。月球场景无 Step2，因此全路径线性。
///
/// 按 ADR 0013：这是解析解边界条件，不是黄金样本比对。
#[test]
fn test_tide_delta_zero_when_k_love_zero() {
    load_kernels();

    let et = 0.0; // J2000
                  // 月球系数（随便填，仅作 base 参照）
    let nn = 5usize; // degree+1 = 5
    let base_c: Vec<f64> = (0..nn * nn).map(|i| 1e-6 * (i as f64 + 1.0)).collect();
    let base_s: Vec<f64> = (0..nn * nn).map(|i| 1e-7 * (i as f64 + 1.0)).collect();
    let k_love_zero = vec![0.0_f64; 25];
    let tide = TideConfig {
        mode: TideMode::Solid,
        k_love_flat: k_love_zero,
        k_plus_flat: None,
    };

    let (c_eff, s_eff) = effective_coefficients(
        et, "MOON", &base_c, &base_s, 4902.8001, // mu_moon (km³/s²)
        1738.0,    // radius_moon (km)
        &tide,
    )
    .expect("effective_coefficients failed");

    // 有效系数应与 base 逐元素完全相同（全机器精度容差）
    for i in 0..base_c.len() {
        let diff_c = (c_eff[i] - base_c[i]).abs();
        assert!(
            diff_c < 1e-30,
            "k_love=0: c_eff[{i}] = {}, base = {}, diff = {diff_c}",
            c_eff[i],
            base_c[i],
        );
        let diff_s = (s_eff[i] - base_s[i]).abs();
        assert!(
            diff_s < 1e-30,
            "k_love=0: s_eff[{i}] = {}, base = {}, diff = {diff_s}",
            s_eff[i],
            base_s[i],
        );
    }
}

// ── 测试 2：缓存命中位置与 cspice 直接查一致 ────────────────────────────

/// EphemCache 三次样条插值与 cspice spkezr 直接查询的位置误差 < 1e-6 km（亚米级）。
///
/// 依据（ADR 0013 / 物理定义）：两者查的是同一星历数据（de430.bsp），误差
/// 纯粹来自三次样条插值（dt=10s 网格下，月球位移精度 ~1e-6 km，亚米级）。
/// 不是黄金样本——是插值逼近理论一致性的数量界限。
#[test]
fn test_perturber_position_cache_hit_consistency() {
    load_kernels();

    let et_start = 0.0; // J2000
    let et_end = 86400.0; // 1 天
    let dt = 10.0; // 10s 网格，样条精度 ~1e-6 km
    enable_moon_tide_cache(et_start, et_end, dt);

    // 采样 5 个时刻（在缓存覆盖范围内，留 margin）
    let ets = [1000.0, 20000.0, 43200.0, 70000.0, 85000.0];

    for &et in &ets {
        let cached = ephem_cache::lookup_body_position("MOON", "EARTH", et)
            .expect("cache lookup failed")
            .expect("cache miss (should have been covered)");

        let (st, _) =
            spice_ffi::spkezr("MOON", et, "J2000", "NONE", "EARTH").expect("spkezr failed");
        let direct = [st[0], st[1], st[2]];

        for k in 0..3 {
            let diff = (cached[k] - direct[k]).abs();
            assert!(
                diff < 1e-6,
                "et={et}: pos[{k}] cache={cached_k}, direct={direct_k}, diff={diff:.2e} > 1e-6 km",
                cached_k = cached[k],
                direct_k = direct[k],
            );
        }
    }

    ephem_cache::disable();
}

// ── 测试 3：帧旋转矩阵缓存与 pxform 一致 ────────────────────────────────

/// EphemCache 三次样条插值的 MOON_PA→J2000 旋转矩阵与 pxform 直接查
/// 逐元素误差 < 1e-10（旋转矩阵元素量级 O(1)，10s 网格三次样条精度 ~1e-10）。
///
/// 依据（ADR 0013 / 物理定义）：两者查的是同一 PCK 数据（MOON_PA / SPICELuna*），
/// 误差纯粹来自插值，不依赖任何外部软件或黄金样本。
#[test]
fn test_frame_matrix_cache_hit_consistency() {
    load_kernels();

    let et_start = 0.0;
    let et_end = 86400.0;
    let dt = 10.0;
    enable_moon_tide_cache(et_start, et_end, dt);

    let ets = [1000.0, 20000.0, 43200.0, 70000.0, 85000.0];

    for &et in &ets {
        let cached = ephem_cache::lookup_frame_matrix("MOON_PA", "J2000", et)
            .expect("frame cache lookup failed")
            .expect("frame cache miss");

        let direct = spice_ffi::pxform("MOON_PA", "J2000", et).expect("pxform failed");

        for i in 0..3 {
            for j in 0..3 {
                let diff = (cached[i][j] - direct[i][j]).abs();
                assert!(
                    diff < 1e-10,
                    "et={et}: R[{i}][{j}] cache={cv}, direct={dv}, diff={diff:.2e} > 1e-10",
                    cv = cached[i][j],
                    dv = direct[i][j],
                );
            }
        }
    }

    ephem_cache::disable();
}
