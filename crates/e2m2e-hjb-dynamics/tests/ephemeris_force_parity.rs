//! 星历 Hamiltonian 内部合力与 `compute_total_acceleration` 的逐点对拍
//! （ADR 0034 决策 6）。
//!
//! 同一 (et, state) 下，`EphemerisPlanar::inertial_accel` 经系变换后的
//! 惯性合力应与直接调用 `compute_total_acceleration` 一致。星历用解析
//! 圆化月轨 + 面内定点太阳经 `EphemCache::from_raw_grids` 合成（无内核
//! 依赖），双方走同一条缓存查询路径，差异只剩样条采样误差。

#![cfg(feature = "ephemeris")]

use e2m2e_forces::forces::compiled::{compute_total_acceleration, CompiledForce};
use e2m2e_hjb_dynamics::EphemerisPlanar;
use e2m2e_spice::ephem_cache::{self, EphemCache};

const MU_EARTH: f64 = 398600.435436;
const MU_MOON: f64 = 4902.800066;
const MU_SUN: f64 = 1.32712440018e11;
const L: f64 = 384400.0;
const OMEGA: f64 = 2.649146625237848e-6;
const SUN_POS: [f64; 3] = [1.5e8, 0.0, 0.0];

fn forces() -> Vec<CompiledForce> {
    vec![
        CompiledForce::PointMass { mu: MU_EARTH },
        CompiledForce::ThirdBody {
            body: "MOON".to_string(),
            mu: MU_MOON,
        },
        CompiledForce::ThirdBody {
            body: "SUN".to_string(),
            mu: MU_SUN,
        },
    ]
}

/// 圆化月轨解析状态（相对地球，惯性系）。
fn moon_state(et: f64) -> [f64; 6] {
    let (s, c) = (OMEGA * et).sin_cos();
    [L * c, L * s, 0.0, -L * OMEGA * s, L * OMEGA * c, 0.0]
}

#[test]
fn inertial_accel_matches_compute_total_acceleration() {
    let et_start = 0.0;
    let et_end = 10.0 * 86400.0;
    let dt = 600.0;
    let n = ((et_end - et_start) / dt) as usize + 1;
    let t_grid: Vec<f64> = (0..n).map(|i| et_start + i as f64 * dt).collect();

    let moon: Vec<[f64; 6]> = t_grid.iter().map(|&et| moon_state(et)).collect();
    let sun: Vec<[f64; 6]> = t_grid
        .iter()
        .map(|_| [SUN_POS[0], 0.0, 0.0, 0.0, 0.0, 0.0])
        .collect();
    let to_entry = |name: &str, states: &Vec<[f64; 6]>| {
        ((name.to_string(), "EARTH".to_string()), states.clone())
    };
    let cache = EphemCache::from_raw_grids(
        &t_grid,
        &[to_entry("MOON", &moon), to_entry("SUN", &sun)],
        &[],
        &[],
    )
    .expect("合成星历构造");
    ephem_cache::enable(cache);

    let span = (0.0_f64, et_end);
    let ham = EphemerisPlanar::new(
        forces(),
        et_start,
        span,
        1.0,
        300.0,
        9.80665,
        0.1,
        false,
        1000.0,
    );

    for k in 0..30 {
        let t = (k as f64 + 0.37) * 0.31 * 86400.0;
        let s = |i: usize| ((k * 13 + i * 29) % 101) as f64 / 50.0 - 1.0;
        let (x, y) = (s(0) + 0.3, s(1));
        let fs = ham.frame(t);
        let (r_vec, a_rot) = ham.inertial_accel(&fs, x, y);
        let state6 = [r_vec[0], r_vec[1], r_vec[2], 0.0, 0.0, 0.0];
        let a_ref = compute_total_acceleration(&forces(), ham.et0 + t, &state6, "EARTH").unwrap();
        let ref_rot = [
            a_ref[0] * fs.e_r[0] + a_ref[1] * fs.e_r[1] + a_ref[2] * fs.e_r[2],
            a_ref[0] * fs.e_theta[0] + a_ref[1] * fs.e_theta[1] + a_ref[2] * fs.e_theta[2],
        ];
        // 差异只剩样条采样误差（600s 网格下 << 1e-9 km/s²）。
        for d in 0..2 {
            assert!(
                (a_rot[d] - ref_rot[d]).abs() < 1e-9,
                "t={t} ({x},{y}) 第 {d} 维：{} vs {}",
                a_rot[d],
                ref_rot[d]
            );
        }
    }

    ephem_cache::disable();
}
