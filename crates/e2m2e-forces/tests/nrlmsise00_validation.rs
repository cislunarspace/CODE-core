//! NRLMSISE-00 数值对拍：与 nyx 的验证夹具逐点比较密度与温度。
//!
//! # 数据来源
//!
//! `tests/data/nrlmsise00_validation.csv` 由 nyx-space/nyx 的
//! `data/03_tests/nrlmsise00_validation.json`（AGPL 项目发布的对照数据文件，
//! 仅作数值 oracle 使用，未参考其实现代码）转录；nyx 侧又经 pymsis / 官方
//! Fortran 基线交叉验证。夹具覆盖 100 / 120 / 483 km、纬度 0/15/45/80°、
//! 静态与暴时 Ap 史共 72 个工况。
//!
//! # 误差口径
//!
//! 夹具的 `expected_total_density_kg_m3` 与 `expected_temperature_k` **全部是
//! f32 可精确表示的值**（如 `1.88600997924804688e+02`），即该 JSON 由 f32 精度
//! 管线（含 f32 中间量积累）产生。因此本 f64 实现与夹具的偏差主要来自对方的
//! 精度损失，而非本实现的算法误差。
//!
//! 开发期已用独立 oracle（NRL 公有领域 C 参考实现 `nrlmsise-00.c`，release
//! 20041227）在同样 72 个工况上逐点比对：本实现与其中的 f64 结果**逐位相同**
//! （密度与温度相对误差均恰为 0）。对照夹具实测最大相对误差：密度 1.4e-6、
//! 温度 1.8e-7；断言容差取 1e-5（不宽于 1e-4 的上限），余量留给夹具精度噪声。

use std::path::PathBuf;

use e2m2e_forces::nrlmsise00::{density, Nrlmsise00Input};

/// 对拍容差：夹具为 f32 精度，实测最大偏差 ~1.4e-6。
const REL_TOL: f64 = 1e-5;

/// 单个工况（列顺序见 CSV 表头）。
struct Case {
    input: Nrlmsise00Input,
    storm: bool,
    density_kg_m3: f64,
    temperature_k: f64,
}

fn load_cases() -> Vec<Case> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join("nrlmsise00_validation.csv");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("读取夹具 {} 失败: {e}", path.display()));
    text.lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let f: Vec<f64> = line
                .split(',')
                .map(|tok| tok.trim().parse::<f64>().expect("CSV 数值"))
                .collect();
            assert_eq!(f.len(), 17, "列数不符: {line}");
            Case {
                input: Nrlmsise00Input {
                    altitude_km: f[0],
                    geodetic_lat_deg: f[1],
                    geodetic_lon_deg: f[2],
                    day_of_year: f[3] as u16,
                    ut_seconds: f[4],
                    f107_daily: f[5],
                    f107_avg: f[6],
                    ap: [f[7], f[8], f[9], f[10], f[11], f[12], f[13]],
                },
                storm: f[14] != 0.0,
                density_kg_m3: f[15],
                temperature_k: f[16],
            }
        })
        .collect()
}

/// 逐点对拍：密度与温度的相对误差均 ≤ [`REL_TOL`]。
#[test]
fn matches_nyx_validation_fixture() {
    let cases = load_cases();
    assert_eq!(cases.len(), 72, "夹具工况数");

    let mut worst_density = 0.0_f64;
    let mut worst_temp = 0.0_f64;
    let mut failures = Vec::new();

    for (i, c) in cases.iter().enumerate() {
        let out = density(&c.input, c.storm);
        let rel_d = (out.density_kg_m3 - c.density_kg_m3).abs() / c.density_kg_m3.abs();
        let rel_t = (out.temperature_k - c.temperature_k).abs() / c.temperature_k.abs();
        worst_density = worst_density.max(rel_d);
        worst_temp = worst_temp.max(rel_t);
        if rel_d > REL_TOL || rel_t > REL_TOL {
            failures.push(format!(
                "行 {i}: alt={} lat={} doy={} ut={} storm={} | ρ got={:.17e} want={:.17e} rel={rel_d:e} | T got={:.17e} want={:.17e} rel={rel_t:e}",
                c.input.altitude_km,
                c.input.geodetic_lat_deg,
                c.input.day_of_year,
                c.input.ut_seconds,
                c.storm,
                out.density_kg_m3,
                c.density_kg_m3,
                out.temperature_k,
                c.temperature_k,
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "{} / 72 工况超差（max rel ρ = {worst_density:e}, T = {worst_temp:e}）:\n{}",
        failures.len(),
        failures.join("\n"),
    );
}
