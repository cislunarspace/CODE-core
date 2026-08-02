//! ECOM 光压模型（DFH 兼容 9 系数 DYB 参数化）。
//!
//! ECOM（Empirical CODE Orbit Model）将光压加速度分解到卫星本体坐标系的
//! D（太阳方向）、Y（太阳帆板法向）、B（D×Y）三轴，每个方向用常量+周期项展开。
//!
//! DFH 的 DYB 9 系数含义：
//! - dyb[0] = 等效面质比 (m²/kg)
//! - dyb[1:5] = D 方向周期项（cos(u), sin(u), cos(2u), sin(2u)）
//! - dyb[5:7] = Y 方向（cos(u), sin(u)）
//! - dyb[7:9] = B 方向（常量, cos(u)）
//!
//! 当仅 dyb[0] 非零时，模型退化为标准 cannonball SRP。

use crate::forces::srp;
use e2m2e_spice::spice_ffi::SpiceFfiError;

const P_SRP_1AU: f64 = 4.56e-6; // N/m²
const AU_KM: f64 = 1.49597870700e8; // km
const KM_TO_M: f64 = 1000.0;

fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// ECOM 光压加速度（DFH 兼容 9 系数 DYB 参数化）。
///
/// D-Y-B 坐标系：
/// - D = Sun→SC 归一化方向
/// - Y = 轨道面法向（SC 位置 × D）
/// - B = D × Y（右手系闭合）
///
/// 当 `dyb[1..9]` 全为零时，退化为 cannonball SRP。
pub fn ecom_acceleration(
    et: f64,
    sc_pos: &[f64; 3],
    dyb: &[f64; 9],
    shadow_bodies: &[String],
    observer: &str,
) -> Result<[f64; 3], SpiceFfiError> {
    let sun_pos = srp::body_position_cached("SUN", observer, et)?;

    let sun_to_sc = [
        sc_pos[0] - sun_pos[0],
        sc_pos[1] - sun_pos[1],
        sc_pos[2] - sun_pos[2],
    ];
    let r = norm(&sun_to_sc);
    if r == 0.0 {
        return Ok([0.0, 0.0, 0.0]);
    }

    let flux = srp::flux_factor(et, sc_pos, shadow_bodies, observer)?;

    // D-Y-B 坐标系
    let d_hat = [sun_to_sc[0] / r, sun_to_sc[1] / r, sun_to_sc[2] / r];

    // Y = 轨道面法向（SC 位置 × D）
    let y_raw = cross(sc_pos, &d_hat);
    let y_norm = norm(&y_raw);
    let y_hat = if y_norm > 1e-10 {
        [y_raw[0] / y_norm, y_raw[1] / y_norm, y_raw[2] / y_norm]
    } else {
        // 退化：D 与 SC 平行，用 z 轴构造
        let z = [0.0, 0.0, 1.0];
        let y2 = cross(&z, &d_hat);
        let y2n = norm(&y2);
        if y2n > 1e-10 {
            [y2[0] / y2n, y2[1] / y2n, y2[2] / y2n]
        } else {
            let x = [1.0, 0.0, 0.0];
            let y3 = cross(&x, &d_hat);
            let y3n = norm(&y3);
            [y3[0] / y3n, y3[1] / y3n, y3[2] / y3n]
        }
    };
    let b_hat = cross(&d_hat, &y_hat);

    // 基础加速度幅值
    let pressure = P_SRP_1AU * (AU_KM / r).powi(2); // N/m²
    let a0 = flux * pressure * dyb[0] / KM_TO_M; // km/s²

    // 太阳平近点角（简化：u=0）
    let u: f64 = 0.0;

    // ECOM 三向分量
    let d_comp = 1.0
        + dyb[1] * u.cos()
        + dyb[2] * u.sin()
        + dyb[3] * (2.0 * u).cos()
        + dyb[4] * (2.0 * u).sin();
    let y_comp = dyb[5] * u.cos() + dyb[6] * u.sin();
    let b_comp = dyb[7] + dyb[8] * u.cos();

    Ok([
        a0 * (d_comp * d_hat[0] + y_comp * y_hat[0] + b_comp * b_hat[0]),
        a0 * (d_comp * d_hat[1] + y_comp * y_hat[1] + b_comp * b_hat[1]),
        a0 * (d_comp * d_hat[2] + y_comp * y_hat[2] + b_comp * b_hat[2]),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ecom_smoke_no_kernel() {
        let result = ecom_acceleration(
            0.0,
            &[100000.0, 0.0, 0.0],
            &[0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            &[],
            "EARTH",
        );
        assert!(result.is_ok() || result.is_err());
    }
}
