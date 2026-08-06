//! 大气阻力力模型 Rust 移植（仅 `spice` feature 下编译）。
//!
//! 1:1 移植自 Python `drag.py:DragModel.compute_acceleration`：
//! - ITRF93 pxform 帧旋转（替代原 ITRFApproxAxes，决策 1b）
//! - 密度：`atmosphere::density(altitude_km, f107, ap)`
//! - 雅可比：中心差分 FD，扰动 J2000 状态 6 分量计算 ∂a/∂r 与 ∂a/∂v

use crate::atmosphere;
use e2m2e_spice::spice_ffi::{mat3_mul_vec, mat3_t_mul_vec, pxform, SpiceFfiError};

/// 地球赤道半径（km），与 Python `R_EARTH = 6378.1363` 一致。
const EARTH_RADIUS_KM: f64 = 6378.1363;

/// 千米 → 米换算因子，与 Python `KM_TO_M = 1000.0` 一致。
const KM_TO_M: f64 = 1000.0;

/// 阻力加速度 + 雅可比结果。
pub struct AccelDrag {
    pub acc: [f64; 3],
    pub jac_da_dr: [[f64; 3]; 3],
    pub jac_da_dv: [[f64; 3]; 3],
}

// ── 纯物理（无 SPICE 依赖，可独立单元测试）─────────────────────────────────

/// 在 body-fixed 系中计算阻力加速度（纯物理公式，不含坐标变换）。
///
/// 公式（与 Python `_compute_drag_in_itrf` 逐位一致）：
/// ```text
/// altitude = |r| - R_EARTH
/// ρ = atmosphere::density(altitude, f107_default, ap_default)
/// v_si = v * 1000
/// a_km = [-1/2 * ρ * BC * |v_si| * v_si] / 1000
///      = -1/2 * ρ * BC * 1000 * |v| * v
/// ```
///
/// # 注意
/// 本函数假设输入已在 ITRF（或等价 body-fixed 系）中，不做 pxform 旋转。
/// 供 `drag_accel` 管线内部调用，也供单元测试直接使用（不依赖 SPICE 内核）。
pub fn drag_accel_in_body_fixed(
    r_bf: &[f64; 3],
    v_bf: &[f64; 3],
    area: f64,
    mass: f64,
    cd: f64,
) -> [f64; 3] {
    let r_norm = (r_bf[0] * r_bf[0] + r_bf[1] * r_bf[1] + r_bf[2] * r_bf[2]).sqrt();
    let altitude_km = r_norm - EARTH_RADIUS_KM;
    let rho = atmosphere::density(altitude_km, 150.0, 15.0); // kg/m³，默认 f107/ap

    let v_mag = (v_bf[0] * v_bf[0] + v_bf[1] * v_bf[1] + v_bf[2] * v_bf[2]).sqrt();
    if rho == 0.0 || v_mag == 0.0 {
        return [0.0; 3];
    }

    let bc = cd * area / mass; // m²/kg

    // a_km = −½·ρ·BC·1000·|v_km|·v_km
    // 推导（与 Python drag.py:128-140 一致）：
    //   v_si = v_km·1000, |v_si| = |v_km|·1000
    //   a_si = −½·ρ·BC·|v_si|·v_si = −½·ρ·BC·1000²·|v_km|·v_km
    //   a_km = a_si / 1000 = −½·ρ·BC·1000·|v_km|·v_km
    let factor = -0.5 * rho * bc * KM_TO_M * v_mag;
    [factor * v_bf[0], factor * v_bf[1], factor * v_bf[2]]
}

// ── 完整管线（依赖 SPICE pxform，需 spicespice feature + 内核已加载）─────

/// 计算 drag 加速度（J2000 propagation frame → ITRF 旋转 → 阻力公式 → 旋转回 J2000）。
///
/// 物理流程：
/// 1. 查 ITRF93 → propagation_frame 帧旋转矩阵（优先走星历预采样缓存）
/// 2. state_J2000 → state_ITRF（R^T rotation，与 GravityField pxform 模式一致）
/// 3. `drag_accel_in_body_fixed` 算 ITRF 系内阻力
/// 4. a_ITRF → a_J2000（正向 R rotation）
pub fn drag_accel(
    et: f64,
    state: &[f64; 6],
    area: f64,
    mass: f64,
    cd: f64,
    propagation_frame: &str,
) -> Result<[f64; 3], SpiceFfiError> {
    // Step 1: 查 ITRF93 → propagation_frame（"J2000"）。
    let r_itrf_to_prop: [[f64; 3]; 3] =
        match e2m2e_spice::ephem_cache::lookup_frame_matrix("ITRF93", propagation_frame, et) {
            Ok(Some(m)) => m,
            Ok(None) => pxform("ITRF93", propagation_frame, et)?,
            Err(e) => return Err(e.into()),
        };

    // Step 2: 旋转到 ITRF（body-fixed）。
    let r_j2000 = [state[0], state[1], state[2]];
    let v_j2000 = [state[3], state[4], state[5]];
    let r_itrf = mat3_t_mul_vec(&r_itrf_to_prop, &r_j2000);
    let v_itrf = mat3_t_mul_vec(&r_itrf_to_prop, &v_j2000);

    let a_itrf = drag_accel_in_body_fixed(&r_itrf, &v_itrf, area, mass, cd);

    // Step 5: 旋转回 propagation frame。
    let a_prop = mat3_mul_vec(&r_itrf_to_prop, &a_itrf);
    Ok(a_prop)
}

/// 计算 drag 加速度 + 有限差分雅可比矩阵（∂a/∂r, ∂a/∂v）。
///
/// FD 在 J2000 分量上实施中心差分、封装完整管线（12 次 accel 评估）。
pub fn drag_accel_and_jacobian(
    et: f64,
    state: &[f64; 6],
    area: f64,
    mass: f64,
    cd: f64,
    propagation_frame: &str,
) -> Result<AccelDrag, SpiceFfiError> {
    let acc0 = drag_accel(et, state, area, mass, cd, propagation_frame)?;

    // 中心差分步长：√ε · max(1, |component|)，与 GravityField/SRP 统一
    let h_step = |val: f64| -> f64 { (f64::EPSILON.sqrt() * val.abs().max(1.0)).max(1e-6) };

    // ∂a/∂r：扰动 state[0:3]
    let mut jac_da_dr = [[0.0_f64; 3]; 3];
    for dim in 0..3 {
        let h = h_step(state[dim]);
        let mut s_plus = *state;
        let mut s_minus = *state;
        s_plus[dim] += h;
        s_minus[dim] -= h;
        let a_plus = drag_accel(et, &s_plus, area, mass, cd, propagation_frame)?;
        let a_minus = drag_accel(et, &s_minus, area, mass, cd, propagation_frame)?;
        for i in 0..3 {
            jac_da_dr[i][dim] = (a_plus[i] - a_minus[i]) / (2.0 * h);
        }
    }

    // ∂a/∂v：扰动 state[3:6]
    let mut jac_da_dv = [[0.0_f64; 3]; 3];
    for dim in 0..3 {
        let h = h_step(state[3 + dim]);
        let mut s_plus = *state;
        let mut s_minus = *state;
        s_plus[3 + dim] += h;
        s_minus[3 + dim] -= h;
        let a_plus = drag_accel(et, &s_plus, area, mass, cd, propagation_frame)?;
        let a_minus = drag_accel(et, &s_minus, area, mass, cd, propagation_frame)?;
        for i in 0..3 {
            jac_da_dv[i][dim] = (a_plus[i] - a_minus[i]) / (2.0 * h);
        }
    }

    Ok(AccelDrag {
        acc: acc0,
        jac_da_dr,
        jac_da_dv,
    })
}

// ── 单元测试（纯物理公式，不依赖 SPICE）────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    /// 验证纯物理公式通过 `atmosphere::density` 调出密度（量级正确的非零值）。
    #[test]
    fn test_drag_density_via_atmosphere() {
        let rho = atmosphere::density(400.0, 150.0, 15.0);
        assert!(rho > 1e-13);
        assert!(rho < 1e-10);
    }

    /// 零速度 → 零加速度（边界条件）。
    #[test]
    fn test_drag_zero_velocity_is_zero() {
        let r = [6778.0, 0.0, 0.0];
        let v = [0.0, 0.0, 0.0];
        let acc = drag_accel_in_body_fixed(&r, &v, 10.0, 1000.0, 2.2);
        assert_eq!(acc, [0.0; 3]);
    }

    /// 高度超出大气模型上限（1000 km）→ 零加速度。
    #[test]
    fn test_drag_above_ceiling_is_zero() {
        let r = [7578.0, 0.0, 0.0];
        let v = [6.0, 0.0, 0.0];
        let acc = drag_accel_in_body_fixed(&r, &v, 10.0, 1000.0, 2.2);
        assert_eq!(acc, [0.0; 3]);
    }

    /// 阻力加速度方向与速度方向相反（速度在 +y，加速度应在 -y）。
    #[test]
    fn test_drag_opposes_velocity() {
        let r = [6778.0, 0.0, 0.0];
        let v = [0.0, 7.7, 0.0];
        let acc = drag_accel_in_body_fixed(&r, &v, 10.0, 1000.0, 2.2);
        assert!(acc[1] < 0.0, "阻力 y 分量应为负，got {:?}", acc);
        assert!(
            acc[0].abs() < acc[1].abs() * 1e-12,
            "x 分量应可忽略，got {:?}",
            acc
        );
        assert!(
            acc[2].abs() < acc[1].abs() * 1e-12,
            "z 分量应可忽略，got {:?}",
            acc
        );
    }

    /// 阻力加速度量级与解析公式对照（与 Python test_drag_magnitude_matches_formula 一致）。
    #[test]
    fn test_drag_magnitude_matches_formula() {
        let altitude = 400.0; // km
        let r_sc = EARTH_RADIUS_KM + altitude; // ~6778 km
        let r = [r_sc, 0.0, 0.0];
        let v = [0.0, 7.7, 0.0];
        let area = 10.0;
        let mass = 1000.0;
        let cd = 2.2;

        let acc = drag_accel_in_body_fixed(&r, &v, area, mass, cd);

        let rho = atmosphere::density(altitude, 150.0, 15.0);
        let bc = cd * area / mass;
        let v_si = 7.7 * KM_TO_M;
        let a_expected_si = 0.5 * rho * bc * v_si * v_si;
        let a_expected_km = a_expected_si / KM_TO_M;

        let a_mag = (acc[0] * acc[0] + acc[1] * acc[1] + acc[2] * acc[2]).sqrt();
        let rel = (a_mag - a_expected_km).abs() / a_expected_km;
        assert!(
            rel <= 1e-10,
            "acc mag={a_mag:e}, expected={a_expected_km:e}, rel={rel:e}"
        );
    }

    /// 不同高度的阻力加速度量级随高度增大而衰减。
    #[test]
    fn test_drag_decays_with_altitude() {
        let area = 10.0;
        let mass = 1000.0;
        let cd = 2.2;
        let v = [0.0, 7.7, 0.0];

        let acc_400 = {
            let r = [EARTH_RADIUS_KM + 400.0, 0.0, 0.0];
            let a = drag_accel_in_body_fixed(&r, &v, area, mass, cd);
            (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
        };
        let acc_800 = {
            let r = [EARTH_RADIUS_KM + 800.0, 0.0, 0.0];
            let a = drag_accel_in_body_fixed(&r, &v, area, mass, cd);
            (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
        };

        assert!(acc_400 > 0.0);
        assert!(acc_800 > 0.0);
        // 800 km 处密度约 1.137e-14 <= 400 km 处 ~2.803e-12 的 1/100
        assert!(acc_800 < acc_400 / 100.0);
    }
}
