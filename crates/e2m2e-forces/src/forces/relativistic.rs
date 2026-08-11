//! RelativisticCorrection Rust 移植（仅 `spice` feature）。
//!
//! 1:1 移植自 Python ``relativistic_correction.py``：
//! - Schwarzschild（中心质量 1-PN）
//! - Lense-Thirring（自转角动量 LT，需要 sxform + 旋转矩阵导数）
//! - de Sitter（geodesic precession）

use e2m2e_propagation::constants::{
    EARTH_GRAVITY_REF_RADIUS_KM, JUPITER_MEAN_RADIUS_KM, MARS_MEAN_RADIUS_KM, MOON_MEAN_RADIUS_KM,
    SPEED_OF_LIGHT_KMS, SUN_MEAN_RADIUS_KM,
};
use e2m2e_spice::spice_ffi::{spkezr, sxform, SpiceFfiError};

/// 默认光速（km/s）。
const C_DEFAULT: f64 = SPEED_OF_LIGHT_KMS;

/// 中心天体赤道半径默认表（与 Python _DEFAULT_BODY_RADII_KM 一致）。
fn default_body_radius(body: &str) -> Option<f64> {
    match body {
        "EARTH" => Some(EARTH_GRAVITY_REF_RADIUS_KM),
        "MOON" => Some(MOON_MEAN_RADIUS_KM),
        "SUN" => Some(SUN_MEAN_RADIUS_KM),
        "MARS" => Some(MARS_MEAN_RADIUS_KM),
        "JUPITER" => Some(JUPITER_MEAN_RADIUS_KM),
        _ => None,
    }
}

/// body_fixed frame 名（与 Python ITRFSpiceAxes 默认一致）。
fn body_fixed_frame(body: &str) -> &str {
    match body {
        "EARTH" => "ITRF93",
        "MOON" => "MOON_PA",
        _ => "IAU_EARTH",
    }
}

/// RelativisticCorrection 完整加速度（Schwarzschild + LT + de Sitter）。
///
/// # 参数
/// - `et`: SPICE et 秒
/// - `state`: 航天器状态 [x, y, z, vx, vy, vz] km, km/s（observer 系下）
/// - `central_body`: 中心天体名（如 "EARTH"）
/// - `primary_body`: de Sitter 项的主导天体（通常 "SUN"），或 None 跳过 de Sitter
/// - `mu_central`: 中心天体 GM（km³/s²）
/// - `mu_primary`: 主导天体 GM（km³/s²），de Sitter 用
/// - `enable_schwarzschild` / `enable_lt` / `enable_de_sitter`: 三项开关
/// - `angular_momentum_vector`: 用户显式传入的 J 向量，或 None 自动算（LT 用）
/// - `body_radius_override`: 用户传入的天体半径，或 None 用默认表
#[allow(clippy::too_many_arguments)]
pub fn relativistic_acceleration(
    et: f64,
    state: &[f64; 6],
    central_body: &str,
    primary_body: Option<&str>,
    mu_central: f64,
    mu_primary: Option<f64>,
    enable_schwarzschild: bool,
    enable_lense_thirring: bool,
    enable_de_sitter: bool,
    angular_momentum_vector: Option<&[f64; 3]>,
    body_radius_override: Option<f64>,
    gamma: f64,
) -> Result<[f64; 3], SpiceFfiError> {
    let rv = [state[0], state[1], state[2]];
    let vv = [state[3], state[4], state[5]];
    let c = C_DEFAULT;
    let c2 = c * c;
    let r = norm(&rv);
    let v = norm(&vv);

    let mut acc = [0.0_f64; 3];

    // Schwarzschild
    if enable_schwarzschild {
        let s1 = mu_central / (c2 * r * r * r);
        let s2_0 = (4.0 * mu_central / r) - v * v;
        let rv_dot_vv = rv[0] * vv[0] + rv[1] * vv[1] + rv[2] * vv[2];
        // s2 = s2_0 * rv
        // s3 = 4 * rv_dot_vv * vv
        // acc += gamma * s1 * (s2 + s3)
        for k in 0..3 {
            let val = s2_0 * rv[k] + 4.0 * rv_dot_vv * vv[k];
            acc[k] += gamma * s1 * val;
        }
    }

    // Lense-Thirring
    if enable_lense_thirring {
        let j_vec: [f64; 3] = if let Some(j) = angular_momentum_vector {
            *j
        } else {
            compute_angular_momentum(et, central_body, body_radius_override)?
        };
        // rv_cross_vv = rv × vv
        let rv_cross_vv = cross(&rv, &vv);
        // vv_cross_j = vv × j
        let vv_cross_j = cross(&vv, &j_vec);
        let lt1 = 2.0 * mu_central / (c2 * r * r * r);
        let lt2 = (3.0 / (r * r)) * dot(&rv, &j_vec);
        for k in 0..3 {
            acc[k] += lt1 * (lt2 * rv_cross_vv[k] + vv_cross_j[k]);
        }
    }

    // de Sitter
    if enable_de_sitter {
        if let (Some(primary), Some(mu_p)) = (primary_body, mu_primary) {
            let omega = compute_de_sitter_omega(et, central_body, primary, mu_p)?;
            // acc += 2 * omega × vv
            let w_cross_v = cross(&omega, &vv);
            for k in 0..3 {
                acc[k] += 2.0 * w_cross_v[k];
            }
        }
    }

    Ok(acc)
}

/// 计算 body 的归一化角动量向量 J（GMAT 约定）。
///
/// 1:1 移植自 Python _compute_angular_momentum：
/// 1. sxform(body_frame, J2000) → 6×6 矩阵（含 R 和 Rdot）
/// 2. body_spin_vector 3 个分量
/// 3. body_spin_rate = |spin|
/// 4. J1 = [0, 0, (2/5)·R²·spin_rate]
/// 5. J = R @ J1（旋转到 J2000）
fn compute_angular_momentum(
    et: f64,
    central_body: &str,
    body_radius_override: Option<f64>,
) -> Result<[f64; 3], SpiceFfiError> {
    let radius = body_radius_override
        .or_else(|| default_body_radius(central_body))
        .ok_or_else(|| {
            SpiceFfiError::Failed(format!(
                "no default body radius for {:?}; provide override",
                central_body
            ))
        })?;
    let frame = body_fixed_frame(central_body);

    // 优先走星历缓存（strict 模式 miss 即硬 Err，杜绝并行区回退 cspice）
    let xform = match e2m2e_spice::ephem_cache::lookup_sxform(frame, "J2000", et) {
        Ok(Some(m)) => m,
        Ok(None) => sxform(frame, "J2000", et)?,
        Err(e) => return Err(e.into()),
    };
    // xform 是 [[f64;6];6]，前 3×3 是 R，后 3×3 是 Rdot
    let r = [
        [xform[0][0], xform[0][1], xform[0][2]],
        [xform[1][0], xform[1][1], xform[1][2]],
        [xform[2][0], xform[2][1], xform[2][2]],
    ];
    let rdot = [
        [xform[3][0], xform[3][1], xform[3][2]],
        [xform[4][0], xform[4][1], xform[4][2]],
        [xform[5][0], xform[5][1], xform[5][2]],
    ];

    // body_spin_vector（与 Python 公式逐字一致）
    let spin = [
        -r[0][2] * rdot[0][1] - r[1][2] * rdot[1][1] - r[2][2] * rdot[2][1],
        r[0][2] * rdot[0][0] + r[1][2] * rdot[1][0] + r[2][2] * rdot[2][0],
        -r[0][1] * rdot[0][0] - r[1][1] * rdot[1][0] - r[2][1] * rdot[2][0],
    ];
    let spin_rate = norm(&spin);

    // J1 = [0, 0, (2/5)·R²·spin_rate]
    let j1 = [0.0, 0.0, (2.0 / 5.0) * radius * radius * spin_rate];
    // J = R @ J1
    Ok(mat3_mul_vec(&r, &j1))
}

/// de Sitter omega 向量。
///
/// `omega = (3/2 v) × (-mu_primary/(c²r³) r)`
fn compute_de_sitter_omega(
    et: f64,
    central_body: &str,
    primary_body: &str,
    mu_primary: f64,
) -> Result<[f64; 3], SpiceFfiError> {
    // 查 central 和 primary 相对 SSB 的状态
    // 优先走星历缓存（strict 模式 miss 即硬 Err，杜绝并行区回退 cspice）
    let central_pos = match e2m2e_spice::ephem_cache::lookup_body_position(
        central_body,
        "SOLAR SYSTEM BARYCENTER",
        et,
    ) {
        Ok(Some(p)) => p,
        Ok(None) => {
            let (st, _) = spkezr(central_body, et, "J2000", "NONE", "SOLAR SYSTEM BARYCENTER")?;
            [st[0], st[1], st[2]]
        }
        Err(e) => return Err(e.into()),
    };
    let central_vel = match e2m2e_spice::ephem_cache::lookup_body_velocity(
        central_body,
        "SOLAR SYSTEM BARYCENTER",
        et,
    ) {
        Ok(Some(v)) => v,
        Ok(None) => {
            let (st, _) = spkezr(central_body, et, "J2000", "NONE", "SOLAR SYSTEM BARYCENTER")?;
            [st[3], st[4], st[5]]
        }
        Err(e) => return Err(e.into()),
    };
    let primary_pos = match e2m2e_spice::ephem_cache::lookup_body_position(
        primary_body,
        "SOLAR SYSTEM BARYCENTER",
        et,
    ) {
        Ok(Some(p)) => p,
        Ok(None) => {
            let (st, _) = spkezr(primary_body, et, "J2000", "NONE", "SOLAR SYSTEM BARYCENTER")?;
            [st[0], st[1], st[2]]
        }
        Err(e) => return Err(e.into()),
    };
    let primary_vel = match e2m2e_spice::ephem_cache::lookup_body_velocity(
        primary_body,
        "SOLAR SYSTEM BARYCENTER",
        et,
    ) {
        Ok(Some(v)) => v,
        Ok(None) => {
            let (st, _) = spkezr(primary_body, et, "J2000", "NONE", "SOLAR SYSTEM BARYCENTER")?;
            [st[3], st[4], st[5]]
        }
        Err(e) => return Err(e.into()),
    };
    let r_vec = [
        central_pos[0] - primary_pos[0],
        central_pos[1] - primary_pos[1],
        central_pos[2] - primary_pos[2],
    ];
    let v_vec = [
        central_vel[0] - primary_vel[0],
        central_vel[1] - primary_vel[1],
        central_vel[2] - primary_vel[2],
    ];
    let r = norm(&r_vec);
    let c2 = C_DEFAULT * C_DEFAULT;
    let factor = -mu_primary / (c2 * r * r * r);
    let pos = [factor * r_vec[0], factor * r_vec[1], factor * r_vec[2]];
    let vel = [1.5 * v_vec[0], 1.5 * v_vec[1], 1.5 * v_vec[2]];
    Ok(cross(&vel, &pos))
}

fn norm(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn mat3_mul_vec(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}
