//! 基于 cspice 的第三体引力加速度（仅 `spice` feature 下编译）。
//!
//! 1:1 移植自 Python ``third_body_gravity.py`` / ``indirect_term.py``，把
//! "SPICE 查扰动体位置 + 第三体摄动加速度公式" 合并为一次 Rust 调用，
//! 消除每步 Python↔cspice 跨界 + numpy 数组分配开销。
//!
//! 设计：Python 侧 ``ThirdBodyGravity`` 在初始化时调 ``spice_furnsh``
//! 加载内核（与 Python spiceypy 双 furnsh），运行时直接调本模块函数。

/// 非 strict 缓存 miss 时的星历回退：走 [`crate::spice_ffi::spkezr`]
/// （ADR 0051 纯 Rust 后端），返回类型保持 `cspice_rs::Error` 以不改动调用方
/// 签名——错误字段直接构造，详情放 explanation/long_message。
fn fallback_spkezr(target: &str, et: f64, observer: &str) -> Result<[f64; 3], cspice_rs::Error> {
    let (state, _lt) =
        crate::spice_ffi::spkezr(target, et, "J2000", "NONE", observer).map_err(|e| {
            cspice_rs::Error {
                short_message: "SPK_NATIVE".to_string(),
                explanation: e.to_string(),
                long_message: e.to_string(),
                traceback: String::new(),
            }
        })?;
    Ok([state[0], state[1], state[2]])
}

/// 第三体摄动加速度（含直接项 + 间接项）。
///
/// 公式与 ``third_body_gravity.py:ThirdBodyGravity.compute_acceleration`` 逐字对应：
/// ```text
/// a = -μ · [ (r_sc - r_ob) / |r_sc - r_ob|³  +  r_ob / |r_ob|³ ]
/// ```
/// 第一项为直接项（摄动体对航天器的引力），第二项为间接项（扣除摄动体对原点的引力）。
///
/// SPICE 查询：``spkezr(target, et, "J2000", NONE, observer)``。
///
/// 距离低于 ``min_distance`` 钳位（避免除零），与 Python ``MIN_DISTANCE = 1e-6`` km 一致。
///
/// # 参数
/// - `et`: SPICE et 秒（past J2000 TDB）
/// - `target`: 摄动天体名（"MOON"/"SUN"/"5"=JUPITER 等）
/// - `observer`: 原点天体名（通常 "EARTH"）
/// - `sc_pos`: 航天器位置 [x, y, z] km（相对 observer）
/// - `mu`: 摄动天体 GM（km³/s²）
/// - `min_distance`: 防除零钳位，默认 1e-6 km
pub fn third_body_acceleration(
    et: f64,
    target: &str,
    observer: &str,
    sc_pos: &[f64],
    mu: f64,
    min_distance: f64,
) -> Result<[f64; 3], cspice_rs::Error> {
    debug_assert_eq!(sc_pos.len(), 3, "sc_pos must have length 3");

    // 优先查星历缓存（strict 模式下 miss 即硬 Err，杜绝回退星历查询；
    // 非 strict 时 miss 软回退 spice_ffi::spkezr 纯 Rust 后端，ADR 0051），
    // 消除每步 FFI 跨界。
    let r_ob = match crate::ephem_cache::lookup_body_position(target, observer, et) {
        Ok(Some(pos)) => pos,
        Ok(None) => fallback_spkezr(target, et, observer)?,
        Err(e) => return Err(e.into()),
    };

    // r_bsc = r_sc - r_ob
    let r_bsc = [
        sc_pos[0] - r_ob[0],
        sc_pos[1] - r_ob[1],
        sc_pos[2] - r_ob[2],
    ];

    let r_bsc_norm = (r_bsc[0] * r_bsc[0] + r_bsc[1] * r_bsc[1] + r_bsc[2] * r_bsc[2]).sqrt();
    let r_ob_norm = (r_ob[0] * r_ob[0] + r_ob[1] * r_ob[1] + r_ob[2] * r_ob[2]).sqrt();

    let r_bsc_safe = if r_bsc_norm < min_distance {
        min_distance
    } else {
        r_bsc_norm
    };
    let r_ob_safe = if r_ob_norm < min_distance {
        min_distance
    } else {
        r_ob_norm
    };

    let inv_bsc3 = 1.0 / (r_bsc_safe * r_bsc_safe * r_bsc_safe);
    let inv_ob3 = 1.0 / (r_ob_safe * r_ob_safe * r_ob_safe);

    Ok([
        -mu * (r_bsc[0] * inv_bsc3 + r_ob[0] * inv_ob3),
        -mu * (r_bsc[1] * inv_bsc3 + r_ob[1] * inv_ob3),
        -mu * (r_bsc[2] * inv_bsc3 + r_ob[2] * inv_ob3),
    ])
}

/// 第三体摄动加速度 + 雅可比矩阵 ∂a/∂r_sc。
///
/// 与 `third_body_acceleration` 相同的物理模型，但同时返回加速度对航天器位置
/// 的偏导数（3×3 矩阵），用于 STM 变分方程。
///
/// 雅可比公式（仅直接项贡献，间接项不依赖 r_sc）：
/// ```text
/// ∂a/∂r = -μ · [ I/|r_bsc|³ - 3·(r_bsc⊗r_bsc)/|r_bsc|⁵ ]
/// ```
pub fn third_body_acceleration_and_jacobian(
    et: f64,
    target: &str,
    observer: &str,
    sc_pos: &[f64; 3],
    mu: f64,
    min_distance: f64,
) -> Result<([f64; 3], [[f64; 3]; 3]), cspice_rs::Error> {
    let r_ob = match crate::ephem_cache::lookup_body_position(target, observer, et) {
        Ok(Some(pos)) => pos,
        Ok(None) => fallback_spkezr(target, et, observer)?,
        Err(e) => return Err(e.into()),
    };

    let r_bsc = [
        sc_pos[0] - r_ob[0],
        sc_pos[1] - r_ob[1],
        sc_pos[2] - r_ob[2],
    ];

    let r_bsc_norm = (r_bsc[0] * r_bsc[0] + r_bsc[1] * r_bsc[1] + r_bsc[2] * r_bsc[2]).sqrt();
    let r_ob_norm = (r_ob[0] * r_ob[0] + r_ob[1] * r_ob[1] + r_ob[2] * r_ob[2]).sqrt();

    let r_bsc_safe = if r_bsc_norm < min_distance {
        min_distance
    } else {
        r_bsc_norm
    };
    let r_ob_safe = if r_ob_norm < min_distance {
        min_distance
    } else {
        r_ob_norm
    };

    let inv_bsc3 = 1.0 / (r_bsc_safe * r_bsc_safe * r_bsc_safe);
    let inv_ob3 = 1.0 / (r_ob_safe * r_ob_safe * r_ob_safe);

    let acc = [
        -mu * (r_bsc[0] * inv_bsc3 + r_ob[0] * inv_ob3),
        -mu * (r_bsc[1] * inv_bsc3 + r_ob[1] * inv_ob3),
        -mu * (r_bsc[2] * inv_bsc3 + r_ob[2] * inv_ob3),
    ];

    // 雅可比：∂a/∂r = -μ · [I/|r_bsc|³ - 3·r_bsc⊗r_bsc/|r_bsc|⁵]
    // 间接项不依赖 r_sc，所以只贡献直接项。
    let inv_bsc5 = inv_bsc3 / (r_bsc_safe * r_bsc_safe);
    let mut jac = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let delta = if i == j { 1.0 } else { 0.0 };
            jac[i][j] = -mu * (delta * inv_bsc3 - 3.0 * r_bsc[i] * r_bsc[j] * inv_bsc5);
        }
    }

    Ok((acc, jac))
}

/// 第三体间接项加速度：``a = -μ · r_ob / |r_ob|³``。
///
/// 与 ``indirect_term.py:IndirectTerm.compute_acceleration`` 逐字对应。用于
/// ``GravityField`` 中心天体场景（如月球的 GravityField 已含 degree=0 中心项，
/// 间接项需单独补）。
pub fn indirect_term_acceleration(
    et: f64,
    target: &str,
    observer: &str,
    mu: f64,
    min_distance: f64,
) -> Result<[f64; 3], cspice_rs::Error> {
    let r_ob = match crate::ephem_cache::lookup_body_position(target, observer, et) {
        Ok(Some(pos)) => pos,
        Ok(None) => fallback_spkezr(target, et, observer)?,
        Err(e) => return Err(e.into()),
    };
    let r_ob_norm = (r_ob[0] * r_ob[0] + r_ob[1] * r_ob[1] + r_ob[2] * r_ob[2]).sqrt();
    let r_ob_safe = if r_ob_norm < min_distance {
        min_distance
    } else {
        r_ob_norm
    };
    let inv_ob3 = 1.0 / (r_ob_safe * r_ob_safe * r_ob_safe);
    Ok([
        -mu * r_ob[0] * inv_ob3,
        -mu * r_ob[1] * inv_ob3,
        -mu * r_ob[2] * inv_ob3,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load_kernels() {
        // crates/e2m2e-integrators → crates → e2m2e 根
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
                // 双登记：native 注册表 + CSPICE 内核池（见 crate::furnish_kernel）。
                let _ = crate::furnish_kernel(&path.to_string_lossy());
            }
        }
    }

    /// MOON 第三体加速度应为有限值，量级 ~10⁻⁹ km/s²（地心 LEO 处月球潮汐残差）。
    ///
    /// 第三体摄动 = 直接引力 − 间接引力，两项主项（~μ/r_moon² ≈ 3×10⁻⁵ km/s²）
    /// 几乎抵消，残差是潮汐量级 μ·r_sc/r_moon³ ≈ 6×10⁻¹⁰ km/s²，而非引力主项本身。
    #[test]
    fn third_body_moon_finite_and_scale() {
        let _g = crate::lock_spice_for_test();
        load_kernels();
        let et = 0.0; // J2000
        let sc_pos = [7000.0, 0.0, 0.0]; // LEO
        let mu_moon = 4902.8001;
        let a = third_body_acceleration(et, "MOON", "EARTH", &sc_pos, mu_moon, 1e-6)
            .expect("third_body_acceleration failed");
        let norm = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        // LEO 处月球第三体潮汐残差 ~1e-9 km/s² = 1e-6 m/s²；
        // 上限 < 1e-7 可捕获间接项漏算/符号错（届时退化为引力主项 ~3e-5 km/s²）。
        assert!(
            norm > 1e-11 && norm < 1e-7,
            "moon 3rd body |a|={} out of range",
            norm
        );
    }

    /// SUN 第三体加速度量级 ~10⁻⁷ km/s²。
    #[test]
    fn third_body_sun_finite() {
        let _g = crate::lock_spice_for_test();
        load_kernels();
        let et = 0.0;
        let sc_pos = [7000.0, 0.0, 0.0];
        let mu_sun = 1.32712440018e11;
        let a = third_body_acceleration(et, "SUN", "EARTH", &sc_pos, mu_sun, 1e-6)
            .expect("sun third_body failed");
        let norm = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        assert!(norm.is_finite());
        assert!(norm > 0.0);
    }

    /// 间接项：MOON 在地心原点的 -μ·r/|r|³ 应有限。
    #[test]
    fn indirect_term_moon_finite() {
        let _g = crate::lock_spice_for_test();
        load_kernels();
        let et = 0.0;
        let mu_moon = 4902.8001;
        let a = indirect_term_acceleration(et, "MOON", "EARTH", mu_moon, 1e-6)
            .expect("moon indirect failed");
        let norm = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        // 在地心，月球 -μ/r² ≈ 4902.8 / 3.84e5² ≈ 3.3e-8 km/s²
        assert!(norm.is_finite());
        assert!(norm > 0.0);
    }
}
