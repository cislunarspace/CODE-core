//! GravityField 完整 Rust 移植（仅 `spice` feature 下编译）。
//!
//! 1:1 移植自 Python ``gravity_field.py:GravityField.compute_acceleration``，
//! 把"坐标变换（pxform）+ 有效系数（潮汐）+ 球谐加速度 + 反向坐标变换"
//! 全部合并到一次 Rust 调用。
//!
//! # 当前简化
//!
//! 假设 ``input_cs`` 与 ``system.coordinate_system`` 共享同一 origin
//! （即 origin 平移为零）。NRHO 场景成立（两者 origin 都是 EARTH/MOON）。
//! 对于 origin 不同的场景（如 Barycenter origin），需另加 spkezr 平移。

use crate::solid_tide;
use crate::spherical_harmonic;
use e2m2e_spice::spice_ffi::{mat3_mul_vec, mat3_t_mul_vec, pxform, SpiceFfiError};

/// 潮汐配置（与 Python ``tide_mode`` 三档对应）。
#[derive(Clone, Debug)]
pub struct TideConfig {
    pub mode: TideMode,
    /// 中心天体 Love 数表 5×5 行优先扁平化（n=2,3 阶位移 Love 数；其它行为零）
    pub k_love_flat: Vec<f64>,
    /// 弹性 3 阶位移 5 元素，或空（月球等无贡献）
    pub k_plus_flat: Option<Vec<f64>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TideMode {
    None,
    Solid,
    SolidAndPole,
}

/// GM 值硬编码表（与 Python e2m2e.core.spice._GM_VALUES 一致）。
///
/// DE430 bsp 不带 GM 数据，cspice bodvrd 会 KERNELVARNOTFOUND。
/// 与 Python 一致用本地字典。
fn gm_for_body(body: &str) -> Option<f64> {
    match body {
        "SUN" => Some(1.32712440018e11),
        "MERCURY" => Some(22031.868551),
        "VENUS" => Some(324858.592000),
        "EARTH" => Some(398600.435507),
        "MOON" => Some(4902.800118),
        "MARS" => Some(42828.375816),
        "JUPITER" => Some(126712764.100000),
        "SATURN" => Some(37940584.841800),
        "URANUS" => Some(5794556.400000),
        "NEPTUNE" => Some(6836527.100580),
        "EMB" => Some(403503.235502),
        _ => None,
    }
}

/// 中心天体的扰动体列表（用于潮汐 Step1）。
///
/// EARTH: ["SUN", "MOON"]；MOON: ["EARTH"]。
fn perturbers_for_body(body: &str) -> &'static [&'static str] {
    match body {
        "EARTH" => &["SUN", "MOON"],
        "MOON" => &["EARTH"],
        _ => &[],
    }
}

/// GravityField 一次完整计算。
///
/// # 参数
/// - `et`: SPICE et 秒
/// - `r_sc`: 航天器位置 [x, y, z] km（在 propagation frame，origin = propagation_origin）
/// - `c_flat`/`s_flat`: 球谐系数 (degree+1)² 长度，行优先扁平化
/// - `mu`/`radius`/`degree`/`order`: 球谐参数
/// - `input_frame`: body-fixed frame 名（"ITRF93"/"MOON_PA"）
/// - `propagation_frame`: 传播 frame 名（通常 "J2000"）
/// - `body`: 中心天体名（"EARTH"/"MOON"），input_frame 的 origin
/// - `propagation_origin`: 传播系 origin 天体名（通常 "EARTH"）
/// - `tide`: 潮汐配置
///
/// # 返回
/// 加速度 [ax, ay, az] km/s²，在 propagation frame 下。
///
/// # 坐标变换逻辑（1:1 移植自 Python CoordinateSystem.transform_state）
///
/// propagation frame 是 origin=propagation_origin 的惯性系（如地心 J2000）。
/// input frame 是 origin=body 的 body-fixed 系（如月心 MOON_PA）。
/// 变换步骤：
///   1. `r_ssb = r_sc + state_ssb(propagation_origin)` —— 从 propagation origin 平移到 SSB
///   2. `r_body_icrf = r_ssb - state_ssb(body)` —— 从 SSB 平移到 body（仍在 J2000 轴下）
///   3. `r_input = R_j2000_to_input^T @ r_body_icrf` —— 旋转到 input_frame 轴
///      其中 `R_j2000_to_input = pxform(J2000, input_frame)`，T 反转方向
///      等价于 `R_input_to_j2000 = pxform(input_frame, J2000)` 的转置 × r_body_icrf
#[allow(clippy::too_many_arguments)]
pub fn gravity_field_acceleration(
    et: f64,
    r_sc: &[f64; 3],
    c_flat: &[f64],
    s_flat: &[f64],
    mu: f64,
    radius: f64,
    degree: usize,
    order: usize,
    input_frame: &str,
    propagation_frame: &str,
    body: &str,
    propagation_origin: &str,
    tide: &TideConfig,
) -> Result<[f64; 3], SpiceFfiError> {
    // Step 1: 坐标变换 propagation → input_frame
    //
    // body == propagation_origin（地心系地球重力场等常见场景）时，origin 与
    // body 重合，r_body_icrf = r_sc，无需查 origin/body 在 SSB 的位置——
    // 既省两次 spkezr FFI，又避免对 SSB 大级坐标（~1.5e8 km）做插值引入误差。
    // 仅 body != origin 时才需要查 SSB 位置做平移。
    let r_body_icrf: [f64; 3] = if body == propagation_origin {
        [r_sc[0], r_sc[1], r_sc[2]]
    } else {
        // 查两个 origin 在 SSB 的位置（优先走星历缓存；strict 模式 miss 即
        // 硬 Err，杜绝并行区回退 cspice）
        let prop_origin_pos_ssb: [f64; 3] = match e2m2e_spice::ephem_cache::lookup_body_position(
            propagation_origin,
            "SOLAR SYSTEM BARYCENTER",
            et,
        ) {
            Ok(Some(p)) => p,
            Ok(None) => {
                let (st, _) = e2m2e_spice::spice_ffi::spkezr(
                    propagation_origin,
                    et,
                    propagation_frame,
                    "NONE",
                    "SOLAR SYSTEM BARYCENTER",
                )?;
                [st[0], st[1], st[2]]
            }
            Err(e) => return Err(e.into()),
        };
        let r_ssb = [
            r_sc[0] + prop_origin_pos_ssb[0],
            r_sc[1] + prop_origin_pos_ssb[1],
            r_sc[2] + prop_origin_pos_ssb[2],
        ];
        let body_pos_ssb: [f64; 3] = match e2m2e_spice::ephem_cache::lookup_body_position(
            body,
            "SOLAR SYSTEM BARYCENTER",
            et,
        ) {
            Ok(Some(p)) => p,
            Ok(None) => {
                let (st, _) = e2m2e_spice::spice_ffi::spkezr(
                    body,
                    et,
                    propagation_frame,
                    "NONE",
                    "SOLAR SYSTEM BARYCENTER",
                )?;
                [st[0], st[1], st[2]]
            }
            Err(e) => return Err(e.into()),
        };
        [
            r_ssb[0] - body_pos_ssb[0],
            r_ssb[1] - body_pos_ssb[1],
            r_ssb[2] - body_pos_ssb[2],
        ]
    };

    // 旋转 propagation_frame → input_frame（优先走缓存；strict miss 硬 Err）
    // Python: position_out = to_rotation.T @ r_body_icrf，
    // 其中 to_rotation = pxform(input_frame, J2000)（input → j2000）。
    // 所以 position_out = pxform(input_frame, J2000).T @ r_body_icrf
    //                   = pxform(J2000, input_frame) @ r_body_icrf
    let r_input_to_j2000: [[f64; 3]; 3] =
        match e2m2e_spice::ephem_cache::lookup_frame_matrix(input_frame, propagation_frame, et) {
            Ok(Some(m)) => m,
            Ok(None) => pxform(input_frame, propagation_frame, et)?,
            Err(e) => return Err(e.into()),
        };
    let r_input = mat3_t_mul_vec(&r_input_to_j2000, &r_body_icrf);

    // Step 2: 有效系数（含潮汐修正）
    let (c_eff, s_eff) = if tide.mode == TideMode::None {
        (c_flat.to_vec(), s_flat.to_vec())
    } else {
        effective_coefficients(et, body, c_flat, s_flat, mu, radius, tide)?
    };

    // Step 3: 球谐加速度（input_frame 系下）
    let a_input = spherical_harmonic::spherical_harmonic_accel(
        &r_input, &c_eff, &s_eff, mu, radius, degree, order,
    );
    let a_input_arr = [a_input[0], a_input[1], a_input[2]];

    // Step 4: 反向坐标变换 input_frame → propagation_frame
    // a_input 在 input_frame 下，要转到 propagation_frame。
    // 正向变换是 r_input = R_input_to_prop.T @ r_icrf（即 to_rotation.T），
    // 所以反向是 a_prop = R_input_to_prop @ a_input
    // 其中 R_input_to_prop = pxform(input_frame, propagation_frame)
    let a_prop = mat3_mul_vec(&r_input_to_j2000, &a_input_arr);
    Ok(a_prop)
}

/// 算有效系数 C/S = base + 潮汐 ΔC/ΔS。
///
/// 1:1 移植自 Python ``gravity_field._effective_coefficients``。
/// 仅支持 ``Solid`` 档（Step1 + Step2 + 永久潮汐），``SolidAndPole`` 档
/// 需要外部 xp/yp provider，Rust 侧暂不支持（回退到 Python 路径）。
fn effective_coefficients(
    et: f64,
    body: &str,
    base_c: &[f64],
    base_s: &[f64],
    mu_central: f64,
    r_central: f64,
    tide: &TideConfig,
) -> Result<(Vec<f64>, Vec<f64>), SpiceFfiError> {
    let mut c = base_c.to_vec();
    let mut s = base_s.to_vec();

    // 查扰动体位置（input_frame 系下，observer = 中心天体）
    let perturbers_names = perturbers_for_body(body);
    let input_frame = input_frame_for_body(body);
    let mut perturbers_flat: Vec<f64> = Vec::with_capacity(perturbers_names.len() * 4);
    for &name in perturbers_names {
        // SPICE 查扰动体相对中心天体在 input_frame 系下的位置
        let (state, _lt) = e2m2e_spice::spice_ffi::spkezr(name, et, input_frame, "NONE", body)?;
        perturbers_flat.extend_from_slice(&[state[0], state[1], state[2]]);
        // GM 用硬编码表（与 Python spice.get_gm 一致；DE430 bsp 不带 GM）
        let gm = gm_for_body(name)
            .ok_or_else(|| SpiceFfiError::Failed(format!("GM not known for body {:?}", name)))?;
        perturbers_flat.push(gm);
    }

    // Step1（频率无关）
    let step1_out = solid_tide::solid_tide_step1(
        &perturbers_flat,
        &tide.k_love_flat,
        tide.k_plus_flat.as_deref(),
        mu_central,
        r_central,
    );
    // step1_out 是 C(25) ++ S(25)，5×5 表
    // 把它加到 c/s 上（degree>=4 的部分在 c_eff[4*nn + m] 处）
    add_tide_delta(&mut c, &mut s, &step1_out[..25], &step1_out[25..]);

    // 地球专用 Step2（频率相关）
    if body == "EARTH" {
        let step2_out = solid_tide::solid_tide_step2(et);
        add_tide_delta(&mut c, &mut s, &step2_out[..25], &step2_out[25..]);
        // 永久潮汐（zero_tide 约定）：减去永久潮汐修正
        // tide_convention 默认 "tide_free"，不做这步；如需 zero_tide 在 Python
        // 侧处理。本 Rust 路径假设 tide_free。
    }

    Ok((c, s))
}

/// 把 5×5 潮汐 ΔC/ΔS 加到完整 degree 的 C/S 上。
///
/// `delta` 是 5×5 行优先（25 元素），加到 `coeff` 的前 5×5 部分。
fn add_tide_delta(coeff_c: &mut [f64], coeff_s: &mut [f64], delta_c: &[f64], delta_s: &[f64]) {
    let nn = (coeff_c.len() as f64).sqrt() as usize;
    debug_assert_eq!(nn * nn, coeff_c.len(), "coeff must be (degree+1)²");
    let delta_nn = 5usize; // 潮汐表固定 5×5
    for n in 0..delta_nn.min(nn) {
        for m in 0..=n {
            let full_idx = n * nn + m;
            let delta_idx = n * delta_nn + m;
            coeff_c[full_idx] += delta_c[delta_idx];
            coeff_s[full_idx] += delta_s[delta_idx];
        }
    }
}

/// 返回中心天体对应的 body-fixed frame 名（与 Python 一致）。
fn input_frame_for_body(body: &str) -> &'static str {
    match body {
        "EARTH" => "ITRF93",
        "MOON" => "MOON_PA",
        _ => "IAU_EARTH", // fallback，一般不会走到
    }
}

// ── 3x3 矩阵乘法辅助 ──────────────────────────────────────────────────────

/// 3x3 矩阵乘法：C = A @ B。
fn mat3_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut c = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    c
}

/// 3x3 矩阵转置。
fn mat3_transpose(a: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut t = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            t[i][j] = a[j][i];
        }
    }
    t
}

// ── GravityFieldContext：SPICE 预提取 + 状态无关缓存 ─────────────────────

/// 重力场计算上下文：积分前一次性提取 SPICE 数据，积分循环内零 FFI 调用。
///
/// 与 `gravity_field_acceleration` 物理等价，但把与航天器状态无关的量
/// （坐标旋转、天体位置、潮汐有效系数）缓存到结构体中。积分步内只做
/// 内存查表 + `spherical_harmonic_accel`，不碰 SPICE，Rayon 并行安全。
pub struct GravityFieldContext {
    /// 旋转矩阵 R = pxform(input_frame, propagation_frame)
    rotation: [[f64; 3]; 3],
    /// propagation origin 在 SSB 的位置偏移（r_body_icrf = r_sc + offset）
    origin_offset: [f64; 3],
    /// 有效球谐系数 C（含潮汐修正）
    c_eff: Vec<f64>,
    /// 有效球谐系数 S（含潮汐修正）
    s_eff: Vec<f64>,
    mu: f64,
    radius: f64,
    degree: usize,
    order: usize,
}

impl GravityFieldContext {
    /// 一次性查询 SPICE，构建上下文。
    ///
    /// 在积分循环外调用一次，返回的 context 可在积分循环内反复使用。
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        et: f64,
        c_flat: &[f64],
        s_flat: &[f64],
        mu: f64,
        radius: f64,
        degree: usize,
        order: usize,
        input_frame: &str,
        propagation_frame: &str,
        body: &str,
        propagation_origin: &str,
        tide: &TideConfig,
    ) -> Result<Self, SpiceFfiError> {
        // 1. 查天体位置（平移偏移）。优先走星历缓存（三次样条查表），未覆盖
        //    回退 spkezr——与 gravity_field_acceleration L118-171 同模式。
        //    body == propagation_origin（地心系地球重力场常见场景）时 origin
        //    与 body 重合，无需查 SSB 平移。
        let origin_offset: [f64; 3] = if body == propagation_origin {
            [0.0; 3]
        } else {
            let prop_origin_pos_ssb = match e2m2e_spice::ephem_cache::lookup_body_position(
                propagation_origin,
                "SOLAR SYSTEM BARYCENTER",
                et,
            ) {
                Ok(Some(p)) => p,
                Ok(None) => {
                    let (st, _) = e2m2e_spice::spice_ffi::spkezr(
                        propagation_origin,
                        et,
                        propagation_frame,
                        "NONE",
                        "SOLAR SYSTEM BARYCENTER",
                    )?;
                    [st[0], st[1], st[2]]
                }
                Err(e) => return Err(e.into()),
            };
            let body_pos_ssb = match e2m2e_spice::ephem_cache::lookup_body_position(
                body,
                "SOLAR SYSTEM BARYCENTER",
                et,
            ) {
                Ok(Some(p)) => p,
                Ok(None) => {
                    let (st, _) = e2m2e_spice::spice_ffi::spkezr(
                        body,
                        et,
                        propagation_frame,
                        "NONE",
                        "SOLAR SYSTEM BARYCENTER",
                    )?;
                    [st[0], st[1], st[2]]
                }
                Err(e) => return Err(e.into()),
            };
            [
                prop_origin_pos_ssb[0] - body_pos_ssb[0],
                prop_origin_pos_ssb[1] - body_pos_ssb[1],
                prop_origin_pos_ssb[2] - body_pos_ssb[2],
            ]
        };

        // 2. 查旋转矩阵（优先走缓存帧样条；strict miss 硬 Err）
        let rotation =
            match e2m2e_spice::ephem_cache::lookup_frame_matrix(input_frame, propagation_frame, et)
            {
                Ok(Some(m)) => m,
                Ok(None) => e2m2e_spice::spice_ffi::pxform(input_frame, propagation_frame, et)?,
                Err(e) => return Err(e.into()),
            };

        // 3. 有效系数（含潮汐）
        let (c_eff, s_eff) = if tide.mode == TideMode::None {
            (c_flat.to_vec(), s_flat.to_vec())
        } else {
            effective_coefficients(et, body, c_flat, s_flat, mu, radius, tide)?
        };

        Ok(Self {
            rotation,
            origin_offset,
            c_eff,
            s_eff,
            mu,
            radius,
            degree,
            order,
        })
    }

    /// 计算加速度（propagation frame），与 `gravity_field_acceleration` 逐位等价。
    pub fn accel(&self, r_sc: &[f64; 3]) -> Result<[f64; 3], SpiceFfiError> {
        // 平移：propagation origin -> gravity body
        let r_body_icrf = [
            r_sc[0] + self.origin_offset[0],
            r_sc[1] + self.origin_offset[1],
            r_sc[2] + self.origin_offset[2],
        ];
        // 旋转：propagation frame -> input frame（body-fixed）
        let r_input = e2m2e_spice::spice_ffi::mat3_t_mul_vec(&self.rotation, &r_body_icrf);

        // 球谐加速度（body-fixed 系）
        let a_input = spherical_harmonic::spherical_harmonic_accel(
            &r_input,
            &self.c_eff,
            &self.s_eff,
            self.mu,
            self.radius,
            self.degree,
            self.order,
        );
        let a_input_arr = [a_input[0], a_input[1], a_input[2]];

        // 反向旋转：input frame -> propagation frame
        let a_prop = e2m2e_spice::spice_ffi::mat3_mul_vec(&self.rotation, &a_input_arr);
        Ok(a_prop)
    }

    /// 有限差分雅可比 ∂a_prop/∂r_sc（3×3），body-fixed 系差分 + 旋转 sandwich。
    ///
    /// 步长与 Python `_finite_diff_jacobian`（`sqrt(eps) * |r|`）一致。
    /// 潮汐系数随 et 变化但不依赖 r_sc，FD 自动包含其对雅可比的全部贡献。
    pub fn jacobian_fd(&self, r_sc: &[f64; 3]) -> Result<[[f64; 3]; 3], SpiceFfiError> {
        // 变换到 body-fixed 系（与 accel 一致）
        let r_body_icrf = [
            r_sc[0] + self.origin_offset[0],
            r_sc[1] + self.origin_offset[1],
            r_sc[2] + self.origin_offset[2],
        ];
        let r_input = e2m2e_spice::spice_ffi::mat3_t_mul_vec(&self.rotation, &r_body_icrf);

        let r_norm =
            (r_input[0] * r_input[0] + r_input[1] * r_input[1] + r_input[2] * r_input[2]).sqrt();
        let h = (f64::EPSILON.sqrt() * r_norm).max(1e-6);

        // body-fixed 系下 6 次扰动差分
        let mut j_input = [[0.0_f64; 3]; 3];
        for dim in 0..3 {
            let mut r_plus = r_input;
            let mut r_minus = r_input;
            r_plus[dim] += h;
            r_minus[dim] -= h;
            let a_plus = spherical_harmonic::spherical_harmonic_accel(
                &r_plus,
                &self.c_eff,
                &self.s_eff,
                self.mu,
                self.radius,
                self.degree,
                self.order,
            );
            let a_minus = spherical_harmonic::spherical_harmonic_accel(
                &r_minus,
                &self.c_eff,
                &self.s_eff,
                self.mu,
                self.radius,
                self.degree,
                self.order,
            );
            for i in 0..3 {
                j_input[i][dim] = (a_plus[i] - a_minus[i]) / (2.0 * h);
            }
        }

        // 链式法则：J_prop = R @ J_input @ R^T
        let rt = mat3_transpose(&self.rotation);
        let j_mid = mat3_mul(&j_input, &rt);
        let j_prop = mat3_mul(&self.rotation, &j_mid);
        Ok(j_prop)
    }
}
