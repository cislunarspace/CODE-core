//! 六域两层天图的传播内核（Primer §7.3 / Table 4，ADR 0041 Phase 3c）。
//!
//! 模型（物理单位 km / s）：地心惯性系点质量 N 体，摄动天体（月/日）走
//! **固定开普勒椭圆**（星历初值后孤立演化，论文 EM/EMS 定义）：
//!
//! ```text
//! a_sc = −μ⊕·r/|r|³ + Σ_b μ_b·[(r_b−r)/|r_b−r|³ − r_b/|r_b|³]
//! ```
//!
//! 增广 16 维：`[r(3), v(3), δr(3), δv(3), I₁, I₂]`——MEGNO 切变分取
//! **局部切向量**（δv̇ = J·δr，J = ∂a/∂r；天体不随航天器变，REBOUND
//! variational particle 同款语义），累加器与 `megno.rs` 逐项一致
//!（δ·δ̇ 为全相空间 6 维内积、步边界重归一化免补偿）。
//!
//! 命运事件在积分循环内检测（步端点符号变化 + 线性插值定位，地图诊断
//! 精度口径；终端事件早停为登记自由参数，缺省开）：
//!
//! - 再入：``|r| ≤ R⊕``（终端）；
//! - 撞月：``|r − r_☾| ≤ R☾``（终端；月面碰撞判据，Phase 3b 登记）；
//! - 逃逸：``|r| ≥ (r_H)^⊕``（终端）；
//! - 月 Hill 进入：``|r − r_☾| ≤ ρ_H`` 上行计数（非终端）；
//! - 全程最小地心/月心距逐步跟踪。
//!
//! 缺省 max_step = 6 小时：保证近点偶极（下潜-回升）不被步端点采样
//! 漏检（事件只在步端检测的固有约束），也是登记的定标参数。

use e2m2e_propagation::butcher::{explicit_rk_step, suggest_next_step};
use e2m2e_propagation::rk_methods::RkMethod;

/// 最小步长（s），防步长坍缩。
const MIN_STEP_S: f64 = 1e-6;

/// δ 模长重归一化界（与 megno.rs 一致）。
const DELTA_NORM_LO: f64 = 1e-100;
const DELTA_NORM_HI: f64 = 1e100;

/// 缺省最大步长（s，6 小时；近点漏检防护，见模块 docstring）。
pub const DEFAULT_MAX_STEP_S: f64 = 21_600.0;

/// 摄动天体（固定开普勒椭圆，地心）。
#[derive(Clone, Copy)]
pub struct KeplerBody {
    /// 天体 GM（km³/s²）。
    pub gm: f64,
    /// 半长轴 km / 偏心率 / 倾角 rad / 升交点 rad / 近点幅角 rad / 历元平近点角 rad。
    pub a_km: f64,
    pub ecc: f64,
    pub inc_rad: f64,
    pub raan_rad: f64,
    pub argp_rad: f64,
    pub m0_rad: f64,
}

impl KeplerBody {
    /// 历元后 t 秒的地心位置（开普勒方程牛顿解 + 旋转到参考惯性系）。
    pub fn position(&self, t_s: f64, earth_gm: f64) -> [f64; 3] {
        let n = (earth_gm / self.a_km.powi(3)).sqrt();
        let m = self.m0_rad + n * t_s;
        // 牛顿解 E（偏心率 < 0.1 量级，5 次足够；保守 8 次）。
        let mut e_anom = m;
        for _ in 0..8 {
            let residual = e_anom - self.ecc * e_anom.sin() - m;
            e_anom -= residual / (1.0 - self.ecc * e_anom.cos());
        }
        let r = self.a_km * (1.0 - self.ecc * e_anom.cos());
        let cos_f = (e_anom.cos() - self.ecc) / (1.0 - self.ecc * e_anom.cos());
        let sin_f =
            (1.0 - self.ecc * self.ecc).sqrt() * e_anom.sin() / (1.0 - self.ecc * e_anom.cos());
        // 近点焦坐标 → 参考系（Ω、i、ω 常规旋转）。
        let (mut x_p, mut y_p) = (r * cos_f, r * sin_f);
        if x_p.abs() > 1e300 || y_p.abs() > 1e300 {
            x_p = 0.0;
            y_p = 0.0;
        }
        let (so, co) = self.raan_rad.sin_cos();
        let (si, ci) = self.inc_rad.sin_cos();
        let (sw, cw) = self.argp_rad.sin_cos();
        let x = (co * cw - so * sw * ci) * x_p + (-co * sw - so * cw * ci) * y_p;
        let y = (so * cw + co * sw * ci) * x_p + (-so * sw + co * cw * ci) * y_p;
        let z = (sw * si) * x_p + (cw * si) * y_p;
        [x, y, z]
    }
}

/// 单格传播配置（整个网格共享，逐格只换初态）。
pub struct FateMapConfig {
    pub earth_gm: f64,
    pub moon: KeplerBody,
    /// EMS 模型加太阳点质量；None = EM 模型。
    pub sun: Option<KeplerBody>,
    pub earth_radius_km: f64,
    pub moon_radius_km: f64,
    /// 地球 Hill 界（逃逸判据）与月球 Hill 界（进入计数），km。
    pub hill_earth_km: f64,
    pub hill_moon_km: f64,
    /// 积分窗（s）。
    pub span_s: f64,
    /// 终端事件早停（缺省开，Phase 3c 登记自由参数）。
    pub stop_on_terminal: bool,
    /// 输出状态数（≥ 2 时含首末；诊断用）。
    pub n_out: usize,
}

/// 单格结果。
pub struct FateMapCellResult {
    pub states: Vec<[f64; 6]>,
    pub times: Vec<f64>,
    pub y: f64,
    pub ybar: f64,
    pub reentry: bool,
    pub t_reentry_s: Option<f64>,
    pub impact: bool,
    pub t_impact_s: Option<f64>,
    pub escaped: bool,
    pub t_escape_s: Option<f64>,
    pub min_r_geo_km: f64,
    pub min_r_sel_km: f64,
    pub moon_hill_entries: u32,
    /// 终端事件（0=再入、1=撞月、2=逃逸；None=走满窗）。
    pub terminal: Option<u8>,
    pub n_steps: usize,
    pub n_rejected: usize,
}

/// 三体位矢差的安全倒数立方（防除零）。
fn inv_cube(d2: f64) -> f64 {
    let d = d2.sqrt().max(1e-9);
    1.0 / (d * d * d)
}

/// 地心点质量加速度 + ∂a/∂r（3×3 行优先展平）。
#[allow(clippy::too_many_arguments)]
fn accel_and_jacobian(config: &FateMapConfig, r: &[f64; 3], t_s: f64) -> ([f64; 3], [f64; 9]) {
    let mut acc = [0.0_f64; 3];
    let mut jac = [0.0_f64; 9];

    // 中心引力：a = −μ⊕ r/r³，J = −μ⊕(I/r³ − 3rr^T/r⁵)。
    let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    let ir3 = inv_cube(r2);
    let ir5 = ir3 / r2.max(1e-18);
    for i in 0..3 {
        acc[i] -= config.earth_gm * r[i] * ir3;
        for j in 0..3 {
            let delta = if i == j { 1.0 } else { 0.0 };
            jac[i * 3 + j] -= config.earth_gm * (delta * ir3 - 3.0 * r[i] * r[j] * ir5);
        }
    }

    // 摄动天体（月，EMS 再加日）：直接项 + 间接项；J 只有直接项贡献。
    let mut bodies: Vec<[f64; 3]> = Vec::with_capacity(2);
    bodies.push(config.moon.position(t_s, config.earth_gm));
    if let Some(sun) = config.sun.as_ref() {
        bodies.push(sun.position(t_s, config.earth_gm));
    }
    let gms = [config.moon.gm, config.sun.map(|s| s.gm).unwrap_or(0.0)];
    for (rb, gm) in bodies.iter().zip(gms.iter()) {
        let d = [r[0] - rb[0], r[1] - rb[1], r[2] - rb[2]];
        let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        let id3 = inv_cube(d2);
        let id5 = id3 / d2.max(1e-18);
        let rb2 = rb[0] * rb[0] + rb[1] * rb[1] + rb[2] * rb[2];
        let ib3 = inv_cube(rb2);
        for i in 0..3 {
            acc[i] += gm * (-d[i] * id3 - rb[i] * ib3);
            for j in 0..3 {
                let delta = if i == j { 1.0 } else { 0.0 };
                jac[i * 3 + j] -= gm * (delta * id3 - 3.0 * d[i] * d[j] * id5);
            }
        }
    }
    (acc, jac)
}

/// 16 维增广右端项。
fn fate_map_eom(config: &FateMapConfig, t_s: f64, y: &[f64]) -> Vec<f64> {
    let r = [y[0], y[1], y[2]];
    let (acc, jac) = accel_and_jacobian(config, &r, t_s);
    let dr = [y[6], y[7], y[8]];
    let dv = [y[9], y[10], y[11]];

    let mut out = vec![0.0_f64; 16];
    out[0] = y[3];
    out[1] = y[4];
    out[2] = y[5];
    out[3] = acc[0];
    out[4] = acc[1];
    out[5] = acc[2];
    out[6] = dv[0];
    out[7] = dv[1];
    out[8] = dv[2];
    for i in 0..3 {
        out[9 + i] = jac[i * 3] * dr[0] + jac[i * 3 + 1] * dr[1] + jac[i * 3 + 2] * dr[2];
    }
    let delta_sq: f64 = y[6..12].iter().map(|v| v * v).sum();
    let delta_dot: f64 = (0..3)
        .map(|i| dr[i] * dv[i] + dv[i] * out[9 + i])
        .sum::<f64>();
    if delta_sq > 0.0 {
        out[12] = t_s * delta_dot / delta_sq;
        out[13] = if t_s > 0.0 { 2.0 * y[12] / t_s } else { 0.0 };
    }
    out
}

/// 单格事件检查（步端点值对），返回 (终端码, 插值时刻)。
#[allow(clippy::too_many_arguments)]
fn check_events(
    config: &FateMapConfig,
    t0: f64,
    r0: &[f64; 3],
    r_sel0: f64,
    t1: f64,
    r1: &[f64; 3],
    r_sel1: f64,
    state: &mut FateMapCellResult,
) -> Option<u8> {
    let g_re_0 = r0[0].hypot(r0[1].hypot(r0[2])) - config.earth_radius_km;
    let g_re_1 = r1[0].hypot(r1[1].hypot(r1[2])) - config.earth_radius_km;
    let g_im_0 = r_sel0 - config.moon_radius_km;
    let g_im_1 = r_sel1 - config.moon_radius_km;
    let g_es_0 = config.hill_earth_km - r0[0].hypot(r0[1].hypot(r0[2]));
    let g_es_1 = config.hill_earth_km - r1[0].hypot(r1[1].hypot(r1[2]));
    let g_hm_0 = config.hill_moon_km - r_sel0;
    let g_hm_1 = config.hill_moon_km - r_sel1;

    // 月 Hill 上行计数（g 由负转正 = 从界外进界内；起点在界内不计）。
    if g_hm_0 < 0.0 && g_hm_1 >= 0.0 {
        state.moon_hill_entries += 1;
    }

    let frac = |g0: f64, g1: f64| t0 + (t1 - t0) * (g0 / (g0 - g1));
    if g_re_0 > 0.0 && g_re_1 <= 0.0 {
        state.reentry = true;
        state.t_reentry_s = Some(frac(g_re_0, g_re_1));
        return Some(0);
    }
    if g_im_0 > 0.0 && g_im_1 <= 0.0 {
        state.impact = true;
        state.t_impact_s = Some(frac(g_im_0, g_im_1));
        return Some(1);
    }
    if g_es_0 > 0.0 && g_es_1 <= 0.0 {
        state.escaped = true;
        state.t_escape_s = Some(frac(g_es_0, g_es_1));
        return Some(2);
    }
    None
}

/// 单格传播主循环（PD78；误差控制只计前 6 维）。
pub fn propagate_geocentric_fate_map(
    config: &FateMapConfig,
    initial_state_km: &[f64; 6],
    rtol: f64,
    max_step: Option<f64>,
    max_steps: Option<usize>,
) -> Result<FateMapCellResult, String> {
    let method = RkMethod::Pd78;
    let tol = rtol;
    let h_max = max_step.unwrap_or(DEFAULT_MAX_STEP_S);
    let s_max = max_steps.unwrap_or(2_000_000);

    let mut y = vec![0.0_f64; 16];
    y[..6].copy_from_slice(initial_state_km);
    y[6] = 1.0; // 切向量初值：单位 x
    let mut t = 0.0_f64;

    let n_out = config.n_out.max(2);
    let out_dt = config.span_s / (n_out as f64 - 1.0);
    let mut next_out = out_dt; // 首个非零输出点（0 点已随初态入列）
    let mut out_idx = 1usize;

    let mut result = FateMapCellResult {
        states: Vec::with_capacity(n_out),
        times: Vec::with_capacity(n_out),
        y: 0.0,
        ybar: 0.0,
        reentry: false,
        t_reentry_s: None,
        impact: false,
        t_impact_s: None,
        escaped: false,
        t_escape_s: None,
        min_r_geo_km: f64::INFINITY,
        min_r_sel_km: f64::INFINITY,
        moon_hill_entries: 0,
        terminal: None,
        n_steps: 0,
        n_rejected: 0,
    };

    let mut h = h_max.min(config.span_s);
    let r_moon0 = config.moon.position(0.0, config.earth_gm);
    let mut r_prev = [
        initial_state_km[0],
        initial_state_km[1],
        initial_state_km[2],
    ];
    let mut r_sel_prev =
        (r_prev[0] - r_moon0[0]).hypot((r_prev[1] - r_moon0[1]).hypot(r_prev[2] - r_moon0[2]));
    result.min_r_geo_km = r_prev[0].hypot(r_prev[1].hypot(r_prev[2]));
    result.min_r_sel_km = r_sel_prev;
    result.states.push(*initial_state_km);
    result.times.push(0.0);

    while result.n_steps < s_max && t < config.span_s - 1e-9 {
        result.n_steps += 1;
        let h_try = h.min(config.span_s - t).min(h_max);
        if h_try.abs() < MIN_STEP_S {
            return Err(format!("step size collapsed below minimum at t={} s", t));
        }
        let callback =
            |ti: f64, yi: &[f64]| -> Result<Vec<f64>, String> { Ok(fate_map_eom(config, ti, yi)) };
        let (y_new, error) = explicit_rk_step(method.table(), t, &y, h_try, callback, Some(6))
            .map_err(|e| format!("RK step error at t={}: {}", t, e))?;

        if error <= tol {
            let t_new = t + h_try;
            let y_acc = y_new;
            let r_new = [y_acc[0], y_acc[1], y_acc[2]];
            let r_moon_new = config.moon.position(t_new, config.earth_gm);
            let r_sel_new = (r_new[0] - r_moon_new[0])
                .hypot((r_new[1] - r_moon_new[1]).hypot(r_new[2] - r_moon_new[2]));
            result.min_r_geo_km = result
                .min_r_geo_km
                .min(r_new[0].hypot(r_new[1].hypot(r_new[2])));
            result.min_r_sel_km = result.min_r_sel_km.min(r_sel_new);

            if let Some(code) = check_events(
                config,
                t,
                &r_prev,
                r_sel_prev,
                t_new,
                &r_new,
                r_sel_new,
                &mut result,
            ) {
                result.terminal = Some(code);
                result
                    .states
                    .push([y_acc[0], y_acc[1], y_acc[2], y_acc[3], y_acc[4], y_acc[5]]);
                result.times.push(t_new);
                result.y = if t_new > 0.0 {
                    2.0 * y_acc[12] / t_new
                } else {
                    0.0
                };
                result.ybar = if t_new > 0.0 { y_acc[13] / t_new } else { 0.0 };
                if config.stop_on_terminal {
                    return Ok(result);
                }
            }
            r_prev = r_new;
            r_sel_prev = r_sel_new;
            t = t_new;
            y = y_acc;
            let norm: f64 = y[6..12].iter().map(|v| v * v).sum::<f64>().sqrt();
            if !(DELTA_NORM_LO..=DELTA_NORM_HI).contains(&norm) && norm > 0.0 && norm.is_finite() {
                for v in y[6..12].iter_mut() {
                    *v /= norm;
                }
            }
            while out_idx < n_out && t >= next_out - 1e-9 {
                result.states.push([y[0], y[1], y[2], y[3], y[4], y[5]]);
                result.times.push(next_out);
                out_idx += 1;
                next_out += out_dt;
            }
            h = suggest_next_step(h_try, error, tol, method.embedded_order());
        } else {
            result.n_rejected += 1;
            h = suggest_next_step(h_try, error, tol, method.embedded_order());
        }
    }

    result.y = if t > 0.0 { 2.0 * y[12] / t } else { 0.0 };
    result.ybar = if t > 0.0 { y[13] / t } else { 0.0 };
    if result
        .states
        .last()
        .map(|s| s[..3] != y[..3])
        .unwrap_or(true)
    {
        result.states.push([y[0], y[1], y[2], y[3], y[4], y[5]]);
        result.times.push(t);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn em_config(span_s: f64) -> FateMapConfig {
        let moon = KeplerBody {
            gm: 4902.800066163796,
            a_km: 383_397.7725,
            ecc: 0.055545526,
            inc_rad: 0.0900617668,
            raan_rad: 311.07_f64.to_radians(),
            argp_rad: 175.84_f64.to_radians(),
            m0_rad: 0.0,
        };
        FateMapConfig {
            earth_gm: 398_600.4354360959,
            moon,
            sun: None,
            earth_radius_km: 6378.1363,
            moon_radius_km: 1737.4,
            hill_earth_km: 1_496_532.0,
            hill_moon_km: 61_364.0,
            span_s,
            stop_on_terminal: true,
            n_out: 8,
        }
    }

    fn circular_state(a_km: f64, inc_rad: f64, raan_rad: f64, u_rad: f64, gm: f64) -> [f64; 6] {
        let v = (gm / a_km).sqrt();
        let (su, cu) = u_rad.sin_cos();
        let (si, ci) = inc_rad.sin_cos();
        let (so, co) = raan_rad.sin_cos();
        let r_orb = [cu, su * ci, su * si];
        let v_orb = [-su, cu * ci, cu * si];
        let rot = |w: [f64; 3]| [co * w[0] - so * w[1], so * w[0] + co * w[1], w[2]];
        let r = rot(r_orb);
        let vv = rot(v_orb);
        [
            a_km * r[0],
            a_km * r[1],
            a_km * r[2],
            v * vv[0],
            v * vv[1],
            v * vv[2],
        ]
    }

    #[test]
    fn regular_circular_orbit_gives_ybar_two() {
        // Ȳ 收敛速率 ~ 振荡幅度/t：约 50 圈后进入 ±0.05 带内。
        let config = em_config(150.0 * 86_400.0);
        // 低倾圆轨 a = 0.2 a☾（SC 区内、远离月球）：Ȳ → 2（论文 line 1419）。
        let state = circular_state(
            0.2 * 383_397.7725,
            0.09,
            311.07_f64.to_radians(),
            0.0,
            config.earth_gm,
        );
        let result = propagate_geocentric_fate_map(&config, &state, 1e-10, None, None).unwrap();
        assert!((result.ybar - 2.0).abs() < 0.05, "ybar = {}", result.ybar);
        assert!(result.terminal.is_none());
        assert!(result.min_r_sel_km > 200_000.0);
    }

    #[test]
    fn reentry_orbit_is_terminal() {
        let config = em_config(30.0 * 86_400.0);
        // 掠地椭圆：a = 0.4 a☾、e = 0.9 → 近点 ≈ 0.04 a☾ = 15,300 km > R⊕；
        // e = 0.985 → 近点 2,290 km < R⊕（再入）。
        let a = 0.4 * 383_397.7725;
        let e = 0.985;
        let rp = a * (1.0 - e);
        let vp = ((2.0 / rp - 1.0 / a) * config.earth_gm).sqrt();
        let state = [rp, 0.0, 0.0, 0.0, vp, 0.0];
        let result = propagate_geocentric_fate_map(&config, &state, 1e-10, None, None).unwrap();
        assert_eq!(result.terminal, Some(0));
        assert!(result.reentry);
        assert!(result.times.last().unwrap() < &config.span_s);
    }

    #[test]
    fn ems_model_with_sun_runs_and_perturbs() {
        let mut config = em_config(5.0 * 86_400.0);
        config.sun = Some(KeplerBody {
            gm: 132_712_440_041.9393,
            a_km: 1.495978707e8,
            ecc: 0.0167086342,
            inc_rad: 0.0,
            raan_rad: 0.0,
            argp_rad: 1.7967,
            m0_rad: 0.0,
        });
        let state = circular_state(
            0.5 * 383_397.7725,
            0.09,
            311.07_f64.to_radians(),
            0.0,
            config.earth_gm,
        );
        let result = propagate_geocentric_fate_map(&config, &state, 1e-10, None, None).unwrap();
        assert!(result.terminal.is_none());
        assert!(result.ybar.is_finite());
        assert!(result.min_r_sel_km > 100_000.0);
    }

    #[test]
    fn kepler_body_position_round_trip_pericenter() {
        let body = KeplerBody {
            gm: 4902.8,
            a_km: 383_397.7725,
            ecc: 0.0555,
            inc_rad: 0.0,
            raan_rad: 0.0,
            argp_rad: 0.0,
            m0_rad: 0.0,
        };
        let p = body.position(0.0, 398_600.435);
        let r = p[0].hypot(p[1].hypot(p[2]));
        assert!((r - 383_397.7725 * (1.0 - 0.0555)).abs() < 1e-6);
    }
}
