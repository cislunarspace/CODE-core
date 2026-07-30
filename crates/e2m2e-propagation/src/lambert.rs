//! 二体 Lambert 求解器：Izzo (2015) 算法。
//!
//! 无量纲 x 域时间方程 + 三阶 Householder 迭代；初值按 Izzo/pykep 的
//! 分段构造，多圈最小时间用 Halley 迭代定位。蓝本：Izzo 2015
//! 《Revisiting Lambert's Problem》与 jacobwilliams 的 Fortran
//! `lambert_module.f90`（pykep 移植）。
//!
//! 多圈（revs ≥ 1）时每个 tof 存在左右两个分支解，本模块返回右分支
//! （低能解，x > x_min）。

use std::f64::consts::PI;

/// 转移方向：短程（转移角 < π）或长程（> π）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferDirection {
    ShortWay,
    LongWay,
}

/// Householder 收敛容差与最大迭代数（取自 pykep / Fortran 移植）。
const TOL_SINGLE_REV: f64 = 1e-5;
const TOL_MULTI_REV: f64 = 1e-8;
const MAX_HOUSEHOLDER_ITERS: u32 = 15;
/// Halley 迭代（多圈最小时间定位）容差与上限。
const TOL_HALLEY: f64 = 1e-13;
const MAX_HALLEY_ITERS: u32 = 12;

/// 无量纲几何参数 λ 及其幂次。
struct Geometry {
    lambda: f64,
    lambda2: f64,
    lambda3: f64,
    lambda5: f64,
}

fn norm(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// 高斯超几何函数 F(3, 1; 5/2; x)，Battin 级数用。
fn hypergeo(x: f64) -> f64 {
    let mut f = 1.0;
    let mut term = 1.0;
    for i in 0..10000u32 {
        let i = i as f64;
        term = term * (3.0 + i) * (1.0 + i) / (2.5 + i) * x / (i + 1.0);
        f += term;
        if term.abs() <= 1e-11 {
            break;
        }
    }
    f
}

/// 无量纲时间方程 T(x)，按 |x − 1| 分 Lancaster / Lagrange / Battin 三段。
fn compute_tof(g: &Geometry, x: f64, n: u32) -> f64 {
    let dist = (x - 1.0).abs();
    if dist < 0.2 && dist > 0.01 {
        // Lagrange 表达式，Izzo 2015 Eqn. 9
        let a = 1.0 / (1.0 - x * x);
        if a > 0.0 {
            let alpha = 2.0 * x.acos();
            let mut beta = 2.0 * (g.lambda2 / a).sqrt().asin();
            if g.lambda < 0.0 {
                beta = -beta;
            }
            (a * a.sqrt() * ((alpha - alpha.sin()) - (beta - beta.sin()) + 2.0 * PI * n as f64))
                / 2.0
        } else {
            let alpha = 2.0 * x.acosh();
            let mut beta = 2.0 * (-g.lambda2 / a).sqrt().asinh();
            if g.lambda < 0.0 {
                beta = -beta;
            }
            (-a * (-a).sqrt() * ((beta - beta.sinh()) - (alpha - alpha.sinh()))) / 2.0
        }
    } else {
        let e = x * x - 1.0;
        let rho = e.abs();
        let z = (1.0 + g.lambda2 * e).sqrt();
        if dist < 0.01 {
            // Battin 级数（近抛物线），Izzo 2015 Eqn. 20
            let eta = z - g.lambda * x;
            let s1 = (1.0 - g.lambda - x * eta) / 2.0;
            let q = 4.0 / 3.0 * hypergeo(s1);
            (eta * eta * eta * q + 4.0 * g.lambda * eta) / 2.0 + n as f64 * PI / rho.powf(1.5)
        } else {
            // Lancaster-Blanchard 表达式
            let y = rho.sqrt();
            let gg = x * z - g.lambda * e;
            let d = if e < 0.0 {
                n as f64 * PI + gg.clamp(-1.0, 1.0).acos()
            } else {
                (y * (z - g.lambda * x) + gg).ln()
            };
            (x - g.lambda * z - d / y) / e
        }
    }
}

/// T(x) 的一至三阶导数（Izzo 2015 Eqn. 22）。
fn dtdx(g: &Geometry, x: f64, t: f64) -> (f64, f64, f64) {
    let umx2 = 1.0 - x * x;
    let inv = 1.0 / umx2;
    let y = (1.0 - g.lambda2 * umx2).sqrt();
    let y2 = y * y;
    let y3 = y2 * y;
    let y5 = y3 * y2;
    let dt = inv * (3.0 * t * x - 2.0 + 2.0 * g.lambda3 * x / y);
    let d2t = inv * (3.0 * t + 5.0 * x * dt + 2.0 * (1.0 - g.lambda2) * g.lambda3 / y3);
    let d3t = inv * (7.0 * x * d2t + 8.0 * dt - 6.0 * (1.0 - g.lambda2) * g.lambda5 * x / y5);
    (dt, d2t, d3t)
}

/// 三阶 Householder 迭代解 T(x) = t。收敛返回 (x, 迭代数)，否则 None。
fn householder(g: &Geometry, t: f64, mut x: f64, n: u32, eps: f64) -> Option<(f64, u32)> {
    for it in 1..=MAX_HOUSEHOLDER_ITERS {
        let tof = compute_tof(g, x, n);
        let (dt, d2t, d3t) = dtdx(g, x, tof);
        let delta = tof - t;
        let dt2 = dt * dt;
        let term = delta * (dt2 - delta * d2t / 2.0)
            / (dt * (dt2 - delta * d2t) + d3t * delta * delta / 6.0);
        x -= term;
        if term.abs() <= eps {
            return Some((x, it));
        }
    }
    None
}

/// 多圈转移的最小无量纲时间 T_min：Halley 迭代定位 dT/dx = 0。
fn minimum_tof(g: &Geometry, n: u32) -> f64 {
    let mut x = 0.0;
    let mut t_min = compute_tof(g, 0.0, n); // = acos(λ) + λ√(1−λ²) + nπ
    for _ in 0..MAX_HALLEY_ITERS {
        let (dt, d2t, d3t) = dtdx(g, x, t_min);
        let x_new = if dt != 0.0 {
            x - dt * d2t / (d2t * d2t - dt * d3t / 2.0)
        } else {
            x
        };
        let converged = (x - x_new).abs() < TOL_HALLEY;
        x = x_new;
        if converged {
            break;
        }
        t_min = compute_tof(g, x, n);
    }
    t_min
}

/// 单次 Lambert 求解的返回类型：(v0, vf, n_iter) 或错误信息。
pub type LambertResult = Result<([f64; 3], [f64; 3], u32), String>;

/// Izzo 算法解 Lambert 问题。
///
/// `r0`/`rf` 为出发/到达位置（km），`tof` 飞行时间（s），`mu` 中心天体
/// GM（km³/s²），`revs` 完整圈数。返回 (v0, vf, n_iter)，速度单位 km/s；
/// tof 低于 `revs` 圈最小转移时间或迭代不收敛时返回 Err。
pub fn lambert_izzo(
    r0: &[f64; 3],
    rf: &[f64; 3],
    tof: f64,
    mu: f64,
    direction: TransferDirection,
    revs: u32,
) -> LambertResult {
    let r1 = norm(r0);
    let r2 = norm(rf);
    if tof <= 0.0 || mu <= 0.0 || r1 == 0.0 || r2 == 0.0 {
        return Err("lambert_izzo: tof/mu 必须为正，r0/rf 必须非零".to_string());
    }
    let c = [rf[0] - r0[0], rf[1] - r0[1], rf[2] - r0[2]];
    let cmag = norm(&c);
    if cmag == 0.0 {
        return Err("lambert_izzo: r0 与 rf 重合，弦长为零".to_string());
    }

    let s = (cmag + r1 + r2) / 2.0;
    let t = (2.0 * mu / (s * s * s)).sqrt() * tof;

    let r1_hat = [r0[0] / r1, r0[1] / r1, r0[2] / r1];
    let r2_hat = [rf[0] / r2, rf[1] / r2, rf[2] / r2];
    let h = cross(&r1_hat, &r2_hat);
    // π 转移时转移面不定，任选一个法向（Fortran 移植同款处理）
    let h_hat = if norm(&h) == 0.0 {
        [0.0, 0.0, 1.0]
    } else {
        let n = norm(&h);
        [h[0] / n, h[1] / n, h[2] / n]
    };
    let mut it1 = cross(&h_hat, &r1_hat);
    let mut it2 = cross(&h_hat, &r2_hat);

    let lambda2 = 1.0 - cmag / s;
    let mut lambda = lambda2.sqrt();
    if direction == TransferDirection::LongWay {
        lambda = -lambda;
        for i in 0..3 {
            it1[i] = -it1[i];
            it2[i] = -it2[i];
        }
    }
    let g = Geometry {
        lambda,
        lambda2,
        lambda3: lambda * lambda2,
        lambda5: lambda2 * lambda2 * lambda,
    };

    let (x0, tol) = if revs == 0 {
        let t00 = g.lambda.acos() + g.lambda * (1.0 - g.lambda2).sqrt();
        let t1 = 2.0 / 3.0 * (1.0 - g.lambda3);
        let x0 = if t >= t00 {
            -(t - t00) / (t - t00 + 4.0)
        } else if t <= t1 {
            2.5 * (t1 * (t1 - t)) / (t * (1.0 - g.lambda5)) + 1.0
        } else {
            (t / t00).powf(std::f64::consts::LN_2 / (t1 / t00).ln()) - 1.0
        };
        (x0, TOL_SINGLE_REV)
    } else {
        // 多圈：先确认 tof 高于该圈数的最小转移时间
        if t < compute_tof(&g, 0.0, revs) && t < minimum_tof(&g, revs) {
            return Err(format!(
                "lambert_izzo: tof 低于 {revs} 圈转移的最小时间，无解"
            ));
        }
        // 右分支（低能解）初值，Izzo 2015 Eqn. 31
        let term = (8.0 * t / (revs as f64 * PI)).powf(2.0 / 3.0);
        ((term - 1.0) / (term + 1.0), TOL_MULTI_REV)
    };

    let (x, n_iter) = householder(&g, t, x0, revs, tol)
        .ok_or_else(|| "lambert_izzo: Householder 迭代不收敛".to_string())?;

    // 速度重构（Battin 书公式，Izzo 2015 同款）
    let gamma = (mu * s / 2.0).sqrt();
    let rho = (r1 - r2) / cmag;
    let sigma = (1.0 - rho * rho).sqrt();
    let y = (1.0 - g.lambda2 + g.lambda2 * x * x).sqrt();
    let ly = g.lambda * y;
    let vr1 = gamma * ((ly - x) - rho * (ly + x)) / r1;
    let vr2 = -gamma * ((ly - x) + rho * (ly + x)) / r2;
    let vt = gamma * sigma * (y + g.lambda * x);
    let vt1 = vt / r1;
    let vt2 = vt / r2;

    let mut v0 = [0.0; 3];
    let mut vf = [0.0; 3];
    for i in 0..3 {
        v0[i] = vr1 * r1_hat[i] + vt1 * it1[i];
        vf[i] = vr2 * r2_hat[i] + vt2 * it2[i];
    }
    Ok((v0, vf, n_iter))
}

/// N×M 网格批量求解（porkchop 用）：每个几何对每个 tof 各解一次。
///
/// 返回长度 `geometries.len() * tofs.len()`，行优先（几何在外，tof 在内）；
/// 单个组合失败不中断，以 Err 占位。
pub fn lambert_batch(
    geometries: &[([f64; 3], [f64; 3])],
    tofs: &[f64],
    mu: f64,
    direction: TransferDirection,
    revs: u32,
) -> Vec<LambertResult> {
    let mut out = Vec::with_capacity(geometries.len() * tofs.len());
    for (r0, rf) in geometries {
        for &tof in tofs {
            out.push(lambert_izzo(r0, rf, tof, mu, direction, revs));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const MU: f64 = 398600.4418;

    /// 椭圆二体解析传播（Kepler 方程 + f/g），独立于 Lambert 求解器，
    /// 用于交叉验证：Lambert 解出的 v0 传播 tof 后应回到 rf。
    fn propagate_ellipse(r0: &[f64; 3], v0: &[f64; 3], tof: f64, mu: f64) -> [f64; 3] {
        let r0n = norm(r0);
        let v2 = v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2];
        let a = 1.0 / (2.0 / r0n - v2 / mu);
        assert!(a > 0.0, "测试用例应为椭圆");
        let rdotv = r0[0] * v0[0] + r0[1] * v0[1] + r0[2] * v0[2];
        // 偏心率向量
        let ev = [
            (v2 - mu / r0n) * r0[0] / mu - rdotv * v0[0] / mu,
            (v2 - mu / r0n) * r0[1] / mu - rdotv * v0[1] / mu,
            (v2 - mu / r0n) * r0[2] / mu - rdotv * v0[2] / mu,
        ];
        let e = norm(&ev);
        let cos_e0 = (1.0 - r0n / a) / e;
        let sin_e0 = rdotv / (e * (mu * a).sqrt());
        let e0 = sin_e0.atan2(cos_e0);
        let n = (mu / (a * a * a)).sqrt();
        let m0 = e0 - e * e0.sin();
        let m = m0 + n * tof;
        // Newton 解 Kepler 方程
        let mut ecc_anom = m;
        for _ in 0..50 {
            let f = ecc_anom - e * ecc_anom.sin() - m;
            let fp = 1.0 - e * ecc_anom.cos();
            let step = f / fp;
            ecc_anom -= step;
            if step.abs() < 1e-14 {
                break;
            }
        }
        let d_e = ecc_anom - e0;
        let f = 1.0 + a * (d_e.cos() - 1.0) / r0n;
        let gg = tof - (d_e - d_e.sin()) / n;
        [
            f * r0[0] + gg * v0[0],
            f * r0[1] + gg * v0[1],
            f * r0[2] + gg * v0[2],
        ]
    }

    fn assert_vec_close(actual: &[f64; 3], expected: &[f64; 3], abs_tol: f64) {
        for i in 0..3 {
            assert!(
                (actual[i] - expected[i]).abs() < abs_tol,
                "分量 {i}: actual={} expected={}",
                actual[i],
                expected[i]
            );
        }
    }

    /// Lambert 解出 v0 后解析传播 tof，落点与 rf 的相对误差应 < 1e-8。
    fn assert_reaches(r0: &[f64; 3], rf: &[f64; 3], tof: f64, v0: &[f64; 3]) {
        let r_end = propagate_ellipse(r0, v0, tof, MU);
        for i in 0..3 {
            assert!(
                (r_end[i] - rf[i]).abs() < 1e-8 * norm(rf),
                "传播落点分量 {i}: {} vs {}",
                r_end[i],
                rf[i]
            );
        }
    }

    /// Vallado 经典算例（poliastro / Orekit 回归基准）：短程解。
    #[test]
    fn vallado_short_way_benchmark() {
        let r0 = [5000.0, 10000.0, 2100.0];
        let rf = [-14600.0, 2500.0, 7000.0];
        let tof = 3600.0;
        let (v0, vf, n_iter) =
            lambert_izzo(&r0, &rf, tof, MU, TransferDirection::ShortWay, 0).unwrap();
        let exp_v0 = [-5.99249503, 1.92536671, 3.24563805];
        let exp_vf = [-3.31245851, -4.19661901, -0.38529044];
        // 文献值只有 8 位有效数字，容差取 2e-6 km/s
        assert_vec_close(&v0, &exp_v0, 2e-6);
        assert_vec_close(&vf, &exp_vf, 2e-6);
        assert!(n_iter <= 5, "Householder 迭代次数异常: {n_iter}");
        assert_reaches(&r0, &rf, tof, &v0);
    }

    /// 任务指定算例：r0=[15945.34,0,0], rf=[12214.83,10249.46,0], tof=76 min。
    #[test]
    fn vallado_76min_short_way() {
        let r0 = [15945.34, 0.0, 0.0];
        let rf = [12214.83899, 10249.46731, 0.0];
        let tof = 76.0 * 60.0;
        let (v0, _, _) = lambert_izzo(&r0, &rf, tof, MU, TransferDirection::ShortWay, 0).unwrap();
        // 文献基准值（Vallado lambert 测试用例，6 位有效数字）
        assert_vec_close(&v0, &[2.058913, 2.915965, 0.0], 1e-5);
        assert_reaches(&r0, &rf, tof, &v0);
    }

    /// 长程解：无现成文献值，用解析传播交叉验证。
    #[test]
    fn vallado_long_way_consistency() {
        let r0 = [5000.0, 10000.0, 2100.0];
        let rf = [-14600.0, 2500.0, 7000.0];
        let tof = 3600.0 * 4.0;
        let (v0, _, _) = lambert_izzo(&r0, &rf, tof, MU, TransferDirection::LongWay, 0).unwrap();
        assert_reaches(&r0, &rf, tof, &v0);
        // 长程解角动量方向应与短程解相反
        let (v0_short, _, _) =
            lambert_izzo(&r0, &rf, tof, MU, TransferDirection::ShortWay, 0).unwrap();
        let h_long = cross(&r0, &v0);
        let h_short = cross(&r0, &v0_short);
        assert!(h_long[2] * h_short[2] < 0.0);
    }

    /// 多圈 revs = 1, 2：右分支低能解，解析传播交叉验证。
    #[test]
    fn multi_rev_consistency() {
        let r0 = [5000.0, 10000.0, 2100.0];
        let rf = [-14600.0, 2500.0, 7000.0];
        let tof = 3600.0 * 30.0;
        for revs in 1..=2 {
            let (v0, _, _) =
                lambert_izzo(&r0, &rf, tof, MU, TransferDirection::ShortWay, revs).unwrap();
            assert!(v0.iter().all(|v| v.is_finite()));
            assert_reaches(&r0, &rf, tof, &v0);
        }
    }

    /// tof 低于该圈数最小转移时间时应报错而非给出伪解。
    #[test]
    fn multi_rev_below_tmin_errors() {
        let r0 = [5000.0, 10000.0, 2100.0];
        let rf = [-14600.0, 2500.0, 7000.0];
        let result = lambert_izzo(&r0, &rf, 3600.0, MU, TransferDirection::ShortWay, 1);
        assert!(result.is_err());
    }

    /// 近 180° 退化几何：不收敛失败、不发 NaN。
    #[test]
    fn near_pi_transfer_no_nan() {
        let r0 = [10000.0, 0.0, 0.0];
        let rf = [-9999.9, 100.0, 0.0];
        let tof = 5400.0;
        let (v0, vf, _) = lambert_izzo(&r0, &rf, tof, MU, TransferDirection::ShortWay, 0).unwrap();
        assert!(v0.iter().chain(vf.iter()).all(|v| v.is_finite()));
        assert_reaches(&r0, &rf, tof, &v0);
    }

    /// 恰好 180°（叉积为零，转移面不定）：任选一个面，结果仍须有限。
    #[test]
    fn exact_pi_transfer_no_nan() {
        let r0 = [10000.0, 0.0, 0.0];
        let rf = [-10000.0, 0.0, 0.0];
        let tof = 5400.0;
        let (v0, vf, _) = lambert_izzo(&r0, &rf, tof, MU, TransferDirection::ShortWay, 0).unwrap();
        assert!(v0.iter().chain(vf.iter()).all(|v| v.is_finite()));
    }

    /// 批量接口与逐条求解一致；失败组合以 Err 占位。
    #[test]
    fn batch_matches_single() {
        let r0 = [5000.0, 10000.0, 2100.0];
        let rf = [-14600.0, 2500.0, 7000.0];
        let geoms = [(r0, rf), (r0, r0)]; // 第二组弦长为零，必失败
        let tofs = [3600.0, 7200.0];
        let results = lambert_batch(&geoms, &tofs, MU, TransferDirection::ShortWay, 0);
        assert_eq!(results.len(), 4);
        for (j, &tof) in tofs.iter().enumerate() {
            let (v0, vf, _) =
                lambert_izzo(&r0, &rf, tof, MU, TransferDirection::ShortWay, 0).unwrap();
            let (b0, bf, _) = results[j].as_ref().unwrap();
            assert_vec_close(b0, &v0, 0.0 + 1e-12);
            assert_vec_close(bf, &vf, 1e-12);
        }
        assert!(results[2].is_err());
        assert!(results[3].is_err());
    }
}
