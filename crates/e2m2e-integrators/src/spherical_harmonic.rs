//! 球谐引力加速度（1:1 移植自 Python ``gravity_field.py:_compute_acceleration_in_input_frame``）。
//!
//! 满配直推的 32% 耗时来自 Python 版的三层嵌套循环（Legendre 递推 + 加速度累加）。
//! 本模块用纯 Rust 标量循环重写同一算法，由 Python 侧经 PyO3 单次调用，
//! 消除逐元素 Python 开销。算法（球坐标分量法，physics convention 的完全正规化
//! associated Legendre）与 Python 版逐字一致，精度回归 < 1e-12。
//!
//! 输入位置 ``r`` 与输出加速度均在球谐展开坐标系（body-fixed，如 ITRF93 / MOON_PA）
//! 下；坐标变换仍由 Python 侧的 ``coordinate_system`` 完成，本函数不碰 SPICE。

/// 球谐引力加速度（body-fixed 系）。
///
/// 与 ``gravity_field.py:_compute_acceleration_in_input_frame`` 逐字对应：
/// 1. 球坐标 (r, s=sinφ, u=cosφ, lon)
/// 2. 完全正规化 associated Legendre 递推（vertical m=0 / 对角 / 次对角）
/// 3. 纬度偏导 dP
/// 4. 三轴加速度分量 dUdr / dUdφ / dUdλ 累加
/// 5. 球坐标 → 笛卡尔
///
/// # 参数
/// - `r`：位置 (3,)，body-fixed 系
/// - `c_flat`：球谐系数 C，行优先扁平化的 (degree+1)*(degree+1) 数组
/// - `s_flat`：球谐系数 S，同上
/// - `mu`：中心天体引力参数 (km³/s²)
/// - `radius`：参考半径 (km)
/// - `degree`：最大阶 n
/// - `order`：最大次 m（≤ degree）
pub fn spherical_harmonic_accel(
    r: &[f64],
    c_flat: &[f64],
    s_flat: &[f64],
    mu: f64,
    radius: f64,
    degree: usize,
    order: usize,
) -> Vec<f64> {
    debug_assert_eq!(r.len(), 3);
    let nn = degree + 1;
    debug_assert_eq!(c_flat.len(), nn * nn);
    debug_assert_eq!(s_flat.len(), nn * nn);
    debug_assert!(order <= degree);

    let x = r[0];
    let y = r[1];
    let z = r[2];
    let r_norm = (x * x + y * y + z * z).sqrt();
    if r_norm == 0.0 {
        return vec![f64::NAN, f64::NAN, f64::NAN];
    }

    let rho = radius / r_norm;
    let s = z / r_norm; // sin(phi)
    let u = (1.0 - s * s).max(0.0).sqrt(); // cos(phi)
    let lon = y.atan2(x);

    // ── associated Legendre functions (fully normalized, physics convention) ──
    // P[n*nn + m]，与 Python P[n,m] 同布局。
    let mut p = vec![0.0_f64; nn * nn];
    p[0] = 1.0;
    if degree >= 1 {
        p[1 * nn + 0] = 3.0_f64.sqrt() * s;
        p[1 * nn + 1] = -3.0_f64.sqrt() * u;
    }
    for n in 2..=degree {
        // vertical recurrence m=0
        let alpha = ((2 * n + 1) as f64 * (2 * n - 1) as f64 / (n * n) as f64).sqrt();
        let beta = ((2 * n + 1) as f64 * (n - 1) as f64 * (n - 1) as f64
            / (n * n * (2 * n - 3)) as f64)
            .sqrt();
        p[n * nn + 0] = alpha * s * p[(n - 1) * nn + 0] - beta * p[(n - 2) * nn + 0];
    }
    for m in 1..=degree {
        // sub-diagonal P[m+1, m]
        if m + 1 <= degree {
            p[(m + 1) * nn + m] = s * ((2 * m + 3) as f64).sqrt() * p[m * nn + m];
        }
        for n in (m + 2)..=degree {
            let alpha = ((2 * n + 1) as f64 * (2 * n - 1) as f64
                / ((n + m) * (n - m)) as f64)
                .sqrt();
            let beta = ((2 * n + 1) as f64 * (n + m - 1) as f64 * (n - m - 1) as f64
                / ((n + m) * (n - m) * (2 * n - 3)) as f64)
                .sqrt();
            p[n * nn + m] = alpha * s * p[(n - 1) * nn + m] - beta * p[(n - 2) * nn + m];
        }
        // next diagonal P[m+1, m+1]
        if m + 1 <= degree {
            p[(m + 1) * nn + (m + 1)] =
                -u * ((2 * m + 3) as f64 / (2 * m + 2) as f64).sqrt() * p[m * nn + m];
        }
    }

    // ── 纬度偏导 dP ──
    let mut dp = vec![0.0_f64; nn * nn];
    for n in 1..=degree {
        let coef = ((n * (n + 1)) as f64 / 2.0_f64).sqrt();
        dp[n * nn + 0] = -coef * p[n * nn + 1];
    }
    for m in 1..=degree {
        for n in (m + 1)..=degree {
            let term1 = (((n + m) * (n - m + 1)) as f64).sqrt() * p[n * nn + (m - 1)];
            let term2 = if m + 1 <= degree {
                (((n + m + 1) * (n - m)) as f64).sqrt() * p[n * nn + (m + 1)]
            } else {
                0.0
            };
            dp[n * nn + m] = 0.5 * (term1 - term2);
        }
    }

    // ── 加速度分量累加 ──
    let mut dudr = 0.0_f64;
    let mut dudphi = 0.0_f64;
    let mut dudlambda = 0.0_f64;
    for n in 0..=degree {
        let rho_n = rho.powi(n as i32);
        let m_hi = if n < order { n } else { order };
        for m in 0..=m_hi {
            let c_val = c_flat[n * nn + m];
            let s_val = s_flat[n * nn + m];
            if c_val == 0.0 && s_val == 0.0 {
                continue;
            }
            let cm = (m as f64 * lon).cos();
            let sm = (m as f64 * lon).sin();
            let cs = c_val * cm + s_val * sm;
            dudr += rho_n * (n as f64 + 1.0) * p[n * nn + m] * cs;
            dudphi += rho_n * dp[n * nn + m] * cs;
            dudlambda += rho_n * m as f64 * p[n * nn + m] * (-c_val * sm + s_val * cm);
        }
    }

    let dudr = -mu / (r_norm * r_norm) * dudr;
    let dudphi = mu / r_norm * dudphi;
    let dudlambda = mu / r_norm * dudlambda;

    // ── 球坐标 → 笛卡尔 (body-fixed) ──
    let cos_lon = lon.cos();
    let sin_lon = lon.sin();
    let cos_phi = u;
    let sin_phi = s;

    let a_r = dudr;
    let a_phi = dudphi / r_norm;
    let a_lambda = dudlambda / (r_norm * cos_phi);

    let ax = a_r * cos_phi * cos_lon - a_phi * sin_phi * cos_lon - a_lambda * sin_lon;
    let ay = a_r * cos_phi * sin_lon - a_phi * sin_phi * sin_lon + a_lambda * cos_lon;
    let az = a_r * sin_phi + a_phi * cos_phi;

    vec![ax, ay, az]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 中心质量（degree=0）应等价于点质量：a = -μ r / |r|³。
    #[test]
    fn test_central_term_matches_point_mass() {
        let r = [70000.0_f64, 0.0, 0.0];
        // degree=0 → nn=1，C/S 各 1 个元素。C[0,0]=1（归一化的中心项）。
        let c = vec![1.0];
        let s = vec![0.0];
        let mu = 4902.8; // 月球 GM
        let radius = 1737.1;
        let a = spherical_harmonic_accel(&r, &c, &s, mu, radius, 0, 0);
        let point = -mu / (r[0] * r[0]); // -μ/r²
        assert!((a[0] - point).abs() < 1e-9, "a_x={} expected={}", a[0], point);
        assert!(a[1].abs() < 1e-9);
        assert!(a[2].abs() < 1e-9);
    }

    /// 输出维度恒为 3。
    #[test]
    fn test_output_dim() {
        let r = [1e5, 2e5, 3e5];
        let n = 2;
        let nn = n + 1;
        let c = vec![1.0];
        let _ = c;
        let c = vec![1.0_f64; nn * nn];
        let s = vec![0.0_f64; nn * nn];
        let a = spherical_harmonic_accel(&r, &c, &s, 4902.8, 1737.1, n, n);
        assert_eq!(a.len(), 3);
        assert!(a.iter().all(|v| v.is_finite()));
    }
}
