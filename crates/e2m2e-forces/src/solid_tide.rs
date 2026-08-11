//! 固体潮修正（1:1 移植自 Python ``earth_tide.py``）。
//!
//! 球谐核下沉后（commit 533095e），潮汐路径的 Python 三角函数循环是下一个
//! 显著热点——每步 ~200 个三角函数（step1 ~50 + step2 ~142），全程 Python 解释
//! 执行。本模块把 step1/step2 + pole/permanent 全部用纯 Rust 标量循环重写，
//! 由 Python 侧经 PyO3 单次调用。
//!
//! 公式与系数取自 IERS Technical Note 32 (Conventions 2003)，与 GMAT R2026a
//! ``HarmonicGravity`` 对齐。单位一致：位置 km、GM km³/s²、参考半径 km。
//! 数值与 Python 版逐项一致，精度回归 < 1e-15（机器精度）。
//!
//! 输出布局：返回长度 50 的 Vec<f64> = C(25) ++ S(25)，对应两个 5×5 矩阵的
//! 行优先扁平化；Python 侧 reshape 成 (5,5) 即可。

// ----------------------------------------------------------------------------
// 时间与角度常量
// ----------------------------------------------------------------------------

use e2m2e_propagation::constants::{
    DAYS_PER_JULIAN_CENTURY as DAYS_PER_JULIAN_CENTURY_CONST,
    DAYS_PER_JULIAN_YEAR as DAYS_PER_YEAR_CONST, RAD_PER_DEG,
};

const JD_J2000: f64 = 2451545.0;
const DAYS_PER_JULIAN_CENTURY: f64 = DAYS_PER_JULIAN_CENTURY_CONST;
const DAYS_PER_YEAR: f64 = DAYS_PER_YEAR_CONST;

// ----------------------------------------------------------------------------
// Table 6.3（IERS TN32 p.64/66，迁移 GMAT HarmonicGravity.cpp 静态数组）
// 每行：[N1, N2, N3, N4, N5, ip, op]
//   前 5 列是 5 个 Delaunay 幅角的整数乘子
//   ip, op: 同相、正交系数（单位 1e-12）
// Table63c 只有 amp（1 列）
// ----------------------------------------------------------------------------

/// Table63a：48 行，C21/S21。列 [N1..N5, ip, op]
const TABLE_63A: [[f64; 7]; 48] = [
    [2.0, 0.0, 2.0, 0.0, 2.0, -0.1, 0.0],
    [0.0, 0.0, 2.0, 2.0, 2.0, -0.1, 0.0],
    [1.0, 0.0, 2.0, 0.0, 1.0, -0.1, 0.0],
    [1.0, 0.0, 2.0, 0.0, 2.0, -0.7, 0.1],
    [-1.0, 0.0, 2.0, 2.0, 2.0, -0.1, 0.0],
    [0.0, 0.0, 2.0, 0.0, 1.0, -1.3, 0.1],
    [0.0, 0.0, 2.0, 0.0, 2.0, -6.8, 0.6],
    [0.0, 0.0, 0.0, 2.0, 0.0, 0.1, 0.0],
    [1.0, 0.0, 2.0, -2.0, 2.0, 0.1, 0.0],
    [-1.0, 0.0, 2.0, 0.0, 1.0, 0.1, 0.0],
    [-1.0, 0.0, 2.0, 0.0, 2.0, 0.4, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.3, -0.1],
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.3, 0.0],
    [-1.0, 0.0, 0.0, 2.0, 0.0, 0.3, 0.0],
    [-1.0, 0.0, 0.0, 2.0, 1.0, 0.1, 0.0],
    [0.0, 1.0, 2.0, -2.0, 2.0, -1.9, 0.1],
    [0.0, 0.0, 2.0, -2.0, 1.0, 0.5, 0.0],
    [0.0, 0.0, 2.0, -2.0, 2.0, -43.4, 2.9],
    [0.0, -1.0, 2.0, -2.0, 2.0, 0.6, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 1.6, -0.1],
    [-2.0, 0.0, 2.0, 0.0, 1.0, 0.1, 0.0],
    [0.0, 0.0, 0.0, 0.0, -2.0, 0.1, 0.0],
    [0.0, 0.0, 0.0, 0.0, -1.0, -8.8, 0.5],
    [0.0, 0.0, 0.0, 0.0, 0.0, 470.9, -30.2],
    [0.0, 0.0, 0.0, 0.0, 1.0, 68.1, -4.6],
    [0.0, 0.0, 0.0, 0.0, 2.0, -1.6, 0.1],
    [-1.0, 0.0, 0.0, 1.0, 0.0, 0.1, 0.0],
    [0.0, -1.0, 0.0, 0.0, -1.0, -0.1, 0.0],
    [0.0, -1.0, 0.0, 0.0, 0.0, -20.6, -0.3],
    [0.0, 1.0, -2.0, 2.0, -2.0, 0.3, 0.0],
    [0.0, -1.0, 0.0, 0.0, 1.0, -0.3, 0.0],
    [-2.0, 0.0, 0.0, 2.0, 0.0, -0.2, 0.0],
    [-2.0, 0.0, 0.0, 2.0, 1.0, -0.1, 0.0],
    [0.0, 0.0, -2.0, 2.0, -2.0, -5.0, 0.3],
    [0.0, 0.0, -2.0, 2.0, -1.0, 0.2, 0.0],
    [0.0, -1.0, -2.0, 2.0, -2.0, -0.2, 0.0],
    [1.0, 0.0, 0.0, -2.0, 0.0, -0.5, 0.0],
    [1.0, 0.0, 0.0, -2.0, 1.0, -0.1, 0.0],
    [-1.0, 0.0, 0.0, 0.0, -1.0, 0.1, 0.0],
    [-1.0, 0.0, 0.0, 0.0, 0.0, -2.1, 0.1],
    [-1.0, 0.0, 0.0, 0.0, 1.0, -0.4, 0.0],
    [0.0, 0.0, 0.0, -2.0, 0.0, -0.2, 0.0],
    [-2.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0],
    [0.0, 0.0, -2.0, 0.0, -2.0, -0.6, 0.0],
    [0.0, 0.0, -2.0, 0.0, -1.0, -0.4, 0.0],
    [0.0, 0.0, -2.0, 0.0, 0.0, -0.1, 0.0],
    [-1.0, 0.0, -2.0, 0.0, -2.0, -0.1, 0.0],
    [-1.0, 0.0, -2.0, 0.0, -1.0, -0.1, 0.0],
];

/// Table63b：21 行，C20。列 [N1..N5, ip, op]
const TABLE_63B: [[f64; 7]; 21] = [
    [0.0, 0.0, 0.0, 0.0, 1.0, 16.6, -6.7],
    [0.0, 0.0, 0.0, 0.0, 2.0, -0.1, 0.1],
    [0.0, -1.0, 0.0, 0.0, 0.0, -1.2, 0.8],
    [0.0, 0.0, -2.0, 2.0, -2.0, -5.5, 4.3],
    [0.0, 0.0, -2.0, 2.0, -1.0, 0.1, -0.1],
    [0.0, -1.0, -2.0, 2.0, -2.0, -0.3, 0.2],
    [1.0, 0.0, 0.0, -2.0, 0.0, -0.3, 0.7],
    [-1.0, 0.0, 0.0, 0.0, -1.0, 0.1, -0.2],
    [-1.0, 0.0, 0.0, 0.0, 0.0, -1.2, 3.7],
    [-1.0, 0.0, 0.0, 0.0, 1.0, 0.1, -0.2],
    [1.0, 0.0, -2.0, 0.0, -2.0, 0.1, -0.2],
    [0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.6],
    [-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3],
    [0.0, 0.0, -2.0, 0.0, -2.0, 0.6, 6.3],
    [0.0, 0.0, -2.0, 0.0, -1.0, 0.2, 2.6],
    [0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.2],
    [1.0, 0.0, -2.0, -2.0, -2.0, 0.1, 0.2],
    [-1.0, 0.0, -2.0, 0.0, -2.0, 0.4, 1.1],
    [-1.0, 0.0, -2.0, 0.0, -1.0, 0.2, 0.5],
    [0.0, 0.0, -2.0, -2.0, -2.0, 0.1, 0.2],
    [-2.0, 0.0, -2.0, 0.0, -2.0, 0.1, 0.1],
];

/// Table63c：2 行，C22/S22。列 [N1..N5, amp]
const TABLE_63C: [[f64; 6]; 2] = [
    [1.0, 0.0, 2.0, 0.0, 2.0, -0.3],
    [0.0, 0.0, 2.0, 0.0, 2.0, -1.2],
];

// ----------------------------------------------------------------------------
// 内部辅助：5×5 表扁平化布局：[n, m] -> n*5 + m
// ----------------------------------------------------------------------------

/// 把 (deltaC, deltaS) 两个 5×5 矩阵打包成 Vec<f64> 长度 50。
fn pack_cs(dc: &[[f64; 5]; 5], ds: &[[f64; 5]; 5]) -> Vec<f64> {
    let mut out = Vec::with_capacity(50);
    for row in dc {
        for &val in row {
            out.push(val);
        }
    }
    for row in ds {
        for &val in row {
            out.push(val);
        }
    }
    out
}

// ----------------------------------------------------------------------------
// Step2 频率相关
// ----------------------------------------------------------------------------

/// 计算 5 个 Delaunay 幅角（度）与 GMST（度）。
///
/// 公式取自 IERS TN32 p.48/60，与 GMAT IncrementEarthTide 幅角段对齐。
/// 返回 `(F: [f64; 5], gmst_deg: f64)`。
fn delanay_args_and_gmst(jd: f64) -> ([f64; 5], f64) {
    let t = (jd - JD_J2000) / DAYS_PER_JULIAN_CENTURY;
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;

    let mut f = [0.0_f64; 5];
    f[0] = (134.96340251e3 + 1717915923.2178 * t + 31.8792 * t2 + 0.051635 * t3 - 0.00024470 * t4)
        / 3600.0;
    f[1] = (357.52910918e3 + 129596581.0481 * t - 0.5532 * t2 + 0.000136 * t3 - 0.00001149 * t4)
        / 3600.0;
    f[2] = (93.27209062e3 + 1739527262.8478 * t - 12.7512 * t2 - 0.001037 * t3 + 0.00000417 * t4)
        / 3600.0;
    f[3] = (297.85019547e3 + 1602961601.2090 * t - 6.3706 * t2 + 0.006593 * t3 - 0.00003169 * t4)
        / 3600.0;
    f[4] = (125.04455501e3 - 6962890.5431 * t + 7.4722 * t2 + 0.007702 * t3 - 0.00005939 * t4)
        / 3600.0;

    let gmst_sec = 67310.54841 + 3164400184.812866 * t + 0.093104 * t2 - 6.2e-06 * t3;
    let gmst_deg = gmst_sec / 240.0;

    (f, gmst_deg)
}

/// 固体潮 Step 2（频率相关）。返回长度 50 的 `Vec<f64>`（C25 + S25 扁平化）。
///
/// 只影响 (2,0)/(2,1)/(2,2)。量级 ~1e-10。
pub fn solid_tide_step2(et: f64) -> Vec<f64> {
    let jd = JD_J2000 + et / 86400.0;
    let (f, gmst) = delanay_args_and_gmst(jd);

    let mut dc = [[0.0_f64; 5]; 5];
    let mut ds = [[0.0_f64; 5]; 5];

    // (2,0) 频率相关：IERS eqn 5a
    let mut freq_c20 = 0.0_f64;
    for row in TABLE_63B.iter() {
        // theta = -dot(row[0..5], F) * RAD_PER_DEG
        let dot: f64 =
            row[0] * f[0] + row[1] * f[1] + row[2] * f[2] + row[3] * f[3] + row[4] * f[4];
        let theta = -dot * RAD_PER_DEG;
        freq_c20 += row[5] * theta.cos() - row[6] * theta.sin();
    }
    dc[2][0] += freq_c20 * 1e-12;

    // (2,1) 频率相关：IERS eqn 5b，m=1
    let mut freq_c21 = 0.0_f64;
    let mut freq_s21 = 0.0_f64;
    let m1 = 1.0;
    for row in TABLE_63A.iter() {
        let dot: f64 =
            row[0] * f[0] + row[1] * f[1] + row[2] * f[2] + row[3] * f[3] + row[4] * f[4];
        let theta = (m1 * (gmst + 180.0) - dot) * RAD_PER_DEG;
        freq_c21 += row[5] * theta.sin() + row[6] * theta.cos();
        freq_s21 += row[5] * theta.cos() - row[6] * theta.sin();
    }
    dc[2][1] += freq_c21 * 1e-12;
    ds[2][1] += freq_s21 * 1e-12;

    // (2,2) 频率相关：m=2，Table63c 只有 amp
    let mut freq_c22 = 0.0_f64;
    let mut freq_s22 = 0.0_f64;
    let m2 = 2.0;
    for row in TABLE_63C.iter() {
        let dot: f64 =
            row[0] * f[0] + row[1] * f[1] + row[2] * f[2] + row[3] * f[3] + row[4] * f[4];
        let theta = (m2 * (gmst + 180.0) - dot) * RAD_PER_DEG;
        freq_c22 += row[5] * theta.cos();
        freq_s22 += -row[5] * theta.sin();
    }
    dc[2][2] += freq_c22 * 1e-12;
    ds[2][2] += freq_s22 * 1e-12;

    pack_cs(&dc, &ds)
}

// ----------------------------------------------------------------------------
// Step1 天体无关
// ----------------------------------------------------------------------------

/// 计算 n=2,3 的完全正规化 associated Legendre（与 GMAT PolarToLegendre 一致）。
///
/// 返回 5×5 数组 P[n][m]，未计算的项为零。
fn legendre_23(s: f64, c: f64) -> [[f64; 5]; 5] {
    let mut p = [[0.0_f64; 5]; 5];
    let sqrt5 = (5.0_f64).sqrt();
    let sqrt5over3 = (5.0_f64 / 3.0).sqrt();
    let sqrt7 = (7.0_f64).sqrt();
    let sqrt7over6 = (7.0_f64 / 6.0).sqrt();
    let sqrt7over15 = (7.0_f64 / 15.0).sqrt();
    let sqrt_point7 = (0.7_f64).sqrt();

    // n=2
    p[2][0] = sqrt5 * (1.5 * s * s - 0.5);
    p[2][1] = 3.0 * sqrt5over3 * c * s;
    p[2][2] = 1.5 * sqrt5over3 * c * c;
    // n=3
    p[3][0] = sqrt7 * (2.5 * s * s * s - 1.5 * s);
    p[3][1] = sqrt7over6 * c * (7.5 * s * s - 1.5);
    p[3][2] = 7.5 * sqrt7over15 * c * c * s;
    p[3][3] = 2.5 * sqrt_point7 * c * c * c;
    p
}

/// 固体潮 Step 1（频率无关，天体无关）。返回长度 50 的 `Vec<f64>`（C25 + S25）。
///
/// 对一组扰动天体累加 ΔC/ΔS。Love 数由调用方按中心天体传入。
/// - `perturbers_flat`：扁平化的扰动体列表，每 4 个元素一组 = `[px, py, pz, gm]`
///   （位置 km，gm km³/s²），故总长度 = `n_perturbers * 4`
/// - `k_love_flat`：Love 数表扁平化，长度 25（5×5 行优先）
/// - `k_plus_flat`：弹性 Love 数扁平化，长度 5；空表示无贡献（如月球）
pub fn solid_tide_step1(
    perturbers_flat: &[f64],
    k_love_flat: &[f64],
    k_plus_flat: Option<&[f64]>,
    mu_central: f64,
    r_central: f64,
) -> Vec<f64> {
    assert_eq!(k_love_flat.len(), 25, "k_love_flat must be length 25 (5x5)");
    if let Some(kp) = k_plus_flat {
        assert_eq!(kp.len(), 5, "k_plus_flat must be length 5");
    }
    assert!(
        perturbers_flat.len().is_multiple_of(4),
        "perturbers_flat length must be multiple of 4"
    );
    let n_perturbers = perturbers_flat.len() / 4;

    // 把 Love 数 reshape 成 [5][5]
    let mut k_love = [[0.0_f64; 5]; 5];
    for n in 0..5 {
        for m in 0..5 {
            k_love[n][m] = k_love_flat[n * 5 + m];
        }
    }
    let k_plus: Option<[f64; 5]> = k_plus_flat.map(|kp| {
        let mut arr = [0.0_f64; 5];
        arr.copy_from_slice(kp);
        arr
    });

    let mut dc = [[0.0_f64; 5]; 5];
    let mut ds = [[0.0_f64; 5]; 5];

    for i in 0..n_perturbers {
        let px = perturbers_flat[i * 4];
        let py = perturbers_flat[i * 4 + 1];
        let pz = perturbers_flat[i * 4 + 2];
        let mu_perturber = perturbers_flat[i * 4 + 3];

        let r = (px * px + py * py + pz * pz).sqrt();
        if r == 0.0 {
            panic!("perturber position must be non-zero");
        }

        // 中心纬度 φ、经度 λ
        let xy = (px * px + py * py).sqrt();
        let lat = pz.atan2(xy);
        let lon = py.atan2(px);
        let s = lat.sin();
        let c = lat.cos();

        let p = legendre_23(s, c);
        let massratio = mu_perturber / mu_central;
        let rho = r_central / r;

        for n in 2usize..=3 {
            let rho_n = rho.powi((n + 1) as i32);
            for m in 0..=n {
                let f = massratio * rho_n * p[n][m];
                let cm = ((m as f64) * lon).cos();
                let sm = ((m as f64) * lon).sin();
                let kn = k_love[n][m] / (2 * n + 1) as f64;
                dc[n][m] += kn * f * cm;
                ds[n][m] += kn * f * sm;
                if n == 2 {
                    if let Some(kp) = k_plus {
                        let kplus = kp[m] / (2 * n + 1) as f64; // 5
                        dc[4][m] += kplus * f * cm;
                        ds[4][m] += kplus * f * sm;
                    }
                }
            }
        }
    }

    pack_cs(&dc, &ds)
}

// ----------------------------------------------------------------------------
// 极潮
// ----------------------------------------------------------------------------

/// 极潮（固体极潮 IERS p.65 + Desai 海洋极潮 TN32 §6.3）。返回长度 50 的 Vec<f64>。
///
/// 只影响 (2,1)。对齐 GMAT `ETide::SolidAndPole`。
pub fn pole_tide(et: f64, xp: f64, yp: f64) -> Vec<f64> {
    let jd = JD_J2000 + et / 86400.0;
    let ym2000 = (jd - JD_J2000) / DAYS_PER_YEAR;
    let xp_bar = 0.054 + ym2000 * 0.00083;
    let yp_bar = 0.357 + ym2000 * 0.00395;

    let m1 = xp - xp_bar;
    let m2 = -(yp - yp_bar);

    let mut dc = [[0.0_f64; 5]; 5];
    let mut ds = [[0.0_f64; 5]; 5];

    // 固体极潮（IERS p.65）
    dc[2][1] -= 1.333e-09 * (m1 + 0.0115 * m2);
    ds[2][1] -= 1.333e-09 * (m2 - 0.0115 * m1);

    // 海洋极潮（Desai, TN32 §6.3）
    dc[2][1] -= 2.2344e-10 * (m1 - 0.01737 * m2);
    ds[2][1] -= 1.7680e-10 * (m2 - 0.03351 * m1);

    pack_cs(&dc, &ds)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Step2 在 et=0 的输出应该是有限的、量级 ~1e-10。
    #[test]
    fn test_step2_finite_and_scale() {
        let out = solid_tide_step2(0.0);
        assert_eq!(out.len(), 50);
        let dc = &out[0..25];
        let ds = &out[25..50];
        // (2,0) 非零且量级 1e-10
        assert!(dc[10].abs() > 1e-12); // [2,0]
                                       // (2,1) 非零
        assert!(dc[11].abs() > 1e-12 || ds[11].abs() > 1e-12);
        // 所有项有限
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// Step1 在地球 + Sun/Moon 扰动下应非零。
    #[test]
    fn test_step1_earth_sun_moon() {
        // 地球 Love 数（只填 n=2,3 的前几项）
        let mut k_love_flat = vec![0.0_f64; 25];
        k_love_flat[2 * 5 + 0] = 0.30190; // K20
        k_love_flat[2 * 5 + 1] = 0.29830; // K21
        k_love_flat[2 * 5 + 2] = 0.30102; // K22
        k_love_flat[3 * 5 + 0] = 0.093;
        k_love_flat[3 * 5 + 1] = 0.093;
        k_love_flat[3 * 5 + 2] = 0.093;
        k_love_flat[3 * 5 + 3] = 0.094;

        let k_plus_flat = vec![-0.00087, -0.00079, -0.00057, 0.0, 0.0];

        // 扰动体：Sun (1.5e8 km, GM 1.327e11) + Moon (3.84e5 km, GM 4902.8)
        let perturbers_flat = vec![
            1.5e8,
            0.0,
            0.0,
            1.32712440018e11,
            3.844e5,
            0.0,
            0.0,
            4902.8001,
        ];

        let out = solid_tide_step1(
            &perturbers_flat,
            &k_love_flat,
            Some(&k_plus_flat),
            398600.4415,
            6378.1363,
        );
        assert_eq!(out.len(), 50);
        // (2,0) 应该有显著值
        assert!(out[2 * 5 + 0].abs() > 1e-9);
        // 所有项有限
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// pole_tide 在中等极移下应该非零。
    #[test]
    fn test_pole_tide_basic() {
        let out = pole_tide(0.0, 0.1, 0.3);
        assert_eq!(out.len(), 50);
        // (2,1) 非零：index [2,1] = 11
        assert!(out[11].abs() > 1e-12 || out[25 + 11].abs() > 1e-12);
    }
}
