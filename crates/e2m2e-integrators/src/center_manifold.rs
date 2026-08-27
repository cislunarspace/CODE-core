//! 中心流形化简数值内核：两步 Lie 同调 + 频域 W + Poisson 链。
//!
//! 完整下沉 `CenterManifoldReducer.reduce` 的数值语义：
//! - Step 1 `"invariant"`：消去双曲方向不平衡项（实值 W 求解器）；
//! - Step 2 `"center"`：消去中心非共振耦合（复值 + MAD W 求解器）；
//! - 虚/实基底变换、特征频率、`list_deriv`、`ad_W^n/n!` 与 Python 一致。
//!
//! 多项式核（`poly_poisson` / `polylist_simplify`）在本模块内嵌。

use num_complex::Complex64;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustfft::num_complex::Complex as FftComplex;
use rustfft::FftPlanner;
use std::collections::HashMap;
use std::f64::consts::PI;

/// 6 维幂次（q1,q2,q3,p1,p2,p3）。
type Pow = [i64; 6];
type Poly = HashMap<Pow, Vec<Complex64>>;
type PolyByOrder = HashMap<i64, Poly>;

const N_VARS: usize = 6;
const DEFAULT_EXTENSION_RATIO: f64 = 0.2;
const DEFAULT_MAD_KVAL: f64 = 1e6;
const DEFAULT_DERIV_N: usize = 14;
const EPS_SIMPLIFY: f64 = 1e-15;
const M_ROW_TOL: f64 = 1e-12;

// ---------------------------------------------------------------------------
// 复基底 D / D⁻¹（与 Python center_manifold._D 逐元素一致）
// ---------------------------------------------------------------------------

fn inv_sqrt2() -> f64 {
    1.0 / 2.0_f64.sqrt()
}

/// 非零条目列表：`(row, col, value)`。
fn matrix_d_entries() -> [(usize, usize, Complex64); 10] {
    let s = inv_sqrt2();
    let j = Complex64::new(0.0, 1.0);
    [
        (0, 0, Complex64::new(1.0, 0.0)),
        (1, 1, Complex64::new(s, 0.0)),
        (1, 4, j * s),
        (2, 2, Complex64::new(s, 0.0)),
        (2, 5, j * s),
        (3, 3, Complex64::new(1.0, 0.0)),
        (4, 4, Complex64::new(s, 0.0)),
        (4, 1, j * s),
        (5, 5, Complex64::new(s, 0.0)),
        (5, 2, j * s),
    ]
}

fn matrix_d_inv_entries() -> [(usize, usize, Complex64); 10] {
    let s = inv_sqrt2();
    let mj = Complex64::new(0.0, -1.0);
    [
        (0, 0, Complex64::new(1.0, 0.0)),
        (1, 1, Complex64::new(s, 0.0)),
        (1, 4, mj * s),
        (2, 2, Complex64::new(s, 0.0)),
        (2, 5, mj * s),
        (3, 3, Complex64::new(1.0, 0.0)),
        (4, 1, mj * s),
        (4, 4, Complex64::new(s, 0.0)),
        (5, 2, mj * s),
        (5, 5, Complex64::new(s, 0.0)),
    ]
}

fn row_nonzeros(entries: &[(usize, usize, Complex64)], row: usize) -> Vec<(usize, Complex64)> {
    entries
        .iter()
        .filter(|(r, _, v)| *r == row && v.norm() > M_ROW_TOL)
        .map(|(_, c, v)| (*c, *v))
        .collect()
}

// ---------------------------------------------------------------------------
// 多项式核
// ---------------------------------------------------------------------------

fn zero_series(n: usize) -> Vec<Complex64> {
    vec![Complex64::new(0.0, 0.0); n]
}

fn is_zero_series(v: &[Complex64]) -> bool {
    v.iter().all(|c| c.norm() == 0.0)
}

fn series_mean_abs(v: &[Complex64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().map(|c| c.norm()).sum::<f64>() / v.len() as f64
}

fn series_add_assign(dst: &mut [Complex64], src: &[Complex64]) {
    let n = dst.len().min(src.len());
    for i in 0..n {
        dst[i] += src[i];
    }
}

fn series_scale(v: &[Complex64], s: Complex64) -> Vec<Complex64> {
    v.iter().map(|c| *c * s).collect()
}

fn series_mul(a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    let n = a.len().min(b.len());
    (0..n).map(|i| a[i] * b[i]).collect()
}

/// 6-DOF 辛 Poisson：`{f,g} = Σ_k (∂f/∂q_k ∂g/∂p_k − ∂f/∂p_k ∂g/∂q_k)`。
fn poly_poisson(poly1: &Poly, poly2: &Poly, n_samples: usize) -> Poly {
    let mut result: Poly = HashMap::new();
    for (pow1, coef1) in poly1 {
        if is_zero_series(coef1) {
            continue;
        }
        for (pow2, coef2) in poly2 {
            if is_zero_series(coef2) {
                continue;
            }
            for k in 0..3 {
                let a_k = pow1[k];
                let a_pk = pow1[k + 3];
                let b_k = pow2[k];
                let b_pk = pow2[k + 3];
                if !((a_k != 0 && b_pk != 0) || (a_pk != 0 && b_k != 0)) {
                    continue;
                }
                let mut new_pow = [0i64; N_VARS];
                for i in 0..N_VARS {
                    new_pow[i] = pow1[i] + pow2[i];
                }
                new_pow[k] -= 1;
                new_pow[k + 3] -= 1;
                if new_pow.iter().any(|&p| p < 0) {
                    continue;
                }
                let factor = (a_k * b_pk - a_pk * b_k) as f64;
                if factor == 0.0 {
                    continue;
                }
                let mut coef = series_mul(coef1, coef2);
                for c in &mut coef {
                    *c *= factor;
                }
                if is_zero_series(&coef) {
                    continue;
                }
                result
                    .entry(new_pow)
                    .and_modify(|e| series_add_assign(e, &coef))
                    .or_insert(coef);
            }
        }
    }
    if result.is_empty() {
        result.insert([0; N_VARS], zero_series(n_samples));
    }
    result
}

fn polylist_simplify(poly: &Poly, eps: f64) -> Poly {
    let mut result: Poly = HashMap::new();
    let mut sample_n = 1usize;
    for (pow, coef) in poly {
        sample_n = coef.len().max(1);
        let mean_abs = series_mean_abs(coef);
        if mean_abs <= eps {
            continue;
        }
        result
            .entry(*pow)
            .and_modify(|e| series_add_assign(e, coef))
            .or_insert_with(|| coef.clone());
    }
    // 合并后再滤一次（累加可能抵消）
    result.retain(|_, v| series_mean_abs(v) > eps);
    if result.is_empty() {
        result.insert([0; N_VARS], zero_series(sample_n));
    }
    result
}

// ---------------------------------------------------------------------------
// 线性基底变换
// ---------------------------------------------------------------------------

fn binom_u(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    let mut r = 1.0;
    for i in 0..k {
        r = r * (n - i) as f64 / (i + 1) as f64;
    }
    r
}

fn linear_basis_change(h_by_order: &PolyByOrder, inv: bool) -> PolyByOrder {
    let entries: Vec<(usize, usize, Complex64)> = if inv {
        matrix_d_inv_entries().to_vec()
    } else {
        matrix_d_entries().to_vec()
    };
    let mut out: PolyByOrder = HashMap::new();
    for (&order, poly) in h_by_order {
        let mut acc: Poly = HashMap::new();
        for (pow_tuple, coef) in poly {
            // {new_pow: 复标量系数}
            let mut terms: HashMap<Pow, Complex64> = HashMap::new();
            terms.insert(*pow_tuple, Complex64::new(1.0, 0.0));
            for i in 0..N_VARS {
                let ni = pow_tuple[i] as usize;
                if ni == 0 {
                    continue;
                }
                let row = row_nonzeros(&entries, i);
                if row.is_empty() {
                    continue;
                }
                if row.len() == 1 {
                    let (j0, m0) = row[0];
                    let factor = m0.powi(ni as i32);
                    let old = std::mem::take(&mut terms);
                    for (mut p, c) in old {
                        p[i] -= ni as i64;
                        p[j0] += ni as i64;
                        terms.insert(p, c * factor);
                    }
                } else {
                    let (j0, m0) = row[0];
                    let (j1, m1) = row[1];
                    let mut new_terms: HashMap<Pow, Complex64> = HashMap::new();
                    for (mut p, c) in terms.drain() {
                        p[i] -= ni as i64;
                        for s in 0..=ni {
                            let mut np = p;
                            np[j0] += (ni - s) as i64;
                            np[j1] += s as i64;
                            let cnew =
                                c * binom_u(ni, s) * m0.powi((ni - s) as i32) * m1.powi(s as i32);
                            *new_terms.entry(np).or_insert(Complex64::new(0.0, 0.0)) += cnew;
                        }
                    }
                    terms = new_terms;
                }
            }
            for (new_pow, c) in terms {
                let contrib = series_scale(coef, c);
                acc.entry(new_pow)
                    .and_modify(|e| series_add_assign(e, &contrib))
                    .or_insert(contrib);
            }
        }
        out.insert(order, acc);
    }
    out
}

// ---------------------------------------------------------------------------
// 频域 ODE 求解器
// ---------------------------------------------------------------------------

fn characteristic_freq(pow: &Pow, lam: f64, wp: f64, wv: f64) -> Complex64 {
    let (i1, i2, i3, j1, j2, j3) = (pow[0], pow[1], pow[2], pow[3], pow[4], pow[5]);
    Complex64::new(
        (j1 - i1) as f64 * lam,
        (j2 - i2) as f64 * wp + (j3 - i3) as f64 * wv,
    )
}

fn is_nearly_constant(f: &[Complex64]) -> bool {
    if f.is_empty() {
        return true;
    }
    let f0 = f[0];
    let max_abs = f.iter().map(|c| c.norm()).fold(0.0_f64, f64::max);
    let max_dev = f.iter().map(|c| (*c - f0).norm()).fold(0.0_f64, f64::max);
    max_dev <= 1e-2 * max_abs.max(1.0)
}

fn fft_forward(input: &[Complex64]) -> Vec<Complex64> {
    let n = input.len();
    let mut buf: Vec<FftComplex<f64>> = input.iter().map(|c| FftComplex::new(c.re, c.im)).collect();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buf);
    buf.into_iter()
        .map(|c| Complex64::new(c.re, c.im))
        .collect()
}

fn fft_inverse(input: &[Complex64]) -> Vec<Complex64> {
    let n = input.len();
    let mut buf: Vec<FftComplex<f64>> = input.iter().map(|c| FftComplex::new(c.re, c.im)).collect();
    let mut planner = FftPlanner::<f64>::new();
    let ifft = planner.plan_fft_inverse(n);
    ifft.process(&mut buf);
    // rustfft 逆变换不归一化；numpy.ifft 除以 n
    let scale = 1.0 / n as f64;
    buf.into_iter()
        .map(|c| Complex64::new(c.re * scale, c.im * scale))
        .collect()
}

/// `numpy.fft.fftfreq(n, d=dt)` 等价。
fn fftfreq(n: usize, dt: f64) -> Vec<f64> {
    let val = 1.0 / (n as f64 * dt);
    let n_half = (n - 1) / 2 + 1;
    let mut freq: Vec<f64> = (0..n_half).map(|i| i as f64 * val).collect();
    freq.extend((n_half..n).map(|i| (i as f64 - n as f64) * val));
    freq
}

fn mirror_extend(f: &[Complex64], m: usize) -> Vec<Complex64> {
    let n = f.len();
    let mut ext = Vec::with_capacity(n + 2 * m);
    // f[:M][::-1]
    for i in (0..m.min(n)).rev() {
        ext.push(f[i]);
    }
    // 若 m > n，numpy 会再翻；此处 m = ceil(0.2*N) < N（N>=2）
    for &v in f {
        ext.push(v);
    }
    for i in (n.saturating_sub(m)..n).rev() {
        ext.push(f[i]);
    }
    ext
}

fn solve_wfunc_fft(tlist: &[f64], forcing: &[Complex64], k: Complex64) -> Vec<Complex64> {
    let n = tlist.len();
    if n < 2 {
        return zero_series(n);
    }
    // 常数输入：代数特解 W = -f/k
    if is_nearly_constant(forcing) {
        let f0 = forcing[0];
        return vec![-f0 / k; n];
    }
    let dt = tlist[1] - tlist[0];
    let m = ((DEFAULT_EXTENSION_RATIO * n as f64).ceil() as usize).max(1);
    let f_ext = mirror_extend(forcing, m);
    let n_ext = f_ext.len();
    let f_hat = fft_forward(&f_ext);
    let freq = fftfreq(n_ext, dt);
    let mut y_hat = vec![Complex64::new(0.0, 0.0); n_ext];
    for i in 0..n_ext {
        let mut omega = 2.0 * PI * freq[i];
        if omega.abs() < 1e-12 {
            omega = 1e-12;
        }
        let h = Complex64::new(1.0, 0.0) / (Complex64::new(0.0, omega) - k);
        y_hat[i] = f_hat[i] * h;
    }
    let y_ext = fft_inverse(&y_hat);
    y_ext[m..m + n].to_vec()
}

fn limit_fft_outliers_mad(fft_result: &[Complex64], k_val: f64) -> (Vec<Complex64>, bool) {
    let n = fft_result.len();
    let amps: Vec<f64> = fft_result.iter().map(|c| c.norm()).collect();
    let mut sorted = amps.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med_a = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    };
    let mut devs: Vec<f64> = amps.iter().map(|a| (a - med_a).abs()).collect();
    devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = if n % 2 == 1 {
        devs[n / 2]
    } else {
        0.5 * (devs[n / 2 - 1] + devs[n / 2])
    };
    if mad == 0.0 {
        return (fft_result.to_vec(), false);
    }
    let threshold = med_a + k_val * (mad / 0.6745);
    let mut corrected = false;
    let mut out = fft_result.to_vec();
    for i in 0..n {
        if threshold < amps[i] {
            corrected = true;
            let scale = threshold / amps[i];
            out[i] *= scale;
        }
    }
    (out, corrected)
}

fn solve_wfunc_fft_imag(
    tlist: &[f64],
    forcing: &[Complex64],
    k: Complex64,
) -> (Vec<Complex64>, bool) {
    let n = tlist.len();
    if n < 2 {
        return (zero_series(n), false);
    }
    if is_nearly_constant(forcing) {
        let f0 = forcing[0];
        return (vec![-f0 / k; n], false);
    }
    let dt = tlist[1] - tlist[0];
    let m = ((DEFAULT_EXTENSION_RATIO * n as f64).ceil() as usize).max(1);
    let f_ext = mirror_extend(forcing, m);
    let n_ext = f_ext.len();
    let f_hat = fft_forward(&f_ext);
    let freq = fftfreq(n_ext, dt);
    let mut y_hat = vec![Complex64::new(0.0, 0.0); n_ext];
    for i in 0..n_ext {
        let mut omega = 2.0 * PI * freq[i];
        if omega.abs() < 1e-12 {
            omega = 1e-12;
        }
        let h = Complex64::new(1.0, 0.0) / (Complex64::new(0.0, omega) - k);
        y_hat[i] = f_hat[i] * h;
    }
    let y_ext = fft_inverse(&y_hat);
    let y = y_ext[m..m + n].to_vec();
    let (y_corr_freq, corrected) = limit_fft_outliers_mad(&y_hat, DEFAULT_MAD_KVAL);
    if corrected {
        let y_corr_ext = fft_inverse(&y_corr_freq);
        return (y_corr_ext[m..m + n].to_vec(), true);
    }
    (y, false)
}

// ---------------------------------------------------------------------------
// list_deriv（高阶数值微分）
// ---------------------------------------------------------------------------

fn vandermonde_deriv_coeffs(n: usize, mode: i32) -> Vec<f64> {
    let k: Vec<f64> = match mode {
        0 => {
            let half = n / 2;
            ((-(half as i64))..=(half as i64))
                .map(|x| x as f64)
                .collect()
        }
        1 => (0..=n).map(|x| x as f64).collect(),
        -1 => ((-(n as i64))..=0).map(|x| x as f64).collect(),
        _ => return vec![0.0],
    };
    let m = k.len();
    if m <= 1 {
        return vec![0.0];
    }
    // V[i,j] = k[j]^i （increasing Vandermonde 转置后）
    // 解 V c = b，b = [0,1,0,...]
    let mut a: Vec<Vec<f64>> = (0..m)
        .map(|i| k.iter().map(|&kj| kj.powi(i as i32)).collect())
        .collect();
    let mut b = vec![0.0; m];
    if m > 1 {
        b[1] = 1.0;
    }
    // 高斯消元
    for col in 0..m {
        let mut piv = col;
        for r in (col + 1)..m {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        a.swap(col, piv);
        b.swap(col, piv);
        let diag = a[col][col];
        if diag.abs() < 1e-18 {
            continue;
        }
        for a_col_j in a[col].iter_mut().skip(col) {
            *a_col_j /= diag;
        }
        b[col] /= diag;
        for r in 0..m {
            if r == col {
                continue;
            }
            let f = a[r][col];
            let col_slice: Vec<f64> = a[col][col..].to_vec();
            for (offset, a_col_j) in col_slice.into_iter().enumerate() {
                a[r][col + offset] -= f * a_col_j;
            }
            b[r] -= f * b[col];
        }
    }
    b
}

fn list_deriv_real(y: &[f64], h: f64, n: usize, swi: usize, ord_boundary: usize) -> Vec<f64> {
    let n_pts = y.len();
    let mut dy = vec![0.0; n_pts];
    if n_pts == 0 {
        return dy;
    }
    let half = n / 2;

    if n < n_pts {
        let c = vandermonde_deriv_coeffs(n, 0);
        for i in half..(n_pts - half) {
            let mut acc = 0.0;
            for (idx_k, kk) in ((-(half as i64))..=(half as i64)).enumerate() {
                acc += c[idx_k] * y[(i as i64 + kk) as usize];
            }
            dy[i] = acc / h;
        }
    }

    for i in 0..swi.min(n_pts) {
        let ord_eff = ord_boundary.min(n_pts.saturating_sub(i + 1));
        if ord_eff < 1 {
            dy[i] = if i + 1 < n_pts {
                (y[i + 1] - y[i]) / h
            } else {
                0.0
            };
        } else {
            let c_fwd = vandermonde_deriv_coeffs(ord_eff, 1);
            let mut acc = 0.0;
            for (j, &cj) in c_fwd.iter().enumerate() {
                acc += cj * y[i + j];
            }
            dy[i] = acc / h;
        }
    }

    for i in 0..swi.min(n_pts) {
        let ri = n_pts - 1 - i;
        let ord_eff = ord_boundary.min(ri);
        if ord_eff < 1 {
            dy[ri] = if ri > 0 { (y[ri] - y[ri - 1]) / h } else { 0.0 };
        } else {
            let c_bwd = vandermonde_deriv_coeffs(ord_eff, -1);
            let mut acc = 0.0;
            for (j, &cj) in c_bwd.iter().enumerate() {
                acc += cj * y[ri - ord_eff + j];
            }
            dy[ri] = acc / h;
        }
    }

    for i in swi..half.min(n_pts.saturating_sub(swi)) {
        let span = 2 * i;
        if span < 2 {
            continue;
        }
        let c_center = vandermonde_deriv_coeffs(span, 0);
        let offset = span / 2;
        let mut acc = 0.0;
        for (idx_k, kk) in ((-(offset as i64))..=(offset as i64)).enumerate() {
            acc += c_center[idx_k] * y[(i as i64 + kk) as usize];
        }
        dy[i] = acc / h;
        let ri = n_pts - 1 - i;
        acc = 0.0;
        for (idx_k, kk) in ((-(offset as i64))..=(offset as i64)).enumerate() {
            acc += c_center[idx_k] * y[(ri as i64 + kk) as usize];
        }
        dy[ri] = acc / h;
    }
    dy
}

fn list_deriv_complex(y: &[Complex64], h: f64) -> Vec<Complex64> {
    let re: Vec<f64> = y.iter().map(|c| c.re).collect();
    let im: Vec<f64> = y.iter().map(|c| c.im).collect();
    let dre = list_deriv_real(&re, h, DEFAULT_DERIV_N, 4, 10);
    let dim = list_deriv_real(&im, h, DEFAULT_DERIV_N, 4, 10);
    dre.into_iter()
        .zip(dim)
        .map(|(r, i)| Complex64::new(r, i))
        .collect()
}

// ---------------------------------------------------------------------------
// 判别条件
// ---------------------------------------------------------------------------

fn is_invariant_term(pow: &Pow) -> bool {
    pow[0] == pow[3]
}

fn is_center_term(pow: &Pow) -> bool {
    pow[0] == pow[3] && pow[1] == pow[4] && pow[2] == pow[5]
}

fn delete_invariant(pow: &Pow, _eliminated: &[Pow]) -> bool {
    pow[0] == pow[3]
}

fn delete_center(pow: &Pow, eliminated: &[Pow]) -> bool {
    !eliminated.iter().any(|e| e == pow)
}

// ---------------------------------------------------------------------------
// Lie 变换单步
// ---------------------------------------------------------------------------

fn factorial_f64(n: u64) -> f64 {
    let mut r = 1.0;
    for i in 2..=n {
        r *= i as f64;
    }
    r
}

#[allow(clippy::too_many_arguments)]
fn lie_transform_step(
    h_by_order: &mut PolyByOrder,
    max_order: i64,
    tlist: &[f64],
    lam: f64,
    wp: f64,
    wv: f64,
    keep: fn(&Pow) -> bool,
    delete: fn(&Pow, &[Pow]) -> bool,
    use_imag_solver: bool,
) -> HashMap<i64, Poly> {
    let n = tlist.len();
    let mut w_series: HashMap<i64, Poly> = HashMap::new();

    for order in 3..=max_order {
        h_by_order.entry(order).or_default();
        let h_order_keys: Vec<(Pow, Vec<Complex64>)> = h_by_order
            .get(&order)
            .map(|p| p.iter().map(|(k, v)| (*k, v.clone())).collect())
            .unwrap_or_default();

        let mut w_temp: Poly = HashMap::new();
        let mut wd_temp: Poly = HashMap::new();
        let mut eliminated: Vec<Pow> = Vec::new();

        for (pow_tuple, coef_arr) in h_order_keys {
            if keep(&pow_tuple) {
                w_temp.insert(pow_tuple, zero_series(n));
                continue;
            }
            let k = characteristic_freq(&pow_tuple, lam, wp, wv);
            let (w_func, wd_func) = if !use_imag_solver {
                // Step 1：强迫取实部（与 Python `coef_c.real` 一致）
                let forcing: Vec<Complex64> =
                    coef_arr.iter().map(|c| Complex64::new(c.re, 0.0)).collect();
                let w = solve_wfunc_fft(tlist, &forcing, k);
                let mut wd = Vec::with_capacity(n);
                for i in 0..n {
                    wd.push(-k * w[i] - coef_arr[i]);
                }
                (w, wd)
            } else {
                let (w, corrected) = solve_wfunc_fft_imag(tlist, &coef_arr, k);
                let mut wd = if !corrected {
                    (0..n).map(|i| k * w[i] + coef_arr[i]).collect()
                } else {
                    let dt = if n >= 2 {
                        // mean(diff(tlist))
                        let mut s = 0.0;
                        for i in 1..n {
                            s += tlist[i] - tlist[i - 1];
                        }
                        s / (n - 1) as f64
                    } else {
                        1.0
                    };
                    list_deriv_complex(&w, dt)
                };
                for c in &mut wd {
                    *c = -*c;
                }
                (w, wd)
            };
            w_temp.insert(pow_tuple, w_func);
            wd_temp.insert(pow_tuple, wd_func);
            eliminated.push(pow_tuple);
        }
        w_series.insert(order, w_temp.clone());

        // Poisson 括号链 ad_W^n / n!
        for j in 2..max_order {
            let h_j = match h_by_order.get(&j) {
                Some(p) if !p.is_empty() => p.clone(),
                _ => continue,
            };
            let mut cur_order = j + order - 2;
            if cur_order > max_order {
                continue;
            }
            let mut num: u64 = 1;
            let mut p_prev = h_j;
            while cur_order <= max_order {
                let mut p = poly_poisson(&p_prev, &w_temp, n);
                let fact = factorial_f64(num);
                for v in p.values_mut() {
                    for c in v.iter_mut() {
                        *c /= fact;
                    }
                }
                num += 1;
                let target = h_by_order.entry(cur_order).or_default();
                for (kk, v) in &p {
                    target
                        .entry(*kk)
                        .and_modify(|e| series_add_assign(e, v))
                        .or_insert_with(|| v.clone());
                }
                cur_order += order - 2;
                p_prev = p;
            }
        }

        // 加 Ẇ
        {
            let target = h_by_order.entry(order).or_default();
            for (kk, v) in &wd_temp {
                target
                    .entry(*kk)
                    .and_modify(|e| series_add_assign(e, v))
                    .or_insert_with(|| v.clone());
            }
        }

        // delete_criterion
        if let Some(poly) = h_by_order.get_mut(&order) {
            poly.retain(|k, _| delete(k, &eliminated));
            if poly.is_empty() {
                poly.insert([0; N_VARS], zero_series(n));
            }
        }
    }

    // 化简各阶
    let keys: Vec<i64> = h_by_order.keys().copied().collect();
    for o in keys {
        if let Some(poly) = h_by_order.remove(&o) {
            if !poly.is_empty() {
                h_by_order.insert(o, polylist_simplify(&poly, EPS_SIMPLIFY));
            }
        }
    }
    w_series
}

// ---------------------------------------------------------------------------
// 组装与 reduce
// ---------------------------------------------------------------------------

fn assemble_hamiltonian(
    n: usize,
    lam: f64,
    wp: f64,
    wv: f64,
    max_order: i64,
    higher_pows: &[Pow],
    higher_coefs: &[Vec<f64>],
) -> Result<PolyByOrder, String> {
    let mut h: PolyByOrder = HashMap::new();
    let ones = |s: f64| -> Vec<Complex64> { vec![Complex64::new(s, 0.0); n] };

    let put = |h: &mut PolyByOrder, deg: i64, key: Pow, val: Vec<Complex64>| {
        let poly = h.entry(deg).or_default();
        poly.entry(key)
            .and_modify(|e| series_add_assign(e, &val))
            .or_insert(val);
    };

    put(&mut h, 2, [1, 0, 0, 1, 0, 0], ones(lam));
    put(&mut h, 2, [0, 2, 0, 0, 0, 0], ones(wp / 2.0));
    put(&mut h, 2, [0, 0, 0, 0, 2, 0], ones(wp / 2.0));
    put(&mut h, 2, [0, 0, 2, 0, 0, 0], ones(wv / 2.0));
    put(&mut h, 2, [0, 0, 0, 0, 0, 2], ones(wv / 2.0));

    for (pow, coef) in higher_pows.iter().zip(higher_coefs.iter()) {
        let deg: i64 = pow.iter().sum();
        if deg < 1 || deg > max_order {
            continue;
        }
        let arr: Vec<Complex64> = if coef.len() == 1 {
            ones(coef[0])
        } else if coef.len() == n {
            coef.iter().map(|&x| Complex64::new(x, 0.0)).collect()
        } else {
            return Err(format!(
                "hamiltonian_terms 系数长度 {} 与 tlist 长度 {} 不一致（pow={:?}）",
                coef.len(),
                n,
                pow
            ));
        };
        put(&mut h, deg, *pow, arr);
    }
    Ok(h)
}

fn max_hyperbolic_center_coupling(terms: &[(Pow, Vec<f64>)]) -> f64 {
    let mut mx = 0.0;
    for (pow, coef) in terms {
        if pow[0] == pow[3] {
            continue;
        }
        let center = pow[1] + pow[4] + pow[2] + pow[5];
        if center > 0 {
            let m = coef.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
            if m > mx {
                mx = m;
            }
        }
    }
    mx
}

fn take_real_part(h_by_order: PolyByOrder) -> PolyByOrder {
    let mut out = PolyByOrder::new();
    for (o, poly) in h_by_order {
        let mut p = Poly::new();
        for (k, v) in poly {
            p.insert(
                k,
                v.into_iter().map(|c| Complex64::new(c.re, 0.0)).collect(),
            );
        }
        out.insert(o, p);
    }
    out
}

/// 中心流形化简完整 reduce。
///
/// **参数**
///
/// - `tlist`: 等距时间采样。
/// - `lam`, `wp`, `wv`: 双曲指数与平面/垂直中心频率（来自 QF 的 `D`）。
/// - `max_order`: 截断阶。
/// - `steps`: `"invariant"` / `"center"` 序列。
/// - `higher_pows`: 高阶项幂次 `(K, 6)`。
/// - `higher_coefs`: 高阶项实系数时间序列 `(K, N)` 或 `(K, 1)`。
///
/// **返回**
///
/// `(w_entries, h_pows, h_coefs, pre_coupling, steps_done)`：
/// - `w_entries`: `(step, order, pow[6], re[N], im[N])` 列表；
/// - `h_pows` / `h_coefs`: 化简后实 Hamiltonian；
/// - `pre_coupling`: 化简前双曲-中心耦合度量；
/// - `steps_done`: 实际执行步骤。
#[pyfunction]
#[pyo3(signature = (tlist, lam, wp, wv, max_order, steps, higher_pows, higher_coefs))]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn center_manifold_reduce_py(
    tlist: Vec<f64>,
    lam: f64,
    wp: f64,
    wv: f64,
    max_order: i64,
    steps: Vec<String>,
    higher_pows: Vec<Vec<i64>>,
    higher_coefs: Vec<Vec<f64>>,
) -> PyResult<(
    Vec<(String, i64, Vec<i64>, Vec<f64>, Vec<f64>)>,
    Vec<Vec<i64>>,
    Vec<Vec<f64>>,
    f64,
    Vec<String>,
)> {
    if max_order < 1 {
        return Err(PyValueError::new_err(format!(
            "max_order 必须为正，得到 {max_order}"
        )));
    }
    let valid = ["invariant", "center"];
    let bad: Vec<&String> = steps
        .iter()
        .filter(|s| !valid.contains(&s.as_str()))
        .collect();
    if !bad.is_empty() {
        return Err(PyValueError::new_err(format!(
            "steps 只能含 {:?}, 得到非法值：{:?}",
            valid, bad
        )));
    }
    if higher_pows.len() != higher_coefs.len() {
        return Err(PyValueError::new_err(format!(
            "higher_pows 与 higher_coefs 长度不一致：{} vs {}",
            higher_pows.len(),
            higher_coefs.len()
        )));
    }
    let n = tlist.len();
    if n == 0 {
        return Err(PyValueError::new_err("tlist 不能为空"));
    }

    let mut pows6: Vec<Pow> = Vec::with_capacity(higher_pows.len());
    for p in &higher_pows {
        if p.len() != 6 {
            return Err(PyValueError::new_err(format!(
                "pows 每行必须 6 列，得到 {}",
                p.len()
            )));
        }
        pows6.push([p[0], p[1], p[2], p[3], p[4], p[5]]);
    }

    // 诊断：化简前耦合
    let pre_terms: Vec<(Pow, Vec<f64>)> = pows6
        .iter()
        .zip(higher_coefs.iter())
        .map(|(p, c)| {
            let arr = if c.len() == 1 {
                vec![c[0]; n]
            } else {
                c.clone()
            };
            (*p, arr)
        })
        .collect();
    let pre_coupling = max_hyperbolic_center_coupling(&pre_terms);

    let mut h_by_order = assemble_hamiltonian(n, lam, wp, wv, max_order, &pows6, &higher_coefs)
        .map_err(PyValueError::new_err)?;

    let mut w_all: Vec<(String, i64, Vec<i64>, Vec<f64>, Vec<f64>)> = Vec::new();
    let mut steps_done: Vec<String> = Vec::new();

    for step in &steps {
        // 虚变换
        h_by_order = linear_basis_change(&h_by_order, false);

        let w_step = if step == "invariant" {
            lie_transform_step(
                &mut h_by_order,
                max_order,
                &tlist,
                lam,
                wp,
                wv,
                is_invariant_term,
                delete_invariant,
                false,
            )
        } else {
            lie_transform_step(
                &mut h_by_order,
                max_order,
                &tlist,
                lam,
                wp,
                wv,
                is_center_term,
                delete_center,
                true,
            )
        };

        // 实变换 + 取实部
        h_by_order = linear_basis_change(&h_by_order, true);
        h_by_order = take_real_part(h_by_order);

        // 打包 W（保持复值）
        for (order, poly) in w_step {
            for (pow, coef) in poly {
                let re: Vec<f64> = coef.iter().map(|c| c.re).collect();
                let im: Vec<f64> = coef.iter().map(|c| c.im).collect();
                w_all.push((step.clone(), order, pow.to_vec(), re, im));
            }
        }
        steps_done.push(step.clone());
    }

    // 汇总最终 H
    let mut final_terms: Poly = HashMap::new();
    let mut orders: Vec<i64> = h_by_order.keys().copied().collect();
    orders.sort();
    for o in orders {
        if let Some(poly) = h_by_order.get(&o) {
            for (k, v) in poly {
                final_terms
                    .entry(*k)
                    .and_modify(|e| series_add_assign(e, v))
                    .or_insert_with(|| v.clone());
            }
        }
    }
    let final_terms = polylist_simplify(&final_terms, EPS_SIMPLIFY);

    let mut h_pows: Vec<Vec<i64>> = Vec::new();
    let mut h_coefs: Vec<Vec<f64>> = Vec::new();
    for (p, c) in final_terms {
        h_pows.push(p.to_vec());
        h_coefs.push(c.iter().map(|z| z.re).collect());
    }

    Ok((w_all, h_pows, h_coefs, pre_coupling, steps_done))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poisson_basic_qp() {
        // {q1, p1} = 1
        let mut f = Poly::new();
        f.insert([1, 0, 0, 0, 0, 0], vec![Complex64::new(1.0, 0.0)]);
        let mut g = Poly::new();
        g.insert([0, 0, 0, 1, 0, 0], vec![Complex64::new(1.0, 0.0)]);
        let pb = poly_poisson(&f, &g, 1);
        let c = pb.get(&[0; 6]).unwrap();
        assert!((c[0].re - 1.0).abs() < 1e-12);
    }

    #[test]
    fn characteristic_freq_matches_formula() {
        let k = characteristic_freq(&[1, 2, 0, 0, 0, 0], 2.0, 3.0, 4.0);
        // (0-1)*2 + i*((0-2)*3 + (0-0)*4) = -2 - 6i
        assert!((k.re + 2.0).abs() < 1e-12);
        assert!((k.im + 6.0).abs() < 1e-12);
    }

    #[test]
    fn constant_forcing_algebraic_solution() {
        let t: Vec<f64> = (0..16).map(|i| i as f64 * 0.1).collect();
        let f = vec![Complex64::new(0.5, 0.0); 16];
        let k = Complex64::new(2.0, 0.0);
        let y = solve_wfunc_fft(&t, &f, k);
        for yi in &y {
            assert!((yi.re + 0.25).abs() < 1e-12);
            assert!(yi.im.abs() < 1e-12);
        }
    }

    #[test]
    fn virtual_real_roundtrip_q2_cubed() {
        let ones = vec![Complex64::new(1.0, 0.0); 4];
        let mut h = PolyByOrder::new();
        let mut p = Poly::new();
        p.insert([0, 3, 0, 0, 0, 0], ones.clone());
        h.insert(3, p);
        let hc = linear_basis_change(&h, false);
        let hr = linear_basis_change(&hc, true);
        let got = hr.get(&3).unwrap().get(&[0, 3, 0, 0, 0, 0]).unwrap();
        assert!((got[0].re - 1.0).abs() < 1e-12);
        assert!(got[0].im.abs() < 1e-12);
    }
}
