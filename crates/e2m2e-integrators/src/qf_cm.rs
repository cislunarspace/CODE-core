//! QF ↔ CM：quasi-Floquet 坐标 ↔ 中心流形坐标（高阶 Lie 级数）。
//!
//! 对应 Python ``e2m2e.algorithm.normal_form.coord_trans.qf_cm`` 与 qiao
//! ``qpQF2qpCM`` / ``qpCM2qpQF``。复值 Hamilton 流 ``dX/dt = J·∇W`` 用
//! **12 实维分裂**（``[Re X, Im X]``）走现有实值 DOP853 ``solve_ivp``，
//! 语义与 scipy 对复 ``y0`` 的内部分裂等价（issue #465，关闭 #336 例外）。
//!
//! 完整链：实→复基底 ``D⁻¹`` → 逐阶 Lie 流（正向 W 取反升序；反向不取反降序）
//! → 复→实基底 ``D`` 取实部。阶集合由输入决定，不硬编码截断。

use e2m2e_propagation::pd78::PD78_TABLE as DOP853;
use e2m2e_propagation::solve_ivp::{solve_ivp_impl, MAX_ADAPTIVE_STEPS};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// 复标量（避免引入 num-complex 依赖；仅需加/乘/整数幂）
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default)]
struct C64 {
    re: f64,
    im: f64,
}

impl C64 {
    const ZERO: Self = Self { re: 0.0, im: 0.0 };
    const ONE: Self = Self { re: 1.0, im: 0.0 };

    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    fn from_real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }

    fn add(self, o: Self) -> Self {
        Self {
            re: self.re + o.re,
            im: self.im + o.im,
        }
    }

    fn mul(self, o: Self) -> Self {
        Self {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }

    fn scale(self, s: f64) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }

    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }

    /// 非负整数幂。``z^0 = 1``（含 ``0^0=1``），与 numpy 整数幂一致。
    fn powi_nonneg(self, n: i64) -> Self {
        debug_assert!(n >= 0);
        if n == 0 {
            return Self::ONE;
        }
        let mut acc = Self::ONE;
        let mut base = self;
        let mut e = n as u64;
        while e > 0 {
            if e & 1 == 1 {
                acc = acc.mul(base);
            }
            base = base.mul(base);
            e >>= 1;
        }
        acc
    }
}

// ---------------------------------------------------------------------------
// 实/复基底 D（与 Python ``qf_cm._D`` / qiao ``_complex_basis`` 逐元素一致）
// ---------------------------------------------------------------------------

/// 6×6 复矩阵，按行主序存 ``(re, im)``。
fn d_matrix() -> [[C64; 6]; 6] {
    let s = 1.0 / 2.0_f64.sqrt();
    let mut d = [[C64::ZERO; 6]; 6];
    d[0][0] = C64::from_real(1.0);
    d[1][1] = C64::from_real(s);
    d[1][4] = C64::new(0.0, s); // +i/√2
    d[2][2] = C64::from_real(s);
    d[2][5] = C64::new(0.0, s);
    d[3][3] = C64::from_real(1.0);
    d[4][4] = C64::from_real(s);
    d[4][1] = C64::new(0.0, s);
    d[5][5] = C64::from_real(s);
    d[5][2] = C64::new(0.0, s);
    d
}

/// ``D⁻¹``（预计算解析形式，避免运行时求逆漂移）。
///
/// ``D`` 块对角：双曲 1×1 单位块 + 两个 2×2 中心块
/// ``M = (1/√2) [[1, i], [i, 1]]``，其逆 ``M⁻¹ = (1/√2) [[1, -i], [-i, 1]]``。
fn d_inv_matrix() -> [[C64; 6]; 6] {
    let s = 1.0 / 2.0_f64.sqrt();
    let mut inv = [[C64::ZERO; 6]; 6];
    inv[0][0] = C64::from_real(1.0);
    inv[1][1] = C64::from_real(s);
    inv[1][4] = C64::new(0.0, -s); // -i/√2
    inv[2][2] = C64::from_real(s);
    inv[2][5] = C64::new(0.0, -s);
    inv[3][3] = C64::from_real(1.0);
    inv[4][4] = C64::from_real(s);
    inv[4][1] = C64::new(0.0, -s);
    inv[5][5] = C64::from_real(s);
    inv[5][2] = C64::new(0.0, -s);
    inv
}

fn mat_vec(m: &[[C64; 6]; 6], x: &[C64; 6]) -> [C64; 6] {
    let mut out = [C64::ZERO; 6];
    for i in 0..6 {
        let mut acc = C64::ZERO;
        for j in 0..6 {
            acc = acc.add(m[i][j].mul(x[j]));
        }
        out[i] = acc;
    }
    out
}

fn re_to_im(x_real: &[f64; 6]) -> [C64; 6] {
    let x = [
        C64::from_real(x_real[0]),
        C64::from_real(x_real[1]),
        C64::from_real(x_real[2]),
        C64::from_real(x_real[3]),
        C64::from_real(x_real[4]),
        C64::from_real(x_real[5]),
    ];
    mat_vec(&d_inv_matrix(), &x)
}

fn im_to_re(x_im: &[C64; 6]) -> [f64; 6] {
    let y = mat_vec(&d_matrix(), x_im);
    // 合法输入下虚部应 ~0；取实部丢弃数值残余。
    [y[0].re, y[1].re, y[2].re, y[3].re, y[4].re, y[5].re]
}

// ---------------------------------------------------------------------------
// Hamilton 流 dX/dt = J·∇W
// ---------------------------------------------------------------------------

/// 输出方向 out_j ← (求导坐标 coord, 符号 sign)。
/// dq_i/dt = +∂W/∂p_i；dp_i/dt = −∂W/∂q_i。
const DERIV: [(usize, f64); 6] = [
    (3, 1.0),
    (4, 1.0),
    (5, 1.0),
    (0, -1.0),
    (1, -1.0),
    (2, -1.0),
];

/// 单阶 W 多项式：幂次与复系数（已按 forward 符号预处理）。
struct WPoly {
    exps: Vec<[i64; 6]>,
    coefs: Vec<C64>,
}

impl WPoly {
    fn is_empty(&self) -> bool {
        self.coefs.is_empty()
    }
}

/// 向量化 Hamilton 流右端（降幂写法，``0^0=1``）。
fn hamilton_flow_rhs(x: &[C64; 6], poly: &WPoly) -> [C64; 6] {
    let n_terms = poly.coefs.len();
    if n_terms == 0 {
        return [C64::ZERO; 6];
    }
    // B[k][j] = x[j]^exps[k][j]
    let mut b: Vec<[C64; 6]> = vec![[C64::ZERO; 6]; n_terms];
    for (k, exp) in poly.exps.iter().enumerate() {
        for j in 0..6 {
            b[k][j] = x[j].powi_nonneg(exp[j]);
        }
    }

    let mut dx = [C64::ZERO; 6];
    for (out_j, (coord, sign)) in DERIV.iter().enumerate() {
        let mut acc = C64::ZERO;
        for ((exp, coef), bk) in poly.exps.iter().zip(poly.coefs.iter()).zip(b.iter()) {
            let e_col = exp[*coord];
            if e_col == 0 {
                continue; // 该项对 ∂/∂coord 贡献 0
            }
            // Π_{m≠coord} x_m^{n_m}
            let mut prod_excl = C64::ONE;
            for (m, bm) in bk.iter().enumerate() {
                if m != *coord {
                    prod_excl = prod_excl.mul(*bm);
                }
            }
            // x_coord^(n_coord-1)，n≥1 时 max(n-1,0)=n-1
            let red_pow = (e_col - 1).max(0);
            let qp_red = x[*coord].powi_nonneg(red_pow);
            let term = coef.scale(e_col as f64).mul(prod_excl).mul(qp_red);
            acc = acc.add(term);
        }
        dx[out_j] = acc.scale(*sign);
    }
    dx
}

// ---------------------------------------------------------------------------
// 12 实维分裂积分
// ---------------------------------------------------------------------------

fn pack12(x: &[C64; 6]) -> [f64; 12] {
    [
        x[0].re, x[1].re, x[2].re, x[3].re, x[4].re, x[5].re, x[0].im, x[1].im, x[2].im, x[3].im,
        x[4].im, x[5].im,
    ]
}

fn unpack12(y: &[f64]) -> [C64; 6] {
    debug_assert!(y.len() >= 12);
    [
        C64::new(y[0], y[6]),
        C64::new(y[1], y[7]),
        C64::new(y[2], y[8]),
        C64::new(y[3], y[9]),
        C64::new(y[4], y[10]),
        C64::new(y[5], y[11]),
    ]
}

/// 对单阶 W 从 t=0 积到 t=1。失败返回 ``Err``。
fn integrate_lie_order(
    x0: &[C64; 6],
    poly: &WPoly,
    rtol: f64,
    atol: f64,
) -> Result<[C64; 6], String> {
    let y0 = pack12(x0).to_vec();
    let t_eval = [0.0_f64, 1.0];
    let rhs = |_t: f64, y: &[f64]| -> Result<Vec<f64>, String> {
        if y.len() != 12 {
            return Err(format!("期望 12 维实状态，得到 {}", y.len()));
        }
        let x = unpack12(y);
        let dx = hamilton_flow_rhs(&x, poly);
        Ok(pack12(&dx).to_vec())
    };
    let states = solve_ivp_impl(
        &DOP853,
        rhs,
        (0.0, 1.0),
        &y0,
        &t_eval,
        rtol,
        atol,
        f64::INFINITY,
        MAX_ADAPTIVE_STEPS,
        None,
    );
    if states.len() != t_eval.len() {
        return Err(format!(
            "Lie 流积分未完成：输出 {}/{} 点（步长塌缩或步数耗尽）",
            states.len(),
            t_eval.len()
        ));
    }
    let yf = &states[states.len() - 1];
    if yf.len() != 12 || yf.iter().any(|v| !v.is_finite()) {
        return Err("Lie 流积分末态含非有限值".to_string());
    }
    Ok(unpack12(yf))
}

/// 逐阶应用 Hamilton 流。
///
/// - ``forward=true``（QF→CM）：系数取反，阶升序
/// - ``forward=false``（CM→QF）：系数不取反，阶降序
/// - ``order < 2`` 跳过（与 Python/qiao 一致）
fn apply_lie_series(
    x0: [C64; 6],
    mut orders: Vec<(i64, WPoly)>,
    forward: bool,
    rtol: f64,
    atol: f64,
) -> Result<[C64; 6], String> {
    orders.sort_by(|a, b| {
        if forward {
            a.0.cmp(&b.0)
        } else {
            b.0.cmp(&a.0)
        }
    });
    let mut x = x0;
    for (order, mut poly) in orders {
        if order < 2 || poly.is_empty() {
            continue;
        }
        if forward {
            for c in &mut poly.coefs {
                *c = c.neg();
            }
        }
        x = integrate_lie_order(&x, &poly, rtol, atol)
            .map_err(|e| format!("order={order}: {e}"))?;
    }
    Ok(x)
}

// ---------------------------------------------------------------------------
// 公开纯函数
// ---------------------------------------------------------------------------

/// 一阶 W 的 FFI 友好表示。
pub struct WOrderInput {
    pub order: i64,
    pub exps: Vec<[i64; 6]>,
    pub coefs_re: Vec<f64>,
    pub coefs_im: Vec<f64>,
}

fn to_wpoly(input: &WOrderInput) -> Result<WPoly, String> {
    let n = input.exps.len();
    if input.coefs_re.len() != n || input.coefs_im.len() != n {
        return Err(format!(
            "order={} 的 exps/coefs 长度不一致：exps={}, re={}, im={}",
            input.order,
            n,
            input.coefs_re.len(),
            input.coefs_im.len()
        ));
    }
    let coefs = input
        .coefs_re
        .iter()
        .zip(input.coefs_im.iter())
        .map(|(&re, &im)| C64::new(re, im))
        .collect();
    Ok(WPoly {
        exps: input.exps.clone(),
        coefs,
    })
}

fn parse_x6(x: &[f64]) -> Result<[f64; 6], String> {
    if x.len() != 6 {
        return Err(format!("状态须为 6 维，得到 {}", x.len()));
    }
    Ok([x[0], x[1], x[2], x[3], x[4], x[5]])
}

/// QF → CM。
pub fn qf_to_cm_impl(
    x_qf: &[f64],
    w_orders: &[WOrderInput],
    rtol: f64,
    atol: f64,
) -> Result<[f64; 6], String> {
    let x = parse_x6(x_qf)?;
    let x_im = re_to_im(&x);
    let orders: Result<Vec<_>, _> = w_orders
        .iter()
        .map(|w| to_wpoly(w).map(|p| (w.order, p)))
        .collect();
    let x_out = apply_lie_series(x_im, orders?, true, rtol, atol)?;
    Ok(im_to_re(&x_out))
}

/// CM → QF。
pub fn cm_to_qf_impl(
    x_cm: &[f64],
    w_orders: &[WOrderInput],
    rtol: f64,
    atol: f64,
) -> Result<[f64; 6], String> {
    let x = parse_x6(x_cm)?;
    let x_im = re_to_im(&x);
    let orders: Result<Vec<_>, _> = w_orders
        .iter()
        .map(|w| to_wpoly(w).map(|p| (w.order, p)))
        .collect();
    let x_out = apply_lie_series(x_im, orders?, false, rtol, atol)?;
    Ok(im_to_re(&x_out))
}

// ---------------------------------------------------------------------------
// PyO3 绑定
// ---------------------------------------------------------------------------

/// Python 侧一阶 W：``(order, exps, coefs_re, coefs_im)``。
type WSeriesOrderPy = (i64, Vec<Vec<i64>>, Vec<f64>, Vec<f64>);

/// 解析 Python 侧 ``W_series``：
/// ``[(order, exps: [[i64;6],...], coefs_re: [f64], coefs_im: [f64]), ...]``
fn parse_w_series_py(w_series: Vec<WSeriesOrderPy>) -> PyResult<Vec<WOrderInput>> {
    let mut out = Vec::with_capacity(w_series.len());
    for (order, exps_v, cre, cim) in w_series {
        let mut exps = Vec::with_capacity(exps_v.len());
        for row in exps_v {
            if row.len() != 6 {
                return Err(PyValueError::new_err(format!(
                    "W 幂次每行须 6 列，order={order} 得到 {}",
                    row.len()
                )));
            }
            exps.push([row[0], row[1], row[2], row[3], row[4], row[5]]);
        }
        out.push(WOrderInput {
            order,
            exps,
            coefs_re: cre,
            coefs_im: cim,
        });
    }
    Ok(out)
}

/// quasi-Floquet → 中心流形（高阶 Lie 级数，12 实维 DOP853）。
///
/// Args:
///     x_qf: 长度 6 的实 QF 状态
///     w_series: ``[(order, exps, coefs_re, coefs_im), ...]``
///     rtol / atol: ODE 容差（默认与 Python 一致 1e-11 / 1e-13）
#[pyfunction]
#[pyo3(signature = (x_qf, w_series, rtol=None, atol=None))]
pub fn qf_to_cm_py(
    x_qf: Vec<f64>,
    w_series: Vec<WSeriesOrderPy>,
    rtol: Option<f64>,
    atol: Option<f64>,
) -> PyResult<Vec<f64>> {
    let rtol = rtol.unwrap_or(1e-11);
    let atol = atol.unwrap_or(1e-13);
    if rtol <= 0.0 || atol <= 0.0 {
        return Err(PyValueError::new_err("rtol 与 atol 必须为正"));
    }
    let orders = parse_w_series_py(w_series)?;
    let out = qf_to_cm_impl(&x_qf, &orders, rtol, atol)
        .map_err(|e| PyRuntimeError::new_err(format!("QF→CM Lie 级数积分失败：{e}")))?;
    Ok(out.to_vec())
}

/// 中心流形 → quasi-Floquet（高阶 Lie 级数反向）。
#[pyfunction]
#[pyo3(signature = (x_cm, w_series, rtol=None, atol=None))]
pub fn cm_to_qf_py(
    x_cm: Vec<f64>,
    w_series: Vec<WSeriesOrderPy>,
    rtol: Option<f64>,
    atol: Option<f64>,
) -> PyResult<Vec<f64>> {
    let rtol = rtol.unwrap_or(1e-11);
    let atol = atol.unwrap_or(1e-13);
    if rtol <= 0.0 || atol <= 0.0 {
        return Err(PyValueError::new_err("rtol 与 atol 必须为正"));
    }
    let orders = parse_w_series_py(w_series)?;
    let out = cm_to_qf_impl(&x_cm, &orders, rtol, atol)
        .map_err(|e| PyRuntimeError::new_err(format!("CM→QF Lie 级数积分失败：{e}")))?;
    Ok(out.to_vec())
}

// ---------------------------------------------------------------------------
// 单元测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d_inv_is_left_inverse() {
        let d = d_matrix();
        let inv = d_inv_matrix();
        // inv · D ≈ I
        for (i, inv_row) in inv.iter().enumerate() {
            #[allow(clippy::needless_range_loop)] // j 同时用于索引与 i==j 判别，迭代器化反而绕
            for j in 0..6 {
                let mut acc = C64::ZERO;
                for k in 0..6 {
                    acc = acc.add(inv_row[k].mul(d[k][j]));
                }
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (acc.re - expect).abs() < 1e-14 && acc.im.abs() < 1e-14,
                    "({i},{j}) = {}+{}i",
                    acc.re,
                    acc.im
                );
            }
        }
    }

    #[test]
    fn re_im_roundtrip() {
        let x = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6];
        let y = im_to_re(&re_to_im(&x));
        for i in 0..6 {
            assert!((y[i] - x[i]).abs() < 1e-14);
        }
    }

    #[test]
    fn pow_zero_to_zero_is_one() {
        let z = C64::ZERO.powi_nonneg(0);
        assert_eq!(z.re, 1.0);
        assert_eq!(z.im, 0.0);
    }

    #[test]
    fn hamilton_rhs_linear_p_term() {
        // W = c · p1 → dX/dt: dq1/dt = c, 其余 0
        let poly = WPoly {
            exps: vec![[0, 0, 0, 1, 0, 0]],
            coefs: vec![C64::new(0.7, -0.4)],
        };
        let x = [C64::ZERO; 6];
        let dx = hamilton_flow_rhs(&x, &poly);
        assert!((dx[0].re - 0.7).abs() < 1e-14);
        assert!((dx[0].im + 0.4).abs() < 1e-14);
        for dx_v in &dx[1..] {
            assert!(dx_v.re.abs() < 1e-14 && dx_v.im.abs() < 1e-14);
        }
    }

    #[test]
    fn empty_w_is_identity() {
        let x = [1e-3, -2e-3, 3e-3, 4e-3, -5e-3, 6e-3];
        let out = qf_to_cm_impl(&x, &[], 1e-11, 1e-13).unwrap();
        for i in 0..6 {
            assert!((out[i] - x[i]).abs() < 1e-12, "i={i}");
        }
        let back = cm_to_qf_impl(&out, &[], 1e-11, 1e-13).unwrap();
        for i in 0..6 {
            assert!((back[i] - x[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn multi_order_roundtrip() {
        // 两阶小系数 W：往返应在容差内
        let w = vec![
            WOrderInput {
                order: 3,
                exps: vec![[2, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0]],
                coefs_re: vec![1e-4, -2e-4],
                coefs_im: vec![3e-5, 1e-5],
            },
            WOrderInput {
                order: 4,
                exps: vec![[1, 1, 0, 1, 1, 0]],
                coefs_re: vec![5e-5],
                coefs_im: vec![-2e-5],
            },
        ];
        let x = [1e-3, -1.5e-3, 0.8e-3, 1.2e-3, -0.7e-3, 0.5e-3];
        let cm = qf_to_cm_impl(&x, &w, 1e-11, 1e-13).unwrap();
        let back = cm_to_qf_impl(&cm, &w, 1e-11, 1e-13).unwrap();
        for i in 0..6 {
            assert!(
                (back[i] - x[i]).abs() < 1e-8,
                "i={i}: back={} x={}",
                back[i],
                x[i]
            );
        }
    }

    #[test]
    fn complex_oscillator_via_12_real() {
        // ż = i z，z(0)=1 → z(t)=e^{it}；用 12 维中的前 1 复维验证积分能力
        let y0 = vec![1.0, 0.0]; // re, im of scalar complex
        let t_eval: Vec<f64> = (0..=10).map(|k| k as f64 * 0.1).collect();
        let rhs = |_t: f64, y: &[f64]| -> Result<Vec<f64>, String> {
            // d(re)/dt = -im, d(im)/dt = re  （× i）
            Ok(vec![-y[1], y[0]])
        };
        let sol = solve_ivp_impl(
            &DOP853,
            rhs,
            (0.0, 1.0),
            &y0,
            &t_eval,
            1e-12,
            1e-14,
            f64::INFINITY,
            MAX_ADAPTIVE_STEPS,
            None,
        );
        assert_eq!(sol.len(), t_eval.len());
        for (i, &t) in t_eval.iter().enumerate() {
            let re_ex = t.cos();
            let im_ex = t.sin();
            assert!((sol[i][0] - re_ex).abs() < 1e-10, "t={t} re");
            assert!((sol[i][1] - im_ex).abs() < 1e-10, "t={t} im");
        }
    }
}
