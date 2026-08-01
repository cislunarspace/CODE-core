//! normal_form: 标准形计算的加速内核（H→QF 投影的数值多项式展开）。
//!
//! 对应 qiao ``Code09`` 的 ``X = B·Y`` 线性替换：把平动点偏移坐标下的
//! Hamiltonian 单项式逐项展开为 QF 坐标多项式，逐时刻代入 ``B(t)`` 数值。
//! Python 侧 ``e2m2e/algorithms/normal_form/qf_projection.py`` 在
//! ``e2m2e._integrators`` 可用时调用本模块（数值展开，O(ms)）；符号实现
//! （sympy）保留作回退（星历路径的时间序列系数等）。
//!
//! 展开算法：对每个采样时刻 ``t``、每个输入单项式 ``∏ x_i^{n_i}``，
//! 逐变量展开 ``(Σ_j B[i,j]·y_j)^{n_i}``（multinomial 组合枚举），
//! 合并同幂次项。不依赖符号运算，系数为 ``f64``。

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::{BTreeSet, HashMap};

/// 6 维幂次向量（q1, q2, q3, p1, p2, p3）。
type Pow = [i64; 6];

/// 组合数 C(n, k)（n ≤ 20 的表驱动）。
fn binom(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut r: usize = 1;
    for i in 0..k {
        r = r * (n - i) / (i + 1);
    }
    r
}

/// 把 ``(Σ_j m_j·y_j)^n`` 的展开乘进 ``terms``（合并同幂次）。
///
/// ``row`` 是变换矩阵一行的非零 ``(列号 j, 系数 m_j)`` 列表；``terms``
/// 是当前部分展开表 ``{pow: coef}``。枚举 ``n`` 个因子分配到 ``row``
/// 各列的 multinomial 组合（``C(n; s_0..s_{k-1})·∏ m_j^{s_j}``）。
fn expand_row(terms: &mut HashMap<Pow, f64>, row: &[(usize, f64)], n: usize) {
    let k = row.len();
    if k == 0 || n == 0 {
        return;
    }
    if k == 1 {
        let (j, m) = row[0];
        let factor = m.powi(n as i32);
        let old = std::mem::take(terms);
        for (mut p, c) in old {
            p[j] += n as i64;
            terms.insert(p, c * factor);
        }
        return;
    }
    // 预计算每列幂次 m^s（s = 0..=n）
    let pow_tab: Vec<Vec<f64>> = row
        .iter()
        .map(|(_, m)| {
            let mut v = Vec::with_capacity(n + 1);
            v.push(1.0);
            for s in 1..=n {
                v.push(v[s - 1] * m);
            }
            v
        })
        .collect();
    // 递归枚举组合 (s_0, ..., s_{k-1})，Σ = n；acc 累积 multinomial
    let mut combos: Vec<(Vec<usize>, f64)> = Vec::new();
    let mut combo = vec![0usize; k];
    enumerate_combos(0, n, k, &mut combo, 1.0, &mut combos);
    let old = std::mem::take(terms);
    let mut new_terms: HashMap<Pow, f64> = HashMap::with_capacity(old.len() * combos.len());
    for (p, c) in old {
        for (combo, multinom) in &combos {
            let mut np = p;
            let mut factor = *multinom;
            for (idx, s) in combo.iter().enumerate() {
                np[row[idx].0] += *s as i64;
                factor *= pow_tab[idx][*s];
            }
            let entry = new_terms.entry(np).or_insert(0.0);
            *entry += c * factor;
        }
    }
    *terms = new_terms;
}

/// 递归枚举 ``n`` 个球分到 ``k`` 个箱子的全部组合（组合数 ≤ 3003）。
fn enumerate_combos(
    depth: usize,
    remaining: usize,
    k: usize,
    combo: &mut Vec<usize>,
    acc: f64,
    out: &mut Vec<(Vec<usize>, f64)>,
) {
    if depth == k - 1 {
        combo[depth] = remaining;
        out.push((combo.clone(), acc));
        return;
    }
    for s in 0..=remaining {
        combo[depth] = s;
        let a = acc * binom(remaining, s) as f64;
        enumerate_combos(depth + 1, remaining - s, k, combo, a, out);
    }
}

/// H→QF 投影：``X = B·Y`` 单项式替换展开（对应 qiao ``Code09``）。
///
/// Args:
///     pows: ``(N_in, 6)`` 输入单项式的幂次（平动点偏移坐标）。
///     coefs: ``(N_in,)`` 输入单项式的标量系数（CR3BP 自治，常数）。
///     b_seq: ``(M, 36)`` 每个采样时刻的 6×6 变换矩阵 ``B(t)`` 展平。
///
/// Returns:
///     ``(out_pows, out_coefs)``：``out_pows`` 为 ``(K, 6)`` 展开后幂次
///     并集（排序），``out_coefs`` 为 ``(M, K)`` 逐时刻系数矩阵。
#[pyfunction]
#[pyo3(signature = (pows, coefs, b_seq))]
pub fn project_hamiltonian_qf_py(
    pows: Vec<Vec<i64>>,
    coefs: Vec<f64>,
    b_seq: Vec<Vec<f64>>,
) -> PyResult<(Vec<Vec<i64>>, Vec<Vec<f64>>)> {
    let n_in = pows.len();
    if n_in != coefs.len() {
        return Err(PyValueError::new_err(format!(
            "pows 与 coefs 长度不一致：{} vs {}",
            n_in,
            coefs.len()
        )));
    }
    for p in &pows {
        if p.len() != 6 {
            return Err(PyValueError::new_err(format!(
                "pows 每行必须 6 列，得到 {}",
                p.len()
            )));
        }
    }
    let m = b_seq.len();
    if m == 0 {
        return Ok((vec![], vec![]));
    }
    for b in &b_seq {
        if b.len() != 36 {
            return Err(PyValueError::new_err(format!(
                "b_seq 每行必须 36 列，得到 {}",
                b.len()
            )));
        }
    }
    let pows6: Vec<Pow> = pows
        .iter()
        .map(|p| [p[0], p[1], p[2], p[3], p[4], p[5]])
        .collect();

    // 逐时刻展开
    let mut pow_set: BTreeSet<Pow> = BTreeSet::new();
    let mut per_t: Vec<HashMap<Pow, f64>> = Vec::with_capacity(m);
    for b in &b_seq {
        let b6: [[f64; 6]; 6] = {
            let mut m6 = [[0.0f64; 6]; 6];
            for i in 0..6 {
                for j in 0..6 {
                    m6[i][j] = b[i * 6 + j];
                }
            }
            m6
        };
        let mut out: HashMap<Pow, f64> = HashMap::new();
        for (pow, coef) in pows6.iter().zip(coefs.iter()) {
            let mut terms: HashMap<Pow, f64> = HashMap::new();
            terms.insert([0i64; 6], *coef);
            for i in 0..6 {
                let ni = pow[i];
                if ni == 0 {
                    continue;
                }
                let row: Vec<(usize, f64)> = (0..6)
                    .filter(|j| b6[i][*j].abs() > 1e-14)
                    .map(|j| (j, b6[i][j]))
                    .collect();
                if row.is_empty() {
                    terms.clear();
                    break;
                }
                expand_row(&mut terms, &row, ni as usize);
            }
            for (p, c) in terms {
                *out.entry(p).or_insert(0.0) += c;
            }
        }
        pow_set.extend(out.keys().cloned());
        per_t.push(out);
    }

    let out_pows: Vec<Pow> = pow_set.into_iter().collect();
    let mut out_coefs: Vec<Vec<f64>> = Vec::with_capacity(m);
    for t_map in &per_t {
        out_coefs.push(
            out_pows
                .iter()
                .map(|p| t_map.get(p).copied().unwrap_or(0.0))
                .collect(),
        );
    }
    Ok((out_pows.iter().map(|p| p.to_vec()).collect(), out_coefs))
}

/// CR3BP Hamiltonian 构造（Jorba-Masdemont ``c_n·ρⁿ·P_n(x/ρ)`` 形式）。
///
/// ``H = ½‖p‖² + y·p_x − x·p_y − Σ_{n≥2} c_n·ρⁿ·P_n(x/ρ)``（地心会合系、
/// 平动点偏移坐标），``c_n = (-1)ⁿ/γ³·[μ + (1−μ)·rho_e_ratioⁿ⁺¹]``
/// （JM 1999 式 1）。``Q_n = ρⁿ·P_n(x/ρ)`` 用递推
/// ``Q_n = ((2n−1)/n)·x·Q_{n−1} − ((n−1)/n)·ρ²·Q_{n−2}`` 展开为
/// ``(x, y, z)`` 的多项式（纯数值，不依赖 sympy）。
///
/// Args:
///     mu: 质量比 μ。
///     gamma: 共线平动点的 γ。
///     rho_e_ratio: ``γ/(1+γ)``（L2 的 JM 展开比）。
///     max_degree: 截断阶数（≥2）。
///
/// Returns:
///     ``(pows, coefs)``：``(K, 6)`` 幂次（后 3 位动量为 0）与 ``(K,)``
///     系数。动能与科里奥利项（``½‖p‖²``、``y·p_x − x·p_y``）已包含。
#[pyfunction]
#[pyo3(signature = (mu, gamma, rho_e_ratio, max_degree))]
pub fn build_cr3bp_hamiltonian_py(
    mu: f64,
    gamma: f64,
    rho_e_ratio: f64,
    max_degree: i64,
) -> PyResult<(Vec<Vec<i64>>, Vec<f64>)> {
    let deg = max_degree;
    if deg < 2 {
        return Err(PyValueError::new_err(format!(
            "max_degree 必须 ≥ 2，得到 {deg}"
        )));
    }
    let mut terms: HashMap<Pow, f64> = HashMap::new();

    // 动能 ½‖p‖² + 科里奥利 y·p_x − x·p_y（系数与平动点无关）
    terms.insert([0, 0, 0, 2, 0, 0], 0.5);
    terms.insert([0, 0, 0, 0, 2, 0], 0.5);
    terms.insert([0, 0, 0, 0, 0, 2], 0.5);
    terms.insert([0, 1, 0, 1, 0, 0], 1.0);
    terms.insert([1, 0, 0, 0, 1, 0], -1.0);

    // Q_n 递推：Q_0 = 1, Q_1 = x,
    // Q_n = a·x·Q_{n-1} − b·ρ²·Q_{n-2}, a=(2n−1)/n, b=(n−1)/n
    let mut q: Vec<HashMap<[i64; 3], f64>> = Vec::with_capacity(deg as usize + 1);
    let mut q0: HashMap<[i64; 3], f64> = HashMap::new();
    q0.insert([0, 0, 0], 1.0);
    q.push(q0);
    let mut q1: HashMap<[i64; 3], f64> = HashMap::new();
    q1.insert([1, 0, 0], 1.0);
    q.push(q1);

    for n in 2..=deg {
        let a = (2 * n - 1) as f64 / n as f64;
        let b = (n - 1) as f64 / n as f64;
        let mut qn: HashMap<[i64; 3], f64> = HashMap::new();
        // a·x·Q_{n-1}
        for (p, c) in &q[n as usize - 1] {
            let np = [p[0] + 1, p[1], p[2]];
            *qn.entry(np).or_insert(0.0) += a * c;
        }
        // −b·ρ²·Q_{n-2}, ρ² = x²+y²+z²
        for (p, c) in &q[n as usize - 2] {
            for (dp, dc) in [([2, 0, 0], 1.0), ([0, 2, 0], 1.0), ([0, 0, 2], 1.0)] {
                let np = [p[0] + dp[0], p[1] + dp[1], p[2] + dp[2]];
                *qn.entry(np).or_insert(0.0) -= b * dc * c;
            }
        }
        qn.retain(|_, v| v.abs() > 1e-15);
        q.push(qn);
    }

    // H_grav = −Σ c_n·Q_n, c_n = (−1)ⁿ/γ³·[μ + (1−μ)·rho_e_ratioⁿ⁺¹]
    for n in 2..=deg {
        let cn = ((-1.0f64).powi(n as i32) / gamma.powi(3))
            * (mu + (1.0 - mu) * rho_e_ratio.powi(n as i32 + 1));
        for (p, c) in &q[n as usize] {
            let full = [p[0], p[1], p[2], 0, 0, 0];
            *terms.entry(full).or_insert(0.0) += -cn * c;
        }
    }
    terms.retain(|_, v| v.abs() > 1e-15);

    let mut pows: Vec<Vec<i64>> = Vec::with_capacity(terms.len());
    let mut coefs: Vec<f64> = Vec::with_capacity(terms.len());
    for (p, c) in terms.iter() {
        pows.push(p.to_vec());
        coefs.push(*c);
    }
    Ok((pows, coefs))
}
