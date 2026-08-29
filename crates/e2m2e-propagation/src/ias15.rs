//! IAS15：15 阶 Gauss-Radau 自适应积分器（变阶预测-校正 + 补偿求和）。
//!
//! 算法描述见 Rein & Spiegel (2015, MNRAS 446, 1424) 与 Everhart (1985)。
//! 本实现按论文公开的算法自行编写（REBOUND/ASSIST 为 GPL，不引用其代码）：
//!
//! - 状态分两段：位置/速度按二阶系统处理（x 由加速度经 `(s−u)` 权重的
//!   二重积分更新），额外一阶分量（STM 列、参数敏感列）走单积分。
//! - 加速度在 8 个左 Radau 节点上用 Newton 基插值多项式表示
//!   `a(s) = Σ G_j·P_j(s)`，`P_j(s) = ∏_{i<j}(s−s_i)`。逐步外加上一步的
//!   多项式作为预测，逐节点 Gauss-Seidel 校正（节点 k 的残差只改 `G_k`，
//!   因为 `P_j(s_k)=0` 对 `j>k` 成立），迭代至收敛即 15 阶方法。
//! - 无嵌入对：误差用最高阶项 `G_7` 估计（`G_7 ∝ h⁷`，步长指数 1/7）。
//! - 全局状态累加用 Neumaier-Kahan 补偿求和，长弧段误差增长趋近
//!   Brouwer 律 `n^(1/2)`（Rein & Spiegel 2015 §2）。
//!
//! 右端项闭包约定：输入完整状态 `y = [x(3), v(3), extra(m)]`，返回全长
//! 导数 `dy`；引擎取 `dy[3..6]` 为加速度、`dy[6..]` 为额外分量变化率，
//! `dy[0..3]` 忽略（`x' = v` 由引擎构造保证）。

use std::sync::OnceLock;

/// 左 Radau 节点（含端点 0）：勒让德多项式 `P₇+P₈` 在 `[-1,1]` 上的根
/// 映射到 `[0,1]`。与 Everhart (1985) RADAU-15 的 spacing 一致
/// （已用 numpy.polynomial.legroots 数值核对到机器精度）。
#[allow(clippy::excessive_precision)]
const S: [f64; 8] = [
    0.0,
    0.05626256053692215,
    0.18024069173689236,
    0.35262471711316963,
    0.54715362633055538,
    0.73421017721541053,
    0.88532094683909576,
    0.97752061356128750,
];

/// 预测-校正单步最大扫描次数（与 IAS15 论文一致）。
const MAX_SWEEPS: usize = 12;

/// 校正收敛阈值：节点残差相对采样量级的上限。
const CONV_TOL: f64 = 1e-16;

/// 最小步长（相对积分区间），防止步长坍缩。
const MIN_STEP_REL: f64 = 1e-12;

/// 预计算的 Newton 基系数（全部由节点 `S` 推出，编译期无硬编码积分常数）。
struct Coeff {
    /// `peval[k][j] = P_j(S[k])`：校正器中由多项式预测节点采样值。
    peval: [[f64; 8]; 8],
    /// `pval[k] = P_k(S[k])`：校正器分母；index 0 不使用。
    pval: [f64; 8],
    /// `ip[k][j] = ∫₀^{s_k} P_j(u) du`；行 1..=7 对应节点 `S[k]`，行 8 对应 `s=1`。
    ip: [[f64; 8]; 9],
    /// `iip[k][j] = ∫₀^{s_k} (s_k−u)·P_j(u) du`（二重积分核），行序同 `ip`。
    iip: [[f64; 8]; 9],
}

/// Newton 基多项式 `P_j(s) = ∏_{i<j}(s − S[i])`（连乘形式，避免单项展开相消）。
fn eval_p(j: usize, s: f64) -> f64 {
    S[..j].iter().fold(1.0, |acc, &si| acc * (s - si))
}

/// 16 点 Gauss-Legendre 节点与权重（`[-1,1]`，Newton 迭代求 P₁₆ 的根）。
///
/// 对次数 ≤ 31 的多项式积分精确到舍入；本模块被积函数最高 8 次
/// （`(s−u)·P₇(u)`），故 `ip`/`iip` 常数精确到机器精度。
fn gauss_legendre_16() -> ([f64; 16], [f64; 16]) {
    const N: usize = 16;
    let mut x = [0.0_f64; N];
    let mut w = [0.0_f64; N];
    for i in 0..N / 2 {
        // Tricomi 初值
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (N as f64 + 0.5)).cos();
        loop {
            // 递推求 P_N(z)（p1）与 P_{N-1}(z)（p2）
            let mut p1 = 1.0;
            let mut p2 = 0.0;
            for j in 1..=N {
                let p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j as f64 - 1.0) * z * p2 - (j as f64 - 1.0) * p3) / j as f64;
            }
            let pp = N as f64 * (z * p1 - p2) / (z * z - 1.0);
            let z_next = z - p1 / pp;
            if (z_next - z).abs() < 1e-15 {
                z = z_next;
                break;
            }
            z = z_next;
        }
        // 收敛后再算一次 P' 求权重
        let mut p1 = 1.0;
        let mut p2 = 0.0;
        for j in 1..=N {
            let p3 = p2;
            p2 = p1;
            p1 = ((2.0 * j as f64 - 1.0) * z * p2 - (j as f64 - 1.0) * p3) / j as f64;
        }
        let pp = N as f64 * (z * p1 - p2) / (z * z - 1.0);
        let wgt = 2.0 / ((1.0 - z * z) * pp * pp);
        x[i] = -z;
        x[N - 1 - i] = z;
        w[i] = wgt;
        w[N - 1 - i] = wgt;
    }
    (x, w)
}

fn build_coeff() -> Coeff {
    let (gx, gw) = gauss_legendre_16();
    let mut c = Coeff {
        peval: [[0.0; 8]; 8],
        pval: [0.0; 8],
        ip: [[0.0; 8]; 9],
        iip: [[0.0; 8]; 9],
    };
    for (k, (&sk, peval_row)) in S.iter().zip(c.peval.iter_mut()).enumerate() {
        for (j, cell) in peval_row.iter_mut().enumerate() {
            *cell = eval_p(j, sk);
        }
        c.pval[k] = eval_p(k, sk);
    }
    // 行 1..=7 积分到节点 S[k]，行 8 积分到 s=1；行 0（s=0）全零，不使用。
    for (row, &sk) in S.iter().enumerate().skip(1) {
        let lim = sk;
        for j in 0..8 {
            let mut ip = 0.0;
            let mut iip = 0.0;
            for q in 0..16 {
                let u = 0.5 * lim * (gx[q] + 1.0);
                let wq = 0.5 * lim * gw[q];
                let pj = eval_p(j, u);
                ip += wq * pj;
                iip += wq * (lim - u) * pj;
            }
            c.ip[row][j] = ip;
            c.iip[row][j] = iip;
        }
    }
    // 行 8：积分到 s=1（区间端点）
    let lim = 1.0_f64;
    for j in 0..8 {
        let mut ip = 0.0;
        let mut iip = 0.0;
        for q in 0..16 {
            let u = 0.5 * lim * (gx[q] + 1.0);
            let wq = 0.5 * lim * gw[q];
            let pj = eval_p(j, u);
            ip += wq * pj;
            iip += wq * (lim - u) * pj;
        }
        c.ip[8][j] = ip;
        c.iip[8][j] = iip;
    }
    c
}

fn coeff() -> &'static Coeff {
    static COEFF: OnceLock<Coeff> = OnceLock::new();
    COEFF.get_or_init(build_coeff)
}

/// Neumaier-Kahan 补偿累加：把上一步丢失的低位折进本步增量，
/// 累加后更新补偿项。长弧段全局舍入误差由此按 `n^(1/2)` 增长。
#[inline]
fn accum(y: &mut f64, comp: &mut f64, d: f64) {
    let d_eff = d + *comp;
    let sum = *y + d_eff;
    *comp = if y.abs() >= d_eff.abs() {
        (*y - sum) + d_eff
    } else {
        (d_eff - sum) + *y
    };
    *y = sum;
}

/// 步长控制：`G_7 ∝ h⁷`，故误差按 `(tol/eps)^(1/7)` 回调。
fn next_h(h: f64, eps: f64, tol: f64) -> f64 {
    if eps == 0.0 {
        return h * 10.0;
    }
    h * (0.9 * (tol / eps).powf(1.0 / 7.0)).clamp(0.1, 10.0)
}

/// 接受步的端点更新：s=1 处的单/二重积分，补偿求和累加进全局状态。
fn accept_step(
    y: &mut [f64],
    comp: &mut [f64],
    g: &[[f64; 8]],
    coeff: &Coeff,
    h: f64,
    n_extra: usize,
) {
    for i in 0..3 {
        let mut dv = 0.0;
        let mut dx = 0.0;
        for (j, &gj) in g[i].iter().enumerate() {
            dv += gj * coeff.ip[8][j];
            dx += gj * coeff.iip[8][j];
        }
        let d_x = h * y[3 + i] + h * h * dx;
        accum(&mut y[i], &mut comp[i], d_x);
        accum(&mut y[3 + i], &mut comp[3 + i], h * dv);
    }
    for c in 0..n_extra {
        let mut dq = 0.0;
        for (j, &gj) in g[3 + c].iter().enumerate() {
            dq += gj * coeff.ip[8][j];
        }
        accum(&mut y[6 + c], &mut comp[6 + c], h * dq);
    }
}

/// IAS15 传播结果。
pub struct Ias15Result {
    pub times: Vec<f64>,
    pub states: Vec<Vec<f64>>,
    pub n_steps: usize,
    pub n_rejected: usize,
}

/// IAS15 自适应传播：从 `t_span.0` 积到 `t_span.1`，在 `t_eval` 处输出。
///
/// - `tol` 是相对容差（对加速度采样量级归一），语义对齐 IAS15 论文。
/// - `breaks`：右端项不连续时刻（如推力开关机边界），步长必落在其上，
///   避免插值多项式跨过间断。须按时间升序。
/// - 支持反向积分（`t_span.1 < t_span.0`，此时 `t_eval` 须降序）。
///
/// 步长截断到输出点：输出点处状态就是该时刻的积分值，不做插值。
#[allow(clippy::too_many_arguments)]
pub fn propagate_ias15<F, E>(
    f: F,
    t_span: (f64, f64),
    t_eval: &[f64],
    y0: &[f64],
    tol: f64,
    max_step: Option<f64>,
    max_steps: Option<usize>,
    breaks: &[f64],
) -> Result<Ias15Result, E>
where
    F: Fn(f64, &[f64]) -> Result<Vec<f64>, E>,
    E: From<String>,
{
    let n = y0.len();
    if n < 6 {
        return Err(E::from(format!(
            "ias15: state length must be >= 6, got {n}"
        )));
    }
    if t_eval.is_empty() {
        return Err(E::from("ias15: t_eval must not be empty".to_string()));
    }
    let coeff = coeff();
    let (t0, t1) = t_span;
    let span = (t1 - t0).abs();
    let dir = (t1 - t0).signum();
    if dir == 0.0 {
        return Ok(Ias15Result {
            times: t_eval.to_vec(),
            states: vec![y0.to_vec(); t_eval.len()],
            n_steps: 0,
            n_rejected: 0,
        });
    }

    let n_extra = n - 6;
    let h_max = max_step.unwrap_or(f64::INFINITY);
    let s_max = max_steps.unwrap_or(500_000);

    let mut y = y0.to_vec();
    let mut comp = vec![0.0_f64; n];
    // Newton 基系数：行 0..3 为加速度三分量，行 3.. 为额外一阶分量。
    let mut g = vec![[0.0_f64; 8]; 3 + n_extra];
    let mut yk = vec![0.0_f64; n];

    let mut t = t0;
    let mut h = dir * (span / 100.0).min(h_max);

    let mut times: Vec<f64> = Vec::with_capacity(t_eval.len());
    let mut states: Vec<Vec<f64>> = Vec::with_capacity(t_eval.len());
    let mut eval_idx = 0usize;
    if (t_eval[0] - t0).abs() <= 1e-9 {
        times.push(t0);
        states.push(y.clone());
        eval_idx = 1;
    }

    let mut n_steps = 0usize;
    let mut n_rejected = 0usize;
    // 噪声地板处理：星历采样（SPICE 求值、第三体直接/间接项相消）的加速度
    // 有效光滑度约 1e-11 相对量级，7 阶均差把它放大成不随步长下降的误差
    // 估计。连续拒步 eps 不再显著下降即到达地板：把有效容差抬到地板上方
    // 接受当前步并继续（地板以下的精度对当前力模型本就不可达）。
    let mut tol_eff = tol;
    let mut best_reject_eps = f64::INFINITY;

    while (t1 - t) * dir > 0.0 && n_steps < s_max {
        n_steps += 1;

        // 本步目标时刻：最近的输出点 / 间断点 / 终点
        let mut t_next = t1;
        if eval_idx < t_eval.len() {
            let te = t_eval[eval_idx];
            if (te - t) * dir < (t_next - t) * dir {
                t_next = te;
            }
        }
        for &b in breaks {
            if (b - t) * dir > 0.0 && (b - t) * dir < (t_next - t) * dir {
                t_next = b;
            }
        }
        if (t + h - t_next) * dir > 0.0 {
            h = t_next - t;
        }
        if h.abs() > h_max {
            h = dir * h_max;
        }
        if h.abs() < MIN_STEP_REL * span && (t_next - t) * dir > 1e-9 {
            // 剩余区间大于输出记录 slack（1e-9 s）才算坍缩；落点截断的
            // 浮点残余（≤1 ulp）走一步即达，不算失败。
            return Err(E::from(format!(
                "ias15: step size collapsed below minimum after {n_steps} steps"
            )));
        }

        // 节点 0 采样并重锚定 G_0（P_j(0)=0 对 j≥1 成立，故多项式过 (0, G_0)）
        let dy0 = f(t, &y)?;
        if dy0.len() != n {
            return Err(E::from(format!(
                "ias15: rhs returned length {}, expected {n}",
                dy0.len()
            )));
        }
        let mut scale_acc = 0.0_f64;
        for i in 0..3 {
            g[i][0] = dy0[3 + i];
            scale_acc = scale_acc.max(dy0[3 + i].abs());
        }
        let mut scale_ext = vec![0.0_f64; n_extra];
        for c in 0..n_extra {
            g[3 + c][0] = dy0[6 + c];
            scale_ext[c] = dy0[6 + c].abs();
        }

        // 逐节点 Gauss-Seidel 校正扫描
        for sweep in 0..MAX_SWEEPS {
            let mut max_resid = 0.0_f64;
            for k in 1..8 {
                let sk = S[k];
                let tk = t + h * sk;
                for i in 0..3 {
                    let mut vi = 0.0;
                    let mut xi = 0.0;
                    for (j, &gj) in g[i].iter().enumerate() {
                        vi += gj * coeff.ip[k][j];
                        xi += gj * coeff.iip[k][j];
                    }
                    yk[3 + i] = y[3 + i] + h * vi;
                    yk[i] = y[i] + y[3 + i] * h * sk + h * h * xi;
                }
                for c in 0..n_extra {
                    let mut qi = 0.0;
                    for (j, &gj) in g[3 + c].iter().enumerate() {
                        qi += gj * coeff.ip[k][j];
                    }
                    yk[6 + c] = y[6 + c] + h * qi;
                }
                let dyk = f(tk, &yk)?;
                for i in 0..3 {
                    let a_new = dyk[3 + i];
                    scale_acc = scale_acc.max(a_new.abs());
                    let mut a_pred = 0.0;
                    for (j, &gj) in g[i].iter().enumerate() {
                        a_pred += gj * coeff.peval[k][j];
                    }
                    let delta = a_new - a_pred;
                    g[i][k] += delta / coeff.pval[k];
                    max_resid = max_resid.max(delta.abs() / (scale_acc + 1e-300));
                }
                for c in 0..n_extra {
                    let e_new = dyk[6 + c];
                    scale_ext[c] = scale_ext[c].max(e_new.abs());
                    let mut e_pred = 0.0;
                    for (j, &gj) in g[3 + c].iter().enumerate() {
                        e_pred += gj * coeff.peval[k][j];
                    }
                    let delta = e_new - e_pred;
                    g[3 + c][k] += delta / coeff.pval[k];
                    max_resid = max_resid.max(delta.abs() / (scale_ext[c] + 1e-300));
                }
            }
            if sweep >= 1 && max_resid <= CONV_TOL {
                break;
            }
        }

        // 误差估计：最高阶项 G_7 对*状态更新*的贡献。P₇ 在 [0,1] 上有 7 个
        // 零点，其积分权重 ip[8][7]/iip[8][7] 因相消远小于节点值 pval[7]，
        // 用积分权重标定的误差不会被节点值的噪声地板淹没（本仓 Newton 基
        // 实现与 IAS15 论文的 b 系数基只差基变换，误差语义一致）。
        // 分母取各阶项对更新贡献的最大值（量级参照，h 无关），故 eps ∝ h⁷。
        let ip87 = coeff.ip[8][7].abs();
        let iip87 = coeff.iip[8][7].abs();
        let mut eps = 0.0_f64;
        for row in g.iter() {
            let mut scale_ip = 0.0_f64;
            let mut scale_iip = 0.0_f64;
            for (j, &gj) in row.iter().enumerate() {
                scale_ip = scale_ip.max((gj * coeff.ip[8][j]).abs());
                scale_iip = scale_iip.max((gj * coeff.iip[8][j]).abs());
            }
            eps = eps.max(row[7].abs() * ip87 / (scale_ip + 1e-300));
            eps = eps.max(row[7].abs() * iip87 / (scale_iip + 1e-300));
        }

        if eps <= tol_eff {
            // 接受：端点 s=1 更新（行 8 积分常数），补偿求和累加
            accept_step(&mut y, &mut comp, &g, coeff, h, n_extra);
            t += h;

            while eval_idx < t_eval.len() && (t - t_eval[eval_idx]) * dir >= -1e-9 {
                times.push(t_eval[eval_idx]);
                states.push(y.clone());
                eval_idx += 1;
            }
            h = next_h(h, eps, tol_eff);
            best_reject_eps = f64::INFINITY;
        } else {
            // 噪声地板检测：连续拒步 eps 不再显著下降（不足 2 倍）即到达
            // 地板——把有效容差抬到地板上方，接受当前步并继续。
            let stagnated = eps > 0.5 * best_reject_eps;
            if stagnated {
                tol_eff = tol_eff.max(2.0 * best_reject_eps.min(eps));
                accept_step(&mut y, &mut comp, &g, coeff, h, n_extra);
                t += h;
                while eval_idx < t_eval.len() && (t - t_eval[eval_idx]) * dir >= -1e-9 {
                    times.push(t_eval[eval_idx]);
                    states.push(y.clone());
                    eval_idx += 1;
                }
                best_reject_eps = f64::INFINITY;
            } else {
                best_reject_eps = best_reject_eps.min(eps);
                n_rejected += 1;
                h = next_h(h, eps, tol_eff);
            }
        }
    }

    if states.len() != t_eval.len() {
        return Err(E::from(format!(
            "ias15: output length mismatch: got {} time points, expected {} (max_steps={s_max})",
            states.len(),
            t_eval.len()
        )));
    }

    Ok(Ias15Result {
        times,
        states,
        n_steps,
        n_rejected,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kepler_rhs(mu: f64) -> impl Fn(f64, &[f64]) -> Result<Vec<f64>, String> {
        move |_t, y| {
            let r = (y[0] * y[0] + y[1] * y[1] + y[2] * y[2]).sqrt();
            let f = -mu / (r * r * r);
            Ok(vec![y[3], y[4], y[5], f * y[0], f * y[1], f * y[2]])
        }
    }

    fn energy(y: &[f64], mu: f64) -> f64 {
        let r = (y[0] * y[0] + y[1] * y[1] + y[2] * y[2]).sqrt();
        let v2 = y[3] * y[3] + y[4] * y[4] + y[5] * y[5];
        0.5 * v2 - mu / r
    }

    #[test]
    fn coeff_sanity() {
        // P_0 = 1：IP(s)=s，IIP(s)=s²/2；P_1 = s：IP(s)=s²/2，IIP(s)=s³/6
        let c = coeff();
        for (row, &sk) in S.iter().enumerate().skip(1) {
            assert!((c.ip[row][0] - sk).abs() < 1e-15, "ip row {row}");
            assert!(
                (c.iip[row][0] - sk * sk / 2.0).abs() < 1e-15,
                "iip row {row}"
            );
            assert!((c.ip[row][1] - sk * sk / 2.0).abs() < 1e-15);
            assert!((c.iip[row][1] - sk * sk * sk / 6.0).abs() < 1e-15);
        }
        assert!((c.ip[8][0] - 1.0).abs() < 1e-15);
        assert!((c.iip[8][0] - 0.5).abs() < 1e-15);
        // P_j(s_k) = 0 对 j > k（校正器局部更新的依据）
        for k in 0..8 {
            for j in (k + 1)..8 {
                assert!(c.peval[k][j].abs() < 1e-15, "peval[{k}][{j}]");
            }
        }
        for k in 1..8 {
            assert!(c.pval[k] != 0.0);
        }
    }

    #[test]
    fn kepler_circular_ten_periods() {
        // 圆轨道 μ=1, r=1, v=1，周期 2π。tol=1e-13 时 10 圈末态与解析解比较。
        let f = kepler_rhs(1.0);
        let y0 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let t1 = 10.0 * 2.0 * std::f64::consts::PI;
        let res = propagate_ias15(f, (0.0, t1), &[t1], &y0, 1e-13, None, None, &[]).unwrap();
        let y = &res.states[0];
        let theta = t1;
        let exact = [
            theta.cos(),
            theta.sin(),
            0.0,
            -theta.sin(),
            theta.cos(),
            0.0,
        ];
        let err: f64 = y
            .iter()
            .zip(exact.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-9, "circular orbit endpoint error {err:e}");
        assert!((energy(y, 1.0) + 0.5).abs() < 1e-11, "energy drift");
    }

    #[test]
    fn kepler_eccentric_close_approach() {
        // e=0.9 椭圆，近拱点强加速考验步长自适应；一圈后回到远拱点初态。
        let mu = 1.0;
        let e: f64 = 0.9;
        let r_a = 1.0 + e;
        let v_a = ((1.0 - e) / (1.0 + e)).sqrt();
        let y0 = [r_a, 0.0, 0.0, 0.0, v_a, 0.0];
        let period = 2.0 * std::f64::consts::PI; // a=1 → T=2π
        let f = kepler_rhs(mu);
        let res =
            propagate_ias15(f, (0.0, period), &[period], &y0, 1e-12, None, None, &[]).unwrap();
        let y = &res.states[0];
        let err: f64 = y
            .iter()
            .zip(y0.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-7, "eccentric orbit closure error {err:e}");
        // 误差应主要来自相位（近拱点被精确解析，步数应显著少于低阶方法）
        assert!(res.n_steps < 500, "n_steps = {}", res.n_steps);
    }

    #[test]
    fn first_order_extra_component() {
        // 额外一阶分量 q' = −q（单积分路径），q(t) = e^{−t}
        let f = |_: f64, y: &[f64]| -> Result<Vec<f64>, String> {
            let r = (y[0] * y[0] + y[1] * y[1] + y[2] * y[2]).sqrt();
            let f = -1.0 / (r * r * r);
            Ok(vec![y[3], y[4], y[5], f * y[0], f * y[1], f * y[2], -y[6]])
        };
        let y0 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let t1 = 5.0;
        let res = propagate_ias15(f, (0.0, t1), &[t1], &y0, 1e-13, None, None, &[]).unwrap();
        let q = res.states[0][6];
        assert!(
            (q - (-t1).exp()).abs() < 1e-11,
            "exp decay error: {q} vs {}",
            (-t1).exp()
        );
    }

    #[test]
    fn out_and_back_reversibility() {
        // 正向积到 T 再反向积回 0（IAS15 非时间对称，残余即真实截断+舍入）。
        let y0 = [1.0, 0.0, 0.0, 0.0, 1.1, 0.2];
        let t1 = 11.3;
        let f = kepler_rhs(1.0);
        let fwd = propagate_ias15(&f, (0.0, t1), &[t1], &y0, 1e-13, None, None, &[]).unwrap();
        let y_mid = &fwd.states[0];
        let bwd = propagate_ias15(&f, (t1, 0.0), &[0.0], y_mid, 1e-13, None, None, &[]).unwrap();
        let y_back = &bwd.states[0];
        let err: f64 = y_back
            .iter()
            .zip(y0.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-8, "out-and-back residual {err:e}");
    }

    #[test]
    fn long_run_energy_growth_sublinear() {
        // 补偿求和的效果：100 圈圆轨道，能量误差应远小于朴素累加的线性增长。
        // 宽松上界：tol=1e-13 时 100 圈 |ΔE| < 1e-10。
        let f = kepler_rhs(1.0);
        let y0 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let t1 = 100.0 * 2.0 * std::f64::consts::PI;
        let res = propagate_ias15(f, (0.0, t1), &[t1], &y0, 1e-13, None, None, &[]).unwrap();
        let e = energy(&res.states[0], 1.0);
        assert!((e + 0.5).abs() < 1e-10, "long-run energy drift {}", e + 0.5);
    }
}
