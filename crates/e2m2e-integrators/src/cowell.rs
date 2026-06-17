//! Cowell (Störmer-Cowell) 8 阶 PECE，用于二阶 ODE x'' = a(t, x)。
//!
//! 双重积分多步：通过 Störmer（显式）预测器 + Cowell（隐式）校正器，
//! 直接由加速度采样积分位置，避免一阶速度方程。这种形式在轨道力学中收益明显
//!（位置上的高阶精度）。
//!
//! # 阶数与后向差分形式
//! 8 阶非求和 Störmer-Cowell，后向差分（BD）形式：
//! - 预测器：`x_{n+1} = 2 x_n − x_{n-1} + h²·Σ_{j=0}^{7} λ_j·∇ʲa_n`
//! - 校正器：`x_{n+1} = 2 x_n − x_{n-1} + h²·Σ_{j=0}^{7} q_j·∇ʲa_{n+1}`
//!
//! **为何 8 项而非 7 项？** 校正器的全局阶数由其首项截断项 `q_{k+1}·∇^{k+1}a` 决定
//!（局部截断误差 LTE `O(h^{k+3})` → 全局阶 `k+1`）。经典 4 阶 Störmer-Cowell
//! 截断于 k=2（3 项），但因 **q_3 = 0** 而免费跳一级，首项残差变为 q_4·∇⁴a。
//! 这一幸运零值不再复现：q_7 = −19/6048 ≠ 0，因此 7 项方法（k=6）只有 7 阶。
//! 达到 8 阶需要截断于 k=7（8 项），使首项残差为 q_8·∇⁸a（LTE O(h¹⁰)，全局 8 阶）。
//! 下方收敛测试（以及项数扫描：5→5、6→6、7→7、8→8）已实证验证。
//!
//! # 历史缓冲
//! `history` = `[x_{n-1}, x_n, a_{n-7}, a_{n-6}, ..., a_n]` —— 2 个位置采样
//! 后接 8 个加速度采样（由旧到新），共 10 个位置维度的向量。这种混合布局
//! 是 Cowell 无法复用 `multistep_step` 历史缓冲（纯导数采样）的原因。
//!
//! # 限制
//! - 右端项为 `a(t, x)`（仅位置相关的加速度）。速度相关力（如大气阻力）需另作处理。
//! - 输出仅含位置（`x_new`）；速度需由有限差分恢复。
//! - 固定步长；改变 `h` 需重新初始化历史缓冲。
//!
//! # 来源
//! 系数转录自 Berry, *A Variable-Step Double-Integration Multi-Step Integrator*,
//! Virginia Tech PhD thesis, 2004, §2.3.5：
//! - 校正器 `q_i` —— 式 2.56（`L²` 展开，`q_i = Σ_k c_k c_{i-k}`，其中 `c`
//!   为式 2.39 的 Adams-Moulton 系数）；
//! - 预测器 `λ_i` —— 式 2.60（`EL²` 展开），满足 `λ_i = Σ_{k≤i} q_k`。
//! 与 Montenbruck & Gill, *Satellite Orbits* §3.3 及 Henrici,
//! *Discrete Variable Methods in ODEs* 交叉核对。4 阶截断（λ_0..λ_2 / q_0..q_2）
//! 展开后即为教科书坐标权重：预测器 [13/12, −2/12, 1/12]、校正器 [1/12, 10/12, 1/12]。
//!
//! ⚠️ Berry 的 PDF（网页提取）将若干 `q_i` 误读为 0 —— q_5（实际 −1/240）
//! 与 q_7（实际 −19/6048）。本文所有 q_i 均通过卷积 `q_i = Σ_k c_k c_{i-k}`
//! 重新计算，并通过一致性测试 `λ_i = Σ_{k≤i} q_k` 与 λ 表交叉锁定。

/// Störmer（显式）预测器后向差分系数 λ_0..λ_7。
const PRED_LAMBDA: [f64; 8] = [
    1.0,
    0.0,
    1.0 / 12.0,
    1.0 / 12.0,
    19.0 / 240.0,
    3.0 / 40.0,
    863.0 / 12096.0,
    275.0 / 4032.0,
];

/// Cowell（隐式）校正器后向差分系数 q_0..q_7。
const CORR_Q: [f64; 8] = [
    1.0,
    -1.0,
    1.0 / 12.0,
    0.0,
    -1.0 / 240.0,
    -1.0 / 240.0,
    -221.0 / 60480.0,
    -19.0 / 6048.0,
];

/// 方法每步消耗的加速度回溯点数。
pub const COWELL_N_ACCEL: usize = 8;
/// 历史缓冲总长度：2 个位置采样 + 8 个加速度采样。
pub const COWELL_HISTORY_LEN: usize = 10;
/// 用于步长建议的阶数（Milne 估计按此阶数缩放）。
pub const COWELL_EMBEDDED_ORDER: usize = 8;

/// 最新采样处的 `j` 阶后向差分 `∇ʲ`。
///
/// `samples` 由旧到新；返回 `Σ_{m=0}^{j} C(j,m)·(−1)^m·samples[k−m]`，
/// 其中 `k = samples.len() − 1` 为最新索引。递推二项式
/// `C(j,m+1) = C(j,m)·(j−m)/(m+1)` 避免重复计算阶乘。
fn backward_diff(samples: &[Vec<f64>], j: usize) -> Vec<f64> {
    let k = samples.len() - 1;
    let n = samples[0].len();
    let mut result = vec![0.0; n];
    let mut binom = 1.0; // C(j, 0)
    for m in 0..=j {
        let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
        for l in 0..n {
            result[l] += sign * binom * samples[k - m][l];
        }
        binom *= (j - m) as f64 / (m + 1) as f64;
    }
    result
}

/// 一次 Störmer-Cowell 8 阶 PECE 步。
///
/// `history` = `[x_{n−1}, x_n, a_{n−7}, ..., a_n]`（10 个位置维度的向量）。
/// `accel` 计算 `a(t, x)`。返回 `(x_{n+1}, milne_error, new_history)`，
/// 其中 `new_history = [x_n, x_{n+1}, a_{n−6}, ..., a_n, a_{n+1}]`，
/// 误差为 `‖x_corrector − x_predictor‖`（Milne 风格局部估计）。
pub fn cowell_step<F, E>(
    t: f64,
    h: f64,
    history: &[Vec<f64>],
    accel: F,
) -> Result<(Vec<f64>, f64, Vec<Vec<f64>>), E>
where
    F: Fn(f64, &[f64]) -> Result<Vec<f64>, E>,
{
    debug_assert_eq!(history.len(), COWELL_HISTORY_LEN);
    let n = history[0].len();
    let x_prev = &history[0]; // x_{n-1}
    let x_n = &history[1]; // x_n
    let a_stored = &history[2..10]; // a_{n-7}..a_n (oldest first, 8 samples)

    let h2 = h * h;

    // Predictor (Störmer explicit): backward differences centred at a_n.
    let mut x_pred = vec![0.0; n];
    for l in 0..n {
        x_pred[l] = 2.0 * x_n[l] - x_prev[l];
    }
    for j in 0..COWELL_N_ACCEL {
        if PRED_LAMBDA[j] == 0.0 {
            continue;
        }
        let dj = backward_diff(a_stored, j);
        for l in 0..n {
            x_pred[l] += h2 * PRED_LAMBDA[j] * dj[l];
        }
    }

    // Evaluate a at the predictor (PECE: first Evaluate).
    let a_pred = accel(t + h, &x_pred)?;

    // Corrector (Cowell implicit): backward differences centred at a_{n+1}.
    // Corrector samples (oldest first): a_{n-6}..a_n plus the predictor a_{n+1}.
    let mut corr_samples: Vec<Vec<f64>> = a_stored[1..8].to_vec();
    corr_samples.push(a_pred);
    let mut x_corr = vec![0.0; n];
    for l in 0..n {
        x_corr[l] = 2.0 * x_n[l] - x_prev[l];
    }
    for j in 0..COWELL_N_ACCEL {
        if CORR_Q[j] == 0.0 {
            continue;
        }
        let dj = backward_diff(&corr_samples, j);
        for l in 0..n {
            x_corr[l] += h2 * CORR_Q[j] * dj[l];
        }
    }

    // Milne-style error estimate (predictor − corrector norm).
    let error = x_corr
        .iter()
        .zip(x_pred.iter())
        .map(|(c, p)| (c - p).powi(2))
        .sum::<f64>()
        .sqrt();

    // Evaluate a at the corrector (PECE: second Evaluate) and roll history.
    let a_new = accel(t + h, &x_corr)?;
    let new_history = vec![
        x_n.clone(),         // new x_{n-1} = old x_n
        x_corr.clone(),      // new x_n = x_{n+1}
        a_stored[1].clone(), // a_{n-6} (drop oldest a_{n-7})
        a_stored[2].clone(), // a_{n-5}
        a_stored[3].clone(), // a_{n-4}
        a_stored[4].clone(), // a_{n-3}
        a_stored[5].clone(), // a_{n-2}
        a_stored[6].clone(), // a_{n-1}
        a_stored[7].clone(), // a_n
        a_new,               // a_{n+1}
    ];

    Ok((x_corr, error, new_history))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lambda_is_cumulative_sum_of_q() {
        // Berry eq 2.59: λ_i = Σ_{k=0}^{i} q_k. Cross-locks both tables, so a
        // transcription error in either is caught.
        let mut acc = 0.0;
        for i in 0..COWELL_N_ACCEL {
            acc += CORR_Q[i];
            assert!(
                (acc - PRED_LAMBDA[i]).abs() < 1e-14,
                "λ_{i} = {} but Σ_{{k≤{i}}} q_k = {acc}",
                PRED_LAMBDA[i]
            );
        }
        // Constant-acceleration exactness requires the leading coefficient 1.
        assert!((CORR_Q[0] - 1.0).abs() < 1e-15);
        assert!((PRED_LAMBDA[0] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn harmonic_oscillator_one_step_is_eighth_order_accurate() {
        // x'' = -x, x(0)=1, x'(0)=0 → x(t)=cos t, a(t) = -cos t.
        let accel = |_t: f64, x: &[f64]| -> Result<Vec<f64>, std::convert::Infallible> {
            Ok(vec![-x[0]])
        };

        let h: f64 = 1e-3;
        // History at t = 0: [x(-h), x(0), a(-7h), ..., a(-h), a(0)].
        let mut history: Vec<Vec<f64>> = vec![vec![h.cos()], vec![1.0]];
        for k in (1..=7).rev() {
            history.push(vec![-((k as f64) * h).cos()]);
        }
        history.push(vec![-1.0]);

        let (x1, error, new_history) = cowell_step(0.0, h, &history, accel).unwrap();

        // Exact x(h) = cos h; 8th-order at h=1e-3 sits at roundoff.
        assert!(
            (x1[0] - h.cos()).abs() < 1e-12,
            "x(h) = {} vs cos(h) = {}",
            x1[0],
            h.cos()
        );
        assert!(error < 1e-12, "Milne error {error} too large");
        assert_eq!(new_history.len(), COWELL_HISTORY_LEN);
    }

    #[test]
    fn exponential_converges_at_eighth_order() {
        // x'' = e^t, x(0)=1, x'(0)=1 → x(t)=e^t, a(t)=e^t. Unlike the harmonic
        // oscillator (whose alternating Taylor structure cancels the method's
        // leading error terms and muddies the observed order near roundoff), the
        // exponential has same-sign derivatives, giving a clean asymptotic slope.
        // Halving h should shrink the global error by ~2^8 = 256.
        let accel = |t: f64, _x: &[f64]| -> Result<Vec<f64>, std::convert::Infallible> {
            Ok(vec![t.exp()])
        };
        let target = 1.0_f64;

        let error_at = |h: f64| -> f64 {
            // History at t=0: [x(-h), x(0), a(-7h), ..., a(-h), a(0)] = [e^-h, 1, e^-7h, ..., 1].
            let mut history: Vec<Vec<f64>> = vec![vec![(-h).exp()], vec![1.0]];
            for k in (1..=7).rev() {
                history.push(vec![(-((k as f64) * h)).exp()]);
            }
            history.push(vec![1.0]);
            let mut t = 0.0;
            let n = (target / h).round() as usize;
            for _ in 0..n {
                let (_x, _e, new_h) = cowell_step(t, h, &history, accel).unwrap();
                history = new_h;
                t += h;
            }
            (history[1][0] - target.exp()).abs()
        };

        let e1 = error_at(0.1);
        let e2 = error_at(0.05);
        let observed_order = (e1 / e2).log2();
        assert!(
            (7.5..=8.5).contains(&observed_order),
            "observed order {observed_order} not ~8 (e1={e1:e}, e2={e2:e})"
        );
    }
}
