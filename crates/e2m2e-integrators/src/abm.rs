//! Adams-Bashforth-Moulton 4 步预测-校正（PECE）。
//!
//! 4 阶方法：Adams-Bashforth 预测器 + Adams-Moulton 校正器，
//! 附带 Milne 误差估计 `|19/270·(c−p)|`。固定步长 —— 历史缓冲存储等间距 `h` 的导数采样；
//! 改变 `h` 需要重新初始化历史缓冲（Python 侧见 `initialize_abm_history`）。
//!
//! # 来源
//! 权重逐字转录自 GMAT R2026a
//! `src/base/propagator/AdamsBashforthMoulton.cpp::SetWeights()`（第 146 行），
//! Milne 因子 `eeFactor = 19/270` 来自第 70 行。

/// Adams-Bashforth 4 步预测器权重，由旧到新
/// (`history[0]·(−9/24) + … + history[3]·(55/24)`)。
const PWEIGHTS: [f64; 4] = [-9.0 / 24.0, 37.0 / 24.0, -59.0 / 24.0, 55.0 / 24.0];

/// Adams-Moulton 4 步校正器权重。`CWEIGHTS[0..2]` 乘 `history[1..3]`；
/// `CWEIGHTS[3]` 乘新求得的 `f(predictor)`。
const CWEIGHTS: [f64; 4] = [1.0 / 24.0, -5.0 / 24.0, 19.0 / 24.0, 9.0 / 24.0];

/// 4 阶 ABM 的 Milne 误差因子（GMAT `eeFactor`）。
pub const EE_FACTOR: f64 = 19.0 / 270.0;

/// 方法步数（4 步 → 历史缓冲保存 4 个导数采样）。
pub const ABM_STEPS: usize = 4;

/// 用于步长建议的嵌入阶数（Milne 估计按此阶数缩放）。
pub const ABM_EMBEDDED_ORDER: usize = 4;

/// 一次 ABM 4 阶预测-校正步（PECE）。
///
/// `history` 必须恰好保存 [`ABM_STEPS`] 个导数采样
/// `[f_{n-3}, f_{n-2}, f_{n-1}, f_n]`（由旧到新），每个长度与 `y.len()` 相同，
/// 等间距 `h`。返回 `(corrector, milne_error, rolled_history)`，
/// 其中 `rolled_history = [f_{n-2}, f_{n-1}, f_n, f_{n+1}]`，
/// `f_{n+1} = f(t+h, corrector)`。
pub fn abm_step<F, E>(
    t: f64,
    y: &[f64],
    h: f64,
    history: &[Vec<f64>],
    f: F,
) -> Result<(Vec<f64>, f64, Vec<Vec<f64>>), E>
where
    F: Fn(f64, &[f64]) -> Result<Vec<f64>, E>,
{
    let n = y.len();
    debug_assert_eq!(history.len(), ABM_STEPS);

    // Predict (Adams-Bashforth): y_p = y + h · Σ pweights[i]·f_{n-3+i}
    let mut predict = y.to_vec();
    for i in 0..ABM_STEPS {
        for l in 0..n {
            predict[l] += h * PWEIGHTS[i] * history[i][l];
        }
    }

    // Evaluate f at the predictor (PECE: first Evaluate).
    let f_pred = f(t + h, &predict)?;

    // Correct (Adams-Moulton): y_c = y + h·(cweights[3]·f_pred + Σ_{i=1..3} cweights[i-1]·history[i])
    let mut correct = y.to_vec();
    for l in 0..n {
        correct[l] += h * CWEIGHTS[3] * f_pred[l];
    }
    for i in 1..ABM_STEPS {
        for l in 0..n {
            correct[l] += h * CWEIGHTS[i - 1] * history[i][l];
        }
    }

    // Milne error estimate.
    let error = correct
        .iter()
        .zip(predict.iter())
        .map(|(c, p)| (EE_FACTOR * (c - p)).powi(2))
        .sum::<f64>()
        .sqrt();

    // Evaluate f at the corrector (PECE: second Evaluate) and roll history.
    let f_next = f(t + h, &correct)?;
    let mut new_history = Vec::with_capacity(ABM_STEPS);
    new_history.push(history[1].clone());
    new_history.push(history[2].clone());
    new_history.push(history[3].clone());
    new_history.push(f_next);

    Ok((correct, error, new_history))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predictor_weights_sum_to_one() {
        // Convergence: a consistent linear multistep method has Σ weights = 1.
        let s: f64 = PWEIGHTS.iter().sum();
        assert!((s - 1.0).abs() < 1e-15, "∑pweights = {s} != 1");
    }

    #[test]
    fn corrector_weights_sum_to_one() {
        let s: f64 = CWEIGHTS.iter().sum();
        assert!((s - 1.0).abs() < 1e-15, "∑cweights = {s} != 1");
    }

    #[test]
    fn harmonic_oscillator_one_step() {
        // y'' = -y as [y, v]: y' = v, v' = -y. y(0)=[1,0], f(t)=[-sin t, -cos t].
        let f = |_t: f64, s: &[f64]| -> Result<Vec<f64>, std::convert::Infallible> {
            Ok(vec![s[1], -s[0]])
        };

        let h: f64 = 1e-3;
        // Exact history at t = -3h, -2h, -h, 0. f(t) = [-sin t, -cos t], so
        // f(-kh) = [sin(kh), -cos(kh)].
        let history: Vec<Vec<f64>> = vec![
            vec![(3.0 * h).sin(), -(3.0 * h).cos()],
            vec![(2.0 * h).sin(), -(2.0 * h).cos()],
            vec![h.sin(), -h.cos()],
            vec![0.0, -1.0],
        ];

        let (y1, error, new_history) = abm_step(0.0, &[1.0, 0.0], h, &history, f).unwrap();

        // Exact solution at t = h: [cos h, -sin h].
        let exact = vec![h.cos(), -h.sin()];
        let num_err = y1
            .iter()
            .zip(exact.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        assert!(num_err < 1e-10, "numerical error {num_err} too large");
        assert!(error < 1e-10, "Milne error {error} too large");
        assert_eq!(new_history.len(), ABM_STEPS);
        // Oldest sample (f at -3h) must have been dropped.
        assert!((new_history[0][1] - (-(2.0 * h).cos())).abs() < 1e-12);
    }
}
