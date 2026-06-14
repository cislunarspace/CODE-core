//! Cowell (Störmer-Cowell) 8th-order PECE for the second-order ODE x'' = a(t, x).
//!
//! Double-integration multistep: integrates position directly from acceleration
//! samples via the Störmer (explicit) predictor + Cowell (implicit) corrector,
//! avoiding the first-order velocity equation. This is the form that pays off
//! for orbital mechanics (high-order on position).
//!
//! # Order and back-difference form
//! 8th-order non-summed Störmer-Cowell in backward-difference (BD) form:
//! - predictor: `x_{n+1} = 2 x_n − x_{n-1} + h²·Σ_{j=0}^{7} λ_j·∇ʲa_n`
//! - corrector: `x_{n+1} = 2 x_n − x_{n-1} + h²·Σ_{j=0}^{7} q_j·∇ʲa_{n+1}`
//!
//! **Why 8 terms, not 7?** The corrector's global order is fixed by its leading
//! truncation term `q_{k+1}·∇^{k+1}a` (LTE `O(h^{k+3})` → global order `k+1`).
//! For a method truncating at index `k`, order = `k+1` *unless* the next
//! coefficient is zero. The classic 4th-order Störmer-Cowell truncates at k=2
//! (3 terms) yet reaches order 4 because **q_3 = 0** gives a free skip — the
//! leading residual becomes q_4·∇⁴a. That lucky zero does **not** recur: q_7 =
//! −19/6048 ≠ 0, so a 7-term method (k=6) is only order 7. Reaching order 8
//! requires truncating at k=7 (8 terms), making the leading residual q_8·∇⁸a
//! (LTE O(h¹⁰), global order 8). Verified empirically by the convergence test
//! below (and by a term-sweep: 5→5, 6→6, 7→7, 8→8).
//!
//! # History
//! `history` = `[x_{n-1}, x_n, a_{n-7}, a_{n-6}, ..., a_n]` — 2 position samples
//! followed by 8 acceleration samples (oldest first), 10 vectors of the position
//! dimension. This mixed layout is why Cowell cannot share `multistep_step`'s
//! history wire (pure derivative samples): a deviation from #107's original
//! "reuse multistep_step" note, recorded in the issue.
//!
//! # Limitations
//! - RHS is `a(t, x)` (acceleration depending on position). Velocity-dependent
//!   forces (atmospheric drag) need a different formulation.
//! - Output is position only (`x_new`); recover velocity by finite differences.
//! - Fixed step; changing `h` requires re-initialising the history.
//!
//! # Source
//! Coefficients transcribed from Berry, *A Variable-Step Double-Integration
//! Multi-Step Integrator*, Virginia Tech PhD thesis, 2004, §2.3.5:
//! - corrector `q_i` — eq 2.56 (`L²` expansion, `q_i = Σ_k c_k c_{i-k}` where
//!   `c` are the Adams-Moulton coefficients of eq 2.39);
//! - predictor `λ_i` — eq 2.60 (`EL²` expansion), satisfying `λ_i = Σ_{k≤i} q_k`.
//! Cross-checked against Montenbruck & Gill, *Satellite Orbits* §3.3, and
//! Henrici, *Discrete Variable Methods in ODEs*. The 4th-order truncation
//! (λ_0..λ_2 / q_0..q_2) expands to the textbook ordinate weights
//! predictor [13/12, −2/12, 1/12] and corrector [1/12, 10/12, 1/12].
//!
//! ⚠️ Berry's PDF (web extraction) misreads several `q_i` as 0 — q_5 (actually
//! −1/240) and q_7 (actually −19/6048). All q_i here were recomputed via the
//! convolution `q_i = Σ_k c_k c_{i-k}` and cross-locked against the λ table by
//! the consistency test `λ_i = Σ_{k≤i} q_k` below.

/// Störmer (explicit) predictor backward-difference coefficients λ_0..λ_7.
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

/// Cowell (implicit) corrector backward-difference coefficients q_0..q_7.
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

/// Number of acceleration backpoints the method consumes per step.
pub const COWELL_N_ACCEL: usize = 8;
/// Total history length: 2 position samples + 8 acceleration samples.
pub const COWELL_HISTORY_LEN: usize = 10;
/// Order used for step-size suggestion (Milne estimate scales at this order).
pub const COWELL_EMBEDDED_ORDER: usize = 8;

/// `j`-th backward difference `∇ʲ` at the newest sample.
///
/// `samples` are oldest-first; returns `Σ_{m=0}^{j} C(j,m)·(−1)^m·samples[k-m]`
/// where `k = samples.len() − 1` is the newest index. The running binomial
/// `C(j,m+1) = C(j,m)·(j−m)/(m+1)` avoids recomputing factorials.
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

/// One Störmer-Cowell 8th-order PECE step.
///
/// `history` = `[x_{n-1}, x_n, a_{n-7}, ..., a_n]` (10 vectors of the position
/// dimension). `accel` evaluates `a(t, x)`. Returns `(x_{n+1}, milne_error,
/// new_history)` where `new_history = [x_n, x_{n+1}, a_{n-6}, ..., a_n, a_{n+1}]`
/// and the error is `‖x_corrector − x_predictor‖` (Milne-style local estimate).
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
