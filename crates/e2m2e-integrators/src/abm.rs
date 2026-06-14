//! Adams-Bashforth-Moulton 4-step predictor-corrector (PECE).
//!
//! 4th-order method: Adams-Bashforth predictor + Adams-Moulton corrector,
//! with Milne error estimate `|19/270·(c−p)|`. Fixed-step — the history stores
//! derivative samples at equal spacing `h`; changing `h` requires
//! re-initialising the history (see `initialize_abm_history` on the Python side).
//!
//! # Source
//! Weights transcribed verbatim from GMAT R2026a
//! `src/base/propagator/AdamsBashforthMoulton.cpp::SetWeights()` (line 146),
//! Milne factor `eeFactor = 19/270` from line 70.

/// Adams-Bashforth 4-step predictor weights, oldest→newest
/// (`history[0]·(−9/24) + … + history[3]·(55/24)`).
const PWEIGHTS: [f64; 4] = [-9.0 / 24.0, 37.0 / 24.0, -59.0 / 24.0, 55.0 / 24.0];

/// Adams-Moulton 4-step corrector weights. `CWEIGHTS[0..2]` multiply
/// `history[1..3]`; `CWEIGHTS[3]` multiplies the newly-evaluated `f(predictor)`.
const CWEIGHTS: [f64; 4] = [1.0 / 24.0, -5.0 / 24.0, 19.0 / 24.0, 9.0 / 24.0];

/// Milne error factor for 4th-order ABM (GMAT `eeFactor`).
pub const EE_FACTOR: f64 = 19.0 / 270.0;

/// Step count of the method (4-step → history holds 4 derivative samples).
pub const ABM_STEPS: usize = 4;

/// Embedded order used for step-size suggestion (Milne estimate is O(h⁵)).
pub const ABM_EMBEDDED_ORDER: usize = 4;

/// One ABM 4th-order predictor-corrector step (PECE).
///
/// `history` must hold exactly [`ABM_STEPS`] derivative samples
/// `[f_{n-3}, f_{n-2}, f_{n-1}, f_n]` (oldest first), each of length `y.len()`,
/// at equal spacing `h`. Returns `(corrector, milne_error, rolled_history)`
/// where `rolled_history = [f_{n-2}, f_{n-1}, f_n, f_{n+1}]` and
/// `f_{n+1} = f(t+h, corrector)`.
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
