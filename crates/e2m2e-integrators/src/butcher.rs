//! Generic explicit Runge-Kutta driver, parameterised over a Butcher tableau.
//!
//! All explicit embedded RK methods (PD45, PD78, RK89, ...) share the same
//! single-step structure; only the Butcher coefficients differ. This module
//! hosts the shared [`explicit_rk_step`] driver and the order-aware
//! [`suggest_next_step`] heuristic, so each concrete method only contributes a
//! [`ButcherTable`] constant.

/// A Butcher tableau for an explicit embedded Runge-Kutta method.
///
/// All coefficient slices are `&'static` — tables are compile-time constants
/// sourced from GMAT / literature. Overall dimensions are validated by
/// [`ButcherTable::new`] at construction time (at compile time when the table
/// is a `const`); per-row lower-triangular shape is covered by each method's
/// own unit tests.
pub struct ButcherTable {
    /// Number of stages `s`.
    pub stages: usize,
    /// Order `p` of the primary (higher-order) solution.
    pub order: usize,
    /// Order of the embedded (lower-order) solution used for error estimate.
    pub embedded_order: usize,
    /// Time nodes `c[i]` (length `stages`); `c[0]` is conventionally 0.
    pub c: &'static [f64],
    /// Runge-Kutta matrix rows `a[i]`; row `i` has length `i` (lower triangular).
    pub a: &'static [&'static [f64]],
    /// Weights `b` for the primary (higher-order) solution (length `stages`).
    pub b: &'static [f64],
    /// Weights `b_star` for the embedded (lower-order) solution (length `stages`).
    pub b_star: &'static [f64],
}

impl ButcherTable {
    /// Construct a tableau, validating overall dimension consistency.
    ///
    /// Intended for `const` initialisers so that a malformed table fails to
    /// compile. Asserts `c`, `b`, `b_star`, and `a` each have `stages` entries.
    /// Per-row length (`a[i].len() == i`) is enforced by each method's tests,
    /// since per-element slice indexing is not yet allowed in `const fn`.
    pub const fn new(
        stages: usize,
        order: usize,
        embedded_order: usize,
        c: &'static [f64],
        a: &'static [&'static [f64]],
        b: &'static [f64],
        b_star: &'static [f64],
    ) -> Self {
        assert!(c.len() == stages);
        assert!(b.len() == stages);
        assert!(b_star.len() == stages);
        assert!(a.len() == stages);
        Self {
            stages,
            order,
            embedded_order,
            c,
            a,
            b,
            b_star,
        }
    }
}

/// Take a single explicit Runge-Kutta step using `table`.
///
/// Returns the primary (higher-order) solution and the L2 norm of the
/// difference between the primary and embedded solutions (the local error
/// estimate used for step-size control).
pub fn explicit_rk_step<F, E>(
    table: &ButcherTable,
    t: f64,
    y: &[f64],
    h: f64,
    f: F,
) -> Result<(Vec<f64>, f64), E>
where
    F: Fn(f64, &[f64]) -> Result<Vec<f64>, E>,
{
    let n = y.len();
    let s = table.stages;

    debug_assert_eq!(table.c.len(), s);
    debug_assert_eq!(table.b.len(), s);
    debug_assert_eq!(table.b_star.len(), s);
    debug_assert_eq!(table.a.len(), s);

    let mut k = vec![vec![0.0; n]; s];
    k[0] = f(t, y)?;

    for i in 1..s {
        let ti = t + h * table.c[i];
        let mut yi = y.to_vec();
        let ai = table.a[i];
        for (j, &aij) in ai.iter().enumerate() {
            for l in 0..n {
                yi[l] += h * aij * k[j][l];
            }
        }
        k[i] = f(ti, &yi)?;
    }

    let mut y_high = vec![0.0; n];
    let mut y_low = vec![0.0; n];
    for l in 0..n {
        for i in 0..s {
            y_high[l] += table.b[i] * k[i][l];
            y_low[l] += table.b_star[i] * k[i][l];
        }
        y_high[l] = y[l] + h * y_high[l];
        y_low[l] = y[l] + h * y_low[l];
    }

    let error = y_high
        .iter()
        .zip(y_low.iter())
        .map(|(hi, lo)| (hi - lo).powi(2))
        .sum::<f64>()
        .sqrt();

    Ok((y_high, error))
}

/// Suggest the next step size from the local error estimate.
///
/// Standard controller: `h_next = h · clamp(0.9 · (tol/error)^(1/(p+1)), 0.1, 5)`
/// where `p = embedded_order`. The exponent matches the order of the embedded
/// error estimate.
pub fn suggest_next_step(h: f64, error: f64, tol: f64, embedded_order: usize) -> f64 {
    if error == 0.0 {
        return h * 5.0;
    }
    let ratio = tol / error;
    let p = embedded_order as f64;
    let factor = 0.9 * ratio.powf(1.0 / (p + 1.0));
    h * factor.clamp(0.1, 5.0)
}
