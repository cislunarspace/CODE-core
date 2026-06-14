//! RungeKutta89 tableau (Verner 9(8), 16 stages).
//!
//! Only the coefficient table lives here — the shared step logic is in
//! [`crate::butcher::explicit_rk_step`].
//!
//! # Source
//! Coefficients transcribed verbatim from GMAT R2026a
//! `src/base/propagator/RungeKutta89.cpp::SetCoefficients()` (line 152), which
//! declares `RungeKutta(16, 9)` — 16 stages, 9th-order solution with 8th-order
//! embedded error control. GMAT stores the time nodes `ai`, the RK matrix
//! `bij`, the 9th-order weights `cj`, and the error-estimate coefficients `ee`
//! (where `ee = cj - cj_hat`). The 8th-order embedded weights `cj_hat` are
//! reconstructed here as `b_star = cj - ee` and verified by the `ee_consistency`
//! test against GMAT's `ee` literal. The √6 terms are characteristic of
//! Verner's family of methods.

use crate::butcher::ButcherTable;

/// sqrt(6.0), matching GMAT's `Real rt6 = sqrt(6.0)`.
const RT6: f64 = 2.449489742783178; // sqrt(6.0), matches GMAT `Real rt6 = sqrt(6.0)`

/// RungeKutta89 (Verner 9(8)): 16 stages, order 9, embedded order 8.
pub const RK89_TABLE: ButcherTable = ButcherTable::new(
    16,
    9,
    8,
    // c (GMAT ai)
    &[
        0.0,
        1.0 / 12.0,
        1.0 / 9.0,
        1.0 / 6.0,
        (2.0 + 2.0 * RT6) / 15.0,
        (6.0 + RT6) / 15.0,
        (6.0 - RT6) / 15.0,
        2.0 / 3.0,
        1.0 / 2.0,
        1.0 / 3.0,
        1.0 / 4.0,
        4.0 / 3.0,
        5.0 / 6.0,
        1.0,
        1.0 / 6.0,
        1.0,
    ],
    // a (GMAT bij, strictly lower triangular — diagonal zeros dropped)
    &[
        &[],
        &[1.0 / 12.0],
        &[1.0 / 27.0, 2.0 / 27.0],
        &[1.0 / 24.0, 0.0, 1.0 / 8.0],
        &[
            (4.0 + 94.0 * RT6) / 375.0,
            0.0,
            (-94.0 - 84.0 * RT6) / 125.0,
            (328.0 + 208.0 * RT6) / 375.0,
        ],
        &[
            (9.0 - RT6) / 150.0,
            0.0,
            0.0,
            (312.0 + 32.0 * RT6) / 1425.0,
            (69.0 + 29.0 * RT6) / 570.0,
        ],
        &[
            (927.0 - 347.0 * RT6) / 1250.0,
            0.0,
            0.0,
            (-16248.0 + 7328.0 * RT6) / 9375.0,
            (-489.0 + 179.0 * RT6) / 3750.0,
            (14268.0 - 5798.0 * RT6) / 9375.0,
        ],
        &[
            2.0 / 27.0,
            0.0,
            0.0,
            0.0,
            0.0,
            (16.0 - RT6) / 54.0,
            (16.0 + RT6) / 54.0,
        ],
        &[
            19.0 / 256.0,
            0.0,
            0.0,
            0.0,
            0.0,
            (118.0 - 23.0 * RT6) / 512.0,
            (118.0 + 23.0 * RT6) / 512.0,
            -9.0 / 256.0,
        ],
        &[
            11.0 / 144.0,
            0.0,
            0.0,
            0.0,
            0.0,
            (266.0 - RT6) / 864.0,
            (266.0 + RT6) / 864.0,
            -1.0 / 16.0,
            -8.0 / 27.0,
        ],
        &[
            (5034.0 - 271.0 * RT6) / 61440.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            (7859.0 - 1626.0 * RT6) / 10240.0,
            (-2232.0 + 813.0 * RT6) / 20480.0,
            (-594.0 + 271.0 * RT6) / 960.0,
            (657.0 - 813.0 * RT6) / 5120.0,
        ],
        &[
            (5996.0 - 3794.0 * RT6) / 405.0,
            0.0,
            0.0,
            0.0,
            0.0,
            (-4342.0 - 338.0 * RT6) / 9.0,
            (154922.0 - 40458.0 * RT6) / 135.0,
            (-4176.0 + 3794.0 * RT6) / 45.0,
            (-340864.0 + 242816.0 * RT6) / 405.0,
            (26304.0 - 15176.0 * RT6) / 45.0,
            -26624.0 / 81.0,
        ],
        &[
            (3793.0 + 2168.0 * RT6) / 103680.0,
            0.0,
            0.0,
            0.0,
            0.0,
            (4042.0 + 2263.0 * RT6) / 13824.0,
            (-231278.0 + 40717.0 * RT6) / 69120.0,
            (7947.0 - 2168.0 * RT6) / 11520.0,
            (1048.0 - 542.0 * RT6) / 405.0,
            (-1383.0 + 542.0 * RT6) / 720.0,
            2624.0 / 1053.0,
            3.0 / 1664.0,
        ],
        &[
            -137.0 / 1296.0,
            0.0,
            0.0,
            0.0,
            0.0,
            (5642.0 - 337.0 * RT6) / 864.0,
            (5642.0 + 337.0 * RT6) / 864.0,
            -299.0 / 48.0,
            184.0 / 81.0,
            -44.0 / 9.0,
            -5120.0 / 1053.0,
            -11.0 / 468.0,
            16.0 / 9.0,
        ],
        &[
            (33617.0 - 2168.0 * RT6) / 518400.0,
            0.0,
            0.0,
            0.0,
            0.0,
            (-3846.0 + 31.0 * RT6) / 13824.0,
            (155338.0 - 52807.0 * RT6) / 345600.0,
            (-12537.0 + 2168.0 * RT6) / 57600.0,
            (92.0 + 542.0 * RT6) / 2025.0,
            (-1797.0 - 542.0 * RT6) / 3600.0,
            320.0 / 567.0,
            -1.0 / 1920.0,
            4.0 / 105.0,
            0.0,
        ],
        &[
            (-36487.0 - 30352.0 * RT6) / 279600.0,
            0.0,
            0.0,
            0.0,
            0.0,
            (-29666.0 - 4499.0 * RT6) / 7456.0,
            (2779182.0 - 615973.0 * RT6) / 186400.0,
            (-94329.0 + 91056.0 * RT6) / 93200.0,
            (-232192.0 + 121408.0 * RT6) / 17475.0,
            (101226.0 - 22764.0 * RT6) / 5825.0,
            -169984.0 / 9087.0,
            -87.0 / 30290.0,
            492.0 / 1165.0,
            0.0,
            1260.0 / 233.0,
        ],
    ],
    // b (GMAT cj): 9th-order weights
    &[
        23.0 / 525.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        171.0 / 1400.0,
        86.0 / 525.0,
        93.0 / 280.0,
        -2048.0 / 6825.0,
        -3.0 / 18200.0,
        39.0 / 175.0,
        0.0,
        9.0 / 25.0,
        233.0 / 4200.0,
    ],
    // b_star = cj - ee: 8th-order embedded weights
    &[
        103.0 / 1680.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -27.0 / 140.0,
        76.0 / 105.0,
        -201.0 / 280.0,
        1024.0 / 1365.0,
        3.0 / 7280.0,
        12.0 / 35.0,
        9.0 / 280.0,
        0.0,
        0.0,
    ],
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butcher::explicit_rk_step;

    // GMAT's error-estimate coefficients `ee` (RungeKutta89.cpp:348-363),
    // transcribed verbatim. Used to verify b - b_star == ee.
    const GMAT_EE: [f64; 16] = [
        -7.0 / 400.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        63.0 / 200.0,
        -14.0 / 25.0,
        21.0 / 20.0,
        -1024.0 / 975.0,
        -21.0 / 36400.0,
        -3.0 / 25.0,
        -9.0 / 280.0,
        9.0 / 25.0,
        233.0 / 4200.0,
    ];

    #[test]
    fn row_lengths_are_lower_triangular() {
        for i in 0..RK89_TABLE.stages {
            assert_eq!(RK89_TABLE.a[i].len(), i, "row {i} should have length {i}");
        }
    }

    #[test]
    fn butcher_rows_sum_to_c_nodes() {
        for i in 1..RK89_TABLE.stages {
            let row_sum: f64 = RK89_TABLE.a[i].iter().sum();
            assert!(
                (row_sum - RK89_TABLE.c[i]).abs() < 1e-10,
                "row {i} sum {row_sum} != c {}",
                RK89_TABLE.c[i]
            );
        }
    }

    #[test]
    fn weights_sum_to_one() {
        let b_sum: f64 = RK89_TABLE.b.iter().sum();
        let b_star_sum: f64 = RK89_TABLE.b_star.iter().sum();
        assert!((b_sum - 1.0).abs() < 1e-12, "b sum {b_sum} != 1");
        assert!((b_star_sum - 1.0).abs() < 1e-12, "b* sum {b_star_sum} != 1");
    }

    #[test]
    fn ee_consistency() {
        // b - b_star must equal GMAT's ee literal (verifies b_star = cj - ee).
        for i in 0..RK89_TABLE.stages {
            let diff = RK89_TABLE.b[i] - RK89_TABLE.b_star[i];
            assert!(
                (diff - GMAT_EE[i]).abs() < 1e-12,
                "b[{i}] - b_star[{i}] = {diff} != ee {}",
                GMAT_EE[i]
            );
        }
    }

    #[test]
    fn harmonic_oscillator_small_step_error() {
        // y'' = -y as first-order system [y, v]: y' = v, v' = -y.
        let f = |_t: f64, state: &[f64]| -> Result<Vec<f64>, std::convert::Infallible> {
            Ok(vec![state[1], -state[0]])
        };

        let y0 = vec![1.0, 0.0];
        let h = 1e-3;

        let (y1, error) = explicit_rk_step(&RK89_TABLE, 0.0, &y0, h, f).unwrap();

        let y_exact = vec![h.cos(), -h.sin()];
        let num_err = y1
            .iter()
            .zip(y_exact.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        // 9th-order method at h=1e-3: error ~ h^9 ~ 1e-27.
        assert!(num_err < 1e-13, "numerical error {num_err} too large");
        assert!(error < 1e-13, "local truncation estimate {error} too large");
    }
}
