//! Prince-Dormand 5(4) 表（Dormand & Prince, 1980）。
//!
//! 仅系数表存放于此 —— 共享步进逻辑在 [`crate::butcher::explicit_rk_step`]。

use crate::butcher::ButcherTable;

/// Prince-Dormand 5(4)：7 级，主阶 5，嵌入阶 4。
///
/// 系数与先前手写的逐位实现一致。
pub const PD45_TABLE: ButcherTable = ButcherTable::new(
    7,
    5,
    4,
    &[0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0],
    &[
        &[],
        &[1.0 / 5.0],
        &[3.0 / 40.0, 9.0 / 40.0],
        &[44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0],
        &[
            19372.0 / 6561.0,
            -25360.0 / 2187.0,
            64448.0 / 6561.0,
            -212.0 / 729.0,
        ],
        &[
            9017.0 / 3168.0,
            -355.0 / 33.0,
            46732.0 / 5247.0,
            49.0 / 176.0,
            -5103.0 / 18656.0,
        ],
        &[
            35.0 / 384.0,
            0.0,
            500.0 / 1113.0,
            125.0 / 192.0,
            -2187.0 / 6784.0,
            11.0 / 84.0,
        ],
    ],
    // 5th-order weights b (identical to the FSAL row a[6])
    &[
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
        0.0,
    ],
    // 4th-order embedded weights b*
    &[
        5179.0 / 57600.0,
        0.0,
        7571.0 / 16695.0,
        393.0 / 640.0,
        -92097.0 / 339200.0,
        187.0 / 2100.0,
        1.0 / 40.0,
    ],
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butcher::explicit_rk_step;

    #[test]
    fn row_lengths_are_lower_triangular() {
        for i in 0..PD45_TABLE.stages {
            assert_eq!(PD45_TABLE.a[i].len(), i, "row {i} should have length {i}");
        }
    }

    #[test]
    fn butcher_rows_sum_to_c_nodes() {
        for i in 1..PD45_TABLE.stages {
            let row_sum: f64 = PD45_TABLE.a[i].iter().sum();
            assert!(
                (row_sum - PD45_TABLE.c[i]).abs() < 1e-15,
                "row {i} sum {row_sum} != c {}",
                PD45_TABLE.c[i]
            );
        }
    }

    #[test]
    fn weights_sum_to_one() {
        let b_sum: f64 = PD45_TABLE.b.iter().sum();
        let b_star_sum: f64 = PD45_TABLE.b_star.iter().sum();
        assert!((b_sum - 1.0).abs() < 1e-15, "b sum {b_sum} != 1");
        assert!((b_star_sum - 1.0).abs() < 1e-15, "b* sum {b_star_sum} != 1");
    }

    #[test]
    fn harmonic_oscillator_small_step_error() {
        // y'' = -y as first-order system [y, v]: y' = v, v' = -y.
        let f = |_t: f64, state: &[f64]| -> Result<Vec<f64>, std::convert::Infallible> {
            Ok(vec![state[1], -state[0]])
        };

        let y0 = vec![1.0, 0.0];
        let h = 1e-4;

        let (y1, error) = explicit_rk_step(&PD45_TABLE, 0.0, &y0, h, f, None).unwrap();

        // Analytic solution at t=h: cos(h), -sin(h).
        let y_exact = vec![h.cos(), -h.sin()];
        let num_err = y1
            .iter()
            .zip(y_exact.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        assert!(num_err < 1e-10, "numerical error {num_err} too large");
        assert!(error < 1e-10, "local truncation estimate {error} too large");
    }
}
