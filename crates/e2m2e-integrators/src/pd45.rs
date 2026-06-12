/// Coefficients for the Prince-Dormand 5(4) method.
pub struct Pd45Coefficients;

impl Pd45Coefficients {
    pub const C: [f64; 7] = [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0];

    pub const A: [[f64; 7]; 7] = [
        [0.0; 7],
        [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0, 0.0],
        [
            19372.0 / 6561.0,
            -25360.0 / 2187.0,
            64448.0 / 6561.0,
            -212.0 / 729.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            9017.0 / 3168.0,
            -355.0 / 33.0,
            46732.0 / 5247.0,
            49.0 / 176.0,
            -5103.0 / 18656.0,
            0.0,
            0.0,
        ],
        [
            35.0 / 384.0,
            0.0,
            500.0 / 1113.0,
            125.0 / 192.0,
            -2187.0 / 6784.0,
            11.0 / 84.0,
            0.0,
        ],
    ];

    // 5th-order weights (b)
    pub const B5: [f64; 7] = [
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
        0.0,
    ];

    // 4th-order weights (b*)
    pub const B4: [f64; 7] = [
        5179.0 / 57600.0,
        0.0,
        7571.0 / 16695.0,
        393.0 / 640.0,
        -92097.0 / 339200.0,
        187.0 / 2100.0,
        1.0 / 40.0,
    ];
}

pub fn pd45_step<F, E>(t: f64, y: &[f64], h: f64, f: F) -> Result<(Vec<f64>, f64), E>
where
    F: Fn(f64, &[f64]) -> Result<Vec<f64>, E>,
{
    let n = y.len();
    let mut k = vec![vec![0.0; n]; 7];

    k[0] = f(t, y)?;

    for i in 1..7 {
        let ti = t + h * Pd45Coefficients::C[i];
        let mut yi = y.to_vec();
        for j in 0..i {
            let aij = Pd45Coefficients::A[i][j];
            for l in 0..n {
                yi[l] += h * aij * k[j][l];
            }
        }
        k[i] = f(ti, &yi)?;
    }

    let mut y5 = vec![0.0; n];
    let mut y4 = vec![0.0; n];
    for l in 0..n {
        for i in 0..7 {
            y5[l] += Pd45Coefficients::B5[i] * k[i][l];
            y4[l] += Pd45Coefficients::B4[i] * k[i][l];
        }
        y5[l] = y[l] + h * y5[l];
        y4[l] = y[l] + h * y4[l];
    }

    let error = y5
        .iter()
        .zip(y4.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();

    Ok((y5, error))
}

pub fn suggest_next_step(h: f64, error: f64, tol: f64) -> f64 {
    if error == 0.0 {
        return h * 5.0;
    }
    let ratio = tol / error;
    let factor = 0.9 * ratio.powf(0.2);
    h * factor.clamp(0.1, 5.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn butcher_rows_sum_to_c_nodes() {
        for i in 1..7 {
            let row_sum: f64 = Pd45Coefficients::A[i][..i].iter().sum();
            assert!(
                (row_sum - Pd45Coefficients::C[i]).abs() < 1e-15,
                "row {i} sum {row_sum} != c {}",
                Pd45Coefficients::C[i]
            );
        }
    }

    #[test]
    fn harmonic_oscillator_small_step_error() {
        // y'' = -y written as first-order system: [y, v], y' = v, v' = -y
        let f = |_t: f64, state: &[f64]| -> Result<Vec<f64>, std::convert::Infallible> {
            Ok(vec![state[1], -state[0]])
        };

        let y0 = vec![1.0, 0.0]; // initial condition y=cos(t), v=-sin(t)
        let t0 = 0.0;
        let h = 1e-4;

        let (y1, error) = pd45_step(t0, &y0, h, f).unwrap();

        // Analytic solution at t=h: cos(h), -sin(h)
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
