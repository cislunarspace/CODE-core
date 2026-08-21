//! 阶段 2 验证：Burgers 方程（对应 `Examples/OsherShu/burgersLF.m`）。
//!
//! `φ_t + (φ_x)²/2 = 0`，周期域初值 φ₀ = -cos x。激波形成时刻 t = 1，
//! 在 t = 0.5（光滑解阶段）用 Hopf–Lax 公式构造精确解逐点对比：
//! `φ(x,t) = min_ξ { (x−ξ)²/(2t) + φ₀(ξ) }`，最小值点满足足点方程
//! `x = ξ + t·φ₀'(ξ)`（t·max|φ₀''| < 1 时单调，二分求 ξ）。

mod common;

use common::Burgers1d;
use e2m2e_levelset::derivative::UpwindFirstENO2;
use e2m2e_levelset::dissipation::ArtificialDissipationGLF;
use e2m2e_levelset::grid::Grid;
use e2m2e_levelset::integrator::{ode_cfl2, CflOptions};
use e2m2e_levelset::term::LaxFriedrichsTerm;

fn burgers_error(n: usize, t_end: f64) -> f64 {
    let grid = Grid::new(&[0.0], &[2.0 * std::f64::consts::PI], &[n]);
    let phi0 =
        ndarray::Array1::from(grid.axis(0).iter().map(|x| -x.cos()).collect::<Vec<_>>()).into_dyn();

    let mut options = CflOptions::default();
    let mut term = LaxFriedrichsTerm::new(
        grid.clone(),
        Burgers1d,
        UpwindFirstENO2,
        ArtificialDissipationGLF,
    );
    let result = ode_cfl2(&mut term, 0.0, t_end, phi0.clone(), &mut options);

    // 特征线精确解：g(ξ) = ξ + t·sin ξ - x = 0 在 [0, 2π) 上二分。
    let mut worst = 0.0f64;
    for (i, x) in grid.axis(0).iter().enumerate() {
        let g = |xi: f64| xi + t_end * xi.sin() - x;
        let (mut lo, mut hi) = (-0.5f64, 2.0 * std::f64::consts::PI + 0.5);
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            if g(mid) < 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let xi = 0.5 * (lo + hi);
        // Hopf–Lax：φ = (x−ξ)²/(2t) + φ₀(ξ)，其中 (x−ξ)/t = φ₀'(ξ)。
        let exact = (x - xi).powi(2) / (2.0 * t_end) - xi.cos();
        worst = worst.max((result.y[i] - exact).abs());
    }
    worst
}

#[test]
fn burgers_光滑解特征线对比() {
    let err = burgers_error(401, 0.5);
    assert!(err < 0.02, "L∞ 误差 {err:.4e}（ENO2 + GLF，N=401）");
}

#[test]
fn burgers_网格加密收敛() {
    let coarse = burgers_error(101, 0.5);
    let fine = burgers_error(201, 0.5);
    assert!(
        fine < coarse * 0.75,
        "加密网格误差应下降：{coarse:.4e} → {fine:.4e}"
    );
}

#[test]
fn burgers_激波后解有界() {
    // t = 1.5 > 激波时刻，格式应保持 L∞ 稳定（TVD）。
    let err = burgers_error(201, 1.5);
    // 与解析解（含多值截断）不做逐点对比，只确认解保持在 [−1, 1] 量级。
    assert!(err < 1.5, "t > 激波时刻解应保持有界（误差 {err:.3e}）");
}
