//! RK 方法二体问题解析解对照（ADR 0013：按定义验证，不用 golden file）。
//!
//! 使用两种尺度的圆轨道二体问题：
//! - 真实轨道（GM=398600.44, r₀=8000 km）：单步误差在舍入噪声级别
//! - 单位轨道（GM=1, r₀=1, ω=1, T=2π）：截断误差超机器精度，可观测收敛
//!
//! 依据：圆形 Kepler 轨道解析解是二体问题的"定义"，RK 方法应随步长减小
//! 向其收敛。断言基于数学恒等式和物理定义，不依赖任何外部软件或 golden 文件。

use e2m2e_propagation::butcher::explicit_rk_step;
use e2m2e_propagation::rk_methods::RkMethod;

#[derive(Clone, Copy)]
struct Circ {
    r0: f64,
    omega: f64,
}
impl Circ {
    fn new(r: f64, mu: f64) -> Self {
        let v = (mu / r).sqrt();
        Self {
            r0: r,
            omega: v / r,
        }
    }
    fn state(&self, t: f64) -> [f64; 6] {
        let c = (self.omega * t).cos();
        let s = (self.omega * t).sin();
        let v0 = self.r0 * self.omega;
        [self.r0 * c, self.r0 * s, 0.0, -v0 * s, v0 * c, 0.0]
    }
}

fn rhs(gm: f64) -> impl Fn(f64, &[f64]) -> Result<Vec<f64>, std::convert::Infallible> {
    move |_t, y| {
        let r3 = (y[0].powi(2) + y[1].powi(2) + y[2].powi(2)).powf(1.5);
        let a = -gm / r3;
        Ok(vec![y[3], y[4], y[5], a * y[0], a * y[1], a * y[2]])
    }
}

fn l2(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

// ── test 1: 真实二体圆轨道 — 小步长单步精度合理 ──────────────────────

#[test]
fn test_earth_orbit_small_step() {
    let orbit = Circ::new(8000.0, 398600.4418);
    let f = rhs(398600.4418);
    let h = 1e-4;
    for method in &[RkMethod::Pd45, RkMethod::Pd78, RkMethod::Rk89] {
        let table = method.table();
        let y0 = orbit.state(0.0).to_vec();
        let exact = orbit.state(h);
        let (y1, _) = explicit_rk_step(table, 0.0, &y0, h, &f, None).unwrap();
        let err = l2(&y1, &exact);
        assert!(err < 1e-10, "{method:?} h={h}: err {err:.2e} > 1e-10");
    }
}

// ── test 2: 单位轨道 — 步长减小误差减小且排序合理 ──────────────────────

/// 每种 RK 方法在步长减半、步数加倍（等总时长）下误差缩小。
/// 比值应远大于 1（至少 8 倍，即 h 减半后至少加速减少）。
#[test]
fn test_error_decreases_with_h() {
    let orbit = Circ::new(1.0, 1.0);

    struct Case {
        method: RkMethod,
        h: f64,
        n: usize,
    }

    fn go(
        table: &e2m2e_propagation::butcher::ButcherTable,
        gm: f64,
        h: f64,
        n: usize,
        orbit: &Circ,
    ) -> f64 {
        let mut y = orbit.state(0.0).to_vec();
        let mut t = 0.0;
        let f = rhs(gm);
        for _ in 0..n {
            let (yn, _) = explicit_rk_step(table, t, &y, h, &f, None).unwrap();
            y = yn;
            t += h;
        }
        l2(&y, &orbit.state(t))
    }

    let cases = [
        Case {
            method: RkMethod::Pd45,
            h: 0.4,
            n: 40,
        },
        Case {
            method: RkMethod::Pd45,
            h: 0.2,
            n: 80,
        },
        Case {
            method: RkMethod::Pd78,
            h: 0.4,
            n: 40,
        },
        Case {
            method: RkMethod::Pd78,
            h: 0.2,
            n: 80,
        },
        Case {
            method: RkMethod::Rk89,
            h: 0.4,
            n: 40,
        },
        Case {
            method: RkMethod::Rk89,
            h: 0.2,
            n: 80,
        },
    ];

    for w in cases.chunks(2) {
        let c1 = &w[0];
        let c2 = &w[1];
        let e1 = go(c1.method.table(), 1.0, c1.h, c1.n, &orbit);
        let e2 = go(c2.method.table(), 1.0, c2.h, c2.n, &orbit);

        // 误差应随步长减小而显著降低
        let ratio = e1 / e2;
        assert!(
            ratio > 8.0,
            "{:?} h={} err {:.2e} → h={} err {:.2e}, ratio {ratio:.1} ≤ 8",
            c1.method,
            c1.h,
            e1,
            c2.h,
            e2
        );
    }
}
