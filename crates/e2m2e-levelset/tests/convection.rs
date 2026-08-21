//! 阶段 1 验证：对流输运（对应 `Examples/Basic/convectionDemo.m`）。
//!
//! 周期域上的圆被常速场平移，t = 1 后圆心位移应等于速度 × 时间；
//! 一阶格式的耗散使面积略缩，ENO2 + odeCFL2 明显更锐。

mod common;

use e2m2e_levelset::derivative::{UpwindFirstENO2, UpwindFirstFirst};
use e2m2e_levelset::dissipation::ArtificialDissipationGLF;
use e2m2e_levelset::grid::{BoundaryCondition, Grid};
use e2m2e_levelset::hamiltonian::Advection;
use e2m2e_levelset::integrator::{ode_cfl1, ode_cfl2, CflOptions};
use e2m2e_levelset::shape::Shape;
use e2m2e_levelset::term::{LaxFriedrichsTerm, Term};
use ndarray::ArrayD;

fn centroid_inside(grid: &Grid, phi: &ArrayD<f64>) -> (f64, f64) {
    let (mut sx, mut sy, mut n) = (0.0f64, 0.0f64, 0usize);
    for (idx, v) in phi.indexed_iter() {
        if *v <= 0.0 {
            sx += grid.axis(0)[idx[0]];
            sy += grid.axis(1)[idx[1]];
            n += 1;
        }
    }
    (sx / n as f64, sy / n as f64)
}

fn area_inside(phi: &ArrayD<f64>) -> f64 {
    phi.iter().filter(|v| **v <= 0.0).count() as f64
}

fn advection_case(deriv: UpwindKind, n: usize) -> (f64, f64, f64) {
    let grid =
        Grid::new(&[0.0, 0.0], &[1.0, 1.0], &[n, n]).with_boundary_all(BoundaryCondition::Periodic);
    let phi0 = Shape::Sphere {
        center: vec![0.3, 0.5],
        radius: 0.15,
    }
    .implicit(&grid);

    let ham = Advection {
        velocity: vec![0.2, 0.0],
    };
    let mut options = CflOptions {
        factor_cfl: 0.5,
        ..Default::default()
    };
    let result = match deriv {
        UpwindKind::First => {
            let mut term = LaxFriedrichsTerm::new(
                grid.clone(),
                ham,
                UpwindFirstFirst,
                ArtificialDissipationGLF,
            );
            ode_cfl1(&mut term, 0.0, 1.0, phi0.clone(), &mut options)
        }
        UpwindKind::Eno2 => {
            let mut term = LaxFriedrichsTerm::new(
                grid.clone(),
                ham,
                UpwindFirstENO2,
                ArtificialDissipationGLF,
            );
            ode_cfl2(&mut term, 0.0, 1.0, phi0.clone(), &mut options)
        }
    };
    let (cx, cy) = centroid_inside(&grid, &result.y);
    let area_ratio = area_inside(&result.y) / area_inside(&phi0);
    (cx, cy, area_ratio)
}

enum UpwindKind {
    First,
    Eno2,
}

#[test]
fn 一阶格式平流圆的质心与面积() {
    let (cx, cy, area_ratio) = advection_case(UpwindKind::First, 81);
    // 期望圆心 (0.5, 0.5)，一阶耗散下质心误差允许约 1.5 dx（dx ≈ 0.0123）。
    assert!((cx - 0.5).abs() < 0.02, "x 质心 {cx}");
    assert!((cy - 0.5).abs() < 1e-12, "y 质心 {cy}（无 y 向速度）");
    assert!(
        (area_ratio - 1.0).abs() < 0.05,
        "面积比 {area_ratio}（一阶耗散允许略缩）"
    );
}

#[test]
fn 二阶格式平流圆更锐() {
    let (_, _, area_eno) = advection_case(UpwindKind::Eno2, 81);
    let (_, _, area_first) = advection_case(UpwindKind::First, 81);
    // ENO2 + odeCFL2 的面积损失应小于一阶格式。
    assert!(
        (area_eno - 1.0).abs() < (area_first - 1.0).abs(),
        "ENO2 面积比 {area_eno} 应比一阶 {area_first} 更接近 1"
    );
}

#[test]
fn 单步模式只走一步() {
    let grid = Grid::new(&[0.0], &[1.0], &[41]);
    let ham = Advection {
        velocity: vec![1.0],
    };
    let phi0 = Shape::Sphere {
        center: vec![0.5],
        radius: 0.2,
    }
    .implicit(&grid);
    let mut term = LaxFriedrichsTerm::new(
        grid.clone(),
        ham,
        UpwindFirstFirst,
        ArtificialDissipationGLF,
    );
    // 先取一步真实步长。
    let r1 = term.rhs(0.0, &phi0);
    let mut options = CflOptions {
        factor_cfl: 0.5,
        single_step: true,
        ..Default::default()
    };
    let result = ode_cfl1(&mut term, 0.0, 1.0, phi0, &mut options);
    assert_eq!(result.steps, 1);
    assert!((result.t - 0.5 * r1.step_bound).abs() < 1e-12);
}
