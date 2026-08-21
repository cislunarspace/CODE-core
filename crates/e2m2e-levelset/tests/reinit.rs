//! 阶段 3 验证：重初始化与符号距离（对应
//! `Examples/RussoSmereka/reinitCircle.m` 与 `signedDistanceIterative.m`）。

mod common;

use e2m2e_levelset::grid::{BoundaryCondition, Grid};
use e2m2e_levelset::shape::Shape;
use e2m2e_levelset::signed_distance::{signed_distance_iterative, SignedDistanceOptions};

fn circle_grid(n: usize) -> Grid {
    Grid::new(&[-1.0, -1.0], &[1.0, 1.0], &[n, n]).with_boundary_all(BoundaryCondition::Extrapolate)
}

#[test]
fn 缩放距离函数重初始化回真距离() {
    let n = 81;
    let grid = circle_grid(n);
    let center = [0.0, 0.0];
    let radius = 0.5;
    // 初值是真距离的 0.3 倍（同一零等值面，但不是距离函数）。
    let phi0 = 0.3
        * &Shape::Sphere {
            center: center.to_vec(),
            radius,
        }
        .implicit(&grid);

    let phi = signed_distance_iterative(&grid, &phi0, &SignedDistanceOptions::default());

    let dx = grid.dx()[0];
    let (mut worst, mut sum) = (0.0f64, 0.0f64);
    for (idx, v) in phi.indexed_iter() {
        let r = ((grid.axis(0)[idx[0]] - center[0]).powi(2)
            + (grid.axis(1)[idx[1]] - center[1]).powi(2))
        .sqrt();
        let exact = r - radius;
        let err = (v - exact).abs();
        worst = worst.max(err);
        sum += err;
    }
    let mean = sum / grid.size() as f64;
    // Russo-Smereka 亚网格修正下界面附近误差 O(dx)，远场几何收敛。
    assert!(worst < 2.5 * dx, "最大误差 {worst:.4e}（dx = {dx:.4e}）");
    assert!(mean < 0.5 * dx, "平均误差 {mean:.4e}（dx = {dx:.4e}）");
}

#[test]
fn zalesak_圆盘保持缺口拓扑() {
    let n = 81;
    let grid = circle_grid(n);
    let phi0 = Shape::ZalesakDisk {
        center: vec![0.0, 0.15],
        radius: 0.4,
        width: 0.15,
        height: 0.55,
    }
    .implicit(&grid);
    let phi = signed_distance_iterative(&grid, &phi0, &SignedDistanceOptions::default());

    // 缺口区域（圆盘内、槽内）重初始化后仍应为正（外部）。
    let inside_slot = |x1: f64, x2: f64| x1.abs() < 0.05 && x2 < 0.15;
    let mut slot_max = f64::NEG_INFINITY;
    for (idx, v) in phi.indexed_iter() {
        let (x1, x2) = (grid.axis(0)[idx[0]], grid.axis(1)[idx[1]]);
        if inside_slot(x1, x2) {
            slot_max = slot_max.max(*v);
        }
    }
    assert!(slot_max > 0.0, "缺口内应保持外部（正），最大值 {slot_max}");
    // 缺口两侧的圆盘体应仍为负。
    let body_phi = phi
        .indexed_iter()
        .find(|(idx, _)| {
            let (x1, x2) = (grid.axis(0)[idx[0]], grid.axis(1)[idx[1]]);
            (x1 - 0.25).abs() < 0.03 && (x2 + 0.1).abs() < 0.03
        })
        .map(|(_, v)| *v)
        .expect("网格上应有点 (0.25, -0.1)");
    assert!(body_phi < 0.0, "圆盘体内应保持内部（负）");
}

#[test]
fn 固定步数模式() {
    let grid = circle_grid(41);
    let phi0 = Shape::Sphere {
        center: vec![0.0, 0.0],
        radius: 0.5,
    }
    .implicit(&grid);
    let options = SignedDistanceOptions {
        t_max_steps: Some(3),
        ..Default::default()
    };
    let phi = signed_distance_iterative(&grid, &phi0, &options);
    // 三步后界面附近应已接近距离函数（亚网格修正一步即起效）。
    let dx = grid.dx()[0];
    let mut near_err = 0.0f64;
    for (idx, v) in phi.indexed_iter() {
        let r = (grid.axis(0)[idx[0]].powi(2) + grid.axis(1)[idx[1]].powi(2)).sqrt();
        if (r - 0.5).abs() < 2.0 * dx {
            near_err = near_err.max((*v - (r - 0.5)).abs());
        }
    }
    assert!(near_err < 1.5 * dx, "界面附近误差 {near_err:.4e}");
}
