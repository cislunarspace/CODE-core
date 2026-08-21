//! 阶段 4 验证：双积分器最短到达时间（对应
//! `Examples/TimeToReach/doubleIntegratorTTR.m`）。
//!
//! 复刻示例的缺省流程：域 [-1,1]²、101 节点、Extrapolate 边界、
//! 目标半径 0.2、GLF 耗散、ENO2 格式、odeCFL2、factorCFL = 0.75、
//! 解析初值、termRestrictUpdate(positive=0) 包裹、postTimestepTTR 记录，
//! 积分到 t = 2 后与解析解逐点对比。

mod common;

use common::{analytic_double_integrator_ttr, DoubleIntegrator};
use e2m2e_levelset::derivative::UpwindFirstENO2;
use e2m2e_levelset::dissipation::ArtificialDissipationGLF;
use e2m2e_levelset::grid::{BoundaryCondition, Grid};
use e2m2e_levelset::integrator::{
    ode_cfl2, CflOptions, ConvergeNorm, PostTimestepTtrRecorder, TerminalEvent,
};
use e2m2e_levelset::term::{LaxFriedrichsTerm, RestrictUpdateTerm};
use ndarray::ArrayD;

/// 共享 TTR 记录器的钩子适配器：测试侧保留 Rc 句柄，积分后读取 ttr。
struct TtrHook(std::rc::Rc<std::cell::RefCell<PostTimestepTtrRecorder>>);

impl e2m2e_levelset::integrator::PostTimestep for TtrHook {
    fn post_step(&mut self, t: f64, y: &mut ArrayD<f64>) {
        self.0.borrow_mut().post_step(t, y);
    }
}

fn run_ttr(n: usize, llf: bool) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let grid = Grid::new(&[-1.0, -1.0], &[1.0, 1.0], &[n, n])
        .with_boundary_all(BoundaryCondition::Extrapolate);
    let input_bound = 1.0;
    let target_radius = 0.2;
    let t_max = 2.0;

    // 解析初值（示例 whichIC = 'analytic'，第 99 行最终生效的取值）：
    // φ₀ = τ_analytic − r。TTR 后处理要求初值是 TTR 的平移——此时
    // φ(x,t) = τ(x) − r − t 是精确解，零等值面按真实到达时刻传播。
    // 圆形初值下集合位置最终正确，但各节点穿越时刻无意义。
    let data0 = ArrayD::from_shape_fn(grid.shape(), |idx| {
        analytic_double_integrator_ttr(grid.axis(0)[idx[0]], grid.axis(1)[idx[1]]) - target_radius
    });

    let ham = DoubleIntegrator { input_bound };
    let inner: Box<dyn e2m2e_levelset::term::Term> = if llf {
        Box::new(LaxFriedrichsTerm::new(
            grid.clone(),
            ham,
            UpwindFirstENO2,
            e2m2e_levelset::dissipation::ArtificialDissipationLLF,
        ))
    } else {
        Box::new(LaxFriedrichsTerm::new(
            grid.clone(),
            ham,
            UpwindFirstENO2,
            ArtificialDissipationGLF,
        ))
    };
    let mut term = RestrictUpdateTerm {
        inner,
        positive: false,
    };

    let recorder = std::rc::Rc::new(std::cell::RefCell::new(PostTimestepTtrRecorder::new(&grid)));
    recorder.borrow_mut().record(0.0, &data0);
    let mut options = CflOptions {
        factor_cfl: 0.75,
        ..Default::default()
    };
    options
        .post_timestep
        .push(Box::new(TtrHook(recorder.clone())));

    let result = ode_cfl2(&mut term, 0.0, t_max, data0, &mut options);
    let recorder = recorder.borrow();

    let (mut ttr_vals, mut attr_vals, mut phi_vals) = (Vec::new(), Vec::new(), Vec::new());
    for (idx, v) in recorder.ttr.indexed_iter() {
        let (x1, x2) = (grid.axis(0)[idx[0]], grid.axis(1)[idx[1]]);
        ttr_vals.push(*v);
        attr_vals.push((analytic_double_integrator_ttr(x1, x2) - target_radius).max(0.0));
        phi_vals.push(result.y[idx.clone()]);
    }
    (ttr_vals, attr_vals, phi_vals)
}

/// 统计远离切换曲线与 t_max 边界带的 TTR 误差（最大值、中位数）。
fn ttr_error_stats(n: usize) -> (f64, f64, usize) {
    let (ttr, attr, _) = run_ttr(n, false);
    let mut errs = Vec::new();
    for (i, (mttr, attr)) in ttr.iter().zip(&attr).enumerate() {
        let (x1, x2) = (
            -1.0 + ((i % n) as f64 + 0.5) * 2.0 / n as f64,
            -1.0 + ((i / n) as f64 + 0.5) * 2.0 / n as f64,
        );
        let switch_dist = (x1 + 0.5 * x2 * x2.abs()).abs();
        if (0.3..1.7).contains(attr) && mttr.is_finite() && switch_dist > 0.15 {
            errs.push((mttr - attr).abs());
        }
    }
    errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (errs[errs.len() - 1], errs[errs.len() / 2], errs.len())
}

#[test]
fn 双积分器_ttr_对比解析解() {
    // 双积分器的 ∂H/∂p 与 p 无关，GLF/LLF/LLLF 三种耗散等价；
    // LF 类格式的固有耗散使 TTR 误差随网格一阶收敛（101 节点实测
    // 最大 ~0.41、201 节点 ~0.28），与 MATLAB 原版行为一致。
    let (worst, _median, count) = ttr_error_stats(101);
    assert!(count > 1000, "有效对比节点数 {count} 应覆盖大部分网格");
    assert!(worst < 0.45, "远离切换曲线的 TTR 最大误差 {worst:.4}");

    // 分辨率加倍，误差应明显下降（格式收敛性的回归防线）。
    let (worst_coarse, _, _) = ttr_error_stats(51);
    let (worst_fine, _, _) = ttr_error_stats(101);
    assert!(
        worst_fine < worst_coarse * 0.9,
        "加密网格误差应下降：{worst_coarse:.4} → {worst_fine:.4}"
    );
}

#[test]
fn 双积分器_可达集边界对比解析解() {
    let (_, attr, phi) = run_ttr(101, false);
    // t = 2 时刻的可达集（φ ≤ 0）应与解析解 attr ≤ 2 的集合基本重合。
    let (mut mismatch, mut boundary) = (0usize, 0usize);
    for (attr, phi) in attr.iter().zip(&phi) {
        let reached_numeric = *phi <= 0.0;
        let reached_analytic = *attr <= 2.0;
        if *attr >= 1.8 && *attr <= 2.2 {
            boundary += 1;
        }
        if reached_numeric != reached_analytic && !(1.7..2.3).contains(attr) {
            mismatch += 1;
        }
    }
    let _ = boundary;
    let frac = mismatch as f64 / attr.len() as f64;
    assert!(frac < 0.02, "集合错分比例 {frac:.4}（排除 t≈2 边界带）");
}

#[test]
fn 收敛终止事件提前停止() {
    // 重初始化到收敛：更新量趋零时事件变号，积分应早于 t_max 停止。
    use e2m2e_levelset::shape::Shape;

    let grid = Grid::new(&[-1.0, -1.0], &[1.0, 1.0], &[41, 41])
        .with_boundary_all(BoundaryCondition::Extrapolate);
    // 初值取 0.3 倍距离（非距离函数）：前几步更新量大（事件值 +1），
    // 收敛后更新量趋零（事件值翻为 -1），变号触发提前终止。
    let phi0 = 0.3
        * &Shape::Sphere {
            center: vec![0.0, 0.0],
            radius: 0.5,
        }
        .implicit(&grid);
    let mut term = e2m2e_levelset::term::ReinitTerm {
        grid: grid.clone(),
        deriv: Box::new(e2m2e_levelset::derivative::UpwindFirstENO2),
        initial: phi0.clone(),
        subcell_fix: true,
    };
    let mut options = CflOptions {
        factor_cfl: 0.5,
        terminal_event: Some(TerminalEvent::Converge {
            abs_tol: 1e-4,
            rel_tol: 1e-4,
            norm: ConvergeNorm::Average,
        }),
        ..Default::default()
    };
    let result = e2m2e_levelset::integrator::ode_cfl2(&mut term, 0.0, 1.0, phi0, &mut options);
    assert!(
        result.t < 1.0,
        "应在 t_max 前收敛停止，实际 t = {}",
        result.t
    );
    assert!(result.steps >= 2, "至少两步才能检测事件变号");
}
