//! 迭代法符号距离函数：对应 ToolboxLS 的
//! `Helper/SignedDistance/signedDistanceIterative.m`。
//!
//! 通过迭代求解重初始化方程，把隐式表面函数转换为符号距离函数。信息
//! 从零等值面以单位速度向外传播，要得到至少 `k` 个网格宽度的有效带，
//! `t_max` 至少取 `k * max(dx)`。

use crate::derivative::{
    UpwindDerivative, UpwindFirstENO2, UpwindFirstENO3, UpwindFirstFirst, UpwindFirstWENO5,
};
use crate::grid::Grid;
use crate::integrator::{ode_cfl1, ode_cfl2, ode_cfl3, CflOptions, CflResult};
use crate::term::ReinitTerm;
use ndarray::{ArrayD, Zip};

/// 精度档位（`signedDistanceIterative.m` 第 23-26 行）：
/// 积分器与导数格式的组合。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ReinitAccuracy {
    /// odeCFL1 + upwindFirstFirst。
    Low,
    /// odeCFL2 + upwindFirstENO2（默认）。
    Medium,
    /// odeCFL3 + upwindFirstENO3。
    High,
    /// odeCFL3 + upwindFirstWENO5。
    VeryHigh,
}

impl ReinitAccuracy {
    /// 组装该档位对应的（积分器代号, 导数格式）。
    fn parts(self) -> (u8, Box<dyn UpwindDerivative>) {
        match self {
            ReinitAccuracy::Low => (1, Box::new(UpwindFirstFirst)),
            ReinitAccuracy::Medium => (2, Box::new(UpwindFirstENO2)),
            ReinitAccuracy::High => (3, Box::new(UpwindFirstENO3)),
            ReinitAccuracy::VeryHigh => (3, Box::new(UpwindFirstWENO5)),
        }
    }
}

/// 重初始化选项，默认值对应 `signedDistanceIterative.m` 第 27-35 行。
#[derive(Clone, Debug)]
pub struct SignedDistanceOptions {
    pub accuracy: ReinitAccuracy,
    /// 迭代终止时间（默认域对角线 `max(grid.max - grid.min)`）。
    pub t_max: Option<f64>,
    /// 固定重初始化步数（对应 MATLAB `tMax < 0` 时的 `-round(tMax)`
    /// 语义）；设置后忽略 `t_max`。
    pub t_max_steps: Option<usize>,
    /// 单步平均更新量低于 `error_max * max(dx)` 时提前收敛（默认 1e-3）。
    pub error_max: f64,
}

impl Default for SignedDistanceOptions {
    fn default() -> Self {
        SignedDistanceOptions {
            accuracy: ReinitAccuracy::Medium,
            t_max: None,
            t_max_steps: None,
            error_max: 1e-3,
        }
    }
}

/// 走一个单步（三个积分器共用入口）。
fn single_step(
    integrator: u8,
    term: &mut ReinitTerm,
    t_now: f64,
    t_max: f64,
    y: ArrayD<f64>,
    options: &mut CflOptions,
) -> CflResult {
    match integrator {
        1 => ode_cfl1(term, t_now, t_max, y, options),
        2 => ode_cfl2(term, t_now, t_max, y, options),
        _ => ode_cfl3(term, t_now, t_max, y, options),
    }
}

/// 把隐式表面函数 `phi0` 转换为符号距离函数。
///
/// 与 MATLAB 版一致：CFL 系数 0.95，单步推进 + 步间收敛检查（与上一步
/// 起点的 1 范数更新量低于 `error_max * max(dx) * 节点数` 即停）；
/// `initial` 固定为原始 `phi0`，不随迭代更新。
pub fn signed_distance_iterative(
    grid: &Grid,
    phi0: &ArrayD<f64>,
    options: &SignedDistanceOptions,
) -> ArrayD<f64> {
    grid.check_data(phi0);
    let (integrator, deriv) = options.accuracy.parts();
    let mut term = ReinitTerm {
        grid: grid.clone(),
        deriv,
        initial: phi0.clone(),
        subcell_fix: true,
    };
    let small = 100.0 * f64::EPSILON;
    let max_dx = grid.dx().iter().fold(0.0f64, |m, v| m.max(*v));
    // 收敛判据见 `signedDistanceIterative.m` 第 124、149 行：
    // norm(y - y_step_start, 1) < error_max * max(dx) * prod(N)。
    let delta_max = options.error_max * max_dx * grid.size() as f64;
    let mut single = CflOptions {
        factor_cfl: 0.95,
        single_step: true,
        ..Default::default()
    };

    match options.t_max_steps {
        // 固定步数模式：走指定步数，不做收敛检查。
        Some(steps) => {
            let mut y = phi0.clone();
            let mut t_now = 0.0f64;
            for _ in 0..steps {
                let r = single_step(integrator, &mut term, t_now, f64::INFINITY, y, &mut single);
                y = r.y;
                t_now = r.t;
            }
            y
        }
        None => {
            let t_max = options.t_max.unwrap_or_else(|| {
                grid.max()
                    .iter()
                    .zip(grid.min())
                    .map(|(a, b)| a - b)
                    .fold(0.0f64, |m, v| m.max(v))
            });
            let mut y = phi0.clone();
            let mut y_step_start = phi0.clone();
            let mut t_now = 0.0f64;
            while t_max - t_now >= small * t_max {
                if t_now > 0.0 {
                    let l1 = Zip::from(&y)
                        .and(&y_step_start)
                        .fold(0.0, |acc, a, b| acc + (a - b).abs());
                    if l1 < delta_max {
                        break;
                    }
                }
                y_step_start = y.clone();
                let r = single_step(
                    integrator,
                    &mut term,
                    t_now,
                    t_max,
                    y_step_start.clone(),
                    &mut single,
                );
                y = r.y;
                t_now = r.t;
            }
            y
        }
    }
}
