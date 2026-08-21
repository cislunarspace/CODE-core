//! CFL 约束时间积分器：对应 ToolboxLS 的 `ExplicitIntegration/Integrators/`
//! 下的 `odeCFL1/2/3.m` 与选项结构 `odeCFLset.m`。
//!
//! 与 e2m2e-propagation 的自适应积分器不同，这类积分器不做误差控制，
//! 步长由右端项返回的 CFL 上界直接决定，适用于双曲型 PDE 的方法线
//! 时间推进。三个积分器共用同一个驱动循环（含 postTimestep、
//! terminalEvent、singleStep、stats），仅子步组合不同。

use crate::boundary::matlab_sign;
use crate::grid::Grid;
use crate::signed_distance::{signed_distance_iterative, ReinitAccuracy, SignedDistanceOptions};
use crate::term::{Term, TermRhs};
use ndarray::{ArrayD, Zip};

/// 积分选项，字段对应 `odeCFLset.m` 第 90-95 行的默认值。
/// 含 `Box<dyn PostTimestep>` 钩子，不可 Clone。
pub struct CflOptions {
    /// CFL 系数，实际步长 = `factor_cfl * step_bound`（MATLAB 默认 0.5）。
    pub factor_cfl: f64,
    /// 步长上限（MATLAB 默认 realmax，此处无穷）。
    pub max_step: f64,
    /// 只推进一步（`singleStep`）。
    pub single_step: bool,
    /// 打印步数统计（`stats`）。
    pub stats: bool,
    /// 每步结束后依次调用的后处理钩子（`postTimestep`，可多个）。
    pub post_timestep: Vec<Box<dyn PostTimestep>>,
    /// 终止事件（`terminalEvent`）：事件值与上一步符号相反即停。
    pub terminal_event: Option<TerminalEvent>,
}

impl Default for CflOptions {
    fn default() -> Self {
        CflOptions {
            factor_cfl: 0.5,
            max_step: f64::INFINITY,
            single_step: false,
            stats: false,
            post_timestep: Vec::new(),
            terminal_event: None,
        }
    }
}

/// 每步之后的钩子，可修改解。MATLAB 原型
/// `postTimestep(t, y, schemeData)` 中的 `schemeData` 依赖（如掩码、网格）
/// 由各实现的结构体字段持有。
pub trait PostTimestep {
    fn post_step(&mut self, t: f64, y: &mut ArrayD<f64>);
}

/// 掩码钳制（`postTimestepMask.m`）：每步结束后把掩码内（`mask` 非零）
/// 节点重置为初值，用于固定障碍/保护区。
pub struct PostTimestepMask {
    /// 掩码区域节点集合（与网格同形，非零即在掩码内）。
    pub mask: ArrayD<f64>,
    /// 初值（掩码内节点回到此值）。
    pub initial: ArrayD<f64>,
}

impl PostTimestep for PostTimestepMask {
    fn post_step(&mut self, _t: f64, y: &mut ArrayD<f64>) {
        Zip::from(y)
            .and(&self.mask)
            .and(&self.initial)
            .for_each(|v, m, init| {
                if *m != 0.0 {
                    *v = *init;
                }
            });
    }
}

/// 每步做一次迭代重初始化（`postTimestepReinit.m`）。默认迭代到
/// `tMax = 5 * max(dx)`（约五个节点宽的带），对应 MATLAB 不给
/// `reinitSteps` 时的默认。
pub struct PostTimestepReinit {
    pub grid: Grid,
    pub accuracy: ReinitAccuracy,
    /// 指定重初始化步数（`reinitSteps`）；None 表示按默认时间收敛。
    pub steps: Option<usize>,
    /// 平均更新量收敛阈值（`reinitErrorMax`，默认 1e-3）。
    pub error_max: f64,
}

impl PostTimestep for PostTimestepReinit {
    fn post_step(&mut self, _t: f64, y: &mut ArrayD<f64>) {
        let mut options = SignedDistanceOptions {
            accuracy: self.accuracy,
            error_max: self.error_max,
            ..Default::default()
        };
        match self.steps {
            Some(steps) => options.t_max_steps = Some(steps),
            None => {
                options.t_max = Some(5.0 * self.grid.dx().iter().fold(0.0f64, |m, v| m.max(*v)))
            }
        }
        *y = signed_distance_iterative(&self.grid, y, &options);
    }
}

/// 最短到达时间记录器（`postTimestepTTR.m`）：不改解，记录零等值面
/// 越过每个节点的时刻（相邻步之间线性插值），未到达节点保持无穷。
///
/// 与 MATLAB 版一致：初始时刻不算步进结果，若要记录 t₀ 时刻的零等值面，
/// 需在开始积分前手动调用一次 [`PostTimestepTtrRecorder::record`]。
pub struct PostTimestepTtrRecorder {
    /// 每个节点首次被零等值面穿越的时刻，初值无穷。
    pub ttr: ArrayD<f64>,
    last_t: f64,
    last_y: ArrayD<f64>,
}

impl PostTimestepTtrRecorder {
    pub fn new(grid: &Grid) -> Self {
        PostTimestepTtrRecorder {
            ttr: ArrayD::from_elem(grid.shape(), f64::INFINITY),
            last_t: f64::NAN,
            last_y: ArrayD::zeros(grid.shape()),
        }
    }

    /// 记录初始状态：零等值面内（φ ≤ 0）节点的 TTR 记为 `t`，其余无穷。
    pub fn record(&mut self, t: f64, y: &ArrayD<f64>) {
        Zip::from(&mut self.ttr).and(y).for_each(|v, cur| {
            if *cur <= 0.0 {
                *v = t;
            }
        });
        self.last_t = t;
        self.last_y = y.clone();
    }
}

impl PostTimestep for PostTimestepTtrRecorder {
    fn post_step(&mut self, t: f64, y: &mut ArrayD<f64>) {
        if self.last_t.is_nan() {
            // 未初始化：按 MATLAB 首次调用路径处理。
            let snapshot = y.clone();
            self.record(t, &snapshot);
            return;
        }
        // 上一步为正、本步 ≤ 0 的节点做线性插值求穿越时刻。
        let snapshot = y.clone();
        Zip::from(&mut self.ttr)
            .and(&snapshot)
            .and(&self.last_y)
            .for_each(|v, cur, prev| {
                if *cur <= 0.0 && *prev > 0.0 {
                    *v = self.last_t - (t - self.last_t) * *prev / (*cur - *prev);
                }
            });
        self.last_t = t;
        self.last_y = snapshot;
    }
}

/// 终止事件（`terminalEvent` 原型：`(t, y, tOld, yOld) -> 标量`，
/// 相邻两次调用返回值符号相反则终止积分）。
///
/// 只有收敛检测一种实现（`terminalEventConverge.m`），故用枚举而非 trait。
#[derive(Clone, Debug)]
pub enum TerminalEvent {
    /// 更新量范数降到容差以下触发变号（返回 -1），否则 +1
    /// （`terminalEventConverge.m` 第 79-93 行）。
    Converge {
        /// 绝对容差（默认 1e-6）。
        abs_tol: f64,
        /// 相对容差（默认 1e-3）。
        rel_tol: f64,
        norm: ConvergeNorm,
    },
}

/// 收敛度量范数（`terminalEventConverge.m` 第 39-43 行），
/// 按紧到松为逐点 / 最大 / 平均。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConvergeNorm {
    /// 全节点平均（MATLAB 默认）。
    Average,
    /// 全节点最大。
    Maximum,
    /// 每个节点单独检验（全体都低于容差才算收敛）。
    Pointwise,
}

impl TerminalEvent {
    /// 事件值，符号翻转即终止（由积分器检测）。
    pub fn event_value(&self, _t: f64, y: &ArrayD<f64>, _t_old: f64, y_old: &ArrayD<f64>) -> f64 {
        let Self::Converge {
            abs_tol,
            rel_tol,
            norm,
        } = self;
        let diff = Zip::from(y).and(y_old).map_collect(|a, b| a - b);
        let tol = |v: f64| -> f64 { (rel_tol * v).max(*abs_tol) };
        let converged = match norm {
            ConvergeNorm::Average => {
                let mean =
                    |a: &ArrayD<f64>| a.iter().map(|v| v.abs()).sum::<f64>() / a.len() as f64;
                mean(&diff) < tol(mean(y))
            }
            ConvergeNorm::Maximum => {
                let max = |a: &ArrayD<f64>| a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                max(&diff) < tol(max(y))
            }
            ConvergeNorm::Pointwise => Zip::from(&diff).and(y).all(|d, v| d.abs() < tol(v.abs())),
        };
        if converged {
            -1.0
        } else {
            1.0
        }
    }
}

/// 积分结果。MATLAB 版的 `tspan` 多时刻输出（`odeCFLmultipleSteps.m`）
/// 只是反复调用单段积分，需要时在调用侧循环即可，不再单独提供。
pub struct CflResult {
    /// 终止时刻（到达 `tf` 或事件触发，均返回实际时刻）。
    pub t: f64,
    /// 终止时刻的水平集函数。
    pub y: ArrayD<f64>,
    /// 实际推进的步数。
    pub steps: usize,
}

/// 子步组合器：消耗首级右端项，完成其余子步，返回新解与全部子步的
/// 最小步长上界（用于 CFL 违约告警）。
type Stage<'a> =
    dyn FnMut(&mut dyn Term, f64, &ArrayD<f64>, f64, TermRhs) -> (ArrayD<f64>, f64) + 'a;

/// 三个积分器共用的驱动循环（对应 odeCFL1/2/3 的主循环结构）。
fn cfl_integrate<T: Term>(
    term: &mut T,
    t0: f64,
    tf: f64,
    y0: ArrayD<f64>,
    options: &mut CflOptions,
    stage: &mut Stage<'_>,
) -> CflResult {
    // 子步 CFL 违约阈值：比用户系数多 20%，封顶 1.0（odeCFL3.m 第 85 行）。
    let safety = (1.2 * options.factor_cfl).min(1.0);
    // 到达终点的相对判据（odeCFL1.m 第 74 行）。
    let small = 100.0 * f64::EPSILON;

    let mut t = t0;
    let mut y = y0;
    let mut steps = 0usize;
    let mut event_old: Option<f64> = None;

    while tf - t >= small * tf.abs() {
        // 首级右端项决定本步步长，整步固定（odeCFL3.m 第 126-144 行）。
        let r1 = term.rhs(t, &y);
        let dt = (options.factor_cfl * r1.step_bound)
            .min(tf - t)
            .min(options.max_step);

        // 终止事件需要上一步的 (t, y)。
        let t_old = t;
        let y_old = options.terminal_event.as_ref().map(|_| y.clone());

        let (y_new, sb_min) = stage(term as &mut dyn Term, t, &y, dt, r1);
        y = y_new;
        t += dt;
        steps += 1;

        if dt > safety * sb_min {
            eprintln!("警告：子步 CFL 违约，dt = {dt:.6}，子步上界 {sb_min:.6}");
        }

        for post in &mut options.post_timestep {
            post.post_step(t, &mut y);
        }

        if options.single_step {
            break;
        }

        if let (Some(event), Some(y_old)) = (&options.terminal_event, &y_old) {
            let value = event.event_value(t, &y, t_old, y_old);
            if let Some(old) = event_old {
                if matlab_sign(value) != matlab_sign(old) {
                    break;
                }
            }
            event_old = Some(value);
        }
    }

    if options.stats {
        println!("\t{steps} steps from {t0} to {t}");
    }

    CflResult { t, y, steps }
}

/// `y ← y + dt·k`。
fn axpy(y: &ArrayD<f64>, k: &ArrayD<f64>, dt: f64) -> ArrayD<f64> {
    let mut out = y.clone();
    Zip::from(&mut out).and(k).for_each(|v, kk| *v += dt * kk);
    out
}

/// 一阶时间积分（前向 Euler，`odeCFL1.m`）。
pub fn ode_cfl1<T: Term>(
    term: &mut T,
    t0: f64,
    tf: f64,
    y0: ArrayD<f64>,
    options: &mut CflOptions,
) -> CflResult {
    let mut stage = |_term: &mut dyn Term, _t: f64, y: &ArrayD<f64>, dt: f64, r1: TermRhs| {
        (axpy(y, &r1.dphi_dt, dt), r1.step_bound)
    };
    cfl_integrate(term, t0, tf, y0, options, &mut stage)
}

/// 二阶 TVD Runge-Kutta（`odeCFL2.m`，Heun 型）：
/// `y1 = y + dt·k1`；`y2 = y1 + dt·k2(t+dt, y1)`；`y ← (y + y2)/2`。
pub fn ode_cfl2<T: Term>(
    term: &mut T,
    t0: f64,
    tf: f64,
    y0: ArrayD<f64>,
    options: &mut CflOptions,
) -> CflResult {
    let mut stage = |term: &mut dyn Term, t: f64, y: &ArrayD<f64>, dt: f64, r1: TermRhs| {
        let y1 = axpy(y, &r1.dphi_dt, dt);
        let r2 = term.rhs(t + dt, &y1);
        let y2 = axpy(&y1, &r2.dphi_dt, dt);
        let mut y_new = y.clone();
        Zip::from(&mut y_new)
            .and(&y2)
            .for_each(|v, v2| *v = 0.5 * (*v + *v2));
        (y_new, r1.step_bound.min(r2.step_bound))
    };
    cfl_integrate(term, t0, tf, y0, options, &mut stage)
}

/// 三阶 TVD Runge-Kutta（`odeCFL3.m`，Shu-Osher 三阶段格式）。
///
/// 子步（`odeCFL3.m` 第 146-265 行）：
///
/// 1. `y1 = y + dt·k1(t, y)`；
/// 2. `y2 = y1 + dt·k2(t + dt, y1)`；`y_half = (3y + y2)/4`；
/// 3. `y3 = y_half + dt·k3(t + dt/2, y_half)`；`y ← (y + 2·y3)/3`。
pub fn ode_cfl3<T: Term>(
    term: &mut T,
    t0: f64,
    tf: f64,
    y0: ArrayD<f64>,
    options: &mut CflOptions,
) -> CflResult {
    let mut stage = |term: &mut dyn Term, t: f64, y: &ArrayD<f64>, dt: f64, r1: TermRhs| {
        let y1 = axpy(y, &r1.dphi_dt, dt);
        let r2 = term.rhs(t + dt, &y1);
        let y2 = axpy(&y1, &r2.dphi_dt, dt);
        // t_half = (3t + t2)/4 = t + dt/2。
        let mut y_half = y.clone();
        Zip::from(&mut y_half)
            .and(&y2)
            .for_each(|v, v2| *v = 0.25 * (3.0 * *v + *v2));
        let r3 = term.rhs(t + 0.5 * dt, &y_half);
        let y3 = axpy(&y_half, &r3.dphi_dt, dt);
        let mut y_new = y.clone();
        Zip::from(&mut y_new)
            .and(&y3)
            .for_each(|v, v3| *v = (*v + 2.0 * *v3) / 3.0);
        let sb = r1.step_bound.min(r2.step_bound).min(r3.step_bound);
        (y_new, sb)
    };
    cfl_integrate(term, t0, tf, y0, options, &mut stage)
}
