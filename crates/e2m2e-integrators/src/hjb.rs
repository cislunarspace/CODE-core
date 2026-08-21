//! HJB 结构网格求解的 Python 绑定（issue #497）。
//!
//! 通用入口 [`solve_hjb_py`]：动力学标识 + 参数表，不为每种动力学各写
//! 一个绑定函数（ADR 0032 决策 3）。兼容包装
//! [`solve_planar_lowthrust_hjb_py`] 保持 geo-nrho `algorithm/dp.py`
//! 既有调用签名。
//!
//! 求解语义：终端代价 ψ 在 tf 给定，值函数满足 −V_t = H\*(x, ∇V)。
//! levelset 积分器只正向推进 D_τ φ = −H，故本层做时间反转：以
//! τ = tf − t 正向积分，喂给求解器的 Hamiltonian 取 −H\*（包一层
//! [`NegHamiltonian`]），返回的时刻换算回 t 并升序排列。
//!
//! 数值格式固定为 ENO2 + GLF 耗散 + odeCFL2 + 外插边界（ToolboxLS 示例
//! 的缺省组合，与 e2m2e-levelset 阶段 4 验证一致）。

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use ndarray::{ArrayD, IxDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use e2m2e_hjb_dynamics::{Cr3bpSynodic, PlanarDoubleIntegrator};
use e2m2e_levelset::derivative::UpwindFirstENO2;
use e2m2e_levelset::dissipation::ArtificialDissipationGLF;
use e2m2e_levelset::grid::{BoundaryCondition, Grid};
use e2m2e_levelset::hamiltonian::Hamiltonian;
use e2m2e_levelset::integrator::{ode_cfl2, CflOptions, PostTimestep};
use e2m2e_levelset::term::LaxFriedrichsTerm;

/// 快照占内存的上界：值函数时间序列按快照数降采样，防止高维网格的
/// 完整时间序列撑爆内存（40⁵ 网格单帧约 0.8 GB）。
const MAX_SNAPSHOT_BYTES: usize = 4 << 30;
/// 快照数上限。
const MAX_SNAPSHOTS: usize = 64;

/// 时间反转适配器：HJB 反向求解等价于以 −H\* 正向推进。
struct NegHamiltonian<H>(H);

impl<H: Hamiltonian> Hamiltonian for NegHamiltonian<H> {
    fn hamiltonian(
        &self,
        t: f64,
        grid: &Grid,
        phi: &ArrayD<f64>,
        p: &[ArrayD<f64>],
    ) -> ArrayD<f64> {
        -&self.0.hamiltonian(t, grid, phi, p)
    }

    fn partial_bound(
        &self,
        t: f64,
        grid: &Grid,
        phi: &ArrayD<f64>,
        p_min: &[ArrayD<f64>],
        p_max: &[ArrayD<f64>],
        dim: usize,
    ) -> ArrayD<f64> {
        // 包络是绝对值，取负不变。
        self.0.partial_bound(t, grid, phi, p_min, p_max, dim)
    }
}

/// 等距快照记录器：步数未知且可能远超快照上限，记录时按 stride 抽样，
/// 超限时 stride 加倍并抽稀已有快照，保证快照在时间上近似等距且
/// 内存有界（任意时刻至多 2 倍上限的快照驻留）。
struct SnapshotRecorder {
    stride: usize,
    counter: usize,
    max_snapshots: usize,
    taus: Vec<f64>,
    flat: Vec<f64>,
    nodes: usize,
}

impl SnapshotRecorder {
    fn new(max_snapshots: usize, nodes: usize) -> Self {
        Self {
            stride: 1,
            counter: 0,
            max_snapshots,
            taus: Vec::new(),
            flat: Vec::new(),
            nodes,
        }
    }

    fn record(&mut self, tau: f64, y: &ArrayD<f64>) {
        if self.counter.is_multiple_of(self.stride) {
            self.taus.push(tau);
            self.flat.extend(y.iter());
            if self.taus.len() > self.max_snapshots {
                self.decimate();
            }
        }
        self.counter += 1;
    }

    /// stride 加倍，已有快照隔帧丢弃。记录的计数原本是 stride 的倍数，
    /// 保留偶数序号后仍是新 stride 的倍数，抽样节奏一致。
    fn decimate(&mut self) {
        self.stride *= 2;
        let keep = self.taus.len().div_ceil(2);
        let mut taus = Vec::with_capacity(keep);
        let mut flat = Vec::with_capacity(keep * self.nodes);
        for (k, tau) in self.taus.iter().enumerate() {
            if k.is_multiple_of(2) {
                taus.push(*tau);
                flat.extend_from_slice(&self.flat[k * self.nodes..(k + 1) * self.nodes]);
            }
        }
        self.taus = taus;
        self.flat = flat;
    }
}

impl PostTimestep for SnapshotRecorder {
    fn post_step(&mut self, t: f64, y: &mut ArrayD<f64>) {
        self.record(t, y);
    }
}

/// Rc 句柄适配器：钩子交给积分器后，调用侧仍能取回记录（同
/// e2m2e-levelset tests/ttr.rs 的 TtrHook 模式）。
struct RecorderHandle(Rc<RefCell<SnapshotRecorder>>);

impl PostTimestep for RecorderHandle {
    fn post_step(&mut self, t: f64, y: &mut ArrayD<f64>) {
        self.0.borrow_mut().post_step(t, y);
    }
}

/// 求解结果：升序时刻、逐快照拼接的值函数（C 序）、实际步数。
struct HjbSolution {
    times: Vec<f64>,
    values: Vec<f64>,
    steps: usize,
}

fn solve_hjb<H: Hamiltonian>(
    grid: Grid,
    terminal: ArrayD<f64>,
    t0: f64,
    tf: f64,
    hamiltonian: H,
    cfl: f64,
    max_step: f64,
) -> HjbSolution {
    let nodes = terminal.len();
    let bytes_per_snapshot = nodes * std::mem::size_of::<f64>();
    let max_snapshots = (MAX_SNAPSHOT_BYTES / bytes_per_snapshot.max(1)).clamp(2, MAX_SNAPSHOTS);

    let recorder = Rc::new(RefCell::new(SnapshotRecorder::new(max_snapshots, nodes)));
    recorder.borrow_mut().record(0.0, &terminal);

    let mut term = LaxFriedrichsTerm::new(
        grid.clone(),
        NegHamiltonian(hamiltonian),
        UpwindFirstENO2,
        ArtificialDissipationGLF,
    );
    let mut options = CflOptions {
        factor_cfl: cfl,
        max_step,
        ..Default::default()
    };
    options
        .post_timestep
        .push(Box::new(RecorderHandle(recorder.clone())));

    // τ = tf − t：从终端代价正向推进 τ ∈ [0, tf − t0]。
    let result = ode_cfl2(&mut term, 0.0, tf - t0, terminal, &mut options);

    let recorder = recorder.borrow();
    let count = recorder.taus.len();
    // τ 升序 ↔ t = tf − τ 降序；时刻与快照一并反转为 t 升序。
    let mut times: Vec<f64> = recorder.taus.iter().map(|tau| tf - tau).collect();
    times.reverse();
    let mut values = vec![0.0; recorder.flat.len()];
    for k in 0..count {
        let src = &recorder.flat[k * nodes..(k + 1) * nodes];
        let dst = &mut values[(count - 1 - k) * nodes..(count - k) * nodes];
        dst.copy_from_slice(src);
    }

    HjbSolution {
        times,
        values,
        steps: result.steps,
    }
}

fn check_params(dynamics: &str, params: &HashMap<String, f64>, required: &[&str]) -> PyResult<()> {
    for key in required {
        if !params.contains_key(*key) {
            return Err(PyValueError::new_err(format!(
                "动力学 {dynamics} 缺少参数 {key}（需要：{required:?}）"
            )));
        }
    }
    for key in params.keys() {
        if !required.contains(&key.as_str()) {
            return Err(PyValueError::new_err(format!(
                "动力学 {dynamics} 不接受参数 {key}（需要：{required:?}）"
            )));
        }
    }
    Ok(())
}

fn check_control_params(params: &HashMap<String, f64>) -> PyResult<()> {
    let max_accel = params["max_accel"];
    let fuel_weight = params["fuel_weight"];
    if !max_accel.is_finite() || max_accel <= 0.0 {
        return Err(PyValueError::new_err("max_accel 必须为正的有限值"));
    }
    if !fuel_weight.is_finite() || fuel_weight < 0.0 {
        return Err(PyValueError::new_err("fuel_weight 必须为非负有限值"));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_grid_args(
    terminal: &[f64],
    minimum: &[f64],
    maximum: &[f64],
    shape: &[usize],
    t0: f64,
    tf: f64,
    cfl: f64,
    max_step: f64,
    dim: usize,
) -> PyResult<()> {
    if minimum.len() != dim || maximum.len() != dim || shape.len() != dim {
        return Err(PyValueError::new_err(format!(
            "网格维数必须为 {dim}（minimum/maximum/shape 长度一致）"
        )));
    }
    if shape.iter().any(|&n| n < 3) {
        return Err(PyValueError::new_err("每维节点数至少为 3"));
    }
    if minimum
        .iter()
        .zip(maximum.iter())
        .any(|(lo, hi)| !lo.is_finite() || !hi.is_finite() || hi <= lo)
    {
        return Err(PyValueError::new_err(
            "网格上下界必须为有限值且上界大于下界",
        ));
    }
    let nodes: usize = shape.iter().product();
    if terminal.len() != nodes {
        return Err(PyValueError::new_err(format!(
            "terminal 长度 {} 与网格节点数 {nodes} 不一致",
            terminal.len()
        )));
    }
    if terminal.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("terminal 含非有限值"));
    }
    if !t0.is_finite() || !tf.is_finite() || tf <= t0 {
        return Err(PyValueError::new_err("需要 t0 < tf 且均为有限值"));
    }
    if !cfl.is_finite() || cfl <= 0.0 || cfl > 1.0 {
        return Err(PyValueError::new_err("cfl 必须在 (0, 1] 内"));
    }
    if !max_step.is_finite() || max_step <= 0.0 {
        return Err(PyValueError::new_err("max_step 必须为正的有限值"));
    }
    Ok(())
}

fn build_result_dict<'py>(
    py: Python<'py>,
    grid: &Grid,
    solution: HjbSolution,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("times", solution.times)?;
    dict.set_item("values", solution.values)?;
    let axes: Vec<Vec<f64>> = (0..grid.dim()).map(|d| grid.axis(d).to_vec()).collect();
    dict.set_item("axes", axes)?;
    dict.set_item("steps", solution.steps)?;
    Ok(dict)
}

/// 通用 HJB 求解入口。
///
/// `dynamics` 为动力学标识，当前支持：
///
/// - ``"planar_double_integrator"``：平面双积分器，参数 drift_x、drift_y、
///   max_accel、fuel_weight；
/// - ``"cr3bp_synodic"``：地月会合系无量纲平面 CR3BP，参数 mu、
///   max_accel、fuel_weight。
///
/// `terminal` 为终端代价 ψ 的扁平数组（C 序，长度等于网格节点数）。
/// 返回字典：``times`` 升序时刻、``values`` 逐快照拼接的值函数
/// （C 序，形状 (len(times), ×shape)）、``axes`` 各维节点坐标、
/// ``steps`` 实际积分步数。快照按时间近似等距抽样，数量有上界，
/// 高维网格不会返回完整时间序列。
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn solve_hjb_py<'py>(
    py: Python<'py>,
    terminal: Vec<f64>,
    minimum: Vec<f64>,
    maximum: Vec<f64>,
    shape: Vec<usize>,
    t0: f64,
    tf: f64,
    dynamics: &str,
    params: HashMap<String, f64>,
    cfl: f64,
    max_step: f64,
) -> PyResult<Bound<'py, PyDict>> {
    // 期望维数由动力学标识决定，不是入口级常量：五维含质量等后续
    // 动力学加入时只动自己的分支（ADR 0032 决策 3 的通用入口定位）。
    let expected_dim = match dynamics {
        "planar_double_integrator" | "cr3bp_synodic" => 4,
        other => {
            return Err(PyValueError::new_err(format!(
                "未知动力学标识 {other}（支持：planar_double_integrator、cr3bp_synodic）"
            )));
        }
    };
    validate_grid_args(
        &terminal,
        &minimum,
        &maximum,
        &shape,
        t0,
        tf,
        cfl,
        max_step,
        expected_dim,
    )?;
    let grid =
        Grid::new(&minimum, &maximum, &shape).with_boundary_all(BoundaryCondition::Extrapolate);
    let terminal_arr = ArrayD::from_shape_vec(IxDyn(&shape), terminal)
        .map_err(|e| PyValueError::new_err(format!("terminal 形状不匹配：{e}")))?;

    let solution = match dynamics {
        "planar_double_integrator" => {
            check_params(
                dynamics,
                &params,
                &["drift_x", "drift_y", "max_accel", "fuel_weight"],
            )?;
            check_control_params(&params)?;
            if !params["drift_x"].is_finite() || !params["drift_y"].is_finite() {
                return Err(PyValueError::new_err("drift_accel 必须为有限值"));
            }
            let ham = PlanarDoubleIntegrator::new(
                [params["drift_x"], params["drift_y"]],
                params["max_accel"],
                params["fuel_weight"],
            );
            py.allow_threads(|| solve_hjb(grid.clone(), terminal_arr, t0, tf, ham, cfl, max_step))
        }
        "cr3bp_synodic" => {
            check_params(dynamics, &params, &["mu", "max_accel", "fuel_weight"])?;
            check_control_params(&params)?;
            let mu = params["mu"];
            if !mu.is_finite() || mu <= 0.0 || mu >= 1.0 {
                return Err(PyValueError::new_err("mu 必须在 (0, 1) 内"));
            }
            let ham = Cr3bpSynodic::new(mu, params["max_accel"], params["fuel_weight"]);
            py.allow_threads(|| solve_hjb(grid.clone(), terminal_arr, t0, tf, ham, cfl, max_step))
        }
        other => {
            unreachable!("动力学标识已在入口校验：{other}")
        }
    };
    // 参数校验先于构造完成，上面的 ::new 不会 panic。
    build_result_dict(py, &grid, solution)
}

/// geo-nrho 既有调用签名的兼容包装：平面双积分器低推力 HJB。
///
/// 参数顺序与 geo-nrho `algorithm/dp.py` 的 `solve_low_dim_hjb` 调用一致，
/// 语义等同 `solve_hjb_py(dynamics="planar_double_integrator", ...)`。
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn solve_planar_lowthrust_hjb_py<'py>(
    py: Python<'py>,
    terminal: Vec<f64>,
    minimum: Vec<f64>,
    maximum: Vec<f64>,
    shape: Vec<usize>,
    t0: f64,
    tf: f64,
    drift_accel: (f64, f64),
    max_accel: f64,
    fuel_weight: f64,
    cfl: f64,
    max_step: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let params = HashMap::from([
        ("drift_x".to_string(), drift_accel.0),
        ("drift_y".to_string(), drift_accel.1),
        ("max_accel".to_string(), max_accel),
        ("fuel_weight".to_string(), fuel_weight),
    ]);
    solve_hjb_py(
        py,
        terminal,
        minimum,
        maximum,
        shape,
        t0,
        tf,
        "planar_double_integrator",
        params,
        cfl,
        max_step,
    )
}
