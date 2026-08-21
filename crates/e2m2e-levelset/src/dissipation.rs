//! Lax-Friedrichs 耗散：对应 ToolboxLS 的
//! `ExplicitIntegration/Dissipation/artificialDissipationGLF/LLF/LLLF.m`。
//!
//! MATLAB 原型（`termLaxFriedrichs.m` 第 173-174 行的调用方式）：
//!
//! ```matlab
//! [diss, stepBound] = feval(schemeData.dissFunc, t, data, derivL, derivR, schemeData);
//! ```
//!
//! GLF 的算法（`artificialDissipationGLF.m` 第 72-102 行）：
//!
//! ```text
//! p_min[i] = min(min(derivL[i]), min(derivR[i]))   （全网格标量，广播）
//! p_max[i] = max(max(derivL[i]), max(derivR[i]))
//! diss     = Σ_i 0.5 * α_i * (derivR[i] - derivL[i])
//! stepBound = 1 / Σ_i max(α_i) / dx[i]
//! ```
//!
//! LLF 只把当前维 `i` 的包络换成节点邻域的逐点 min/max（其余维仍用全局
//! 标量，`artificialDissipationLLF.m` 第 96-118 行）；LLLF 把所有维都换成
//! 逐点包络（`artificialDissipationLLLF.m` 第 78-102 行）。

use crate::grid::Grid;
use crate::hamiltonian::Hamiltonian;
use ndarray::{ArrayD, Zip};

/// LF 耗散格式。MATLAB 的 `schemeData.dissFunc` 函数句柄物化为实现本
/// trait 的结构体。
pub trait Dissipation: Send + Sync {
    /// 计算逐节点耗散项与 CFL 步长上界。
    fn dissipation(
        &self,
        t: f64,
        grid: &Grid,
        phi: &ArrayD<f64>,
        deriv_l: &[ArrayD<f64>],
        deriv_r: &[ArrayD<f64>],
        ham: &dyn Hamiltonian,
    ) -> (ArrayD<f64>, f64);
}

/// 由耗散系数场与梯度差组装 `(diss, step_bound)`（三种格式共用的收尾）。
fn assemble(
    grid: &Grid,
    deriv_l: &[ArrayD<f64>],
    deriv_r: &[ArrayD<f64>],
    alphas: &[ArrayD<f64>],
) -> (ArrayD<f64>, f64) {
    let mut diss = ArrayD::zeros(grid.shape());
    let mut step_bound_inv = 0.0f64;
    for i in 0..grid.dim() {
        // diss += 0.5 * α_i * (derivR_i - derivL_i)
        Zip::from(&mut diss)
            .and(&alphas[i])
            .and(&deriv_r[i])
            .and(&deriv_l[i])
            .for_each(|d, a, r, l| *d += 0.5 * a * (r - l));
        step_bound_inv += alphas[i].iter().fold(0.0f64, |m, v| m.max(*v)) / grid.dx()[i];
    }
    (diss, 1.0 / step_bound_inv)
}

/// 全局 LF（`artificialDissipationGLF.m`）：梯度包络取全网格极值。
pub struct ArtificialDissipationGLF;

impl Dissipation for ArtificialDissipationGLF {
    fn dissipation(
        &self,
        t: f64,
        grid: &Grid,
        phi: &ArrayD<f64>,
        deriv_l: &[ArrayD<f64>],
        deriv_r: &[ArrayD<f64>],
        ham: &dyn Hamiltonian,
    ) -> (ArrayD<f64>, f64) {
        let dim = grid.dim();
        let shape = grid.shape();
        let mut p_min = Vec::with_capacity(dim);
        let mut p_max = Vec::with_capacity(dim);
        for i in 0..dim {
            let lo = deriv_l[i]
                .iter()
                .chain(deriv_r[i].iter())
                .fold(f64::INFINITY, |m, v| m.min(*v));
            let hi = deriv_l[i]
                .iter()
                .chain(deriv_r[i].iter())
                .fold(f64::NEG_INFINITY, |m, v| m.max(*v));
            p_min.push(ArrayD::from_elem(shape.clone(), lo));
            p_max.push(ArrayD::from_elem(shape.clone(), hi));
        }
        let alphas: Vec<ArrayD<f64>> = (0..dim)
            .map(|i| ham.partial_bound(t, grid, phi, &p_min, &p_max, i))
            .collect();
        assemble(grid, deriv_l, deriv_r, &alphas)
    }
}

/// 局部 LF（`artificialDissipationLLF.m`）：第 `i` 维的包络取该维左右
/// 导数的逐点 min/max，其余维用全网格标量。
pub struct ArtificialDissipationLLF;

impl Dissipation for ArtificialDissipationLLF {
    fn dissipation(
        &self,
        t: f64,
        grid: &Grid,
        phi: &ArrayD<f64>,
        deriv_l: &[ArrayD<f64>],
        deriv_r: &[ArrayD<f64>],
        ham: &dyn Hamiltonian,
    ) -> (ArrayD<f64>, f64) {
        let dim = grid.dim();
        let shape = grid.shape();
        // 全局标量包络（各维共用）。
        let g_min: Vec<f64> = (0..dim)
            .map(|i| {
                deriv_l[i]
                    .iter()
                    .chain(deriv_r[i].iter())
                    .fold(f64::INFINITY, |m, v| m.min(*v))
            })
            .collect();
        let g_max: Vec<f64> = (0..dim)
            .map(|i| {
                deriv_l[i]
                    .iter()
                    .chain(deriv_r[i].iter())
                    .fold(f64::NEG_INFINITY, |m, v| m.max(*v))
            })
            .collect();
        let broadcast_min: Vec<ArrayD<f64>> = g_min
            .iter()
            .map(|v| ArrayD::from_elem(shape.clone(), *v))
            .collect();
        let broadcast_max: Vec<ArrayD<f64>> = g_max
            .iter()
            .map(|v| ArrayD::from_elem(shape.clone(), *v))
            .collect();

        let mut alphas = Vec::with_capacity(dim);
        for i in 0..dim {
            let mut p_min = broadcast_min.clone();
            let mut p_max = broadcast_max.clone();
            // 当前维换成逐点包络。
            p_min[i] = Zip::from(&deriv_l[i])
                .and(&deriv_r[i])
                .map_collect(|l, r| (*l).min(*r));
            p_max[i] = Zip::from(&deriv_l[i])
                .and(&deriv_r[i])
                .map_collect(|l, r| (*l).max(*r));
            alphas.push(ham.partial_bound(t, grid, phi, &p_min, &p_max, i));
        }
        assemble(grid, deriv_l, deriv_r, &alphas)
    }
}

/// 局部局部 LF（`artificialDissipationLLLF.m`）：所有维的包络都取逐点
/// min/max。
pub struct ArtificialDissipationLLLF;

impl Dissipation for ArtificialDissipationLLLF {
    fn dissipation(
        &self,
        t: f64,
        grid: &Grid,
        phi: &ArrayD<f64>,
        deriv_l: &[ArrayD<f64>],
        deriv_r: &[ArrayD<f64>],
        ham: &dyn Hamiltonian,
    ) -> (ArrayD<f64>, f64) {
        let dim = grid.dim();
        let p_min: Vec<ArrayD<f64>> = (0..dim)
            .map(|i| {
                Zip::from(&deriv_l[i])
                    .and(&deriv_r[i])
                    .map_collect(|l, r| (*l).min(*r))
            })
            .collect();
        let p_max: Vec<ArrayD<f64>> = (0..dim)
            .map(|i| {
                Zip::from(&deriv_l[i])
                    .and(&deriv_r[i])
                    .map_collect(|l, r| (*l).max(*r))
            })
            .collect();
        let alphas: Vec<ArrayD<f64>> = (0..dim)
            .map(|i| ham.partial_bound(t, grid, phi, &p_min, &p_max, i))
            .collect();
        assemble(grid, deriv_l, deriv_r, &alphas)
    }
}
