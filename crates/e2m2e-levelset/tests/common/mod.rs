// 各测试二进制按需包含本模块，未用到的部分不算死代码。
#![allow(dead_code)]

//! 各集成测试共用的哈密顿量实现与解析解（对应 MATLAB 示例中的
//! hamFunc / partialFunc 子函数）。平流哈密顿量直接复用 crate 的
//! [`e2m2e_levelset::hamiltonian::Advection`]。

use e2m2e_levelset::grid::Grid;
use e2m2e_levelset::hamiltonian::Hamiltonian;
use ndarray::{ArrayD, Zip};

/// 一维无粘 Burgers `H = p²/2`（OsherShu burgersLF 的哈密顿量）。
pub struct Burgers1d;

impl Hamiltonian for Burgers1d {
    fn hamiltonian(
        &self,
        _t: f64,
        _grid: &Grid,
        _phi: &ArrayD<f64>,
        p: &[ArrayD<f64>],
    ) -> ArrayD<f64> {
        0.5 * &(&p[0] * &p[0])
    }

    fn partial_bound(
        &self,
        _t: f64,
        _grid: &Grid,
        _phi: &ArrayD<f64>,
        p_min: &[ArrayD<f64>],
        p_max: &[ArrayD<f64>],
        dim: usize,
    ) -> ArrayD<f64> {
        assert_eq!(dim, 0, "一维 Burgers 只有第 0 维");
        // ∂H/∂p = p，在 [p_min, p_max] 上的极值为两端绝对值的最大者。
        let lo = p_min[0].mapv(f64::abs);
        let hi = p_max[0].mapv(f64::abs);
        Zip::from(&lo).and(&hi).map_collect(|a, b| (*a).max(*b))
    }
}

/// 双积分器（doubleIntegratorTTR.m 第 356-384 行）：
/// `H = -(x₂·p₁ - inputBound·|p₂|)`。
pub struct DoubleIntegrator {
    pub input_bound: f64,
}

impl Hamiltonian for DoubleIntegrator {
    fn hamiltonian(
        &self,
        _t: f64,
        grid: &Grid,
        _phi: &ArrayD<f64>,
        p: &[ArrayD<f64>],
    ) -> ArrayD<f64> {
        // x₂ 场：各节点第 1 维坐标。
        // hamValue = -(x₂·p₁ - a·|p₂|) = -x₂·p₁ + a·|p₂|
        // （doubleIntegratorTTR.m 第 383-384 行）。
        let x2 = ArrayD::from_shape_fn(grid.shape(), |idx| grid.axis(1)[idx[1]]);
        Zip::from(&x2).and(&p[0]).map_collect(|v, p1| -(v * p1))
            + self.input_bound * &p[1].mapv(f64::abs)
    }

    fn partial_bound(
        &self,
        _t: f64,
        grid: &Grid,
        _phi: &ArrayD<f64>,
        _p_min: &[ArrayD<f64>],
        _p_max: &[ArrayD<f64>],
        dim: usize,
    ) -> ArrayD<f64> {
        match dim {
            // ∂H/∂p₁ = -x₂ → |α| = |x₂|。
            0 => ArrayD::from_shape_fn(grid.shape(), |idx| grid.axis(1)[idx[1]].abs()),
            // ∂H/∂p₂ = ±inputBound。
            1 => ArrayD::from_elem(grid.shape(), self.input_bound.abs()),
            _ => panic!("双积分器只有两维"),
        }
    }
}

/// 双积分器到原点的最短时间解析解（analyticDoubleIntegratorTTR.m）。
/// `ttr = -0.5·x₂·|x₂|` 为切换曲线。
pub fn analytic_double_integrator_ttr(x1: f64, x2: f64) -> f64 {
    let switch = -0.5 * x2 * x2.abs();
    if x1 > switch {
        x2 + (4.0 * x1 + 2.0 * x2 * x2).sqrt()
    } else if x1 < switch {
        -x2 + (-4.0 * x1 + 2.0 * x2 * x2).sqrt()
    } else {
        x2.abs()
    }
}
