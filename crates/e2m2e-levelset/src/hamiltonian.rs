//! 用户提供的哈密顿量：对应 ToolboxLS 的 `schemeData.hamFunc` 与
//! `schemeData.partialFunc` 两个回调（协议见 `termLaxFriedrichs.m`
//! 第 52-71 行与 `artificialDissipationGLF.m` 第 34-47 行的注释）。
//!
//! HJ PDE 为
//!
//! ```text
//! D_t φ = -H(x, t, φ, ∇φ)
//! ```
//!
//! MATLAB 中这两个回调配合弱类型 `schemeData` 使用（用户把动力学参数
//! 塞进自由字段，`hamFunc` 还可借助 `nargout` 探测返回修改后的
//! `schemeData`）。Rust 里的对应做法：把动力学参数放进实现本 trait 的
//! 结构体字段，`hamFunc` 的"原位修改 schemeData"能力不再需要——需要
//! 随时间演化的状态本就属于哈密顿量的实现方，用普通字段表达。

use crate::grid::Grid;
use ndarray::ArrayD;

/// 解析哈密顿量 `H(x, t, φ, p)` 及其代价导数包络。
pub trait Hamiltonian: Send + Sync {
    /// 在全部网格节点上计算 `H(t, x, p)`，返回形状与网格相同。
    ///
    /// `p` 是各维中心差分梯度（`termLaxFriedrichs.m` 第 130 行
    /// `derivC = 0.5 * (derivL + derivR)` 传入 `hamFunc` 的 `deriv`），
    /// `p[i]` 形状与 `phi` 相同。节点坐标经 `grid.axis(i)` 获取。
    fn hamiltonian(&self, t: f64, grid: &Grid, phi: &ArrayD<f64>, p: &[ArrayD<f64>])
        -> ArrayD<f64>;

    /// 第 `dim` 维的耗散系数包络（逐节点，形状与网格相同）：
    ///
    /// ```text
    /// α_dim(x) = max_{p ∈ [p_min, p_max]} |∂H/∂p_dim|
    /// ```
    ///
    /// `p_min` / `p_max` 是耗散格式选定的梯度上下界——GLF 为全网格极值
    /// 的广播，LLF/LLLF 为节点邻域极值（见各 `artificialDissipation*.m`）。
    /// 线性哈密顿量（∂H/∂p 与 p 无关）可直接返回常值场。
    fn partial_bound(
        &self,
        t: f64,
        grid: &Grid,
        phi: &ArrayD<f64>,
        p_min: &[ArrayD<f64>],
        p_max: &[ArrayD<f64>],
        dim: usize,
    ) -> ArrayD<f64>;
}

/// 常速平流 `H = Σᵢ vᵢ·pᵢ`（PDE 为 `φ_t + v·∇φ = 0`），
/// 最简单实用的 [`Hamiltonian`] 实现。
pub struct Advection {
    /// 各维速度。
    pub velocity: Vec<f64>,
}

impl Hamiltonian for Advection {
    fn hamiltonian(
        &self,
        _t: f64,
        grid: &Grid,
        _phi: &ArrayD<f64>,
        p: &[ArrayD<f64>],
    ) -> ArrayD<f64> {
        let mut out = ArrayD::zeros(grid.shape());
        for (i, v) in self.velocity.iter().enumerate() {
            ndarray::Zip::from(&mut out)
                .and(&p[i])
                .for_each(|o, d| *o += v * d);
        }
        out
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
        ArrayD::from_elem(grid.shape(), self.velocity[dim].abs())
    }
}
