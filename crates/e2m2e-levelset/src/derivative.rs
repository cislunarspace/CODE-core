//! 迎风一阶导数格式：对应 ToolboxLS 的 `SpatialDerivative/UpwindFirst/`。
//!
//! MATLAB 原型（`termLaxFriedrichs.m` 第 129 行的调用方式）：
//!
//! ```matlab
//! [derivL{i}, derivR{i}] = feval(schemeData.derivFunc, grid, data, i);
//! ```
//!
//! 实现方式：先沿目标维填充鬼单元，再把数组按该维切成 1 维 lane，
//! 逐 lane 套用与 MATLAB 逐行对应的一维修式。

use crate::boundary::pad_ghost_dim;
use crate::grid::Grid;
use ndarray::ArrayD;

/// 迎风导数格式。MATLAB 的 `schemeData.derivFunc` 函数句柄物化为实现本
/// trait 的结构体；高阶格式（ENO/WENO）基于均差表
/// （`dividedDifferenceTable.m`）选光滑模板。
pub trait UpwindDerivative: Send + Sync {
    /// 每侧需要的鬼单元层数（一阶 = 1，ENO2 = 2，ENO3/WENO5 = 3）。
    fn stencil_layers(&self) -> usize;

    /// 第 `dim` 维（0-based）的左、右迎风一阶导数，形状与 `data` 相同。
    fn deriv(&self, grid: &Grid, data: &ArrayD<f64>, dim: usize) -> (ArrayD<f64>, ArrayD<f64>);
}

/// 逐 lane 的一维修式：`(dx, n, 填充后的 lane, 左导数输出, 右导数输出)`。
type Scheme1d = fn(f64, usize, &[f64], &mut [f64], &mut [f64]);

/// C 序（行主序）各维元素步长。
fn c_strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

/// 通用驱动：填充 `layers` 层鬼单元，沿 `dim` 切 lane 逐条套用一维修式。
/// 本函数创建的三个数组（填充数组与两个输出）都是标准 C 布局，可直接用
/// 扁平切片做下标运算（ndarray 0.16 的 lanes 迭代器不支持动态维度）。
fn deriv_lanes(
    grid: &Grid,
    data: &ArrayD<f64>,
    dim: usize,
    layers: usize,
    scheme: Scheme1d,
) -> (ArrayD<f64>, ArrayD<f64>) {
    let padded = pad_ghost_dim(grid, data, dim, layers);
    let n = grid.n()[dim];
    let dx = grid.dx()[dim];
    let mut dl = ArrayD::zeros(grid.shape());
    let mut dr = ArrayD::zeros(grid.shape());

    let pshape = padded.shape().to_vec();
    let oshape = grid.shape();
    let pstride = c_strides(&pshape);
    let ostride = c_strides(&oshape);

    let pslice = padded.as_slice().expect("填充数组为标准布局");
    let (dls, drs) = (
        dl.as_slice_mut().expect("输出数组为标准布局"),
        dr.as_slice_mut().expect("输出数组为标准布局"),
    );

    // 除 dim 外各维的大小的 C 序枚举。
    let outer: Vec<usize> = oshape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != dim)
        .map(|(_, v)| *v)
        .collect();
    let num_lanes: usize = outer.iter().product();
    let mut om = vec![0usize; outer.len()];
    let mut left = vec![0.0; n];
    let mut right = vec![0.0; n];
    for o in 0..num_lanes {
        // o → 各外维坐标（C 序 → 多索引）。
        let mut r = o;
        for k in (0..outer.len()).rev() {
            om[k] = r % outer[k];
            r /= outer[k];
        }
        // 组装基址（dim 维坐标为 0）。
        let mut k = 0;
        let (mut pbase, mut obase) = (0usize, 0usize);
        for i in 0..oshape.len() {
            if i == dim {
                continue;
            }
            pbase += om[k] * pstride[i];
            obase += om[k] * ostride[i];
            k += 1;
        }
        // 填充后的整条 lane（含两侧鬼单元）。
        let m = pshape[dim];
        let g: Vec<f64> = (0..m).map(|j| pslice[pbase + j * pstride[dim]]).collect();
        scheme(dx, n, &g, &mut left, &mut right);
        for j in 0..n {
            dls[obase + j * ostride[dim]] = left[j];
            drs[obase + j * ostride[dim]] = right[j];
        }
    }
    (dl, dr)
}

/// 一阶迎风（`upwindFirstFirst.m`）：填充一层鬼单元后取相邻差分，
/// 左导数取差分序列前 n 个、右导数取后 n 个（源码第 57-70 行）。
fn first_1d(dx: f64, n: usize, g: &[f64], dl: &mut [f64], dr: &mut [f64]) {
    let inv = 1.0 / dx;
    for j in 0..n {
        dl[j] = inv * (g[j + 1] - g[j]);
        dr[j] = inv * (g[j + 2] - g[j + 1]);
    }
}

/// 一阶迎风格式。
pub struct UpwindFirstFirst;

impl UpwindDerivative for UpwindFirstFirst {
    fn stencil_layers(&self) -> usize {
        1
    }

    fn deriv(&self, grid: &Grid, data: &ArrayD<f64>, dim: usize) -> (ArrayD<f64>, ArrayD<f64>) {
        deriv_lanes(grid, data, dim, 1, first_1d)
    }
}

/// 二阶 ENO（`upwindFirstENO2.m`）：两个候选二阶近似中选二阶均差
/// 模较小的那个。
fn eno2_1d(dx: f64, n: usize, g: &[f64], dl: &mut [f64], dr: &mut [f64]) {
    // 未去头的一阶差分（m - 1 = n + 3 个）。
    let d1f: Vec<f64> = (0..n + 3).map(|j| (g[j + 1] - g[j]) / dx).collect();
    // 去掉首尾各一个得 D1（n + 1 个，对应半节点导数）。
    let d1 = &d1f[1..n + 2];
    // 二阶均差（n + 2 个）。
    let d2: Vec<f64> = (0..n + 2)
        .map(|j| 0.5 / dx * (d1f[j + 1] - d1f[j]))
        .collect();
    for k in 0..n {
        let dl1 = d1[k] + dx * d2[k];
        let dl2 = d1[k] + dx * d2[k + 1];
        let dr1 = d1[k + 1] - dx * d2[k + 1];
        let dr2 = d1[k + 1] - dx * d2[k + 2];
        let smaller = d2[k].abs() < d2[k + 1].abs();
        let smaller_r = d2[k + 1].abs() < d2[k + 2].abs();
        dl[k] = if smaller { dl1 } else { dl2 };
        dr[k] = if smaller_r { dr1 } else { dr2 };
    }
}

/// 二阶 ENO 格式。
pub struct UpwindFirstENO2;

impl UpwindDerivative for UpwindFirstENO2 {
    fn stencil_layers(&self) -> usize {
        2
    }

    fn deriv(&self, grid: &Grid, data: &ArrayD<f64>, dim: usize) -> (ArrayD<f64>, ArrayD<f64>) {
        deriv_lanes(grid, data, dim, 2, eno2_1d)
    }
}

/// ENO3 的三个候选近似 + 未去头的一阶均差（WENO5 权重也要用）。
struct Eno3Candidates {
    dl: [Vec<f64>; 3],
    dr: [Vec<f64>; 3],
    /// 未去头的一阶均差（n + 5 个）。
    d1f: Vec<f64>,
    /// 去头后的二阶均差（n + 2 个）。
    d2s: Vec<f64>,
    /// 三阶均差（n + 3 个，MATLAB 不去头）。
    d3f: Vec<f64>,
}

/// `upwindFirstENO3aHelper.m`（stencil = 3，三层鬼单元）。
fn eno3_candidates(dx: f64, n: usize, g: &[f64]) -> Eno3Candidates {
    let d1f: Vec<f64> = (0..n + 5).map(|j| (g[j + 1] - g[j]) / dx).collect(); // m - 1 = n + 5
    let d2f: Vec<f64> = (0..n + 4)
        .map(|j| 0.5 / dx * (d1f[j + 1] - d1f[j]))
        .collect(); // n + 4
    let d3f: Vec<f64> = (0..n + 3)
        .map(|j| (d2f[j + 1] - d2f[j]) / (3.0 * dx))
        .collect(); // n + 3

    let d1 = &d1f[2..n + 3]; // 去掉首尾各两个，n + 1 个
    let d2s = &d2f[1..n + 3]; // 去掉首尾各一个，n + 2 个

    let mut dl = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    let mut dr = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    let dx2 = dx * dx;
    for k in 0..n {
        // 一阶基底：左近似取左半差分，右近似取右半差分。
        let (bl, br) = (d1[k], d1[k + 1]);
        // 二阶项（源码第 136-152 行）。
        dl[0][k] = bl + dx * d2s[k];
        dl[1][k] = bl + dx * d2s[k];
        dl[2][k] = bl + dx * d2s[k + 1];
        dr[0][k] = br - dx * d2s[k + 1];
        dr[1][k] = br - dx * d2s[k + 1];
        dr[2][k] = br - dx * d2s[k + 2];
        // 三阶项（源码第 169-187 行）。
        dl[0][k] += 2.0 * dx2 * d3f[k];
        dl[1][k] += 2.0 * dx2 * d3f[k + 1];
        dl[2][k] -= dx2 * d3f[k + 2];
        dr[0][k] -= dx2 * d3f[k + 1];
        dr[1][k] -= dx2 * d3f[k + 2];
        dr[2][k] += 2.0 * dx2 * d3f[k + 3];
    }
    Eno3Candidates {
        dl,
        dr,
        d1f,
        d2s: d2s.to_vec(),
        d3f,
    }
}

/// 三阶 ENO（`upwindFirstENO3.m`，默认走 3a 变体）：先按二阶均差的
/// 模选方向，再在该方向上按三阶均差的模选模板。
fn eno3_1d(dx: f64, n: usize, g: &[f64], dl: &mut [f64], dr: &mut [f64]) {
    let c = eno3_candidates(dx, n, g);
    // smaller_l2：|D2s[k]| < |D2s[k+1]|，k = 0..n（n + 1 个）。
    let smaller_l2: Vec<bool> = (0..n + 1)
        .map(|k| c.d2s[k].abs() < c.d2s[k + 1].abs())
        .collect();
    // smaller_t：|D3[k]| < |D3[k+1]|，k = 0..n+1（n + 2 个）。
    let smaller_t: Vec<bool> = (0..n + 2)
        .map(|k| c.d3f[k].abs() < c.d3f[k + 1].abs())
        .collect();

    let pick = |k: usize| -> usize {
        // 返回 0/1/2 对应候选 LL / M / RR（源码第 121-141 行）。
        let (ll, m) = (
            smaller_t[k] && smaller_l2[k],
            (smaller_t[k + 1] && !smaller_l2[k]) || (!smaller_t[k] && smaller_l2[k]),
        );
        if ll {
            0
        } else if m {
            1
        } else {
            2
        }
    };
    for k in 0..n {
        dl[k] = c.dl[pick(k)][k];
        dr[k] = c.dr[pick(k + 1)][k];
    }
}

/// 三阶 ENO 格式（默认 3a 变体）。
pub struct UpwindFirstENO3;

impl UpwindDerivative for UpwindFirstENO3 {
    fn stencil_layers(&self) -> usize {
        3
    }

    fn deriv(&self, grid: &Grid, data: &ArrayD<f64>, dim: usize) -> (ArrayD<f64>, ArrayD<f64>) {
        deriv_lanes(grid, data, dim, 3, eno3_1d)
    }
}

/// 五阶 WENO（`upwindFirstWENO5.m` → `upwindFirstWENO5a.m`）：三个 ENO3
/// 候选按光滑度加权。ε 采用源码第 63 行实际启用的 `maxOverGrid` 方案：
/// `1e-6 * max(D1²) + 1e-99`。
fn weno5_1d(dx: f64, n: usize, g: &[f64], dl: &mut [f64], dr: &mut [f64]) {
    let c = eno3_candidates(dx, n, g);
    let eps = 1e-6 * c.d1f.iter().map(|v| v * v).fold(0.0, f64::max) + 1e-99;

    // 光滑度指示子（源码第 107-123 行），p 为 v₁ = D1f[p] 的下标。
    let smooth = |p: usize| -> [f64; 3] {
        let (v1, v2, v3, v4, v5) = (
            c.d1f[p],
            c.d1f[p + 1],
            c.d1f[p + 2],
            c.d1f[p + 3],
            c.d1f[p + 4],
        );
        let s1 = (13.0 / 12.0) * (v1 - 2.0 * v2 + v3).powi(2)
            + 0.25 * (v1 - 4.0 * v2 + 3.0 * v3).powi(2);
        let s2 = (13.0 / 12.0) * (v2 - 2.0 * v3 + v4).powi(2) + 0.25 * (v2 - v4).powi(2);
        let s3 = (13.0 / 12.0) * (v3 - 2.0 * v4 + v5).powi(2)
            + 0.25 * (3.0 * v3 - 4.0 * v4 + v5).powi(2);
        [s1, s2, s3]
    };
    let weight = |s: [f64; 3], w: [f64; 3], d: [f64; 3]| -> f64 {
        let alpha: Vec<f64> = (0..3).map(|i| w[i] / (s[i] + eps).powi(2)).collect();
        let sum: f64 = alpha.iter().sum();
        (0..3).map(|i| alpha[i] * d[i]).sum::<f64>() / sum
    };
    // 线性权重（源码第 133、146 行）。
    let (wl, wr) = ([0.1, 0.6, 0.3], [0.3, 0.6, 0.1]);
    for k in 0..n {
        dl[k] = weight(smooth(k), wl, [c.dl[0][k], c.dl[1][k], c.dl[2][k]]);
        dr[k] = weight(smooth(k + 1), wr, [c.dr[0][k], c.dr[1][k], c.dr[2][k]]);
    }
}

/// 五阶 WENO 格式。
pub struct UpwindFirstWENO5;

impl UpwindDerivative for UpwindFirstWENO5 {
    fn stencil_layers(&self) -> usize {
        3
    }

    fn deriv(&self, grid: &Grid, data: &ArrayD<f64>, dim: usize) -> (ArrayD<f64>, ArrayD<f64>) {
        deriv_lanes(grid, data, dim, 3, weno5_1d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 周期网格上 sin(x) 的导数逼近误差（左右导数的最大模差）。
    pub(super) fn max_deriv_error(scheme: &dyn UpwindDerivative, n: usize) -> f64 {
        let grid = Grid::new(&[0.0], &[2.0 * std::f64::consts::PI], &[n]);
        let data = ndarray::Array1::from(grid.axis(0).iter().map(|x| x.sin()).collect::<Vec<_>>())
            .into_dyn();
        let (dl, dr) = scheme.deriv(&grid, &data, 0);
        let mut err = 0.0f64;
        for (i, x) in grid.axis(0).iter().enumerate() {
            let exact = x.cos();
            err = err.max((dl[i] - exact).abs()).max((dr[i] - exact).abs());
        }
        err
    }

    pub(super) fn observed_order(scheme: &dyn UpwindDerivative, n1: usize, n2: usize) -> f64 {
        let (e1, e2) = (max_deriv_error(scheme, n1), max_deriv_error(scheme, n2));
        (e1 / e2).log2() / (n2 as f64 / n1 as f64).log2()
    }

    #[test]
    fn 各阶格式的收敛阶() {
        assert!(
            observed_order(&UpwindFirstFirst, 40, 80) > 0.8,
            "一阶格式应接近 1 阶"
        );
        assert!(
            observed_order(&UpwindFirstENO2, 40, 80) > 1.8,
            "ENO2 应接近 2 阶"
        );
        assert!(
            observed_order(&UpwindFirstENO3, 40, 80) > 2.5,
            "ENO3 应接近 3 阶"
        );
        assert!(
            observed_order(&UpwindFirstWENO5, 40, 80) > 4.0,
            "WENO5 应接近 5 阶"
        );
    }

    #[test]
    fn 二维逐维求导() {
        // f(x, y) = sin x + cos y，对第 0 维求导应为 cos x。
        let grid = Grid::new(&[0.0, 0.0], &[2.0 * std::f64::consts::PI; 2], &[32, 17]);
        let mut data = ArrayD::zeros(grid.shape());
        for (idx, v) in data.indexed_iter_mut() {
            let (x, y) = (grid.axis(0)[idx[0]], grid.axis(1)[idx[1]]);
            *v = x.sin() + y.cos();
        }
        let (dl, _) = UpwindFirstFirst.deriv(&grid, &data, 0);
        // 一阶格式在 32 节点下误差应小于 dx = 2π/32 ≈ 0.2。
        let mut worst = 0.0f64;
        for (idx, v) in dl.indexed_iter() {
            let x = grid.axis(0)[idx[0]];
            worst = worst.max((*v - x.cos()).abs());
        }
        assert!(worst < 0.2, "一阶导数误差 {worst} 应小于 dx");
    }
}
