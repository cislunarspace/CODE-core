//! 鬼单元填充：对应 ToolboxLS 的 `addGhost*.m`。
//!
//! MATLAB 原型（见 `upwindFirstFirst.m` 第 44 行的调用方式）：
//!
//! ```matlab
//! gdata = feval(grid.bdry{dim}, data, dim, stencil, grid.bdryData{dim});
//! ```
//!
//! 即只沿第 `dim` 维在数据两侧各加 `layers` 层鬼单元，其余维不变。

use crate::grid::{BoundaryCondition, Grid};
use ndarray::{ArrayD, Dimension};

/// 沿第 `dim` 维为 `data` 两侧各填充 `layers` 层鬼单元，按该维边界条件取值。
pub fn pad_ghost_dim(grid: &Grid, data: &ArrayD<f64>, dim: usize, layers: usize) -> ArrayD<f64> {
    grid.check_data(data);
    match grid.boundary(dim) {
        BoundaryCondition::Periodic => pad_periodic(grid, data, dim, layers),
        BoundaryCondition::Dirichlet(value) => pad_dirichlet(grid, data, dim, layers, *value),
        BoundaryCondition::Neumann(value) => pad_neumann(grid, data, dim, layers, *value),
        BoundaryCondition::Extrapolate => pad_extrapolate(grid, data, dim, layers),
        BoundaryCondition::Extrapolate2 => pad_extrapolate2(grid, data, dim, layers),
    }
}

/// 生成沿 `dim` 维加长 `2 * layers` 的全零输出数组。
fn padded_zeros(grid: &Grid, dim: usize, layers: usize) -> ArrayD<f64> {
    let mut shape = grid.shape();
    shape[dim] += 2 * layers;
    ArrayD::zeros(shape)
}

/// 把动态维度索引拷成 Vec（ndarray 0.16 的 Dim 没有直接的 slice 视图）。
fn idx_vec(idx: &ndarray::IxDyn) -> Vec<usize> {
    (0..idx.ndim()).map(|d| idx[d]).collect()
}

/// MATLAB 的 sign()：0 返回 0（f64::signum 对 0 返回 1，不可用）。
pub(crate) fn matlab_sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// `addGhostPeriodic.m`：鬼单元从对侧边界回卷。
fn pad_periodic(grid: &Grid, data: &ArrayD<f64>, dim: usize, layers: usize) -> ArrayD<f64> {
    let n = grid.n()[dim];
    let mut out = padded_zeros(grid, dim, layers);
    for (idx, v) in out.indexed_iter_mut() {
        let j = (idx[dim] as isize - layers as isize).rem_euclid(n as isize);
        let mut s = idx_vec(&idx);
        s[dim] = j as usize;
        *v = data[&s[..]];
    }
    out
}

/// `addGhostDirichlet.m`：鬼单元取常值（上下侧同值，即 MATLAB 只给
/// `lowerValue` 时的默认行为）。
fn pad_dirichlet(
    grid: &Grid,
    data: &ArrayD<f64>,
    dim: usize,
    layers: usize,
    value: f64,
) -> ArrayD<f64> {
    let n = grid.n()[dim];
    let mut out = padded_zeros(grid, dim, layers);
    for (idx, v) in out.indexed_iter_mut() {
        *v = if idx[dim] < layers || idx[dim] >= n + layers {
            value
        } else {
            let mut s = idx_vec(&idx);
            s[dim] = idx[dim] - layers;
            data[&s[..]]
        };
    }
    out
}

/// `addGhostNeumann.m`：法向导数为 `value`（按"每节点数据增量"理解，
/// 与 MATLAB 一致地不除以 dx），即鬼单元 = 边缘节点 + gap * value。
fn pad_neumann(
    grid: &Grid,
    data: &ArrayD<f64>,
    dim: usize,
    layers: usize,
    value: f64,
) -> ArrayD<f64> {
    let n = grid.n()[dim];
    let mut out = padded_zeros(grid, dim, layers);
    for (idx, v) in out.indexed_iter_mut() {
        let i = idx[dim];
        *v = if (layers..n + layers).contains(&i) {
            let mut s = idx_vec(&idx);
            s[dim] = i - layers;
            data[&s[..]]
        } else {
            // gap：距内缘第一个真实节点的层数（最内侧鬼单元 gap = 1）。
            let (edge, gap) = if i < layers {
                (0usize, layers - i)
            } else {
                (n - 1, i - (n + layers) + 1)
            };
            let mut s = idx_vec(&idx);
            s[dim] = edge;
            data[&s[..]] + gap as f64 * value
        };
    }
    out
}

/// `addGhostExtrapolate.m`：一阶线性外推。斜率符号调整为与边缘节点
/// 数据同号（对应 MATLAB 默认 slopeMultiplier = +1，即远离零等值面）。
fn pad_extrapolate(grid: &Grid, data: &ArrayD<f64>, dim: usize, layers: usize) -> ArrayD<f64> {
    let n = grid.n()[dim];
    let mut out = padded_zeros(grid, dim, layers);
    for (idx, v) in out.indexed_iter_mut() {
        let i = idx[dim];
        *v = if (layers..n + layers).contains(&i) {
            let mut s = idx_vec(&idx);
            s[dim] = i - layers;
            data[&s[..]]
        } else {
            let (e0, e1, gap) = if i < layers {
                (0usize, 1usize, layers - i)
            } else {
                (n - 1, n - 2, i - (n + layers) + 1)
            };
            let mut s0 = idx_vec(&idx);
            s0[dim] = e0;
            let mut s1 = idx_vec(&idx);
            s1[dim] = e1;
            let d0 = data[&s0[..]];
            let slope = matlab_sign(d0) * (d0 - data[&s1[..]]).abs();
            d0 + gap as f64 * slope
        };
    }
    out
}

/// `addGhostExtrapolate2.m`：二阶外推（一次斜率 + 二阶差分修正；
/// MATLAB 源码中向零等值面调整符号的分支被禁用，此处同样不做调整）。
fn pad_extrapolate2(grid: &Grid, data: &ArrayD<f64>, dim: usize, layers: usize) -> ArrayD<f64> {
    let n = grid.n()[dim];
    let mut out = padded_zeros(grid, dim, layers);
    for (idx, v) in out.indexed_iter_mut() {
        let i = idx[dim];
        *v = if (layers..n + layers).contains(&i) {
            let mut s = idx_vec(&idx);
            s[dim] = i - layers;
            data[&s[..]]
        } else {
            let (e0, e1, e2, gap) = if i < layers {
                (0usize, 1usize, 2usize, layers - i)
            } else {
                (n - 1, n - 2, n - 3, i - (n + layers) + 1)
            };
            let at = |e: usize| {
                let mut s = idx_vec(&idx);
                s[dim] = e;
                data[&s[..]]
            };
            let (d0, d1, d2) = (at(e0), at(e1), at(e2));
            let g = gap as f64;
            d0 + g * (d0 - d1) + 0.5 * g * (g + 1.0) * (d0 - 2.0 * d1 + d2)
        };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid_1d(n: usize, bc: BoundaryCondition) -> Grid {
        Grid::new(&[0.0], &[1.0], &[n]).with_boundary_all(bc)
    }

    #[test]
    fn 周期回卷() {
        let g = grid_1d(5, BoundaryCondition::Periodic);
        let data = ndarray::Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]).into_dyn();
        let out = pad_ghost_dim(&g, &data, 0, 2);
        let v = out.iter().cloned().collect::<Vec<f64>>();
        assert_eq!(v, vec![4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0]);
    }

    #[test]
    fn 常值与斜率边界() {
        let g = grid_1d(4, BoundaryCondition::Dirichlet(9.0));
        let data = ndarray::Array1::from(vec![1.0, 2.0, 3.0, 4.0]).into_dyn();
        let out = pad_ghost_dim(&g, &data, 0, 1);
        let v = out.iter().cloned().collect::<Vec<f64>>();
        assert_eq!(v, vec![9.0, 1.0, 2.0, 3.0, 4.0, 9.0]);

        // Neumann 斜率 0.5：下侧 1 + 1*0.5，上侧 4 + 1*0.5
        let g = grid_1d(4, BoundaryCondition::Neumann(0.5));
        let out = pad_ghost_dim(&g, &data, 0, 1);
        let v = out.iter().cloned().collect::<Vec<f64>>();
        assert_eq!(v, vec![1.5, 1.0, 2.0, 3.0, 4.0, 4.5]);
    }

    #[test]
    fn 一二阶外推的语义() {
        let data: ArrayD<f64> =
            ndarray::Array1::from((1..=6).map(|i| i as f64).collect::<Vec<_>>()).into_dyn();

        // 一阶外推（addGhostExtrapolate）不是线性延拓：斜率取
        // sign(边缘值)*|斜率|，即向远离零等值面方向推。数据递增且边缘为正
        // 时，两侧鬼单元都取更大的正值（与 MATLAB 逐位一致）。
        let g = grid_1d(6, BoundaryCondition::Extrapolate);
        let out = pad_ghost_dim(&g, &data, 0, 2);
        let v = out.iter().cloned().collect::<Vec<f64>>();
        assert_eq!(v, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        // 二阶外推（addGhostExtrapolate2）不带符号调整，线性数据下是精确
        // 的线性延拓。
        let g2 = g.with_boundary_all(BoundaryCondition::Extrapolate2);
        let out2 = pad_ghost_dim(&g2, &data, 0, 2);
        let v2 = out2.iter().cloned().collect::<Vec<f64>>();
        assert_eq!(v2, vec![-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn 二维只填充目标维() {
        let g = Grid::new(&[0.0, 0.0], &[1.0, 1.0], &[3, 2]).with_boundaries(vec![
            BoundaryCondition::Dirichlet(0.0),
            BoundaryCondition::Periodic,
        ]);
        let data = ArrayD::from_shape_vec(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        // 第 0 维加一层鬼单元：形状 (4, 2)，上下两行取 0。
        let out = pad_ghost_dim(&g, &data, 0, 1);
        assert_eq!(out.shape(), &[5, 2]);
        let v = out.iter().cloned().collect::<Vec<f64>>();
        assert_eq!(v, vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0]);
    }
}
