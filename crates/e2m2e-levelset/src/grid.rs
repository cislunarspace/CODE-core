//! 结构网格：对应 ToolboxLS 的 `processGrid.m`。
//!
//! 节点为单元中心（cell-centered）：第 `dim` 维第 `i` 个节点（0-based）位于
//! `min[dim] + (i + 0.5) * dx[dim]`，该维共 `n[dim]` 个节点，`dx[dim] =
//! (max[dim] - min[dim]) / n[dim]`。与 MATLAB 版一致，维度上限 5。

use ndarray::ArrayD;

/// 每维的边界条件。
///
/// 对应 MATLAB `grid.bdry{dim}` 的函数句柄与 `grid.bdryData{dim}`，函数句柄
/// 物化为枚举，鬼单元的填充算法见 [`crate::boundary`]：
///
/// - `Dirichlet(v)`   ← `addGhostDirichlet`（`bdryData.value = v`，仅支持标量）
/// - `Neumann(v)`     ← `addGhostNeumann`（`bdryData.value = v`，仅支持标量）
/// - `Periodic`       ← `addGhostPeriodic`
/// - `Extrapolate`    ← `addGhostExtrapolate`（一阶线性外推）
/// - `Extrapolate2`   ← `addGhostExtrapolate2`（二阶外推）
#[derive(Clone, Debug, PartialEq)]
pub enum BoundaryCondition {
    Dirichlet(f64),
    Neumann(f64),
    Periodic,
    Extrapolate,
    Extrapolate2,
}

/// 结构网格。所有网格函数（`ArrayD<f64>`）的形状等于 `n`。
#[derive(Clone, Debug)]
pub struct Grid {
    dim: usize,
    min: Vec<f64>,
    max: Vec<f64>,
    n: Vec<usize>,
    dx: Vec<f64>,
    /// 每维节点坐标（单元中心），`vs[i]` 长度为 `n[i]`。
    vs: Vec<Vec<f64>>,
    boundary: Vec<BoundaryCondition>,
}

impl Grid {
    /// 由下界、上界与每维节点数构造网格，边界条件默认全周期
    /// （与 `processGrid.m` 的 `defaultBdry = @addGhostPeriodic` 一致）。
    ///
    /// 输入非法时 panic 并给出原因（对应 `processGrid.m` 的 `error` 调用）。
    pub fn new(min: &[f64], max: &[f64], n: &[usize]) -> Grid {
        let dim = min.len();
        assert!(dim == max.len() && dim == n.len(), "min/max/n 长度必须一致");
        assert!(
            (1..=5).contains(&dim),
            "维度必须在 1..=5 之间（processGrid maxDimension）"
        );
        for i in 0..dim {
            assert!(n[i] >= 1, "第 {i} 维节点数必须为正");
            assert!(max[i] > min[i], "第 {i} 维必须 max > min");
        }

        let dx: Vec<f64> = (0..dim).map(|i| (max[i] - min[i]) / n[i] as f64).collect();
        let vs: Vec<Vec<f64>> = (0..dim)
            .map(|i| {
                (0..n[i])
                    .map(|j| min[i] + (j as f64 + 0.5) * dx[i])
                    .collect()
            })
            .collect();

        Grid {
            dim,
            min: min.to_vec(),
            max: max.to_vec(),
            n: n.to_vec(),
            dx,
            vs,
            boundary: vec![BoundaryCondition::Periodic; dim],
        }
    }

    /// 所有维使用同一边界条件。
    pub fn with_boundary_all(mut self, bc: BoundaryCondition) -> Self {
        self.boundary = vec![bc; self.dim];
        self
    }

    /// 逐维指定边界条件，长度必须等于维度。
    pub fn with_boundaries(mut self, bc: Vec<BoundaryCondition>) -> Self {
        assert_eq!(bc.len(), self.dim, "边界条件数量必须等于网格维度");
        self.boundary = bc;
        self
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn min(&self) -> &[f64] {
        &self.min
    }

    pub fn max(&self) -> &[f64] {
        &self.max
    }

    /// 每维节点数。
    pub fn n(&self) -> &[usize] {
        &self.n
    }

    /// 每维网格间距。
    pub fn dx(&self) -> &[f64] {
        &self.dx
    }

    /// 第 `dim` 维节点坐标（单元中心）。
    pub fn axis(&self, dim: usize) -> &[f64] {
        &self.vs[dim]
    }

    /// 第 `dim` 维边界条件。
    pub fn boundary(&self, dim: usize) -> &BoundaryCondition {
        &self.boundary[dim]
    }

    /// 网格函数的 ndarray 形状。
    pub fn shape(&self) -> Vec<usize> {
        self.n.clone()
    }

    /// 网格节点总数。
    pub fn size(&self) -> usize {
        self.n.iter().product()
    }

    /// 校验网格函数形状与网格一致，不一致时 panic。
    pub fn check_data(&self, data: &ArrayD<f64>) {
        assert_eq!(
            data.shape(),
            &self.n[..],
            "网格函数形状 {:?} 与网格 {:?} 不一致",
            data.shape(),
            &self.n[..]
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 单元中心坐标与间距() {
        let g = Grid::new(&[0.0], &[1.0], &[101]);
        assert!((g.dx()[0] - 1.0 / 101.0).abs() < 1e-15);
        assert!((g.axis(0)[0] - 0.5 / 101.0).abs() < 1e-15);
        assert!((g.axis(0)[100] - 100.5 / 101.0).abs() < 1e-15);
    }

    #[test]
    fn 标量广播到各维() {
        let g = Grid::new(&[-1.0, -1.0], &[1.0, 1.0], &[41, 41]);
        assert_eq!(g.dim(), 2);
        assert_eq!(g.shape(), vec![41, 41]);
        assert_eq!(g.size(), 41 * 41);
        assert_eq!(g.boundary(0), &BoundaryCondition::Periodic);
        let g = g.with_boundary_all(BoundaryCondition::Dirichlet(0.0));
        assert_eq!(g.boundary(1), &BoundaryCondition::Dirichlet(0.0));
    }

    #[test]
    #[should_panic]
    fn 维度超限报错() {
        let ones = vec![1.0; 6];
        let ns = vec![11; 6];
        Grid::new(&ones, &ones, &ns);
    }
}
