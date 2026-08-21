//! 隐式形状初值与集合运算：对应 ToolboxLS 的 `InitialConditions/`。
//!
//! 隐式表面约定：函数值在形状内部为负、外部为正，零等值面即边界；
//! 下列基础形状的函数直接取（符号）距离。

use crate::grid::Grid;
use ndarray::ArrayD;

/// 基础形状，对应 `BasicShapes/shapeSphere/Cylinder/Hyperplane/Rectangle.m`
/// 与 `OtherShapes/shapeZalesakDisk.m`。
pub enum Shape {
    /// 超球（任意维）：`{x : ‖x - center‖ ≤ radius}`。
    Sphere { center: Vec<f64>, radius: f64 },
    /// 无限圆柱：沿 `axis` 维无界，其余维距 `center` 不超过 `radius`。
    Cylinder {
        center: Vec<f64>,
        radius: f64,
        axis: usize,
    },
    /// 半空间：`{x : (x - point) · normal ≤ 0}`。
    Hyperplane { point: Vec<f64>, normal: Vec<f64> },
    /// 轴对齐盒子（二维为矩形）。
    Box { lower: Vec<f64>, upper: Vec<f64> },
    /// Zalesak 缺口圆盘（`shapeZalesakDisk.m`），重初始化精度的标准算例。
    /// 缺口沿最后一维朝负方向。
    ZalesakDisk {
        center: Vec<f64>,
        radius: f64,
        /// 缺口宽度。
        width: f64,
        /// 缺口自圆心向外延伸的长度。
        height: f64,
    },
}

impl Shape {
    /// 在网格节点上生成隐式表面函数（内部为负、外部为正）。
    pub fn implicit(&self, grid: &Grid) -> ArrayD<f64> {
        let dim = grid.dim();
        let center_of = |c: &[f64]| -> Vec<f64> { c.to_vec() };
        match self {
            Shape::Sphere { center, radius } => {
                let c = center_of(center);
                assert_eq!(c.len(), dim, "球心维数必须等于网格维度");
                ArrayD::from_shape_fn(grid.shape(), |idx| {
                    let d2: f64 = (0..dim)
                        .map(|i| {
                            let x = grid.axis(i)[idx[i]] - c[i];
                            x * x
                        })
                        .sum();
                    d2.sqrt() - radius
                })
            }
            Shape::Cylinder {
                center,
                radius,
                axis,
            } => {
                let c = center_of(center);
                assert!(*axis < dim, "圆柱轴向越界");
                ArrayD::from_shape_fn(grid.shape(), |idx| {
                    let d2: f64 = (0..dim)
                        .filter(|i| *i != *axis)
                        .map(|i| {
                            let x = grid.axis(i)[idx[i]] - c[i];
                            x * x
                        })
                        .sum();
                    d2.sqrt() - radius
                })
            }
            Shape::Hyperplane { point, normal } => {
                let (p, mut n) = (center_of(point), center_of(normal));
                assert_eq!(n.len(), dim, "法向维数必须等于网格维度");
                let norm = n.iter().map(|v| v * v).sum::<f64>().sqrt();
                for v in &mut n {
                    *v /= norm;
                }
                ArrayD::from_shape_fn(grid.shape(), |idx| {
                    (0..dim).map(|i| n[i] * (grid.axis(i)[idx[i]] - p[i])).sum()
                })
            }
            Shape::Box { lower, upper } => {
                let (lo, hi) = (center_of(lower), center_of(upper));
                assert_eq!(lo.len(), dim);
                ArrayD::from_shape_fn(grid.shape(), |idx| {
                    (0..dim)
                        .map(|i| {
                            let x = grid.axis(i)[idx[i]];
                            (x - hi[i]).max(lo[i] - x)
                        })
                        .fold(f64::NEG_INFINITY, |m, v| m.max(v))
                })
            }
            Shape::ZalesakDisk {
                center,
                radius,
                width,
                height,
            } => {
                // 圆盘减去竖直缺口矩形（shapeZalesakDisk.m 第 50-84 行）：
                // 缺口中心沿最后一维下移 -r + 0.5*slot_length，
                // 宽度 [slot_width, 2r × (dim-2), slot_length]。
                let c = center_of(center);
                let mut slot_center = c.clone();
                slot_center[dim - 1] += -radius + 0.5 * height;
                let mut widths = vec![*width; dim];
                for w in widths.iter_mut().take(dim).skip(1) {
                    *w = 2.0 * radius;
                }
                widths[dim - 1] = *height;
                let lower: Vec<f64> = slot_center
                    .iter()
                    .zip(&widths)
                    .map(|(a, w)| a - w / 2.0)
                    .collect();
                let upper: Vec<f64> = slot_center
                    .iter()
                    .zip(&widths)
                    .map(|(a, w)| a + w / 2.0)
                    .collect();
                let circle = Shape::Sphere {
                    center: c,
                    radius: *radius,
                }
                .implicit(grid);
                let slot = Shape::Box { lower, upper }.implicit(grid);
                shape_difference(&circle, &slot)
            }
        }
    }
}

/// 并集（`shapeUnion.m`）：`min(a, b)`。
pub fn shape_union(a: &ArrayD<f64>, b: &ArrayD<f64>) -> ArrayD<f64> {
    combine(a, b, |x, y| x.min(y))
}

/// 交集（`shapeIntersection.m`）：`max(a, b)`。
pub fn shape_intersection(a: &ArrayD<f64>, b: &ArrayD<f64>) -> ArrayD<f64> {
    combine(a, b, |x, y| x.max(y))
}

/// 差集 A \ B（`shapeDifference.m`）：`max(a, -b)`。
pub fn shape_difference(a: &ArrayD<f64>, b: &ArrayD<f64>) -> ArrayD<f64> {
    combine(a, b, |x, y| x.max(-y))
}

/// 补集（`shapeComplement.m`）：`-a`。
pub fn shape_complement(a: &ArrayD<f64>) -> ArrayD<f64> {
    let data = a.iter().map(|x| -x).collect::<Vec<f64>>();
    ArrayD::from_shape_vec(a.raw_dim(), data).expect("形状不变，不会失败")
}

fn combine(a: &ArrayD<f64>, b: &ArrayD<f64>, f: impl Fn(f64, f64) -> f64) -> ArrayD<f64> {
    assert_eq!(a.shape(), b.shape(), "集合运算要求两隐式函数同形");
    let data = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| f(*x, *y))
        .collect::<Vec<f64>>();
    ArrayD::from_shape_vec(a.raw_dim(), data).expect("形状不变，不会失败")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 集合运算的组合语义() {
        let a = ArrayD::from_shape_vec(vec![2], vec![-1.0, 2.0]).unwrap();
        let b = ArrayD::from_shape_vec(vec![2], vec![3.0, -0.5]).unwrap();
        let to_vec = |x: &ArrayD<f64>| x.iter().cloned().collect::<Vec<f64>>();
        // 隐式函数内负外正：并集取 min、交集取 max、差集取 max(a, -b)
        assert_eq!(to_vec(&shape_union(&a, &b)), vec![-1.0, -0.5]);
        assert_eq!(to_vec(&shape_intersection(&a, &b)), vec![3.0, 2.0]);
        assert_eq!(to_vec(&shape_difference(&a, &b)), vec![-1.0, 2.0]);
        assert_eq!(to_vec(&shape_complement(&a)), vec![1.0, -2.0]);
    }
}
