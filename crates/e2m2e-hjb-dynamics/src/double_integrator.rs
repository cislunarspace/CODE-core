//! 带常值漂移的平面双积分器 HJB Hamiltonian。
//!
//! 状态 (x, y, vx, vy)，无控动力学 v̇ = drift（常值漂移加速度），
//! 控制与 crate 级文档的 bang-bang 模型一致。这是 geo-nrho 现有
//! 低维 HJB 管线（`algorithm/dp.py` 的 `LowDimHjbProblem`）所用的动力学，
//! 经 `solve_planar_lowthrust_hjb_py` 兼容包装暴露。

use e2m2e_levelset::grid::Grid;
use e2m2e_levelset::hamiltonian::Hamiltonian;
use ndarray::ArrayD;

use crate::control_hamiltonian;

/// 平面双积分器（常值漂移 + 有界推力）。
pub struct PlanarDoubleIntegrator {
    /// 常值漂移加速度 (drift_x, drift_y)。
    pub drift: [f64; 2],
    /// 推力加速度上界 a（常值）。
    pub max_accel: f64,
    /// 燃料权重 w：运行代价 L = w·δ。
    pub fuel_weight: f64,
}

impl PlanarDoubleIntegrator {
    /// 构造并校验参数。max_accel > 0，fuel_weight ≥ 0。
    pub fn new(drift: [f64; 2], max_accel: f64, fuel_weight: f64) -> Self {
        assert!(drift.iter().all(|v| v.is_finite()), "drift 必须为有限值");
        assert!(
            max_accel.is_finite() && max_accel > 0.0,
            "max_accel 必须为正的有限值"
        );
        assert!(
            fuel_weight.is_finite() && fuel_weight >= 0.0,
            "fuel_weight 必须为非负有限值"
        );
        Self {
            drift,
            max_accel,
            fuel_weight,
        }
    }

    /// 单点 H\*(x, p)。
    pub fn hamiltonian_at(&self, state: [f64; 4], p: [f64; 4]) -> f64 {
        let pv_norm = p[2].hypot(p[3]);
        p[0] * state[2]
            + p[1] * state[3]
            + p[2] * self.drift[0]
            + p[3] * self.drift[1]
            + control_hamiltonian(pv_norm, self.max_accel, self.fuel_weight)
    }

    /// 第 `dim` 维的耗散包络：位置维 |v|，速度维 |drift_d| + a。
    pub fn partial_bound_at(&self, state: [f64; 4], dim: usize) -> f64 {
        match dim {
            0 => state[2].abs(),
            1 => state[3].abs(),
            2 => self.drift[0].abs() + self.max_accel,
            3 => self.drift[1].abs() + self.max_accel,
            _ => panic!("PlanarDoubleIntegrator 只有四维"),
        }
    }
}

impl Hamiltonian for PlanarDoubleIntegrator {
    fn hamiltonian(
        &self,
        _t: f64,
        grid: &Grid,
        _phi: &ArrayD<f64>,
        p: &[ArrayD<f64>],
    ) -> ArrayD<f64> {
        assert_eq!(grid.dim(), 4, "PlanarDoubleIntegrator 是四维模型");
        let mut out = ArrayD::zeros(grid.shape());
        for (idx, o) in out.indexed_iter_mut() {
            let state = [
                grid.axis(0)[idx[0]],
                grid.axis(1)[idx[1]],
                grid.axis(2)[idx[2]],
                grid.axis(3)[idx[3]],
            ];
            let pvec = [
                p[0][idx.clone()],
                p[1][idx.clone()],
                p[2][idx.clone()],
                p[3][idx.clone()],
            ];
            *o = self.hamiltonian_at(state, pvec);
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
        assert_eq!(grid.dim(), 4, "PlanarDoubleIntegrator 是四维模型");
        ArrayD::from_shape_fn(grid.shape(), |idx| {
            let state = [
                grid.axis(0)[idx[0]],
                grid.axis(1)[idx[1]],
                grid.axis(2)[idx[2]],
                grid.axis(3)[idx[3]],
            ];
            self.partial_bound_at(state, dim)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 无漂移、无控制时退化为平流：H = vx·p_x + vy·p_y。
    #[test]
    fn drift_free_control_off_is_advection() {
        let h = PlanarDoubleIntegrator::new([0.0, 0.0], 0.5, 0.1);
        let state = [0.3, -0.7, 0.4, 0.9];
        let p = [1.0, 2.0, 0.0, 0.0];
        assert!((h.hamiltonian_at(state, p) - (0.4 + 1.8)).abs() < 1e-15);
    }

    /// partial_bound_at 覆盖数值导数。
    #[test]
    fn partial_bound_covers_numerical_derivative() {
        let h = PlanarDoubleIntegrator::new([0.2, -0.1], 0.5, 0.1);
        let eps = 1e-6;
        for k in 0..20 {
            let s = |i: usize| ((k * 11 + i * 17) % 89) as f64 / 44.0 - 1.0;
            let state = [s(0), s(1), s(2), s(3)];
            let p = [s(4), s(5), s(6) + 2.0, s(7) + 2.0];
            for dim in 0..4 {
                let mut p_up = p;
                let mut p_dn = p;
                p_up[dim] += eps;
                p_dn[dim] -= eps;
                let deriv =
                    (h.hamiltonian_at(state, p_up) - h.hamiltonian_at(state, p_dn)) / (2.0 * eps);
                assert!(deriv.abs() <= h.partial_bound_at(state, dim) + 1e-9);
            }
        }
    }
}
