//! 地月会合系无量纲平面 CR3BP 的 HJB Hamiltonian（issue #497）。
//!
//! 标准无量纲化：长度单位为主星间距，时间单位使会合系角速度 ω = 1。
//! 质量比 μ 为唯一动力学参数；主星（质量 1−μ）固定在 x = −μ，
//! 次星（质量 μ）固定在 x = 1−μ。状态 (x, y, vx, vy)，无控动力学：
//!
//! ```text
//! v̇x = 2·vy + x − (1−μ)(x+μ)/r1³ − μ(x−1+μ)/r2³
//! v̇y = −2·vx + y − (1−μ)·y/r1³ − μ·y/r2³
//! ```
//!
//! r1、r2 为到两主星的距离。公式与 e2m2e-forces 的 `cr3bp_eom`
//! （`CR3BP_Dynamics` 的 Rust 移植）逐项一致，对拍见
//! `tests/cr3bp_parity.rs`。

use e2m2e_levelset::grid::Grid;
use e2m2e_levelset::hamiltonian::Hamiltonian;
use ndarray::ArrayD;

use crate::control_hamiltonian;

/// 到主星距离的下限截断，与 e2m2e-forces cr3bp 模块的 `MIN_DISTANCE` 一致。
const MIN_DISTANCE: f64 = 1e-10;

/// 平面 CR3BP 会合系 Hamiltonian。构造参数构造时固定，求值无内部状态。
pub struct Cr3bpSynodic {
    /// 质量比 μ = m₂/(m₁+m₂)，地月取 0.01215。
    pub mu: f64,
    /// 推力加速度上界 a（常值）。
    pub max_accel: f64,
    /// 燃料权重 w：运行代价 L = w·δ。
    pub fuel_weight: f64,
}

impl Cr3bpSynodic {
    /// 构造并校验参数。μ ∈ (0, 1)，max_accel > 0，fuel_weight ≥ 0。
    pub fn new(mu: f64, max_accel: f64, fuel_weight: f64) -> Self {
        assert!(mu.is_finite() && mu > 0.0 && mu < 1.0, "μ 必须在 (0, 1) 内");
        assert!(
            max_accel.is_finite() && max_accel > 0.0,
            "max_accel 必须为正的有限值"
        );
        assert!(
            fuel_weight.is_finite() && fuel_weight >= 0.0,
            "fuel_weight 必须为非负有限值"
        );
        Self {
            mu,
            max_accel,
            fuel_weight,
        }
    }

    /// 无控向量场 f(x, 0) = (vx, vy, ax, ay)。
    pub fn vector_field(&self, state: [f64; 4]) -> [f64; 4] {
        let [x, y, vx, vy] = state;
        let mu = self.mu;
        let r1 = ((x + mu).powi(2) + y * y).sqrt().max(MIN_DISTANCE);
        let r2 = ((x - 1.0 + mu).powi(2) + y * y).sqrt().max(MIN_DISTANCE);
        let inv_r1_3 = 1.0 / r1.powi(3);
        let inv_r2_3 = 1.0 / r2.powi(3);
        let ax = 2.0 * vy + x - (1.0 - mu) * (x + mu) * inv_r1_3 - mu * (x - 1.0 + mu) * inv_r2_3;
        let ay = -2.0 * vx + y - (1.0 - mu) * y * inv_r1_3 - mu * y * inv_r2_3;
        [vx, vy, ax, ay]
    }

    /// 单点 H\*(x, p)。网格方法与测试共用这一份公式，避免两处漂移。
    pub fn hamiltonian_at(&self, state: [f64; 4], p: [f64; 4]) -> f64 {
        let f = self.vector_field(state);
        let pv_norm = p[2].hypot(p[3]);
        p[0] * f[0]
            + p[1] * f[1]
            + p[2] * f[2]
            + p[3] * f[3]
            + control_hamiltonian(pv_norm, self.max_accel, self.fuel_weight)
    }

    /// 第 `dim` 维的耗散包络（与 p 无关的保守上界）。
    ///
    /// 位置维（0、1）：∂H/∂p_r = v，包络 |v|，精确。
    /// 速度维（2、3）：∂H/∂p_v = g(x) + u\*，‖u\*‖ ≤ a，
    /// 包络 |g_d(x)| + a；不要求 p 的取值盒覆盖全部推力方向，
    /// 是比真实逐盒极大值略保守的上界，耗散安全。
    pub fn partial_bound_at(&self, state: [f64; 4], dim: usize) -> f64 {
        let f = self.vector_field(state);
        match dim {
            0 => state[2].abs(),
            1 => state[3].abs(),
            2 => f[2].abs() + self.max_accel,
            3 => f[3].abs() + self.max_accel,
            _ => panic!("Cr3bpSynodic 只有四维"),
        }
    }
}

impl Hamiltonian for Cr3bpSynodic {
    fn hamiltonian(
        &self,
        _t: f64,
        grid: &Grid,
        _phi: &ArrayD<f64>,
        p: &[ArrayD<f64>],
    ) -> ArrayD<f64> {
        assert_eq!(grid.dim(), 4, "Cr3bpSynodic 是四维模型 (x, y, vx, vy)");
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
        assert_eq!(grid.dim(), 4, "Cr3bpSynodic 是四维模型 (x, y, vx, vy)");
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

    const MU_EARTH_MOON: f64 = 0.01215;

    fn ham() -> Cr3bpSynodic {
        Cr3bpSynodic::new(MU_EARTH_MOON, 0.5, 0.1)
    }

    /// 三角平动点 L4/L5：(0.5−μ, ±√3/2) 处无控加速度为零。
    #[test]
    fn triangular_libration_points_are_equilibria() {
        let h = ham();
        for sign in [1.0, -1.0] {
            let state = [0.5 - MU_EARTH_MOON, sign * 3.0_f64.sqrt() / 2.0, 0.0, 0.0];
            let f = h.vector_field(state);
            assert!(
                f.iter().all(|v| v.abs() < 1e-12),
                "L4/L5 应为平衡点，得到 {f:?}"
            );
        }
    }

    /// 共线平动点 L1/L2/L3：在 y = 0 轴上一分法求 ∂Ω/∂x = 0 的根
    /// （括号区间取 μ = 0.01215 下文献位置的邻域），该点无控加速度为零。
    /// y = 0 时 ay 自动为零，只需验证 ax。
    #[test]
    fn collinear_libration_points_are_equilibria() {
        let h = ham();
        let brackets = [(0.5, 0.95), (1.0, 1.2), (-1.1, -0.9)];
        for (lo, hi) in brackets {
            let f = |x: f64| h.vector_field([x, 0.0, 0.0, 0.0])[2];
            let (mut a, mut b) = (lo, hi);
            assert!(f(a) * f(b) < 0.0, "括号 [{a}, {b}] 内应有变号");
            for _ in 0..80 {
                let mid = 0.5 * (a + b);
                if f(a) * f(mid) <= 0.0 {
                    b = mid;
                } else {
                    a = mid;
                }
            }
            let x = 0.5 * (a + b);
            let field = h.vector_field([x, 0.0, 0.0, 0.0]);
            assert!(
                field.iter().all(|v| v.abs() < 1e-9),
                "共线点应为平衡点：x = {x}，f = {field:?}"
            );
        }
    }

    /// 控制项的开关结构：p_v = 0 时不推力（贡献 0），
    /// a·‖p_v‖ > w 时满推力（贡献 w − a·‖p_v‖）。
    #[test]
    fn control_term_switching() {
        let h = ham();
        let state = [0.8, 0.1, 0.2, -0.3];
        // p_v = 0：min(0, w) = 0（w ≥ 0）。
        let h_off = h.hamiltonian_at(state, [1.0, 1.0, 0.0, 0.0]);
        let f = h.vector_field(state);
        assert!((h_off - (f[0] + f[1])).abs() < 1e-15);
        // 大 p_v：控制贡献 w − a·‖p_v‖ < 0。
        let p = [0.0, 0.0, 3.0, 4.0];
        let h_on = h.hamiltonian_at(state, p);
        let expect = 3.0 * f[2] + 4.0 * f[3] + (h.fuel_weight - h.max_accel * 5.0);
        assert!((h_on - expect).abs() < 1e-12);
    }

    /// partial_bound_at 是 |∂H/∂p_dim| 的上界：随机状态与协态采样下
    /// 中心差分不超过包络（避开开关面邻域，那里导数有折点）。
    #[test]
    fn partial_bound_covers_numerical_derivative() {
        let h = ham();
        let eps = 1e-6;
        // 确定性伪随机采样（避免引入 rand 依赖）。
        for k in 0..50 {
            let s = |i: usize| ((k * 7 + i * 13) % 97) as f64 / 48.0 - 1.0;
            let state = [s(0) + 0.5, s(1), s(2), s(3)];
            let p = [s(4), s(5), s(6) + 2.0, s(7) + 2.0]; // p_v 远离开关面
            for dim in 0..4 {
                let mut p_up = p;
                let mut p_dn = p;
                p_up[dim] += eps;
                p_dn[dim] -= eps;
                let deriv =
                    (h.hamiltonian_at(state, p_up) - h.hamiltonian_at(state, p_dn)) / (2.0 * eps);
                assert!(
                    deriv.abs() <= h.partial_bound_at(state, dim) + 1e-6,
                    "dim {dim}：|∂H/∂p| = {} 超过包络 {}",
                    deriv.abs(),
                    h.partial_bound_at(state, dim)
                );
            }
        }
    }

    /// 网格方法与单点方法逐节点一致。
    #[test]
    fn grid_method_matches_pointwise() {
        let h = ham();
        let grid = Grid::new(
            &[0.5, -1.0, -1.0, -1.0],
            &[1.5, 1.0, 1.0, 1.0],
            &[3, 3, 3, 3],
        );
        let p: Vec<ArrayD<f64>> = (0..4)
            .map(|d| ArrayD::from_shape_fn(grid.shape(), move |idx| 0.1 * (idx[d] as f64) - 0.2))
            .collect();
        let phi = ArrayD::zeros(grid.shape());
        let out = h.hamiltonian(0.0, &grid, &phi, &p);
        for (idx, v) in out.indexed_iter() {
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
            assert!((v - h.hamiltonian_at(state, pvec)).abs() < 1e-15);
        }
    }
}
