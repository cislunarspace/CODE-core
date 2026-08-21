//! e2m2e-hjb-dynamics：HJB 结构网格求解器的动力学 Hamiltonian 实现。
//!
//! 求解器内核在 e2m2e-levelset（ToolboxLS 移植，ACM 非商业许可），本 crate
//! 是原创动力学层（Apache-2.0），经 [`e2m2e_levelset::hamiltonian::Hamiltonian`]
//! trait 接入求解器。归属、许可边界与演化谱系见
//! `docs/architecture/hjb-subsystem.md` 与 ADR 0032。
//!
//! 控制模型统一为有界推力 bang-bang：控制 u = δ·a·û，油门 δ ∈ [0,1]，
//! a 为常值加速度上界（`max_accel`），û 方向自由；运行代价 L = w·δ
//! （w 为燃料权重 `fuel_weight`）。最优控制解析给出：û\* = −p_v/‖p_v‖，
//! δ\* = 1 当且仅当 a·‖p_v‖ > w。消去控制后的 Hamiltonian（最优控制意义下
//! H\* = min_u \[L + p·f\]）：
//!
//! ```text
//! H*(x, p) = p_r·v + p_v·g(x) + min(0, w − a·‖p_v‖)
//! ```
//!
//! 其中 g(x) 是各实现的无控加速度场。
//!
//! 时间方向的约定：本 crate 实现的是 H\* 本身。终端代价 HJB 问题
//! （−V_t = H\*，V(tf) = ψ）须从 tf 向 t0 反向求解，时间反转
//! （τ = tf − t，即把 −H\* 交给求解器正向推进）由调用侧负责，
//! 见 e2m2e-integrators 的 HJB 绑定层。

mod cr3bp;
mod double_integrator;

pub use cr3bp::Cr3bpSynodic;
pub use double_integrator::PlanarDoubleIntegrator;

/// bang-bang 控制对 Hamiltonian 的贡献：`min(0, w − a·‖p_v‖)`。
///
/// 对油门 δ 取 min：δ·(w − a·‖p_v‖) 在 δ ∈ [0,1] 上的极小，
/// 开关面 a·‖p_v‖ = w 右侧满推力。
pub(crate) fn control_hamiltonian(pv_norm: f64, max_accel: f64, fuel_weight: f64) -> f64 {
    (fuel_weight - max_accel * pv_norm).min(0.0)
}
