//! e2m2e-forces: Force models (N-body, gravity field, STM).
//!
//! 从 e2m2e-integrators 拆分，包含力模型和 STM 变分方程。

#[cfg(feature = "spice")]
pub mod forces;
// 纯数学模块，无 spice 依赖；保持无条件 pub，其 #[cfg(test)] 测试在
// 默认 feature 的 cargo test 下照常执行。integrators 的 tide/harmonic 绑定
// 也直接复用本 crate 的这两个模块。
pub mod atmosphere;
pub mod bcr4bp;
pub mod cr3bp;
pub mod solid_tide;
pub mod spherical_harmonic;
