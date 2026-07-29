//! e2m2e-forces: Force models (N-body, gravity field, STM).
//!
//! 从 e2m2e-integrators 拆分，包含力模型和 STM 变分方程。

#[cfg(feature = "spice")]
pub mod forces;
// gravity_field 的内部依赖；integrators 使用自己的本地副本，不经过本 crate。
#[cfg(feature = "spice")]
pub(crate) mod solid_tide;
#[cfg(feature = "spice")]
pub(crate) mod spherical_harmonic;
