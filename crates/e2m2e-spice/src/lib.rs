//! e2m2e-spice: SPICE FFI bindings.
//!
//! 从 e2m2e-integrators 拆分，只包含 SPICE 相关功能。

#[cfg(feature = "spice")]
pub mod spice_ffi;
#[cfg(feature = "spice")]
pub mod spk_accel;
