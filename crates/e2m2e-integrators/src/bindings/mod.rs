//! e2m2e-integrators 的 PyO3 绑定子模块（自 lib.rs 按主题拆分，`#[pymodule]` 注册仍集中在 lib.rs）。

#[cfg(feature = "spice")]
pub mod compiled;
pub mod force_models;
pub mod geometry;
pub mod lambert;
pub mod manifold;
pub mod pal;
pub mod qlaw;
#[cfg(feature = "spice")]
pub mod spice;
pub mod steppers;
pub mod three_body;
pub mod transfer_search;
