//! e2m2e-dyn: 任务设计动力学（转移搜索、流形、MEGNO、Q-law、PAL 延拓）。
//!
//! 从 e2m2e-forces 拆出的非力模块（ADR 0058）：crate 名 forces 不再名不副实，
//! 力模型与 cr3bp/bcr4bp 动力学内核留守 e2m2e-forces，本 crate 以
//! `default-features = false` 依赖 forces 取用内核，自身无 CSPICE 构建链。
//! PyO3 绑定仍在 e2m2e-integrators，Python 符号面与 ABI 不变。

pub mod cartography;
pub mod low_energy_patch;
pub mod manifold;
pub mod megno;
pub mod pal_continuation;
pub mod porkchop;
pub mod qlaw;
pub mod transfer_geometry;
pub mod transfer_grid_search;
pub mod wsb;
