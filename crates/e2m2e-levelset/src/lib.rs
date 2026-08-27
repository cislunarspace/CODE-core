//! e2m2e-levelset: 水平集方法与 Hamilton-Jacobi 可达性求解器。
//!
//! 自 UBC Ian M. Mitchell 的 Toolbox of Level Set Methods（ToolboxLS 1.1）
//! 移植，求解含时 Hamilton-Jacobi PDE
//!
//! ```text
//! D_t φ = -H(x, t, φ, ∇φ)
//! ```
//!
//! 支撑低推力轨迹的可达集、最短到达时间与追逃博弈计算。原始代码采用
//! ACM 非商业许可，全文随附于本 crate 的 `LICENSE-ToolboxLS`。
//!
//! 定位与 e2m2e-propagation / e2m2e-forces 一致：纯数学 crate，不依赖
//! SPICE、不含 Python 绑定；对外暴露经 e2m2e-integrators 按 ABI 戳流程
//! 统一进行（见根仓库 `abi-version.txt` 机制）。
//!
//! 与 MATLAB 原版的范围取舍：
//! - 网格函数统一为 `ndarray::ArrayD<f64>`，形状即各维节点数，不再像
//!   MATLAB 版在数组与列向量之间反复 reshape；
//! - 向量水平集（Examples/Vector，多分量 cell 数组）不移植；
//! - 绘图与动画（Helper/Visualization 及 Examples 各 animate*）不移植，
//!   可视化在 Python 侧完成；
//! - MATLAB 的 1-based 维编号在本文一律为 0-based。
//!
//! 仓库全貌与一条任务链的走读见 README 的仓库怎么读一节。

pub mod boundary;
pub mod derivative;
pub mod dissipation;
pub mod grid;
pub mod hamiltonian;
pub mod integrator;
pub mod shape;
pub mod signed_distance;
pub mod term;
