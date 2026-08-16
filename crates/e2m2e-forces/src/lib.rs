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
pub mod pal_continuation;
pub mod solid_tide;
pub mod spherical_harmonic;
pub mod transfer_geometry;
pub mod transfer_grid_search;

/// 传播失败错误类型。
///
/// 区分"步长塌缩到机器精度地板"与"其他传播错误"，让 Rust→Python FFI 边界
/// 按错误**类型**而非消息字符串决定抛哪种 Python 异常（ADR 0020 决策 2）：
/// [`PropagateError::StepCollapsed`] 在边界表现为
/// ``e2m2e.exceptions.PropagationFailure``，[`PropagateError::Other`] 表现为
/// ``RuntimeError``。改写错误消息措辞不影响 Python 侧的 ``except`` 捕获。
#[derive(Debug, Clone)]
pub enum PropagateError {
    /// 步长塌缩到机器精度地板：自适应控制器不断缩步仍无法满足误差容差，
    /// h 跌破 ``MIN_STEP`` 循环守卫。属确定性传播失败（ADR 0020 决策 2 第 2 级）。
    StepCollapsed(String),
    /// 其他传播错误：参数校验失败、RK 单步内部错误、输出点数不足等。
    Other(String),
}

impl std::fmt::Display for PropagateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PropagateError::StepCollapsed(msg) | PropagateError::Other(msg) => {
                write!(f, "{msg}")
            }
        }
    }
}

impl std::error::Error for PropagateError {}
