use crate::abm::{ABM_EMBEDDED_ORDER, ABM_STEPS};
use pyo3::prelude::*;

/// 暴露给 Python 的显式多步方法。与 `RkMethod` 分开，因为多步方法携带历史缓冲，
/// 不能共享无状态的 `rk_step` 入口。
#[pyclass(eq, eq_int, rename_all = "SCREAMING_SNAKE_CASE")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MultistepMethod {
    Abm,
}

impl MultistepMethod {
    /// 方法每步消耗的历史采样数。
    pub fn steps(self) -> usize {
        match self {
            MultistepMethod::Abm => ABM_STEPS,
        }
    }

    /// 用于步长建议的阶数（与 Milne 估计阶数一致）。
    pub fn embedded_order(self) -> usize {
        match self {
            MultistepMethod::Abm => ABM_EMBEDDED_ORDER,
        }
    }
}
