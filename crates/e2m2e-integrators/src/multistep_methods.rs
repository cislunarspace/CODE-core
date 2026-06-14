use crate::abm::{ABM_EMBEDDED_ORDER, ABM_STEPS};
use pyo3::prelude::*;

/// Explicit multistep methods exposed to Python. Separate from `RkMethod`
/// because multistep methods carry history and cannot share the stateless
/// `rk_step` entry point.
#[pyclass(eq, eq_int, rename_all = "SCREAMING_SNAKE_CASE")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MultistepMethod {
    Abm,
}

impl MultistepMethod {
    /// Number of history samples the method consumes per step.
    pub fn steps(self) -> usize {
        match self {
            MultistepMethod::Abm => ABM_STEPS,
        }
    }

    /// Order used for step-size suggestion (matches the Milne estimate order).
    pub fn embedded_order(self) -> usize {
        match self {
            MultistepMethod::Abm => ABM_EMBEDDED_ORDER,
        }
    }
}
