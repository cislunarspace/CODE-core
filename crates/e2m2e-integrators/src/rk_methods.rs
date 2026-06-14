use crate::butcher::ButcherTable;
use crate::pd45::PD45_TABLE;
use pyo3::prelude::*;

#[pyclass(eq, eq_int, rename_all = "SCREAMING_SNAKE_CASE")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RkMethod {
    Pd45,
}

impl RkMethod {
    /// The Butcher tableau for this method.
    pub fn table(self) -> &'static ButcherTable {
        match self {
            RkMethod::Pd45 => &PD45_TABLE,
        }
    }

    pub fn stages(self) -> usize {
        self.table().stages
    }

    /// Order of the primary (higher-order) solution.
    pub fn order(self) -> usize {
        self.table().order
    }

    /// Order of the embedded (lower-order) solution used for error control.
    pub fn embedded_order(self) -> usize {
        self.table().embedded_order
    }
}
