use pyo3::prelude::*;

#[pyclass(eq, eq_int, rename_all = "SCREAMING_SNAKE_CASE")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RkMethod {
    Pd45,
}

impl RkMethod {
    pub fn stages(self) -> usize {
        match self {
            RkMethod::Pd45 => 7,
        }
    }
}
