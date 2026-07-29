use crate::butcher::ButcherTable;
use crate::pd45::PD45_TABLE;
use crate::pd78::PD78_TABLE;
use crate::rk89::RK89_TABLE;

/// 单步 Runge-Kutta 方法枚举。启用 `pyo3` feature 时暴露给 Python。
///
/// 每种方法对应一张 Butcher 表（见 [`crate::butcher::ButcherTable`]）。
#[cfg_attr(
    feature = "pyo3",
    pyo3::pyclass(eq, eq_int, rename_all = "SCREAMING_SNAKE_CASE")
)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RkMethod {
    Pd45,
    Pd78,
    Rk89,
}

impl RkMethod {
    /// 该方法的 Butcher 表。
    pub fn table(self) -> &'static ButcherTable {
        match self {
            RkMethod::Pd45 => &PD45_TABLE,
            RkMethod::Pd78 => &PD78_TABLE,
            RkMethod::Rk89 => &RK89_TABLE,
        }
    }

    /// 级数（ stages ）。
    pub fn stages(self) -> usize {
        self.table().stages
    }

    /// 主解（高阶解）的阶数。
    pub fn order(self) -> usize {
        self.table().order
    }

    /// 嵌入（低阶）解的阶数，用于误差控制。
    pub fn embedded_order(self) -> usize {
        self.table().embedded_order
    }
}
