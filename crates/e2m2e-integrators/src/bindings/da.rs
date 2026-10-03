//! 微分代数（DACE 截断 Taylor 多项式）原语的 PyO3 绑定（issue #784）。
//!
//! 链路：dace-rs（数值实现）→ e2m2e-da（生命周期与 panic 边界约定）→
//! 本模块（PyO3 绑定）→ Python。原语面刻意最小：上下文管理、多项式
//! 算术与初等函数、系数读写、求值与编译求值、范数/包围、映射求逆；
//! 领域编排留在 algorithm 层，不进 Facade 工具面。

use e2m2e_da::{
    catch_da_panic, cos, exp, log, sin, sqrt, tan, CompiledDa as CompiledDaCore, Da as DaCore,
    DaVector, DaceError, NormType,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// 把 [`DaceError`] 翻译为 Python ``RuntimeError``（运算期错误统一出口）。
fn da_err(e: DaceError) -> PyErr {
    PyRuntimeError::new_err(format!("DA 运算失败: {e}"))
}

/// 执行可能 panic 的 DA 运算并包装为 pyclass。
fn da_binop(op: impl FnOnce() -> DaCore) -> PyResult<Da> {
    catch_da_panic(op).map(Da::from_core).map_err(da_err)
}

/// 二元运算的双操作数：Python 侧 ``Da + Da`` 与 ``Da + float`` 双向可用。
enum DaOperand {
    Num(f64),
    Poly(DaCore),
}

impl<'py> pyo3::FromPyObject<'py> for DaOperand {
    fn extract_bound(ob: &pyo3::Bound<'py, pyo3::PyAny>) -> PyResult<Self> {
        if let Ok(f) = ob.extract::<f64>() {
            Ok(Self::Num(f))
        } else {
            Ok(Self::Poly(ob.extract::<PyRef<'_, Da>>()?.inner.clone()))
        }
    }
}

/// 截断 Taylor 多项式（上游 ``dace_rs::Da``）。
///
/// 禁止裸构造：只能经 ``Da.constant`` / ``Da.variable`` 工厂创建，且必须
/// 先 ``da_init_py`` 初始化上下文。不要跨 ``da_init_py`` 代次混用对象
/// （新旧代次对象混算报 code 1003，见 e2m2e-da crate 文档）。
#[pyclass(name = "Da")]
pub struct Da {
    inner: DaCore,
}

impl Da {
    fn from_core(inner: DaCore) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl Da {
    /// 常数多项式（需已初始化 DA 上下文）。
    #[staticmethod]
    pub fn constant(c: f64) -> PyResult<Da> {
        da_binop(|| DaCore::constant(c))
    }

    /// 第 ``idx`` 个独立变量（1-based；超界返回零多项式并告警）。
    #[staticmethod]
    pub fn variable(idx: u32) -> PyResult<Da> {
        da_binop(|| DaCore::variable(idx))
    }

    fn __add__(&self, other: DaOperand) -> PyResult<Da> {
        da_binop(|| match other {
            DaOperand::Num(f) => self.inner.clone() + f,
            DaOperand::Poly(p) => self.inner.clone() + p,
        })
    }

    fn __radd__(&self, lhs: DaOperand) -> PyResult<Da> {
        da_binop(|| match lhs {
            DaOperand::Num(f) => f + self.inner.clone(),
            DaOperand::Poly(p) => p + self.inner.clone(),
        })
    }

    fn __sub__(&self, other: DaOperand) -> PyResult<Da> {
        da_binop(|| match other {
            DaOperand::Num(f) => self.inner.clone() - f,
            DaOperand::Poly(p) => self.inner.clone() - p,
        })
    }

    fn __rsub__(&self, lhs: DaOperand) -> PyResult<Da> {
        da_binop(|| match lhs {
            DaOperand::Num(f) => f - self.inner.clone(),
            DaOperand::Poly(p) => p - self.inner.clone(),
        })
    }

    fn __mul__(&self, other: DaOperand) -> PyResult<Da> {
        da_binop(|| match other {
            DaOperand::Num(f) => self.inner.clone() * f,
            DaOperand::Poly(p) => self.inner.clone() * p,
        })
    }

    fn __rmul__(&self, lhs: DaOperand) -> PyResult<Da> {
        da_binop(|| match lhs {
            DaOperand::Num(f) => f * self.inner.clone(),
            DaOperand::Poly(p) => p * self.inner.clone(),
        })
    }

    fn __truediv__(&self, other: DaOperand) -> PyResult<Da> {
        da_binop(|| match other {
            DaOperand::Num(f) => self.inner.clone() / f,
            DaOperand::Poly(p) => self.inner.clone() / p,
        })
    }

    fn __rtruediv__(&self, lhs: DaOperand) -> PyResult<Da> {
        da_binop(|| match lhs {
            DaOperand::Num(f) => f / self.inner.clone(),
            DaOperand::Poly(p) => p / self.inner.clone(),
        })
    }

    fn __neg__(&self) -> PyResult<Da> {
        da_binop(|| -self.inner.clone())
    }

    /// 正弦（常数部分经初等函数复合展开到截断阶）。
    pub fn sin(&self) -> PyResult<Da> {
        da_binop(|| sin(&self.inner))
    }

    /// 余弦。
    pub fn cos(&self) -> PyResult<Da> {
        da_binop(|| cos(&self.inner))
    }

    /// 正切（常数部分余弦为零时报 649）。
    pub fn tan(&self) -> PyResult<Da> {
        da_binop(|| tan(&self.inner))
    }

    /// 指数。
    pub fn exp(&self) -> PyResult<Da> {
        da_binop(|| exp(&self.inner))
    }

    /// 自然对数（常数部分非正时报 647）。
    pub fn log(&self) -> PyResult<Da> {
        da_binop(|| log(&self.inner))
    }

    /// 平方根（负常数部分报 645）。
    pub fn sqrt(&self) -> PyResult<Da> {
        da_binop(|| sqrt(&self.inner))
    }

    /// 整数幂。
    pub fn powi(&self, np: i32) -> PyResult<Da> {
        da_binop(|| self.inner.powi(np))
    }

    /// 实数幂（非整数幂的负常数部分报 643）。
    pub fn powf(&self, p: f64) -> PyResult<Da> {
        da_binop(|| self.inner.powf(p))
    }

    /// 常数部分（无该单项式时为 0.0）。
    pub fn cons(&self) -> f64 {
        self.inner.cons()
    }

    /// 线性系数，每变量一个。
    pub fn linear(&self) -> Vec<f64> {
        self.inner.linear()
    }

    /// 指数向量 ``jj`` 对应单项式的系数（缺项返回 0.0）。
    pub fn coefficient(&self, jj: Vec<u32>) -> f64 {
        self.inner.get_coefficient(&jj)
    }

    /// 设置指数向量 ``jj`` 对应单项式的系数（``|c| <= epsilon`` 时删除该项）。
    pub fn set_coefficient(&mut self, jj: Vec<u32>, c: f64) -> PyResult<()> {
        catch_da_panic(|| self.inner.set_coefficient(&jj, c)).map_err(da_err)
    }

    /// 全部存储单项式 ``([指数向量], 系数)`` 列表。
    pub fn monomials(&self) -> Vec<(Vec<u32>, f64)> {
        self.inner.iter_monomials().map(|m| (m.jj, m.c)).collect()
    }

    /// 存储的非零单项式个数。
    pub fn term_count(&self) -> usize {
        self.inner.size()
    }

    /// 在给定点求值（返回标量）。
    pub fn eval(&self, args: Vec<f64>) -> PyResult<f64> {
        catch_da_panic(|| self.inner.eval(&args)).map_err(da_err)
    }

    /// 系数最大绝对值（无穷范数）。
    pub fn norm_inf(&self) -> f64 {
        self.inner.norm(NormType::Infinity)
    }

    /// 系数绝对值之和（1 范数）。
    pub fn norm_one(&self) -> f64 {
        self.inner.norm(NormType::One)
    }

    /// 系数的 p 向量范数（要求 ``p >= 2``；p = 2 为欧氏范数）。
    pub fn norm_power(&self, p: u32) -> PyResult<f64> {
        if p < 2 {
            return Err(PyValueError::new_err("norm_power 需要指数 p >= 2"));
        }
        Ok(self.inner.norm(NormType::Power(p)))
    }

    /// 域 ``[-1,1]^nv`` 上的保守包围 ``(lo, hi)``。
    pub fn bound(&self) -> (f64, f64) {
        let iv = self.inner.bound();
        (iv.lo, iv.hi)
    }

    /// 编译为可复用 Horner 求值树。
    pub fn compile(&self) -> CompiledDa {
        CompiledDa::from_core(self.inner.compile())
    }

    fn __repr__(&self) -> String {
        format!(
            "Da(cons={}, terms={})",
            self.inner.cons(),
            self.inner.size()
        )
    }
}

/// 共享 Horner 求值树（上游 ``dace_rs::CompiledDa``），可对多个分量
/// 一次性重复求值。
#[pyclass(name = "CompiledDa", frozen)]
pub struct CompiledDa {
    /// 分量个数。
    #[pyo3(get)]
    pub dim: u32,
    /// 树的最大阶。
    #[pyo3(get)]
    pub ord: u32,
    /// 使用的变量数。
    #[pyo3(get)]
    pub vars: u32,
    /// 树的节点数（含根）。
    #[pyo3(get)]
    pub terms: u32,
    inner: CompiledDaCore,
}

impl CompiledDa {
    fn from_core(core: CompiledDaCore) -> Self {
        Self {
            dim: core.dim,
            ord: core.ord,
            vars: core.vars,
            terms: core.terms,
            inner: core,
        }
    }
}

#[pymethods]
impl CompiledDa {
    /// 把多个同上下文 ``Da`` 编译为一棵共享求值树（混上下文报 1003）。
    #[staticmethod]
    pub fn from_das(py: Python<'_>, das: Vec<pyo3::Py<Da>>) -> PyResult<Self> {
        let cores: Vec<DaCore> = das.iter().map(|d| d.borrow(py).inner.clone()).collect();
        let core = catch_da_panic(|| CompiledDaCore::from_das(&cores)).map_err(da_err)?;
        Ok(Self::from_core(core))
    }

    /// 在给定点求值，返回每个分量的值。
    pub fn eval(&self, args: Vec<f64>) -> PyResult<Vec<f64>> {
        catch_da_panic(|| self.inner.eval(&args)).map_err(da_err)
    }
}

/// 初始化 DA 上下文（阶数 ``order``、变量数 ``nvars``），可重复调用。
///
/// 低于 1 的参数被上游 clamp 到 1 并告警；编码表超出 32 位索引空间时
/// 返回 ``ValueError``（失败后旧上下文仍有效）。
#[pyfunction]
pub fn da_init_py(order: u32, nvars: u32) -> PyResult<()> {
    e2m2e_da::init(order, nvars).map_err(|e| PyValueError::new_err(format!("DA 初始化失败: {e}")))
}

/// DA 上下文是否已在当前进程初始化。
#[pyfunction]
pub fn da_initialized_py() -> bool {
    e2m2e_da::initialized()
}

/// 当前线程的截断阶（未初始化报 RuntimeError）。
#[pyfunction]
pub fn da_truncation_order_py() -> PyResult<u32> {
    catch_da_panic(e2m2e_da::truncation_order).map_err(da_err)
}

/// 设置当前线程的截断阶（clamp 到 ``[1, 阶数]``），返回旧值。
#[pyfunction]
pub fn da_set_truncation_order_py(order: u32) -> PyResult<u32> {
    catch_da_panic(|| e2m2e_da::set_truncation_order(order)).map_err(da_err)
}

/// 多项式映射求逆：求 ``das`` 的复合逆（要求维数 ≤ 变量数、线性部分非奇异）。
///
/// 内部按递增截断阶迭代，返回前恢复原截断阶。奇异报 642、维数超限报
/// 165、未初始化报 1003，均映射为 ``RuntimeError``。
#[pyfunction]
pub fn da_vector_invert_py(py: Python<'_>, das: Vec<pyo3::Py<Da>>) -> PyResult<Vec<Da>> {
    let cores: Vec<DaCore> = das.iter().map(|d| d.borrow(py).inner.clone()).collect();
    let inverted = catch_da_panic(|| cores.invert()).map_err(da_err)?;
    Ok(inverted.into_iter().map(Da::from_core).collect())
}
