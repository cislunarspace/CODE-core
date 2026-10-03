//! 微分代数（DACE 截断 Taylor 多项式）原语的 PyO3 绑定（issue #784）。
//!
//! 链路：dace-rs（数值实现）→ e2m2e-da（生命周期与 panic 边界约定）→
//! 本模块（PyO3 绑定）→ Python。原语面刻意最小：上下文管理、多项式
//! 算术与初等函数、系数读写、求值与编译求值、范数/包围、映射求逆；
//! 领域编排留在 algorithm 层，不进 Facade 工具面。
//!
//! 例外是 ``da_ads_split``：上游 0.2.0 的 ADS（自动域分裂）驱动器薄封装
//! （issue #787）。调用方给变量盒、映射回调与分裂配置，递归分裂全部在
//! Rust 内完成；分裂判据与驱动本体住上游 ``dace_rs::ads``，本模块只做
//! 参数校验、GIL 边界与回调异常透传，不重实现判据。

use e2m2e_da::ads::{
    split as ads_split, AdsConfig as AdsConfigCore, AdsLeaf as AdsLeafCore,
    AdsResult as AdsResultCore, ToleranceKind,
};
use e2m2e_da::{
    catch_da_panic, cos, exp, log, sin, sqrt, tan, CompiledDa as CompiledDaCore, Da as DaCore,
    DaVector, DaceError, Interval, NormType,
};
use parking_lot::Mutex;
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
#[derive(Clone)]
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

/// ADS 分裂的一片叶子：子盒几何、子盒上的重展开多项式、逐分量包围与达标标记。
///
/// 由 [`da_ads_split`] 返回，禁止裸构造。``values`` 是映射在该子盒单位变量
/// 上的重展开多项式（``x_i = c_i + h_i * variable(i+1)``）；``bounds`` 逐
/// 输出分量给出子盒上的保守包围 ``(lo, hi)``，口径同 ``Da.bound``。
#[derive(Clone)]
#[pyclass(name = "AdsLeaf", frozen)]
pub struct AdsLeaf {
    /// 子盒中心，每变量一项。
    #[pyo3(get)]
    pub center: Vec<f64>,
    /// 子盒半宽，每变量一项。
    #[pyo3(get)]
    pub half_width: Vec<f64>,
    /// 映射在该子盒上的重展开多项式。
    #[pyo3(get)]
    pub values: Vec<Da>,
    /// 逐输出分量的保守包围 ``(lo, hi)``。
    #[pyo3(get)]
    pub bounds: Vec<(f64, f64)>,
    /// 目标分量是否全部满足容差（预算或方向上限触顶的叶子为 False，显式可见）。
    #[pyo3(get)]
    pub met: bool,
}

impl AdsLeaf {
    fn from_core(core: AdsLeafCore) -> Self {
        Self {
            center: core.center,
            half_width: core.half_width,
            values: core.values.into_iter().map(Da::from_core).collect(),
            bounds: core.bounds.into_iter().map(|iv| (iv.lo, iv.hi)).collect(),
            met: core.met,
        }
    }
}

/// 一次 ADS 分裂的结果：确定性叶序的叶子列表、每变量二分次数与达标叶数。
#[pyclass(name = "AdsResult", frozen)]
pub struct AdsResult {
    /// 叶子列表，下半盒优先的深度优先确定性顺序。
    #[pyo3(get)]
    pub leaves: Vec<AdsLeaf>,
    /// 每个变量被二分的次数。
    #[pyo3(get)]
    pub splits_per_var: Vec<u32>,
    /// 目标分量全部满足容差的叶子数。
    #[pyo3(get)]
    pub met_leaves: usize,
}

impl AdsResult {
    fn from_core(core: AdsResultCore) -> Self {
        Self {
            leaves: core.leaves.into_iter().map(AdsLeaf::from_core).collect(),
            splits_per_var: core.splits_per_var,
            met_leaves: core.met_leaves,
        }
    }
}

/// 在 GIL 内调用 Python 映射回调并取回多项式向量。
///
/// 回调异常（非 callable、返回类型错等）原样上抛，由调用方暂存、待驱动器
/// 返回后重抛；异常类别不在此翻译。
fn call_map(py: Python<'_>, cb: &Py<PyAny>, inputs: &[DaCore]) -> PyResult<Vec<DaCore>> {
    let py_inputs: Vec<Py<Da>> = inputs
        .iter()
        .map(|d| Py::new(py, Da::from_core(d.clone())))
        .collect::<PyResult<Vec<Py<Da>>>>()?;
    let out = cb.call(py, (py_inputs,), None)?;
    let das: Vec<Py<Da>> = out.extract(py)?;
    Ok(das.iter().map(|d| d.borrow(py).inner.clone()).collect())
}

/// 自动域分裂（ADS）：把 ``func`` 在 ``domain`` 上递归二分并逐子盒重展开，
/// 直到目标分量的多项式包围宽度满足容差（dace-rs 0.2.0 的 ``ads::split``
/// 驱动器薄封装，issue #787）。
///
/// # 参数
///
/// - ``func``：映射回调。收到已代换到当前子盒的 ``Da`` 列表
///   （``x_i = c_i + h_i * variable(i+1)``），必须由这些输入构造输出并返回
///   ``Da`` 列表；返回空列表报 ``RuntimeError``（code 650）。
/// - ``domain``：每变量一个 ``(lo, hi)`` 区间，须非空且 ``lo <= hi``。
/// - ``tolerances``：逐目标分量的容差，正有限，必填。
/// - ``tolerance_kind``：``absolute``（缺省）判据 ``hi - lo <= tol``；
///   ``relative`` 判据 ``hi - lo <= tol * max(|lo|, |hi|)``。
/// - ``targets``：容差作用的输出分量下标；``None`` 或空列表为全部分量，
///   非空给定时长度须与 ``tolerances`` 一致（空列表的逐分量对齐由上游
///   按实际输出数校验）。
/// - ``max_splits_per_var``：每方向二分上限（缺省 32），``0`` 禁止分裂。
/// - ``max_leaves``：总叶子预算（缺省 1024），含未达标叶子。
///
/// 缺省值与上游 ``AdsConfig::default()`` 一致（1e-8 绝对容差只是上游缺省的
/// 起点，本封装仍要求显式给 ``tolerances``）。
///
/// # 回调契约与并发
///
/// 回调收到的 ``Da`` 是子盒单位变量上的多项式，不是原始域变量；输出须由
/// 这些输入构造，驱动器据其包围宽度决定继续分裂或收叶。驱动期间释放 GIL
/// （``allow_threads``），每次子盒重展开回调时重新获取；回调抛出的 Python
/// 异常原样透传（暂存并在分裂结束后重抛，类别不吞），出错时驱动器内部以
/// 输入克隆作哑结果走完递归，最终结果随后丢弃。
///
/// # 返回保证
///
/// 叶子顺序确定性：下半盒优先的深度优先遍历，方向平局取最小变量下标，
/// 相同输入给相同结果。预算或方向上限触顶的叶子 ``met=False`` 显式可见，
/// 不谎报达标。上游运算期错误（code 650 等）经 panic 边界映射为
/// ``RuntimeError``，与 ``da_vector_invert_py`` 同口径。
// Python 可见参数 7 个（py token 由 pyo3 注入），沿用仓库既有 allow。
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (func, domain, tolerances, tolerance_kind = "absolute", targets = None, max_splits_per_var = 32, max_leaves = 1024))]
pub fn da_ads_split(
    py: Python<'_>,
    func: Py<PyAny>,
    domain: Vec<(f64, f64)>,
    tolerances: Vec<f64>,
    tolerance_kind: &str,
    targets: Option<Vec<usize>>,
    max_splits_per_var: u32,
    max_leaves: usize,
) -> PyResult<AdsResult> {
    if domain.is_empty() {
        return Err(PyValueError::new_err(
            "da_ads_split: domain 至少需要一个 (lo, hi) 区间",
        ));
    }
    for (i, &(lo, hi)) in domain.iter().enumerate() {
        if lo.is_nan() || hi.is_nan() || lo > hi {
            return Err(PyValueError::new_err(format!(
                "da_ads_split: 第 {i} 个区间无效（lo > hi 或 NaN 端点），收到 ({lo}, {hi})"
            )));
        }
    }
    if tolerances.is_empty() {
        return Err(PyValueError::new_err(
            "da_ads_split: tolerances 至少需要一个正的有限容差",
        ));
    }
    for (i, &tol) in tolerances.iter().enumerate() {
        if !tol.is_finite() || tol <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "da_ads_split: 第 {i} 个容差无效，收到 {tol}，须为正的有限值"
            )));
        }
    }
    let kind = match tolerance_kind {
        "absolute" => ToleranceKind::Absolute,
        "relative" => ToleranceKind::Relative,
        other => {
            return Err(PyValueError::new_err(format!(
                "da_ads_split: tolerance_kind 只接受 absolute 或 relative，收到 {other}"
            )))
        }
    };
    if let Some(t) = &targets {
        if !t.is_empty() && t.len() != tolerances.len() {
            return Err(PyValueError::new_err(format!(
                "da_ads_split: targets 长度 {} 与 tolerances 长度 {} 不一致",
                t.len(),
                tolerances.len()
            )));
        }
    }
    if max_leaves == 0 {
        return Err(PyValueError::new_err("da_ads_split: max_leaves 必须为正"));
    }

    let config = AdsConfigCore {
        tolerances,
        tolerance_kind: kind,
        targets: targets.unwrap_or_default(),
        max_splits_per_var,
        max_leaves,
    };
    let domain_iv: Vec<Interval> = domain.iter().map(|&(lo, hi)| Interval { lo, hi }).collect();
    // 回调异常暂存：出错时给驱动器返回输入克隆作哑结果（分量数与输入一致，
    // 不会 panic），驱动器照常走完递归；结果随后丢弃、异常原样重抛。
    let err: Mutex<Option<PyErr>> = Mutex::new(None);
    let cb = func.clone_ref(py);
    let core = catch_da_panic(|| {
        py.allow_threads(|| {
            ads_split(
                |inputs: &[DaCore]| {
                    Python::with_gil(|py| match call_map(py, &cb, inputs) {
                        Ok(cores) => cores,
                        Err(e) => {
                            *err.lock() = Some(e);
                            inputs.to_vec()
                        }
                    })
                },
                &domain_iv,
                &config,
            )
        })
    })
    .map_err(da_err)?;
    if let Some(e) = err.into_inner() {
        return Err(e);
    }
    Ok(AdsResult::from_core(core))
}
