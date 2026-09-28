//! SPICE 内核装卸、诊断查询与星历缓存开关的 PyO3 绑定（整模块 spice 门控）。

use pyo3::prelude::*;
use pyo3::types::PyList;

/// PoC：查询 `target` 相对 `observer` 在 J2000 系下的位置（km）。
///
/// 走 ADR 0051 的纯 Rust SPK 后端（`spice_ffi::spkezr` → `native_spk`），
/// 零 cspice FFI；内核须经 `spice_furnsh`（`furnish_kernel`）双登记。
///
/// 用于验证：maturin 链路是否正常、native 注册表与 Python spiceypy 的
/// 查询是否一致。仅在 `spice` feature 下编译。返回长度 3 的 `Vec<f64>` 。
#[cfg(feature = "spice")]
#[pyfunction]
pub fn spice_poc_body_position(et: f64, target: &str, observer: &str) -> PyResult<Vec<f64>> {
    let (state, _lt) = e2m2e_spice::spice_ffi::spkezr(target, et, "J2000", "NONE", observer)
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("native spkezr failed: {e}"))
        })?;
    Ok(state[..3].to_vec())
}

/// 首次调用时经 Once 触发 Rust CSPICE 实例的行星名别名注册（对称 Python
/// 侧 SPICEManager.load_kernel 的 boddef）。幂等。
#[cfg(feature = "spice")]
pub(crate) fn ensure_bodies_registered() {
    static REGISTERED: std::sync::Once = std::sync::Once::new();
    REGISTERED.call_once(e2m2e_spice::spice_ffi::register_bodies);
}

/// 在 Rust 侧加载一个内核文件（ADR 0051 双登记：native SPK 注册表 +
/// cspice 内核池）。
///
/// Rust cspice 与 Python spiceypy 是**独立的 CSPICE 实例** （静态链接，全局状态
/// 不共享）。Python 侧 furnsh 的内核，Rust 看不见；反之亦然。要让 Rust 查询
/// 可用，必须用本函数在 Rust 侧再 furnsh 一次（同一份文件，两边独立加载）。
/// native SPK 注册表与 CSPICE 池由 [`e2m2e_spice::furnish_kernel`] 一次性
/// 双登记：二进制 DAF 内核进 native 注册表（`spkezr` 求值路径），文本内核
/// 只进 CSPICE 池；任一真实错误上抛。
///
/// 同时在首次加载时把行星名注册到质心/本体 ID（`register_bodies` ），使本
/// 实例对 "MARS"/"JUPITER" 等的解析与 Python spiceypy 实例（那边在
/// manager.load_kernel 里 boddef）以及 DFH 一致，否则 CSPICE 默认表会把
/// "MARS" 解析成不存在的本体 499。
#[cfg(feature = "spice")]
#[pyfunction]
pub fn spice_furnsh(path: &str) -> PyResult<()> {
    ensure_bodies_registered();
    e2m2e_spice::furnish_kernel(path)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("furnsh failed: {e}")))?;
    LOADED_KERNELS.lock().unwrap().push(path.to_string());
    Ok(())
}

/// 已通过 [`spice_furnsh`] 加载到 Rust cspice 内核池的内核路径清单。
///
/// CSPICE `unload_c` 对未加载文件会设置错误信号，而 Python 侧
/// `SPICEManager.unload_kernel` 的语义是幂等的（重复卸载不抛，见
/// tests/data/kernels/test_spice_manager.py::test_load_and_unload）。清单保证
/// [`spice_unload`] 只对确实 furnsh 过的文件调 `unload_c` ，未加载文件静默跳过。
#[cfg(feature = "spice")]
static LOADED_KERNELS: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// 从 Rust cspice 内核池卸载一个内核文件（与 [`spice_furnsh`] 对称）。
///
/// Rust cspice 与 Python spiceypy 独立（见 [`spice_furnsh`] 文档）。
/// `SPICEManager.load_kernel` 双 furnsh，卸载必须对称：否则 Rust 内核池残留
/// 已卸载文件，测试结果依赖同进程执行顺序。只卸载清单中
/// 确已加载的文件，其余静默跳过（保持幂等语义）。
#[cfg(feature = "spice")]
#[pyfunction]
pub fn spice_unload(path: &str) -> PyResult<()> {
    let mut loaded = LOADED_KERNELS.lock().unwrap();
    if !loaded.iter().any(|p| p == path) {
        return Ok(());
    }
    // 三池卸载（ADR 0051/0052）：native SPK 注册表 + 文本池（FK/TPCK/LSK）
    // 幂等移除 + cspice 池卸载。
    let p = std::path::Path::new(path);
    e2m2e_spice::native_spk::unload(p);
    e2m2e_spice::native_frame::unload(p);
    e2m2e_spice::native_time::unload(p);
    cspice_rs::data::unload(path).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("unload failed: {:?}", e))
    })?;
    loaded.retain(|p| p != path);
    Ok(())
}

/// 诊断用：在 Rust CSPICE 实例上查 spkezr（与 spiceypy.spkezr 同名函数对齐）。
///
/// 用于对比 Python（spiceypy）与 Rust（cspice-sys）两个独立 CSPICE 实例的
/// 查询结果，排查内核加载 / boddef 同步问题。常规查询仍走
/// ``SPICEManager`` / spiceypy。返回 ``(state[6], lt)`` 。
#[cfg(feature = "spice")]
#[pyfunction]
pub fn spice_spkezr(
    target: &str,
    et: f64,
    frame: &str,
    abcorr: &str,
    observer: &str,
) -> PyResult<(Vec<f64>, f64)> {
    ensure_bodies_registered();
    let (state, lt) = e2m2e_spice::spice_ffi::spkezr(target, et, frame, abcorr, observer)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    Ok((state.to_vec(), lt))
}

/// 诊断用：在 Rust CSPICE 实例上查 pxform（与 spiceypy.pxform 同名函数对齐）。
///
/// 用于对比 Python（spiceypy）与 Rust（cspice-sys）两个独立 CSPICE 实例的
/// 帧旋转查询，排查内核加载同步问题。常规查询仍走 ``SPICEManager`` /
/// spiceypy。返回 3×3 行优先矩阵。
#[cfg(feature = "spice")]
#[pyfunction]
pub fn spice_pxform(from: &str, to: &str, et: f64) -> PyResult<Vec<Vec<f64>>> {
    ensure_bodies_registered();
    let m = e2m2e_spice::spice_ffi::pxform(from, to, et)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    Ok(m.iter().map(|row| row.to_vec()).collect())
}

/// 激活 Rust 星历预采样缓存。
///
/// 在积分前把要用到的天体状态与帧旋转矩阵在均匀网格上预采样、建三次样条，
/// 装入进程级缓存。此后 Rust 力模型（ThirdBody/IndirectTerm/GravityField/Relativistic）
/// 每步查表，不再调 cspice FFI。需在 SPICE 内核已加载后调用。
///
/// **参数**
///
/// - ``targets``: 要缓存的天体对 ``[(target, observer), ...]`` ，如
///   ``[("MOON", "EARTH"), ("SUN", "EARTH"), ("EARTH", "SOLAR SYSTEM BARYCENTER")]``
/// - ``frame_pairs``: 要缓存的帧旋转对 ``[(from, to), ...]`` ，如
///   ``[("ITRF93", "J2000"), ("MOON_PA", "J2000")]``
/// - ``sxform_pairs``: 要缓存的 6×6 状态变换对 ``[(from, to), ...]`` ，如
///   ``[("ITRF93", "J2000")]`` （Lense-Thirring 用）。关键字参数，默认 ``None`` 。
/// - ``et_start``, ``et_end``: 积分时间范围（SPICE et 秒）
/// - ``dt``: 网格步长（秒），默认 3600
#[cfg(feature = "spice")]
#[pyfunction]
#[pyo3(signature = (targets, frame_pairs, et_start, et_end, dt=3600.0, *, sxform_pairs=None))]
pub fn enable_ephem_cache(
    targets: &Bound<'_, PyList>,
    frame_pairs: &Bound<'_, PyList>,
    et_start: f64,
    et_end: f64,
    dt: f64,
    sxform_pairs: Option<&Bound<'_, PyList>>,
) -> PyResult<()> {
    let mut bodies: Vec<(String, String)> = Vec::new();
    for item in targets.iter() {
        let tup: (String, String) = item.extract()?;
        bodies.push(tup);
    }
    let mut frames: Vec<(String, String)> = Vec::new();
    for item in frame_pairs.iter() {
        let tup: (String, String) = item.extract()?;
        frames.push(tup);
    }
    let mut sxforms: Vec<(String, String)> = Vec::new();
    if let Some(pairs) = sxform_pairs {
        for item in pairs.iter() {
            let tup: (String, String) = item.extract()?;
            sxforms.push(tup);
        }
    }
    let cache = e2m2e_spice::ephem_cache::EphemCache::build(
        &bodies, &frames, &sxforms, et_start, et_end, dt,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("ephem cache build: {e:?}")))?;
    e2m2e_spice::ephem_cache::enable(cache);
    Ok(())
}

/// 关闭 Rust 星历缓存（回到逐次 cspice 查询）。
#[cfg(feature = "spice")]
#[pyfunction]
pub fn disable_ephem_cache() {
    e2m2e_spice::ephem_cache::disable();
}

/// 返回 cspice FFI 调用计数（验证零 cspice 场景用）。
#[cfg(feature = "spice")]
#[pyfunction]
pub fn ephem_ffi_call_count() -> u64 {
    e2m2e_spice::spice_ffi::ffi_call_count()
}

/// 清零 cspice FFI 调用计数。
#[cfg(feature = "spice")]
#[pyfunction]
pub fn reset_ephem_ffi_call_count() {
    e2m2e_spice::spice_ffi::reset_ffi_call_count();
}
