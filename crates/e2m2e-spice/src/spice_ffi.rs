//! cspice-sys 直接 FFI 的 safe 包装（仅 `spice` feature 下编译）。
//!
//! cspice 0.1 高层 API 只覆盖 spk/data/time 等基础功能，缺 pxform/sxform/bodvrd
//! 等坐标变换和物理参数查询。本模块直接通过 cspice-sys 的 unsafe FFI 调用
//! 这些函数，提供 safe Rust 接口。
//!
//! # CSPICE 错误处理
//!
//! CSPICE C 库的错误模型是"set failure flag + 长跳"——出错时设置 `failed_c()`
//! 返回 true，后续调用都短路返回。`reset_c()` 清除错误状态。
//!
//! 本模块的包装在每次调用后检查 `failed_c()`，如果出错就 `reset_c()` 并返回
//! `Err`，避免错误状态泄漏到下次调用。
//!
//! # 线程安全
//!
//! CSPICE 全局状态非线程安全。cspice crate 用 `with_spice_lock_or_panic`
//! 串行化，但本模块直接走 FFI 不加锁。**调用方必须保证单线程使用**（e2m2e
//! 主循环是单线程 Python，OK）。

#[cfg(test)]
use cspice_sys::bodn2c_c;
use cspice_sys::{
    boddef_c, bodvrd_c, erract_c, errdev_c, et2utc_c, failed_c, getmsg_c, ktotal_c, pxform_c,
    qcktrc_c, reset_c, spkezr_c, sxform_c, ConstSpiceChar, SpiceInt,
};
use std::ffi::CString;
use std::os::raw::c_char;
use std::sync::atomic::{AtomicU64, Ordering};

/// cspice FFI 调用计数。验证"零 cspice"用：打靶前后读该计数，应为 0
/// （前提：星历预采样缓存已启用 + strict 模式，力模型查内存样条）。
pub static FFI_CALLS: AtomicU64 = AtomicU64::new(0);

/// 返回累计 cspice FFI 调用次数（pxform/sxform/spkezr 入口）。
pub fn ffi_call_count() -> u64 {
    FFI_CALLS.load(Ordering::Relaxed)
}

/// 清零 cspice FFI 调用计数。
pub fn reset_ffi_call_count() {
    FFI_CALLS.store(0, Ordering::Relaxed);
}

fn bump_ffi_calls() {
    FFI_CALLS.fetch_add(1, Ordering::Relaxed);
}

/// 取 CSPICE 指定类别消息（调用前后状态不限）。`option` 为 "SHORT"/"LONG"/
/// "EXPLAIN" 等（见 getmsg_c 文档）。缓冲在首个 null 处截断。
fn getmsg(option: &str, len: usize) -> String {
    let opt_c = CString::new(option).expect("CSPICE option contains null byte");
    let mut msg_buf = vec![0i8; len];
    unsafe {
        getmsg_c(
            opt_c.as_ptr() as *mut ConstSpiceChar,
            len as SpiceInt,
            msg_buf.as_mut_ptr() as *mut c_char,
        );
    }
    c_chars_to_string(&msg_buf)
}

/// 取 CSPICE traceback（qcktrc_c 包装）。缓冲在首个 null 处截断。
fn qcktrc(len: usize) -> String {
    let mut trace_buf = vec![0i8; len];
    unsafe {
        qcktrc_c(len as SpiceInt, trace_buf.as_mut_ptr() as *mut c_char);
    }
    c_chars_to_string(&trace_buf)
}

/// 取 CSPICE 完整错误信息：SHORT + LONG + traceback（调用前必须 failed_c()==true）。
///
/// SHORT 仅 256 字节、常被截断；LONG 是完整描述，traceback 指向出错调用栈。
/// 三者拼接，空段跳过，让上层异常携带可定位的错误信息。
fn get_full_error_message() -> String {
    let short = getmsg("SHORT", 256);
    let long = getmsg("LONG", 2048);
    let traceback = qcktrc(2048);
    let mut parts: Vec<String> = Vec::new();
    if !short.is_empty() {
        parts.push(short);
    }
    if !long.is_empty() {
        parts.push(long);
    }
    if !traceback.is_empty() {
        parts.push(format!("Traceback:\n{traceback}"));
    }
    parts.join("\n\n")
}

/// 把 CSPICE 返回的 C char 数组转成 Rust String（在首个 null 处截断）。
fn c_chars_to_string(buf: &[i8]) -> String {
    let bytes: Vec<u8> = buf
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).to_string()
}

/// CSPICE FFI 调用错误。
#[derive(Debug)]
pub enum SpiceFfiError {
    /// `failed_c()` 在调用后返回 true；含完整错误描述（SHORT + LONG + traceback）。
    Failed(String),
}

impl std::fmt::Display for SpiceFfiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpiceFfiError::Failed(msg) => write!(f, "CSPICE failure: {}", msg),
        }
    }
}

impl std::error::Error for SpiceFfiError {}

/// 检查 CSPICE 错误状态，如果出错则 reset 并返回错误。
///
/// 调用方应在每次 CSPICE FFI 后调用本函数。
fn check_spice_error() -> Result<(), SpiceFfiError> {
    unsafe {
        let failed: bool = failed_c() != 0;
        if failed {
            let msg = get_full_error_message();
            reset_c();
            return Err(SpiceFfiError::Failed(msg));
        }
    }
    Ok(())
}

/// 把 Rust 字符串转成 CSPICE 期望的 C null-terminated 字符串（CString）。
fn to_cstring(s: &str) -> CString {
    CString::new(s).expect("CSPICE name contains null byte")
}

/// 行星名→NAIF ID 别名表，与 Python 侧
/// `e2m2e/data/kernels/manager.py:_BODY_ID_ALIASES` 保持一致。
///
/// 背景：de440s/de430/de440 全本只含行星**质心**段（水~海王 ID 1~8）+
/// 地球族本体（199/299/399）+ 月球（301）+ 太阳（10），**不含**行星本体段
/// （499/599/…）。但 CSPICE 内置默认表把 "MARS" 解析成 499、把 "MARS
/// BARYCENTER" 解析成 4。DFH（qiao 版 README:308-321）与天体力学惯例对
/// 大行星第三体摄动一律用**质心**（含卫星总质量）；e2m2e 所有非 STM 路径
/// 与 GM 值也已统一用质心。本表把这些名字注册到质心/本体 ID，使本（Rust）
/// CSPICE 实例的解析与 Python spiceypy 实例（那边在 manager.load_kernel 里做同样
/// 的 boddef）以及与 DFH 一致。
const BODY_ALIASES: &[(&str, SpiceInt)] = &[
    ("MERCURY", 1),
    ("VENUS", 2),
    ("EARTH", 399),
    ("MARS", 4),
    ("JUPITER", 5),
    ("SATURN", 6),
    ("URANUS", 7),
    ("NEPTUNE", 8),
    ("MOON", 301),
    ("SUN", 10),
];

/// 显式设置 CSPICE 错误动作与输出设备，消除对上游 cspice crate 初始化
/// 顺序的依赖。
///
/// CSPICE 默认错误动作是 ABORT（出错即 `exit(1)` 杀进程）。cspice 0.1 crate
/// 的 `set_error_defaults`（动作 RETURN、设备 NULL）只在首次 `with_spice_lock`
/// 时触发；本模块的 FFI 包装不经 `with_spice_lock`，若该锁还没被任何路径
/// 触发过，CSPICE 就以默认 ABORT 运行——spkezr/pxform 出错会直接终止
/// Python 进程而非抛异常。本函数把动作显式设为 RETURN（出错置 `failed_c()`
/// 后返回）、设备设为 NULL（错误信息经 `getmsg_c` 取回，不污染 stderr），
/// 保证本实例首次被使用前已进入"出错返回"模式。幂等，重复调用无害。
fn init_error_handling() {
    let set_c = CString::new("SET").unwrap();
    let return_c = CString::new("RETURN").unwrap();
    let null_c = CString::new("NULL").unwrap();
    unsafe {
        erract_c(
            set_c.as_ptr() as *mut ConstSpiceChar,
            0,
            return_c.as_ptr() as *mut ConstSpiceChar,
        );
        errdev_c(
            set_c.as_ptr() as *mut ConstSpiceChar,
            0,
            null_c.as_ptr() as *mut ConstSpiceChar,
        );
    }
}

/// NAIF ID → 名字反查（基于 [`BODY_ALIASES`]）。供星历缓存 key 归一化等场景
/// （to_rust_spec 把天体名转成 ID 字符串，缓存 enable 侧用名字）。未注册的
/// ID 返回 `None`。
pub fn id_to_name(id: SpiceInt) -> Option<&'static str> {
    BODY_ALIASES
        .iter()
        .find(|(_, code)| *code == id)
        .map(|(name, _)| *name)
}

/// 在本 CSPICE 实例注册 [`BODY_ALIASES`] 里的行星名别名（等价 Python
/// spiceypy.boddef）。`boddef_c` 只改名字→ID 映射表，不需要内核加载，
/// 对同一 (name, id) 重复调用幂等。
///
/// 同时显式设置 CSPICE 错误动作（见 [`init_error_handling`]），消除对上游
/// crate 初始化顺序的依赖。应在任何 `spkezr`/`bodvrd` 之前调用一次（见
/// `spice_furnsh` 里的 `Once` 触发）。
pub fn register_bodies() {
    init_error_handling();
    for (name, code) in BODY_ALIASES {
        let name_c = to_cstring(name);
        unsafe {
            boddef_c(name_c.as_ptr() as *mut ConstSpiceChar, *code);
        }
    }
}

/// 查名字→NAIF ID（bodn2c_c 包装），返回 Some(id) 或 None。
#[cfg(test)]
fn bodn2c(name: &str) -> Option<SpiceInt> {
    let name_c = to_cstring(name);
    let mut id: SpiceInt = 0;
    let mut found: SpiceInt = 0;
    unsafe {
        bodn2c_c(name_c.as_ptr() as *mut ConstSpiceChar, &mut id, &mut found);
    }
    if found != 0 {
        Some(id)
    } else {
        None
    }
}

/// ktotal_c 包装：返回当前已加载的指定类型内核数。`kind` 通常为 "ALL"。
///
/// 供 spkezr/pxform 入口预检内核池是否为空（见 ADR 0020：用状态查询预检，
/// 不用 CSPICE 错误码字符串匹配翻译）。
pub fn ktotal(kind: &str) -> Result<i32, SpiceFfiError> {
    let kind_c = to_cstring(kind);
    let mut count: SpiceInt = 0;
    unsafe {
        ktotal_c(kind_c.as_ptr() as *mut ConstSpiceChar, &mut count);
        check_spice_error()?;
    }
    Ok(count as i32)
}

/// 内核池为空时的项目语境错误信息。spkezr/pxform 入口预检复用。
const NO_KERNEL_MSG: &str = "Rust CSPICE 实例无内核加载——请经 SPICEManager.load_kernel 加载";

/// pxform_c 包装：返回 from→to 在 et 时刻的 3×3 旋转矩阵（行优先）。
///
/// 等价于 Python spiceypy.pxform(from, to, et)。
pub fn pxform(from: &str, to: &str, et: f64) -> Result<[[f64; 3]; 3], SpiceFfiError> {
    // 入口预检：内核池为空时直接报项目语境错误，不走 FFI（避免 CSPICE 内部
    // 错误码上冒或 erract 兜底杀进程）。
    if ktotal("ALL")? == 0 {
        return Err(SpiceFfiError::Failed(NO_KERNEL_MSG.into()));
    }
    bump_ffi_calls();
    let from_c = to_cstring(from);
    let to_c = to_cstring(to);
    let mut rotate = [[0.0_f64; 3]; 3];
    unsafe {
        pxform_c(
            from_c.as_ptr() as *mut ConstSpiceChar,
            to_c.as_ptr() as *mut ConstSpiceChar,
            et,
            rotate.as_mut_ptr(),
        );
        check_spice_error()?;
    }
    Ok(rotate)
}

/// sxform_c 包装：返回 from→to 在 et 时刻的 6×6 状态变换矩阵。
///
/// 等价于 Python spiceypy.sxform(from, to, et)。返回 6×6 行优先矩阵。
pub fn sxform(from: &str, to: &str, et: f64) -> Result<[[f64; 6]; 6], SpiceFfiError> {
    bump_ffi_calls();
    let from_c = to_cstring(from);
    let to_c = to_cstring(to);
    let mut xform = [[0.0_f64; 6]; 6];
    unsafe {
        sxform_c(
            from_c.as_ptr() as *mut ConstSpiceChar,
            to_c.as_ptr() as *mut ConstSpiceChar,
            et,
            xform.as_mut_ptr(),
        );
        check_spice_error()?;
    }
    Ok(xform)
}

/// spkezr_c 包装：返回 target 相对 observer 在 frame 系下的状态 [x,y,z,vx,vy,vz] + 光时 lt。
///
/// 等价于 Python spiceypy.spkezr(target, et, frame, abcorr, observer)。
/// `abcorr` 通常为 "NONE"。
pub fn spkezr(
    target: &str,
    et: f64,
    frame: &str,
    abcorr: &str,
    observer: &str,
) -> Result<([f64; 6], f64), SpiceFfiError> {
    // 入口预检：内核池为空时直接报项目语境错误，不走 FFI（避免 CSPICE 内部
    // 错误码上冒或 erract 兜底杀进程）。
    if ktotal("ALL")? == 0 {
        return Err(SpiceFfiError::Failed(NO_KERNEL_MSG.into()));
    }
    bump_ffi_calls();
    let target_c = to_cstring(target);
    let frame_c = to_cstring(frame);
    let abcorr_c = to_cstring(abcorr);
    let observer_c = to_cstring(observer);
    let mut state = [0.0_f64; 6];
    let mut lt = 0.0_f64;
    unsafe {
        spkezr_c(
            target_c.as_ptr() as *mut ConstSpiceChar,
            et,
            frame_c.as_ptr() as *mut ConstSpiceChar,
            abcorr_c.as_ptr() as *mut ConstSpiceChar,
            observer_c.as_ptr() as *mut ConstSpiceChar,
            state.as_mut_ptr(),
            &mut lt,
        );
        check_spice_error()?;
    }
    Ok((state, lt))
}

/// bodvrd_c 包装：读取天体属性（如 GM、RADII）。
///
/// 等价于 Python spiceypy.bodvrd(body, item, maxn)。
/// 返回 values 数组（长度 maxn）+ 实际 dim。
pub fn bodvrd(body: &str, item: &str, maxn: usize) -> Result<(Vec<f64>, i32), SpiceFfiError> {
    let body_c = to_cstring(body);
    let item_c = to_cstring(item);
    let mut values = vec![0.0_f64; maxn];
    let mut dim: SpiceInt = 0;
    unsafe {
        bodvrd_c(
            body_c.as_ptr() as *mut ConstSpiceChar,
            item_c.as_ptr() as *mut ConstSpiceChar,
            maxn as SpiceInt,
            &mut dim,
            values.as_mut_ptr(),
        );
        check_spice_error()?;
    }
    values.truncate(dim as usize);
    Ok((values, dim))
}

/// et2utc_c 包装：ET → UTC ISO 字符串（"ISOC" 格式，prec 位小数秒）。
///
/// 等价于 Python spiceypy.et2utc(et, "ISOC", prec)。供批量 ET→UTC 转换
/// （星历表组装）下沉 Rust 用。
pub fn et2utc(et: f64, prec: i32) -> Result<String, SpiceFfiError> {
    // 入口预检同 spkezr：内核池为空（leapsecond 缺失）时直接报项目语境错误。
    if ktotal("ALL")? == 0 {
        return Err(SpiceFfiError::Failed(NO_KERNEL_MSG.into()));
    }
    bump_ffi_calls();
    let fmt_c = to_cstring("ISOC");
    let mut buf = vec![0i8; 64];
    unsafe {
        et2utc_c(
            et,
            fmt_c.as_ptr() as *mut ConstSpiceChar,
            prec as SpiceInt,
            buf.len() as SpiceInt,
            buf.as_mut_ptr() as *mut c_char,
        );
        check_spice_error()?;
    }
    Ok(c_chars_to_string(&buf))
}

/// 矩阵向量乘：3×3 矩阵 × 3 向量。
pub fn mat3_mul_vec(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// 矩阵转置 × 向量：3×3 转置矩阵 × 3 向量（用于反向变换）。
pub fn mat3_t_mul_vec(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load_kernels() {
        // crates/e2m2e-integrators → crates → e2m2e 根（kernels/ 在 e2m2e 根下）
        let kernel_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent() // crates/
            .and_then(|p| p.parent()) // e2m2e 根
            .unwrap()
            .join("kernels");
        for name in [
            "naif0012.tls",
            "pck00010.tpc",
            "de430.bsp",
            "earth_latest_high_prec.bpc",
            "SPICEEarthPredictedKernel.bpc",
            "SPICELunaFrameKernel.tf",
            "SPICELunaCurrentKernel.bpc",
        ] {
            let path = kernel_dir.join(name);
            if path.exists() {
                let _ = cspice::data::furnish(path.to_string_lossy().to_string());
            }
        }
    }

    /// pxform 在 et=0 的 ITRF93→J2000 应该是有限旋转矩阵。
    #[test]
    fn pxform_itrf93_to_j2000() {
        let _g = crate::lock_spice_for_test();
        load_kernels();
        let r = pxform("ITRF93", "J2000", 0.0).expect("pxform failed");
        // 旋转矩阵的每行模 1
        for row in r.iter() {
            let norm = (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt();
            assert!((norm - 1.0).abs() < 1e-10, "row norm {} != 1", norm);
        }
    }

    /// spkezr 应该与 cspice 0.1 高层 easier_reader 给出相同结果。
    #[test]
    fn spkezr_matches_cspice_high_level() {
        let _g = crate::lock_spice_for_test();
        load_kernels();
        use cspice::common::AberrationCorrection;
        use cspice::spk::easier_reader;
        use cspice::time::Et;
        let et = 0.0_f64;
        let (state, _lt) = easier_reader(
            "MOON",
            Et::from(et),
            "J2000",
            AberrationCorrection::NONE,
            "EARTH",
        )
        .expect("easier_reader failed");
        let (state2, _lt2) = spkezr("MOON", et, "J2000", "NONE", "EARTH").expect("spkezr failed");
        let expected_pos = [
            state.position.x as f64,
            state.position.y as f64,
            state.position.z as f64,
        ];
        for k in 0..3 {
            let diff = (expected_pos[k] - state2[k]).abs();
            assert!(diff < 1e-9, "pos[{}] diff={}", k, diff);
        }
    }

    /// 注册行星别名后，CSPICE 应把 "MARS"/"JUPITER" 解析成质心 ID（4/5），
    /// 且 spkezr(名字) 与 spkezr(ID) 结果一致——否则默认表会把 "MARS"
    /// 解析成 de440s 不含的本体 499 导致 SPKINSUFFDATA。
    #[test]
    fn register_bodies_maps_planets_to_barycenter() {
        let _g = crate::lock_spice_for_test();
        load_kernels();
        register_bodies();
        assert_eq!(bodn2c("MARS"), Some(4));
        assert_eq!(bodn2c("JUPITER"), Some(5));
        assert_eq!(bodn2c("SATURN"), Some(6));
        assert_eq!(bodn2c("EARTH"), Some(399));
        assert_eq!(bodn2c("MOON"), Some(301));

        let et = 0.0_f64;
        for name in ["MARS", "JUPITER"] {
            let (by_name, _) = spkezr(name, et, "J2000", "NONE", "EARTH")
                .unwrap_or_else(|e| panic!("spkezr({}) failed: {:?}", name, e));
            let id = bodn2c(name).unwrap();
            let (by_id, _) = spkezr(&id.to_string(), et, "J2000", "NONE", "EARTH")
                .unwrap_or_else(|e| panic!("spkezr({}) failed: {:?}", id, e));
            for k in 0..6 {
                let diff = (by_name[k] - by_id[k]).abs();
                assert!(diff < 1e-9, "{} state[{}] diff={}", name, k, diff);
            }
        }
    }
}
