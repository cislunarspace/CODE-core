//! cspice-sys FFI 的 safe 包装（仅 `spice` feature 下编译）。
//!
//! 自 ADR 0051/#685 Phase A 起，星历几何查询（spkezr）走纯 Rust 后端
//! （`native_spk`），不跨 FFI；自 ADR 0056/#685 Phase B 起，
//! `pxform`/`sxform`（BPC Type 2 帧旋转 + FK 帧图 + 文本 PCK）与
//! `et2utc`（LSK 时间）也改走纯 Rust 后端（`native_frame`/
//! `native_time`），`ktotal` 改为 native 登记计数。生产入口零 FFI；
//! 原 FFI 包装保留在 [`ffi_oracle`]（`#[doc(hidden)]`），仅供对拍测试
//! 当 oracle（先例：`daf::parse_with`）。
//!
//! # CSPICE 错误处理（oracle 与残余 FFI 面）
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
    boddef_c, erract_c, errdev_c, failed_c, getmsg_c, qcktrc_c, reset_c, ConstSpiceChar, SpiceInt,
};
use std::ffi::CString;
use std::os::raw::c_char;
use std::sync::atomic::{AtomicU64, Ordering};

/// cspice FFI 调用计数。验证"零 cspice"用：打靶前后读该计数，应为 0
/// （前提：星历预采样缓存已启用 + strict 模式，力模型查内存样条）。
///
/// 自 ADR 0056 起 pxform/sxform/et2utc/ktotal 生产入口不再跨 FFI，
/// 该计数在生产路径恒为 0；oracle（[`ffi_oracle`]）不入计数——
/// 计数语义只剩「生产面旁路诊断」。
pub static FFI_CALLS: AtomicU64 = AtomicU64::new(0);

/// 返回累计 cspice FFI 调用次数。
///
/// ADR 0051 后 spkezr、ADR 0056 后 pxform/sxform/et2utc/ktotal 均走纯
/// Rust 后端、不再计数，生产路径恒 0（「零 cspice」断言方向不受影响）。
pub fn ffi_call_count() -> u64 {
    FFI_CALLS.load(Ordering::Relaxed)
}

/// 清零 cspice FFI 调用计数。
pub fn reset_ffi_call_count() {
    FFI_CALLS.store(0, Ordering::Relaxed);
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
/// crate 初始化顺序的依赖。应在任何 `spkezr`/`pxform` 之前调用一次（见
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

/// 名字→NAIF ID（本地解析，不跨 FFI）。供 [`spkezr`] 走纯 Rust 后端前把
/// 天体名归一成 ID。与 CSPICE `bodn2c` 的可用面保持一致：
///
/// 1. [`BODY_ALIASES`] 别名表（大小写不敏感）；
/// 2. CSPICE 内置天体名中本项目实际会用到的 "SOLAR SYSTEM BARYCENTER"（ID 0）；
/// 3. 纯数字串直接按 NAIF ID 解析（spkezr("399", …) 语义）。
///
/// 覆盖面窄于 CSPICE `bodn2c` 内置表："PLUTO"、"LUNA"、"MARS BARYCENTER" 一类
/// 未收录名会硬报「未知天体名」（ADR 0051 Phase A 边界，不静默回退）。仓库内
/// 全部 Rust 调用方传名均在覆盖内（见 R/naif_id_str 侧的数字串路径）。
pub(crate) fn name_to_id(name: &str) -> Option<SpiceInt> {
    let upper = name.trim().to_ascii_uppercase();
    if upper == "SOLAR SYSTEM BARYCENTER" {
        return Some(0);
    }
    if let Some((_, id)) = BODY_ALIASES.iter().find(|(n, _)| *n == upper) {
        return Some(*id);
    }
    upper.parse::<SpiceInt>().ok()
}

/// 当前已加载内核的 native 计数（`ktotal_c` 语义的纯 Rust 版）。
///
/// 自 ADR 0056 起不再跨 FFI：`"ALL"` = native 登记的文件总数（DAF + 文本
/// 池 + LSK）；`"SPK"` = 含 ≥1 个 SPK 段（`center: Some`）的 DAF 文件数；
/// `"PCK"` = 含 ≥1 个 BPC 段的 DAF 文件数；`"FK"`/`"LSK"`/`"TEXT"` =
/// 对应文本池计数（TEXT = FK + 文本 PCK + LSK）。其他 kind → `Err`
/// （`NATIVE_KTOTAL_UNSUPPORTED_KIND`）。生产唯一调用点
/// `nbody_stm.rs::spk_kernels_loaded` 用 `"SPK"`，语义保持。
pub fn ktotal(kind: &str) -> Result<i32, SpiceFfiError> {
    let count = match kind.to_ascii_uppercase().as_str() {
        "ALL" => {
            (crate::native_spk::daf_file_count()
                + crate::native_frame::fk_count()
                + crate::native_frame::tpck_count()
                + crate::native_time::count()) as i32
        }
        "SPK" => {
            let (spk, _) = crate::native_spk::daf_file_segment_counts();
            spk as i32
        }
        "PCK" => {
            let (_, pck) = crate::native_spk::daf_file_segment_counts();
            pck as i32
        }
        "FK" => crate::native_frame::fk_count() as i32,
        "LSK" => crate::native_time::count() as i32,
        "TEXT" => {
            (crate::native_frame::fk_count()
                + crate::native_frame::tpck_count()
                + crate::native_time::count()) as i32
        }
        other => {
            return Err(SpiceFfiError::Failed(format!(
                "NATIVE_KTOTAL_UNSUPPORTED_KIND: ktotal 类别 {other:?} 不受支持（仅 ALL/SPK/PCK/FK/LSK/TEXT）"
            )));
        }
    };
    Ok(count)
}

/// 无内核可用时的项目语境错误信息。pxform 判「native 三池全空」、
/// et2utc 判 LSK 池空、spkezr 判 native 注册表空时复用。
const NO_KERNEL_MSG: &str = "Rust CSPICE 实例无内核加载——请经 SPICEManager.load_kernel 加载";

/// `from → to` 在 `et` 时刻的 3×3 旋转矩阵（行主序）。
///
/// 等价于 Python spiceypy.pxform(from, to, et)。自 ADR 0056/#685 Phase B
/// 起走纯 Rust 后端 [`crate::native_frame::pxform`]（BPC Type 2 + FK 帧
/// 图 + 文本 PCK + 内置帧），与 CSPICE 逐位一致；内核须经理
/// [`crate::furnish_kernel`] 登记（native 注册表 / 文本池）。
pub fn pxform(from: &str, to: &str, et: f64) -> Result<[[f64; 3]; 3], SpiceFfiError> {
    // 入口预检：三池全空说明没有任何内核经 furnish_kernel 登记，直接报
    // 项目语境错误（仅装 LSK 也能查 J2000↔ECLIPJ2000 的现行行为保持）。
    if crate::native_spk::is_empty()
        && crate::native_frame::pools_empty()
        && crate::native_time::is_empty()
    {
        return Err(SpiceFfiError::Failed(NO_KERNEL_MSG.into()));
    }
    crate::native_frame::pxform(from, to, et).map_err(|e| SpiceFfiError::Failed(e.to_string()))
}

/// `from → to` 在 `et` 时刻的 6×6 状态变换矩阵（行主序）。
///
/// 等价于 Python spiceypy.sxform(from, to, et)。自 ADR 0056 起走纯 Rust
/// 后端 [`crate::native_frame::sxform`]（组合语义照搬 frmchg.c）。
pub fn sxform(from: &str, to: &str, et: f64) -> Result<[[f64; 6]; 6], SpiceFfiError> {
    if crate::native_spk::is_empty()
        && crate::native_frame::pools_empty()
        && crate::native_time::is_empty()
    {
        return Err(SpiceFfiError::Failed(NO_KERNEL_MSG.into()));
    }
    crate::native_frame::sxform(from, to, et).map_err(|e| SpiceFfiError::Failed(e.to_string()))
}

/// spkezr 包装：返回 target 相对 observer 在 frame 系下的状态 [x,y,z,vx,vy,vz] + 光时 lt。
///
/// 等价于 Python spiceypy.spkezr(target, et, frame, abcorr, observer)。
///
/// # 纯 Rust 后端（ADR 0051，#639 Phase A）
///
/// 星历几何求值不再跨 CSPICE FFI：走 [`crate::native_spk`] 的纯 Rust DAF +
/// SPK Type 2 读取器，与 CSPICE 逐位一致。Phase A 边界：`frame == "J2000"`
/// 且 `abcorr == "NONE"` 才继续，否则硬报错（消息含
/// `SPK_NATIVE_UNSUPPORTED_FRAME/ABCORR`），不回退 CSPICE。abcorr=NONE 下
/// `lt` 恒返回 0.0（几何链式状态不含光行时；`abcorr=NONE` 下 CSPICE 的 lt 为
/// 几何单向光时，本后端不计算它——仓库内无 lt 消费者，Python 诊断口
/// `spice_spkezr` 会观察到该差异）。内核须经 [`crate::furnish_kernel`] 双登记
/// （native 注册表 + CSPICE 池）。
pub fn spkezr(
    target: &str,
    et: f64,
    frame: &str,
    abcorr: &str,
    observer: &str,
) -> Result<([f64; 6], f64), SpiceFfiError> {
    // 入口预检：native 注册表为空说明没有任何内核经 furnish_kernel 登记
    // （文本内核不产生 native 段），直接报项目语境错误。
    if crate::native_spk::is_empty() {
        return Err(SpiceFfiError::Failed(NO_KERNEL_MSG.into()));
    }
    if !frame.eq_ignore_ascii_case("J2000") {
        return Err(SpiceFfiError::Failed(format!(
            "SPK_NATIVE_UNSUPPORTED_FRAME: 本地 SPK 后端仅支持 J2000，收到 frame={frame:?}"
        )));
    }
    if !abcorr.eq_ignore_ascii_case("NONE") {
        return Err(SpiceFfiError::Failed(format!(
            "SPK_NATIVE_UNSUPPORTED_ABCORR: 本地 SPK 后端仅支持 abcorr=NONE，收到 abcorr={abcorr:?}"
        )));
    }
    let tid = name_to_id(target)
        .ok_or_else(|| SpiceFfiError::Failed(format!("未知天体名 {target:?}：无法解析 NAIF ID")))?;
    let oid = name_to_id(observer).ok_or_else(|| {
        SpiceFfiError::Failed(format!("未知天体名 {observer:?}：无法解析 NAIF ID"))
    })?;
    let state =
        crate::native_spk::state(tid, et, oid).map_err(|e| SpiceFfiError::Failed(e.to_string()))?;
    Ok((state, 0.0))
}

/// ET → UTC ISO 字符串（"ISOC" 格式，prec 位小数秒）。
///
/// 等价于 Python spiceypy.et2utc(et, "ISOC", prec)。供批量 ET→UTC 转换
/// （星历表组装）下沉 Rust 用。自 ADR 0056 起走纯 Rust 后端
/// [`crate::native_time::et2utc_isoc`]（deltet/tunitim/ttrans 语义移植）；
/// 只支持 ISOC（仓库两个调用点均为 ISOC），`prec` ∈ 0..=9。
pub fn et2utc(et: f64, prec: i32) -> Result<String, SpiceFfiError> {
    // 入口预检同 spkezr：LSK 池空（leapsecond 缺失）时直接报项目语境错误。
    if crate::native_time::is_empty() {
        return Err(SpiceFfiError::Failed(NO_KERNEL_MSG.into()));
    }
    crate::native_time::et2utc_isoc(et, prec).map_err(|e| SpiceFfiError::Failed(e.to_string()))
}

/// 民用日期 → 1970-01-01 起的天数（Howard Hinnant 算法，含负数年/日）。
/// `pub(crate)`：`@date` 记号（native_frame::text）与 LSK 的
/// daynum→日历（native_time）复用同一口径。
pub(crate) fn days_from_civil(year: i32, month: u32, day: u32) -> i64 {
    let y = if month <= 2 {
        i64::from(year) - 1
    } else {
        i64::from(year)
    };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400; // [0, 399]
    let mp = i64::from((month + 9) % 12); // 3 月 = 0
    let doy = (153 * mp + 2) / 5 + i64::from(day) - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146097 + doe - 719468
}

/// [`days_from_civil`] 的逆变换。`pub(crate)`：native_time 复用。
pub(crate) fn civil_from_days(days: i64) -> (i32, u32, u32) {
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]，3 月 = 0
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };
    (year as i32, month as u32, day as u32)
}

/// UTC 时标原点：2000-01-01T00:00:00Z 的「1970 起的天数」。
const UTC_EPOCH_DAYS: i64 = 10957;

/// 公历闰年判定。
fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// 公历年月日 → 年积日（1 基；月份越界钳到 [1, 12]，结果钳到 [1, 366]）。
pub fn day_of_year(year: i32, month: u32, day: u32) -> u16 {
    const CUM_DAYS: [u32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let m = month.clamp(1, 12) as usize;
    let mut doy = CUM_DAYS[m - 1] + day;
    if is_leap_year(year) && month > 2 {
        doy += 1;
    }
    doy.min(366) as u16
}

/// 解析 `et2utc(.., "ISOC", 3)` 输出 `"YYYY-MM-DDTHH:MM:SS.sss"`。
pub fn parse_isoc(isoc: &str) -> Result<(i32, u32, u32, f64), SpiceFfiError> {
    let bad = || SpiceFfiError::Failed(format!("et2utc 输出格式异常: {isoc:?}"));
    let (date, time) = isoc.split_once('T').ok_or_else(bad)?;
    let date: Vec<&str> = date.split('-').collect();
    let time: Vec<&str> = time.split(':').collect();
    if date.len() != 3 || time.len() != 3 {
        return Err(bad());
    }
    let year: i32 = date[0].parse().map_err(|_| bad())?;
    let month: u32 = date[1].parse().map_err(|_| bad())?;
    let day: u32 = date[2].parse().map_err(|_| bad())?;
    let hour: f64 = time[0].parse().map_err(|_| bad())?;
    let minute: f64 = time[1].parse().map_err(|_| bad())?;
    let second: f64 = time[2].parse().map_err(|_| bad())?;
    Ok((year, month, day, hour * 3600.0 + minute * 60.0 + second))
}

/// ET → UTC 自 2000-01-01T00:00:00Z 起的秒数（连续单调，含闰秒偏移）。
///
/// 选这条**单调连续**曲线而非 (年积日, 日内秒) 作为缓存量与插值对象：后者在
/// 年/日边界回绕、无法插值。毫秒分辨率（`et2utc` prec=3）对本用途足够。
pub fn et_to_utc_seconds(et: f64) -> Result<f64, SpiceFfiError> {
    let isoc = et2utc(et, 3)?;
    let (year, month, day, seconds_of_day) = parse_isoc(&isoc)?;
    Ok((days_from_civil(year, month, day) - UTC_EPOCH_DAYS) as f64 * 86400.0 + seconds_of_day)
}

/// [`et_to_utc_seconds`] 的逆：UTC 秒 → `(年, 年积日, 日内秒)`。
pub fn utc_seconds_to_calendar(utc_seconds: f64) -> (i32, u16, f64) {
    let days = (utc_seconds / 86400.0).floor();
    let seconds_of_day = utc_seconds - days * 86400.0;
    let (year, month, day) = civil_from_days(days as i64 + UTC_EPOCH_DAYS);
    (year, day_of_year(year, month, day), seconds_of_day)
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

/// 原 FFI 包装的对拍 oracle（`#[doc(hidden)]`，仅供测试）。
///
/// 先例：`daf::parse_with`（ADR 0051）。`pxform`/`sxform`/`et2utc`/`ktotal`
/// 的生产入口自 ADR 0056 起走纯 Rust 后端，但 cspice crate 的高层 API 没有
/// 这些函数的包装，对拍测试无处复用——保留直连 `*_c` 的薄包装当 oracle。
/// oracle 不入 [`FFI_CALLS`] 计数（计数语义 = 生产面旁路诊断，oracle 使用
/// 不得破坏「生产路径恒 0」断言）。
///
/// 前置：内核已经理 [`crate::furnish_kernel`] 双登记（oracle 走 cspice 池，
/// 被测走 native 池）；CSPICE erract 须为 RETURN/NULL（见
/// `init_error_handling`，`register_bodies` 触发）。
#[doc(hidden)]
#[cfg(feature = "spice")]
pub mod ffi_oracle {
    use super::{c_chars_to_string, check_spice_error, to_cstring, SpiceFfiError};
    use cspice_sys::{et2utc_c, ktotal_c, pxform_c, sxform_c, ConstSpiceChar, SpiceInt};

    /// oracle：CSPICE `pxform_c` 直连，返回 3×3 行主序旋转矩阵。
    pub fn pxform(from: &str, to: &str, et: f64) -> Result<[[f64; 3]; 3], SpiceFfiError> {
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

    /// oracle：CSPICE `sxform_c` 直连，返回 6×6 行主序状态变换。
    pub fn sxform(from: &str, to: &str, et: f64) -> Result<[[f64; 6]; 6], SpiceFfiError> {
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

    /// oracle：CSPICE `et2utc_c` 直连（"ISOC"，prec 位小数秒）。
    pub fn et2utc(et: f64, prec: i32) -> Result<String, SpiceFfiError> {
        let fmt_c = to_cstring("ISOC");
        let mut buf = vec![0i8; 64];
        unsafe {
            et2utc_c(
                et,
                fmt_c.as_ptr() as *mut ConstSpiceChar,
                prec as SpiceInt,
                buf.len() as SpiceInt,
                buf.as_mut_ptr() as *mut std::os::raw::c_char,
            );
            check_spice_error()?;
        }
        Ok(c_chars_to_string(&buf))
    }

    /// oracle：CSPICE `ktotal_c` 直连。
    pub fn ktotal(kind: &str) -> Result<i32, SpiceFfiError> {
        let kind_c = to_cstring(kind);
        let mut count: SpiceInt = 0;
        unsafe {
            ktotal_c(kind_c.as_ptr() as *mut ConstSpiceChar, &mut count);
            check_spice_error()?;
        }
        Ok(count as i32)
    }
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
                // 双登记：native 注册表（spkezr 求值路径）+ CSPICE 内核池
                //（easier_reader 等 oracle 路径）。文本内核 native 侧自动跳过。
                let _ = crate::furnish_kernel(&path.to_string_lossy());
            }
        }
    }

    /// 民用日期 ↔ 天数往返；原点与闰日、跨世纪边界。
    #[test]
    fn civil_days_round_trip() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(2000, 1, 1), UTC_EPOCH_DAYS);
        for &(y, m, d) in &[
            (1970, 1, 1),
            (1999, 12, 31),
            (2000, 1, 1),
            (2000, 2, 29),
            (2024, 12, 31),
            (2100, 3, 1),
            (1900, 1, 1),
        ] {
            let z = days_from_civil(y, m, d);
            assert_eq!(civil_from_days(z), (y, m, d), "往返 {y}-{m}-{d}");
        }
    }

    /// 年积日：闰年 2 月之后 +1，世纪年按 400 年规则。
    #[test]
    fn day_of_year_handles_leap_years() {
        assert_eq!(day_of_year(2000, 1, 1), 1);
        assert_eq!(day_of_year(2000, 3, 1), 61); // 闰年
        assert_eq!(day_of_year(2001, 3, 1), 60); // 平年
        assert_eq!(day_of_year(2000, 12, 31), 366);
        assert_eq!(day_of_year(2001, 12, 31), 365);
        assert_eq!(day_of_year(2100, 12, 31), 365); // 2100 非闰年
        assert_eq!(day_of_year(2000, 2, 29), 60);
    }

    /// ISOC 解析：正常输入取分量，畸形输入报错。
    #[test]
    fn parse_isoc_parses_and_rejects() {
        let (y, m, d, s) = parse_isoc("2000-01-01T06:00:00.000").expect("解析");
        assert_eq!((y, m, d), (2000, 1, 1));
        assert_eq!(s, 21600.0);
        let (y, m, d, s) = parse_isoc("2024-12-31T23:59:59.500").expect("解析");
        assert_eq!((y, m, d), (2024, 12, 31));
        assert!((s - 86399.5).abs() < 1e-9);
        assert!(parse_isoc("2000-01-01").is_err());
        assert!(parse_isoc("garbage").is_err());
    }

    /// UTC 秒 → 日历量的锚点：原点、跨年。
    #[test]
    fn utc_seconds_to_calendar_anchors() {
        assert_eq!(utc_seconds_to_calendar(0.0), (2000, 1, 0.0));
        assert_eq!(utc_seconds_to_calendar(21600.0), (2000, 1, 21600.0));
        // 2000 是闰年（366 天）：第 366 天之后进入 2001-01-01。
        let (y, doy, s) = utc_seconds_to_calendar(366.5 * 86400.0);
        assert_eq!((y, doy), (2001, 1));
        assert!((s - 43200.0).abs() < 1e-6, "s={s}");
    }

    /// ET → UTC 秒与 `et2utc` 自洽（需内核）。
    #[test]
    fn et_to_utc_seconds_matches_known_epoch() {
        let _g = crate::lock_spice_for_test();
        load_kernels();
        // 2000-01-01T06:00:00Z 的 ET（Python SPICEManager.utc_to_et 实测值）。
        let et = -21535.816079952438;
        assert!(et2utc(et, 3).unwrap().starts_with("2000-01-01T06:00:00"));
        let s = et_to_utc_seconds(et).expect("et2utc");
        assert!((s - 21600.0).abs() < 1e-3, "s={s}");
        let (year, doy, secs) = utc_seconds_to_calendar(s);
        assert_eq!((year, doy), (2000, 1));
        assert!((secs - 21600.0).abs() < 1e-3, "secs={secs}");
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
        let expected_pos = [state.position.x, state.position.y, state.position.z];
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
