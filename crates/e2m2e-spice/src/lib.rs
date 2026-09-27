//! e2m2e-spice: SPICE FFI bindings.
//!
//! 从 e2m2e-integrators 拆分，只包含 SPICE 相关功能。
//!
//! 仓库全貌与一条任务链的走读见 README 的仓库怎么读一节。

#[cfg(feature = "spice")]
pub mod ephem_cache;
#[cfg(feature = "spice")]
pub mod native_frame;
pub mod native_spk;
#[cfg(feature = "spice")]
pub mod native_time;
#[cfg(feature = "spice")]
pub mod spice_ffi;
#[cfg(feature = "spice")]
pub mod spk_accel;

/// 文本内核分派到对应池（ADR 0052）。无法识别的文本返回 `Ok(false)`
/// （native 侧跳过，CSPICE 侧照常校验）；解析/装载错误上抛。
#[cfg(feature = "spice")]
fn furnish_text_kernel(path: &std::path::Path, content: &str) -> Result<bool, String> {
    match native_frame::text::classify(path, content) {
        Some(native_frame::text::TextKind::Fk) => {
            native_frame::load_fk(path, content)
                .map_err(|e| format!("native FK 登记 {}: {e}", path.display()))?;
            Ok(true)
        }
        Some(native_frame::text::TextKind::Tpck) => {
            native_frame::load_tpck(path, content)
                .map_err(|e| format!("native 文本 PCK 登记 {}: {e}", path.display()))?;
            Ok(true)
        }
        Some(native_frame::text::TextKind::Lsk) => {
            native_time::load_lsk(path, content)
                .map_err(|e| format!("native LSK 登记 {}: {e}", path.display()))?;
            Ok(true)
        }
        None => Ok(false),
    }
}

/// 双登记一个内核文件：先入 native 注册表（[`native_spk`]，纯 Rust DAF/SPK
/// 读取器）或文本池（[`native_frame`] / [`native_time`]，ADR 0052），再
/// furnsh 到 CSPICE 内核池。
///
/// 分类按内容：前 8 字节 `DAF/` → 二进制（native 注册表）；否则按首个非空
/// 行的 `KPL/FK` / `KPL/PCK` / `KPL/LSK` 前缀分派文本池；无法识别的文本
/// native 侧跳过（CSPICE 侧照常 furnsh 校验）。任一真实错误上抛，不静默
/// 丢内核。
#[cfg(feature = "spice")]
pub fn furnish_kernel(path: &str) -> Result<(), String> {
    let p = std::path::Path::new(path);
    // 内容分类只读 8 字节魔数：`DAF/` → 二进制（native 注册表）；
    // 否则整读文本按 `KPL/*` 首行分派文本池（无法识别 → native 跳过，
    // cspice 照常 furnsh 校验）。任一真实错误上抛，不静默丢内核。
    let mut magic = [0_u8; 8];
    let is_daf = match std::fs::File::open(p) {
        Ok(mut f) => {
            use std::io::Read;
            let n = f.read(&mut magic).unwrap_or(0);
            n >= 4 && &magic[..4] == b"DAF/"
        }
        Err(e) => return Err(format!("内核文件读取失败 {path}: {e}")),
    };
    if is_daf {
        native_spk::load(p).map_err(|e| format!("native SPK 登记 {path}: {e}"))?;
    } else {
        // 非 DAF：整读后按内容分类（lossy 解码容忍非 UTF-8 的他类二进制——
        // 分类失败照旧 native 跳过，语义与 Phase A 的 NotDaf 跳过一致）。
        let bytes = std::fs::read(p).map_err(|e| format!("native 文本内核读取失败 {path}: {e}"))?;
        let content = String::from_utf8_lossy(&bytes).into_owned();
        furnish_text_kernel(p, &content)?;
    }
    // 双登记非原子：CSPICE 侧失败时把 native 侧恢复到调用前状态，避免两侧分叉
    // （native 有内核而 CSPICE 池无 —— spkezr 可用而 pxform 报无内核）。区分
    // 「本次新增」与「此前已登记」：后者重装回去，不能一卸了之（同一路径重复
    // furnish 时，CSPICE 池里仍是上次成功加载的那份）。
    let existed_daf = native_spk::is_loaded(p);
    let existed_text = native_frame::is_loaded(p) || native_time::is_loaded(p);
    cspice::data::furnish(path).map_err(|e| {
        if existed_daf {
            let _ = native_spk::load(p);
        } else {
            native_spk::unload(p);
        }
        // 文本池对称回滚：此前已登记 → 重载，否则卸载。
        if existed_text {
            let bytes = std::fs::read(p).ok();
            if let Some(bytes) = bytes {
                let content = String::from_utf8_lossy(&bytes).into_owned();
                let _ = furnish_text_kernel(p, &content);
            }
        } else {
            native_frame::unload(p);
            native_time::unload(p);
        }
        format!("cspice furnish 失败 {path}: {e:?}")
    })
}

/// cspice 全局状态非线程安全，cspice crate 检测到跨线程并发调用会 panic。
/// 产品积分走 ``ephem_cache`` 内存表、不碰 cspice，不受此锁约束；本锁仅让
/// 调 cspice 的单测（spk_accel / spice_ffi 的 sanity test）在 cargo test
/// 默认多线程下串行执行，避免撞 cspice 全局状态。
#[cfg(all(test, feature = "spice"))]
pub(crate) static SPICE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(all(test, feature = "spice"))]
pub(crate) fn lock_spice_for_test() -> std::sync::MutexGuard<'static, ()> {
    SPICE_TEST_LOCK.lock().unwrap()
}
