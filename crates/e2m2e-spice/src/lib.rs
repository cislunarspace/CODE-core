//! e2m2e-spice: SPICE FFI bindings.
//!
//! 从 e2m2e-integrators 拆分，只包含 SPICE 相关功能。
//!
//! 仓库全貌与一条任务链的走读见 README 的仓库怎么读一节。

#[cfg(feature = "spice")]
pub mod ephem_cache;
pub mod native_spk;
#[cfg(feature = "spice")]
pub mod spice_ffi;
#[cfg(feature = "spice")]
pub mod spk_accel;

/// 双登记一个内核文件：先入 native 注册表（[`native_spk`]，纯 Rust DAF/SPK
/// 读取器），再 furnsh 到 CSPICE 内核池。
///
/// 非 DAF 容器（文本内核 tls/tpc/tf，`locidw != "DAF/"`）对 native 侧属预期
/// 跳过（返回 `NotDaf` 不算错误）；CSPICE 侧仍正常 furnsh。任一真实错误上抛，
/// 不静默丢内核。
#[cfg(feature = "spice")]
pub fn furnish_kernel(path: &str) -> Result<(), String> {
    match native_spk::load(std::path::Path::new(path)) {
        Ok(()) => {}
        // 文本内核：native 侧无二进制段可登记，跳过。
        Err(native_spk::DafSpkError::NotDaf) => {}
        Err(e) => return Err(format!("native SPK 登记失败 {path}: {e}")),
    }
    // 双登记非原子：CSPICE 侧失败时把 native 侧恢复到调用前状态，避免两侧分叉
    // （native 有内核而 CSPICE 池无 —— spkezr 可用而 pxform 报无内核）。区分
    // 「本次新增」与「此前已登记」：后者重装回去，不能一卸了之（同一路径重复
    // furnish 时，CSPICE 池里仍是上次成功加载的那份）。
    let p = std::path::Path::new(path);
    let existed = native_spk::is_loaded(p);
    cspice::data::furnish(path).map_err(|e| {
        if existed {
            let _ = native_spk::load(p);
        } else {
            native_spk::unload(p);
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
