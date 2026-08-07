//! e2m2e-spice: SPICE FFI bindings.
//!
//! 从 e2m2e-integrators 拆分，只包含 SPICE 相关功能。

#[cfg(feature = "spice")]
pub mod ephem_cache;
#[cfg(feature = "spice")]
pub mod spice_ffi;
#[cfg(feature = "spice")]
pub mod spk_accel;

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
