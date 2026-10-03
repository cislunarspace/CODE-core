//! 微分代数（DACE 截断 Taylor 多项式）薄封装：上下文生命周期约定与 panic 边界。
//!
//! 上游是 [dace-rs](https://github.com/cislunarspace/dace-rs)（DACE 2.1 的纯 Rust
//! 实现，crates.io 0.1.0），提供截断 Taylor 多项式算术、初等函数、映射求逆、
//! 范数包围与编译求值。本 crate 不做算法，只固定 Python 边界必须依赖的约定，
//! 经 `e2m2e._integrators` 的 `bindings::da` 暴露为最小原语面（issue #784），
//! 作为分区判据区间化、CR3BP 多项式流与自动域分裂等后续工单的公共底座。
//! PyO3 绑定仍在 e2m2e-integrators（同 e2m2e-dyn 的既有模式）。
//!
//! # 上下文生命周期
//!
//! [`init`] 可重复调用：每次调用生成新一代进程级全局上下文（阶数/变量数），
//! 并只重置**调用线程**的 epsilon 与截断阶设置。旧 [`Da`] 对象持有原代次
//! 上下文的快照，`init` 后继续可用；但新旧代次对象混算会以 code 1003 panic。
//! 因此 Python 侧约定：不要跨 `init` 代次混用对象。
//!
//! # 线程本地设置
//!
//! 截断阶（含 [`set_truncation_order`] 与 push/pop 栈）和 epsilon 是线程本地
//! 状态；所有运算都在调用线程执行，跨线程传递 `Da` 对象不传递这些设置。
//!
//! # panic 边界
//!
//! dace-rs 对运算期错误（除零 641、log 非正 647、奇异求逆 642、未初始化 1003、
//! 混上下文 1003、变量超限 165）一律 `panic_any(DaceError)`，只有 [`init`]
//! 返回 `Result`。[`catch_da_panic`] 把这类 panic 转为 `Err(DaceError)`，
//! 绑定层再映射为 Python 异常；非 DaceError 载荷的意外 panic 归一为 code 1006。
//! 默认 panic hook 会在 stderr 留一行噪音，这是不吞 hook 的已知代价。
//!
//! # 包围域约定
//!
//! [`Da::bound`] 给出的上下界针对域 `[-1,1]^nv`（nv 为变量数）：全偶次单项式
//! 把带符号系数贡献到单侧，其余单项式把绝对值贡献到两侧，因此是保守包围
//! 而非精确区间。`invert` 要求向量维数不超过上下文变量数，内部临时下调线程
//! 截断阶迭代求解并在返回前恢复。

pub use dace_rs::elementary::{cos, exp, log, sin, sqrt, tan};
pub use dace_rs::monomial::Monomial;
pub use dace_rs::norm::{Interval, NormType};
pub use dace_rs::vector::DaVector;
pub use dace_rs::{
    init, initialized, set_truncation_order, truncation_order, CompiledDa, Da, DaceError,
};

/// 把 dace-rs 的 `panic_any(DaceError)` 转为 `Err(DaceError)`。
///
/// 非 DaceError 载荷的意外 panic 归一为 code 1006。
pub fn catch_da_panic<T>(op: impl FnOnce() -> T) -> Result<T, DaceError> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(op)).map_err(|payload| {
        payload
            .downcast_ref::<DaceError>()
            .cloned()
            .unwrap_or_else(|| DaceError::new(1006, "non-DACE panic payload in DA operation"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catch_converts_div_zero_panic() {
        let _ = init(6, 2);
        let err = catch_da_panic(|| Da::constant(1.0) / Da::constant(0.0)).unwrap_err();
        assert_eq!(err.code, 641);
    }

    #[test]
    fn mixed_context_is_isolated() {
        let _ = init(6, 2);
        let a = Da::constant(1.0);
        let _ = init(6, 2);
        let b = Da::constant(1.0);
        let err = catch_da_panic(|| a.clone() + b.clone()).unwrap_err();
        assert_eq!(err.code, 1003);
    }
}
