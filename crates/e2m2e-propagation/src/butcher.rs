//! 显式 Runge-Kutta 通用驱动器，由 Butcher 表参数化。
//!
//! 所有显式嵌入 RK 方法（PD45、PD78、RK89 等）共享同样的单步结构；
//! 仅 Butcher 系数不同。本模块存放共享的 [`explicit_rk_step`] 驱动器
//! 与阶数感知的 [`suggest_next_step`] 启发式，各具体方法只需贡献一个
//! [`ButcherTable`] 常量。

/// 显式嵌入 Runge-Kutta 方法的 Butcher 表。
///
/// 所有系数切片均为 `&'static` —— 表为编译期常量，来源为 GMAT / 文献。
/// 整体维度由 [`ButcherTable::new`] 在构造时校验（`const` 情形下在编译期校验）；
/// 每行下三角形状由各方法自身的单元测试覆盖。
pub struct ButcherTable {
    /// 级数 `s`。
    pub stages: usize,
    /// 主解（高阶解）的阶数 `p`。
    pub order: usize,
    /// 用于误差估计的嵌入（低阶）解的阶数。
    pub embedded_order: usize,
    /// 时间节点 `c[i]`（长度 `stages`）；`c[0]` 惯例为 0。
    pub c: &'static [f64],
    /// Runge-Kutta 矩阵行 `a[i]`；第 `i` 行长度为 `i`（严格下三角）。
    pub a: &'static [&'static [f64]],
    /// 主解（高阶）权重 `b`（长度 `stages`）。
    pub b: &'static [f64],
    /// 嵌入（低阶）解的权重 `b_star`（长度 `stages`）。
    pub b_star: &'static [f64],
}

impl ButcherTable {
    /// 构造一张表，校验整体维度一致性。
    ///
    /// 供 `const` 初始化器使用，使格式错误的表在编译期失败。
    /// 断言 `c`、`b`、`b_star`、`a` 各有 `stages` 个元素。
    /// 逐行长度（`a[i].len() == i`）因 `const fn` 中尚不允许逐元素切片索引，
    /// 由各方法的测试强制执行。
    pub const fn new(
        stages: usize,
        order: usize,
        embedded_order: usize,
        c: &'static [f64],
        a: &'static [&'static [f64]],
        b: &'static [f64],
        b_star: &'static [f64],
    ) -> Self {
        assert!(c.len() == stages);
        assert!(b.len() == stages);
        assert!(b_star.len() == stages);
        assert!(a.len() == stages);
        Self {
            stages,
            order,
            embedded_order,
            c,
            a,
            b,
            b_star,
        }
    }
}

/// 使用 `table` 执行一次显式 Runge-Kutta 单步。
///
/// 返回主解（高阶解）与局部误差估计。误差为主解与嵌入解之差的 L2 范数，
/// 仅统计前 ``error_dim`` 个分量（``error_dim`` 为 ``None`` 时统计全部）。
///
/// STM 增广传播时，物理状态占前 6 维、状态转移矩阵展平占后 36 维。后者不
/// 参与步长误差控制（与 GMAT 一致：STM 元素禁用误差控制），否则 STM 分量
/// 会主导 L2 范数、导致步长过小。
pub fn explicit_rk_step<F, E>(
    table: &ButcherTable,
    t: f64,
    y: &[f64],
    h: f64,
    f: F,
    error_dim: Option<usize>,
) -> Result<(Vec<f64>, f64), E>
where
    F: Fn(f64, &[f64]) -> Result<Vec<f64>, E>,
{
    let n = y.len();
    let s = table.stages;

    debug_assert_eq!(table.c.len(), s);
    debug_assert_eq!(table.b.len(), s);
    debug_assert_eq!(table.b_star.len(), s);
    debug_assert_eq!(table.a.len(), s);

    let mut k = vec![vec![0.0; n]; s];
    k[0] = f(t, y)?;

    for i in 1..s {
        let ti = t + h * table.c[i];
        let mut yi = y.to_vec();
        let ai = table.a[i];
        for (j, &aij) in ai.iter().enumerate() {
            for l in 0..n {
                yi[l] += h * aij * k[j][l];
            }
        }
        k[i] = f(ti, &yi)?;
    }

    let mut y_high = vec![0.0; n];
    let mut y_low = vec![0.0; n];
    for l in 0..n {
        for ((bi, bstar_i), ki) in table.b.iter().zip(table.b_star.iter()).zip(k.iter()) {
            y_high[l] += bi * ki[l];
            y_low[l] += bstar_i * ki[l];
        }
        y_high[l] = y[l] + h * y_high[l];
        y_low[l] = y[l] + h * y_low[l];
    }

    // 误差只统计前 error_dim 维（默认全部），让 STM 分量不主导步长控制。
    let dim = error_dim.unwrap_or(n).min(n);
    let error = y_high[..dim]
        .iter()
        .zip(y_low[..dim].iter())
        .map(|(hi, lo)| (hi - lo).powi(2))
        .sum::<f64>()
        .sqrt();

    Ok((y_high, error))
}

/// 由局部误差估计建议下一步步长。
///
/// 标准控制器：`h_next = h · clamp(0.9 · (tol/error)^(1/(p+1)), 0.1, 5)`，
/// 其中 `p = embedded_order`。指数与嵌入误差估计的阶数一致。
pub fn suggest_next_step(h: f64, error: f64, tol: f64, embedded_order: usize) -> f64 {
    if error == 0.0 {
        return h * 5.0;
    }
    let ratio = tol / error;
    let p = embedded_order as f64;
    let factor = 0.9 * ratio.powf(1.0 / (p + 1.0));
    h * factor.clamp(0.1, 5.0)
}
