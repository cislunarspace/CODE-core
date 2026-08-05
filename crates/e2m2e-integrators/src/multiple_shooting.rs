//! 多重打靶法（Multiple Shooting）Rust 实现。
//!
//! 从 Python `e2m2e.algorithms.multiple_shooting` 迁移核心算法，
//! 使用 `compiled_stm.rs` 的传播原语，替代 Python 的迭代循环。
//!
//! ## 算法概述
//!
//! 1. 残差向量构建：F_i = φ(t_{i+1}; t_i, x_i) - x_{i+1}
//! 2. 雅可比矩阵组装：DF[i] = [Φ_i, -I_6]（固定时间）或 [Φ_i, -I_6, -f_i, f_{i+1}]（自由时间）
//! 3. 最小二乘求解：dX = -(DF^T DF)^{-1} DF^T F
//! 4. 更新：x_new = x_old + α * dX

use e2m2e_forces::forces::compiled::{compute_total_acceleration_and_jacobian, CompiledForce};
use e2m2e_forces::forces::compiled_stm::propagate_compiled_stm;
use e2m2e_propagation::rk_methods::RkMethod;
use pyo3::prelude::*;

/// 最小二乘求解结果。
struct LeastSquaresResult {
    dx: Vec<f64>,
}

/// 一段积分的结果：终端 STM、终端状态、段首状态导数、段末状态导数。
type SegmentInfo = ([f64; 36], [f64; 6], [f64; 6], [f64; 6]);

/// 多重打靶迭代结果。
#[pyclass]
#[derive(Clone)]
pub struct MultipleShootingRustResult {
    /// 修正后的时间节点，形状 (N,)
    #[pyo3(get)]
    pub t_patch: Vec<f64>,
    /// 修正后的状态量，形状 (N, 6)
    #[pyo3(get)]
    pub state_patch: Vec<[f64; 6]>,
    /// 是否收敛
    #[pyo3(get)]
    pub converged: bool,
    /// 实际迭代次数
    #[pyo3(get)]
    pub iterations: usize,
    /// 最终最大残差
    #[pyo3(get)]
    pub max_residual: f64,
    /// 每次迭代最大残差的历史
    #[pyo3(get)]
    pub residual_history: Vec<f64>,
}

/// 构建残差向量 F。
///
/// 残差定义：F_i = φ(t_{i+1}; t_i, x_i) - x_{i+1}
/// 即第 i 段积分终端状态与第 i+1 个节点状态的差值。
fn build_residual(final_states: &[[f64; 6]], state_work: &[[f64; 6]], n_seg: usize) -> Vec<f64> {
    let mut f = vec![0.0_f64; n_seg * 6];
    for i in 0..n_seg {
        for j in 0..6 {
            f[i * 6 + j] = final_states[i][j] - state_work[i + 1][j];
        }
    }
    f
}

/// 构建雅可比矩阵 DF（固定时间模式）。
///
/// 自由变量列块 j 对应第 j 个**自由节点**（由 ``free_pos`` 映射：
/// ``free_pos[i]`` = 节点 i 在自由变量序列中的位置，固定节点为 ``None``）。
/// 残差 F_i = φ(x_i) - x_{i+1}，故：
///   ∂F_i/∂x_i     = Φ_i，列块 = free_pos[i]（i 自由时存在）
///   ∂F_i/∂x_{i+1} = -I_6，列块 = free_pos[i+1]（i+1 自由时存在）
/// 固定节点的列块整块省略（对自由变量雅可比无贡献）。
fn build_jacobian_fixed_time(
    stms: &[[f64; 36]],
    n_seg: usize,
    n_vars: usize,
    free_pos: &[Option<usize>],
) -> Vec<f64> {
    let n_constraints = n_seg * 6;
    let mut df = vec![0.0_f64; n_constraints * n_vars];

    for i in 0..n_seg {
        let r_start = i * 6;
        // ∂F_i/∂x_i = Φ_i（仅当 x_i 是自由变量）
        if let Some(col_block) = free_pos[i] {
            for row in 0..6 {
                for col in 0..6 {
                    df[(r_start + row) * n_vars + col_block * 6 + col] = stms[i][row * 6 + col];
                }
            }
        }
        // ∂F_i/∂x_{i+1} = -I_6（仅当 x_{i+1} 是自由变量）
        if let Some(col_block) = free_pos[i + 1] {
            for j in 0..6 {
                df[(r_start + j) * n_vars + col_block * 6 + j] = -1.0;
            }
        }
    }
    df
}

/// 构建雅可比矩阵 DF（自由时间模式）。
///
/// 自由变量列块 j 对应第 j 个自由节点（``free_pos`` 映射）；时间变量接在
/// 状态变量之后，从 ``n_free_nodes*6`` 开始。
///   ∂F_i/∂x_i     = Φ_i，列块 = free_pos[i]
///   ∂F_i/∂x_{i+1} = -I_6，列块 = free_pos[i+1]
///   ∂F_i/∂t_i     = -f(t_i, x_i)
///   ∂F_i/∂t_{i+1} = f(t_{i+1}, φ_i)
fn build_jacobian_variable_time(
    stms: &[[f64; 36]],
    f_starts: &[[f64; 6]],
    f_ends: &[[f64; 6]],
    n_seg: usize,
    n_vars: usize,
    n_free_nodes: usize,
    free_pos: &[Option<usize>],
) -> Vec<f64> {
    let n_constraints = n_seg * 6;
    let mut df = vec![0.0_f64; n_constraints * n_vars];

    for i in 0..n_seg {
        let r_start = i * 6;
        // ∂F_i/∂x_i = Φ_i（仅当 x_i 是自由变量）
        if let Some(col_block) = free_pos[i] {
            for row in 0..6 {
                for col in 0..6 {
                    df[(r_start + row) * n_vars + col_block * 6 + col] = stms[i][row * 6 + col];
                }
            }
        }
        // ∂F_i/∂x_{i+1} = -I_6（仅当 x_{i+1} 是自由变量）
        if let Some(col_block) = free_pos[i + 1] {
            for j in 0..6 {
                df[(r_start + j) * n_vars + col_block * 6 + j] = -1.0;
            }
        }
        // ∂F_i/∂t_i = -f(t_i, x_i)
        for j in 0..6 {
            df[(r_start + j) * n_vars + n_free_nodes * 6 + i] = -f_starts[i][j];
        }
        // ∂F_i/∂t_{i+1} = f(t_{i+1}, φ_i)
        for j in 0..6 {
            df[(r_start + j) * n_vars + n_free_nodes * 6 + i + 1] = f_ends[i][j];
        }
    }
    df
}

/// 阻尼最小二乘求解：dX = (D^T D + λ I)^{-1} D^T (-F)
///
/// 用正规方程 + 高斯消元（部分主元）解：
///   (D^T D + λ I) dX = -D^T F
/// λ 为 Levenberg-Marquardt 阻尼项：
/// - 恰定/过定系统 λ≈0 退化为 Gauss-Newton 最小二乘；
/// - 欠定系统（节点全自由，6N 变量 > 6(N-1) 约束，论文式13 语义）
///   D^T D 半正定奇异，加 λI 正定化后解是
///   min ‖D dX + F‖² + λ‖dX‖²，即最小范数偏好——解停留在初猜附近，
///   这正是"保形"的数学来源（论文 `C_{k+1}=C_k-D^T(DD^T)^{-1}F`）。
///   λ 随线搜索失败增大（见 `multiple_shooting_correct`），成功时趋小。
fn least_squares_solve(
    df: &[f64],
    f: &[f64],
    n_constraints: usize,
    n_vars: usize,
    lam: f64,
) -> LeastSquaresResult {
    // 计算 DF^T (-F)
    let mut dtf = vec![0.0_f64; n_vars];
    for i in 0..n_vars {
        for j in 0..n_constraints {
            dtf[i] += df[j * n_vars + i] * (-f[j]); // 注意负号
        }
    }

    // 计算 DF^T DF + λI
    let mut dtd = vec![0.0_f64; n_vars * n_vars];
    for i in 0..n_vars {
        for j in 0..n_vars {
            let mut s = 0.0;
            for k in 0..n_constraints {
                s += df[k * n_vars + i] * df[k * n_vars + j];
            }
            dtd[i * n_vars + j] = s;
        }
        dtd[i * n_vars + i] += lam;
    }

    // 求解 (DF^T DF + λI) dX = DF^T (-F)
    // 使用高斯消元法（简化版，适合小规模问题）
    let mut augmented = vec![0.0_f64; n_vars * (n_vars + 1)];
    for i in 0..n_vars {
        for j in 0..n_vars {
            augmented[i * (n_vars + 1) + j] = dtd[i * n_vars + j];
        }
        augmented[i * (n_vars + 1) + n_vars] = dtf[i]; // 注意：这里已经是 -DF^T F
    }

    // 前向消元
    for i in 0..n_vars {
        // 选主元
        let mut max_val = augmented[i * (n_vars + 1) + i].abs();
        let mut max_row = i;
        for k in (i + 1)..n_vars {
            if augmented[k * (n_vars + 1) + i].abs() > max_val {
                max_val = augmented[k * (n_vars + 1) + i].abs();
                max_row = k;
            }
        }
        // 交换行
        if max_row != i {
            for j in 0..=n_vars {
                augmented.swap(i * (n_vars + 1) + j, max_row * (n_vars + 1) + j);
            }
        }

        // 消元
        let pivot = augmented[i * (n_vars + 1) + i];
        if pivot.abs() < 1e-15 {
            continue; // 奇异矩阵，跳过
        }
        for j in (i + 1)..n_vars {
            let factor = augmented[j * (n_vars + 1) + i] / pivot;
            for k in i..=n_vars {
                augmented[j * (n_vars + 1) + k] -= factor * augmented[i * (n_vars + 1) + k];
            }
        }
    }

    // 回代
    let mut dx = vec![0.0_f64; n_vars];
    for i in (0..n_vars).rev() {
        let mut sum = augmented[i * (n_vars + 1) + n_vars];
        for j in (i + 1)..n_vars {
            sum -= augmented[i * (n_vars + 1) + j] * dx[j];
        }
        let pivot = augmented[i * (n_vars + 1) + i];
        if pivot.abs() > 1e-15 {
            dx[i] = sum / pivot;
        }
    }

    LeastSquaresResult { dx }
}

/// 多重打靶迭代修正（Rust 实现）。
///
/// 求解器语义对齐朱彦伟 2026《星历模型下基于多重打靶拼接的长期近直线晕
/// 轨道设计方法》式(13)：**阻尼最小二乘 + 回溯线搜索**。
/// - 欠定系统（节点全自由，6N 变量 > 6(N-1) 约束）下 LM 阻尼项使解趋
///   向**最小范数**（停留在初猜附近），即"保形"的数学来源；
/// - 回溯线搜索（α 递减到残差下降）防止不稳定轨道全步长 Gauss-Newton
///   一步发散（STM 谱半径 ~1e7/圈）。
///
/// # Arguments
/// * `forces` - 编译后的力模型列表
/// * `observer` - 坐标原点天体名（如 "EARTH"）
/// * `t_patch` - 初始时间节点，形状 (N,)
/// * `state_patch` - 初始状态量，形状 (N, 6)
/// * `var_time` - 是否启用自由时间修正（自由变量含时间节点，论文式9）
/// * `fix_first_node` - 固定首节点（兼容旧接口）。等价于
///   `fixed_node_mask=[true,false,...]`。
/// * `fixed_node_mask` - 固定任意节点子集（`None` 时用 `fix_first_node`）。
///   拼接/锚定远月点需要固定段首/两端。长度必须等于节点数。
/// * `max_iter` - 最大迭代次数
/// * `tolerance` - 收敛容差
/// * `rtol` - 积分相对容差
/// * `max_step` - 积分最大步长（可选）
/// * `verbose` - 是否输出进度
#[allow(clippy::too_many_arguments)]
pub fn multiple_shooting_correct(
    forces: &[CompiledForce],
    observer: &str,
    t_patch: &[f64],
    state_patch: &[[f64; 6]],
    var_time: bool,
    fix_first_node: bool,
    fixed_node_mask: Option<&[bool]>,
    max_iter: usize,
    tolerance: f64,
    rtol: f64,
    max_step: Option<f64>,
    verbose: bool,
    method: RkMethod,
) -> Result<MultipleShootingRustResult, String> {
    let n_nodes = t_patch.len();
    let n_seg = n_nodes - 1;

    if n_nodes < 2 {
        return Err("need at least 2 patch points".to_string());
    }
    if state_patch.len() != n_nodes {
        return Err(format!(
            "state_patch length {} != t_patch length {}",
            state_patch.len(),
            n_nodes
        ));
    }
    if let Some(mask) = fixed_node_mask {
        if mask.len() != n_nodes {
            return Err(format!(
                "fixed_node_mask length {} != n_nodes {}",
                mask.len(),
                n_nodes
            ));
        }
    }

    let mut t_work = t_patch.to_vec();
    let mut state_work = state_patch.to_vec();
    let mut residual_history = Vec::new();
    let mut converged = false;

    // 自由节点映射：free_pos[i] = 节点 i 在自由变量序列中的位置（固定节点为 None）。
    // 由 fixed_node_mask（若给出）或 fix_first_node（兼容）派生。
    let mut free_pos = vec![None; n_nodes];
    let mut n_free_nodes = 0usize;
    for i in 0..n_nodes {
        let fixed = if let Some(mask) = fixed_node_mask {
            mask[i]
        } else {
            fix_first_node && i == 0
        };
        if !fixed {
            free_pos[i] = Some(n_free_nodes);
            n_free_nodes += 1;
        }
    }
    if n_free_nodes == 0 {
        return Err("all nodes fixed, no free variables".to_string());
    }
    let n_vars = if var_time {
        n_free_nodes * 6 + n_nodes
    } else {
        n_free_nodes * 6
    };

    // Levenberg-Marquardt 阻尼初值：相对 D^T D 对角元量级，欠定系统保 min-norm。
    // **跨迭代保留**（成功迭代减小、失败增大）：标准 LM 做法。若不保留而每次
    // 重置，残差在局部极小处反复尝试同一条失败路径（实测停滞 8e-2 km 压不下，
    // 保留阻尼可到 1.6e-2）。见 multiple_shooting_correct 主循环。
    let lam0 = 1e-6;
    let mut lam = lam0;
    // 停滞计数：连续无改善迭代数（提前停，避免局部极小反复浪费）
    let mut stall_count = 0usize;

    // strict 缓存模式：打靶全程力模型查星历表，miss 即硬 Err，杜绝任何
    // 静默回退 cspice（并行段积分的 cspice 是内核池损坏/panic 的根源）。
    let _strict = e2m2e_spice::ephem_cache::StrictGuard::new();
    // 并行开关：E2M2E_MS_PARALLEL=0 强制串行（验证并行/串行位级一致性用）。
    // 默认并行——前提是缓存已启用且 strict（段积分内零 cspice）。
    let parallel = std::env::var("E2M2E_MS_PARALLEL").map_or(true, |v| v != "0");

    for iteration in 0..max_iter {
        // 第一步：逐段积分，收集 STM、终端状态。
        // 段间相互独立（只依赖本段起始状态），rayon 并行段积分线性加速。
        // 并行安全前提：星历预采样缓存已启用 + strict 模式——段积分内力
        // 模型查内存三次样条，零 cspice FFI（cspice 并发会让多线程同时调
        // easier_reader 触发全局锁损坏，报 SPICE(DAFFRNOTFOUND) 或 panic）。
        // par_iter 保序 + 各段积分确定 → 并行与串行位级一致。
        // 顺带求段两端点的状态导数 [v, a]（自由时间模式雅可比需要）。
        let integrate_seg = |i: usize| -> Result<SegmentInfo, String> {
            let t_span = (t_work[i], t_work[i + 1]);
            let result = propagate_compiled_stm(
                forces,
                observer,
                t_span,
                &[t_work[i], t_work[i + 1]],
                &state_work[i],
                rtol,
                rtol * 0.1,
                max_step,
                Some(500_000),
                method,
            )?;

            let final_state = *result.states.last().ok_or("empty propagation result")?;
            let final_stm = *result.stms.last().ok_or("empty STM result")?;

            let (a0, _) = compute_total_acceleration_and_jacobian(
                forces,
                t_work[i],
                &state_work[i],
                observer,
            )?;
            let (a1, _) = compute_total_acceleration_and_jacobian(
                forces,
                t_work[i + 1],
                &final_state,
                observer,
            )?;
            let f_start = [
                state_work[i][3],
                state_work[i][4],
                state_work[i][5],
                a0[0],
                a0[1],
                a0[2],
            ];
            let f_end = [
                final_state[3],
                final_state[4],
                final_state[5],
                a1[0],
                a1[1],
                a1[2],
            ];

            Ok((final_stm, final_state, f_start, f_end))
        };
        let seg_infos: Vec<Result<SegmentInfo, String>> = if parallel {
            use rayon::prelude::*;
            (0..n_seg).into_par_iter().map(integrate_seg).collect()
        } else {
            (0..n_seg).map(integrate_seg).collect()
        };

        let mut stms = Vec::with_capacity(n_seg);
        let mut final_states = Vec::with_capacity(n_seg);
        let mut f_starts = Vec::with_capacity(n_seg);
        let mut f_ends = Vec::with_capacity(n_seg);
        for info in seg_infos {
            let (final_stm, final_state, f_start, f_end) = info?;
            stms.push(final_stm);
            final_states.push(final_state);
            f_starts.push(f_start);
            f_ends.push(f_end);
        }

        // 第二步：构建残差向量
        let f = build_residual(&final_states, &state_work, n_seg);
        let max_res = f.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        residual_history.push(max_res);

        if verbose {
            eprintln!(
                "Iteration {}: max_residual = {:.2e}",
                iteration + 1,
                max_res
            );
        }

        // 判断收敛
        if max_res < tolerance {
            converged = true;
            break;
        }

        // 停滞检测：连续迭代残差相对改善极小即提前停。阈值取 0.05%（比
        // Gauss-Newton 停滞点更严），避免在 LM 阻尼还能探索的方向上过早截断
        // ——跨迭代 LM 阻尼（成功减、失败增）可越过浅局部极小继续收敛。
        if let Some(prev) = residual_history.iter().rev().nth(1) {
            let improvement = 1.0 - max_res / *prev;
            if improvement < 5e-4 {
                stall_count += 1;
                if stall_count >= 5 {
                    if verbose {
                        eprintln!(
                            "Iteration {}: stalled (improvement {:.3}%), stopping",
                            iteration + 1,
                            improvement
                        );
                    }
                    break;
                }
            } else {
                stall_count = 0;
            }
        }

        // 第三步：构建雅可比矩阵
        let df = if var_time {
            build_jacobian_variable_time(
                &stms,
                &f_starts,
                &f_ends,
                n_seg,
                n_vars,
                n_free_nodes,
                &free_pos,
            )
        } else {
            build_jacobian_fixed_time(&stms, n_seg, n_vars, &free_pos)
        };

        // 第四步：LM 阻尼最小二乘 + 回溯线搜索。
        // 阻尼项 λ 跨迭代保留：成功时减小（朝 Gauss-Newton），失败时增大
        // （朝 min-norm/最速下降）。全步长 Gauss-Newton 对不稳定轨道（STM
        // 谱半径 1e7/圈）一步就发散，线搜索 + 阻尼是收敛的工程必需。
        let mut accepted = false;
        for _ in 0..8 {
            let ls_result = least_squares_solve(&df, &f, n_seg * 6, n_vars, lam);

            // 第五步：更新变量（回溯线搜索，从 α=1 减半，直到残差下降）
            let mut alpha = 1.0_f64;
            for _ in 0..16 {
                let t_try = apply_dx_t(&t_work, &ls_result.dx, n_free_nodes, var_time, alpha);
                let s_try = apply_dx_state(
                    &state_work,
                    &ls_result.dx,
                    &free_pos,
                    var_time,
                    n_free_nodes,
                    alpha,
                );
                match try_residual(forces, observer, &t_try, &s_try, rtol, max_step, method) {
                    Ok(trial_res) if trial_res < max_res => {
                        t_work = t_try;
                        state_work = s_try;
                        accepted = true;
                        break;
                    }
                    _ => {
                        alpha *= 0.5;
                    }
                }
            }
            if accepted {
                lam = (lam * 0.5).max(1e-10); // 成功：减小阻尼
                break;
            }
            lam *= 10.0; // 失败：增大阻尼，朝 min-norm 方向靠
        }
        if !accepted {
            // 线搜索连续失败：残差停滞在局部极小，接受当前解（保形已由
            // min-norm 保证，不再强推数值收敛）
            if verbose {
                eprintln!(
                    "Iteration {}: line search stalled, stopping at residual {:.2e}",
                    iteration + 1,
                    max_res
                );
            }
            break;
        }
    }

    Ok(MultipleShootingRustResult {
        t_patch: t_work,
        state_patch: state_work,
        converged,
        iterations: residual_history.len(),
        max_residual: *residual_history.last().unwrap_or(&f64::INFINITY),
        residual_history,
    })
}

/// 对状态应用修正量 dX（只更新自由节点，按 free_pos 映射）。
#[allow(clippy::too_many_arguments)]
fn apply_dx_state(
    state_work: &[[f64; 6]],
    dx: &[f64],
    free_pos: &[Option<usize>],
    var_time: bool,
    n_free_nodes: usize,
    alpha: f64,
) -> Vec<[f64; 6]> {
    let mut out = state_work.to_vec();
    for (i, s) in out.iter_mut().enumerate() {
        if let Some(pos) = free_pos[i] {
            let base = pos * 6;
            for j in 0..6 {
                s[j] += alpha * dx[base + j];
            }
        }
    }
    let _ = (var_time, n_free_nodes); // 状态更新不依赖时间列，保留签名对称
    out
}

/// 对时间节点应用修正量（仅自由时间模式）。
fn apply_dx_t(
    t_work: &[f64],
    dx: &[f64],
    n_free_nodes: usize,
    var_time: bool,
    alpha: f64,
) -> Vec<f64> {
    let mut out = t_work.to_vec();
    if var_time {
        for (i, t) in out.iter_mut().enumerate() {
            *t += alpha * dx[n_free_nodes * 6 + i];
        }
    }
    out
}

/// 试算一组状态/时间节点下的最大残差（供回溯线搜索验收）。积分失败视为不下降。
fn try_residual(
    forces: &[CompiledForce],
    observer: &str,
    t_work: &[f64],
    state_work: &[[f64; 6]],
    rtol: f64,
    max_step: Option<f64>,
    method: RkMethod,
) -> Result<f64, String> {
    let n_seg = t_work.len() - 1;
    let mut final_states = Vec::with_capacity(n_seg);
    for i in 0..n_seg {
        let result = propagate_compiled_stm(
            forces,
            observer,
            (t_work[i], t_work[i + 1]),
            &[t_work[i], t_work[i + 1]],
            &state_work[i],
            rtol,
            rtol * 0.1,
            max_step,
            Some(500_000),
            method,
        )?;
        final_states.push(*result.states.last().ok_or("empty propagation result")?);
    }
    let f = build_residual(&final_states, state_work, n_seg);
    Ok(f.iter().map(|x| x.abs()).fold(0.0_f64, f64::max))
}

/// PyO3 接口：多重打靶迭代修正。
///
/// `forces` 是 Python 元组列表，每个元组描述一个力模型（格式同 `propagate_compiled`）。
#[pyfunction]
#[pyo3(signature = (forces, observer, t_patch, state_patch, var_time=false, fix_first_node=false, fixed_node_mask=None, max_iter=50, tolerance=1e-8, rtol=1e-10, max_step=None, verbose=false, method=RkMethod::Pd78))]
#[allow(clippy::too_many_arguments)]
pub fn multiple_shooting_correct_py(
    forces: Vec<PyObject>,
    observer: &str,
    t_patch: Vec<f64>,
    state_patch: Vec<Vec<f64>>,
    var_time: bool,
    fix_first_node: bool,
    fixed_node_mask: Option<Vec<bool>>,
    max_iter: usize,
    tolerance: f64,
    rtol: f64,
    max_step: Option<f64>,
    verbose: bool,
    method: RkMethod,
    py: Python<'_>,
) -> PyResult<MultipleShootingRustResult> {
    // 解析 forces: Vec<PyObject> -> Vec<CompiledForce>
    if forces.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "forces must not be empty",
        ));
    }
    let mut compiled_forces: Vec<CompiledForce> = Vec::with_capacity(forces.len());
    for item in &forces {
        compiled_forces.push(crate::parse_force_tuple(&item.bind(py).as_borrowed())?);
    }

    // 转换 state_patch: Vec<Vec<f64>> -> Vec<[f64; 6]>
    let state_array: Vec<[f64; 6]> = state_patch
        .iter()
        .map(|s| {
            if s.len() != 6 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "state must have 6 elements",
                ));
            }
            Ok([s[0], s[1], s[2], s[3], s[4], s[5]])
        })
        .collect::<PyResult<Vec<_>>>()?;

    // 调用 Rust 核心。包 allow_threads：多重打靶段积分用 rayon 并行，核心
    // 纯 Rust 不碰 Python 对象，释放 GIL 让出给其它线程（性能，非正确性）。
    py.allow_threads(move || {
        multiple_shooting_correct(
            &compiled_forces,
            observer,
            &t_patch,
            &state_array,
            var_time,
            fix_first_node,
            fixed_node_mask.as_deref(),
            max_iter,
            tolerance,
            rtol,
            max_step,
            verbose,
            method,
        )
    })
    .map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_residual() {
        let final_states = vec![
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ];
        let state_work = vec![
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ];
        let f = build_residual(&final_states, &state_work, 2);
        assert_eq!(f.len(), 12);
        // 第一段残差：final_states[0] - state_work[1] = [0, 0, 0, 0, 0, 0]
        for i in 0..6 {
            assert!((f[i] - 0.0).abs() < 1e-15);
        }
        // 第二段残差：final_states[1] - state_work[2] = [0, 0, 0, 0, 0, 0]
        for i in 6..12 {
            assert!((f[i] - 0.0).abs() < 1e-15);
        }
    }

    #[test]
    fn test_build_jacobian_fixed_time() {
        let stms = vec![[0.0_f64; 36]; 2];
        // 全自由：free_pos = [Some(0), Some(1), Some(2)]，列块 j 对应 x_j
        let free_pos = [Some(0usize), Some(1), Some(2)];
        let df = build_jacobian_fixed_time(&stms, 2, 18, &free_pos);
        assert_eq!(df.len(), 12 * 18);
        // 验证结构：DF[0:6, 0:6] = Φ_0 = 0（stms 全 0）
        // DF[0:6, 6:12] = -I_6（∂F_0/∂x_1）
        for i in 0..6 {
            assert!((df[i * 18 + 6 + i] - (-1.0)).abs() < 1e-15);
        }
        // DF[6:12, 6:12] = Φ_1 = 0（∂F_1/∂x_1）
        for i in 0..6 {
            assert!((df[6 * 18 + 6 + i] - 0.0).abs() < 1e-15);
        }
        // DF[6:12, 12:18] = -I_6（∂F_1/∂x_2，对角线为 -1）
        for i in 0..6 {
            assert!((df[(6 + i) * 18 + 12 + i] - (-1.0)).abs() < 1e-15);
        }
    }

    #[test]
    fn test_build_jacobian_fixed_time_fix_first_node() {
        let stms = vec![[0.0_f64; 36]; 2];
        // fix_first_node 兼容：3 个节点，固定 x_0，自由变量 = [x_1, x_2]，
        // n_vars = 2*6 = 12。列块 0 对应 x_1，列块 1 对应 x_2。
        let free_pos = [None, Some(0usize), Some(1)];
        let df = build_jacobian_fixed_time(&stms, 2, 12, &free_pos);
        assert_eq!(df.len(), 12 * 12);
        // DF[0:6, 0:6] = -I_6（∂F_0/∂x_1，x_0 固定无列）
        for i in 0..6 {
            assert!((df[i * 12 + i] - (-1.0)).abs() < 1e-15);
        }
        // DF[6:12, 0:6] = Φ_1 = 0（∂F_1/∂x_1）
        for i in 0..6 {
            assert!((df[6 * 12 + i] - 0.0).abs() < 1e-15);
        }
        // DF[6:12, 6:12] = -I_6（∂F_1/∂x_2，对角线为 -1）
        for i in 0..6 {
            assert!((df[(6 + i) * 12 + 6 + i] - (-1.0)).abs() < 1e-15);
        }
    }

    #[test]
    fn test_build_jacobian_fixed_time_fix_both_ends() {
        let stms = vec![[0.0_f64; 36]; 3];
        // 4 节点固定首末（合并段拼接锚点）：free_pos = [None, Some(0), Some(1), None]，
        // n_vars = 2*6 = 12，n_seg = 3。
        let free_pos = [None, Some(0usize), Some(1), None];
        let df = build_jacobian_fixed_time(&stms, 3, 12, &free_pos);
        assert_eq!(df.len(), 18 * 12);
        // DF[0:6, 0:6] = -I_6（∂F_0/∂x_1，x_0 固定、x_1 自由）
        for i in 0..6 {
            assert!((df[i * 12 + i] - (-1.0)).abs() < 1e-15);
        }
        // DF[6:12, 0:6] = Φ_1 = 0（∂F_1/∂x_1，stms 全 0）
        // DF[6:12, 6:12] = -I_6（∂F_1/∂x_2）
        for i in 0..6 {
            assert!((df[(6 + i) * 12 + 6 + i] - (-1.0)).abs() < 1e-15);
        }
        // 第三段（i=2）：∂F_2/∂x_2 = Φ_2 = 0（stms 全 0），
        // ∂F_2/∂x_3 = -I 但 x_3 固定 → 无列。故 DF[12:18, :] 整块全 0。
        for i in 0..6 {
            for j in 0..12 {
                assert!((df[(12 + i) * 12 + j] - 0.0).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn test_least_squares_solve() {
        // 求解 (DF^T DF + λI) dX = DF^T (-F)，λ=0 退化为最小二乘
        // DF = [[1, 1], [1, -1]], F = [3, 1]
        // -F = [-3, -1]
        // 解：dX = [-2, -1]（因为 [[1,1],[1,-1]] [-2,-1] = [-3, -1]）
        let df = vec![1.0, 1.0, 1.0, -1.0];
        let f = vec![3.0, 1.0];
        let result = least_squares_solve(&df, &f, 2, 2, 0.0);
        assert!((result.dx[0] - (-2.0)).abs() < 1e-10);
        assert!((result.dx[1] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_least_squares_solve_underdetermined_damped() {
        // 欠定系统（1 约束 2 变量，论文式13 语义）：DF = [[1, 0]], F = [2]
        // 无阻尼最小范数解 dX = [-2, 0]（min-norm，停留在 y 方向不动）
        let df = vec![1.0, 0.0];
        let f = vec![2.0];
        let result = least_squares_solve(&df, &f, 1, 2, 1e-12);
        // 阻尼极小：接近 min-norm 解 [-2, 0]
        assert!((result.dx[0] - (-2.0)).abs() < 1e-8);
        assert!(result.dx[1].abs() < 1e-8);
    }
}
