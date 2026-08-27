//! 分段打靶拼接法（Segmented Shooting Splicing）Rust 实现。
//!
//! 参考朱彦伟等人的论文《星历模型下基于多重打靶拼接的长期近直线晕轨道设计方法》，
//! 实现分段打靶拼接策略，解决长转移轨迹在多重打靶中不收敛的问题。
//!
//! ## 核心思想
//!
//! 1. 将长轨迹按圈切段（每段 ``revs_per_group`` 圈，节点密集）
//! 2. 每小段独立用多重打靶转换到星历模型（段间独立，rayon 并行）
//! 3. 分层两两合并相邻段（同层配对合并段独立，rayon 并行），继续修正
//! 4. 重复直到所有段合并为一条完整轨迹
//!
//! ## 约束形式
//!
//! 第 1 步与合并层的打靶均不固定节点（``fix_first_node=False``、
//! ``fixed_node_mask=None``），LM 最小范数更新对齐文献。首末锚定会破坏
//! 合并层收敛：各段独立打靶后 seam 不连续，锚定把修正全压给内部节点
//! （60 天合并层残差停在 7.5e-01 km）；去锚定后同工况收敛到
//! 1.4e-03 km，180 天三层合并全程收敛。

use crate::multiple_shooting::{
    multiple_shooting_correct, MultipleShootingRustResult, SolverTermination,
};
use e2m2e_forces::forces::compiled::CompiledForce;
use e2m2e_propagation::rk_methods::RkMethod;
use pyo3::prelude::*;

/// 分段打靶拼接结果。
#[pyclass]
#[derive(Clone)]
pub struct SegmentedShootingResult {
    /// 修正后的时间节点
    #[pyo3(get)]
    pub t_patch: Vec<f64>,
    /// 修正后的状态量
    #[pyo3(get)]
    pub state_patch: Vec<Vec<f64>>,
    /// 最终状态（受控值，与多重打靶结果契约一致）。
    #[pyo3(get)]
    pub status: String,
    /// 最终原因（受控值，与 `status` 一一对应）。
    #[pyo3(get)]
    pub cause: String,
    /// 面向调用方的最终诊断信息。
    #[pyo3(get)]
    pub message: String,
    /// 总迭代次数
    #[pyo3(get)]
    pub total_iterations: usize,
    /// 最终残差
    #[pyo3(get)]
    pub max_residual: f64,
    /// 每阶段的残差历史（第 1 步各段 + 各合并层合并段，按执行顺序）
    #[pyo3(get)]
    pub stage_residuals: Vec<f64>,
    /// 分段数
    #[pyo3(get)]
    pub n_segments: usize,
}

impl SegmentedShootingResult {
    #[allow(clippy::too_many_arguments)]
    fn new(
        t_patch: Vec<f64>,
        state_patch: Vec<Vec<f64>>,
        total_iterations: usize,
        max_residual: f64,
        stage_residuals: Vec<f64>,
        n_segments: usize,
        outcome: SolverTermination,
        failure_site: Option<String>,
    ) -> Self {
        let (status, cause, message) = outcome.contract();
        let message = match failure_site {
            Some(site) => format!("{site}（残差 {max_residual:.3e}）"),
            None => message.to_string(),
        };
        Self {
            t_patch,
            state_patch,
            status: status.to_string(),
            cause: cause.to_string(),
            message,
            total_iterations,
            max_residual,
            stage_residuals,
            n_segments,
        }
    }
}

/// 将轨迹按圈切段。
///
/// 每段 ``revs_per_group`` 圈；相邻段共享 seam 节点（段 i 末节点 = 段 i+1
/// 首节点，同一时刻的同一 tile 节点），这是分层合并连续性的基础。切段
/// 语义与 Python 侧 ``_design_apolune_segmented`` 对齐。
fn segment_trajectory(
    t_patch: &[f64],
    state_patch: &[[f64; 6]],
    per_rev: usize,
    revs_per_group: usize,
) -> Vec<(Vec<f64>, Vec<[f64; 6]>)> {
    let n_total = t_patch.len();
    let n_rev = n_total / per_rev;
    let mut segments = Vec::new();
    let mut k = 0;
    while k < n_rev {
        let end = (k + revs_per_group).min(n_rev);
        let mut lo = k * per_rev;
        let mut hi = end * per_rev;
        if hi >= n_total {
            hi = n_total - 1;
        }
        // 首段含第 0 节点；后续段跳过前一共享节点（段间 seam）
        if !segments.is_empty() {
            lo += 1;
        }
        segments.push((t_patch[lo..=hi].to_vec(), state_patch[lo..=hi].to_vec()));
        k = end;
    }
    segments
}

/// 对单段进行多重打靶修正。
///
/// 不固定任何节点（``fix_first_node=False``、``fixed_node_mask=None``）：
/// 第 1 步与合并层统一走最小范数更新（对齐文献）。首末锚定会破坏合并层
/// 收敛，见模块注释。
#[allow(clippy::too_many_arguments)]
fn correct_segment(
    forces: &[CompiledForce],
    observer: &str,
    t_patch: &[f64],
    state_patch: &[[f64; 6]],
    var_time: bool,
    max_iter: usize,
    tolerance: f64,
    rtol: f64,
    vel_weight: f64,
    verbose: bool,
    method: RkMethod,
) -> Result<MultipleShootingRustResult, String> {
    multiple_shooting_correct(
        forces,
        observer,
        t_patch,
        state_patch,
        var_time,
        false, // fix_first_node：第 1 步/合并层均不固定节点
        None,  // fixed_node_mask：无锚定
        max_iter,
        tolerance,
        rtol,
        None,
        vel_weight,
        verbose,
        method,
    )
}

/// 分段打靶拼接法主函数。
///
/// 第 1 步各段独立打靶与合并层各配对合并段打靶均 rayon 并行：段/合并段
/// 之间相互独立（只依赖本段输入），并行前提与多重打靶段积分相同（strict +
/// 预采样星历缓存，零 cspice FFI）；rayon 保序 collect + 各打靶确定 →
/// 并行与串行位级一致（``E2M2E_MS_PARALLEL=0`` 可强制串行验证）。
#[allow(clippy::too_many_arguments)]
pub fn segmented_shooting_correct(
    forces: &[CompiledForce],
    observer: &str,
    t_patch: &[f64],
    state_patch: &[[f64; 6]],
    revs_per_group: usize,
    per_rev: usize,
    var_time: bool,
    max_iter_per_segment: usize,
    tolerance: f64,
    rtol: f64,
    vel_weight: f64,
    verbose: bool,
    method: RkMethod,
) -> Result<SegmentedShootingResult, String> {
    if t_patch.len() < 2 {
        return Err("need at least 2 patch points".to_string());
    }
    if per_rev == 0 {
        return Err("per_rev must be > 0".to_string());
    }

    let parallel = std::env::var("E2M2E_MS_PARALLEL").map_or(true, |v| v != "0");

    let segments = segment_trajectory(t_patch, state_patch, per_rev, revs_per_group);
    let n_segments = segments.len();
    let mut total_iterations = 0;
    let mut stage_residuals = Vec::new();
    let mut outcome = SolverTermination::Converged;
    let mut failure_site: Option<String> = None;
    let mut max_residual: f64 = 0.0;

    if verbose {
        eprintln!(
            "分段打靶拼接：{} 段，每段 {} 圈（{} 节点/圈）",
            n_segments, revs_per_group, per_rev
        );
    }

    // 第1步：对每小段独立进行多重打靶修正（段间独立，rayon 并行）
    let correct = |(i, (seg_t, seg_s)): (usize, &(Vec<f64>, Vec<[f64; 6]>))| {
        if verbose {
            eprintln!("  修正段 {}/{}: {} 点", i + 1, n_segments, seg_t.len());
        }
        let result = correct_segment(
            forces,
            observer,
            seg_t,
            seg_s,
            var_time,
            max_iter_per_segment,
            tolerance,
            rtol,
            vel_weight,
            verbose,
            method,
        )?;
        Ok::<_, String>((i, result))
    };
    let stage_results: Vec<Result<(usize, MultipleShootingRustResult), String>> = if parallel {
        use rayon::prelude::*;
        segments.par_iter().enumerate().map(correct).collect()
    } else {
        segments.iter().enumerate().map(correct).collect()
    };

    let mut corrected: Vec<(Vec<f64>, Vec<[f64; 6]>)> = Vec::with_capacity(n_segments);
    for res in stage_results {
        let (i, result) = res?;
        total_iterations += result.iterations;
        stage_residuals.push(result.max_residual);
        max_residual = max_residual.max(result.max_residual);
        if result.outcome() != SolverTermination::Converged {
            outcome = result.outcome();
            if failure_site.is_none() {
                failure_site = Some(format!("分段打靶段 {} 未收敛", i + 1));
            }
        }
        corrected.push((result.t_patch, result.state_patch));
    }

    // 第2步：分层两两合并（文献式多重打靶拼接）
    // 每层把当前段序列两两配对（(0,1),(2,3)...），合并出更长的段后各自独立
    // 修正，再进入下一层，直到只剩 1 段。分段首尾相接（共享 seam 节点）保证：
    // 合并后新段的末点时间戳 = 下一段首点时间戳，层间拼接连续。同层配对
    // 合并段独立（rayon 并行）。层数 log₂(n)，每层修正的段长翻倍但有界，
    // 逐层扩大连续弧长。
    let mut current_segments = corrected;
    let mut stage = 1;
    while current_segments.len() > 1 && outcome == SolverTermination::Converged {
        let n_merge = current_segments.len() / 2;
        if verbose {
            eprintln!(
                "合并阶段 {}: {} 段 -> {} 段",
                stage,
                current_segments.len(),
                current_segments.len() - n_merge,
            );
        }

        // 配对拼接：段 i 全部 + 段 i+1 去首（共享 seam 节点）
        let mut pairs: Vec<(Vec<f64>, Vec<[f64; 6]>)> = Vec::with_capacity(n_merge);
        let mut i = 0;
        while i + 1 < current_segments.len() {
            let (t1, s1) = &current_segments[i];
            let (t2, s2) = &current_segments[i + 1];
            let mut t_all = t1.clone();
            t_all.extend_from_slice(&t2[1..]);
            let mut s_all = s1.clone();
            s_all.extend_from_slice(&s2[1..]);
            pairs.push((t_all, s_all));
            i += 2;
        }

        let merge = |(pi, (t_all, s_all)): (usize, &(Vec<f64>, Vec<[f64; 6]>))| {
            if verbose {
                eprintln!("  合并层 {} 段 {}: {} 节点", stage, pi + 1, t_all.len());
            }
            let result = correct_segment(
                forces,
                observer,
                t_all,
                s_all,
                var_time,
                max_iter_per_segment,
                tolerance,
                rtol,
                vel_weight,
                verbose,
                method,
            )?;
            Ok::<_, String>((pi, result))
        };
        let merge_results: Vec<Result<(usize, MultipleShootingRustResult), String>> = if parallel {
            use rayon::prelude::*;
            pairs.par_iter().enumerate().map(merge).collect()
        } else {
            pairs.iter().enumerate().map(merge).collect()
        };

        let mut next_segments: Vec<(Vec<f64>, Vec<[f64; 6]>)> =
            Vec::with_capacity(current_segments.len() - n_merge);
        for res in merge_results {
            let (pi, result) = res?;
            total_iterations += result.iterations;
            stage_residuals.push(result.max_residual);
            max_residual = max_residual.max(result.max_residual);
            if result.outcome() != SolverTermination::Converged {
                outcome = result.outcome();
                if failure_site.is_none() {
                    failure_site = Some(format!("分层合并第 {stage} 层段 {} 未收敛", pi + 1));
                }
            }
            next_segments.push((result.t_patch, result.state_patch));
        }
        // 奇数段：最后一段直接继承（本层不合并）
        if current_segments.len() % 2 == 1 {
            next_segments.push(current_segments.pop().unwrap());
        }

        current_segments = next_segments;
        stage += 1;
    }

    // 组装整条轨迹：段间首尾相接，去重 seam 后拼接
    let mut all_t = Vec::new();
    let mut all_s = Vec::new();
    for (i, (seg_t, seg_s)) in current_segments.into_iter().enumerate() {
        if i == 0 {
            all_t.extend(seg_t);
            all_s.extend(seg_s);
        } else {
            all_t.extend_from_slice(&seg_t[1..]);
            all_s.extend_from_slice(&seg_s[1..]);
        }
    }
    let final_state: Vec<Vec<f64>> = all_s.iter().map(|s| s.to_vec()).collect();

    Ok(SegmentedShootingResult::new(
        all_t,
        final_state,
        total_iterations,
        max_residual,
        stage_residuals,
        n_segments,
        outcome,
        failure_site,
    ))
}

/// PyO3 接口：分段打靶拼接法。
///
/// `forces` 是 Python 元组列表，每个元组描述一个力模型（格式同 `propagate_compiled`）。
/// 与多重打靶入口一致，GIL 内只做参数解析，核心计算包 `allow_threads`
/// 释放 GIL（内部 rayon 并行，纯 Rust 不碰 Python 对象）。
#[pyfunction]
#[pyo3(signature = (forces, observer, t_patch, state_patch, revs_per_group=3, per_rev=12, var_time=false, max_iter_per_segment=50, tolerance=1e-8, rtol=1e-10, vel_weight=1.0, verbose=false, method=RkMethod::Pd78))]
#[allow(clippy::too_many_arguments)]
pub fn segmented_shooting_correct_py(
    forces: Vec<PyObject>,
    observer: &str,
    t_patch: Vec<f64>,
    state_patch: Vec<Vec<f64>>,
    revs_per_group: usize,
    per_rev: usize,
    var_time: bool,
    max_iter_per_segment: usize,
    tolerance: f64,
    rtol: f64,
    vel_weight: f64,
    verbose: bool,
    method: RkMethod,
    py: Python<'_>,
) -> PyResult<SegmentedShootingResult> {
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

    py.allow_threads(move || {
        segmented_shooting_correct(
            &compiled_forces,
            observer,
            &t_patch,
            &state_array,
            revs_per_group,
            per_rev,
            var_time,
            max_iter_per_segment,
            tolerance,
            rtol,
            vel_weight,
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
    fn test_segment_trajectory_by_revs() {
        // 每圈 4 点：12 点 tile（n_rev=3，revs_per_group=2）
        // 段1=[0..8]（圈 0-2），段2=[9..11]（跳过共享 seam 节点 8）
        let t: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let s: Vec<[f64; 6]> = (0..12).map(|i| [i as f64; 6]).collect();

        let segments = segment_trajectory(&t, &s, 4, 2);

        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].0.len(), 9); // 节点 0..=8
        assert_eq!(segments[0].0[0], 0.0);
        assert_eq!(segments[0].0[8], 8.0);
        assert_eq!(segments[1].0.len(), 3); // 节点 9..=11（跳过 seam 节点 8）
        assert_eq!(segments[1].0[0], 9.0);
        assert_eq!(segments[1].0[2], 11.0);
    }

    #[test]
    fn test_segment_trajectory_last_partial() {
        // n_rev=5、revs_per_group=3：段1=圈0-3（节点 0..=12），段2=圈3-5（节点 13..=19）
        let t: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let s: Vec<[f64; 6]> = (0..20).map(|i| [i as f64; 6]).collect();

        let segments = segment_trajectory(&t, &s, 4, 3);

        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].0.len(), 13);
        assert_eq!(segments[0].0[12], 12.0);
        assert_eq!(segments[1].0.len(), 7);
        assert_eq!(segments[1].0[0], 13.0);
        assert_eq!(segments[1].0[6], 19.0);
    }
}
