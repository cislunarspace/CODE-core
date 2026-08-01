//! 分段打靶拼接法（Segmented Shooting Splicing）Rust 实现。
//!
//! 参考朱彦伟等人的论文《星历模型下基于多重打靶拼接的长期近直线晕轨道设计方法》，
//! 实现分段打靶拼接策略，解决长转移轨迹在多重打靶中不收敛的问题。
//!
//! ## 核心思想
//!
//! 1. 将长转移拆分为若干小段（每段约10-20天）
//! 2. 每小段独立用多重打靶转换到星历模型
//! 3. 逐步合并相邻段，继续修正
//! 4. 重复直到所有段合并为一条完整轨迹

use crate::multiple_shooting::{multiple_shooting_correct, MultipleShootingRustResult};
use e2m2e_forces::forces::compiled::CompiledForce;
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
    /// 是否收敛
    #[pyo3(get)]
    pub converged: bool,
    /// 总迭代次数
    #[pyo3(get)]
    pub total_iterations: usize,
    /// 最终残差
    #[pyo3(get)]
    pub max_residual: f64,
    /// 每阶段的残差历史
    #[pyo3(get)]
    pub stage_residuals: Vec<f64>,
    /// 分段数
    #[pyo3(get)]
    pub n_segments: usize,
}

/// 分段配置。
pub struct SegmentationConfig {
    /// 每段的 patch points 数量
    pub points_per_segment: usize,
    /// 重叠点数（相邻段共享的点数）
    pub overlap_points: usize,
    /// 是否启用逐步合并
    pub enable_merging: bool,
}

impl Default for SegmentationConfig {
    fn default() -> Self {
        Self {
            points_per_segment: 10,
            overlap_points: 2,
            enable_merging: true,
        }
    }
}

/// 将轨迹分段。
///
/// 相邻段首尾相接（共享 1 个拼接点：段 i 末点 = 段 i+1 首点），这是分层合并
/// 连续性的基础——合并后新段的末点与下一段首点时间戳相同，可无缝拼接。
/// `overlap_points` 参数保留用于兼容，但内部强制为 1：重叠 >1 会让合并后的
/// 段与段之间时间不连续，破坏拼接。
fn segment_trajectory(
    t_patch: &[f64],
    state_patch: &[[f64; 6]],
    points_per_segment: usize,
    _overlap_points: usize,
) -> Vec<(Vec<f64>, Vec<[f64; 6]>)> {
    let n = t_patch.len();
    let mut segments = Vec::new();
    let overlap_points = 1; // 首尾相接，保证合并连续性

    let mut start = 0;
    while start < n {
        let end = (start + points_per_segment).min(n);
        let segment_t = t_patch[start..end].to_vec();
        let segment_s = state_patch[start..end].to_vec();
        segments.push((segment_t, segment_s));

        if end >= n {
            break;
        }
        start = end - overlap_points;
    }

    segments
}

/// 对单段进行多重打靶修正。
///
/// 固定首节点：段首点是拼接/合并时的锚点（来自上一段的末端或初始初猜），
/// 段内修正不应改变它，否则段间连续性在合并时被破坏。
#[allow(clippy::too_many_arguments)]
fn correct_segment(
    forces: &[CompiledForce],
    observer: &str,
    t_patch: &[f64],
    state_patch: &[[f64; 6]],
    max_iter: usize,
    tolerance: f64,
    rtol: f64,
    verbose: bool,
) -> Result<MultipleShootingRustResult, String> {
    multiple_shooting_correct(
        forces,
        observer,
        t_patch,
        state_patch,
        false, // 固定时间
        true,  // 固定首节点：拼接锚点不可变
        max_iter,
        tolerance,
        rtol,
        None,
        verbose,
    )
}

/// 分段打靶拼接法主函数。
#[allow(clippy::too_many_arguments)]
pub fn segmented_shooting_correct(
    forces: &[CompiledForce],
    observer: &str,
    t_patch: &[f64],
    state_patch: &[[f64; 6]],
    config: &SegmentationConfig,
    max_iter_per_segment: usize,
    tolerance: f64,
    rtol: f64,
    verbose: bool,
) -> Result<SegmentedShootingResult, String> {
    if t_patch.len() < 2 {
        return Err("need at least 2 patch points".to_string());
    }

    let mut segments = segment_trajectory(
        t_patch,
        state_patch,
        config.points_per_segment,
        config.overlap_points,
    );

    let n_segments = segments.len();
    let mut total_iterations = 0;
    let mut stage_residuals = Vec::new();
    let mut converged = true;

    if verbose {
        eprintln!(
            "分段打靶拼接：{} 段，每段 {} 点",
            n_segments, config.points_per_segment
        );
    }

    // 第1步：对每小段独立进行多重打靶修正
    for (i, (seg_t, seg_s)) in segments.iter_mut().enumerate() {
        if verbose {
            eprintln!("  修正段 {}/{}: {} 点", i + 1, n_segments, seg_t.len());
        }

        let result = correct_segment(
            forces,
            observer,
            seg_t,
            seg_s,
            max_iter_per_segment,
            tolerance,
            rtol,
            verbose,
        )?;

        total_iterations += result.iterations;
        stage_residuals.push(result.max_residual);

        if !result.converged {
            converged = false;
            if verbose {
                eprintln!(
                    "    段 {} 未收敛，残差 = {:.2e}",
                    i + 1,
                    result.max_residual
                );
            }
        }

        *seg_t = result.t_patch;
        *seg_s = result.state_patch;
    }

    // 第2步：分层两两合并（文献式多重打靶拼接）
    // 每层把当前段序列两两配对（(0,1),(2,3)...），合并出更长的段后各自独立
    // 修正，再进入下一层，直到只剩 1 段。分段首尾相接（overlap=1）保证：
    // 合并后新段的末点时间戳 = 下一段首点时间戳，层间拼接连续。
    // 注：不能单次合并所有段——多重打靶对超长弧段不收敛（残差震荡、积分
    // 器步长塌缩），分层让每层修正的段长翻倍但有界，逐层扩大连续弧长。
    if config.enable_merging && n_segments > 1 {
        let mut current_segments = segments;
        let mut stage = 1;

        while current_segments.len() > 1 {
            let n_merge = current_segments.len() / 2;
            if verbose {
                eprintln!(
                    "合并阶段 {}: {} 段 -> {} 段",
                    stage,
                    current_segments.len(),
                    current_segments.len() - n_merge,
                );
            }

            let mut corrected = Vec::with_capacity(current_segments.len() - n_merge);
            let mut i = 0;
            while i < current_segments.len() {
                let (t1, s1) = current_segments[i].clone();
                if i + 1 < current_segments.len() {
                    // 合并相邻两段：首尾相接（段1末点 = 段2首点），去重拼接
                    let (t2, s2) = &current_segments[i + 1];
                    let mut t_all = t1.clone();
                    let mut s_all = s1.clone();
                    let last_t = *t_all.last().unwrap();
                    let skip = t2
                        .iter()
                        .position(|&t| (t - last_t).abs() < 1e-10)
                        .map(|idx| idx + 1)
                        .unwrap_or(0);
                    t_all.extend_from_slice(&t2[skip..]);
                    s_all.extend_from_slice(&s2[skip..]);

                    let result = correct_segment(
                        forces,
                        observer,
                        &t_all,
                        &s_all,
                        max_iter_per_segment,
                        tolerance,
                        rtol,
                        verbose,
                    )?;
                    total_iterations += result.iterations;
                    stage_residuals.push(result.max_residual);
                    if !result.converged {
                        converged = false;
                    }
                    corrected.push((result.t_patch, result.state_patch));
                } else {
                    // 奇数段：最后一段直接继承（本层不合并）
                    corrected.push((t1, s1));
                }
                i += 2;
            }

            current_segments = corrected;
            stage += 1;
        }

        let (final_t, final_s) = current_segments.into_iter().next().unwrap();
        let final_state: Vec<Vec<f64>> = final_s.iter().map(|s| s.to_vec()).collect();

        Ok(SegmentedShootingResult {
            t_patch: final_t,
            state_patch: final_state,
            converged,
            total_iterations,
            max_residual: *stage_residuals.last().unwrap_or(&f64::INFINITY),
            stage_residuals,
            n_segments,
        })
    } else {
        let mut all_t = Vec::new();
        let mut all_s = Vec::new();

        for (i, (seg_t, seg_s)) in segments.into_iter().enumerate() {
            if i == 0 {
                all_t.extend(seg_t);
                all_s.extend(seg_s);
            } else {
                let skip = config.overlap_points.min(seg_t.len());
                all_t.extend(seg_t[skip..].iter().cloned());
                all_s.extend(seg_s[skip..].iter().cloned());
            }
        }

        let final_state: Vec<Vec<f64>> = all_s.iter().map(|s| s.to_vec()).collect();

        Ok(SegmentedShootingResult {
            t_patch: all_t,
            state_patch: final_state,
            converged,
            total_iterations,
            max_residual: *stage_residuals.last().unwrap_or(&f64::INFINITY),
            stage_residuals,
            n_segments,
        })
    }
}

/// PyO3 接口：分段打靶拼接法。
///
/// `forces` 是 Python 元组列表，每个元组描述一个力模型（格式同 `propagate_compiled`）。
#[pyfunction]
#[pyo3(signature = (forces, observer, t_patch, state_patch, points_per_segment=10, overlap_points=2, enable_merging=true, max_iter_per_segment=50, tolerance=1e-8, rtol=1e-10, verbose=false))]
#[allow(clippy::too_many_arguments)]
pub fn segmented_shooting_correct_py(
    forces: Vec<PyObject>,
    observer: &str,
    t_patch: Vec<f64>,
    state_patch: Vec<Vec<f64>>,
    points_per_segment: usize,
    overlap_points: usize,
    enable_merging: bool,
    max_iter_per_segment: usize,
    tolerance: f64,
    rtol: f64,
    verbose: bool,
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

    // 构造配置
    let config = SegmentationConfig {
        points_per_segment,
        overlap_points,
        enable_merging,
    };

    // 调用 Rust 核心
    segmented_shooting_correct(
        &compiled_forces,
        observer,
        &t_patch,
        &state_array,
        &config,
        max_iter_per_segment,
        tolerance,
        rtol,
        verbose,
    )
    .map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_trajectory() {
        let t: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let s: Vec<[f64; 6]> = (0..20).map(|i| [i as f64; 6]).collect();

        let segments = segment_trajectory(&t, &s, 10, 2);

        // 20个点，每段10个，首尾相接（overlap=1 强制）：
        // 段1=[0..10], 段2=[9..19], 段3=[18..20]
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].0.len(), 10);
        assert_eq!(segments[0].0[0], 0.0);
        assert_eq!(segments[0].0[9], 9.0);
        // 段 i 末点 = 段 i+1 首点（拼接点）
        assert_eq!(segments[1].0.len(), 10);
        assert_eq!(segments[1].0[0], 9.0);
        assert_eq!(segments[1].0[9], 18.0);
        assert_eq!(segments[2].0.len(), 2);
        assert_eq!(segments[2].0[0], 18.0);
        assert_eq!(segments[2].0[1], 19.0);
    }

}
