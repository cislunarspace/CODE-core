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

use crate::forces::compiled::CompiledForce;
use crate::multiple_shooting::{multiple_shooting_correct, MultipleShootingRustResult};
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
fn segment_trajectory(
    t_patch: &[f64],
    state_patch: &[[f64; 6]],
    points_per_segment: usize,
    overlap_points: usize,
) -> Vec<(Vec<f64>, Vec<[f64; 6]>)> {
    let n = t_patch.len();
    let mut segments = Vec::new();

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

/// 合并相邻段。
fn merge_segments(
    segments: Vec<(Vec<f64>, Vec<[f64; 6]>)>,
    merge_indices: &[(usize, usize)],
) -> Vec<(Vec<f64>, Vec<[f64; 6]>)> {
    let mut merged = Vec::new();
    let mut skip = vec![false; segments.len()];

    for &(i, j) in merge_indices {
        if i >= segments.len() || j >= segments.len() {
            continue;
        }

        let (ref t_i, ref s_i) = segments[i];
        let (ref t_j, ref s_j) = segments[j];

        let mut t_merged = t_i.clone();
        let mut s_merged = s_i.clone();

        // 找到重叠点（t_j 中第一个等于 t_i 最后一个的点）
        let overlap_start = if !t_j.is_empty() && !t_i.is_empty() {
            let last_t = t_i.last().unwrap();
            t_j.iter().position(|&t| (t - last_t).abs() < 1e-10)
        } else {
            None
        };

        // 添加非重叠部分（跳过重叠点）
        let start = match overlap_start {
            Some(idx) => idx + 1, // 跳过重叠点
            None => 0,
        };
        for k in start..t_j.len() {
            t_merged.push(t_j[k]);
            s_merged.push(s_j[k]);
        }

        merged.push((t_merged, s_merged));
        skip[i] = true;
        skip[j] = true;
    }

    // 添加未合并的段
    for (i, segment) in segments.into_iter().enumerate() {
        if !skip[i] {
            merged.push(segment);
        }
    }

    merged
}

/// 对单段进行多重打靶修正。
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
        max_iter,
        tolerance,
        rtol,
        None,
        verbose,
    )
}

/// 分段打靶拼接法主函数。
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

    // 第2步：逐步合并相邻段
    if config.enable_merging && n_segments > 1 {
        let mut current_segments = segments;
        let mut stage = 1;

        while current_segments.len() > 1 {
            if verbose {
                eprintln!(
                    "合并阶段 {}: {} 段 -> {} 段",
                    stage,
                    current_segments.len(),
                    (current_segments.len() + 1) / 2
                );
            }

            let merge_indices: Vec<(usize, usize)> = (0..current_segments.len() - 1)
                .step_by(2)
                .map(|i| (i, i + 1))
                .collect();

            let merged = merge_segments(current_segments, &merge_indices);

            let mut corrected = Vec::new();
            for (i, (seg_t, seg_s)) in merged.into_iter().enumerate() {
                let result = correct_segment(
                    forces,
                    observer,
                    &seg_t,
                    &seg_s,
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
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_trajectory() {
        let t: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let s: Vec<[f64; 6]> = (0..20).map(|i| [i as f64; 6]).collect();

        let segments = segment_trajectory(&t, &s, 10, 2);

        // 20个点，每段10个，重叠2个：段1=[0..10], 段2=[8..18], 段3=[16..20]
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].0.len(), 10);
        assert_eq!(segments[0].0[0], 0.0);
        assert_eq!(segments[0].0[9], 9.0);
        assert_eq!(segments[1].0.len(), 10);
        assert_eq!(segments[1].0[0], 8.0);
        assert_eq!(segments[1].0[9], 17.0);
        assert_eq!(segments[2].0.len(), 4);
        assert_eq!(segments[2].0[0], 16.0);
        assert_eq!(segments[2].0[3], 19.0);
    }

    #[test]
    fn test_merge_segments() {
        let t1 = vec![0.0, 1.0, 2.0, 3.0];
        let s1 = vec![[0.0; 6], [1.0; 6], [2.0; 6], [3.0; 6]];
        let t2 = vec![3.0, 4.0, 5.0];
        let s2 = vec![[3.0; 6], [4.0; 6], [5.0; 6]];

        let segments = vec![(t1, s1), (t2, s2)];
        let merged = merge_segments(segments, &[(0, 1)]);

        assert_eq!(merged.len(), 1);
        // 合并后：[0,1,2,3] + [4,5] = 6个点（去掉重叠的3）
        assert_eq!(merged[0].0.len(), 6);
        assert_eq!(merged[0].0[0], 0.0);
        assert_eq!(merged[0].0[5], 5.0);
    }
}
