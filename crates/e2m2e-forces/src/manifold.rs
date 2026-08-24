//! 不变流形种子生成与批量传播的纯 Rust 数值核。
//!
//! 对应 Python `InvariantManifold` 的数值部分：
//! - 沿周期轨道相位扫掠（STM 传播）
//! - 单值矩阵双曲实特征向量选取
//! - STM 转运 + 位置归一化 + ±ε 扰动得种子
//! - 多种子批量传播（串行 / Rayon），单弧失败跳过
//!
//! 截面事后截断仍在 Python（`sections.py` 有意留 Python）。

use nalgebra::{Matrix6, Vector6};

use crate::cr3bp::{propagate_cr3bp, propagate_cr3bp_stm};
use crate::PropagateError;

/// 与 Python `InvariantManifold._REAL_TOL` / `_UNIT_MARGIN` 对齐。
const REAL_TOL: f64 = 1e-8;
const UNIT_MARGIN: f64 = 1e-6;
/// 幂法/逆幂法迭代上限与收敛阈值。
const POWER_MAX_ITER: usize = 200;
const POWER_TOL: f64 = 1e-12;

/// 流形类型：稳定（反向积分）/ 不稳定（正向积分）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ManifoldKind {
    Stable,
    Unstable,
}

impl ManifoldKind {
    pub fn parse(kind: &str) -> Result<Self, String> {
        match kind {
            "stable" => Ok(Self::Stable),
            "unstable" => Ok(Self::Unstable),
            other => Err(format!(
                "kind 必须为 'stable' 或 'unstable'，当前为 {other:?}"
            )),
        }
    }
}

/// 种子生成结果：相位态、相位 STM（可选缓存）与种子数组。
#[derive(Clone, Debug)]
pub struct ManifoldSeeds {
    /// 形状语义 `(n_points, 6)`，行优先展平。
    pub seeds: Vec<[f64; 6]>,
    /// 相位点状态（末端与首点重合已剔除），长度 `n_points`。
    pub phase_states: Vec<[f64; 6]>,
    /// 首点处流形特征向量（未转运）。
    pub eigvec0: [f64; 6],
}

/// 单条流形弧（密采样）。
#[derive(Clone, Debug)]
pub struct ManifoldArc {
    pub times: Vec<f64>,
    pub states: Vec<[f64; 6]>,
}

/// 批量传播结果：仅含成功弧，顺序与输入种子一致（失败跳过）。
#[derive(Clone, Debug)]
pub struct ManifoldTubeResult {
    pub arcs: Vec<ManifoldArc>,
    /// 与 `arcs` 等长：对应输入种子下标。
    pub seed_indices: Vec<usize>,
    pub n_failures: usize,
}

fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    assert!(n >= 2, "linspace 至少 2 点");
    let mut out = Vec::with_capacity(n);
    let denom = (n - 1) as f64;
    for i in 0..n {
        let w = i as f64 / denom;
        out.push(start + (end - start) * w);
    }
    out
}

fn stm_from_flat(flat: &[f64; 36]) -> Matrix6<f64> {
    // 行优先：flat[i*6+j] = M[i,j]，与 CR3BP STM 输出一致。
    Matrix6::from_row_slice(flat)
}

fn mat_vec(m: &Matrix6<f64>, v: &[f64; 6]) -> [f64; 6] {
    let vv = Vector6::from_column_slice(v);
    let r = m * vv;
    let mut out = [0.0_f64; 6];
    for i in 0..6 {
        out[i] = r[i];
    }
    out
}

/// 从单值矩阵提取流形方向对应的实特征向量（轨道首点处）。
///
/// CR3BP 周期轨道的单值矩阵通常有一对实双曲乘子（|λ|>1 与 1/|λ|，互为
/// 倒数）和单位圆上的中心乘子。病态条件下数 |λ| 可达 10³ 以上，全谱
/// Schur/`eig` 易失败；幂法直接锁定模最大/最小的实方向，与 Python 原
/// 实现（稳定取 |λ|<1 中最小、不稳定取 |λ|>1 中最大）在单双曲对情形
/// 下等价（CR3BP 周期轨道的标准结构）。
pub fn select_eigenvector(monodromy: &[f64; 36], kind: ManifoldKind) -> Result<[f64; 6], String> {
    let m = stm_from_flat(monodromy);
    match kind {
        ManifoldKind::Unstable => {
            let (lambda, vec) = power_iteration(&m)?;
            if lambda.abs() <= 1.0 + UNIT_MARGIN {
                return Err("单值矩阵无 |λ|>1 的实特征值，不稳定流形不存在".to_string());
            }
            // 虚部检测：残差 ||Mv - λv|| 应小；幂法收敛到复模主导时残差大
            if !is_real_eigenpair(&m, lambda, &vec) {
                return Err("单值矩阵主导特征非实，不稳定流形不存在".to_string());
            }
            Ok(vec)
        }
        ManifoldKind::Stable => {
            // 对 M⁻¹ 做幂法 → 得到 1/λ_min 方向，即最小模特征值方向
            let m_inv = m
                .try_inverse()
                .ok_or_else(|| "单值矩阵奇异，无法提取稳定流形特征向量".to_string())?;
            let (inv_lambda, vec) = power_iteration(&m_inv)?;
            if inv_lambda.abs() <= REAL_TOL {
                return Err("单值矩阵逆幂法得到零特征值，稳定流形不存在".to_string());
            }
            let lambda = 1.0 / inv_lambda;
            if lambda.abs() >= 1.0 - UNIT_MARGIN {
                return Err("单值矩阵无 |λ|<1 的实特征值，稳定流形不存在".to_string());
            }
            if !is_real_eigenpair(&m, lambda, &vec) {
                return Err("单值矩阵最小模特征非实，稳定流形不存在".to_string());
            }
            Ok(vec)
        }
    }
}

/// 幂法：返回 (Rayleigh 商 λ, 单位特征向量)。
fn power_iteration(m: &Matrix6<f64>) -> Result<(f64, [f64; 6]), String> {
    let mut v = Vector6::from_column_slice(&[1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]);
    let n0 = v.norm();
    if n0 == 0.0 {
        return Err("幂法初值范数为零".to_string());
    }
    v /= n0;

    let mut lambda = 0.0_f64;
    for _ in 0..POWER_MAX_ITER {
        let w = m * v;
        let n = w.norm();
        if n <= 0.0 {
            return Err("幂法迭代中向量范数塌缩".to_string());
        }
        let v_next = w / n;
        // Rayleigh 商
        lambda = v_next.dot(&(m * v_next));
        let delta = (v_next - v).norm().min((v_next + v).norm()); // 允许符号翻转
        v = v_next;
        if delta < POWER_TOL {
            break;
        }
    }

    let mut out = [0.0_f64; 6];
    for i in 0..6 {
        out[i] = v[i];
    }
    Ok((lambda, out))
}

fn is_real_eigenpair(m: &Matrix6<f64>, lambda: f64, vec: &[f64; 6]) -> bool {
    let v = Vector6::from_column_slice(vec);
    let residual = (m * v - lambda * v).norm();
    residual <= 1e-6 * (1.0 + lambda.abs())
}

/// 生成流形种子。
#[allow(clippy::too_many_arguments)]
pub fn generate_manifold_seeds(
    mu: f64,
    initial_state: &[f64; 6],
    period: f64,
    kind: ManifoldKind,
    branch_sign: f64,
    epsilon: f64,
    n_points: usize,
    rtol: f64,
    atol: f64,
    max_step: Option<f64>,
) -> Result<ManifoldSeeds, String> {
    if n_points < 1 {
        return Err(format!("n_points 必须大于等于 1，当前为 {n_points}"));
    }
    if period <= 0.0 {
        return Err(format!("period 必须为正，当前为 {period}"));
    }
    if epsilon <= 0.0 {
        return Err(format!("epsilon 必须为正数，当前为 {epsilon}"));
    }
    if branch_sign != 1.0 && branch_sign != -1.0 {
        return Err(format!("branch_sign 必须为 ±1，当前为 {branch_sign}"));
    }

    // 沿周期均匀 n_points+1 点（含末端 = 首点），剔除末端。
    let t_eval = linspace(0.0, period, n_points + 1);
    let stm_result = propagate_cr3bp_stm(
        mu,
        (0.0, period),
        &t_eval,
        initial_state,
        rtol,
        atol,
        max_step,
        None,
    )
    .map_err(|e| format!("流形相位扫掠 STM 传播失败: {e}"))?;

    if stm_result.states.len() != n_points + 1 || stm_result.stms.len() != n_points + 1 {
        return Err(format!(
            "相位扫掠输出长度异常: states={}, stms={}, 期望 {}",
            stm_result.states.len(),
            stm_result.stms.len(),
            n_points + 1
        ));
    }

    let monodromy = stm_result.stms[n_points];
    let eigvec0 = select_eigenvector(&monodromy, kind)?;

    let mut phase_states = Vec::with_capacity(n_points);
    let mut seeds = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let state = stm_result.states[i];
        let stm = stm_from_flat(&stm_result.stms[i]);
        let mut v = mat_vec(&stm, &eigvec0);
        let pos_norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if pos_norm <= 0.0 {
            return Err(format!("相位 {i} 处特征向量位置部分为零，无法归一化"));
        }
        for vk in &mut v {
            *vk /= pos_norm;
        }
        let mut seed = [0.0_f64; 6];
        for (sk, (st, vk)) in seed.iter_mut().zip(state.iter().zip(v.iter())) {
            *sk = st + branch_sign * epsilon * vk;
        }
        phase_states.push(state);
        seeds.push(seed);
    }

    Ok(ManifoldSeeds {
        seeds,
        phase_states,
        eigvec0,
    })
}

fn propagate_one_seed(
    mu: f64,
    seed: &[f64; 6],
    t_final: f64,
    t_eval: &[f64],
    rtol: f64,
    atol: f64,
    max_step: Option<f64>,
) -> Result<ManifoldArc, PropagateError> {
    let result = propagate_cr3bp(mu, (0.0, t_final), t_eval, seed, rtol, atol, max_step, None)?;
    if result.states.is_empty() {
        return Err(PropagateError::Other("流形弧积分返回空轨迹".to_string()));
    }
    Ok(ManifoldArc {
        times: result.times,
        states: result.states,
    })
}

/// 批量传播流形弧。`t_final` 带符号（稳定负、不稳定正）。
/// 单弧失败跳过，不中止整管。
#[allow(clippy::too_many_arguments)]
pub fn propagate_manifold_arcs_serial(
    mu: f64,
    seeds: &[[f64; 6]],
    t_final: f64,
    sample_dt: f64,
    rtol: f64,
    atol: f64,
    max_step: Option<f64>,
) -> ManifoldTubeResult {
    let duration = t_final.abs();
    let n_samples = ((duration / sample_dt).ceil() as usize)
        .saturating_add(1)
        .max(2);
    let t_eval = linspace(0.0, t_final, n_samples);

    let mut arcs = Vec::new();
    let mut seed_indices = Vec::new();
    let mut n_failures = 0usize;
    for (i, seed) in seeds.iter().enumerate() {
        match propagate_one_seed(mu, seed, t_final, &t_eval, rtol, atol, max_step) {
            Ok(arc) => {
                arcs.push(arc);
                seed_indices.push(i);
            }
            Err(_) => {
                n_failures += 1;
            }
        }
    }
    ManifoldTubeResult {
        arcs,
        seed_indices,
        n_failures,
    }
}

/// Rayon 并行批量传播；弧序与输入种子一致（失败跳过）。
#[allow(clippy::too_many_arguments)]
pub fn propagate_manifold_arcs_parallel(
    mu: f64,
    seeds: &[[f64; 6]],
    t_final: f64,
    sample_dt: f64,
    rtol: f64,
    atol: f64,
    max_step: Option<f64>,
) -> ManifoldTubeResult {
    use rayon::prelude::*;

    let duration = t_final.abs();
    let n_samples = ((duration / sample_dt).ceil() as usize)
        .saturating_add(1)
        .max(2);
    let t_eval = linspace(0.0, t_final, n_samples);

    let results: Vec<Option<ManifoldArc>> = seeds
        .par_iter()
        .map(|seed| propagate_one_seed(mu, seed, t_final, &t_eval, rtol, atol, max_step).ok())
        .collect();

    let mut arcs = Vec::new();
    let mut seed_indices = Vec::new();
    let mut n_failures = 0usize;
    for (i, item) in results.into_iter().enumerate() {
        match item {
            Some(arc) => {
                arcs.push(arc);
                seed_indices.push(i);
            }
            None => n_failures += 1,
        }
    }
    ManifoldTubeResult {
        arcs,
        seed_indices,
        n_failures,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linspace_endpoints() {
        let t = linspace(0.0, 1.0, 5);
        assert_eq!(t.len(), 5);
        assert!((t[0] - 0.0).abs() < 1e-15);
        assert!((t[4] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn select_identity_has_no_hyperbolic() {
        let mut eye = [0.0_f64; 36];
        for i in 0..6 {
            eye[i * 6 + i] = 1.0;
        }
        assert!(select_eigenvector(&eye, ManifoldKind::Stable).is_err());
        assert!(select_eigenvector(&eye, ManifoldKind::Unstable).is_err());
    }

    #[test]
    fn select_diagonal_hyperbolic() {
        // diag(2, 0.5, 1, 1, 1, 1) → 不稳定取 λ=2，稳定取 λ=0.5
        let mut m = [0.0_f64; 36];
        let diags = [2.0, 0.5, 1.0, 1.0, 1.0, 1.0];
        for i in 0..6 {
            m[i * 6 + i] = diags[i];
        }
        let v_u = select_eigenvector(&m, ManifoldKind::Unstable).unwrap();
        let v_s = select_eigenvector(&m, ManifoldKind::Stable).unwrap();
        // 特征向量应接近 e0 / e1
        assert!(v_u[0].abs() > 0.9, "unstable vec ≈ e0, got {v_u:?}");
        assert!(v_s[1].abs() > 0.9, "stable vec ≈ e1, got {v_s:?}");
    }
}
