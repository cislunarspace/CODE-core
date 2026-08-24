//! 伪弧长延拓（PAL）数值内核（纯 Rust，无 pyo3）。
//!
//! 从 Python `e2m2e/algorithm/solver/continuation.py` 下沉的数值迭代部分：
//! XZ 平面对称约束的 F/dF 组装、约束雅可比零空间切向量、
//! PAL 牛顿迭代。轨道族编排（微分修正、方向反馈、停滞检测、物理合理性
//! 检查）留 Python 侧。
//!
//! 与 numpy 参照实现的对应关系：
//!
//! - 零空间用**广义叉积**（3×4 矩阵行向量的 4 维叉积，各 3×3 余子式）
//!   而非 SVD：满秩 3×4 矩阵的零空间一维，两者同向（符号约定各自
//!   任意，由调用方在与上一步同向化时吸收，与 numpy 路径一致）。
//! - 4×4 牛顿步用部分主元高斯消元（同 `multiple_shooting` 的手写小
//!   矩阵求解惯例），主元精确为零判奇异，对应 LAPACK `dgesv` 的
//!   `info > 0`（numpy `LinAlgError`）。
//! - 传播走 `propagate_cr3bp_stm`，与 Python 路径底层是同一个 Rust
//!   积分器，states/STM 逐位一致。

use crate::cr3bp::{cr3bp_eom, propagate_cr3bp_stm};

/// 牛顿步分量裁剪上限（对应 Python `max_step = [0.04, 0.12, 0.12, 0.08]`，
/// 防止 PAL 收敛到 rx 极大的非物理解）。
pub const NEWTON_STEP_CLIP: [f64; 4] = [0.04, 0.12, 0.12, 0.08];

/// `f_df_symmetric_xz_plane` 的返回类型：约束向量 F（3）、雅可比 dF（3×4）、
/// 半周期末端状态（6）。
pub type FdfTriple = ([f64; 3], [[f64; 4]; 3], [f64; 6]);

/// F/dF + 切向量 + 半周期末端状态。
pub struct FdfTangent {
    pub f: [f64; 3],
    pub df: [[f64; 4]; 3],
    pub tangent: [f64; 4],
    pub final_state: [f64; 6],
}

/// PAL 牛顿迭代结果。无论收敛与否都返回当前 `x_new`（对应 Python 循环
/// `break`/耗尽后继续使用最后值的行为）。
pub struct PalNewtonOutcome {
    pub x_new: [f64; 4],
    pub tangent: [f64; 4],
    pub iterations: usize,
    pub residual: f64,
    pub converged: bool,
    pub singular: bool,
}

/// XZ 平面对称轨道的约束向量与雅可比（对应 Python
/// `compute_F_and_dF_symmetric_xz_plane`）。
///
/// 自由变量 `x = [rx, rz, vy, T/2]`；`F = [vx; vz; ry]`（半周期末端状态）；
/// `dF` 前三列取 STM 相应列，第四列取末端状态导数。
pub fn f_df_symmetric_xz_plane(
    mu: f64,
    x: [f64; 4],
    sv0: [f64; 6],
    rtol: f64,
    atol: f64,
    max_step: f64,
) -> Result<FdfTriple, String> {
    let tf2 = x[3];
    let mut state = sv0;
    state[0] = x[0];
    state[2] = x[1];
    state[4] = x[2];

    let result = propagate_cr3bp_stm(
        mu,
        (0.0, tf2),
        &[tf2],
        &state,
        rtol,
        atol,
        Some(max_step),
        None,
    )?;
    let final_state = *result
        .states
        .last()
        .ok_or("STM propagation returned no states")?;
    let stm = *result
        .stms
        .last()
        .ok_or("STM propagation returned no STM")?;
    let dsv = cr3bp_eom(mu, &final_state);

    let f = [final_state[3], final_state[5], final_state[1]];

    // stm 为 6×6 行优先展平：stm[i*6+j] = ∂state[i]/∂state0[j]。
    let mut df = [[0.0_f64; 4]; 3];
    // 输出行顺序 [vx=3, vz=5, ry=1]，输入列顺序 [rx=0, rz=2, vy=4]。
    let rows = [3_usize, 5, 1];
    let cols = [0_usize, 2, 4];
    for (ri, &r) in rows.iter().enumerate() {
        for (ci, &c) in cols.iter().enumerate() {
            df[ri][ci] = stm[r * 6 + c];
        }
    }
    df[0][3] = dsv[3];
    df[1][3] = dsv[5];
    df[2][3] = dsv[1];

    Ok((f, df, final_state))
}

/// 约束雅可比（3×4）的零空间切向量，单位化（对应 Python
/// `compute_tangent_vector`）。
///
/// 用广义叉积：`n_i = (-1)^i · det(删去第 i 列的 3×3 余子阵)`，与各行
/// 正交（行列式含重复行必为零）。秩亏（< 3）时所有余子式为零，返回零
/// 向量，与 Python `if norm > 0` 保护同义（实际流形上 dF 满秩）。
pub fn tangent_null_vector(df: &[[f64; 4]; 3]) -> [f64; 4] {
    let mut n = [0.0_f64; 4];
    for (i, nv) in n.iter_mut().enumerate() {
        // 删去第 i 列后剩余 3 列的下标
        let cols: [usize; 3] = match i {
            0 => [1, 2, 3],
            1 => [0, 2, 3],
            2 => [0, 1, 3],
            _ => [0, 1, 2],
        };
        let minor = det3x3(df, cols);
        *nv = if i % 2 == 0 { minor } else { -minor };
    }
    let norm = (n.iter().map(|v| v * v).sum::<f64>()).sqrt();
    if norm > 0.0 {
        for v in n.iter_mut() {
            *v /= norm;
        }
    }
    n
}

/// 3×3 行列式（`df` 的三行取 `cols` 指定的三列）。
fn det3x3(df: &[[f64; 4]; 3], cols: [usize; 3]) -> f64 {
    let a = [
        [df[0][cols[0]], df[0][cols[1]], df[0][cols[2]]],
        [df[1][cols[0]], df[1][cols[1]], df[1][cols[2]]],
        [df[2][cols[0]], df[2][cols[1]], df[2][cols[2]]],
    ];
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

/// F/dF + 切向量一步算齐（初始切向量与每步收敛后的刷新用）。
pub fn f_df_tangent(
    mu: f64,
    x: [f64; 4],
    sv0: [f64; 6],
    rtol: f64,
    atol: f64,
    max_step: f64,
) -> Result<FdfTangent, String> {
    let (f, df, final_state) = f_df_symmetric_xz_plane(mu, x, sv0, rtol, atol, max_step)?;
    let tangent = tangent_null_vector(&df);
    Ok(FdfTangent {
        f,
        df,
        tangent,
        final_state,
    })
}

/// 部分主元高斯消元解 4×4 线性系统 `a·x = b`；主元精确为零返回 `None`
/// （对应 numpy `np.linalg.solve` 抛 `LinAlgError`）。
fn solve4(a: [[f64; 4]; 4], b: [f64; 4]) -> Option<[f64; 4]> {
    let mut m = a;
    let mut rhs = b;
    for i in 0..4 {
        // 选主元（严格大于保留首个最大者，与 LAPACK idamax 同约定）
        let mut max_row = i;
        let mut max_val = m[i][i].abs();
        for (k, row) in m.iter().enumerate().skip(i + 1) {
            if row[i].abs() > max_val {
                max_val = row[i].abs();
                max_row = k;
            }
        }
        if max_val == 0.0 {
            return None;
        }
        if max_row != i {
            m.swap(i, max_row);
            rhs.swap(i, max_row);
        }
        let pivot = m[i][i];
        for j in (i + 1)..4 {
            let factor = m[j][i] / pivot;
            // split_at_mut 同时借出主元行（不可变）与消元行（可变）
            let (head, tail) = m.split_at_mut(j);
            let row_pivot = &head[i];
            let row_elim = &mut tail[0];
            for k in i..4 {
                row_elim[k] -= factor * row_pivot[k];
            }
            rhs[j] -= factor * rhs[i];
        }
    }
    let mut x = [0.0_f64; 4];
    for i in (0..4).rev() {
        let mut sum = rhs[i];
        for (j, &xv) in x.iter().enumerate().skip(i + 1) {
            sum -= m[i][j] * xv;
        }
        x[i] = sum / m[i][i];
    }
    Some(x)
}

/// PAL 牛顿迭代（对应 Python `pseudo_arclength_continuation` 内层循环）。
///
/// 从预测点 `x_start` 出发，迭代解 `G = [F; (Xnew - x_ref)·tangent_ref - ds] = 0`。
/// 每迭代：在当前 `x_new` 处算 F/dF 与切向量（与 `tangent_ref` 同向化），
/// **先判收敛**（`‖F‖ < tol` 即停，与 numpy 参照次序一致），再解 4×4
/// 牛顿步、按 [`NEWTON_STEP_CLIP`] 裁剪后更新。
#[allow(clippy::too_many_arguments)]
pub fn pal_newton_step(
    mu: f64,
    x_start: [f64; 4],
    x_ref: [f64; 4],
    sv0: [f64; 6],
    tangent_ref: [f64; 4],
    ds: f64,
    tol: f64,
    iter_max: usize,
    rtol: f64,
    atol: f64,
    max_step: f64,
) -> Result<PalNewtonOutcome, String> {
    let mut x_new = x_start;
    let mut tangent = tangent_ref;
    let mut residual = f64::NAN;
    let mut converged = false;
    let mut singular = false;
    let mut iterations = 0_usize;

    for iter in 0..iter_max {
        iterations = iter + 1;
        let (f, df, _) = f_df_symmetric_xz_plane(mu, x_new, sv0, rtol, atol, max_step)?;
        let mut tangent_new = tangent_null_vector(&df);
        // 切向量同向化（对零空间符号任意性），确保牛顿步稳定
        if dot4(&tangent_new, &tangent_ref) < 0.0 {
            for v in tangent_new.iter_mut() {
                *v = -*v;
            }
        }
        tangent = tangent_new;

        residual = norm3(&f);

        let g = [
            f[0],
            f[1],
            f[2],
            dot4(&sub4(&x_new, &x_ref), &tangent_ref) - ds,
        ];
        let mut dg = [[0.0_f64; 4]; 4];
        dg[..3].copy_from_slice(&df);
        dg[3] = tangent_ref;

        if residual < tol {
            converged = true;
            break;
        }

        let Some(mut delta) = solve4(dg, g) else {
            singular = true;
            break;
        };
        for (d, &clip) in delta.iter_mut().zip(NEWTON_STEP_CLIP.iter()) {
            *d = d.clamp(-clip, clip);
        }
        for (x, d) in x_new.iter_mut().zip(delta.iter()) {
            *x -= *d;
        }
    }

    Ok(PalNewtonOutcome {
        x_new,
        tangent,
        iterations,
        residual,
        converged,
        singular,
    })
}

fn dot4(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn sub4(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}

fn norm3(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tangent_is_orthogonal_to_rows() {
        // 任意满秩 3×4：零空间向量与三行正交且单位化
        let df = [
            [1.0, 2.0, 3.0, 4.0],
            [0.5, -1.0, 2.0, 0.25],
            [3.0, 0.0, -1.0, 1.5],
        ];
        let n = tangent_null_vector(&df);
        for row in df.iter() {
            let dot: f64 = row.iter().zip(n.iter()).map(|(a, b)| a * b).sum();
            assert!(dot.abs() < 1e-14, "row·n = {dot}");
        }
        let norm = n.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-14, "norm = {norm}");
    }

    #[test]
    fn tangent_rank_deficient_returns_zero() {
        // 秩 2（第三行是第一行的倍数）：所有 3×3 余子式为零，返回零向量
        let df = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
        ];
        let n = tangent_null_vector(&df);
        assert!(n.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn solve4_matches_cramer() {
        let a = [
            [2.0, 1.0, -1.0, 0.5],
            [-3.0, -1.0, 2.0, 0.0],
            [-2.0, 1.0, 2.0, -1.0],
            [1.0, 0.0, 1.0, 3.0],
        ];
        let b = [8.0, -11.0, -3.0, 7.0];
        let x = solve4(a, b).expect("well-conditioned");
        for (i, row) in a.iter().enumerate() {
            let lhs: f64 = row.iter().zip(x.iter()).map(|(a, x)| a * x).sum();
            assert!((lhs - b[i]).abs() < 1e-12, "row {i}: {lhs} != {}", b[i]);
        }
    }

    #[test]
    fn solve4_singular_returns_none() {
        let a = [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
        ];
        assert!(solve4(a, [1.0, 2.0, 3.0, 4.0]).is_none());
    }
}
