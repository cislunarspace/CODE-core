//! 伪弧长延拓（PAL）F/dF/切向量与牛顿迭代的 PyO3 绑定。

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::propagate_error_to_pyerr;

/// PAL 延拓：XZ 平面对称约束的 F/dF/切向量单次计算。
///
/// 对应 Python `continuation.compute_F_and_dF_symmetric_xz_plane` +
/// `compute_tangent_vector` （纯数值，非 SPICE 门控）。供延拓收敛轨道后的
/// 切向量刷新；初始切向量两后端统一走 Python 参照计算（零空间符号约定
/// 在 SVD 与广义叉积间无保证，首步方向由 Python 侧锁定）。
///
/// # 返回
/// Python dict：`{"f": [3], "df": [[4], [4], [4]], "tangent": [4],
/// "final_state": [6]}`。
#[pyfunction]
#[pyo3(signature = (mu, x, sv0, rtol, atol, max_step))]
pub fn pal_f_df_tangent_py(
    mu: f64,
    x: Vec<f64>,
    sv0: Vec<f64>,
    rtol: f64,
    atol: f64,
    max_step: f64,
    py: Python<'_>,
) -> PyResult<PyObject> {
    use e2m2e_dyn::pal_continuation::f_df_tangent;

    if x.len() != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "x must have length 4, got {}",
            x.len()
        )));
    }
    if sv0.len() != 6 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "sv0 must have length 6, got {}",
            sv0.len()
        )));
    }
    let mut x_arr = [0.0_f64; 4];
    x_arr.copy_from_slice(&x);
    let mut sv0_arr = [0.0_f64; 6];
    sv0_arr.copy_from_slice(&sv0);

    // 内核含 STM 传播（纯 Rust），释 GIL 同 propagate_cr3bp_stm_py。
    let r = py
        .allow_threads(|| f_df_tangent(mu, x_arr, sv0_arr, rtol, atol, max_step))
        .map_err(|e| propagate_error_to_pyerr(py, "PAL F/dF computation failed", e))?;

    let dict = PyDict::new(py);
    dict.set_item("f", r.f.to_vec())?;
    let df_rows: Vec<Vec<f64>> = r.df.iter().map(|row| row.to_vec()).collect();
    dict.set_item("df", df_rows)?;
    dict.set_item("tangent", r.tangent.to_vec())?;
    dict.set_item("final_state", r.final_state.to_vec())?;
    Ok(dict.into())
}

/// PAL 延拓：单步牛顿迭代。
///
/// 对应 Python `pseudo_arclength_continuation` 的内层牛顿循环：从预测点
/// `x_start` 出发解 `G = [F; (Xnew - x_ref)·tangent_ref - ds] = 0` ，先判
/// 收敛再更新，牛顿步按 [0.04, 0.12, 0.12, 0.08] 分量裁剪。无论收敛与否
/// 都返回当前 `x_new` （对应 Python 循环 break/耗尽后继续用最后值）。
///
/// # 返回
/// Python dict：`{"x_new": [4], "tangent": [4], "iterations": int,
/// "residual": float, "converged": bool, "singular": bool}`。
#[pyfunction]
#[pyo3(signature = (mu, x_start, x_ref, sv0, tangent_ref, ds, tol, iter_max, rtol, atol, max_step))]
#[allow(clippy::too_many_arguments)]
pub fn pal_newton_step_py(
    mu: f64,
    x_start: Vec<f64>,
    x_ref: Vec<f64>,
    sv0: Vec<f64>,
    tangent_ref: Vec<f64>,
    ds: f64,
    tol: f64,
    iter_max: usize,
    rtol: f64,
    atol: f64,
    max_step: f64,
    py: Python<'_>,
) -> PyResult<PyObject> {
    use e2m2e_dyn::pal_continuation::pal_newton_step;

    for (name, v, n) in [
        ("x_start", &x_start, 4),
        ("x_ref", &x_ref, 4),
        ("tangent_ref", &tangent_ref, 4),
        ("sv0", &sv0, 6),
    ] {
        if v.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{name} must have length {n}, got {}",
                v.len()
            )));
        }
    }
    let mut x_start_arr = [0.0_f64; 4];
    x_start_arr.copy_from_slice(&x_start);
    let mut x_ref_arr = [0.0_f64; 4];
    x_ref_arr.copy_from_slice(&x_ref);
    let mut sv0_arr = [0.0_f64; 6];
    sv0_arr.copy_from_slice(&sv0);
    let mut tangent_arr = [0.0_f64; 4];
    tangent_arr.copy_from_slice(&tangent_ref);

    let r = py
        .allow_threads(|| {
            pal_newton_step(
                mu,
                x_start_arr,
                x_ref_arr,
                sv0_arr,
                tangent_arr,
                ds,
                tol,
                iter_max,
                rtol,
                atol,
                max_step,
            )
        })
        .map_err(|e| propagate_error_to_pyerr(py, "PAL Newton step failed", e))?;

    let dict = PyDict::new(py);
    dict.set_item("x_new", r.x_new.to_vec())?;
    dict.set_item("tangent", r.tangent.to_vec())?;
    dict.set_item("iterations", r.iterations)?;
    dict.set_item("residual", r.residual)?;
    dict.set_item("converged", r.converged)?;
    dict.set_item("singular", r.singular)?;
    Ok(dict.into())
}
