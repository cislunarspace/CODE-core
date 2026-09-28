//! 不变流形种子生成与批量弧传播的 PyO3 绑定。

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Python 接口：不变流形种子生成。
///
/// 相位扫掠（STM）、单值矩阵双曲实特征选取、转运归一化与 ±ε 扰动均在 Rust
/// 完成；全程 ``py.allow_threads`` ，无 Python 数值回退。
#[pyfunction]
#[pyo3(signature = (mu, initial_state, period, kind, branch_sign, epsilon, n_points, rtol, atol, max_step=None))]
#[allow(clippy::too_many_arguments)]
pub fn manifold_seeds_py(
    mu: f64,
    initial_state: Vec<f64>,
    period: f64,
    kind: &str,
    branch_sign: f64,
    epsilon: f64,
    n_points: usize,
    rtol: f64,
    atol: f64,
    max_step: Option<f64>,
    py: Python<'_>,
) -> PyResult<PyObject> {
    if initial_state.len() != 6 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "initial_state 须含 6 个分量，得到 {}",
            initial_state.len()
        )));
    }
    let kind = e2m2e_dyn::manifold::ManifoldKind::parse(kind)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let mut state0 = [0.0_f64; 6];
    state0.copy_from_slice(&initial_state);

    let result = py
        .allow_threads(|| {
            e2m2e_dyn::manifold::generate_manifold_seeds(
                mu,
                &state0,
                period,
                kind,
                branch_sign,
                epsilon,
                n_points,
                rtol,
                atol,
                max_step,
            )
        })
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    let seeds: Vec<Vec<f64>> = result.seeds.iter().map(|s| s.to_vec()).collect();
    let phase_states: Vec<Vec<f64>> = result.phase_states.iter().map(|s| s.to_vec()).collect();
    let dict = PyDict::new(py);
    dict.set_item("seeds", seeds)?;
    dict.set_item("phase_states", phase_states)?;
    dict.set_item("eigvec0", result.eigvec0.to_vec())?;
    Ok(dict.into())
}

/// Python 接口：不变流形批量弧传播。
///
/// 输入展平种子 ``n*6`` ；`kind` 决定积分方向；单弧失败跳过。`n_workers>1`
/// 或环境变量 ``E2M2E_MANIFOLD_PARALLEL!=0`` 时走 Rayon。
#[pyfunction]
#[pyo3(signature = (mu, seeds, kind, t_span, sample_dt, rtol, atol, max_step=None, *, n_workers=None, parallel=None))]
#[allow(clippy::too_many_arguments)]
pub fn manifold_propagate_py(
    mu: f64,
    seeds: Vec<f64>,
    kind: &str,
    t_span: f64,
    sample_dt: f64,
    rtol: f64,
    atol: f64,
    max_step: Option<f64>,
    n_workers: Option<usize>,
    parallel: Option<bool>,
    py: Python<'_>,
) -> PyResult<PyObject> {
    if !seeds.len().is_multiple_of(6) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "seeds 展平长度必须是 6 的倍数",
        ));
    }
    if sample_dt <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "sample_dt 必须为正，当前为 {sample_dt}"
        )));
    }
    let duration = t_span.abs();
    if duration <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "t_span 必须非零，当前为 {t_span}"
        )));
    }
    let kind = e2m2e_dyn::manifold::ManifoldKind::parse(kind)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let t_final = match kind {
        e2m2e_dyn::manifold::ManifoldKind::Stable => -duration,
        e2m2e_dyn::manifold::ManifoldKind::Unstable => duration,
    };

    let n = seeds.len() / 6;
    let mut seed_arr = Vec::with_capacity(n);
    for i in 0..n {
        let mut s = [0.0_f64; 6];
        s.copy_from_slice(&seeds[i * 6..i * 6 + 6]);
        seed_arr.push(s);
    }

    let use_parallel = parallel.unwrap_or_else(|| {
        if n_workers.map(|n| n > 1).unwrap_or(false) {
            true
        } else {
            std::env::var("E2M2E_MANIFOLD_PARALLEL").is_ok_and(|v| v != "0")
        }
    });
    let pool = match n_workers {
        Some(n) if use_parallel => Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(n.max(1))
                .build()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
        ),
        _ => None,
    };

    let result = py.allow_threads(move || {
        let work = || {
            if use_parallel {
                e2m2e_dyn::manifold::propagate_manifold_arcs_parallel(
                    mu, &seed_arr, t_final, sample_dt, rtol, atol, max_step,
                )
            } else {
                e2m2e_dyn::manifold::propagate_manifold_arcs_serial(
                    mu, &seed_arr, t_final, sample_dt, rtol, atol, max_step,
                )
            }
        };
        if let Some(pool) = pool.as_ref() {
            pool.install(work)
        } else {
            work()
        }
    });

    let arcs_list = PyList::empty(py);
    for arc in &result.arcs {
        let states: Vec<Vec<f64>> = arc.states.iter().map(|s| s.to_vec()).collect();
        let d = PyDict::new(py);
        d.set_item("times", &arc.times)?;
        d.set_item("states", states)?;
        arcs_list.append(d)?;
    }
    let dict = PyDict::new(py);
    dict.set_item("arcs", arcs_list)?;
    dict.set_item("seed_indices", result.seed_indices)?;
    dict.set_item("n_failures", result.n_failures)?;
    Ok(dict.into())
}
