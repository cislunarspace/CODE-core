//! 二体 Lambert（Izzo）与 universal-variable Kepler 封闭解传播的 PyO3 绑定。

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// 二体 Lambert 求解（Izzo 算法）的 Python 接口。
///
/// # 参数
/// - `r0`/`rf` ：出发/到达位置 [x, y, z]（km）
/// - `tof` ：飞行时间（s）
/// - `mu` ：中心天体 GM（km³/s²）
/// - `long_way` ：True 取长程解（转移角 > π）
/// - `revs` ：完整圈数（≥ 1 时返回右分支低能解）
///
/// # 返回
/// Python dict：`{"v0": [3], "vf": [3], "n_iter": int}` ；无解/不收敛抛 ValueError。
#[pyfunction]
#[pyo3(signature = (r0, rf, tof, mu, long_way, revs))]
pub fn lambert_izzo_py(
    r0: Vec<f64>,
    rf: Vec<f64>,
    tof: f64,
    mu: f64,
    long_way: bool,
    revs: u32,
    py: Python<'_>,
) -> PyResult<PyObject> {
    use e2m2e_propagation::lambert::{lambert_izzo, TransferDirection};

    if r0.len() != 3 || rf.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "r0/rf must have length 3, got {} and {}",
            r0.len(),
            rf.len()
        )));
    }
    let direction = if long_way {
        TransferDirection::LongWay
    } else {
        TransferDirection::ShortWay
    };
    let r0_arr = [r0[0], r0[1], r0[2]];
    let rf_arr = [rf[0], rf[1], rf[2]];
    let (v0, vf, n_iter) = lambert_izzo(&r0_arr, &rf_arr, tof, mu, direction, revs)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    let dict = PyDict::new(py);
    dict.set_item("v0", v0.to_vec())?;
    dict.set_item("vf", vf.to_vec())?;
    dict.set_item("n_iter", n_iter)?;
    Ok(dict.into())
}

/// universal-variable 二体 Kepler 封闭解传播（可选解析 STM）。
///
/// 算法为 Vallado 2013 §2-5（Algorithm 8）的 universal Kepler 方程
/// Newton 解 + f & g 闭式（定义性公式，ADR 0055 决策 5 ①）；解析 STM 由
/// f & g 表达式的隐函数定理链式求导得到。
///
/// # 参数
/// - `state0`：初始状态 [x, y, z, vx, vy, vz]（km, km/s）
/// - `t_eval`：采样时刻（s）；各元素为相对 state0 历元的流逝秒，可为负
/// - `mu`：中心天体 GM（km³/s²）
/// - `with_stm`：True 时返回逐点 6×6 行主序解析 STM
///
/// # 返回
/// Python dict：`{"time": [n], "states": [n][6]}`；`with_stm=True` 时额外
/// 含 `"stm": [n][36]`。迭代不收敛或参数非法抛 ValueError。
#[pyfunction]
#[pyo3(signature = (state0, t_eval, mu, *, with_stm = false))]
pub fn propagate_kepler_py(
    state0: Vec<f64>,
    t_eval: Vec<f64>,
    mu: f64,
    with_stm: bool,
    py: Python<'_>,
) -> PyResult<PyObject> {
    if state0.len() != 6 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "state0 必须为 6 维 [x, y, z, vx, vy, vz]，得到 {} 维",
            state0.len()
        )));
    }
    if t_eval.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("t_eval 不能为空"));
    }
    if !mu.is_finite() || mu <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "mu 必须为正的有限值",
        ));
    }
    let r0_sq = state0[0] * state0[0] + state0[1] * state0[1] + state0[2] * state0[2];
    if r0_sq == 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("r0 不能为零向量"));
    }
    let mut s0 = [0.0f64; 6];
    s0.copy_from_slice(&state0);
    let result = e2m2e_propagation::kepler::propagate_kepler(&s0, &t_eval, mu, with_stm)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    let dict = PyDict::new(py);
    dict.set_item("time", result.times)?;
    let states: Vec<Vec<f64>> = result.states.iter().map(|s| s.to_vec()).collect();
    dict.set_item("states", states)?;
    if with_stm {
        let stms: Vec<Vec<f64>> = result.stms.iter().map(|s| s.to_vec()).collect();
        dict.set_item("stm", stms)?;
    }
    Ok(dict.into())
}

/// N×M 网格批量 Lambert 求解（porkchop 用）的 Python 接口。
///
/// # 参数
/// - `geometries` ：几何列表，每项 `[r0x, r0y, r0z, rfx, rfy, rfz]` （km）
/// - `tofs` ：飞行时间列表（s），对每个几何都求解一遍
/// - `mu`/`long_way`/`revs` ：同 `lambert_izzo_py`
///
/// # 返回
/// 长度 `len(geometries) * len(tofs)` 的 list（几何在外，tof 在内），
/// 每项为 dict 或 None（该组合无解）。
#[pyfunction]
#[pyo3(signature = (geometries, tofs, mu, long_way, revs))]
pub fn lambert_batch_py(
    geometries: Vec<Vec<f64>>,
    tofs: Vec<f64>,
    mu: f64,
    long_way: bool,
    revs: u32,
    py: Python<'_>,
) -> PyResult<PyObject> {
    use e2m2e_propagation::lambert::{lambert_batch, TransferDirection};

    let mut geoms: Vec<([f64; 3], [f64; 3])> = Vec::with_capacity(geometries.len());
    for g in &geometries {
        if g.len() != 6 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "each geometry must have length 6 [r0, rf], got {}",
                g.len()
            )));
        }
        geoms.push(([g[0], g[1], g[2]], [g[3], g[4], g[5]]));
    }
    let direction = if long_way {
        TransferDirection::LongWay
    } else {
        TransferDirection::ShortWay
    };
    let results = lambert_batch(&geoms, &tofs, mu, direction, revs);

    let list = PyList::empty(py);
    for res in results {
        match res {
            Ok((v0, vf, n_iter)) => {
                let dict = PyDict::new(py);
                dict.set_item("v0", v0.to_vec())?;
                dict.set_item("vf", vf.to_vec())?;
                dict.set_item("n_iter", n_iter)?;
                list.append(dict)?;
            }
            Err(_) => list.append(py.None())?,
        }
    }
    Ok(list.into())
}
