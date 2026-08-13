//! synodic ↔ J2000 批量坐标转换与批量 ET→UTC。
//!
//! 下沉自 Python ``e2m2e.algorithm.coordinate.synodic_j2000`` 的逐点循环
//! （星历表组装一年约 8766 点，逐点在 Python↔C 边界查 SPICE 并做小矩阵
//! 运算）。数值语义逐位对齐 Python 侧：
//!
//! - 会合轴旋转矩阵 ``R = [e1 | e2 | e3]`` 由月球相对地球状态（J2000）
//!   构造：e1 地月连线、e3 轨道角动量、e2 右手补齐；
//! - 轴速率 ``Rdot = (R(et+1s) - R(et-1s)) / 2``（数值差分，
//!   对齐 ``SynodicAxes._DEFAULT_RATE_STEP = 1.0``）；
//! - 特征长度 ``l_c = |r_moon_earth|``；
//! - 两个坐标系原点同为地球，``transform_state`` 的平移项抵消，只余旋转。
//!
//! 线程模型：CSPICE 全局状态非单线程不安全，本模块与其余 spice_* 入口
//! 一致不释放 GIL（主循环单线程调用）。

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use e2m2e_spice::spice_ffi::{et2utc, mat3_mul_vec, mat3_t_mul_vec, spkezr, SpiceFfiError};

/// 轴速率差分步长（秒），对齐 Python ``SynodicAxes._DEFAULT_RATE_STEP``。
const RATE_STEP: f64 = 1.0;

/// 3×3 行优先矩阵（旋转矩阵/轴速率）。
type Mat3 = [[f64; 3]; 3];

/// 批量 ET→UTC 的输出：年/月/日/时/分/秒六分量数组。
type UtcFields = (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<f64>);

/// Rust cspice 实例的行星名注册（Once 保护，见 lib.rs::ensure_bodies_registered）。
fn spkezr_or_err(target: &str, et: f64, observer: &str) -> PyResult<[f64; 6]> {
    crate::ensure_bodies_registered();
    spkezr(target, et, "J2000", "NONE", observer)
        .map(|(s, _lt)| s)
        .map_err(spice_err)
}

fn spice_err(e: SpiceFfiError) -> PyErr {
    PyRuntimeError::new_err(format!("{e}"))
}

/// 由月球相对地球位置/速度构造会合轴旋转矩阵（列 = e1, e2, e3）。
fn synodic_rotation(r: [f64; 3], v: [f64; 3]) -> Mat3 {
    let rn = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
    let e1 = [r[0] / rn, r[1] / rn, r[2] / rn];
    let h = [
        r[1] * v[2] - r[2] * v[1],
        r[2] * v[0] - r[0] * v[2],
        r[0] * v[1] - r[1] * v[0],
    ];
    let hn = (h[0] * h[0] + h[1] * h[1] + h[2] * h[2]).sqrt();
    let e3 = [h[0] / hn, h[1] / hn, h[2] / hn];
    let e2 = [
        e3[1] * e1[2] - e3[2] * e1[1],
        e3[2] * e1[0] - e3[0] * e1[2],
        e3[0] * e1[1] - e3[1] * e1[0],
    ];
    // 列向量 e1, e2, e3 → 行优先 R[i][j] = e_j[i]
    [
        [e1[0], e2[0], e3[0]],
        [e1[1], e2[1], e3[1]],
        [e1[2], e2[2], e3[2]],
    ]
}

/// 查询时刻 ``et`` 的会合轴旋转矩阵、轴速率与地月距离。
fn rotation_and_rate(et: f64) -> PyResult<(Mat3, Mat3, f64)> {
    let s = spkezr_or_err("MOON", et, "EARTH")?;
    let lc = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
    let r_now = synodic_rotation([s[0], s[1], s[2]], [s[3], s[4], s[5]]);
    let s_before = spkezr_or_err("MOON", et - RATE_STEP, "EARTH")?;
    let s_after = spkezr_or_err("MOON", et + RATE_STEP, "EARTH")?;
    let r_before = synodic_rotation(
        [s_before[0], s_before[1], s_before[2]],
        [s_before[3], s_before[4], s_before[5]],
    );
    let r_after = synodic_rotation(
        [s_after[0], s_after[1], s_after[2]],
        [s_after[3], s_after[4], s_after[5]],
    );
    let mut rate = [[0.0_f64; 3]; 3];
    for (i, row) in rate.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (r_after[i][j] - r_before[i][j]) / (2.0 * RATE_STEP);
        }
    }
    Ok((r_now, rate, lc))
}

/// 校验批量输入形状：状态数组长度必须为时间点数的 6 倍。
fn check_len(name: &str, n_states: usize, n_times: usize) -> PyResult<()> {
    if n_states != 6 * n_times {
        return Err(PyValueError::new_err(format!(
            "{name} 长度 {n_states} 与时间点数 {n_times} 不匹配（应为 6×n）"
        )));
    }
    Ok(())
}

/// 单点 synodic（无量纲）→ J2000（km, km/s）。纯函数，供批量循环与
/// 解析恒等式测试共用（不依赖 SPICE，数值语义同 Python 逐点版）。
fn syn_to_j2000(
    rot: &[[f64; 3]; 3],
    rate: &[[f64; 3]; 3],
    lc: f64,
    mu: f64,
    t_c: f64,
    s: &[f64; 6],
) -> [f64; 6] {
    // position_in = (r_syn + [μ,0,0])·l_c；velocity_in = v_syn·l_c / t_c
    let p = [(s[0] + mu) * lc, s[1] * lc, s[2] * lc];
    let v = [s[3] * lc / t_c, s[4] * lc / t_c, s[5] * lc / t_c];
    let pos = mat3_mul_vec(rot, &p);
    let rp = mat3_mul_vec(rot, &v);
    let rdot_p = mat3_mul_vec(rate, &p);
    [
        pos[0],
        pos[1],
        pos[2],
        rp[0] + rdot_p[0],
        rp[1] + rdot_p[1],
        rp[2] + rdot_p[2],
    ]
}

/// 单点 J2000（km, km/s）→ synodic（无量纲）。纯函数，同 [`syn_to_j2000`]。
fn j2000_to_syn(
    rot: &[[f64; 3]; 3],
    rate: &[[f64; 3]; 3],
    lc: f64,
    mu: f64,
    t_c: f64,
    s: &[f64; 6],
) -> [f64; 6] {
    let p_j = [s[0], s[1], s[2]];
    let v_j = [s[3], s[4], s[5]];
    // 原点同为地球：平移项抵消，position_out = Rᵀ @ p_j
    let p_out = mat3_t_mul_vec(rot, &p_j);
    // velocity_out = Rᵀ @ (v_j - Rdot @ p_out)
    let rdot_p = mat3_mul_vec(rate, &p_out);
    let v_out = mat3_t_mul_vec(
        rot,
        &[v_j[0] - rdot_p[0], v_j[1] - rdot_p[1], v_j[2] - rdot_p[2]],
    );
    // 无量纲化：r_syn = p_out / lc - offset；v_syn = v_out·t_c / l_c
    [
        p_out[0] / lc - mu,
        p_out[1] / lc,
        p_out[2] / lc,
        v_out[0] * t_c / lc,
        v_out[1] * t_c / lc,
        v_out[2] * t_c / lc,
    ]
}

/// 批量 synodic（无量纲）→ J2000（km, km/s）状态转换。
///
/// 语义对齐 ``SynodicJ2000System.synodic_to_j2000``（逐点）：
/// ``et = et0 + t_syn·t_c``，``pos = R @ ((r_syn + [μ,0,0])·l_c)``，
/// ``vel = R @ (v_syn·l_c/t_c) + Rdot @ ((r_syn + [μ,0,0])·l_c)``。
#[pyfunction]
#[pyo3(signature = (states_syn, t_syn, et0, mu, t_c))]
pub fn batch_synodic_to_j2000_py(
    states_syn: Vec<f64>,
    t_syn: Vec<f64>,
    et0: f64,
    mu: f64,
    t_c: f64,
) -> PyResult<Vec<f64>> {
    let n = t_syn.len();
    check_len("states_syn", states_syn.len(), n)?;
    let mut out = vec![0.0_f64; 6 * n];
    for i in 0..n {
        let et = et0 + t_syn[i] * t_c;
        let (rot, rate, lc) = rotation_and_rate(et)?;
        let s: [f64; 6] = states_syn[6 * i..6 * i + 6].try_into().expect("len 6");
        let res = syn_to_j2000(&rot, &rate, lc, mu, t_c, &s);
        out[6 * i..6 * i + 6].copy_from_slice(&res);
    }
    Ok(out)
}

/// 批量 J2000（km, km/s）→ synodic（无量纲）状态转换。
///
/// 语义对齐 ``SynodicJ2000System.j2000_to_synodic``（逐点）：
/// ``pos_syn = Rᵀ @ r_j2000 / l_c - [μ,0,0]``，
/// ``vel_syn = (Rᵀ @ (v_j2000 - Rdot @ (Rᵀ @ r_j2000)))·t_c / l_c``。
#[pyfunction]
#[pyo3(signature = (states_j2000, t_syn, et0, mu, t_c))]
pub fn batch_j2000_to_synodic_py(
    states_j2000: Vec<f64>,
    t_syn: Vec<f64>,
    et0: f64,
    mu: f64,
    t_c: f64,
) -> PyResult<Vec<f64>> {
    let n = t_syn.len();
    check_len("states_j2000", states_j2000.len(), n)?;
    let mut out = vec![0.0_f64; 6 * n];
    for i in 0..n {
        let et = et0 + t_syn[i] * t_c;
        let (rot, rate, lc) = rotation_and_rate(et)?;
        let s: [f64; 6] = states_j2000[6 * i..6 * i + 6].try_into().expect("len 6");
        let res = j2000_to_syn(&rot, &rate, lc, mu, t_c, &s);
        out[6 * i..6 * i + 6].copy_from_slice(&res);
    }
    Ok(out)
}

/// 批量天体状态查询：``target`` 相对 ``observer`` 在 J2000 下的 6 维状态。
///
/// 对齐 ``SPICEManager.get_body_state(target, et, "J2000", observer)`` 的
/// 逐点循环（ELFO 月心根数提取等场景）。返回行优先 ``(n, 6)`` 展平。
#[pyfunction]
#[pyo3(signature = (target, observer, ets))]
pub fn batch_body_states_py(target: &str, observer: &str, ets: Vec<f64>) -> PyResult<Vec<f64>> {
    let mut out = Vec::with_capacity(6 * ets.len());
    for et in ets {
        let s = spkezr_or_err(target, et, observer)?;
        out.extend_from_slice(&s);
    }
    Ok(out)
}

/// 批量 ET → UTC 日历分量（年/月/日/时/分/秒）。
///
/// 对齐 ``SPICEManager.et_to_utc``（"ISOC" 格式、0 位小数秒）的逐点循环，
/// 直接返回六分量数组，免去 Python 侧 ``datetime.fromisoformat`` 字符串解析。
/// 秒为浮点（ISOC prec=0 下为整数秒）。
#[pyfunction]
pub fn batch_et_to_utc_py(et: Vec<f64>) -> PyResult<UtcFields> {
    crate::ensure_bodies_registered();
    let n = et.len();
    let mut year = Vec::with_capacity(n);
    let mut month = Vec::with_capacity(n);
    let mut day = Vec::with_capacity(n);
    let mut hour = Vec::with_capacity(n);
    let mut minute = Vec::with_capacity(n);
    let mut second = Vec::with_capacity(n);
    for t in et {
        let s = et2utc(t, 0).map_err(spice_err)?;
        // "ISOC" prec=0 输出固定 19 字符 "YYYY-MM-DDTHH:MM:SS"；
        // 年份位数异常（非 4 位）时无法按位解析，显式报错。
        let parse = |lo: usize, hi: usize| -> PyResult<i32> {
            s.get(lo..hi)
                .and_then(|x| x.parse::<i32>().ok())
                .ok_or_else(|| PyRuntimeError::new_err(format!("et2utc 输出格式异常: {s:?}")))
        };
        year.push(parse(0, 4)?);
        month.push(parse(5, 7)?);
        day.push(parse(8, 10)?);
        hour.push(parse(11, 13)?);
        minute.push(parse(14, 16)?);
        let sec = s
            .get(17..)
            .and_then(|x| x.parse::<f64>().ok())
            .ok_or_else(|| PyRuntimeError::new_err(format!("et2utc 输出格式异常: {s:?}")))?;
        second.push(sec);
    }
    Ok((year, month, day, hour, minute, second))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 旋转矩阵构造的几何性质：R 正交、列归一。
    #[test]
    fn rotation_is_orthonormal() {
        let r = [1.0_f64, 2.0, 3.0];
        let v = [0.1_f64, -0.2, 0.05];
        let m = synodic_rotation(r, v);
        // 列点积
        for a in 0..3 {
            for b in 0..3 {
                let dot = m[0][a] * m[0][b] + m[1][a] * m[1][b] + m[2][a] * m[2][b];
                let expect = if a == b { 1.0 } else { 0.0 };
                assert!((dot - expect).abs() < 1e-12, "col {a}·{b} = {dot}");
            }
        }
    }

    /// 形状校验：状态长度不是时间点数 6 倍时报错。
    #[test]
    fn len_mismatch_errors() {
        assert!(check_len("x", 5, 1).is_err());
        assert!(check_len("x", 6, 1).is_ok());
    }

    /// ISOC 解析路径（不依赖 SPICE 内核）：格式样例解析。
    #[test]
    fn iso_parse_shape() {
        // 仅验证 Rust 侧解析器接受合法 ISOC 输出；et2utc 本身依赖内核，
        // 集成验证在 Python 侧对拍。
        let s = "2024-01-01T00:00:00";
        assert_eq!(s.get(0..4).and_then(|x| x.parse::<i32>().ok()), Some(2024));
        assert_eq!(s.get(17..).and_then(|x| x.parse::<f64>().ok()), Some(0.0));
    }

    /// 往返恒等式（对称性）：任意 (R, Rdot, l_c) 与状态，syn→J2000→syn 应恢复。
    #[test]
    fn syn_j2000_round_trip_identity() {
        let r = [1.0_f64, 2.0, 3.0];
        let v = [0.1_f64, -0.2, 0.05];
        let rot = synodic_rotation(r, v);
        let rate = [
            [0.001_f64, -0.002, 0.0],
            [0.0005, 0.001, -0.001],
            [0.0, 0.002, 0.0003],
        ];
        let (lc, mu, t_c) = (3.8e5_f64, 0.012, 3.7e5_f64);
        let s = [1.05_f64, 0.03, -0.02, 0.001, -0.3, 0.002];
        let fwd = syn_to_j2000(&rot, &rate, lc, mu, t_c, &s);
        let back = j2000_to_syn(&rot, &rate, lc, mu, t_c, &fwd);
        for k in 0..6 {
            assert!(
                (back[k] - s[k]).abs() < 1e-9,
                "分量 {k}: 往返 {} vs 原值 {}",
                back[k],
                s[k]
            );
        }
    }

    /// 尺度恒等式（定义）：月球位置 (1-μ,0,0) 转 J2000 的位置范数应为地月距离 l_c。
    #[test]
    fn moon_position_scale_identity() {
        let r = [1.0_f64, 2.0, 3.0];
        let v = [0.1_f64, -0.2, 0.05];
        let rot = synodic_rotation(r, v);
        let rate = [[0.0_f64; 3]; 3];
        let (lc, mu, t_c) = (3.8e5_f64, 0.012, 3.7e5_f64);
        let moon_syn = [1.0 - mu, 0.0, 0.0, 0.0, 0.0, 0.0];
        let j2000 = syn_to_j2000(&rot, &rate, lc, mu, t_c, &moon_syn);
        let norm = (j2000[0] * j2000[0] + j2000[1] * j2000[1] + j2000[2] * j2000[2]).sqrt();
        assert!(
            (norm - lc).abs() < 1e-6,
            "月球位置范数 {norm} 应等于特征长度 {lc}"
        );
        // 反向：J2000 月球位置应回到 (1-μ, 0, 0)
        let back = j2000_to_syn(&rot, &rate, lc, mu, t_c, &j2000);
        assert!((back[0] - (1.0 - mu)).abs() < 1e-12);
        assert!(back[1].abs() < 1e-12 && back[2].abs() < 1e-12);
    }
}
