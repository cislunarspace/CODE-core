//! conic 档二体 Kepler 传播：universal-variable（UV）封闭解 + f & g 解析 STM。
//!
//! 算法依据 Vallado 2013 §2-5（Algorithm 8：universal Kepler 方程 Newton
//! 迭代 + Stumpff 函数 + f & g 闭式），属定义性公式（ADR 0055 决策 5 ①）。
//! 解析 STM 由 f & g 表达式对初始状态做隐函数定理链式求导得到（χ 由
//! universal Kepler 方程 F(χ; y0) = 0 隐式定义，∂χ/∂y0 = −(∂F/∂y0)/(∂F/∂χ)），
//! 全程无数值微分；正确性由辛性（det Φ = 1）与有限差分对照测试钉住。
//!
//! 单位：km、km/s、s；`mu` 单位 km³/s²。无 SPICE、无 pyo3 依赖；
//! Python 经 `e2m2e-integrators` 的 `propagate_kepler_py` 调用。

/// Stumpff 函数级数展开阈值：|z| ≤ 阈值时用幂级数，避免闭式 0/0。
const STUMPFF_EPS: f64 = 1e-8;
/// Newton 收敛容差：|F| < REL·√μ·|dt|（时间量纲下的相对判据）。
const NEWTON_TOL_F_REL: f64 = 1e-12;
/// Newton 收敛容差：|Δχ|。
const NEWTON_TOL_DX: f64 = 1e-12;
/// Newton 迭代上限。
const MAX_NEWTON_ITERS: u32 = 200;

/// 单点传播的返回：末态 (km, km/s) 与可选的 6×6 行主序 STM。
pub struct KeplerPropResult {
    /// 与 `t_eval` 原样一致（相对 state0 历元的流逝秒）。
    pub times: Vec<f64>,
    /// 逐 `t_eval` 点的 [x, y, z, vx, vy, vz]。
    pub states: Vec<[f64; 6]>,
    /// 6×6 行主序展平的 Φ = ∂y/∂y0；仅 `with_stm = true` 时填充，否则为空。
    pub stms: Vec<[f64; 36]>,
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(v: &[f64; 3]) -> f64 {
    dot(v, v).sqrt()
}

fn scale(v: [f64; 6], s: f64) -> [f64; 6] {
    [v[0] * s, v[1] * s, v[2] * s, v[3] * s, v[4] * s, v[5] * s]
}

fn add(a: [f64; 6], b: [f64; 6]) -> [f64; 6] {
    [
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
        a[3] + b[3],
        a[4] + b[4],
        a[5] + b[5],
    ]
}

/// Stumpff 函数 c2(z)、c3(z)（Vallado 2013 §2-5 定义性公式）。
///
/// z > 阈值用三角闭式，z < −阈值用双曲闭式，|z| ≤ 阈值用幂级数
/// c2 = 1/2 − z/24 + z²/720，c3 = 1/6 − z/120 + z²/5040。
fn stumpff(z: f64) -> (f64, f64) {
    if z > STUMPFF_EPS {
        let s = z.sqrt();
        ((1.0 - s.cos()) / z, (s - s.sin()) / (s * s * s))
    } else if z < -STUMPFF_EPS {
        let s = (-z).sqrt();
        ((1.0 - s.cosh()) / z, (s.sinh() - s) / (s * s * s))
    } else {
        (
            0.5 - z / 24.0 + z * z / 720.0,
            1.0 / 6.0 - z / 120.0 + z * z / 5040.0,
        )
    }
}

/// c2、c3 对 z 的一阶导数。
///
/// |z| > 阈值用恒等式（由幂级数逐项求导可证，对任意 z ≠ 0 成立）
/// c2' = (1 − z·c3 − 2·c2)/(2z)，c3' = (c2 − 3·c3)/(2z)；
/// |z| ≤ 阈值用幂级数逐项导数，避免 0/0。
fn stumpff_deriv(z: f64, c2: f64, c3: f64) -> (f64, f64) {
    if z.abs() > STUMPFF_EPS {
        (
            (1.0 - z * c3 - 2.0 * c2) / (2.0 * z),
            (c2 - 3.0 * c3) / (2.0 * z),
        )
    } else {
        (
            -1.0 / 24.0 + z / 360.0 - z * z / 13440.0,
            -1.0 / 120.0 + z / 2520.0,
        )
    }
}

/// Newton 解 universal Kepler 方程（Vallado 2013 Algorithm 8）。
///
/// F(χ) = (r0·v0/√μ)·χ²·c2 + (1 − α·r0)·χ³·c3 + r0·χ − √μ·dt = 0，
/// 其中 α = 1/a，ψ = αχ²。F 对 χ 单调递增，初值按圆锥曲线类型分段。
fn solve_universal_kepler(
    r0: f64,
    rdotv: f64,
    alpha: f64,
    sqmu: f64,
    dt: f64,
) -> Result<f64, String> {
    let mut chi = if alpha > 0.0 {
        sqmu * dt * alpha
    } else {
        sqmu * dt / r0
    };
    let tol_f = NEWTON_TOL_F_REL * sqmu * dt.abs();
    for _ in 0..MAX_NEWTON_ITERS {
        let psi = alpha * chi * chi;
        let (c2, c3) = stumpff(psi);
        let f_val =
            (rdotv / sqmu) * chi * chi * c2 + (1.0 - alpha * r0) * chi * chi * chi * c3 + r0 * chi
                - sqmu * dt;
        if f_val.abs() < tol_f {
            return Ok(chi);
        }
        let df = (rdotv / sqmu) * chi * (1.0 - psi * c3) + (1.0 - alpha * r0) * chi * chi * c2 + r0;
        let delta = f_val / df;
        chi -= delta;
        if delta.abs() < NEWTON_TOL_DX {
            return Ok(chi);
        }
    }
    Err(format!(
        "propagate_kepler: universal Kepler 方程 Newton 迭代未收敛 (dt={dt})"
    ))
}

/// 单点 UV 传播：dt 为相对 state0 历元的流逝秒（可负）。
///
/// 返回 (末态, 可选 STM)。dt == 0 短路返回初值与 Φ = I。
fn propagate_point(
    state0: &[f64; 6],
    dt: f64,
    mu: f64,
    with_stm: bool,
) -> Result<([f64; 6], Option<[f64; 36]>), String> {
    if dt == 0.0 {
        let mut phi = [0.0f64; 36];
        if with_stm {
            for i in 0..6 {
                phi[i * 6 + i] = 1.0;
            }
        }
        return Ok((*state0, if with_stm { Some(phi) } else { None }));
    }

    let r0v = [state0[0], state0[1], state0[2]];
    let v0v = [state0[3], state0[4], state0[5]];
    let r0 = norm(&r0v);
    let rdotv = dot(&r0v, &v0v);
    let alpha = 2.0 / r0 - dot(&v0v, &v0v) / mu;
    let sqmu = mu.sqrt();

    let chi = solve_universal_kepler(r0, rdotv, alpha, sqmu, dt)?;

    let psi = alpha * chi * chi;
    let (c2, c3) = stumpff(psi);
    let chi2 = chi * chi;
    let chi3 = chi2 * chi;
    let chi4 = chi2 * chi2;

    let f = 1.0 - chi2 * c2 / r0;
    let g = dt - chi3 * c3 / sqmu;
    let rv = [
        f * r0v[0] + g * v0v[0],
        f * r0v[1] + g * v0v[1],
        f * r0v[2] + g * v0v[2],
    ];
    let rm = norm(&rv);
    let fdot = sqmu / (rm * r0) * chi * (psi * c3 - 1.0);
    let gdot = 1.0 - chi2 * c2 / rm;
    let vv = [
        fdot * r0v[0] + gdot * v0v[0],
        fdot * r0v[1] + gdot * v0v[1],
        fdot * r0v[2] + gdot * v0v[2],
    ];
    let state = [rv[0], rv[1], rv[2], vv[0], vv[1], vv[2]];
    if !with_stm {
        return Ok((state, None));
    }

    // ---- 解析 STM：f & g 偏导数 + 隐函数定理链式法 ----
    //
    // 记 y0 的各标量函数 r0、rdotv、α、ψ = αχ²。所有标量对 y0 的梯度
    // 先在 χ 视为独立的条件下展开，χ 自身的依赖经 grad_chi = −(∂F/∂y0)/(∂F/∂χ)
    // 以 ∂y/∂χ·grad_chi 并入。行主序 Φ[i][j] = ∂y_i/∂y0_j。
    let (c2p, c3p) = stumpff_deriv(psi, c2, c3);

    let grad_r0 = [r0v[0] / r0, r0v[1] / r0, r0v[2] / r0, 0.0, 0.0, 0.0];
    let grad_rdotv = [v0v[0], v0v[1], v0v[2], r0v[0], r0v[1], r0v[2]];
    let r0_2 = r0 * r0;
    let r0_3 = r0_2 * r0;
    let grad_alpha = [
        -2.0 * r0v[0] / r0_3,
        -2.0 * r0v[1] / r0_3,
        -2.0 * r0v[2] / r0_3,
        -2.0 * v0v[0] / mu,
        -2.0 * v0v[1] / mu,
        -2.0 * v0v[2] / mu,
    ];
    let grad_psi = scale(grad_alpha, chi2);

    let a_coef = rdotv / sqmu;
    let b_coef = 1.0 - alpha * r0;
    // ∂F/∂ψ：F 的 ψ 通道系数（A·χ²·c2' + B·χ³·c3'）。ψ = αχ² 只经 α 槽
    // 进入 ∇F（α 固定时 ∂ψ/∂r0 = 0），不得再挂到 r0 槽重复计入。
    let df_dpsi = a_coef * chi2 * c2p + b_coef * chi3 * c3p;
    let df_dchi = a_coef * chi * (1.0 - psi * c3) + b_coef * chi2 * c2 + r0;
    let grad_fund = add(
        add(
            scale(grad_r0, chi - alpha * chi3 * c3),
            scale(grad_rdotv, chi2 * c2 / sqmu),
        ),
        scale(grad_alpha, -r0 * chi3 * c3 + chi2 * df_dpsi),
    );
    let grad_chi = scale(grad_fund, -1.0 / df_dchi);

    // 固定 χ 下 f、g 对 y0 的梯度（ψ 通道经 grad_psi，系数只带一个 χ²）。
    let grad_f = add(
        scale(grad_r0, chi2 * c2 / r0_2),
        scale(grad_psi, -chi2 * c2p / r0),
    );
    let grad_g = scale(grad_psi, -chi3 * c3p / sqmu);

    // ∂y/∂χ（位置、速度两段）。
    let df_dchi = -(2.0 * chi * c2 + 2.0 * alpha * chi3 * c2p) / r0;
    let dg_dchi = -(3.0 * chi2 * c3 + 2.0 * alpha * chi4 * c3p) / sqmu;
    let dr_dchi = [
        df_dchi * r0v[0] + dg_dchi * v0v[0],
        df_dchi * r0v[1] + dg_dchi * v0v[1],
        df_dchi * r0v[2] + dg_dchi * v0v[2],
    ];

    // 固定 χ 下 rm = |f·r0v + g·v0v| 对 y0 的梯度。
    let mut grad_rm = [0.0f64; 6];
    for j in 0..6 {
        let mut col = [0.0f64; 3];
        for i in 0..3 {
            let mut d = r0v[i] * grad_f[j] + v0v[i] * grad_g[j];
            if j < 3 {
                if i == j {
                    d += f;
                }
            } else if i == j - 3 {
                d += g;
            }
            col[i] = d;
        }
        grad_rm[j] = dot(&rv, &col) / rm;
    }

    // 固定 χ 下 ġ = 1 − χ²c2/rm 与 ḟ = K/(rm·r0)（K = √μ·χ·(ψc3 − 1)）的梯度。
    // （grad_rm 为 [f64; 6] 值，先算 ġ 再算 ḟ 以免 move 后再用。）
    let grad_gdot = add(
        scale(grad_psi, -chi2 * c2p / rm),
        scale(grad_rm, chi2 * c2 / (rm * rm)),
    );
    let grad_fdot = add(
        add(
            scale(grad_psi, sqmu * chi * (c3 + psi * c3p) / (rm * r0)),
            scale(grad_rm, -fdot / rm),
        ),
        scale(grad_r0, -fdot / r0),
    );

    // ∂ḟ/∂χ、∂ġ/∂χ（含 rm 对 χ 的依赖）。
    let k_val = sqmu * chi * (psi * c3 - 1.0);
    let dk_dchi = sqmu * ((psi * c3 - 1.0) + 2.0 * alpha * chi2 * (c3 + psi * c3p));
    let drm_dchi = dot(&rv, &dr_dchi) / rm;
    let dfdot_dchi = dk_dchi / (rm * r0) - k_val * drm_dchi / (rm * rm * r0);
    let dgdot_dchi =
        -(2.0 * chi * c2 + 2.0 * alpha * chi3 * c2p) / rm + chi2 * c2 * drm_dchi / (rm * rm);
    let dv_dchi = [
        dfdot_dchi * r0v[0] + dgdot_dchi * v0v[0],
        dfdot_dchi * r0v[1] + dgdot_dchi * v0v[1],
        dfdot_dchi * r0v[2] + dgdot_dchi * v0v[2],
    ];

    // Φ[:, j] = (∂y/∂χ)·(∂χ/∂y0_j) + (∂y/∂y0_j)|_χ。
    let mut phi = [0.0f64; 36];
    for j in 0..6 {
        let w = grad_chi[j];
        for i in 0..3 {
            let eye_r = if j < 3 {
                if i == j {
                    f
                } else {
                    0.0
                }
            } else if i == j - 3 {
                g
            } else {
                0.0
            };
            phi[i * 6 + j] = dr_dchi[i] * w + r0v[i] * grad_f[j] + v0v[i] * grad_g[j] + eye_r;
            let eye_v = if j < 3 {
                if i == j {
                    fdot
                } else {
                    0.0
                }
            } else if i == j - 3 {
                gdot
            } else {
                0.0
            };
            phi[(3 + i) * 6 + j] =
                dv_dchi[i] * w + r0v[i] * grad_fdot[j] + v0v[i] * grad_gdot[j] + eye_v;
        }
    }

    Ok((state, Some(phi)))
}

/// UV 二体 Kepler 封闭解传播：对 `t_eval` 每个流逝秒独立求解。
///
/// `state0` 为 [x, y, z, vx, vy, vz]（km, km/s），`mu` 单位 km³/s²；
/// `t_eval` 各元素为相对 state0 历元的流逝秒，可为负（后向传播）。
/// 迭代不收敛时返回 Err（Vallado 2013 §2-5 定义性公式，ADR 0055 决策 5 ①）。
pub fn propagate_kepler(
    state0: &[f64; 6],
    t_eval: &[f64],
    mu: f64,
    with_stm: bool,
) -> Result<KeplerPropResult, String> {
    let mut times = Vec::with_capacity(t_eval.len());
    let mut states = Vec::with_capacity(t_eval.len());
    let mut stms = Vec::new();
    if with_stm {
        stms = Vec::with_capacity(t_eval.len());
    }
    for &dt in t_eval {
        let (state, phi) = propagate_point(state0, dt, mu, with_stm)?;
        times.push(dt);
        states.push(state);
        if let Some(p) = phi {
            stms.push(p);
        }
    }
    Ok(KeplerPropResult {
        times,
        states,
        stms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const MU: f64 = 398600.4418;

    /// 6×6 行主序矩阵的行列式（部分主元高斯消去）。
    fn det6(m: &[f64; 36]) -> f64 {
        let mut a = *m;
        let mut det = 1.0;
        for col in 0..6 {
            let (pivot, _) = (col..6)
                .map(|r| (r, a[r * 6 + col].abs()))
                .fold((col, 0.0), |acc, x| if x.1 > acc.1 { x } else { acc });
            if pivot != col {
                for k in 0..6 {
                    a.swap(col * 6 + k, pivot * 6 + k);
                }
                det = -det;
            }
            let p = a[col * 6 + col];
            det *= p;
            if p.abs() < 1e-300 {
                return 0.0;
            }
            for r in (col + 1)..6 {
                let factor = a[r * 6 + col] / p;
                for k in col..6 {
                    a[r * 6 + k] -= factor * a[col * 6 + k];
                }
            }
        }
        det
    }

    #[test]
    fn stumpff_at_zero_matches_series() {
        let (c2, c3) = stumpff(0.0);
        assert_eq!(c2, 0.5);
        assert_eq!(c3, 1.0 / 6.0);
    }

    #[test]
    fn circular_orbit_full_period_closure() {
        let r = 8000.0;
        let v = (MU / r).sqrt();
        let state0 = [r, 0.0, 0.0, 0.0, v, 0.0];
        let period = 2.0 * PI * (r * r * r / MU).sqrt();
        let out = propagate_kepler(&state0, &[period], MU, false).unwrap();
        let s = out.states[0];
        for k in 0..3 {
            assert!(
                (s[k] - state0[k]).abs() < 1e-9,
                "位置未闭合: dr = {}",
                (s[k] - state0[k]).abs()
            );
        }
        for k in 3..6 {
            assert!(
                (s[k] - state0[k]).abs() < 1e-9,
                "速度未闭合: dv = {}",
                (s[k] - state0[k]).abs()
            );
        }
    }

    #[test]
    fn stm_det_equals_one() {
        // 椭圆（e ≈ 0.0075 近圆）与双曲（v0 > v_esc）各取正负多个 Δt。
        let elliptic = [7000.0, 0.0, 0.0, 0.0, 7.5, 1.0];
        let hyperbolic = [7000.0, 0.0, 0.0, 0.0, 11.0, 0.0];
        let cases = [
            ("elliptic", elliptic, [600.0, -1800.0, 3600.0]),
            ("hyperbolic", hyperbolic, [600.0, -1800.0, 1800.0]),
        ];
        for (name, state0, dts) in cases {
            let out = propagate_kepler(&state0, &dts, MU, true).unwrap();
            for (k, stm) in out.stms.iter().enumerate() {
                let det = det6(stm);
                assert!(
                    (det - 1.0).abs() < 1e-12,
                    "{name} dt={} |det−1| = {}",
                    dts[k],
                    (det - 1.0).abs()
                );
            }
        }
    }
}
