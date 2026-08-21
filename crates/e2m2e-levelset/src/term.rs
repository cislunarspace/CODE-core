//! HJ PDE 的时间项：对应 ToolboxLS 的 `ExplicitIntegration/Term/`。
//!
//! MATLAB 原型（`odeCFL3.m` 第 26-33 行）：
//!
//! ```matlab
//! [ydot, stepBound, schemeData] = schemeFunc(t, y, schemeData)
//! ```
//!
//! 其中 `schemeData` 是弱类型参数包。Rust 里每个项是一个结构体，
//! MATLAB `schemeData` 的必备字段成为结构体字段，用户自由附加的参数
//! 放进 [`crate::hamiltonian::Hamiltonian`] 的实现里。
//!
//! 与 MATLAB 版的取舍：向量水平集（`termLaxFriedrichs.m` 第 74-79 行
//! 仅取第一个分量的扩展）不移植，所有项只处理单个网格函数。

use crate::boundary::matlab_sign;
use crate::derivative::UpwindDerivative;
use crate::dissipation::Dissipation;
use crate::grid::Grid;
use crate::hamiltonian::Hamiltonian;
use ndarray::{ArrayD, Zip};

/// 一个时间步的右端项求值结果。
pub struct TermRhs {
    /// `D_t φ`，形状与网格相同。
    pub dphi_dt: ArrayD<f64>,
    /// CFL 步长上界，实际步长由积分器乘以 `factor_cfl` 决定。
    pub step_bound: f64,
}

/// CFL 约束的时间项，即方法线（method of lines）后的 ODE 右端。
pub trait Term {
    fn rhs(&mut self, t: f64, phi: &ArrayD<f64>) -> TermRhs;
}

/// 一般哈密顿量项（`termLaxFriedrichs.m`），求解
/// `D_t φ = -H(x, t, φ, ∇φ)`。
///
/// MATLAB `schemeData` 必备字段（第 31-49 行）与本体字段的对应：
/// `.grid → grid`，`.derivFunc → deriv`，`.dissFunc → dissipation`，
/// `.hamFunc/.partialFunc → hamiltonian`（两个回调合并为一个 trait）。
///
/// 更新公式（第 176-182 行）：`dphi_dt = -(ham(derivC) - diss)`。
pub struct LaxFriedrichsTerm<D, P, S> {
    pub grid: Grid,
    pub hamiltonian: D,
    pub deriv: P,
    pub dissipation: S,
}

impl<D, P, S> LaxFriedrichsTerm<D, P, S>
where
    D: Hamiltonian,
    P: UpwindDerivative,
    S: Dissipation,
{
    pub fn new(grid: Grid, hamiltonian: D, deriv: P, dissipation: S) -> Self {
        Self {
            grid,
            hamiltonian,
            deriv,
            dissipation,
        }
    }
}

impl<D, P, S> Term for LaxFriedrichsTerm<D, P, S>
where
    D: Hamiltonian,
    P: UpwindDerivative,
    S: Dissipation,
{
    fn rhs(&mut self, t: f64, phi: &ArrayD<f64>) -> TermRhs {
        self.grid.check_data(phi);
        let dim = self.grid.dim();
        let mut deriv_l = Vec::with_capacity(dim);
        let mut deriv_r = Vec::with_capacity(dim);
        let mut deriv_c = Vec::with_capacity(dim);
        for i in 0..dim {
            let (l, r) = self.deriv.deriv(&self.grid, phi, i);
            deriv_c.push(0.5 * (&l + &r));
            deriv_l.push(l);
            deriv_r.push(r);
        }
        let ham = self.hamiltonian.hamiltonian(t, &self.grid, phi, &deriv_c);
        let (diss, step_bound) =
            self.dissipation
                .dissipation(t, &self.grid, phi, &deriv_l, &deriv_r, &self.hamiltonian);
        let mut dphi_dt = ArrayD::zeros(self.grid.shape());
        Zip::from(&mut dphi_dt)
            .and(&ham)
            .and(&diss)
            .for_each(|o, h, d| *o = -(h - d));
        TermRhs {
            dphi_dt,
            step_bound,
        }
    }
}

/// 法向速度项（`termNormal.m`），求解
/// `D_t φ + a(x,t) ‖∇φ‖ = 0`（O&F 第 6.2 节的 Godunov 格式）。
pub struct NormalTerm {
    pub grid: Grid,
    pub deriv: Box<dyn UpwindDerivative>,
    pub speed: NormalSpeed,
}

/// 动态法向速度回调：`a = f(t, grid, phi)`，形状与网格相同。
pub type SpeedFn = Box<dyn Fn(f64, &Grid, &ArrayD<f64>) -> ArrayD<f64> + Send + Sync>;

/// 法向速度 `a(x,t)`，对应 `schemeData.speed` 的三种给法
/// （`termNormal.m` 第 35-43 行）：常数、网格函数或回调。
pub enum NormalSpeed {
    Constant(f64),
    /// 与网格同形的速度场。
    Grid(ArrayD<f64>),
    Dynamic(SpeedFn),
}

impl NormalTerm {
    /// 解析当前时刻的速度场（常数广播为数组），返回 (速度场, max|a|)。
    fn resolve_speed(&self, t: f64, phi: &ArrayD<f64>) -> (ArrayD<f64>, f64) {
        match &self.speed {
            NormalSpeed::Constant(c) => (ArrayD::from_elem(self.grid.shape(), *c), c.abs()),
            NormalSpeed::Grid(a) => {
                self.grid.check_data(a);
                let m = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                (a.clone(), m)
            }
            NormalSpeed::Dynamic(f) => {
                let a = f(t, &self.grid, phi);
                self.grid.check_data(&a);
                let m = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                (a, m)
            }
        }
    }
}

impl Term for NormalTerm {
    fn rhs(&mut self, t: f64, phi: &ArrayD<f64>) -> TermRhs {
        self.grid.check_data(phi);
        let (speed, _) = self.resolve_speed(t, phi);
        let mut magnitude: ArrayD<f64> = ArrayD::zeros(self.grid.shape());
        // 逐节点累计的 CFL 倒数（termNormal.m 第 143、173-175 行）。
        let mut step_bound_inv: ArrayD<f64> = ArrayD::zeros(self.grid.shape());
        for i in 0..self.grid.dim() {
            let (deriv_l, deriv_r) = self.deriv.deriv(&self.grid, phi, i);
            let dx_inv = 1.0 / self.grid.dx()[i];
            Zip::from(&mut magnitude)
                .and(&mut step_bound_inv)
                .and(&speed)
                .and(&deriv_l)
                .and(&deriv_r)
                .for_each(|mag, sbi, a, l, r| {
                    // Godunov 通量的特征方向判定（termNormal.m 第 160-163 行）。
                    let (pl, pr) = (a * l, a * r);
                    let (ml, mr) = (pl.abs(), pr.abs());
                    let flow_l = (pl >= 0.0 && pr >= 0.0) || (pl >= 0.0 && pr <= 0.0 && ml >= mr);
                    let flow_r = (pl <= 0.0 && pr <= 0.0) || (pl >= 0.0 && pr <= 0.0 && ml < mr);
                    // flowL / flowR 互斥（发散特征两者皆假，梯度贡献为 0）。
                    if flow_l {
                        *mag += l * l;
                        *sbi += ml * dx_inv;
                    }
                    if flow_r {
                        *mag += r * r;
                        *sbi += mr * dx_inv;
                    }
                });
        }
        magnitude.mapv_inplace(|v: f64| v.sqrt());
        let mut dphi_dt = ArrayD::zeros(self.grid.shape());
        let mut worst_inv = 0.0f64;
        Zip::from(&mut dphi_dt)
            .and(&speed)
            .and(&magnitude)
            .and(&step_bound_inv)
            .for_each(|d, a, m, inv| {
                *d = -(a * m);
                if *m > 0.0 {
                    worst_inv = worst_inv.max(inv / m);
                }
            });
        TermRhs {
            dphi_dt,
            step_bound: 1.0 / worst_inv,
        }
    }
}

/// 重初始化项（`termReinit.m`），求解
/// `D_t φ = -sign(φ₀) (‖∇φ‖ - 1)`（Godunov 格式，Sussman-Smereka-Osher
/// 1994），把隐式表面函数修复为符号距离函数。亚网格修正采用 Russo &
/// Smereka (JCP 2000) 的鲁棒变体（源码第 114-120 行的 robust_subcell）。
pub struct ReinitTerm {
    pub grid: Grid,
    pub deriv: Box<dyn UpwindDerivative>,
    /// 冻结的初值 φ₀（对应 `schemeData.initial`）。
    pub initial: ArrayD<f64>,
    /// Russo-Smereka 亚网格修正开关（`schemeData.subcell_fix_order`，
    /// 0 关闭 / 1 开启，默认开启）。
    pub subcell_fix: bool,
}

impl ReinitTerm {
    /// 节点是否在零等值面附近（`isNearInterface.m`，非严格比较：相邻
    /// 节点符号不同即双方都算近）。
    fn near_interface(&self) -> ArrayD<bool> {
        let init = |idx: &ndarray::IxDyn| -> f64 { self.initial[idx.clone()] };
        let mut near =
            ArrayD::from_shape_fn(self.grid.shape(), |idx| matlab_sign(init(&idx)) == 0.0);
        for d in 0..self.grid.dim() {
            let n = self.grid.n()[d];
            // 相邻对 (j, j+1) 符号不同则两者都标近。
            let mut pairs = Vec::new();
            for (idx, _) in near.indexed_iter() {
                if idx[d] + 1 < n {
                    let mut j = idx.clone();
                    j[d] += 1;
                    let (s0, s1) = (matlab_sign(init(&idx)), matlab_sign(init(&j)));
                    if s0 != s1 {
                        pairs.push((idx.clone(), j));
                    }
                }
            }
            for (a, b) in pairs {
                near[a.clone()] = true;
                near[b] = true;
            }
        }
        near
    }

    /// Russo-Smereka 亚网格距离 D（`termReinit.m` 第 268-308 行的鲁棒版）。
    fn subcell_distance(&self) -> ArrayD<f64> {
        let shape = self.grid.shape();
        let mut denom = ArrayD::zeros(shape.clone());
        let robust_eps = 1e6 * f64::EPSILON;
        let init = |idx: &ndarray::IxDyn| -> f64 { self.initial[idx.clone()] };
        for d in 0..self.grid.dim() {
            let n = self.grid.n()[d];
            let dx_inv = 1.0 / self.grid.dx()[d];
            // 长差分：相邻半节点中心差分的平方；末节点置零。
            let mut diff2 = ArrayD::from_shape_fn(shape.clone(), |idx| {
                let j = idx[d];
                if j + 1 < n {
                    let mut k = idx.clone();
                    k[d] = j + 1;
                    let half = 0.5 * dx_inv * (init(&k) - init(&idx));
                    half * half
                } else {
                    0.0
                }
            });
            // 短差分（整节点间距）的平方与长差分取 max，写入左右两个节点
            // 位置，并抬底 robust_eps²（源码第 283-294 行）。先收集再写，
            // 避免在同一数组上边读边写。
            let mut updates: Vec<(ndarray::IxDyn, f64)> = Vec::new();
            for (idx, v) in diff2.indexed_iter() {
                let j = idx[d];
                if j + 1 >= n {
                    updates.push((idx.clone(), (*v).max(robust_eps * robust_eps)));
                    continue;
                }
                let mut k = idx.clone();
                k[d] = j + 1;
                let short = dx_inv * (init(&k) - init(&idx));
                let s2 = short * short;
                updates.push((idx.clone(), (*v).max(s2)));
                let cur_right = diff2[k.clone()];
                updates.push((k, cur_right.max(s2)));
            }
            for (idx, v) in updates {
                diff2[idx] = v.max(robust_eps * robust_eps);
            }
            denom = &denom + &diff2;
        }
        denom.mapv_inplace(|v: f64| v.sqrt());
        &self.initial / &denom
    }
}

impl Term for ReinitTerm {
    fn rhs(&mut self, _t: f64, phi: &ArrayD<f64>) -> TermRhs {
        self.grid.check_data(phi);
        let dim = self.grid.dim();
        // 符号函数：开启亚网格修正时用纯 sign（远离界面无需平滑），
        // 否则用平滑近似 smearedSign = φ/sqrt(φ² + max(dx)²)（O&F 7.5）。
        let s: ArrayD<f64> = if self.subcell_fix {
            self.initial.mapv(matlab_sign)
        } else {
            let sgn_factor = self.grid.dx().iter().fold(0.0f64, |m, v| m.max(*v)).powi(2);
            self.initial.mapv(|v| v / (v * v + sgn_factor).sqrt())
        };

        // 逐维 Godunov 导数（termReinit.m 第 184-216 行）。
        let mut derivs = Vec::with_capacity(dim);
        let mut step_bound_inv = 0.0f64;
        for i in 0..dim {
            let (deriv_l, deriv_r) = self.deriv.deriv(&self.grid, phi, i);
            let mut deriv = ArrayD::zeros(self.grid.shape());
            Zip::from(&mut deriv)
                .and(&s)
                .and(&deriv_l)
                .and(&deriv_r)
                .for_each(|d, s, l, r| {
                    let (sl, sr) = (s * l, s * r);
                    let mut flow_l = sr <= 0.0 && sl <= 0.0;
                    let mut flow_r = sr >= 0.0 && sl >= 0.0;
                    // 收敛特征：比较两侧到达时间（第 201-212 行）。
                    if sr < 0.0 && sl > 0.0 {
                        let denom = r - l;
                        let t = s * (r.abs() - l.abs()) / denom;
                        flow_l |= t < 0.0;
                        flow_r |= t >= 0.0;
                    }
                    // 注意交叉：向右流动取左差分，向左流动取右差分。
                    *d = l * (flow_r as u8 as f64) + r * (flow_l as u8 as f64);
                });
            derivs.push(deriv);
        }
        // 梯度模与有效速度（第 220-243 行）。
        let mut mag = ArrayD::zeros(self.grid.shape());
        for d in &derivs {
            Zip::from(&mut mag).and(d).for_each(|m, v| *m += v * v);
        }
        mag.mapv_inplace(|v: f64| v.sqrt().max(f64::EPSILON));
        let mut delta = s.mapv(|v| -v);
        for (i, d) in derivs.iter().enumerate() {
            Zip::from(&mut delta)
                .and(&s)
                .and(d)
                .and(&mag)
                .for_each(|del, s, dv, m| {
                    let v = s * dv / m;
                    *del += v * dv;
                });
            // 有效速度的 CFL 约束。
            let v_field = Zip::from(&s)
                .and(d)
                .and(&mag)
                .map_collect(|s, dv, m| s * dv / m);
            let v_max = v_field.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            step_bound_inv += v_max / self.grid.dx()[i];
        }

        // 亚网格修正（第 246-330 行）。
        if self.subcell_fix {
            let d_sub = self.subcell_distance();
            let near = self.near_interface();
            let max_dx = self.grid.dx().iter().fold(0.0f64, |m, v| m.max(*v));
            Zip::from(&mut delta)
                .and(&s)
                .and(phi)
                .and(&d_sub)
                .and(&near)
                .for_each(|del, s, cur, dist, near| {
                    if *near {
                        *del = (s * cur.abs() - dist) / max_dx;
                    }
                });
        }

        // MATLAB 末行 `ydot = -delta(:)`：组装的是 δ，右端项取负。
        delta.mapv_inplace(|v: f64| -v);
        TermRhs {
            dphi_dt: delta,
            step_bound: 1.0 / step_bound_inv,
        }
    }
}

/// 项之和（`termSum.m`）：`dphi_dt` 逐项相加，步长上界取调和组合
/// `1 / Σ(1/step_bound_i)`。
pub struct SumTerm(pub Vec<Box<dyn Term>>);

impl Term for SumTerm {
    fn rhs(&mut self, t: f64, phi: &ArrayD<f64>) -> TermRhs {
        assert!(!self.0.is_empty(), "SumTerm 至少要包含一个项");
        let mut sum = None;
        let mut step_bound_inv = 0.0f64;
        for term in &mut self.0 {
            let r = term.rhs(t, phi);
            step_bound_inv += 1.0 / r.step_bound;
            sum = Some(match sum {
                None => r.dphi_dt,
                Some(mut acc) => {
                    Zip::from(&mut acc)
                        .and(&r.dphi_dt)
                        .for_each(|a: &mut f64, b: &f64| *a += b);
                    acc
                }
            });
        }
        TermRhs {
            dphi_dt: sum.expect("非空"),
            step_bound: if step_bound_inv == 0.0 {
                f64::INFINITY
            } else {
                1.0 / step_bound_inv
            },
        }
    }
}

/// 更新限制项（`termRestrictUpdate.m`）：把内层项的更新钳制为非负
/// （`positive = true`，可达集只增长的常用设置）或非正，使积分单调。
pub struct RestrictUpdateTerm {
    pub inner: Box<dyn Term>,
    /// true 时更新钳为 ≥ 0，false 时钳为 ≤ 0（MATLAB 默认 true）。
    pub positive: bool,
}

impl Term for RestrictUpdateTerm {
    fn rhs(&mut self, t: f64, phi: &ArrayD<f64>) -> TermRhs {
        let mut r = self.inner.rhs(t, phi);
        if self.positive {
            r.dphi_dt.mapv_inplace(|v| v.max(0.0));
        } else {
            r.dphi_dt.mapv_inplace(|v| v.min(0.0));
        }
        r
    }
}
