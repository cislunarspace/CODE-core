//! Euler 旋转数学核（纯 std）。
//!
//! 行为地面真值：vendored CSPICE 的 `rotate.c`/`rotmat.c`/`eul2m.c`（经
//! `xf2eul.c` 内的 `eul2xf_`）与 `tisbod.c`/`pcke02.c` 用到的 Fortran `Mod`。
//! 全部**逐操作移植**，操作顺序不得重排——与 CSPICE 逐位一致（
//! `f64::to_bits` 相等）依赖浮点运算的精确顺序。
//!
//! 约定（SPICE 口径）：旋转矩阵作用于坐标（frame rotation），`rotate(axis,
//! angle)` 的矩阵把向量从「旋转后坐标系」表出到「原坐标系」；`eul2xf` 的
//! 输入角序为 `[角1, 角2, 角3, d1, d2, d3]`，矩阵
//! `r = [角1]_axis1 · [角2]_axis2 · [角3]_axis3`（角 1 最先施加、位于最右）。

/// `indexs[5] = { 3,1,2,3,1 }`（rotate.c/rotmat.c 共用的环形轴表，1 基）。
const AXIS_INDEX: [usize; 5] = [3, 1, 2, 3, 1];

/// `next[3] = { 2,3,1 }`（eul2xf 的轴序推进表，1 基）。
const NEXT_AXIS: [usize; 3] = [2, 3, 1];

/// `delta[3][3]`（eul2xf 的叉积符号表，1 基 [a][b]，列主序展平
/// `{0,-1,1, 1,0,-1, -1,1,0}`）。
const DELTA: [[f64; 3]; 3] = [[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]];

/// SPICELIB `rotate`：绕 `axis` 轴转 `angle` 弧度的坐标旋转矩阵。
///
/// `rotate.c` 逐操作移植：`s = sin(angle)`、`c = cos(angle)`，随后按
/// `(i1, i2, i3) = indexs[axis..axis+2]` 写入 9 个元素。
pub fn rotate(axis: i32, angle: f64) -> [[f64; 3]; 3] {
    let s = angle.sin();
    let c = angle.cos();
    let temp = axis.rem_euclid(3); // C: (iaxis % 3 + 3) % 3，0 基表下标
    let i1 = AXIS_INDEX[temp as usize]; // 1 基轴号
    let i2 = AXIS_INDEX[temp as usize + 1];
    let i3 = AXIS_INDEX[temp as usize + 2];
    let mut m = [[0.0_f64; 3]; 3];
    m[i1 - 1][i1 - 1] = 1.0;
    m[i2 - 1][i2 - 1] = c;
    m[i3 - 1][i2 - 1] = -s;
    m[i1 - 1][i2 - 1] = 0.0;
    m[i2 - 1][i3 - 1] = s;
    m[i3 - 1][i3 - 1] = c;
    m
}

/// SPICELIB `rotmat`：`r = R(axis, angle) · m`（左乘）。
///
/// `rotmat.c` 逐操作移植：第 i1 行原样，第 i2/i3 行按 `c·x + s·y` /
/// `-s·x + c·y` 两项和组合。
pub fn rotmat(m: &[[f64; 3]; 3], angle: f64, axis: i32) -> [[f64; 3]; 3] {
    let s = angle.sin();
    let c = angle.cos();
    let temp = axis.rem_euclid(3);
    let i1 = AXIS_INDEX[temp as usize];
    let i2 = AXIS_INDEX[temp as usize + 1];
    let i3 = AXIS_INDEX[temp as usize + 2];
    let mut out = [[0.0_f64; 3]; 3];
    for j in 0..3 {
        out[i1 - 1][j] = m[i1 - 1][j];
        out[i2 - 1][j] = c * m[i2 - 1][j] + s * m[i3 - 1][j];
        out[i3 - 1][j] = -s * m[i2 - 1][j] + c * m[i3 - 1][j];
    }
    out
}

/// SPICELIB `eul2m`：`r = [a3]_ax3 · [a2]_ax2 · [a1]_ax1`。
///
/// `eul2m.c` 逐操作移植：`r = rotate(ax1, a1)` 后连续 `rotmat` 左乘。
/// 相邻轴号不等（ax1≠ax2 且 ax2≠ax3，ax1 与 ax3 可同，如 3-1-3）由调用方
/// 保证（CSPICE 在此报 SPICE(BADAXISNUMBERS)；本仓调用面 axes 均合法，
/// 非法轴按 `panic` 快速失败）。
pub fn eul2m(a1: f64, a2: f64, a3: f64, ax1: i32, ax2: i32, ax3: i32) -> [[f64; 3]; 3] {
    assert!(
        ax1 != ax2 && ax2 != ax3,
        "EUL2M 相邻轴号必须不等: ({ax1}, {ax2}, {ax3})"
    );
    let r = rotate(ax1, a1);
    let r = rotmat(&r, a2, ax2);
    rotmat(&r, a3, ax3)
}

/// SPICELIB `eul2xf`：Euler 角 + 角变率 → 6×6 状态变换矩阵。
///
/// `xf2eul.c` 内 `eul2xf_`（`xf2eul_0_` 的 select=1 分支）逐操作移植：
/// 输入 `[角1, 角2, 角3, d1, d2, d3]`，矩阵
/// `r = [角1]_axisa · [角2]_axisb · [角3]_axisc`，输出
/// `[[r, 0], [dr/dt, r]]`。
pub fn eul2xf(eulang: &[f64; 6], axisa: i32, axisb: i32, axisc: i32) -> [[f64; 6]; 6] {
    let mut locang = *eulang;
    let mut locaxb = axisb;
    // 轴号退化（axisb 与 axisa/axisc 相同）时把第二角折进第一/三角
    // （eul2xf.c 原样；3-1-3 等常规因子化不走此分支）。
    if axisb == axisa || axisb == axisc {
        let i = if axisb == axisa { 1usize } else { 3usize };
        locang[i - 1] += locang[1];
        locang[1] = 0.0;
        locang[i + 1] += locang[4];
        locang[4] = 0.0;
        locaxb = if axisc == NEXT_AXIS[(axisa - 1) as usize] as i32 {
            NEXT_AXIS[(axisc - 1) as usize] as i32
        } else {
            NEXT_AXIS[(axisa - 1) as usize] as i32
        };
    }
    // eul2m 调用：C 的 eul2m(angle3, angle2, angle1, axis3, axis2, axis1)
    // 把 locang[0] 作为最左因子（r = [locang[0]]_axisa · [locang[1]] ·
    // [locang[2]]_axisc）；本仓 eul2m(a1,a2,a3,ax1,ax2,ax3) 把 a1 作为最右
    // 因子，故实参按位反转。
    let r = eul2m(locang[2], locang[1], locang[0], axisc, locaxb, axisa);
    let a = axisa as usize;
    let b = locaxb as usize;
    let l = 6 - a - b;
    let d = DELTA[a - 1][b - 1];
    let ca = locang[0].cos();
    let sa = locang[0].sin();
    let (u, v) = if axisa == axisc {
        (locang[1].cos(), d * locang[1].sin())
    } else {
        (-d * locang[1].sin(), locang[1].cos())
    };
    // solutn 3×3（行优先）：domega = S · [d1, d2, d3]（mxv，3 项和）。
    // 行 0 = (-d, 0, -d*u)，行 1 = (0, -d*ca, -sa*v)，行 2 = (0, sa, -d*ca*v)。
    let domega = [
        -d * locang[3] + 0.0 * locang[4] + (-d * u) * locang[5],
        0.0 * locang[3] + (-d * ca) * locang[4] + (-sa * v) * locang[5],
        0.0 * locang[3] + sa * locang[4] + (-d * ca * v) * locang[5],
    ];
    // drdtrt 反对称角速度矩阵（eul2xf.c 的 6 个赋值原样）。
    let mut drdtrt = [[0.0_f64; 3]; 3];
    drdtrt[l - 1][b - 1] = domega[0];
    drdtrt[b - 1][l - 1] = -domega[0];
    drdtrt[a - 1][l - 1] = domega[1];
    drdtrt[l - 1][a - 1] = -domega[1];
    drdtrt[b - 1][a - 1] = domega[2];
    drdtrt[a - 1][b - 1] = -domega[2];
    // drdt = drdtrt · r（mxm，3 项和）。
    let mut drdt = [[0.0_f64; 3]; 3];
    for (i, row) in drdt.iter_mut().enumerate() {
        for (j, out) in row.iter_mut().enumerate() {
            *out = drdtrt[i][0] * r[0][j] + drdtrt[i][1] * r[1][j] + drdtrt[i][2] * r[2][j];
        }
    }
    // 组装 6×6：[[r, 0], [drdt, r]]（eul2xf.c 的 4 段写入原样；
    // f2c 列主序缓冲经 eul2xf_c 的 xpose6 转回行主序，等价于直接按
    // 数学矩阵装配）。
    let mut xform = [[0.0_f64; 6]; 6];
    for i in 0..3 {
        for j in 0..3 {
            xform[i][j] = r[i][j];
            xform[i + 3][j + 3] = r[i][j];
            xform[i + 3][j] = drdt[i][j];
            xform[i][j + 3] = 0.0;
        }
    }
    xform
}

/// Fortran `d_mod(x, y)`——**mice 编译版**（`IEEE_drem` 未定义，走 f2c
/// 兜底分支）：`q = trunc(x/y)`（商来自**舍入后的除法**），结果
/// `x − y·q`，符号随 x。
///
/// 注意不能用 Rust 的 `%`（= 精确 IEEE fmod，商精确）：对大角度
/// （IAU W 多项式、BPC 第三角累积角）两者相差 1e-13 量级，破坏与
/// CSPICE 的逐位一致（tisbod/pcke02 实测钉死）。`pcke02.c` 的第三角
/// `mod 2π` 与 `tisbod.c` 的 `w = d_mod(w, twopi)` 用。
pub fn pos_mod(x: f64, y: f64) -> f64 {
    let mut quotient = x / y;
    if quotient >= 0.0 {
        quotient = quotient.floor();
    } else {
        quotient = -((-quotient).floor());
    }
    x - y * quotient
}

/// 3×3 矩阵乘（行主序）。乘加顺序与 CSPICE `mxm` 一致：3 项顺序和。
pub(crate) fn mxm(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0_f64; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, out) in row.iter_mut().enumerate() {
            *out = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

/// 3×3 转置（逐元素拷贝，`xpose.c` 语义；旋转矩阵的逆）。
pub(crate) fn xpose(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

/// 6×6 状态变换矩阵的逆（SPICELIB `invstm` = `xposbl`）：仅对角 3×3 块
/// 转置，零块原样。`[[r,0],[drdt,r]]⁻¹ = [[rᵀ,0],[drdtᵀ,rᵀ]]`。
pub(crate) fn invstm(m: &[[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut out = [[0.0_f64; 6]; 6];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = m[j][i];
            out[i + 3][j + 3] = m[j + 3][i + 3];
            out[i + 3][j] = m[j + 3][i];
        }
    }
    out
}

/// 6×6 状态变换矩阵乘（行主序），块结构与乘加顺序与 CSPICE `zzmsxf`
/// 一致：左上 3 项和、左下 6 项和、右上置零、右下拷贝左上。
pub(crate) fn msxf(a: &[[f64; 6]; 6], b: &[[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut out = [[0.0_f64; 6]; 6];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    for i in 3..6 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j]
                + a[i][1] * b[1][j]
                + a[i][2] * b[2][j]
                + a[i][3] * b[3][j]
                + a[i][4] * b[4][j]
                + a[i][5] * b[5][j];
        }
    }
    for i in 3..6 {
        for j in 3..6 {
            out[i][j] = out[i - 3][j - 3];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f64, b: f64, tol: f64, what: &str) {
        assert!((a - b).abs() <= tol, "{what}: {a} != {b}");
    }

    /// rotate：已知旋转 + 正交性。
    #[test]
    fn rotate_known_and_orthonormal() {
        let half_pi = std::f64::consts::FRAC_PI_2;
        // 绕 z 转 90°（坐标旋转口径 R = [[c,s,0],[-s,c,0],[0,0,1]]）：
        // x 基向量 (1,0,0) 表出为 (0,-1,0)。
        let r = rotate(3, half_pi);
        let v = crate::spice_ffi::mat3_mul_vec(&r, &[1.0, 0.0, 0.0]);
        assert_close(v[0], 0.0, 1e-15, "x'");
        assert_close(v[1], -1.0, 1e-15, "y'");
        // 正交：r·rᵀ = I。
        let rt = xpose(&r);
        let p = mxm(&r, &rt);
        for (i, row) in p.iter().enumerate() {
            for (j, x) in row.iter().enumerate() {
                assert_close(*x, if i == j { 1.0 } else { 0.0 }, 1e-14, "r·rᵀ");
            }
        }
    }

    /// eul2m 与 rotate/rotmat 组合一致；rotmat 左乘语义。
    #[test]
    fn eul2m_matches_rotmat_composition() {
        let (a1, a2, a3) = (0.3, -0.7, 1.1);
        let m = eul2m(a1, a2, a3, 3, 1, 3);
        let mut acc = rotate(3, a1);
        acc = rotmat(&acc, a2, 1);
        acc = rotmat(&acc, a3, 3);
        for (r, o) in m.iter().zip(acc.iter()) {
            for (x, y) in r.iter().zip(o.iter()) {
                assert_eq!(x.to_bits(), y.to_bits(), "eul2m 与逐级 rotmat 逐位一致");
            }
        }
    }

    /// eul2xf：无速率时旋转块 == eul2m（按 eul2xf 的因子序：r =
    /// [角1]_3 [角2]_1 [角3]_3，角1 最左；本仓 eul2m 首参最右，故反转）。
    #[test]
    fn eul2xf_zero_rates_matches_eul2m() {
        let angles = [0.4, -0.2, 0.9];
        let mut eulang = [0.0_f64; 6];
        eulang[..3].copy_from_slice(&angles);
        let xf = eul2xf(&eulang, 3, 1, 3);
        let m = eul2m(angles[2], angles[1], angles[0], 3, 1, 3);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(xf[i][j].to_bits(), m[i][j].to_bits(), "旋转块逐位一致");
                assert_eq!(xf[i + 3][j + 3].to_bits(), m[i][j].to_bits(), "右下块一致");
                assert_eq!(xf[i][j + 3], 0.0, "右上块为零");
            }
        }
    }

    /// pos_mod：mice 编译版 d_mod（商舍入后 trunc，符号随 x）语义。
    #[test]
    fn pos_mod_matches_fortran() {
        let two_pi = std::f64::consts::TAU;
        // 大角：与「商舍入 + 乘回」的 C 序逐位一致（不是精确 fmod！）。
        let w = 2830.4382262844674_f64; // tisbod 实测场景
        let q = (w / two_pi).trunc();
        assert_eq!(pos_mod(w, two_pi).to_bits(), (w - two_pi * q).to_bits());
        assert_close(
            pos_mod(w, two_pi),
            3.004838053653657,
            1e-12,
            "tisbod 实测值",
        );
        // 负角：Fortran MOD 符号随 x（不是 Euclidean modulo）。
        assert_eq!(pos_mod(-0.3, two_pi), -0.3);
        assert_eq!(pos_mod(0.0, two_pi), 0.0);
        assert_eq!(pos_mod(2.0, 2.0), 0.0);
    }

    /// eul2xf 与 cspice_sys::eul2xf_c 逐位一致（BPC 派生角，tisbod
    /// 实测场景钉死；sin/cos 实现差异在本仓角度域内不出现）。
    #[test]
    fn eul2xf_bitwise_vs_cspice_sys() {
        let _g = crate::lock_spice_for_test();
        let e = [
            -0.06429580721124545_f64,
            0.39744798782727736,
            3.0048380536536143,
            -2.0558197069941127e-9,
            1.773969130186627e-9,
            2.6637932645112567e-6,
        ];
        let mine = eul2xf(&e, 3, 1, 3);
        let mut theirs = [[0.0_f64; 6]; 6];
        let mut e_in = e;
        unsafe {
            cspice_sys::eul2xf_c(e_in.as_mut_ptr(), 3, 1, 3, theirs.as_mut_ptr());
        }
        for i in 0..6 {
            for j in 0..6 {
                assert_eq!(mine[i][j].to_bits(), theirs[i][j].to_bits(), "[{i}][{j}]");
            }
        }
    }

    /// invstm：块转置后与原矩阵相乘得单位阵（[[r,0],[drdt,r]] 的逆）。
    #[test]
    fn invstm_inverts_state_transform() {
        let eulang = [0.2, -0.5, 1.3, 0.01, -0.02, 0.03];
        let xf = eul2xf(&eulang, 3, 1, 3);
        let inv = invstm(&xf);
        let p = msxf(&xf, &inv);
        for (i, row) in p.iter().enumerate() {
            for (j, x) in row.iter().enumerate() {
                assert_close(*x, if i == j { 1.0 } else { 0.0 }, 1e-14, "xf·invstm(xf)");
            }
        }
    }
}
