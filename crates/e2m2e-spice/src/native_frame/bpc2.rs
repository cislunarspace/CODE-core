//! BPC Type 2 段求值（Chebyshev，3 向量仅角度）。
//!
//! 行为地面真值：vendored CSPICE 的 `pckr02.c`（记录定位）、`pcke02.c`
//! （求值 + 第三角 mod 2π）与 `pckmat.c`（角度重排 + `eul2xf`）。
//! `pckr02.c` 与 `spkr02.c` 同构，记录定位/钳位逻辑直接照搬
//! [`crate::native_spk::spk2::evaluate`]；Chebyshev–Clenshaw 核复用
//! [`crate::native_spk::spk2::chbint`]（ADR 0056）。
//!
//! Type 2 BPC 记录布局与 SPK Type 2 完全一致：每条逻辑记录
//! `[MID, RADIUS, 角1系数, 角2系数, 角3系数]`，3 个分量是 Euler 角
//! （`pcke02` 语义下依次为 phi、delta、w），导数为角变率。

use crate::native_spk::daf::{read_words, word_at, DafSegment};
use crate::native_spk::DafSpkError;

use super::euler::pos_mod;

/// 求 BPC Type 2 段在 `et` 时刻的 Euler 角与角变率
/// `[角1, 角2, 角3, d角1, d角2, d角3]`（pcke02 语义顺序，弧度、弧度/秒）。
///
/// `pcke02.c` 原样：`spke02` 求值后**仅对第三角**做 `mod 2π`
/// （Fortran `d_mod`，结果 ∈ [0, 2π)），导数不动。
pub(crate) fn evaluate(bytes: &[u8], seg: &DafSegment, et: f64) -> Result<[f64; 6], DafSpkError> {
    let (baddr, eaddr) = seg.data_words;

    // 段尾 4 字：INIT / INTLEN / RSIZE / N（pckr02.c：dafgda(end-3, end)）。
    if eaddr < baddr + 4 {
        return Err(DafSpkError::BadFormat(format!(
            "BPC Type 2 段数据区过短: baddr={baddr} eaddr={eaddr}"
        )));
    }
    let init = word_at(bytes, eaddr - 3)?;
    let intlen = word_at(bytes, eaddr - 2)?;
    let rsize = word_at(bytes, eaddr - 1)? as i64;
    let nrec = word_at(bytes, eaddr)? as i64;
    if intlen <= 0.0 || rsize < 5 || nrec < 1 {
        return Err(DafSpkError::BadFormat(format!(
            "BPC Type 2 段描述非法: intlen={intlen} rsize={rsize} nrec={nrec}"
        )));
    }

    // recno = (integer)((et - INIT) / INTLEN) + 1，仅 min 钳位（pckr02.c
    // 原样：Fortran INT() 向零截断；覆盖性由选段层 begin <= et <= end 保证）。
    let recno = (((et - init) / intlen) as i64 + 1).min(nrec);
    let recadr = (recno - 1)
        .checked_mul(rsize)
        .and_then(|o| o.checked_add(baddr as i64));
    let recadr = match recadr {
        Some(a) if a > 0 => a as usize,
        _ => {
            return Err(DafSpkError::BadFormat(format!(
                "BPC Type 2 记录地址越界: recno={recno} rsize={rsize} baddr={baddr}"
            )));
        }
    };
    let rec_end = recadr + rsize as usize - 1;
    if rec_end > eaddr - 4 {
        return Err(DafSpkError::BadFormat(format!(
            "BPC Type 2 记录越过段尾: recadr={recadr} rec_end={rec_end} eaddr={eaddr}"
        )));
    }

    // 记录 = [MID, RADIUS, 角1系数, 角2系数, 角3系数]（pcke02.c 调
    // spke02 逐分量求值；ncof=(RSIZE-2)/3，系数从记录第 4 字起连续排布）。
    let record = read_words(bytes, recadr, rec_end)?;
    let ncof = ((rsize - 2) / 3) as usize;
    if ncof < 1 || 2 + 3 * ncof != rsize as usize {
        return Err(DafSpkError::BadFormat(format!(
            "BPC Type 2 记录尺寸非法: rsize={rsize} ncof={ncof}"
        )));
    }
    let mid = record[0];
    let radius = record[1];
    if radius <= 0.0 {
        return Err(DafSpkError::BadFormat(format!(
            "BPC Type 2 记录半径非法: radius={radius}"
        )));
    }
    let degp = ncof - 1;
    let x2s = [mid, radius];
    let mut eulang = [0.0_f64; 6];
    for axis in 0..3 {
        let cp = &record[2 + axis * ncof..2 + (axis + 1) * ncof];
        let (p, dp) = crate::native_spk::spk2::chbint(cp, degp, x2s, et);
        eulang[axis] = p;
        eulang[axis + 3] = dp;
    }

    // 仅对第三角做 mod 2π（pcke02.c 原样："We do this because we've
    // always done this."），导数不动。
    eulang[2] = pos_mod(eulang[2], std::f64::consts::TAU);
    Ok(eulang)
}

/// [`evaluate`] 的负控钩子：以调用方给定的轴序做 `eul2xf` 并返回旋转块。
///
/// 与 `daf::parse_with` 同例（ADR 0051）：仅供对拍测试证明「错误输入可被
/// 检出」——axes 固定 (3,1,3) 之外**不做任何语义承诺**（数据是按 3-1-3
/// 装段 Chebyshev 系数，换轴序只是数学上错误的读法），
/// 供 `euler_313_negative_control` 用 1-2-3 序列证明对拍能报错。
///
/// 段必须为 Type 2；`et` 需落段覆盖内（负控自行保证）。
#[doc(hidden)]
pub fn evaluate_with_axes(
    bytes: &[u8],
    seg: &DafSegment,
    et: f64,
    axes: [i32; 3],
) -> Result<[[f64; 3]; 3], DafSpkError> {
    if seg.dtype != 2 {
        return Err(DafSpkError::UnsupportedType(seg.dtype));
    }
    let eulang = evaluate(bytes, seg, et)?;
    let mut estate = [0.0_f64; 6];
    // pckmat.c 的重排：estate = [角3, 角2, 角1, d3, d2, d1]，随后按
    // eul2xf(estate, 3, 1, 3) 得 X(段参考系 → 体固)。负控在此换轴序。
    estate[0] = eulang[2];
    estate[1] = eulang[1];
    estate[2] = eulang[0];
    estate[3] = eulang[5];
    estate[4] = eulang[4];
    estate[5] = eulang[3];
    let xf = super::euler::eul2xf(&estate, axes[0], axes[1], axes[2]);
    let mut r = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = xf[i][j];
        }
    }
    Ok(r)
}

/// BPC Type 2 → `X(段参考系 → 体固)` 的 6×6 状态变换（`pckmat.c` 逐操作
/// 移植）：角度重排后 `eul2xf(estate, 3, 1, 3)`。
///
/// 段参考系不是 J2000 时（如 ECLIPJ2000）由调用方按 `tisbod.c` 的
/// pcref 调整逻辑把常值阵折进矩阵。
pub(crate) fn tsipm(bytes: &[u8], seg: &DafSegment, et: f64) -> Result<[[f64; 6]; 6], DafSpkError> {
    let eulang = evaluate(bytes, seg, et)?;
    let mut estate = [0.0_f64; 6];
    estate[0] = eulang[2];
    estate[1] = eulang[1];
    estate[2] = eulang[0];
    estate[3] = eulang[5];
    estate[4] = eulang[4];
    estate[5] = eulang[3];
    Ok(super::euler::eul2xf(&estate, 3, 1, 3))
}
