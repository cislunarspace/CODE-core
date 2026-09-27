//! SPK Type 2 段求值（纯 std）。
//!
//! 行为地面真值：vendored CSPICE 的 `spkr02.c`（记录定位）与 `spke02.c` +
//! `chbint.c`（Chebyshev–Clenshaw 求值）。递推**逐操作移植**，操作顺序不得
//! 重排——逐位一致（`f64::to_bits` 相等）依赖浮点运算的精确顺序。
//!
//! Type 2 段布局（DAF 字地址区间 `[baddr, eaddr]`）：
//!
//! ```text
//! [逻辑记录 0] [逻辑记录 1] ... [逻辑记录 N-1] [INIT] [INTLEN] [RSIZE] [N]
//! ```
//!
//! 每条逻辑记录：`[MID, RADIUS, x 系数×ncof, y 系数×ncof, z 系数×ncof]`，
//! `ncof = (RSIZE - 2)/3`。段尾 4 个字为 `[INIT, INTLEN, RSIZE, N]`。
//! `recno = (integer)((et - INIT)/INTLEN) + 1`，仅做 `min(recno, N)` 钳位
//! （CSPICE `spkr02.c` 原样；越界由选段层 `dc0 <= et <= dc1` 拦截）。

use super::daf::{read_words, word_at, DafSegment};
use super::DafSpkError;

/// Chebyshev 多项式值与一阶导（`chbint.c` 逐操作移植）。
///
/// `cp[0..ncof]` 为系数（`degp = ncof - 1` 次多项式），`x2s = [MID, RADIUS]`
/// 为域变换参数，`x` 为求值时刻（ET 秒）。返回 `(p, dp/dx)`。
fn chbint(cp: &[f64], degp: usize, x2s: [f64; 2], x: f64) -> (f64, f64) {
    let s = (x - x2s[0]) / x2s[1];
    let s2 = s * 2.;
    let mut w = [0.0f64; 3];
    let mut dw = [0.0f64; 3];
    let mut j = degp + 1;
    while j > 1 {
        w[2] = w[1];
        w[1] = w[0];
        w[0] = cp[j - 1] + (s2 * w[1] - w[2]);
        dw[2] = dw[1];
        dw[1] = dw[0];
        dw[0] = w[1] * 2. + dw[1] * s2 - dw[2];
        j -= 1;
    }
    let p = cp[0] + (s * w[0] - w[1]);
    let mut dpdx = w[0] + s * dw[0] - dw[1];
    dpdx /= x2s[1];
    (p, dpdx)
}

/// 求 Type 2 段在 `et`（TDB 秒）时刻的状态 `[x, y, z, vx, vy, vz]`
/// （km, km/s）。
pub fn evaluate(bytes: &[u8], seg: &DafSegment, et: f64) -> Result<[f64; 6], DafSpkError> {
    let (baddr, eaddr) = seg.data_words;

    // 段尾 4 字：INIT / INTLEN / RSIZE / N（spkr02.c：dafgda(end-3, end)）。
    if eaddr < baddr + 4 {
        return Err(DafSpkError::BadFormat(format!(
            "Type 2 段数据区 [{baddr},{eaddr}] 不足以容纳段尾常数"
        )));
    }
    let init = word_at(bytes, eaddr - 3)?;
    let intlen = word_at(bytes, eaddr - 2)?;
    let rsize = word_at(bytes, eaddr - 1)? as i64;
    let nrec = word_at(bytes, eaddr)? as i64;
    if intlen <= 0.0 || rsize < 5 || nrec < 1 {
        return Err(DafSpkError::BadFormat(format!(
            "Type 2 段尾常数非法：INTLEN={intlen} RSIZE={rsize} N={nrec}"
        )));
    }

    // recno = (integer)((et - INIT) / INTLEN) + 1，仅 min 钳位（spkr02.c 原样：
    // Fortran INT() 向零截断；et ≥ INIT 时与 floor 一致，覆盖性由选段层保证）。
    let recno = (((et - init) / intlen) as i64 + 1).min(nrec);
    let recadr = (recno - 1)
        .checked_mul(rsize)
        .and_then(|o| o.checked_add(baddr as i64));
    let recadr = match recadr {
        Some(a) if a >= baddr as i64 => a as usize,
        _ => {
            return Err(DafSpkError::BadFormat(format!(
                "Type 2 记录号 {recno} 定位越界（recadr 计算溢出）"
            )))
        }
    };
    let rec_end = recadr + rsize as usize - 1;
    if rec_end > eaddr - 4 {
        return Err(DafSpkError::BadFormat(format!(
            "Type 2 记录 {recno}（地址 [{recadr},{rec_end}]）越出段数据区 [{baddr},{}]",
            eaddr - 4
        )));
    }

    // 记录 = [MID, RADIUS, x 系数, y 系数, z 系数]（spke02.c：ncof=(RSIZE-2)/3，
    // 系数从记录第 4 字起连续排布；x2s=[MID, RADIUS]）。
    let record = read_words(bytes, recadr, rec_end)?;
    let ncof = ((rsize - 2) / 3) as usize;
    if ncof < 1 || 2 + 3 * ncof != rsize as usize {
        return Err(DafSpkError::BadFormat(format!(
            "Type 2 记录系数数 NCOF={ncof} 与 RSIZE={rsize} 不自洽"
        )));
    }
    let mid = record[0];
    let radius = record[1];
    if radius <= 0.0 {
        return Err(DafSpkError::BadFormat(format!(
            "Type 2 记录区间半径必须为正，得到 RADIUS={radius}"
        )));
    }
    let degp = ncof - 1;
    let x2s = [mid, radius];
    let mut state = [0.0f64; 6];
    for axis in 0..3 {
        let cp = &record[2 + ncof * axis..2 + ncof * (axis + 1)];
        let (p, dpdx) = chbint(cp, degp, x2s, et);
        state[axis] = p;
        state[axis + 3] = dpdx;
    }
    Ok(state)
}
