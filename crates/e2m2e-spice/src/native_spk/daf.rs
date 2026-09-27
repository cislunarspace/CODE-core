//! DAF 容器解析（纯 std，无 CSPICE 依赖）。
//!
//! 行为地面真值：vendored CSPICE 的 `dafrfr_c.c`/`dafgda.c`/`dafus.c`。
//! 两个易错规格点（见 ADR 0051）：
//!
//! 1. `FWARD`/`BWARD`/摘要记录控制字 `next`/`prev` 是 **Fortran 1 基记录号**，
//!    不是 DAF 字地址；换算 `字地址 = (记录号-1)*128 + 1`（字节偏移
//!    `(记录号-1)*1024`）。只有 `FREE` 本身是字地址。
//! 2. 整数分量按 2 个 i32 打包进 1 个 f64、低字在前：读出 f64 后按位重解释，
//!    低 32 位是序号小的整数分量。
//!
//! 全部多字节量按小端（`LOCFMT == "LTL…"`）读取，与 NAIF LTLIEEE 布局一致。

use std::path::{Path, PathBuf};

use super::DafSpkError;

/// DAF 逻辑记录的固定字长（双精度字数）；一个记录 1024 字节。
const WORDS_PER_RECORD: usize = 128;

fn read_f64(bytes: &[u8], byte_off: usize) -> Option<f64> {
    let end = byte_off.checked_add(8)?;
    if end > bytes.len() {
        return None;
    }
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[byte_off..end]);
    Some(f64::from_le_bytes(buf))
}

fn read_i32(bytes: &[u8], byte_off: usize) -> Option<i32> {
    let end = byte_off.checked_add(4)?;
    if end > bytes.len() {
        return None;
    }
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&bytes[byte_off..end]);
    Some(i32::from_le_bytes(buf))
}

/// 读一个 DAF 字（f64）并按低字在前规则取第 `half` 个整数分量（0=低 32 位，
/// 1=高 32 位）。
fn word_int(bytes: &[u8], byte_off: usize, half: usize) -> Option<i32> {
    let bits = read_f64(bytes, byte_off)?.to_bits();
    let v = if half == 0 {
        bits as u32
    } else {
        (bits >> 32) as u32
    };
    Some(v as i32)
}

/// 正确的记录号→字节偏移映射：`字地址 = (记录号-1)*128 + 1`。
fn record_byte_offset(recno: i64) -> Option<usize> {
    if recno <= 0 {
        return None;
    }
    (recno as usize)
        .checked_sub(1)?
        .checked_mul(WORDS_PER_RECORD * 8)
}

/// 负控映射：把记录号**当字地址**解释（字节偏移 = (记录号-1)*8）——
/// 原型首版的真实 bug 形态。
fn word_address_byte_offset(w: i64) -> Option<usize> {
    if w <= 0 {
        return None;
    }
    (w as usize - 1).checked_mul(8)
}

/// 解析出的 DAF 段描述符。
///
/// SPK（`NI == 6`）有 `center`；BPC（`NI == 5`）无 `center`，`center: None`
/// （BPC 段只解析描述符供对拍，不参与 `state()` 选段）。
///
/// `data_words` 是段数据区的 `(起始字地址, 结束字地址)`，即 DAF 摘要整数分量
/// 里的 `[begin_addr, end_addr]`（1 基、含端点，与 `dafgda` 语义一致）。
#[derive(Debug, Clone, PartialEq)]
pub struct DafSegment {
    pub body: i32,
    pub center: Option<i32>,
    pub frame: i32,
    pub dtype: i32,
    /// 段覆盖区间起点（ET 秒）。
    pub begin: f64,
    /// 段覆盖区间终点（ET 秒）。
    pub end: f64,
    /// DC(1)，对 SPK/BPC 即 `begin`（按位保留，供对拍）。
    pub dc0: f64,
    /// DC(2)，即 `end`。
    pub dc1: f64,
    /// (段数据起始字地址, 段数据结束字地址)，1 基含端点。
    pub data_words: (usize, usize),
}

/// 解析完成的 DAF 文件：文件记录头 + 全部段描述符。
#[derive(Debug, Clone)]
pub struct DafFile {
    pub path: PathBuf,
    pub nd: usize,
    pub ni: usize,
    pub segments: Vec<DafSegment>,
}

/// 解析 DAF 容器（SPK/BPC）。整文件读入 `bytes`，零拷贝切片读取。
pub fn parse(path: &Path, bytes: &[u8]) -> Result<DafFile, DafSpkError> {
    parse_map(path, bytes, record_byte_offset)
}

/// [`parse`] 的负控变体：把 `FWARD`/`next` 记录号统一偏移 `record_base` 个
/// 记录再取字节（默认 0 = 正确行为；非 0 仅供负控测试，模拟记录号/字地址
/// 混淆类错误的可检测性，见 ADR 0051 验证节）。
#[doc(hidden)]
pub fn parse_with(path: &Path, bytes: &[u8], record_base: usize) -> Result<DafFile, DafSpkError> {
    parse_map(path, bytes, move |rec| {
        record_byte_offset(rec)
            .and_then(|off| off.checked_add(record_base.checked_mul(WORDS_PER_RECORD * 8)?))
    })
}

/// [`parse`] 的负控变体：把 `FWARD`/`next` 记录号**按字地址解释**
/// （字节偏移 = (记录号-1)*8）。对 de440s 这类内核会踩进文件记录得到垃圾
/// 控制字，解析必须失败或得到错误段表，不得静默返回合法结果。
#[doc(hidden)]
pub fn parse_word_address(path: &Path, bytes: &[u8]) -> Result<DafFile, DafSpkError> {
    parse_map(path, bytes, word_address_byte_offset)
}

fn parse_map(
    path: &Path,
    bytes: &[u8],
    offset_of_record: impl Fn(i64) -> Option<usize>,
) -> Result<DafFile, DafSpkError> {
    let header = FileHeader::parse(path, bytes)?;

    // 摘要宽度 SS = ND + ⌈NI/2⌉；一条记录至多容纳 125 个摘要字。
    let ss = header.nd + header.ni.div_ceil(2);
    if ss == 0 || ss > WORDS_PER_RECORD - 3 {
        return Err(DafSpkError::BadFormat(format!(
            "{}: 摘要宽度 ND+⌈NI/2⌉={} 越界",
            path.display(),
            ss
        )));
    }

    let max_sum_per_record = (WORDS_PER_RECORD - 3) / ss;

    // 摘要记录链：从 FWARD 沿 next 走到 next == 0；段数逐记录用 nsum。
    let mut segments = Vec::new();
    let mut rec = header.fward;
    let max_records = bytes.len() / (WORDS_PER_RECORD * 8) + 1;
    for _ in 0..=max_records {
        let base = offset_of_record(rec).ok_or_else(|| {
            DafSpkError::BadFormat(format!(
                "{}: 摘要记录号 {rec} 非法（FWARD={}）",
                path.display(),
                header.fward
            ))
        })?;
        if base + WORDS_PER_RECORD * 8 > bytes.len() {
            return Err(DafSpkError::BadFormat(format!(
                "{}: 摘要记录 {rec} 越出文件（文件 {} B）",
                path.display(),
                bytes.len()
            )));
        }
        // 控制区：3 个 double（next / prev / nsum）。
        let next = read_f64(bytes, base)
            .ok_or_else(|| DafSpkError::BadFormat(format!("{}: 读 next 失败", path.display())))?
            as i64;
        let nsum = read_f64(bytes, base + 16)
            .ok_or_else(|| DafSpkError::BadFormat(format!("{}: 读 nsum 失败", path.display())))?
            as i64;
        if nsum < 0 || nsum as usize > max_sum_per_record {
            return Err(DafSpkError::BadFormat(format!(
                "{}: 摘要记录 {rec} 的 nsum={nsum} 越界（每记录至多 {max_sum_per_record}）",
                path.display()
            )));
        }
        // 摘要区从记录内第 24 字节（字 4）起，逐条 nd 个 double + 打包整数。
        for k in 0..nsum {
            let sum_off = base + 24 + (k as usize) * ss * 8;
            let mut dc = [0.0f64; 2];
            for (d, slot) in dc.iter_mut().enumerate() {
                *slot = read_f64(bytes, sum_off + d * 8).ok_or_else(|| {
                    DafSpkError::BadFormat(format!(
                        "{}: 记录 {rec} 段 {k} DC 读取越界",
                        path.display()
                    ))
                })?;
            }
            let ic_off = sum_off + header.nd * 8;
            let mut ic = [0i32; 6];
            for (i, slot) in ic.iter_mut().enumerate() {
                *slot = word_int(bytes, ic_off + (i / 2) * 8, i % 2).ok_or_else(|| {
                    DafSpkError::BadFormat(format!(
                        "{}: 记录 {rec} 段 {k} IC 读取越界",
                        path.display()
                    ))
                })?;
            }
            let seg = header.build_segment(dc, &ic).ok_or_else(|| {
                DafSpkError::BadFormat(format!(
                    "{}: 不支持的摘要布局 ND={} NI={}",
                    path.display(),
                    header.nd,
                    header.ni
                ))
            })?;
            // 数据区地址必须落在文件内。
            let (baddr, eaddr) = seg.data_words;
            let end_ok = word_byte_offset(eaddr).is_some_and(|o| o + 8 <= bytes.len());
            if baddr == 0 || eaddr < baddr || !end_ok {
                return Err(DafSpkError::BadFormat(format!(
                    "{}: 段数据地址 [{baddr},{eaddr}] 越出文件",
                    path.display()
                )));
            }
            segments.push(seg);
        }
        if next == 0 {
            return Ok(DafFile {
                path: path.to_path_buf(),
                nd: header.nd,
                ni: header.ni,
                segments,
            });
        }
        rec = next;
    }
    Err(DafSpkError::BadFormat(format!(
        "{}: 摘要记录链超过文件记录数（FWARD={} 处 next 链成环或越界）",
        path.display(),
        header.fward
    )))
}

fn word_byte_offset(addr: usize) -> Option<usize> {
    addr.checked_sub(1)?.checked_mul(8)
}

/// DAF 文件记录头（记录 1）。
struct FileHeader {
    nd: usize,
    ni: usize,
    fward: i64,
}

impl FileHeader {
    fn parse(path: &Path, bytes: &[u8]) -> Result<Self, DafSpkError> {
        if bytes.len() < WORDS_PER_RECORD * 8 {
            return Err(DafSpkError::BadFormat(format!(
                "{}: 文件 {} B 不足一个 DAF 记录",
                path.display(),
                bytes.len()
            )));
        }
        // locidw（字节 0..8）："DAF/" 前缀；非 DAF 容器（文本内核等）→ NotDaf，
        // 供 furnish 层区分文本内核跳过。
        if bytes.len() < 4 || &bytes[0..4] != b"DAF/" {
            return Err(DafSpkError::NotDaf);
        }
        // locfmt（字节 88..96）：仅支持 LTL（小端 IEEE）。
        if &bytes[88..91] != b"LTL" {
            let fmt = String::from_utf8_lossy(&bytes[88..96]).to_string();
            return Err(DafSpkError::BadFormat(format!(
                "{}: 仅支持 LTL（小端）字节序，LOCFMT={fmt:?}",
                path.display()
            )));
        }
        let nd = read_i32(bytes, 8)
            .ok_or_else(|| DafSpkError::BadFormat(format!("{}: 读 ND 失败", path.display())))?;
        let ni = read_i32(bytes, 12)
            .ok_or_else(|| DafSpkError::BadFormat(format!("{}: 读 NI 失败", path.display())))?;
        let fward = read_i32(bytes, 76)
            .ok_or_else(|| DafSpkError::BadFormat(format!("{}: 读 FWARD 失败", path.display())))?;
        if nd != 2 || !matches!(ni, 5 | 6) {
            return Err(DafSpkError::BadFormat(format!(
                "{}: 不支持的摘要布局 ND={nd} NI={ni}（仅 SPK ND=2/NI=6 与 BPC ND=2/NI=5）",
                path.display()
            )));
        }
        Ok(Self {
            nd: nd as usize,
            ni: ni as usize,
            fward: fward as i64,
        })
    }

    /// 由 DC/IC 组装段描述符。支持 SPK（NI=6：body/center/frame/type/baddr/eaddr）
    /// 与 BPC（NI=5：body/frame/type/baddr/eaddr，无 center）。
    fn build_segment(&self, dc: [f64; 2], ic: &[i32; 6]) -> Option<DafSegment> {
        let (body, center, frame, dtype, baddr, eaddr) = match self.ni {
            6 => (ic[0], Some(ic[1]), ic[2], ic[3], ic[4], ic[5]),
            5 => (ic[0], None, ic[1], ic[2], ic[3], ic[4]),
            _ => return None,
        };
        if baddr < 0 || eaddr < 0 {
            return None;
        }
        Some(DafSegment {
            body,
            center,
            frame,
            dtype,
            begin: dc[0],
            end: dc[1],
            dc0: dc[0],
            dc1: dc[1],
            data_words: (baddr as usize, eaddr as usize),
        })
    }
}

/// 读单个 DAF 字（1 基字地址 → f64），越界即 BadFormat。
pub(crate) fn word_at(bytes: &[u8], addr: usize) -> Result<f64, DafSpkError> {
    let off = addr
        .checked_sub(1)
        .and_then(|a| a.checked_mul(8))
        .ok_or_else(|| DafSpkError::BadFormat(format!("DAF 字地址 {addr} 非法")))?;
    if off + 8 > bytes.len() {
        return Err(DafSpkError::BadFormat(format!(
            "DAF 字地址 {addr} 越出文件（{} B）",
            bytes.len()
        )));
    }
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[off..off + 8]);
    Ok(f64::from_le_bytes(buf))
}

/// 读 DAF 字地址区间 `[baddr, eaddr]` 的 f64 数据（1 基、含端点），语义对齐
/// `dafgda`。供 [`super::spk2`] 求值使用。
pub(crate) fn read_words(
    bytes: &[u8],
    baddr: usize,
    eaddr: usize,
) -> Result<Vec<f64>, DafSpkError> {
    if baddr == 0 || eaddr < baddr {
        return Err(DafSpkError::BadFormat(format!(
            "非法 DAF 字地址区间 [{baddr},{eaddr}]"
        )));
    }
    let start = word_byte_offset(baddr)
        .ok_or_else(|| DafSpkError::BadFormat(format!("DAF 字地址 {baddr} 越界")))?;
    let end = word_byte_offset(eaddr)
        .ok_or_else(|| DafSpkError::BadFormat(format!("DAF 字地址 {eaddr} 越界")))?
        + 8;
    if end > bytes.len() {
        return Err(DafSpkError::BadFormat(format!(
            "DAF 字地址区间 [{baddr},{eaddr}] 越出文件（{} B）",
            bytes.len()
        )));
    }
    let n = eaddr - baddr + 1;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&bytes[start + i * 8..start + i * 8 + 8]);
        out.push(f64::from_le_bytes(buf));
    }
    Ok(out)
}
