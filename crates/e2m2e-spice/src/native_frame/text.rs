//! 文本内核解析（FK `.tf` / 文本 PCK `.tpc` / LSK `.tls`，纯 std）。
//!
//! 一个解析器三种消费者：先把 `\begindata` 区段解析成关键字 → 值表的
//! [`KernelPool`]（未知关键字**忽略不报错**——`.tpc` 里有 RADII 等大量
//! 无关量），再由 [`FkFile`] / [`TpckFile`] / [`Lsk`] 按各自关键字集解释。
//!
//! 语法口径（kernel.req 子集，仓库三类内核全部覆盖）：
//! - `KEYWORD = ( v1 v2 ... )`（括号可跨行）或 `KEYWORD = v`；
//! - 值：带引号字符串（`''` 转义引号）、数字（含 Fortran `D` 指数，
//!   如 `1.657D-3`）、`@1972-JAN-1` 日期记号（→ J2000 起算秒，与 CSPICE
//!   内核池装载数值一致：`(JD - JD(J2000)) · 86400`，实测钉死）；
//! - 逗号视为分隔符（`AXES = ( 3, 2, 1 )`）；
//! - `\begintext` 切回注释、`\begindata` 切回数据；缺 `)`、非法数字等
//!   解析失败 → 硬错误。
//!
//! 值域转换失败的语义错误（类别未知、UNITS 不支持等）由消费者报出，
//! 不在本模块。

use std::collections::HashMap;

/// 文本内核解析错误（硬错误）。
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError(pub String);

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "文本内核解析失败: {}", self.0)
    }
}

impl std::error::Error for ParseError {}

/// 内核池值。
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// 带引号字符串（去引号后的原文，保留大小写）。
    Str(String),
    /// 数字（含 `D` 指数与 `@date` 记号，均已换算成 f64）。
    Num(f64),
}

impl Value {
    pub fn as_num(&self) -> Option<f64> {
        match self {
            Value::Num(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// 解析出的内核池：关键字（大写）→ 值序列（同关键字多次赋值按 CSPICE
/// 内核池语义追加）。
#[derive(Debug, Clone, Default)]
pub struct KernelPool {
    entries: HashMap<String, Vec<Value>>,
}

impl KernelPool {
    pub fn get(&self, key: &str) -> Option<&Vec<Value>> {
        self.entries.get(&key.to_ascii_uppercase())
    }

    pub fn nums(&self, key: &str) -> Option<Vec<f64>> {
        self.get(key).map(|v| {
            // 值序列按关键字用途保证全数字（消费者负责语义校验）；
            // 这里惰性过滤，避免为无关量多分配。
            v.iter().filter_map(Value::as_num).collect::<Vec<_>>()
        })
    }

    fn insert(&mut self, key: String, values: Vec<Value>) {
        self.entries.entry(key).or_default().extend(values);
    }
}

/// 民用日期 → J2000（JD 2451545.0）起算的秒数。`@date` 记号的内核池
/// 装载口径：`(JD(0时) - 2451545.0) · 86400`，时间部分按 UTC 秒追加
/// （本仓内核只用 0 点日期）。
fn date_token_to_seconds(year: i32, month: u32, day: u32, sec_of_day: f64) -> f64 {
    let days = crate::spice_ffi::days_from_civil(year, month, day) as f64;
    (days + 2440587.5 - 2451545.0) * 86400.0 + sec_of_day
}

const MONTHS: [&str; 12] = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
];

/// 解析 `@YYYY-MON-DD[THH:MM:SS]` 记号 → f64。
fn parse_date_token(token: &str) -> Result<f64, ParseError> {
    let body = token
        .strip_prefix('@')
        .ok_or_else(|| ParseError(format!("非法日期记号: {token:?}")))?;
    let (date, time) = match body.split_once('T') {
        Some((d, t)) => (d, Some(t)),
        None => (body, None),
    };
    let parts: Vec<&str> = date.split('-').collect();
    if parts.len() != 3 {
        return Err(ParseError(format!("非法日期记号: {token:?}")));
    }
    let year: i32 = parts[0]
        .parse()
        .map_err(|_| ParseError(format!("非法年份: {token:?}")))?;
    let mon_upper = parts[1].to_ascii_uppercase();
    let month = MONTHS
        .iter()
        .position(|m| *m == mon_upper)
        .map(|i| i as u32 + 1)
        .ok_or_else(|| ParseError(format!("非法月份: {token:?}")))?;
    let day: u32 = parts[2]
        .parse()
        .map_err(|_| ParseError(format!("非法日: {token:?}")))?;
    let mut sec_of_day = 0.0_f64;
    if let Some(t) = time {
        let tparts: Vec<&str> = t.split(':').collect();
        if tparts.len() != 3 {
            return Err(ParseError(format!("非法日期记号时间部分: {token:?}")));
        }
        let h: f64 = tparts[0]
            .parse()
            .map_err(|_| ParseError(format!("非法时: {token:?}")))?;
        let m: f64 = tparts[1]
            .parse()
            .map_err(|_| ParseError(format!("非法分: {token:?}")))?;
        let s: f64 = tparts[2]
            .parse()
            .map_err(|_| ParseError(format!("非法秒: {token:?}")))?;
        sec_of_day = h * 3600.0 + m * 60.0 + s;
    }
    Ok(date_token_to_seconds(year, month, day, sec_of_day))
}

/// 解析数字字面量（含 Fortran `D` 指数）。
fn parse_number(token: &str) -> Result<f64, ParseError> {
    let normalized = token.replace(['D', 'd'], "E");
    normalized
        .parse::<f64>()
        .map_err(|_| ParseError(format!("非法数字: {token:?}")))
}

/// 解析一行内的值序列。`*in_list` 为 true 表示在带括号列表内（遇 `)`
/// 置回 false 并返回）；字符串 `''` 转义；`@date` 与数字按前缀分派。
fn consume_values(
    text: &str,
    vals: &mut Vec<Value>,
    in_list: &mut bool,
    lineno: usize,
) -> Result<(), ParseError> {
    let mut chars = text.char_indices().peekable();
    while let Some((idx, ch)) = chars.next() {
        match ch {
            ' ' | '\t' | ',' => continue,
            ')' => {
                if !*in_list {
                    return Err(ParseError(format!(
                        "第 {lineno} 行出现不匹配的 ')': {}",
                        &text[idx..]
                    )));
                }
                // 列表闭合；闭合状态写回调用方，行尾剩余内容忽略。
                *in_list = false;
                return Ok(());
            }
            '(' => {
                return Err(ParseError(format!(
                    "第 {lineno} 行出现不匹配的 '(': {}",
                    &text[idx..]
                )));
            }
            '\'' => {
                // 带引号字符串，'' 转义。
                let mut s = String::new();
                let mut closed = false;
                while let Some((_, c)) = chars.next() {
                    if c == '\'' {
                        if chars.peek().map(|(_, n)| *n) == Some('\'') {
                            chars.next();
                            s.push('\'');
                        } else {
                            closed = true;
                            break;
                        }
                    } else {
                        s.push(c);
                    }
                }
                if !closed {
                    return Err(ParseError(format!("第 {lineno} 行字符串未闭合: {text:?}")));
                }
                vals.push(Value::Str(s));
            }
            '@' => {
                // 日期记号：取到下一个分隔符。
                let start = idx;
                let mut end = text.len();
                for (j, c) in chars.by_ref() {
                    if c == ' ' || c == '\t' || c == ',' || c == ')' {
                        end = j;
                        break;
                    }
                }
                vals.push(Value::Num(parse_date_token(&text[start..end])?));
            }
            _ => {
                // 数字：取到下一个分隔符。
                let start = idx;
                let mut end = text.len();
                for (j, c) in chars.by_ref() {
                    if c == ' ' || c == '\t' || c == ',' || c == ')' || c == '(' {
                        end = j;
                        break;
                    }
                }
                vals.push(Value::Num(parse_number(&text[start..end])?));
            }
        }
    }
    Ok(())
}

/// 解析整文件成 [`KernelPool`]。只解析 `\begindata` 区段；`\begintext`
/// 区段与文件头注释跳过；未知关键字忽略；断句失败 → 硬错误。
pub fn parse_pool(content: &str) -> Result<KernelPool, ParseError> {
    let mut pool = KernelPool::default();
    let mut in_data = false;
    // 待续语句：Some(关键字) 表示「= 之后已见」；`in_list` 表示在带括号
    // 列表内（可能跨行）。
    let mut pending_key: Option<String> = None;
    let mut pending_vals: Vec<Value> = Vec::new();
    let mut in_list = false;

    for (lineno, raw_line) in content.lines().enumerate() {
        let line = raw_line.trim();
        // 区段记号必须独占一行（kernel.req 口径；pck00010 的注释区有
        // "\begindata token." 一类行内提及，不得误触发）。
        let lower = line.to_ascii_lowercase();
        if lower == "\\begintext" {
            in_data = false;
            continue;
        }
        if lower == "\\begindata" {
            in_data = true;
            continue;
        }
        if !in_data || line.is_empty() {
            continue;
        }

        if pending_key.is_none() {
            // 新语句：必须以 KEYWORD = 开头。
            let Some((key, rest)) = line.split_once('=') else {
                return Err(ParseError(format!("第 {lineno} 行缺 '=': {line:?}")));
            };
            let key = key.trim().to_ascii_uppercase();
            if key.is_empty() {
                return Err(ParseError(format!("第 {lineno} 行关键字为空: {line:?}")));
            }
            pending_key = Some(key);
            let rest = rest.trim_start();
            if let Some(stripped) = rest.strip_prefix('(') {
                in_list = true;
                consume_values(stripped, &mut pending_vals, &mut in_list, lineno)?;
            } else {
                consume_values(rest, &mut pending_vals, &mut in_list, lineno)?;
            }
        } else {
            // 带括号列表的续行。
            consume_values(line, &mut pending_vals, &mut in_list, lineno)?;
        }

        if !in_list {
            let key = pending_key
                .take()
                .ok_or_else(|| ParseError(format!("第 {lineno} 行语句状态不一致")))?;
            pool.insert(key, std::mem::take(&mut pending_vals));
        }
    }
    if pending_key.is_some() || in_list {
        return Err(ParseError("语句在文件结束前未闭合（缺 ')'）".into()));
    }
    Ok(pool)
}

// ---------------------------------------------------------------------------
// 文件分类
// ---------------------------------------------------------------------------

/// 按内容分类的文本内核种类。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextKind {
    Fk,
    Tpck,
    Lsk,
}

/// 按内容分类文件类型（furnish 分派用）：首个非空行前缀 `KPL/FK` /
/// `KPL/PCK` / `KPL/LSK` → 对应文本池；其余文本 → `None`
/// （native 侧跳过，CSPICE 侧照常校验）。
pub fn classify(path: &std::path::Path, content: &str) -> Option<TextKind> {
    let _ = path;
    let head = content
        .lines()
        .map(str::trim_start)
        .find(|l| !l.is_empty())
        .unwrap_or("");
    let upper: String = head
        .chars()
        .take(8)
        .collect::<String>()
        .to_ascii_uppercase();
    if upper.starts_with("KPL/FK") {
        Some(TextKind::Fk)
    } else if upper.starts_with("KPL/PCK") {
        Some(TextKind::Tpck)
    } else if upper.starts_with("KPL/LSK") {
        Some(TextKind::Lsk)
    } else {
        None
    }
}

/// 文件是否为 DAF 二进制容器（首 8 字节 `DAF/`）。
pub fn is_daf(bytes: &[u8]) -> bool {
    bytes.len() >= 8 && &bytes[..4] == b"DAF/"
}

// ---------------------------------------------------------------------------
// FK（frames kernel）
// ---------------------------------------------------------------------------

/// TK 帧规格（class 4）。
#[derive(Debug, Clone, PartialEq)]
pub enum TkSpec {
    /// `TKFRAME_<id>_MATRIX`：矩阵按文本行序 = `X(tk → relative)` 的行
    /// （tkfram.c 原样直取）。
    Matrix { relative: String, m: [[f64; 3]; 3] },
    /// `TKFRAME_<id>_ANGLES`：角度 + 轴序 + 单位；`eul2m` 因子化。
    Angles {
        relative: String,
        angles_rad: [f64; 3],
        axes: [i32; 3],
    },
}

/// FK 帧定义。
#[derive(Debug, Clone, PartialEq)]
pub struct FrameDef {
    pub class: i32,
    pub class_id: i32,
    pub center: i32,
    /// class 4 专有。
    pub tk: Option<TkSpec>,
}

/// 解析后的 FK：帧 id → 定义，名字（大写）→ id。
#[derive(Debug, Clone, Default)]
pub struct FkFile {
    pub frames: HashMap<i32, FrameDef>,
    pub name_to_id: HashMap<String, i32>,
}

impl FkFile {
    pub fn from_pool(pool: &KernelPool) -> Result<FkFile, ParseError> {
        let mut fk = FkFile::default();
        // FRAME_<name> = id（名字键，FRAME_ 后为非数字）。
        // FRAME_<id>_NAME = 'name'（id 键）。
        for (key, values) in pool.entries.iter() {
            if let Some(rest) = key.strip_prefix("FRAME_") {
                if rest.chars().next().is_some_and(|c| c.is_ascii_digit()) {
                    continue; // FRAME_<id>_... 由下面的 id 通道处理
                }
                let id = values
                    .first()
                    .and_then(Value::as_num)
                    .ok_or_else(|| ParseError(format!("FRAME_{rest} 值非法")))?;
                let name = rest.to_string();
                fk.name_to_id.insert(name, id as i32);
            }
        }
        // 收集 id 键：FRAME_<id>_NAME / _CLASS / _CLASS_ID / _CENTER。
        let mut ids: Vec<i32> = Vec::new();
        for key in pool.entries.keys() {
            if let Some(rest) = key.strip_prefix("FRAME_") {
                if let Some((id_str, suffix)) = rest.split_once('_') {
                    if let Ok(id) = id_str.parse::<i32>() {
                        ids.push(id);
                        let _ = suffix;
                    }
                }
            }
        }
        ids.sort_unstable();
        ids.dedup();
        for id in ids {
            let idk = id.to_string();
            let class = required_num(pool, &format!("FRAME_{idk}_CLASS"))? as i32;
            let class_id = required_num(pool, &format!("FRAME_{idk}_CLASS_ID"))? as i32;
            let center = required_num(pool, &format!("FRAME_{idk}_CENTER"))? as i32;
            let name = pool
                .get(&format!("FRAME_{idk}_NAME"))
                .and_then(|v| v.first())
                .and_then(Value::as_str)
                .map(str::to_string);
            if let Some(name) = name {
                fk.name_to_id.insert(name.to_ascii_uppercase(), id);
            }
            // class 4：TK 规格。
            let tk = if class == 4 {
                let spec = pool
                    .get(&format!("TKFRAME_{idk}_SPEC"))
                    .and_then(|v| v.first())
                    .and_then(Value::as_str)
                    .map(str::to_ascii_uppercase);
                let relative = pool
                    .get(&format!("TKFRAME_{idk}_RELATIVE"))
                    .and_then(|v| v.first())
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .ok_or_else(|| ParseError(format!("TKFRAME_{id} 缺 RELATIVE")))?;
                match spec.as_deref() {
                    Some("MATRIX") => {
                        let flat = required_nums(pool, &format!("TKFRAME_{idk}_MATRIX"), 9)?;
                        let mut m = [[0.0_f64; 3]; 3];
                        for (i, row) in m.iter_mut().enumerate() {
                            row.copy_from_slice(&flat[i * 3..i * 3 + 3]);
                        }
                        Some(TkSpec::Matrix { relative, m })
                    }
                    Some("ANGLES") => {
                        let angles = required_nums(pool, &format!("TKFRAME_{idk}_ANGLES"), 3)?;
                        let axes_vals = required_nums(pool, &format!("TKFRAME_{idk}_AXES"), 3)?;
                        let units = pool
                            .get(&format!("TKFRAME_{idk}_UNITS"))
                            .and_then(|v| v.first())
                            .and_then(Value::as_str)
                            .unwrap_or("RADIANS")
                            .to_ascii_uppercase();
                        let factor = match units.as_str() {
                            "ARCSECONDS" => std::f64::consts::PI / 648000.0,
                            "DEGREES" => std::f64::consts::PI / 180.0,
                            "RADIANS" => 1.0,
                            other => {
                                return Err(ParseError(format!(
                                    "NATIVE_FRAME_UNSUPPORTED_UNITS: {other}"
                                )));
                            }
                        };
                        let mut angles_rad = [0.0_f64; 3];
                        for (a, v) in angles_rad.iter_mut().zip(angles) {
                            *a = v * factor;
                        }
                        let mut axes = [0_i32; 3];
                        for (a, v) in axes.iter_mut().zip(axes_vals) {
                            *a = v as i32;
                        }
                        Some(TkSpec::Angles {
                            relative,
                            angles_rad,
                            axes,
                        })
                    }
                    other => {
                        return Err(ParseError(format!("TKFRAME_{id} SPEC 不支持: {other:?}")));
                    }
                }
            } else {
                None
            };
            fk.frames.insert(
                id,
                FrameDef {
                    class,
                    class_id,
                    center,
                    tk,
                },
            );
        }
        Ok(fk)
    }
}

fn required_num(pool: &KernelPool, key: &str) -> Result<f64, ParseError> {
    pool.get(key)
        .and_then(|v| v.first())
        .and_then(Value::as_num)
        .ok_or_else(|| ParseError(format!("缺关键字 {key} 或值非数字")))
}

fn required_nums(pool: &KernelPool, key: &str, n: usize) -> Result<Vec<f64>, ParseError> {
    let vals = pool
        .nums(key)
        .ok_or_else(|| ParseError(format!("缺关键字 {key}")))?;
    if vals.len() < n {
        return Err(ParseError(format!("关键字 {key} 值不足 {n} 个")));
    }
    Ok(vals[..n].to_vec())
}

// ---------------------------------------------------------------------------
// 文本 PCK（IAU 多项式）
// ---------------------------------------------------------------------------

/// 单天体的 IAU 旋转多项式（BODY<id>_POLE_RA / POLE_DEC / PM，各 3 系数，
/// 缺项补 0）+ 可选 NUT_PREC 级数。
#[derive(Debug, Clone, Default)]
pub struct IauBody {
    pub ra: [f64; 3],
    pub dec: [f64; 3],
    pub pm: [f64; 3],
    pub nut_ra: Option<Vec<f64>>,
    pub nut_dec: Option<Vec<f64>>,
    pub nut_pm: Option<Vec<f64>>,
}

/// 解析后的文本 PCK。`nut_prec_angles` 以 **关键字标签体号** 为键
/// （`BODY3_NUT_PREC_ANGLES` → 3）：tisbod 按 `zzbodbry(body)`（百位
/// 收缩到系统质心）查该表。
#[derive(Debug, Clone, Default)]
pub struct TpckFile {
    pub bodies: HashMap<i32, IauBody>,
    pub nut_prec_angles: HashMap<i32, Vec<[f64; 2]>>,
}

impl TpckFile {
    pub fn from_pool(pool: &KernelPool) -> Result<TpckFile, ParseError> {
        let mut tpck = TpckFile::default();
        for (key, values) in pool.entries.iter() {
            let Some(rest) = key.strip_prefix("BODY") else {
                continue;
            };
            let Some((id_str, suffix)) = rest.split_once('_') else {
                continue;
            };
            let Ok(id) = id_str.parse::<i32>() else {
                continue;
            };
            match suffix {
                "POLE_RA" => poly3(pool, key).map(|p| tpck.bodies.entry(id).or_default().ra = p)?,
                "POLE_DEC" => {
                    poly3(pool, key).map(|p| tpck.bodies.entry(id).or_default().dec = p)?
                }
                "PM" => poly3(pool, key).map(|p| tpck.bodies.entry(id).or_default().pm = p)?,
                "NUT_PREC_RA" => {
                    tpck.bodies.entry(id).or_default().nut_ra = Some(nums_of(values));
                }
                "NUT_PREC_DEC" => {
                    tpck.bodies.entry(id).or_default().nut_dec = Some(nums_of(values));
                }
                "NUT_PREC_PM" => {
                    tpck.bodies.entry(id).or_default().nut_pm = Some(nums_of(values));
                }
                "NUT_PREC_ANGLES" => {
                    let flat = nums_of(values);
                    if !flat.len().is_multiple_of(2) {
                        return Err(ParseError(format!("{key} 值个数为奇数")));
                    }
                    let pairs: Vec<[f64; 2]> = flat
                        .chunks(2)
                        .map(|c| {
                            let mut p = [0.0_f64; 2];
                            p.copy_from_slice(c);
                            p
                        })
                        .collect();
                    tpck.nut_prec_angles.insert(id, pairs);
                }
                "CONSTANTS_JED_EPOCH"
                | "CONSTS_JED_EPOCH"
                | "CONSTANTS_REF_FRAME"
                | "CONSTS_REF_FRAME" => {
                    return Err(ParseError(format!(
                        "NATIVE_FRAME_UNSUPPORTED_PCK_EPOCH: {key} 不支持（本仓内核不含该键）"
                    )));
                }
                "MAX_PHASE_DEGREE" => {
                    return Err(ParseError(format!(
                        "NATIVE_FRAME_UNSUPPORTED_MAX_PHASE_DEGREE: {key} 不支持（本仓内核缺省相位角度数 1）"
                    )));
                }
                _ => {}
            }
        }
        Ok(tpck)
    }
}

/// 取 BODY<id>_POLE_* 三系数（不足补 0，bodvcd 前 cleard_ 语义）。
fn poly3(pool: &KernelPool, key: &str) -> Result<[f64; 3], ParseError> {
    let vals = pool.nums(key).unwrap_or_default();
    let mut p = [0.0_f64; 3];
    for (i, v) in vals.iter().take(3).enumerate() {
        p[i] = *v;
    }
    Ok(p)
}

fn nums_of(values: &[Value]) -> Vec<f64> {
    values.iter().filter_map(Value::as_num).collect()
}

// ---------------------------------------------------------------------------
// LSK（leapseconds kernel）
// ---------------------------------------------------------------------------

/// 解析后的 LSK。
#[derive(Debug, Clone)]
pub struct Lsk {
    pub delta_t_a: f64,
    pub k: f64,
    pub eb: f64,
    pub m: [f64; 2],
    /// `(J2000 起算秒, DELTA_AT)` 对，按装载顺序。
    pub leap: Vec<(f64, i64)>,
}

impl Lsk {
    pub fn from_pool(pool: &KernelPool) -> Result<Lsk, ParseError> {
        let dta = required_num(pool, "DELTET/DELTA_T_A")?;
        let k = required_num(pool, "DELTET/K")?;
        let eb = required_num(pool, "DELTET/EB")?;
        let m = required_nums(pool, "DELTET/M", 2)?;
        let flat = pool
            .nums("DELTET/DELTA_AT")
            .ok_or_else(|| ParseError("缺关键字 DELTET/DELTA_AT".into()))?;
        if flat.len() % 2 != 0 {
            return Err(ParseError("DELTET/DELTA_AT 值个数为奇数".into()));
        }
        let leap: Vec<(f64, i64)> = flat
            .chunks(2)
            // 内核列表顺序 = (DELTA_AT 值, @date 秒) 交替。
            .map(|c| (c[1], c[0] as i64))
            .collect();
        Ok(Lsk {
            delta_t_a: dta,
            k,
            eb,
            m: [m[0], m[1]],
            leap,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 三种 KPL 头的内容分类。
    #[test]
    fn classify_by_content() {
        assert_eq!(
            classify(&std::path::PathBuf::from("x"), "KPL/FK\n..."),
            Some(TextKind::Fk)
        );
        assert_eq!(
            classify(&std::path::PathBuf::from("x"), "KPL/PCK\n..."),
            Some(TextKind::Tpck)
        );
        assert_eq!(
            classify(&std::path::PathBuf::from("x"), "KPL/LSK\n..."),
            Some(TextKind::Lsk)
        );
        assert_eq!(
            classify(&std::path::PathBuf::from("x"), "随便一个文本文件"),
            None
        );
        assert_eq!(
            classify(&std::path::PathBuf::from("x"), "\n\nKPL/LSK"),
            Some(TextKind::Lsk)
        );
    }

    /// DAF 头识别。
    #[test]
    fn is_daf_checks_magic() {
        assert!(is_daf(b"DAF/xxxxxxx"));
        assert!(!is_daf(b"DAX/xxxxxxx"));
        assert!(!is_daf(b"KPL/LSK"));
    }

    /// D 指数、括号跨行、引号串、@日期、未知键忽略、`\\begindata` 边界。
    #[test]
    fn pool_parses_kernel_syntax() {
        let content = "\
KPL/LSK
注释区 do_not_parse 之前都忽略

\\begindata
DELTET/DELTA_T_A       =   32.184
DELTET/K               =    1.657D-3
DELTET/M               = (  6.239996D0   1.99096871D-7 )
DELTET/DELTA_AT        = ( 10,   @1972-JAN-1
                           11,   @1972-JUL-1 )
FRAME_31007_AXES            = (   3,        2,        1       )
SOMETHING_ELSE              = 'unknown key kept as string'
QUOTED                      = 'it''s fine'
TRAIL                       = ( 1 0 0
                                0 1 0
                                0 0 1 )
\\begintext
这里全是注释 = ( 1 2 3 )
\\begindata
AFTER_TEXT                  = 5
";
        let pool = parse_pool(content).expect("解析");
        assert_eq!(pool.nums("DELTET/DELTA_T_A"), Some(vec![32.184]));
        assert_eq!(pool.nums("DELTET/K"), Some(vec![1.657e-3]));
        assert_eq!(pool.nums("DELTET/M"), Some(vec![6.239996, 1.99096871e-7]));
        let leap = pool.nums("DELTET/DELTA_AT").expect("DELTA_AT");
        assert_eq!(leap[0], 10.0);
        assert_eq!(leap[1], date_token_to_seconds(1972, 1, 1, 0.0));
        assert_eq!(leap[3], date_token_to_seconds(1972, 7, 1, 0.0));
        // (JD(1972-01-01) - 2451545)*86400 = -883656000（CSPICE 装载实测值）。
        assert_eq!(leap[1], -883656000.0);
        assert_eq!(pool.nums("FRAME_31007_AXES"), Some(vec![3.0, 2.0, 1.0]));
        assert_eq!(
            pool.get("SOMETHING_ELSE")
                .and_then(|v| v.first())
                .and_then(Value::as_str),
            Some("unknown key kept as string")
        );
        assert_eq!(
            pool.get("QUOTED")
                .and_then(|v| v.first())
                .and_then(Value::as_str),
            Some("it's fine")
        );
        assert_eq!(
            pool.nums("TRAIL"),
            Some(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        );
        assert!(pool.get("这里全是注释").is_none(), "\\begintext 区段不解析");
        assert_eq!(pool.nums("AFTER_TEXT"), Some(vec![5.0]));
    }

    /// 解析失败必须硬报错：缺右括号、非法数字。
    #[test]
    fn parse_errors_are_hard() {
        assert!(parse_pool("KPL/X\n\\begindata\nA = ( 1 2").is_err());
        assert!(parse_pool("KPL/X\n\\begindata\nA = xyz12ab").is_err());
        assert!(parse_pool("KPL/X\n\\begindata\n= 5").is_err());
    }

    /// @日期换算与 CSPICE 内核池装载一致（1972-01-01 → -883656000 s）。
    #[test]
    fn date_token_matches_cspice_pool_value() {
        assert_eq!(parse_date_token("@1972-JAN-1"), Ok(-883656000.0));
    }
}
