//! 纯 Rust LSK 时间后端（`deltet`/`et2utc` 的 ISOC 路径，ADR 0056）。
//!
//! 行为地面真值：vendored CSPICE 的 `deltet.c`、`unitim.c`（TDB→TAI 的
//! 三次不动点迭代）与 `et2utc.c`（ISOC 路径，含 `ttrans.c` 的 TAI→YMD
//! 闰秒日拆分与秒位进位链）。全部**逐操作移植**。
//!
//! 池语义与 `native_spk::STORE` / `native_frame` 同款：load 幂等
//! （先移除旧同路径再追加），查询逆序扫描 = **后加载者生效**——
//! naif0011/naif0012 并存时最后 furnsh 者生效（被
//! `tests/data/kernels/test_spice_manager.py` 的常量钉死）。
//!
//! 边界：只实现 ISOC（仓库两个调用点均为 ISOC：`et2utc(t, 0)`、
//! `et2utc(et, 3)`）；`prec` 超出 0..=9 硬报错（CSPICE 对 prec 是钳位，
//! 本仓调用面只用 0/3，超界按错误面拒绝）。

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use super::native_frame::text::{self, Lsk};

/// 时间后端错误。
#[derive(Debug, Clone, PartialEq)]
pub enum NativeTimeError {
    /// 未装载任何 LSK。
    NoLsk,
    /// LSK 数据语义错误。
    BadKernel(String),
    /// ISOC 小数位超出支持范围。
    UnsupportedPrec(i32),
}

impl std::fmt::Display for NativeTimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NativeTimeError::NoLsk => write!(
                f,
                "Rust CSPICE 实例无内核加载——请经 SPICEManager.load_kernel 加载"
            ),
            NativeTimeError::BadKernel(msg) => write!(f, "NATIVE_TIME_BAD_KERNEL: {msg}"),
            NativeTimeError::UnsupportedPrec(p) => write!(
                f,
                "NATIVE_TIME_UNSUPPORTED_PREC: ISOC 小数位 {p} 不受支持（仅 0..=9）"
            ),
        }
    }
}

impl std::error::Error for NativeTimeError {}

static LSK_STORE: RwLock<Vec<(PathBuf, Arc<Lsk>)>> = RwLock::new(Vec::new());

fn lsk_store() -> std::sync::RwLockReadGuard<'static, Vec<(PathBuf, Arc<Lsk>)>> {
    LSK_STORE.read().unwrap_or_else(|e| e.into_inner())
}

/// 解析并加载 LSK 到池。重复 load 同一路径幂等（后加载者生效）。
pub fn load_lsk(path: &Path, content: &str) -> Result<(), NativeTimeError> {
    let pool = text::parse_pool(content).map_err(|e| NativeTimeError::BadKernel(e.to_string()))?;
    let lsk = Lsk::from_pool(&pool).map_err(|e| NativeTimeError::BadKernel(e.to_string()))?;
    let mut store = LSK_STORE.write().unwrap_or_else(|e| e.into_inner());
    store.retain(|(p, _)| p != path);
    store.push((path.to_path_buf(), Arc::new(lsk)));
    Ok(())
}

/// 从 LSK 池卸载一个内核（幂等）。
pub fn unload(path: &Path) {
    LSK_STORE
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .retain(|(p, _)| p != path);
}

/// 指定路径是否已在 LSK 池。
pub fn is_loaded(path: &Path) -> bool {
    lsk_store().iter().any(|(p, _)| p == path)
}

/// LSK 池是否为空（et2utc 入口预检用）。
pub fn is_empty() -> bool {
    lsk_store().is_empty()
}

/// LSK 池条目计数（ktotal 语义）。
pub fn count() -> usize {
    lsk_store().len()
}

/// 生效的 LSK 快照：逆序扫描 = 后加载者生效。快照 `Arc` 克隆后即释放
/// 读锁，求值不持锁。
fn snapshot() -> Result<Arc<Lsk>, NativeTimeError> {
    let store = lsk_store();
    store
        .iter()
        .next_back()
        .map(|(_, l)| Arc::clone(l))
        .ok_or(NativeTimeError::NoLsk)
}

// ---------------------------------------------------------------------------
// DELTET（deltet.c 逐操作移植）
// ---------------------------------------------------------------------------

/// Fortran `NINT`：四舍五入到最近整数（.5 远离零）。Rust `round` 同口径。
fn d_nint(x: f64) -> f64 {
    x.round()
}

/// `deltet(epoch, 'UTC'|'ET')`：ET − UTC。`utc_in = true` 表示
/// `epoch` 为 UTC 秒（'UTC'），否则为 ET 秒（'ET'）。诊断/对拍用；
/// 可逆性断言（D1 == D2）由对拍测试承担。
pub fn deltet(epoch: f64, utc_in: bool) -> Result<f64, NativeTimeError> {
    deltet_with(epoch, utc_in, snapshot()?.as_ref())
}

/// [`deltet`] 的显式 LSK 变体（测试/内部复用）。
#[doc(hidden)]
pub fn deltet_with(epoch: f64, utc_in: bool, lsk: &Lsk) -> Result<f64, NativeTimeError> {
    let dta = lsk.delta_t_a;
    let k = lsk.k;
    let eb = lsk.eb;
    let m = lsk.m;
    let nleap = lsk.leap.len();
    if nleap > 200 {
        return Err(NativeTimeError::BadKernel(format!(
            "闰秒对数 {nleap} 超过缓冲上限 200（deltet.c 原样拒绝）"
        )));
    }
    if nleap == 0 {
        return Err(NativeTimeError::BadKernel(
            "DELTET/DELTA_AT 为空，无法计算 DELTET".into(),
        ));
    }

    // leaps 初值 = dleap[0] - 1（deltet.c 原样：首项 DELTA_AT − 1）。
    let mut leaps = lsk.leap[0].1 - 1;

    if utc_in {
        for leap in &lsk.leap {
            if epoch >= leap.0 {
                leaps = leap.1;
            }
        }
    } else {
        for leap in &lsk.leap {
            if epoch > leap.0 {
                let et_leap = leap.0 + dta + leap.1 as f64;
                let aet = d_nint(et_leap);
                let ma = m[0] + m[1] * aet;
                let ea = ma + eb * ma.sin();
                let ettai = k * ea.sin();
                let et_i = leap.0 + dta + leap.1 as f64 + ettai;
                if epoch >= et_i {
                    leaps = leap.1;
                }
            }
        }
    }

    let aet = if utc_in {
        d_nint(epoch + dta + leaps as f64)
    } else {
        d_nint(epoch)
    };
    let ma = m[0] + m[1] * aet;
    let ea = ma + eb * ma.sin();
    let ettai = k * ea.sin();
    Ok(dta + leaps as f64 + ettai)
}

/// `unitim(et, "TDB", "TAI")`（unitim.c 原样）：TDT 与 TDB 间的周期项
/// 三次不动点迭代后减 ΔT_A。
fn tdb_to_tai(et: f64, lsk: &Lsk) -> f64 {
    let mut tdt = et;
    for _ in 0..3 {
        tdt = et
            - lsk.k
                * (lsk.m[0] + lsk.m[1] * tdt + lsk.eb * (lsk.m[0] + lsk.m[1] * tdt).sin()).sin();
    }
    tdt - lsk.delta_t_a
}

// ---------------------------------------------------------------------------
// TTRANS("TAI", "YMD")（ttrans.c TAI→formal 路径移植）
// ---------------------------------------------------------------------------

/// `lstled`：taitab 中 ≤ tai 的最后下标（0 基；无 → −1）。
fn lstled(tai: f64, taitab: &[f64]) -> i64 {
    let mut j: i64 = -1;
    for (i, v) in taitab.iter().enumerate() {
        if *v <= tai {
            j = i as i64;
        } else {
            break;
        }
    }
    j
}

/// `lstlei`：daytab 中 ≤ daynum 的最后下标（0 基；无 → −1）。
fn lstlei(daynum: f64, daytab: &[f64]) -> i64 {
    let mut j: i64 = -1;
    for (i, v) in daytab.iter().enumerate() {
        if *v <= daynum {
            j = i as i64;
        } else {
            break;
        }
    }
    j
}

/// TAI 秒（J2000 起算、整数）→ `(年, 月, 日, 时, 分, 秒)`。
/// `ttrans.c` 的表构建 + uniform→YMD 拆分逐操作移植；闰秒日拆分为
/// 23:59:60、秒位进位链（exsecs）原样。
fn ttrans_tai_to_ymd(tai_whole: f64, lsk: &Lsk) -> (f64, f64, f64, f64, f64, f64) {
    const SECSPD: f64 = 86400.0;
    const HALFD: f64 = 43200.0;
    const DN2000: f64 = 2451545.0;

    let leaps = &lsk.leap;
    let nref = leaps.len() * 2;
    // taitab/daytab 构建（ttrans.c 初始化段原样）。
    let mut taitab = vec![0.0_f64; nref];
    let mut daytab = vec![0.0_f64; nref];
    let mut lastdt = leaps[0].1 as f64 - 1.0;
    for (k, (formal, dt)) in leaps.iter().enumerate() {
        let formal = *formal;
        let dt = *dt as f64;
        taitab[2 * k] = formal - SECSPD + lastdt;
        taitab[2 * k + 1] = formal + dt;
        let daynum = ((formal + HALFD) / SECSPD).trunc() + DN2000;
        daytab[2 * k] = daynum - 1.0;
        daytab[2 * k + 1] = daynum;
        lastdt = dt;
    }

    let mut daynum;
    let mut secs;
    // uniform（TAI）→ formal（YMD）。
    let mut tai = tai_whole;
    let mut j = lstled(tai, &taitab);
    if j >= 0 && j % 2 == 0 {
        // 1 基奇位：闰秒日的 daytab 直接命中。
        daynum = daytab[j as usize];
        secs = tai - taitab[j as usize];
    } else {
        let jj = j.max(0) as usize;
        let d = tai - taitab[jj];
        let q = (d / SECSPD).floor();
        secs = d - q * SECSPD;
        daynum = q + daytab[jj];
    }
    // formal→formal 的边界校正（ttrans.c：secs > secspd-1 或 < 0）。
    if secs > SECSPD - 1.0 || secs < 0.0 {
        let dayptr = lstlei(daynum, &daytab).max(0) as usize;
        secs += (daynum - daytab[dayptr]) * SECSPD;
        tai = taitab[dayptr] + secs;
        j = lstled(tai, &taitab);
        if j >= 0 && j % 2 == 0 {
            daynum = daytab[j as usize];
            secs = tai - taitab[j as usize];
        } else {
            let jj = j.max(0) as usize;
            let d = tai - taitab[jj];
            let q = (d / SECSPD).floor();
            secs = d - q * SECSPD;
            daynum = q + daytab[jj];
        }
    }
    // 越日回绕（forml[pto] && secs > secspd；= secspd 的闰秒尾不回绕）。
    if secs > SECSPD {
        daynum += 1.0;
        secs = 0.0;
    }
    // 时分秒拆分（ttrans.c：exsecs 进位链）。
    let exsecs = (secs - SECSPD + 1.0).max(0.0);
    let tsecs = secs - exsecs;
    let hours = (tsecs / 3600.0).floor();
    let tempd = tsecs - hours * 3600.0;
    let mins = (tempd / 60.0).floor();
    let mut tsecs = tempd - mins * 60.0;
    tsecs += exsecs;

    // daynum（正午基整数 JD）→ 年月日。
    let (year, month, day) = civil_from_noon_jd(daynum);
    (year, month, day, hours, mins, tsecs)
}

/// 正午基整数 JD → `(年, 月, 日)`（`days_from_civil` 逆，ttrans 的
/// 400/100/4/1 分解与之等价）。
fn civil_from_noon_jd(daynum: f64) -> (f64, f64, f64) {
    // daynum 为正午 12:00 的整数 JD；1970-01-01 = JD 2440587.5 →
    // days70 = daynum − 2440588。
    let days70 = (daynum - 2440588.0) as i64;
    let (y, m, d) = crate::spice_ffi::civil_from_days(days70);
    (y as f64, m as f64, d as f64)
}

// ---------------------------------------------------------------------------
// ET2UTC（et2utc.c ISOC 路径移植）
// ---------------------------------------------------------------------------

/// `et2utc(et, "ISOC", prec)`：ET → UTC ISO 串
/// `"YYYY-MM-DDTHH:MM:SS[.fff]"`。只支持 ISOC（仓库调用面）；
/// `prec` ∈ 0..=9，超界 → [`NativeTimeError::UnsupportedPrec`]。
pub fn et2utc_isoc(et: f64, prec: i32) -> Result<String, NativeTimeError> {
    et2utc_isoc_with(et, prec, snapshot()?.as_ref())
}

/// [`et2utc_isoc`] 的显式 LSK 变体（测试/内部复用）。
#[doc(hidden)]
pub fn et2utc_isoc_with(et: f64, prec: i32, lsk: &Lsk) -> Result<String, NativeTimeError> {
    if !(0..=9).contains(&prec) {
        return Err(NativeTimeError::UnsupportedPrec(prec));
    }
    // tai = unitim(et, "TDB", "TAI")（et2utc.c 原样）。
    let tai = tdb_to_tai(et, lsk);

    // 整秒 + 小数秒的舍入进位链（et2utc.c 原样：INT 截断 + 负值 floor
    // 修正 + scale 舍入 + 进位回吐）。
    let mut whlsec = tai as i64; // Fortran INT() 向零截断
    if tai < 0.0 && tai != whlsec as f64 {
        whlsec -= 1;
    }
    let myprec = prec as u32;
    let scale = d_nint(10_f64.powi(myprec as i32));
    let mut frcsec = d_nint(scale * (tai - whlsec as f64));
    if frcsec == scale {
        whlsec += 1;
        frcsec = 0.0;
    }
    frcsec /= scale;

    // ttrans("TAI", "YMD") → 日历分量，逐项 i_dnnt。
    let (y, mo, d, h, mi, s) = ttrans_tai_to_ymd(whlsec as f64, lsk);
    let year = d_nint(y) as i64;
    let month = d_nint(mo) as i64;
    let day = d_nint(d) as i64;
    let hour = d_nint(h) as i64;
    let minute = d_nint(mi) as i64;
    let second = d_nint(s) as i64;

    // 年份：>= 1000 直接十进制；1..999 同样不带前导零（ISO 无 AD 前缀）；
    // <= 0 硬报错（et2utc.c 的 SPICE(YEAROUTOFRANGE)）。
    if year <= 0 {
        return Err(NativeTimeError::BadKernel(format!(
            "ISO 格式不支持公元前的年份: {year}"
        )));
    }
    let mut out = format!("{year}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}");
    if myprec > 0 {
        // frcsec += 1 后按 myprec+1 位打印、跳过首位 "1"（dpstrf 链），
        // 等价于小数部分 zero-padded myprec 位。
        let digits = d_nint(frcsec * scale) as i64;
        out.push_str(&format!(".{digits:0width$}", width = myprec as usize));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// naif0012.tls 实测常量：32.184 / 1.657e-3 / 1.671e-2 / M 对 /
    /// 28 项闰秒（首项 @1972-JAN-1 = −883656000 s）。
    #[test]
    fn lsk_parses_naif0012_values() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .unwrap()
            .join("kernels");
        let Ok(content) = std::fs::read_to_string(dir.join("naif0012.tls")) else {
            return; // 内核未就绪时跳过（make setup 产出）
        };
        let pool = text::parse_pool(&content).expect("解析");
        let lsk = Lsk::from_pool(&pool).expect("LSK");
        assert_eq!(lsk.delta_t_a, 32.184);
        assert_eq!(lsk.k, 1.657e-3);
        assert_eq!(lsk.eb, 1.671e-2);
        assert_eq!(lsk.m, [6.239996, 1.99096871e-7]);
        assert_eq!(lsk.leap.len(), 28);
        assert_eq!(lsk.leap[0], (-883656000.0, 10));
        assert_eq!(lsk.leap[27], (536500800.0, 37));
    }

    /// et2utc：闰秒日显示 ：60、 prec=0/3 的整秒链（有内核时跑）。
    #[test]
    fn et2utc_isoc_around_leap_second() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .unwrap()
            .join("kernels");
        let Ok(content) = std::fs::read_to_string(dir.join("naif0012.tls")) else {
            return;
        };
        let pool = text::parse_pool(&content).expect("解析");
        let lsk = Lsk::from_pool(&pool).expect("LSK");
        // 2017-01-01 闰秒边界（spiceypy str2et 实测 ET 值）：
        //   et0−2 → 23:59:59 / et0−1 → 23:59:60 / et0 → 00:00:00。
        let et0 = 536500869.1839298_f64;
        assert_eq!(
            et2utc_isoc_with(et0 - 2.0, 3, &lsk).unwrap(),
            "2016-12-31T23:59:59.000"
        );
        assert_eq!(
            et2utc_isoc_with(et0 - 1.0, 3, &lsk).unwrap(),
            "2016-12-31T23:59:60.000"
        );
        assert_eq!(
            et2utc_isoc_with(et0, 3, &lsk).unwrap(),
            "2017-01-01T00:00:00.000"
        );
        // prec=0：0.5 s 向上进位（round half away）→ 秒位进位链显示 :60
        //（CSPICE et2utc(et0-1.5,'ISOC',0) 实测同值）。
        assert_eq!(
            et2utc_isoc_with(et0 - 1.5, 0, &lsk).unwrap(),
            "2016-12-31T23:59:60"
        );
    }

    /// deltet 可逆性：D1 = deltet(utc,'UTC')，D2 = deltet(utc+D1,'ET')，
    /// 逐位相等（deltet.c 语义）。
    #[test]
    fn deltet_reversible() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .unwrap()
            .join("kernels");
        let Ok(content) = std::fs::read_to_string(dir.join("naif0012.tls")) else {
            return;
        };
        let pool = text::parse_pool(&content).expect("解析");
        let lsk = Lsk::from_pool(&pool).expect("LSK");
        for &utc in &[-883656000.0_f64, 0.0, 536500800.0, 1.0e9] {
            let d1 = deltet_with(utc, true, &lsk).unwrap();
            let d2 = deltet_with(utc + d1, false, &lsk).unwrap();
            assert_eq!(d1.to_bits(), d2.to_bits(), "deltet 可逆性于 utc={utc}");
        }
    }
}
