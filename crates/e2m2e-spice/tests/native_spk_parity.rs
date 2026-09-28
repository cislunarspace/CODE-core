//! native_spk 与 CSPICE 的对拍测试（ADR 0051，issue #684 验收）。
//!
//! 断言全部为 `f64::to_bits` 逐位相等（非容差）。CSPICE 侧 oracle 一律直连
//! `cspice_rs::spk::easier_reader` / `cspice_rs_sys` DAF 例程（FFI 参照身份），绝不
//! 经 `spice_ffi::spkezr`（那已是新后端）。
//!
//! 运行约定：`make test-rust` 以 `--test-threads=1` 执行；裸跑（默认多线程）
//! 时本文件用文件级互斥锁串行（先例：`nbody_stm.rs::TEST_LOCK`）。

use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, Once};

use cspice_rs::common::AberrationCorrection;
use cspice_rs::spk::easier_reader;
use cspice_rs::time::Et;
use cspice_rs_sys::{
    dafcls_c, daffna_c, dafopr_c, dafrfr_c, dafus_c, erract_c, errdev_c, failed_c, reset_c,
    SpiceBoolean, SpiceDouble, SpiceInt,
};
use e2m2e_spice::native_spk::{self, daf};

/// 本文件的 CSPICE/注册表隔离锁：所有用例先取锁，保证裸跑多线程时不串池。
static FILE_LOCK: Mutex<()> = Mutex::new(());

fn lock() -> MutexGuard<'static, ()> {
    FILE_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// CSPICE 错误动作显式设为 RETURN/NULL（防 ABORT 杀进程；与
/// spice_ffi::init_error_handling 同款，oracle 侧保险）。
fn ensure_cspice_return_mode() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        let set = CString::new("SET").unwrap();
        let ret = CString::new("RETURN").unwrap();
        let null = CString::new("NULL").unwrap();
        erract_c(set.as_ptr() as *mut _, 0, ret.as_ptr() as *mut _);
        errdev_c(set.as_ptr() as *mut _, 0, null.as_ptr() as *mut _);
    });
}

fn check_cspice(context: &str) {
    unsafe {
        if failed_c() != 0 {
            reset_c();
            panic!("CSPICE oracle 调用失败: {context}");
        }
    }
}

// ── 内核定位 ─────────────────────────────────────────────────────────────

fn kernel_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .join("kernels")
}

fn kernel_path(name: &str) -> PathBuf {
    kernel_dir().join(name)
}

/// 6 个二进制内核（91 段的验收集合）。
const BINARY_KERNELS: [&str; 6] = [
    "de421.bsp",
    "de430.bsp",
    "de440s.bsp",
    "earth_latest_high_prec.bpc",
    "SPICEEarthPredictedKernel.bpc",
    "SPICELunaCurrentKernel.bpc",
];

/// 验收期望的段数（issue #684 Agent Brief 表格）。
const EXPECTED_SEGMENTS: [(&str, usize); 6] = [
    ("de421.bsp", 15),
    ("de430.bsp", 14),
    ("de440s.bsp", 14),
    ("earth_latest_high_prec.bpc", 10),
    ("SPICEEarthPredictedKernel.bpc", 37),
    ("SPICELunaCurrentKernel.bpc", 1),
];

/// 内核缺失时跳过该用例（CI 干净 clone 可能没有 LFS 资产）。
fn require_kernel(name: &str) -> Option<PathBuf> {
    let path = kernel_path(name);
    if path.exists() {
        Some(path)
    } else {
        eprintln!("跳过：内核 {name} 不存在（kernels/ 未下载）");
        None
    }
}

/// 从 native 注册表卸载全部二进制内核（用例间隔离）。
fn native_reset() {
    for name in BINARY_KERNELS {
        native_spk::unload(&kernel_path(name));
    }
}

// ── CSPICE DAF 扫描 oracle ───────────────────────────────────────────────

/// CSPICE 扫描出的段描述符（与 `daf::DafSegment` 可比）。
#[derive(Debug, Clone, PartialEq)]
struct OracleSegment {
    body: i32,
    center: Option<i32>,
    frame: i32,
    dtype: i32,
    dc0: f64,
    dc1: f64,
    data_words: (usize, usize),
}

/// 用 `cspice_rs_sys` 直调 DAF 例程逐段取描述符（FFI oracle，与 native 解析对拍）。
fn cspice_scan_segments(path: &Path) -> Vec<OracleSegment> {
    ensure_cspice_return_mode();
    unsafe {
        let name = CString::new(path.to_string_lossy().as_bytes()).unwrap();
        let mut handle: SpiceInt = 0;
        dafopr_c(name.as_ptr() as *mut _, &mut handle);
        assert_eq!(failed_c(), 0, "dafopr_c 失败: {}", path.display());
        let mut nd: SpiceInt = 0;
        let mut ni: SpiceInt = 0;
        let mut ifname = [0i8; 61];
        let mut fward: SpiceInt = 0;
        let mut bward: SpiceInt = 0;
        let mut free_: SpiceInt = 0;
        dafrfr_c(
            handle,
            61,
            &mut nd,
            &mut ni,
            ifname.as_mut_ptr(),
            &mut fward,
            &mut bward,
            &mut free_,
        );
        assert_eq!(failed_c(), 0, "dafrfr_c 失败: {}", path.display());

        dafbfs_c_impl(handle);
        let mut segments = Vec::new();
        loop {
            let mut found: SpiceBoolean = 0;
            daffna_c(&mut found);
            assert_eq!(failed_c(), 0, "daffna_c 失败: {}", path.display());
            if found == 0 {
                break;
            }
            let mut sum = [0.0f64; 125];
            dafgs_c_impl(sum.as_mut_ptr());
            let mut dc = [0.0f64; 2];
            let mut ic = [0i32; 6];
            dafus_c(sum.as_mut_ptr(), nd, ni, dc.as_mut_ptr(), ic.as_mut_ptr());
            check_cspice("dafus_c");
            let (body, center, frame, dtype, baddr, eaddr) = if ni == 6 {
                (ic[0], Some(ic[1]), ic[2], ic[3], ic[4], ic[5])
            } else if ni == 5 {
                (ic[0], None, ic[1], ic[2], ic[3], ic[4])
            } else {
                panic!("未预期的 NI={ni}: {}", path.display());
            };
            segments.push(OracleSegment {
                body,
                center,
                frame,
                dtype,
                dc0: dc[0],
                dc1: dc[1],
                data_words: (baddr as usize, eaddr as usize),
            });
        }
        dafcls_c(handle);
        check_cspice("dafcls_c");
        segments
    }
}

// bindgen 生成的符号按 C 原型导出；这里包一层避免 unsafe 块内书写细节。
unsafe fn dafbfs_c_impl(handle: SpiceInt) {
    cspice_rs_sys::dafbfs_c(handle);
    check_cspice("dafbfs_c");
}

unsafe fn dafgs_c_impl(sum: *mut SpiceDouble) {
    cspice_rs_sys::dafgs_c(sum);
    check_cspice("dafgs_c");
}

// ── ET 网格 ──────────────────────────────────────────────────────────────

/// Howard Hinnant 民用日算法（天）。
fn days_from_civil(y: i32, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y } as i64;
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m as i64 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// 验收网格：1900-01-01 .. 2076-01-01、30 天步长（2143 点）。
fn parity_grid() -> (f64, f64, f64, usize) {
    let et0 = (days_from_civil(1900, 1, 1) - days_from_civil(2000, 1, 1)) as f64 * 86400.0;
    let et1 = (days_from_civil(2076, 1, 1) - days_from_civil(2000, 1, 1)) as f64 * 86400.0;
    let step = 30.0 * 86400.0;
    let n = (((et1 - et0) / step).floor() as usize) + 1;
    (et0, et1, step, n)
}

// ═════════════════════════════════════════════════════════════════════════
// 1. 段描述符逐位一致（6 内核 91 段）
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn descriptors_bit_identical_91_segments() {
    let _g = lock();
    ensure_cspice_return_mode();
    // 验收依赖 6 个内核全在：任一缺失即整体跳过（而不是被下方的 91 断言
    // 击穿——干净 clone / CI 未拉 LFS 时属外部能力缺失，应跳过而非失败）。
    if BINARY_KERNELS
        .iter()
        .any(|name| !kernel_path(name).exists())
    {
        eprintln!("跳过：6 个二进制内核未全部就位（kernels/ 未下载或未拉 LFS）");
        return;
    }
    let mut total = 0usize;
    for (name, expected) in EXPECTED_SEGMENTS {
        let Some(path) = require_kernel(name) else {
            continue;
        };
        let bytes = std::fs::read(&path).unwrap();
        let parsed = daf::parse(&path, &bytes).expect("native 解析失败");
        let oracle = cspice_scan_segments(&path);
        assert_eq!(parsed.segments.len(), oracle.len(), "{name}: 段数不一致");
        assert_eq!(parsed.segments.len(), expected, "{name}: 段数与验收不符");
        for (i, (mine, theirs)) in parsed.segments.iter().zip(&oracle).enumerate() {
            assert_eq!(mine.body, theirs.body, "{name} 段{i}: body");
            assert_eq!(mine.center, theirs.center, "{name} 段{i}: center");
            assert_eq!(mine.frame, theirs.frame, "{name} 段{i}: frame");
            assert_eq!(mine.dtype, theirs.dtype, "{name} 段{i}: dtype");
            assert_eq!(
                mine.dc0.to_bits(),
                theirs.dc0.to_bits(),
                "{name} 段{i}: dc0"
            );
            assert_eq!(
                mine.dc1.to_bits(),
                theirs.dc1.to_bits(),
                "{name} 段{i}: dc1"
            );
            assert_eq!(mine.data_words, theirs.data_words, "{name} 段{i}: 数据地址");
        }
        total += parsed.segments.len();
        println!(
            "{name}: {} 段全等（描述符 + to_bits）",
            parsed.segments.len()
        );
    }
    assert_eq!(total, 91, "6 内核合计段数应为 91（SPK 43 + BPC 48）");
    // 地球预测 BPC 的跨摘要记录链：37 段意味着 FWARD 链必须走过 2 条摘要记录。
    // （若只走了第一条记录，该内核只会解析出 < 37 段并在此处失败。）
    println!("合计 {total} 段全等");
}

// ═════════════════════════════════════════════════════════════════════════
// 2. 状态逐位一致（de440s / de430，全 ID 对 × 网格）
// ═════════════════════════════════════════════════════════════════════════

/// 对单个 SPK 内核做全 ID 对 × 网格逐位对拍。返回比对状态数。
fn state_parity_for_kernel(name: &str) -> usize {
    let Some(path) = require_kernel(name) else {
        return 0;
    };
    native_reset();
    // 双登记：native 注册表（求值路径）+ CSPICE 池（easier_reader oracle）。
    // 若池中残留早前 furnsh 的其他 bsp，本内核最后 furnsh、按「后加载者生效」
    // 恰为 oracle 实际使用的内核，与 native 注册表只含本内核一致。
    furnish_kernel_checked(&path);

    let bytes = std::fs::read(&path).unwrap();
    let file = daf::parse(&path, &bytes).unwrap();
    let mut ids: Vec<i32> = file
        .segments
        .iter()
        .flat_map(|s| {
            let mut v = vec![s.body];
            if let Some(c) = s.center {
                v.push(c);
            }
            v
        })
        .filter(|&id| id != 0)
        .collect();
    ids.sort_unstable();
    ids.dedup();
    let span_lo = file
        .segments
        .iter()
        .map(|s| s.dc0)
        .fold(f64::INFINITY, f64::min);
    let span_hi = file
        .segments
        .iter()
        .map(|s| s.dc1)
        .fold(f64::NEG_INFINITY, f64::max);

    let mut pairs: Vec<(i32, i32)> = Vec::new();
    for i in 0..ids.len() {
        for j in i + 1..ids.len() {
            pairs.push((ids[i], ids[j]));
        }
    }

    let (et0, et1, step, _) = parity_grid();
    let lo = et0.max(span_lo);
    let hi = et1.min(span_hi);
    let mut ets = Vec::new();
    let mut et = et0;
    while et <= et1 + 0.5 {
        if et >= lo && et <= hi {
            ets.push(et);
        }
        et += step;
    }

    let mut compared = 0usize;
    let mut oracle_unsupported = 0usize;
    let mut samples: Vec<String> = Vec::new();
    for &(target, observer) in &pairs {
        for &et in &ets {
            let native = match native_spk::state(target, et, observer) {
                Ok(s) => s,
                Err(e) => panic!("{name}: native 求值失败 ({target},{observer}) et={et}: {e}"),
            };
            match easier_reader(
                &target.to_string(),
                Et::from(et),
                "J2000",
                AberrationCorrection::NONE,
                &observer.to_string(),
            ) {
                Ok((oracle, _lt)) => {
                    let oracle = [
                        oracle.position.x,
                        oracle.position.y,
                        oracle.position.z,
                        oracle.velocity.0[0],
                        oracle.velocity.0[1],
                        oracle.velocity.0[2],
                    ];
                    for k in 0..6 {
                        assert_eq!(
                            native[k].to_bits(),
                            oracle[k].to_bits(),
                            "{name}: ({target},{observer}) et={et} state[{k}] 逐位不一致"
                        );
                    }
                    compared += 1;
                }
                Err(e) => {
                    oracle_unsupported += 1;
                    if samples.len() < 3 {
                        samples.push(format!("({target},{observer}) et={et}: {e}"));
                    }
                }
            }
        }
    }
    assert_eq!(
        oracle_unsupported, 0,
        "{name}: oracle 自身不支持点应为 0；样本 {:?}",
        samples
    );
    println!(
        "{name}: {} ID 对 × {} 点（覆盖内），比对 {} 组状态，全部 to_bits 相等，零跳过",
        pairs.len(),
        ets.len(),
        compared
    );
    compared
}

fn furnish_kernel_checked(path: &Path) {
    e2m2e_spice::furnish_kernel(&path.to_string_lossy())
        .unwrap_or_else(|e| panic!("furnish_kernel {} 失败: {e}", path.display()));
}

#[test]
fn states_bit_identical_de440s_and_de430() {
    let _g = lock();
    ensure_cspice_return_mode();
    let de440s_present = kernel_path("de440s.bsp").exists();
    let de430_present = kernel_path("de430.bsp").exists();
    if !de440s_present && !de430_present {
        eprintln!("跳过：de440s.bsp / de430.bsp 均不存在（kernels/ 未下载）");
        return;
    }
    let mut total = 0usize;
    let mut kernels = 0usize;
    if de440s_present {
        total += state_parity_for_kernel("de440s.bsp");
        kernels += 1;
    }
    if de430_present {
        total += state_parity_for_kernel("de430.bsp");
        kernels += 1;
    }
    native_reset();
    assert_eq!(
        total,
        390_026 / 2 * kernels,
        "{kernels} 内核合计应比对 {} 组状态（91 对 × 2143 点 × {kernels}）",
        195_013 * kernels
    );
    println!("状态逐位一致合计：{total} 组，零跳过");
}

// ═════════════════════════════════════════════════════════════════════════
// 3. 链拓扑显式用例
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn chain_topologies_bit_identical() {
    let _g = lock();
    ensure_cspice_return_mode();
    if require_kernel("de440s.bsp").is_none() {
        return;
    }
    native_reset();
    // 生产装载面：全部内核（含文本与 BPC），验证 BPC 段不混入选段、
    // 「后加载者生效」下 native 与 CSPICE 选到同一段。
    for name in [
        "naif0012.tls",
        "pck00010.tpc",
        "de440s.bsp",
        "earth_latest_high_prec.bpc",
        "SPICEEarthPredictedKernel.bpc",
        "SPICELunaFrameKernel.tf",
        "SPICELunaCurrentKernel.bpc",
    ] {
        if let Some(p) = require_kernel(name) {
            furnish_kernel_checked(&p);
        }
    }
    e2m2e_spice::spice_ffi::register_bodies();

    // (target, observer, 链长说明)
    // de440s 实际段（spiceypy DAF 扫描核实）：(1..10)→0、(199)→1、(299)→2、
    // (301)→3、(399)→3。故 (199,1) 与 (10,0) 是直连段，(301,399) 经 3 汇合，
    // (10,399) 为 3 跳（10→0 与 399→3→0）。
    let cases: [((i32, i32), &str); 4] = [
        ((199, 1), "直连：199→1"),
        ((301, 399), "2 跳：301→3←399"),
        ((10, 399), "3 跳：10→0 与 399→3→0"),
        ((10, 0), "直连：10→0（SSB 端点）"),
    ];
    let et = 1.0e8_f64; // J2000 后约 3.17 年
    for ((target, observer), note) in cases {
        let native = native_spk::state(target, et, observer)
            .unwrap_or_else(|e| panic!("native ({target},{observer}) 失败: {e}"));
        let (oracle, _lt) = easier_reader(
            &target.to_string(),
            Et::from(et),
            "J2000",
            AberrationCorrection::NONE,
            &observer.to_string(),
        )
        .unwrap_or_else(|e| panic!("oracle ({target},{observer}) 失败: {e}"));
        let oracle = [
            oracle.position.x,
            oracle.position.y,
            oracle.position.z,
            oracle.velocity.0[0],
            oracle.velocity.0[1],
            oracle.velocity.0[2],
        ];
        for k in 0..6 {
            assert_eq!(
                native[k].to_bits(),
                oracle[k].to_bits(),
                "({target},{observer}) et={et} state[{k}]"
            );
        }
        println!("({target},{observer}) [{note}] 6 分量 to_bits 全等");
    }
    native_reset();
}

// ═════════════════════════════════════════════════════════════════════════
// 4. 负控回归：把记录号当字地址必须被发现
// ═════════════════════════════════════════════════════════════════════════

/// 断言「错误解释」的结果必不等于正确结果：Err 或段数 != 14。
fn assert_misparse_detected(result: Result<daf::DafFile, native_spk::DafSpkError>, label: &str) {
    match result {
        Err(e) => println!("{label}: 解析报错（负控生效）: {e}"),
        Ok(f) => {
            println!(
                "{label}: 未报错但得到 {} 段（≠14，负控生效）",
                f.segments.len()
            );
            assert_ne!(
                f.segments.len(),
                14,
                "{label}: 错误解释竟然得到正确段数——负控失效"
            );
        }
    }
}

#[test]
fn fward_word_address_misparse_must_fail() {
    let _g = lock();
    let Some(path) = require_kernel("de440s.bsp") else {
        return;
    };
    let bytes = std::fs::read(&path).unwrap();
    // 正向基线：正确解释得到 14 段。
    let ok = daf::parse(&path, &bytes).expect("正确解析不应失败");
    assert_eq!(ok.segments.len(), 14);

    // 负控 1：记录号整体偏移 1 条记录（记录号/字地址混淆类的链错位）。
    let shifted = daf::parse_with(&path, &bytes, 1);
    assert_misparse_detected(shifted, "parse_with(record_base=1)");

    // 负控 2：把 FWARD/next 直接当字地址（字节偏移 (记录号-1)*8）——
    // 原型首版的真实 bug 形态。
    let word_addressed = daf::parse_word_address(&path, &bytes);
    assert_misparse_detected(word_addressed, "parse_word_address");
}

// ═════════════════════════════════════════════════════════════════════════
// 5. 不支持项硬报错
// ═════════════════════════════════════════════════════════════════════════

/// 构造最小 DAF 字节流：1 条 Type 3 SPK 段（301→399，覆盖 [0,1e5]）。
/// 6 个真实内核全为 Type 2，无法自然触发 UnsupportedType，按计划手工构造。
fn synthetic_type3_kernel() -> Vec<u8> {
    let mut b = vec![0u8; 4096]; // 4 条记录
    b[0..8].copy_from_slice(b"DAF/SPK ");
    b[8..12].copy_from_slice(&2_i32.to_le_bytes()); // ND
    b[12..16].copy_from_slice(&6_i32.to_le_bytes()); // NI
    b[76..80].copy_from_slice(&2_i32.to_le_bytes()); // FWARD
    b[80..84].copy_from_slice(&2_i32.to_le_bytes()); // BWARD
    b[88..96].copy_from_slice(b"LTLIEEE "); // LOCFMT
                                            // 摘要记录（记录 2）：next=0, prev=0, nsum=1；摘要区从字节 24 起。
    let rec2 = 1024usize;
    b[rec2..rec2 + 8].copy_from_slice(&0.0_f64.to_le_bytes()); // next
    b[rec2 + 8..rec2 + 16].copy_from_slice(&0.0_f64.to_le_bytes()); // prev
    b[rec2 + 16..rec2 + 24].copy_from_slice(&1.0_f64.to_le_bytes()); // nsum
    let sum = rec2 + 24;
    b[sum..sum + 8].copy_from_slice(&0.0_f64.to_le_bytes()); // DC0
    b[sum + 8..sum + 16].copy_from_slice(&100000.0_f64.to_le_bytes()); // DC1
    let word = |lo: i32, hi: i32| ((hi as i64) << 32 | (lo as i32) as i64).to_le_bytes();
    b[sum + 16..sum + 24].copy_from_slice(&word(301, 3)); // body/center
    b[sum + 24..sum + 32].copy_from_slice(&word(1, 3)); // frame/type=3
    b[sum + 32..sum + 40].copy_from_slice(&word(4, 11)); // baddr/eaddr
                                                         // 数据区（字 4..11，记录 3）：Type 3 内容不会被求值，填充即可。
    for w in 0..8 {
        let off = 2048 + w * 8;
        b[off..off + 8].copy_from_slice(&1.0_f64.to_le_bytes());
    }
    b
}

#[test]
fn unsupported_type_frame_abcorr_hard_errors() {
    let _g = lock();
    ensure_cspice_return_mode();

    // dtype != 2：合成 Type 3 内核 → 命中段即硬报错。
    let tmp = std::env::temp_dir().join("e2m2e-native-spk-type3-test.bsp");
    std::fs::write(&tmp, synthetic_type3_kernel()).unwrap();
    native_reset();
    native_spk::load(&tmp).expect("合成内核应可解析");
    let err = native_spk::state(301, 0.0, 399).unwrap_err();
    assert_eq!(err, native_spk::DafSpkError::UnsupportedType(3));
    assert!(err.to_string().contains("SPK_NATIVE_UNSUPPORTED_TYPE"));
    native_spk::unload(&tmp);
    let _ = std::fs::remove_file(&tmp);

    // frame != J2000 / abcorr != NONE：经 spice_ffi::spkezr 入口硬报错。
    // 需注册表非空 → 先登记真实内核。
    let Some(path) = require_kernel("de440s.bsp") else {
        return;
    };
    native_reset();
    furnish_kernel_checked(&path);

    let e = e2m2e_spice::spice_ffi::spkezr("MOON", 0.0, "ITRF93", "NONE", "EARTH").unwrap_err();
    assert!(
        e.to_string().contains("SPK_NATIVE_UNSUPPORTED_FRAME"),
        "{e}"
    );

    let e = e2m2e_spice::spice_ffi::spkezr("MOON", 0.0, "J2000", "LT+S", "EARTH").unwrap_err();
    assert!(
        e.to_string().contains("SPK_NATIVE_UNSUPPORTED_ABCORR"),
        "{e}"
    );

    // 名字解析：SSB 与纯数字串均应可走通（不改状态语义，仅验证不报名字错误）。
    let (state, lt) =
        e2m2e_spice::spice_ffi::spkezr("301", 1.0e8, "J2000", "NONE", "SOLAR SYSTEM BARYCENTER")
            .expect("数字串 + SSB 应可解析");
    assert_eq!(lt, 0.0);
    let (state2, _) = e2m2e_spice::spice_ffi::spkezr("MOON", 1.0e8, "J2000", "NONE", "0")
        .expect("MOON 别名 + 数字 observer 应可解析");
    for k in 0..6 {
        assert_eq!(state[k].to_bits(), state2[k].to_bits());
    }
    native_reset();
}

// ═════════════════════════════════════════════════════════════════════════
// 5b. 残缺链拓扑（spkgeo.c 观察者链前重置 found）
// ═════════════════════════════════════════════════════════════════════════

/// 构造只含两段的 DAF：`301→3` 与 `399→3`（均 Type 2、覆盖 [0, 1e6]、
/// 常量位置 `p_a` / `p_b`、零速度）。用于验证 target 链在 `3` 处断裂
/// （无 `3→0` 段）时，observer 链仍继续上溯并命中公共节点 3 —— 对应
/// `spkgeo.c:978` 在观察者循环前无条件 `found = TRUE_`。若沿用 target 链
/// 遗留的 `found = false`，此查询会错误地硬报 `NoSegment`。
fn synthetic_partial_chain_kernel(p_a: [f64; 3], p_b: [f64; 3]) -> Vec<u8> {
    let mut b = vec![0u8; 4096];
    let put_f64 = |b: &mut Vec<u8>, addr: usize, v: f64| {
        let off = (addr - 1) * 8;
        b[off..off + 8].copy_from_slice(&v.to_le_bytes());
    };
    // 文件记录：ND=2、NI=6、FWARD=BWARD=2、LTL。
    b[0..8].copy_from_slice(b"DAF/SPK ");
    b[8..12].copy_from_slice(&2_i32.to_le_bytes());
    b[12..16].copy_from_slice(&6_i32.to_le_bytes());
    b[76..80].copy_from_slice(&2_i32.to_le_bytes());
    b[80..84].copy_from_slice(&2_i32.to_le_bytes());
    b[88..96].copy_from_slice(b"LTLIEEE ");
    // 摘要记录（记录 2）：next=prev=0、nsum=2、摘要区自记录内字节 24 起。
    let rec2 = 1024usize;
    b[rec2 + 16..rec2 + 24].copy_from_slice(&2.0_f64.to_le_bytes());
    let mut sum = rec2 + 24;
    let mut put_summary = |body: i32, center: i32, baddr: i32, eaddr: i32| {
        b[sum..sum + 8].copy_from_slice(&0.0_f64.to_le_bytes()); // DC0
        b[sum + 8..sum + 16].copy_from_slice(&1.0e6_f64.to_le_bytes()); // DC1
        let word = |lo: i32, hi: i32| ((hi as i64) << 32 | (lo as i32) as i64).to_le_bytes();
        b[sum + 16..sum + 24].copy_from_slice(&word(body, center));
        b[sum + 24..sum + 32].copy_from_slice(&word(1, 2)); // frame=1(J2000), type=2
        b[sum + 32..sum + 40].copy_from_slice(&word(baddr, eaddr));
        sum += 40; // SS = ND + ⌈NI/2⌉ = 5 字
    };
    // 数据区：段 0 字 257..265（记录 5 字 + 段尾 4 字），段 1 字 266..274。
    put_summary(301, 3, 257, 265);
    put_summary(399, 3, 266, 274);
    for (base, p) in [(257usize, p_a), (266usize, p_b)] {
        put_f64(&mut b, base, 0.0); // MID
        put_f64(&mut b, base + 1, 5.0e5); // RADIUS = INTLEN/2
        put_f64(&mut b, base + 2, p[0]); // ncof=1 → 常量位置、零速度
        put_f64(&mut b, base + 3, p[1]);
        put_f64(&mut b, base + 4, p[2]);
        put_f64(&mut b, base + 5, 0.0); // INIT
        put_f64(&mut b, base + 6, 1.0e6); // INTLEN
        put_f64(&mut b, base + 7, 5.0); // RSIZE
        put_f64(&mut b, base + 8, 1.0); // N
    }
    b
}

#[test]
fn partial_target_chain_still_finds_common_node() {
    let _g = lock();
    let p_a = [10.0_f64, 20.0, 30.0];
    let p_b = [100.0_f64, 200.0, 300.0];
    let tmp = std::env::temp_dir().join("e2m2e-native-spk-partial-chain-test.bsp");
    std::fs::write(&tmp, synthetic_partial_chain_kernel(p_a, p_b)).unwrap();
    native_reset();
    native_spk::load(&tmp).expect("合成内核应可解析");

    // target 链 301→3 断裂（无 3→0 段）；公共节点 3 只可能由 observer 链找到。
    let state = native_spk::state(301, 0.0, 399)
        .expect("残缺链下不应报 NoSegment（spkgeo.c 观察者链前重置 found）");
    let expect = [
        p_a[0] - p_b[0],
        p_a[1] - p_b[1],
        p_a[2] - p_b[2],
        0.0,
        0.0,
        0.0,
    ];
    assert_eq!(state, expect, "常量段状态应为两段位置之差、零速度");

    // 反向：observer=301 同样经 observer 链命中公共节点。
    let rev = native_spk::state(399, 0.0, 301).expect("反向查询同样应成功");
    assert_eq!(rev, [-expect[0], -expect[1], -expect[2], 0.0, 0.0, 0.0]);

    native_spk::unload(&tmp);
    let _ = std::fs::remove_file(&tmp);
}

// ═════════════════════════════════════════════════════════════════════════
// 6. 性能：native ≥ 3× easier_reader（同进程相对比较）
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn native_beats_ffi_3x() {
    let _g = lock();
    ensure_cspice_return_mode();
    let Some(path) = require_kernel("de440s.bsp") else {
        return;
    };
    native_reset();
    furnish_kernel_checked(&path);

    let bytes = std::fs::read(&path).unwrap();
    let file = daf::parse(&path, &bytes).unwrap();
    let span = |body: i32, other: i32| -> (f64, f64) {
        let s301 = file
            .segments
            .iter()
            .find(|s| s.body == body && s.center == Some(other))
            .unwrap();
        (s301.dc0, s301.dc1)
    };
    let (lo301, hi301) = span(301, 3);
    let (lo399, hi399) = span(399, 3);
    let lo = lo301.max(lo399);
    let hi = hi301.min(hi399);

    const N: usize = 100_000;
    let ets: Vec<f64> = (0..N)
        .map(|i| lo + (hi - lo) * (i as f64) / ((N - 1) as f64))
        .collect();

    // warm
    for &et in &ets[..1_000] {
        native_spk::state(301, et, 399).unwrap();
        easier_reader(
            "301",
            Et::from(et),
            "J2000",
            AberrationCorrection::NONE,
            "399",
        )
        .unwrap();
    }

    let t0 = std::time::Instant::now();
    for &et in &ets {
        native_spk::state(301, et, 399).unwrap();
    }
    let native_dt = t0.elapsed();

    let t1 = std::time::Instant::now();
    for &et in &ets {
        easier_reader(
            "301",
            Et::from(et),
            "J2000",
            AberrationCorrection::NONE,
            "399",
        )
        .unwrap();
    }
    let ffi_dt = t1.elapsed();

    let native_qps = N as f64 / native_dt.as_secs_f64();
    let ffi_qps = N as f64 / ffi_dt.as_secs_f64();
    let ratio = native_qps / ffi_qps;
    println!(
        "性能：native {native_qps:.3e} qps，easier_reader(FFI) {ffi_qps:.3e} qps，比值 {ratio:.2}×"
    );
    // 阈值 3.0（issue #684 验收口径：≥ batch_body_states_py 的 3×；#639 原型实测
    // 3.6×）。此处基线取逐次 `easier_reader`（同进程、无 PyO3 边界的 FFI 代理），
    // 比 batch_body_states_py 更快，故断言偏保守——release 实测 4.2× 同时覆盖两种口径。
    // 只在优化构建下断言：cspice-sys 链接的是 NAIF 预编译 -O2 静态库，debug
    // profile 下 C 侧仍优化而本 crate 的 Clenshaw 未优化，比值失真（debug 实测
    // ~0.4×）。验收证据取 `cargo test --release -p e2m2e-spice`（见 ADR 0051
    // 验证节与 issue #684 运行日志）。
    if cfg!(debug_assertions) {
        println!("debug 构建：仅打印比值，不断言（CSPICE 为预编译 -O2，对比失真）");
    } else {
        assert!(ratio >= 3.0, "native/FFI qps 比值 {ratio:.2} < 3.0");
    }
    native_reset();
}
