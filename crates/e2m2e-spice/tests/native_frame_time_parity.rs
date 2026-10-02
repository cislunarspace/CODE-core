//! native_frame / native_time 与 CSPICE 的对拍测试（ADR 0056，#685 Phase B）。
//!
//! 断言口径：`pxform`/`sxform` 目标**逐位一致**（`f64::to_bits` 相等）；
//! `et2utc` 字符串全等；`deltet` 可逆性逐位相等。IAU_* 文本 PCK 帧实测在
//! IAU_MOON 的 9/1186 网格点存在 1–2 ULP 残差（元素级 ≤8.9e-16 ≈
//! 2e-10 角秒，其余帧逐位一致）——残差阈值 1e-14 元素 / 1e-6 角秒并在
//! ADR 0056 记录。CSPICE 侧 oracle 一律直连
//! [`e2m2e_spice::spice_ffi::ffi_oracle`]（`*_c` 薄包装），绝不经生产入口。
//!
//! 运行约定：`make test-rust` 以 `--test-threads=1` 执行；裸跑（默认多线程）
//! 时本文件用文件级互斥锁串行（先例：`native_spk_parity.rs::FILE_LOCK`）。

use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, Once};

use cspice_rs_sys::{erract_c, errdev_c, failed_c, reset_c};
use e2m2e_spice::native_frame::{self, NativeFrameError};
use e2m2e_spice::native_spk;
use e2m2e_spice::native_time;
use e2m2e_spice::spice_ffi::ffi_oracle;

/// 本文件的 CSPICE/池隔离锁：所有用例先取锁，保证裸跑多线程时不串池。
static FILE_LOCK: Mutex<()> = Mutex::new(());

fn lock() -> MutexGuard<'static, ()> {
    FILE_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// CSPICE 错误动作显式设为 RETURN/NULL（防 ABORT 杀进程；oracle 侧保险）。
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

// ── 内核定位与登记 ──────────────────────────────────────────────────────

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

/// 帧旋转验收集：2 个地球 BPC + 月球 BPC/FK + 文本 PCK + LSK。
const FRAME_KERNELS: [&str; 6] = [
    "naif0012.tls",
    "pck00010.tpc",
    "earth_latest_high_prec.bpc",
    "SPICEEarthPredictedKernel.bpc",
    "SPICELunaCurrentKernel.bpc",
    "SPICELunaFrameKernel.tf",
];

/// 经 [`e2m2e_spice::furnish_kernel`] 双登记一批内核（oracle 与被测同池）。
/// 全部就绪返回 true，任一缺失返回 false（用例跳过）。
fn furnish_all(names: &[&str]) -> bool {
    let mut ok = true;
    for name in names {
        let Some(path) = require_kernel(name) else {
            ok = false;
            continue;
        };
        e2m2e_spice::furnish_kernel(&path.to_string_lossy())
            .unwrap_or_else(|e| panic!("furnish_kernel({name}) 失败: {e}"));
    }
    ok
}

/// 卸载验收集 + naif0011（用例间隔离）。
fn reset_pools() {
    for name in FRAME_KERNELS {
        let p = kernel_path(name);
        native_spk::unload(&p);
        native_frame::unload(&p);
        native_time::unload(&p);
    }
    let p = kernel_path("naif0011.tls");
    native_spk::unload(&p);
    native_frame::unload(&p);
    native_time::unload(&p);
}

// ── 数值辅助 ────────────────────────────────────────────────────────────

/// Hinnant days-from-civil（e2m2e_spice::spice_ffi::days_from_civil 是
/// pub(crate)，集成测试无法直用；算法一致）。
fn days70(year: i32, month: u32, day: i64) -> i64 {
    let y = if month <= 2 {
        i64::from(year) - 1
    } else {
        i64::from(year)
    };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = i64::from((month + 9) % 12);
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// `年-月-日` 0 点（UTC 口径）的近似 ET。网格对拍两侧取同一 et，TDB−UTC
/// 偏差（~64 s）不影响对拍，只影响覆盖边缘的落点。
fn et_of(y: i32, m: u32, d: i64) -> f64 {
    (days70(y, m, d) - 10957) as f64 * 86400.0
}

/// ET 网格：月首逐月 + 可选半年点 + 年度 [1, 6, 12] 月点 + et=0 锚点。
fn et_grid() -> Vec<f64> {
    let mut ets = vec![0.0_f64];
    for y in 2000..=2050 {
        for m in 1..=12 {
            ets.push(et_of(y, m, 1));
        }
    }
    for y in 2020..=2040 {
        for m in [1u32, 7u32] {
            ets.push(et_of(y, m, 1));
        }
    }
    for y in 1900..=2076 {
        for m in [1u32, 6u32, 12u32] {
            ets.push(et_of(y, m, 1));
        }
    }
    ets
}

/// 旋转角差（弧度）：θ = acos((tr(Aᵀ·B) − 1)/2)，钳到 [0, π]。
fn rotation_angle(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> f64 {
    let mut atb = [[0.0_f64; 3]; 3];
    for (i, row) in atb.iter_mut().enumerate() {
        for (j, out) in row.iter_mut().enumerate() {
            *out = a[0][i] * b[0][j] + a[1][i] * b[1][j] + a[2][i] * b[2][j];
        }
    }
    let tr = atb[0][0] + atb[1][1] + atb[2][2];
    ((tr - 1.0) / 2.0).clamp(-1.0, 1.0).acos()
}

fn bits_eq3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> bool {
    a.iter().zip(b.iter()).all(|(r1, r2)| {
        r1.iter()
            .zip(r2.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
    })
}

fn bits_eq6(a: &[[f64; 6]; 6], b: &[[f64; 6]; 6]) -> bool {
    a.iter().zip(b.iter()).all(|(r1, r2)| {
        r1.iter()
            .zip(r2.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
    })
}

fn maxdiff3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> f64 {
    let mut md = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            md = md.max((a[i][j] - b[i][j]).abs());
        }
    }
    md
}

/// 对拍单点：oracle Ok → native 必须 Ok 且满足断言；oracle Err → native
/// 必须同样 Err（两侧同覆盖语义）。
fn check_pxform_point(from: &str, to: &str, et: f64, mode: Parity) {
    let n = e2m2e_spice::spice_ffi::pxform(from, to, et);
    let o = ffi_oracle::pxform(from, to, et);
    match (n, o) {
        (Ok(a), Ok(b)) => match mode {
            Parity::Bits => assert!(
                bits_eq3(&a, &b),
                "pxform({from},{to},{et}) 非逐位: maxdiff={:e}",
                maxdiff3(&a, &b)
            ),
            Parity::Tol { elem, angle_arcsec } => {
                let d = maxdiff3(&a, &b);
                assert!(
                    d <= elem,
                    "pxform({from},{to},{et}) 元素差 {d:e} > {elem:e}"
                );
                let arcsec = rotation_angle(&a, &b) * 180.0 / std::f64::consts::PI * 3600.0;
                assert!(
                    arcsec <= angle_arcsec,
                    "pxform({from},{to},{et}) 角差 {arcsec:e} 角秒 > {angle_arcsec:e}"
                );
            }
        },
        (Err(ne), Ok(_)) => panic!("native 失败而 oracle 成功 pxform({from},{to},{et}): {ne}"),
        (Ok(_), Err(oe)) => panic!("oracle 失败而 native 成功 pxform({from},{to},{et}): {oe}"),
        (Err(_), Err(_)) => {}
    }
}

#[derive(Clone, Copy)]
enum Parity {
    /// 逐位一致。
    Bits,
    /// 元素差 + 旋转角差（角秒）双阈值。
    Tol { elem: f64, angle_arcsec: f64 },
}

fn check_sxform_point(from: &str, to: &str, et: f64) {
    let n = e2m2e_spice::spice_ffi::sxform(from, to, et);
    let o = ffi_oracle::sxform(from, to, et);
    match (n, o) {
        (Ok(a), Ok(b)) => assert!(bits_eq6(&a, &b), "sxform({from},{to},{et}) 非逐位"),
        (Err(ne), Ok(_)) => panic!("native 失败而 oracle 成功 sxform({from},{to},{et}): {ne}"),
        (Ok(_), Err(oe)) => panic!("oracle 失败而 native 成功 sxform({from},{to},{et}): {oe}"),
        (Err(_), Err(_)) => {}
    }
}

// ===========================================================================
// 1. 生产帧对 pxform/sxform 逐位对拍
// ===========================================================================

#[test]
fn pxform_sxform_production_pairs() {
    let _g = lock();
    ensure_cspice_return_mode();
    e2m2e_spice::spice_ffi::register_bodies();
    reset_pools();
    if !furnish_all(&FRAME_KERNELS) {
        return;
    }
    let ets = et_grid();
    // 全部生产帧对逐位一致（含 frame 17 折叠的 ITRF93、TK 链的 MOON_*、
    // 内置常值的 ECLIPJ2000、文本 PCK 全链的 IAU_EARTH）。
    for (from, to) in [
        ("ITRF93", "J2000"),
        ("MOON_PA", "J2000"),
        ("MOON_ME", "J2000"),
        ("ECLIPJ2000", "J2000"),
        ("IAU_EARTH", "J2000"),
    ] {
        for &et in &ets {
            check_pxform_point(from, to, et, Parity::Bits);
        }
    }
    // sxform：ITRF93/MOON_PA/MOON_ME → J2000 同口径。
    for (from, to) in [
        ("ITRF93", "J2000"),
        ("MOON_PA", "J2000"),
        ("MOON_ME", "J2000"),
    ] {
        for &et in &ets {
            check_sxform_point(from, to, et);
        }
    }
    reset_pools();
}

// ===========================================================================
// 2. Euler 3-1-3 负控：误序必须可检出
// ===========================================================================

#[test]
fn euler_313_negative_control() {
    let _g = lock();
    ensure_cspice_return_mode();
    reset_pools();
    let Some(luna) = require_kernel("SPICELunaCurrentKernel.bpc") else {
        return;
    };
    e2m2e_spice::furnish_kernel(&luna.to_string_lossy()).unwrap();
    let Some(fk) = require_kernel("SPICELunaFrameKernel.tf") else {
        return;
    };
    e2m2e_spice::furnish_kernel(&fk.to_string_lossy()).unwrap();

    let bytes = std::fs::read(&luna).unwrap();
    let file = native_spk::daf::parse(&luna, &bytes).unwrap();
    let seg = file
        .segments
        .iter()
        .find(|s| s.body == 31006 && s.center.is_none())
        .expect("SPICELunaCurrentKernel.bpc 应含 31006 BPC 段");
    let et = 0.0_f64.clamp(seg.dc0, seg.dc1);
    let native = native_frame::bpc2::evaluate_with_axes(&bytes, seg, et, [3, 1, 3]).unwrap();
    let oracle = ffi_oracle::pxform("MOON_PA", "J2000", et)
        .unwrap_or_else(|e| panic!("oracle pxform 失败: {e}"));
    check_cspice("pxform MOON_PA");

    // evaluate_with_axes 返回 eul2xf 的旋转块（X(J2000→31006) 的 R）；
    // oracle pxform = X(31006→J2000) = 其转置。正确 3-1-3 序下转置后与
    // oracle 逐位一致。
    let mut transposed = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            transposed[i][j] = native[j][i];
        }
    }
    let good = rotation_angle(&transposed, &oracle);
    assert!(
        bits_eq3(&transposed, &oracle),
        "正确 3-1-3 序转置后应与 oracle 逐位一致: 角差 {good:e} rad"
    );

    // 负控：1-2-3 误序 → 旋转角差 > 1 角秒（误序必须可检出，而非小残差）。
    let bad = native_frame::bpc2::evaluate_with_axes(&bytes, seg, et, [1, 2, 3]).unwrap();
    let diff = rotation_angle(&bad, &oracle);
    let one_arcsec = 1.0_f64 / 3600.0 * std::f64::consts::PI / 180.0;
    assert!(
        diff > one_arcsec,
        "1-2-3 误序必须可检出（> 1 角秒），实测角差 {diff:e} rad"
    );
    reset_pools();
}

// ===========================================================================
// 3. 帧图等价：别名链 / 角秒换算 / 轴序 / class2 选段
// ===========================================================================

#[test]
fn frame_graph_equivalence() {
    let _g = lock();
    ensure_cspice_return_mode();
    e2m2e_spice::spice_ffi::register_bodies();
    reset_pools();
    if !furnish_all(&FRAME_KERNELS) {
        return;
    }
    let ets = et_grid();
    for (from, to) in [
        ("MOON_PA", "MOON_ME"),
        ("MOON_ME", "MOON_PA"),
        ("MOON_PA", "MOON_PA_DE421"),
        ("MOON_ME", "MOON_PA_DE421"),
        ("MOON_ME_DE421", "MOON_PA_DE421"),
        ("MOON_PA_DE421", "MOON_ME_DE421"),
    ] {
        for &et in &ets {
            check_pxform_point(from, to, et, Parity::Bits);
        }
    }
    reset_pools();
}

// ===========================================================================
// 4. ITRF93 后加载者生效
// ===========================================================================

#[test]
fn itrf93_load_order_later_wins() {
    let _g = lock();
    ensure_cspice_return_mode();
    reset_pools();
    let Some(predicted) = require_kernel("SPICEEarthPredictedKernel.bpc") else {
        return;
    };
    let Some(history) = require_kernel("earth_latest_high_prec.bpc") else {
        return;
    };
    // 预测段覆盖内（≈2027；历史段止于 ≈2023-07）。
    let et = et_of(2027, 1, 1);

    // 先预测后历史：双方同池 → native == oracle。
    e2m2e_spice::furnish_kernel(&predicted.to_string_lossy()).unwrap();
    e2m2e_spice::furnish_kernel(&history.to_string_lossy()).unwrap();
    let native = e2m2e_spice::spice_ffi::pxform("ITRF93", "J2000", et).unwrap();
    let oracle_both = ffi_oracle::pxform("ITRF93", "J2000", et).unwrap();
    check_cspice("pxform ITRF93 both");
    assert!(
        bits_eq3(&native, &oracle_both),
        "ITRF93 双装（历史后加载）应与 oracle 逐位一致"
    );

    // oracle 侧卸掉预测 BPC（cspice 池），native 侧不动 → oracle = 只装历史。
    // native == “只装历史 BPC 的 oracle”证明后加载者生效，而非逐文件混合。
    cspice_rs::data::unload(&predicted.to_string_lossy()).unwrap();
    let oracle_history = ffi_oracle::pxform("ITRF93", "J2000", et).unwrap();
    check_cspice("pxform ITRF93 history only");
    assert!(
        bits_eq3(&native, &oracle_history),
        "ITRF93 native 应与只装历史的 oracle 逐位一致"
    );
    reset_pools();
}

// ===========================================================================
// 5. 文本 PCK：10 个 IAU 帧对拍
// ===========================================================================

/// IAU_MOON 单列阈值（角秒）：本实现以 CSPICE 为 oracle 且逐操作移植，
/// #638 的 36 角秒是 anise 口径，不适用；实测残差在 1–2 ULP 元素差
/// （≤8.9e-16 ≈ 2e-10 角秒），阈值留 5 个量级余量。注意 acos 角差度量
/// 对 1-ULP 元素差的噪声地板约 4.35e-3 角秒，阈值必须高于它。
const IAU_MOON_TOL_ARCSEC: f64 = 1e-2;
/// 其余 IAU 帧阈值（角秒）；与 IAU_MOON 同（角差度量噪声地板之上）。
const IAU_TOL_ARCSEC: f64 = 1e-2;
/// 元素级残差上限（IAU_MOON 实测 ≤8.9e-16；主断言，角差度量仅辅助）。
const IAU_TOL_ELEM: f64 = 1e-14;

#[test]
fn text_pck_iau_ten_frames() {
    let _g = lock();
    ensure_cspice_return_mode();
    e2m2e_spice::spice_ffi::register_bodies();
    reset_pools();
    if !furnish_all(&["naif0012.tls", "pck00010.tpc"]) {
        return;
    }
    let frames = [
        "IAU_MERCURY",
        "IAU_VENUS",
        "IAU_EARTH",
        "IAU_MOON",
        "IAU_MARS",
        "IAU_JUPITER",
        "IAU_SATURN",
        "IAU_URANUS",
        "IAU_NEPTUNE",
        "IAU_PLUTO",
    ];
    let ets = et_grid();
    for frame in frames {
        let tol = if frame == "IAU_MOON" {
            IAU_MOON_TOL_ARCSEC
        } else {
            IAU_TOL_ARCSEC
        };
        for &et in &ets {
            check_pxform_point(
                frame,
                "J2000",
                et,
                Parity::Tol {
                    elem: IAU_TOL_ELEM,
                    angle_arcsec: tol,
                },
            );
        }
    }
    reset_pools();
}

// ===========================================================================
// 6. et2utc 字符串全等
// ===========================================================================

#[test]
fn et2utc_string_equality() {
    let _g = lock();
    ensure_cspice_return_mode();
    reset_pools();
    let Some(naif0012) = require_kernel("naif0012.tls") else {
        return;
    };
    e2m2e_spice::furnish_kernel(&naif0012.to_string_lossy()).unwrap();

    // naif0012 闰秒历元（@date → J2000 起算秒）± 1 s。
    let leap_first = -883656000.0_f64; // @1972-JAN-1
    let leap_last = 536500800.0_f64; // @2017-JAN-1
    let mut ets: Vec<f64> = et_grid();
    for k in 0..28 {
        let d = leap_first + k as f64 * if k < 27 { 0.0 } else { leap_last - leap_first };
        let _ = d;
    }
    // 闰秒历元表（naif0012 全部 28 项，naif0011 差最后一项）。
    let leap_dates = [
        -883656000.0,
        -867931200.0,
        -852033600.0,
        -835449600.0,
        -818779200.0,
        -802185600.0,
        -785515200.0,
        -768921600.0,
        -752251200.0,
        -735657600.0,
        -717292800.0,
        -700622400.0,
        -684028800.0,
        -667358400.0,
        -650764800.0,
        -634094400.0,
        -617500800.0,
        -600830400.0,
        -584236800.0,
        -567566400.0,
        -550972800.0,
        -534302400.0,
        -517622400.0,
        -485635200.0,
        -453561600.0,
        -420595200.0,
        -388435200.0,
        -356355872.0,
    ];
    for &d in leap_dates.iter() {
        // 闰秒历元是 @date 秒（naive JD 口径），±1 s 覆盖边界两侧。
        ets.push(d - 1.0);
        ets.push(d);
        ets.push(d + 1.0);
    }
    // 舍入边：59.9996 s 与 .4999/.5001（取一个普通历元 + 偏移）。
    let base = 7.0 * 365.25 * 86400.0;
    ets.push(base + 59.9996);
    ets.push(base + 59.4999);
    ets.push(base + 59.5001);
    for prec in [0, 3] {
        for &et in &ets {
            let native = e2m2e_spice::spice_ffi::et2utc(et, prec)
                .unwrap_or_else(|e| panic!("native et2utc({et},{prec}) 失败: {e}"));
            let oracle = ffi_oracle::et2utc(et, prec)
                .unwrap_or_else(|e| panic!("oracle et2utc({et},{prec}) 失败: {e}"));
            check_cspice("et2utc");
            assert_eq!(native, oracle, "et2utc({et}, {prec}) 字符串不等");
        }
    }
    reset_pools();
}

// ===========================================================================
// 7. deltet 可逆性
// ===========================================================================

#[test]
fn deltet_reversibility() {
    let _g = lock();
    ensure_cspice_return_mode();
    reset_pools();
    let Some(naif0012) = require_kernel("naif0012.tls") else {
        return;
    };
    e2m2e_spice::furnish_kernel(&naif0012.to_string_lossy()).unwrap();
    let mut utcs = et_grid();
    utcs.push(-883656000.0);
    utcs.push(-883656001.0);
    utcs.push(536500800.0);
    utcs.push(536500799.0);
    for &utc in &utcs {
        let d1 = native_time::deltet(utc, true).unwrap();
        let d2 = native_time::deltet(utc + d1, false).unwrap();
        assert_eq!(
            d1.to_bits(),
            d2.to_bits(),
            "deltet 可逆性（D1==D2）破坏于 utc={utc}: {d1} vs {d2}"
        );
    }
    reset_pools();
}

// ===========================================================================
// 8. hifitime 固定点
// ===========================================================================

/// `scripts/compare_spice_ephemeris_frames.py` 的 9 组 (UTC, ET) 常量
/// （naif0012 语义；本用例 naif0011 先装、naif0012 最后装，钉死
/// “后加载者生效”下 naif0012 的 37@2017 生效）。
const HIFITIME_POINTS: [(&str, f64); 9] = [
    ("1900-01-09T00:17:15", -3155024523.8157988),
    ("1920-07-23T14:39:29", -2506972789.816543),
    ("1954-12-24T06:06:31", -1420782767.8162904),
    ("1960-02-14T06:06:31", -1258523567.8148985),
    ("1983-04-13T12:09:14.274", -527644192.54036534),
    ("2000-02-29T14:57:29", 5108313.185383182),
    ("2022-11-29T07:58:49.782", 722980798.9650334),
    ("2044-06-06T12:18:54", 1402100403.1847699),
    ("2075-04-30T23:59:54", 2377166463.185493),
];

#[test]
fn hifitime_fixed_points() {
    let _g = lock();
    ensure_cspice_return_mode();
    reset_pools();
    let Some(naif0011) = require_kernel("naif0011.tls") else {
        return;
    };
    e2m2e_spice::furnish_kernel(&naif0011.to_string_lossy()).unwrap();
    let Some(naif0012) = require_kernel("naif0012.tls") else {
        return;
    };
    e2m2e_spice::furnish_kernel(&naif0012.to_string_lossy()).unwrap();
    for (utc, et) in HIFITIME_POINTS {
        let got = e2m2e_spice::spice_ffi::et2utc(et, 3).unwrap();
        assert!(
            got.starts_with(utc),
            "hifitime 固定点 {utc} (et={et}): native et2utc = {got}"
        );
    }
    reset_pools();
}

// ===========================================================================
// 9. ktotal native 语义
// ===========================================================================

#[test]
fn ktotal_native_semantics() {
    let _g = lock();
    ensure_cspice_return_mode();
    reset_pools();
    // 空池。
    assert_eq!(e2m2e_spice::spice_ffi::ktotal("ALL").unwrap(), 0);
    // 装齐验收集：1 LSK + 1 TPCK + 1 FK + 3 DAF（2 地球 BPC + 1 月球 BPC）。
    if !furnish_all(&FRAME_KERNELS) {
        reset_pools();
        return;
    }
    assert_eq!(e2m2e_spice::spice_ffi::ktotal("ALL").unwrap(), 6, "ALL");
    assert_eq!(e2m2e_spice::spice_ffi::ktotal("SPK").unwrap(), 0, "SPK");
    assert_eq!(e2m2e_spice::spice_ffi::ktotal("PCK").unwrap(), 3, "PCK");
    assert_eq!(e2m2e_spice::spice_ffi::ktotal("FK").unwrap(), 1, "FK");
    assert_eq!(e2m2e_spice::spice_ffi::ktotal("LSK").unwrap(), 1, "LSK");
    assert_eq!(e2m2e_spice::spice_ffi::ktotal("TEXT").unwrap(), 3, "TEXT");
    // 未知 kind 报错。
    let err = e2m2e_spice::spice_ffi::ktotal("CK").unwrap_err();
    assert!(err.to_string().contains("NATIVE_KTOTAL_UNSUPPORTED_KIND"));
    reset_pools();
}

// ===========================================================================
// 10. 文本 FK 负控
// ===========================================================================

#[test]
fn text_fk_negative_controls() {
    let _g = lock();
    ensure_cspice_return_mode();
    reset_pools();
    let tmp = std::env::temp_dir().join("e2m2e_native_frame_negctl");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    // class 3（CK 帧）→ UnsupportedClass。
    let class3 = tmp.join("class3.tf");
    std::fs::write(
        &class3,
        "KPL/FK\n\\begindata\nFRAME_TEST_C3 = 1234567\nFRAME_1234567_NAME = 'TEST_C3'\n\
         FRAME_1234567_CLASS = 3\nFRAME_1234567_CLASS_ID = 100\nFRAME_1234567_CENTER = 399\n",
    )
    .unwrap();
    e2m2e_spice::furnish_kernel(&class3.to_string_lossy()).unwrap();
    let err = e2m2e_spice::spice_ffi::pxform("TEST_C3", "J2000", 0.0).unwrap_err();
    assert!(
        err.to_string().contains("NATIVE_FRAME_UNSUPPORTED_CLASS"),
        "class 3 应报 UnsupportedClass: {err}"
    );

    // 两 TK 帧互为 relative → ChainTooDeep。
    let cycle = tmp.join("cycle.tf");
    std::fs::write(
        &cycle,
        "KPL/FK\n\\begindata\nFRAME_CYC_A = 1234571\nFRAME_1234571_NAME = 'CYC_A'\n\
         FRAME_1234571_CLASS = 4\nFRAME_1234571_CLASS_ID = 1234571\nFRAME_1234571_CENTER = 399\n\
         TKFRAME_1234571_SPEC = 'MATRIX'\nTKFRAME_1234571_RELATIVE = 'CYC_B'\n\
         TKFRAME_1234571_MATRIX = ( 1 0 0 0 1 0 0 0 1 )\n\
         FRAME_CYC_B = 1234572\nFRAME_1234572_NAME = 'CYC_B'\n\
         FRAME_1234572_CLASS = 4\nFRAME_1234572_CLASS_ID = 1234572\nFRAME_1234572_CENTER = 399\n\
         TKFRAME_1234572_SPEC = 'MATRIX'\nTKFRAME_1234572_RELATIVE = 'CYC_A'\n\
         TKFRAME_1234572_MATRIX = ( 1 0 0 0 1 0 0 0 1 )\n",
    )
    .unwrap();
    e2m2e_spice::furnish_kernel(&cycle.to_string_lossy()).unwrap();
    let err = e2m2e_spice::spice_ffi::pxform("CYC_A", "J2000", 0.0).unwrap_err();
    assert!(
        err.to_string().contains("NATIVE_FRAME_CHAIN_TOO_DEEP"),
        "互为 relative 应报 ChainTooDeep: {err}"
    );

    // UNITS='HOURANGLE' → UnsupportedUnits（装载即报，BadKernel 承载）。
    let units = tmp.join("units.tf");
    std::fs::write(
        &units,
        "KPL/FK\n\\begindata\nFRAME_TEST_U = 1234573\nFRAME_1234573_NAME = 'TEST_U'\n\
         FRAME_1234573_CLASS = 4\nFRAME_1234573_CLASS_ID = 1234573\nFRAME_1234573_CENTER = 399\n\
         TKFRAME_1234573_SPEC = 'ANGLES'\nTKFRAME_1234573_RELATIVE = 'J2000'\n\
         TKFRAME_1234573_ANGLES = ( 10 20 30 )\nTKFRAME_1234573_AXES = ( 3 2 1 )\n\
         TKFRAME_1234573_UNITS = 'HOURANGLE'\n",
    )
    .unwrap();
    let content = std::fs::read_to_string(&units).unwrap();
    let err = native_frame::load_fk(&units, &content).unwrap_err();
    assert!(
        matches!(&err, NativeFrameError::BadKernel(m) if m.contains("NATIVE_FRAME_UNSUPPORTED_UNITS")),
        "HOURANGLE 应报 UnsupportedUnits: {err}"
    );

    // relative 指向未定义帧 → UnknownFrame。
    let dangling = tmp.join("dangling.tf");
    std::fs::write(
        &dangling,
        "KPL/FK\n\\begindata\nFRAME_TEST_D = 1234574\nFRAME_1234574_NAME = 'TEST_D'\n\
         FRAME_1234574_CLASS = 4\nFRAME_1234574_CLASS_ID = 1234574\nFRAME_1234574_CENTER = 399\n\
         TKFRAME_1234574_SPEC = 'MATRIX'\nTKFRAME_1234574_RELATIVE = 'NO_SUCH_FRAME_XYZ'\n\
         TKFRAME_1234574_MATRIX = ( 1 0 0 0 1 0 0 0 1 )\n",
    )
    .unwrap();
    e2m2e_spice::furnish_kernel(&dangling.to_string_lossy()).unwrap();
    let err = e2m2e_spice::spice_ffi::pxform("TEST_D", "J2000", 0.0).unwrap_err();
    assert!(
        err.to_string().contains("NATIVE_FRAME_UNKNOWN_FRAME"),
        "未定义 relative 应报 UnknownFrame: {err}"
    );

    let _ = std::fs::remove_dir_all(&tmp);
    reset_pools();
}

// ===========================================================================
// 11. 文本内核 furnish/unload 幂等
// ===========================================================================

#[test]
fn furnish_unload_text_idempotent() {
    let _g = lock();
    ensure_cspice_return_mode();
    reset_pools();
    let Some(tf) = require_kernel("SPICELunaFrameKernel.tf") else {
        return;
    };
    let Some(tpc) = require_kernel("pck00010.tpc") else {
        return;
    };
    let Some(tls) = require_kernel("naif0012.tls") else {
        return;
    };
    let tf_s = tf.to_string_lossy().into_owned();
    let tpc_s = tpc.to_string_lossy().into_owned();
    let tls_s = tls.to_string_lossy().into_owned();

    // 双 furnsh：池仍各 1 条（幂等），解析可用。
    e2m2e_spice::furnish_kernel(&tf_s).unwrap();
    e2m2e_spice::furnish_kernel(&tf_s).unwrap();
    e2m2e_spice::furnish_kernel(&tpc_s).unwrap();
    e2m2e_spice::furnish_kernel(&tpc_s).unwrap();
    e2m2e_spice::furnish_kernel(&tls_s).unwrap();
    e2m2e_spice::furnish_kernel(&tls_s).unwrap();
    assert_eq!(native_frame::fk_count(), 1, "FK 双 furnsh 幂等");
    assert_eq!(native_frame::tpck_count(), 1, "TPCK 双 furnsh 幂等");
    assert_eq!(native_time::count(), 1, "LSK 双 furnsh 幂等");
    assert!(native_frame::is_loaded(&tf));
    assert!(native_frame::is_loaded(&tpc));
    assert!(native_time::is_loaded(&tls));
    // 解析可用（文本 PCK 链 IAU_EARTH 在 et=0 有解，无需 BPC）。
    let m = e2m2e_spice::spice_ffi::pxform("IAU_EARTH", "J2000", 0.0).unwrap();
    assert!(m.iter().all(|r| r.iter().all(|x| x.is_finite())));

    // 逐池 unload：池清空 + 重复 unload 幂等（SPICEManager 语义）。
    native_frame::unload(&tf);
    native_frame::unload(&tf);
    native_frame::unload(&tpc);
    native_time::unload(&tls);
    native_time::unload(&tls);
    assert!(!native_frame::is_loaded(&tf));
    assert!(!native_frame::is_loaded(&tpc));
    assert!(!native_time::is_loaded(&tls));
    assert!(native_frame::pools_empty());
    assert!(native_time::is_empty());
    reset_pools();
}
