//! 纯 Rust 帧旋转与文本内核后端（ADR 0052，#685 Phase B）。
//!
//! 把 `pxform`/`sxform` 的剩余 FFI 路径换成纯 Rust：
//!
//! - **池**：`FK_STORE` / `TPCK_STORE` 与 `native_spk::STORE` 同款
//!   （load 幂等 = 先移除旧同路径再追加；查询克隆 `Arc` 快照不持锁求值；
//!   逆序扫描 = 后加载者生效，对齐 CSPICE 内核池）。BPC 段不建新池：
//!   直接读 `native_spk` 快照（卸载/重载语义天然同步）。
//! - **帧图**：`pxform`/`sxform` 的组合语义照搬 vendored CSPICE 的
//!   `refchg.c`（3×3 链）/ `frmchg.c`（6×6 链）：`from` 链上行到 J2000
//!   或命中 `to`；否则 `to` 链上行到首个公共节点；末端用 3×3 转置
//!   （`xpose`）或 6×6 块转置（`invstm`）折入 `from` 链。乘加顺序逐操作
//!   一致——逐位一致的前提。
//! - **PCK 类帧**（class 2）：`tisbod` 语义——先搜 BPC 段
//!   （`pckmat`/`pcke02`：Type 2 + 第三角 mod 2π + `eul2xf(3,1,3)` 重排），
//!   无段则文本 PCK IAU 多项式 + NUT_PREC 级数（`tisbod.c` 逐操作移植）。
//! - **TK 类帧**（class 4）：`tkfram` 语义——MATRIX 原样或 ANGLES
//!   `eul2m` 因子化（单位换算口径与 `convrt` 一致）。
//! - **内置帧**：`J2000`（根）、`ECLIPJ2000`（常值偏移，`chgirf` 口径）、
//!   `ITRF93`（→ class 2 / class_id 3000）。
//!
//! 模块与子模块均挂 `cfg(feature = "spice")`（依赖 `spice_ffi::name_to_id`）。

pub mod bpc2;
pub mod euler;
pub mod text;

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use crate::native_spk::daf::DafSegment;
use crate::native_spk::DafSpkError;

use euler::{eul2m, eul2xf, invstm, msxf, mxm, rotate, xpose};

/// SPICELIB `rpd()`：π/180。
const RPD: f64 = std::f64::consts::PI / 180.0;
/// SPICELIB `halfpi()`。
const HALFPI: f64 = std::f64::consts::FRAC_PI_2;
/// SPICELIB `twopi()`。
const TWOPI: f64 = std::f64::consts::TAU;
/// 秒/日。
const SPD: f64 = 86400.0;
/// 秒/儒略世纪。
const SPC: f64 = SPD * 36525.0;
/// ECLIPJ2000 黄赤交角（角秒，chgirf 内置定义）。
const OBLIQUITY_ARCSEC: f64 = 84381.448;
/// `convrt(角度, "ARCSECONDS", "RADIANS")` 的算子口径：预乘因子
/// π/648000 后一次乘（与 CSPICE 逐位一致的实测口径）。
const ARCSEC_TO_RAD: f64 = std::f64::consts::PI / 648000.0;

/// 帧图求值错误。全部硬失败——不存在回退 CSPICE 的路径。
#[derive(Debug, Clone, PartialEq)]
pub enum NativeFrameError {
    UnknownFrame(String),
    UnsupportedClass {
        frame: String,
        class: i32,
    },
    UnsupportedUnits(String),
    UnsupportedSegmentFrame {
        frame_id: i32,
    },
    UnsupportedPckType(i32),
    NoCoverage {
        frame: String,
        class_id: i32,
        et: f64,
    },
    ChainTooDeep {
        from: String,
        to: String,
    },
    BadKernel(String),
}

impl std::fmt::Display for NativeFrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NativeFrameError::UnknownFrame(n) => {
                write!(f, "NATIVE_FRAME_UNKNOWN_FRAME: 未知参考帧 {n:?}")
            }
            NativeFrameError::UnsupportedClass { frame, class } => write!(
                f,
                "NATIVE_FRAME_UNSUPPORTED_CLASS: 帧 {frame:?} 类别 {class} 不受支持（仅 class 2/4）"
            ),
            NativeFrameError::UnsupportedUnits(u) => {
                write!(f, "NATIVE_FRAME_UNSUPPORTED_UNITS: TK 帧角度单位 {u:?} 不受支持")
            }
            NativeFrameError::UnsupportedSegmentFrame { frame_id } => write!(
                f,
                "NATIVE_FRAME_UNSUPPORTED_SEGMENT_FRAME: BPC 段参考帧 {frame_id} 不受支持（仅 1=J2000 / 17=ECLIPJ2000）"
            ),
            NativeFrameError::UnsupportedPckType(t) => write!(
                f,
                "NATIVE_FRAME_UNSUPPORTED_PCK_TYPE: BPC 段类型 {t} 不受支持（仅 Type 2）"
            ),
            NativeFrameError::NoCoverage { frame, class_id, et } => write!(
                f,
                "NATIVE_FRAME_NO_COVERAGE: 帧 {frame:?}（class_id {class_id}）在 et={et} 无旋转数据（无 BPC 段且文本 PCK 无该体多项式）"
            ),
            NativeFrameError::ChainTooDeep { from, to } => write!(
                f,
                "NATIVE_FRAME_CHAIN_TOO_DEEP: {from:?} → {to:?} 帧链深超过 16（疑似环）"
            ),
            NativeFrameError::BadKernel(msg) => write!(f, "NATIVE_FRAME_BAD_KERNEL: {msg}"),
        }
    }
}

impl std::error::Error for NativeFrameError {}

impl From<DafSpkError> for NativeFrameError {
    fn from(e: DafSpkError) -> Self {
        match e {
            DafSpkError::UnsupportedType(t) => NativeFrameError::UnsupportedPckType(t),
            other => NativeFrameError::BadKernel(other.to_string()),
        }
    }
}

// ---------------------------------------------------------------------------
// 池
// ---------------------------------------------------------------------------

static FK_STORE: RwLock<Vec<(PathBuf, Arc<text::FkFile>)>> = RwLock::new(Vec::new());
static TPCK_STORE: RwLock<Vec<(PathBuf, Arc<text::TpckFile>)>> = RwLock::new(Vec::new());

fn fk_store() -> std::sync::RwLockReadGuard<'static, Vec<(PathBuf, Arc<text::FkFile>)>> {
    FK_STORE.read().unwrap_or_else(|e| e.into_inner())
}

fn tpck_store() -> std::sync::RwLockReadGuard<'static, Vec<(PathBuf, Arc<text::TpckFile>)>> {
    TPCK_STORE.read().unwrap_or_else(|e| e.into_inner())
}

/// 解析并加载 FK 到池。重复 load 同一路径幂等（先移除旧条目再追加，
/// 保持「后加载者生效」）。解析/语义错误 → `BadKernel` 系硬错误。
pub fn load_fk(path: &Path, content: &str) -> Result<(), NativeFrameError> {
    let pool = text::parse_pool(content).map_err(|e| NativeFrameError::BadKernel(e.to_string()))?;
    let fk =
        text::FkFile::from_pool(&pool).map_err(|e| NativeFrameError::BadKernel(e.to_string()))?;
    // FK 重定义内置名 → 硬报错（内置表优先级高于 FK，重定义只会造成
    // 「装了不生效」的错觉，直接拒绝）。
    for name in fk.name_to_id.keys() {
        if is_builtin_name(name) {
            return Err(NativeFrameError::BadKernel(format!(
                "FK 重定义内置帧名 {name:?}（内置表不可覆盖）"
            )));
        }
    }
    let mut store = FK_STORE.write().unwrap_or_else(|e| e.into_inner());
    store.retain(|(p, _)| p != path);
    store.push((path.to_path_buf(), Arc::new(fk)));
    Ok(())
}

/// 解析并加载文本 PCK 到池。重复 load 同一路径幂等。
pub fn load_tpck(path: &Path, content: &str) -> Result<(), NativeFrameError> {
    let pool = text::parse_pool(content).map_err(|e| NativeFrameError::BadKernel(e.to_string()))?;
    let tpck =
        text::TpckFile::from_pool(&pool).map_err(|e| NativeFrameError::BadKernel(e.to_string()))?;
    let mut store = TPCK_STORE.write().unwrap_or_else(|e| e.into_inner());
    store.retain(|(p, _)| p != path);
    store.push((path.to_path_buf(), Arc::new(tpck)));
    Ok(())
}

/// 从文本池卸载一个内核（幂等，未加载时静默跳过）。
pub fn unload(path: &Path) {
    FK_STORE
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .retain(|(p, _)| p != path);
    TPCK_STORE
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .retain(|(p, _)| p != path);
}

/// 指定路径是否已在任一文本池（furnish 回滚区分「本次新增」用）。
pub fn is_loaded(path: &Path) -> bool {
    fk_store().iter().any(|(p, _)| p == path) || tpck_store().iter().any(|(p, _)| p == path)
}

/// 两个文本池是否全空（pxform/et2utc 入口预检用）。
pub fn pools_empty() -> bool {
    fk_store().is_empty() && tpck_store().is_empty()
}

/// FK 池条目计数（ktotal 语义）。
pub fn fk_count() -> usize {
    fk_store().len()
}

/// 文本 PCK 池条目计数（ktotal 语义）。
pub fn tpck_count() -> usize {
    tpck_store().len()
}

// ---------------------------------------------------------------------------
// 帧名解析
// ---------------------------------------------------------------------------

/// 内置帧名（不可被 FK 重定义）。
fn is_builtin_name(upper: &str) -> bool {
    matches!(upper, "J2000" | "ECLIPJ2000" | "ITRF93")
}

/// 解析后的帧节点。
#[derive(Debug, Clone)]
enum Node {
    /// 惯性根（frame 1）。
    J2000,
    /// ECLIPJ2000（frame 17）：到 J2000 的常值旋转。
    EclipJ2000,
    /// class 2 PCK 帧：`tisbod` 语义（BPC 段优先，回退文本池）。
    Pck { class_id: i32, name: String },
    /// class 4 TK 帧：常值旋转到 `relative`。
    Tk { name: String, spec: text::TkSpec },
}

impl Node {
    fn name(&self) -> String {
        match self {
            Node::J2000 => "J2000".into(),
            Node::EclipJ2000 => "ECLIPJ2000".into(),
            Node::Pck { name, .. } => name.clone(),
            Node::Tk { name, .. } => name.clone(),
        }
    }
}

/// 帧名（大小写不敏感）→ 节点。顺序：内置表 → FK 池 `FRAME_<NAME>` →
/// `IAU_<BODY>` 模式 → 否则 `UnknownFrame`。
fn resolve(name: &str) -> Result<Node, NativeFrameError> {
    let upper = name.trim().to_ascii_uppercase();
    match upper.as_str() {
        "J2000" => return Ok(Node::J2000),
        "ECLIPJ2000" => return Ok(Node::EclipJ2000),
        "ITRF93" => {
            return Ok(Node::Pck {
                class_id: 3000,
                name: "ITRF93".into(),
            });
        }
        _ => {}
    }
    // FK 池：逆序（后加载者生效）。
    {
        let store = fk_store();
        for (_, fk) in store.iter().rev() {
            if let Some(id) = fk.name_to_id.get(&upper) {
                let Some(def) = fk.frames.get(id) else {
                    return Err(NativeFrameError::BadKernel(format!(
                        "FK 定义 FRAME_{upper} = {id} 但缺帧规格关键字"
                    )));
                };
                return match def.class {
                    2 => Ok(Node::Pck {
                        class_id: def.class_id,
                        name: upper.clone(),
                    }),
                    4 => {
                        let spec = def.tk.clone().ok_or_else(|| {
                            NativeFrameError::BadKernel(format!(
                                "FK 帧 {upper:?}（class 4）缺 TKFRAME 规格"
                            ))
                        })?;
                        Ok(Node::Tk {
                            name: upper.clone(),
                            spec,
                        })
                    }
                    class => Err(NativeFrameError::UnsupportedClass {
                        frame: upper.clone(),
                        class,
                    }),
                };
            }
        }
    }
    // IAU_<BODY> 模式：class_id = 天体本体 ID（IAU_EARTH→399、IAU_MOON→301，
    // 与 pck00010 的 BODY399_/BODY301_ 键一致）。
    if let Some(body_name) = upper.strip_prefix("IAU_") {
        if let Some(class_id) = iau_body_id(body_name) {
            return Ok(Node::Pck {
                class_id,
                name: upper.clone(),
            });
        }
    }
    Err(NativeFrameError::UnknownFrame(name.trim().to_string()))
}

/// `IAU_<天体名>` → class_id（天体**本体**号，非质心号）。
///
/// 与 [`crate::spice_ffi::name_to_id`] 的质心映射刻意不同：IAU_* 帧的
/// class_id 是天体本体号（pck00010 的 BODY199_/BODY299_/… 键）；CSPICE 的
/// `bodn2c` 对行星名返回本体号（MERCURY→199），而 BODY_ALIASES 为第三体
/// 摄动把行星名映射到质心号。本表只服务 IAU_* 帧解析，数字串按 NAIF ID
/// 直取（IAU_199 语义）。
fn iau_body_id(name: &str) -> Option<i32> {
    match name {
        "MERCURY" => Some(199),
        "VENUS" => Some(299),
        "EARTH" => Some(399),
        "MARS" => Some(499),
        "JUPITER" => Some(599),
        "SATURN" => Some(699),
        "URANUS" => Some(799),
        "NEPTUNE" => Some(899),
        "PLUTO" => Some(999),
        "MOON" => Some(301),
        "SUN" => Some(10),
        other => other.parse::<i32>().ok(),
    }
}

// ---------------------------------------------------------------------------
// 腿（leg）求值
// ---------------------------------------------------------------------------

/// 一条上行的链腿：`X(node → parent)`。3×3 与 6×6 形式一并给出——
/// `pxform` 走 3×3 链（refchg）、`sxform` 走 6×6 链（frmchg），两者在
/// CSPICE 内是独立组合路径，数值上不保证相同。
#[derive(Debug, Clone)]
struct Leg {
    r3: [[f64; 3]; 3],
    x6: [[f64; 6]; 6],
}

impl Leg {
    /// 常值旋转腿（TK / ECLIPJ2000）：6×6 = [[r, 0], [0, r]]
    /// （frmget class 1/4 的 6×6 装配原样）。
    fn constant(r3: [[f64; 3]; 3]) -> Leg {
        let mut x6 = [[0.0_f64; 6]; 6];
        for i in 0..3 {
            for j in 0..3 {
                x6[i][j] = r3[i][j];
                x6[i + 3][j + 3] = r3[i][j];
            }
        }
        Leg { r3, x6 }
    }
}

/// 节点 → (上行腿, 父节点名)。
fn node_leg(node: &Node, et: f64) -> Result<(Leg, String), NativeFrameError> {
    match node {
        Node::J2000 => Err(NativeFrameError::BadKernel(
            "J2000 是惯性根，无上行腿".into(),
        )),
        Node::EclipJ2000 => {
            // chgirf：trans(17) = R(1, +ε)（「J2000 → 帧」）；irfrot(17, 1)
            // = trans(17)ᵀ = X(ECLIPJ2000 → J2000)（et=0 oracle 实测钉死）。
            let eps = OBLIQUITY_ARCSEC * ARCSEC_TO_RAD;
            Ok((Leg::constant(xpose(&rotate(1, eps))), "J2000".into()))
        }
        Node::Tk { spec, .. } => {
            // tkfram.c 的 ANGLES 路径：eul2m(angles[0], angles[1], angles[2],
            // axes[0], axes[1], axes[2]) —— C 的 eul2m(angle3,…,axis3,…) 把
            // angles[0] 作为最左因子；本仓 eul2m(a1,…) 把 a1 作为最右因子，
            // 故实参按位反转。
            let r3 = match spec {
                text::TkSpec::Matrix { m, .. } => *m,
                text::TkSpec::Angles {
                    angles_rad, axes, ..
                } => eul2m(
                    angles_rad[2],
                    angles_rad[1],
                    angles_rad[0],
                    axes[2],
                    axes[1],
                    axes[0],
                ),
            };
            let parent = match spec {
                text::TkSpec::Matrix { relative, .. } => relative.clone(),
                text::TkSpec::Angles { relative, .. } => relative.clone(),
            };
            Ok((Leg::constant(r3), parent.to_ascii_uppercase()))
        }
        Node::Pck { class_id, name } => {
            let tsipm = pck_tsipm(*class_id, name, et)?;
            // `zzrotgt0` 的 `xpose(tipm)`（3×3）/ `frmget` 的
            // `invstm(tsipm)`（6×6）。
            let mut r3 = [[0.0_f64; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    r3[i][j] = tsipm[i][j];
                }
            }
            Ok((
                Leg {
                    r3: xpose(&r3),
                    x6: invstm(&tsipm),
                },
                "J2000".into(),
            ))
        }
    }
}

/// `tisbod("J2000", class_id, et)`：返回 `X(J2000 → 体固)` 6×6。
/// BPC 段优先（`pckmat`），无段回退文本 PCK（IAU 多项式）。
fn pck_tsipm(class_id: i32, name: &str, et: f64) -> Result<[[f64; 6]; 6], NativeFrameError> {
    if let Some((bytes, seg)) = find_bpc_segment(class_id, et) {
        if seg.dtype != 2 {
            return Err(NativeFrameError::UnsupportedPckType(seg.dtype));
        }
        let mut tsipm = bpc2::tsipm(&bytes, &seg, et)?;
        // tisbod 的 pcref 调整：段参考系非 J2000 时把常值阵折进矩阵
        // （tipm·req2pc / dtipm·req2pc，块级 mxm 原样）。
        if seg.frame != 1 {
            if seg.frame != 17 {
                return Err(NativeFrameError::UnsupportedSegmentFrame {
                    frame_id: seg.frame,
                });
            }
            // irfrot(1, 17) = trans(17)·trans(1)ᵀ = X(J2000 → ECLIPJ2000)
            // = rotate(1, +ε)（未转置）。
            let eps = OBLIQUITY_ARCSEC * ARCSEC_TO_RAD;
            let req2pc = rotate(1, eps);
            let mut tipm = [[0.0_f64; 3]; 3];
            let mut dtipm = [[0.0_f64; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    tipm[i][j] = tsipm[i][j];
                    dtipm[i][j] = tsipm[i + 3][j];
                }
            }
            let xtipm = mxm(&tipm, &req2pc);
            let xdtipm = mxm(&dtipm, &req2pc);
            for i in 0..3 {
                for j in 0..3 {
                    tsipm[i][j] = xtipm[i][j];
                    tsipm[i + 3][j + 3] = xtipm[i][j];
                    tsipm[i + 3][j] = xdtipm[i][j];
                }
            }
        }
        return Ok(tsipm);
    }
    text_pck_tsipm(class_id, name, et)
}

/// BPC 段搜索：`native_spk` 快照逆序（内核逆加载序、段逆文件内顺序），
/// 首个 `body == class_id && begin <= et <= end` 命中者胜出
/// （`pcksfs` 后加载者生效）。返回 (文件字节, 段描述符)。
fn find_bpc_segment(class_id: i32, et: f64) -> Option<(Arc<Vec<u8>>, DafSegment)> {
    for (bytes, seg) in crate::native_spk::bpc_segments().iter().rev() {
        if seg.body == class_id && seg.center.is_none() && seg.dc0 <= et && et <= seg.dc1 {
            return Some((Arc::clone(bytes), seg.clone()));
        }
    }
    None
}

/// 文本 PCK 求值（`tisbod.c` 文本分支逐操作移植）：
/// `X(J2000 → 体固)` 6×6。
fn text_pck_tsipm(class_id: i32, name: &str, et: f64) -> Result<[[f64; 6]; 6], NativeFrameError> {
    // 快照：读锁下克隆 Arc 随即释放，求值不持锁。
    let tpck = {
        let store = tpck_store();
        store
            .iter()
            .rev()
            .find(|(_, t)| t.bodies.contains_key(&class_id))
            .map(|(_, t)| Arc::clone(t))
    };
    let Some(tpck) = tpck else {
        return Err(NativeFrameError::NoCoverage {
            frame: name.to_string(),
            class_id,
            et,
        });
    };
    let body = &tpck.bodies[&class_id];
    // NUT_PREC_ANGLES 按 `zzbodbry`（百位收缩到系统质心）查表；
    // 级数系数按天体本键查（bodvcd/bodfnd 的 body）。
    let refid = barycenter(class_id);
    let angles = tpck.nut_prec_angles.get(&refid);
    let na = body.nut_ra.as_ref().map_or(0, Vec::len);
    let nd = body.nut_dec.as_ref().map_or(0, Vec::len);
    let nw = body.nut_pm.as_ref().map_or(0, Vec::len);
    let nphase = angles.map_or(0, Vec::len);
    let nneeded = na.max(nd).max(nw);
    if nneeded > nphase {
        return Err(NativeFrameError::BadKernel(format!(
            "NATIVE_FRAME_INSUFFICIENT_ANGLES: 天体 {class_id} 的 NUT_PREC 级数需要 {nneeded} 个相位角，BODY{refid}_NUT_PREC_ANGLES 仅提供 {nphase}"
        )));
    }

    // 常数纪元缺省 J2000（BODY#_CONSTANTS_JED_EPOCH 不支持，解析层硬错）。
    // td = 日数、tc = 世纪数（tisbod.c：td = epoch/d，tc = epoch/t）。
    let td = et / SPD;
    let tc = et / SPC;

    // 时间多项式（tisbod.c 原样的 Horner 一次式）与导数：
    // RA/DEC 用 tc，W 用 td；导数除以 t（或 d）换算成每秒。
    let ra0 = body.ra[0] + tc * (body.ra[1] + tc * body.ra[2]);
    let dec0 = body.dec[0] + tc * (body.dec[1] + tc * body.dec[2]);
    let w0 = body.pm[0] + td * (body.pm[1] + td * body.pm[2]);
    let mut ra = ra0;
    let mut dec = dec0;
    let mut w = w0;
    let mut dra = (body.ra[1] + (tc * 2.) * body.ra[2]) / SPC;
    let mut ddec = (body.dec[1] + (tc * 2.) * body.dec[2]) / SPC;
    let mut dw = (body.pm[1] + (td * 2.) * body.pm[2]) / SPD;

    // 相位角（nphsco = 2 的公共分支）：θ = (a + tc·b)·rpd，
    // dθ = b/t·rpd；正弦/余弦及其导数入表。
    let mut sinth = vec![0.0_f64; nphase];
    let mut costh = vec![0.0_f64; nphase];
    let mut dsinth = vec![0.0_f64; nphase];
    let mut dcosth = vec![0.0_f64; nphase];
    if let Some(angles) = angles {
        for (i, pair) in angles.iter().enumerate() {
            let theta = (pair[0] + tc * pair[1]) * RPD;
            let dtheta = pair[1] / SPC * RPD;
            let s = theta.sin();
            let c = theta.cos();
            sinth[i] = s;
            costh[i] = c;
            dsinth[i] = c * dtheta;
            dcosth[i] = -s * dtheta;
        }
    }
    // 级数修正（vdotg 语义：先整段点积再叠加）。
    // DEC 配 cos、RA/PM 配 sin（pck.req 口径，tisbod.c 原样）。
    if let Some(ac) = body.nut_ra.as_deref() {
        let mut dot = 0.0_f64;
        for j in 0..na {
            dot += ac[j] * sinth[j];
        }
        ra += dot;
        let mut ddot = 0.0_f64;
        for j in 0..na {
            ddot += ac[j] * dsinth[j];
        }
        dra += ddot;
    }
    if let Some(dc) = body.nut_dec.as_deref() {
        let mut dot = 0.0_f64;
        for j in 0..nd {
            dot += dc[j] * costh[j];
        }
        dec += dot;
        let mut ddot = 0.0_f64;
        for j in 0..nd {
            ddot += dc[j] * dcosth[j];
        }
        ddec += ddot;
    }
    if let Some(wc) = body.nut_pm.as_deref() {
        let mut dot = 0.0_f64;
        for j in 0..nw {
            dot += wc[j] * sinth[j];
        }
        w += dot;
        let mut ddot = 0.0_f64;
        for j in 0..nw {
            ddot += wc[j] * dsinth[j];
        }
        dw += ddot;
    }

    // 度 → 弧度，W mod 2π，转 Euler（φ = RA + π/2，δ = π/2 − DEC）。
    let ra = ra * RPD;
    let dec = dec * RPD;
    let mut w = w * RPD;
    let dra = dra * RPD;
    let ddec = ddec * RPD;
    let dw = dw * RPD;
    w = euler::pos_mod(w, TWOPI);
    let phi = ra + HALFPI;
    let delta = HALFPI - dec;

    // Euler 状态包 = [w, δ, φ, dw, dδ, dφ]（vpack 原样），3-1-3 因子化。
    let eulsta = [w, delta, phi, dw, -ddec, dra];
    Ok(eul2xf(&eulsta, 3, 1, 3))
}

/// `zzbodbry`：行星系统天体号收缩到系统质心号。
fn barycenter(body: i32) -> i32 {
    if (100..=999).contains(&body) {
        body / 100
    } else if (10000..=99999).contains(&body) {
        body / 10000
    } else {
        body
    }
}

// ---------------------------------------------------------------------------
// pxform / sxform（refchg / frmchg 组合语义）
// ---------------------------------------------------------------------------

/// 上行链：从 `start` 走到 J2000 或命中 `stop`。返回 (帧名链, 腿链)；
/// 帧名链含起点与终点（终点 = J2000 或 stop），腿链比帧名链短一。
fn walk_chain(
    start: &Node,
    stop: Option<&str>,
    et: f64,
) -> Result<(Vec<String>, Vec<Leg>), NativeFrameError> {
    let mut frames = vec![start.name()];
    let mut legs = Vec::new();
    let mut node = start.clone();
    loop {
        if matches!(node, Node::J2000) {
            return Ok((frames, legs));
        }
        if let Some(stop) = stop {
            if frames.last().map(|f| f.as_str() == stop).unwrap_or(false) {
                return Ok((frames, legs));
            }
        }
        if legs.len() >= 16 {
            return Err(NativeFrameError::ChainTooDeep {
                from: frames[0].clone(),
                to: stop.unwrap_or("J2000").to_string(),
            });
        }
        let (leg, parent) = node_leg(&node, et)?;
        legs.push(leg);
        node = resolve(&parent)?;
        frames.push(node.name());
    }
}

/// 折叠腿链（右折叠，等价 `zzrxr`/`zzmsxf` 的「后项·前项」链乘）：
/// acc = L1；acc = Lk·acc；空链 = 单位阵。
fn fold3(legs: &[[[f64; 3]; 3]]) -> [[f64; 3]; 3] {
    let mut acc = match legs.first() {
        Some(l) => *l,
        None => return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };
    for leg in &legs[1..] {
        acc = mxm(leg, &acc);
    }
    acc
}

/// [`fold3`] 的 6×6 版本。
fn fold6(legs: &[[[f64; 6]; 6]]) -> [[f64; 6]; 6] {
    let mut acc = match legs.first() {
        Some(l) => *l,
        None => {
            let mut x = [[0.0_f64; 6]; 6];
            for (i, row) in x.iter_mut().enumerate() {
                row[i] = 1.0;
            }
            return x;
        }
    };
    for leg in &legs[1..] {
        acc = msxf(leg, &acc);
    }
    acc
}

/// `pxform`：返回 `from → to` 在 `et` 时刻的 3×3 旋转矩阵（行主序）。
///
/// 组合语义照搬 `refchg.c`：`from` 链上行至 J2000 或命中 `to`；命中即
/// 直接折叠；否则 `to` 链上行至首个公共节点，末端 `xpose` 折入。
pub fn pxform(from: &str, to: &str, et: f64) -> Result<[[f64; 3]; 3], NativeFrameError> {
    let from_node = resolve(from)?;
    let to_node = resolve(to)?;
    let to_name = to_node.name();
    if from_node.name() == to_name {
        return Ok([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    }
    let (frames1, legs1) = walk_chain(&from_node, Some(&to_name), et)?;
    if frames1
        .last()
        .map(|f| f.as_str() == to_name)
        .unwrap_or(false)
    {
        // from 链直接命中 to（CSPICE：frame[node] == frame2 →
        // zzrxr(rot, node-1)）。
        let mut acc = legs1[0].r3;
        for leg in &legs1[1..] {
            acc = mxm(&leg.r3, &acc);
        }
        return Ok(acc);
    }
    // from 链到根（J2000）：走 to 链找首个公共节点。
    let (frames2, legs2) = walk_chain(&to_node, None, et)?;
    let pos2 = frames2
        .iter()
        .position(|f| frames1.iter().any(|g| g == f))
        .ok_or_else(|| NativeFrameError::BadKernel(format!("帧链无公共节点: {from:?} → {to:?}")))?;
    // 公共节点在 from 链上的位置（CSPICE zzrxr(rot, cmnode)：rot[1..cmnode]
    // 是 from 链到公共节点的腿，rot[cmnode] 被 xpose 的 to 链覆盖）。
    let pos1 = frames1
        .iter()
        .position(|f| *f == frames2[pos2])
        .ok_or_else(|| NativeFrameError::BadKernel(format!("帧链无公共节点: {from:?} → {to:?}")))?;
    // to 链折叠到公共节点：X(to → common)。
    let r2: Vec<[[f64; 3]; 3]> = legs2[..pos2].iter().map(|l| l.r3).collect();
    let to_common = fold3(&r2);
    // from 链折叠到公共节点，末端 xpose(to_common) 折入
    // （refchg.c：xpose(rot2) → rot[cmnode] → zzrxr(rot, cmnode)）。
    let mut r1: Vec<[[f64; 3]; 3]> = legs1[..pos1].iter().map(|l| l.r3).collect();
    r1.push(xpose(&to_common));
    Ok(fold3(&r1))
}

/// [`pxform`] 的 6×6 版本（`frmchg.c` 组合语义：末端 `invstm` 折入）。
pub fn sxform(from: &str, to: &str, et: f64) -> Result<[[f64; 6]; 6], NativeFrameError> {
    let from_node = resolve(from)?;
    let to_node = resolve(to)?;
    let to_name = to_node.name();
    if from_node.name() == to_name {
        let mut x = [[0.0_f64; 6]; 6];
        for (i, row) in x.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        return Ok(x);
    }
    let (frames1, legs1) = walk_chain(&from_node, Some(&to_name), et)?;
    if frames1
        .last()
        .map(|f| f.as_str() == to_name)
        .unwrap_or(false)
    {
        let mut acc = legs1[0].x6;
        for leg in &legs1[1..] {
            acc = msxf(&leg.x6, &acc);
        }
        return Ok(acc);
    }
    let (frames2, legs2) = walk_chain(&to_node, None, et)?;
    let pos2 = frames2
        .iter()
        .position(|f| frames1.iter().any(|g| g == f))
        .ok_or_else(|| NativeFrameError::BadKernel(format!("帧链无公共节点: {from:?} → {to:?}")))?;
    let pos1 = frames1
        .iter()
        .position(|f| *f == frames2[pos2])
        .ok_or_else(|| NativeFrameError::BadKernel(format!("帧链无公共节点: {from:?} → {to:?}")))?;
    let x2: Vec<[[f64; 6]; 6]> = legs2[..pos2].iter().map(|l| l.x6).collect();
    let to_common = fold6(&x2);
    let mut x1: Vec<[[f64; 6]; 6]> = legs1[..pos1].iter().map(|l| l.x6).collect();
    x1.push(invstm(&to_common));
    Ok(fold6(&x1))
}
