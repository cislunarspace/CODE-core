//! 纯 Rust 星历注册表与链式求值（ADR 0051，#639 Phase A）。
//!
//! - 注册表：进程级 `RwLock<Vec<Arc<LoadedKernel>>>`，按**加载顺序**存放；
//!   查询路径在读锁下克隆 `Arc` 快照，求值期间不持锁（Rayon 并行安全）。
//! - 选段优先级复刻 CSPICE `spksfs` 的「后加载者生效」：内核按逆加载序、
//!   段按逆文件内顺序扫描，首个 `body == 节点 && dc0 <= et <= dc1` 的段胜出。
//! - 链式求值复刻 `spkgeo.c` 的**首个公共节点**语义：target 与 observer 各自
//!   沿段图 `center` 向上累加状态到首个公共节点后相减。加法折叠顺序与
//!   `spkgeo.c` 逐操作一致（逐位一致的关键，不得改成各自归算 SSB 再相减）。
//! - BPC 段（`center: None`）不参与选段；`dtype != 2`、`frame != 1` 命中即
//!   硬报错，任何情况不回退 CSPICE（ADR 0020）。

pub mod daf;
pub mod spk2;

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// 解析/求值错误。全部硬失败——不存在回退 CSPICE 的路径。
#[derive(Debug, Clone, PartialEq)]
pub enum DafSpkError {
    /// 非 DAF 容器（`locidw != "DAF/"`）。文本内核（tls/tpc/tf）属预期情况，
    /// `furnish_kernel` 据此跳过 native 登记。
    NotDaf,
    /// 文件读取失败。
    Io(String),
    /// 容器格式错误（控制字越界、链断裂、数据区越界等）。
    BadFormat(String),
    /// 命中了不支持的 SPK data type（Phase A 只做 Type 2）。
    UnsupportedType(i32),
    /// 命中了不支持的参考系（Phase A 只做 J2000，ID=1）。
    UnsupportedFrame(i32),
    /// 走不到公共节点或无覆盖段。
    NoSegment { target: i32, observer: i32, et: f64 },
}

impl std::fmt::Display for DafSpkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DafSpkError::NotDaf => write!(f, "非 DAF 容器（文本内核或他类文件）"),
            DafSpkError::Io(msg) => write!(f, "内核文件读取失败: {msg}"),
            DafSpkError::BadFormat(msg) => write!(f, "DAF/SPK 容器格式错误: {msg}"),
            DafSpkError::UnsupportedType(t) => write!(
                f,
                "SPK_NATIVE_UNSUPPORTED_TYPE: 本地 SPK 后端仅支持 Type 2，命中 Type {t}"
            ),
            DafSpkError::UnsupportedFrame(id) => write!(
                f,
                "SPK_NATIVE_UNSUPPORTED_FRAME: 本地 SPK 后端仅支持 J2000(frame 1)，命中 frame {id}"
            ),
            DafSpkError::NoSegment {
                target,
                observer,
                et,
            } => write!(
                f,
                "本地 SPK 后端无法在 et={et} 求 {target} 相对 {observer} 的状态：无覆盖段或走不到公共节点"
            ),
        }
    }
}

impl std::error::Error for DafSpkError {}

/// 已加载内核：路径、整文件字节与解析结果。
pub struct LoadedKernel {
    pub path: PathBuf,
    pub bytes: Arc<Vec<u8>>,
    pub file: daf::DafFile,
}

/// 进程级内核注册表。查询侧只克隆 `Arc` 快照，不持锁求值。
static STORE: RwLock<Vec<Arc<LoadedKernel>>> = RwLock::new(Vec::new());

fn store() -> std::sync::RwLockReadGuard<'static, Vec<Arc<LoadedKernel>>> {
    STORE.read().unwrap_or_else(|e| e.into_inner())
}

/// 加载一个 DAF 内核到 native 注册表。重复 load 同一路径幂等（先移除旧条目
/// 再追加，保持「后加载者生效」优先级）。文本内核返回
/// [`DafSpkError::NotDaf`]，由 `furnish_kernel` 决定是否视为跳过。
pub fn load(path: &Path) -> Result<(), DafSpkError> {
    let bytes =
        std::fs::read(path).map_err(|e| DafSpkError::Io(format!("{}: {e}", path.display())))?;
    let file = daf::parse(path, &bytes)?;
    let mut store = STORE.write().unwrap_or_else(|e| e.into_inner());
    store.retain(|k| k.path != path);
    store.push(Arc::new(LoadedKernel {
        path: path.to_path_buf(),
        bytes: Arc::new(bytes),
        file,
    }));
    Ok(())
}

/// 从 native 注册表卸载一个内核。未加载时静默跳过（与 `spice_unload` 语义
/// 一致，幂等）。
pub fn unload(path: &Path) {
    let mut store = STORE.write().unwrap_or_else(|e| e.into_inner());
    store.retain(|k| k.path != path);
}

/// 注册表是否为空（`spkezr` 入口预检用）。
pub fn is_empty() -> bool {
    store().is_empty()
}

/// 指定路径是否已在注册表中（`furnish_kernel` 回滚时区分「本次新增」与
/// 「此前已登记」用）。
pub fn is_loaded(path: &Path) -> bool {
    store().iter().any(|k| k.path == path)
}

/// 全部已注册段的只读快照（测试与对拍用）。
pub fn segments() -> Vec<(PathBuf, daf::DafSegment)> {
    store()
        .iter()
        .flat_map(|k| {
            k.file
                .segments
                .iter()
                .cloned()
                .map(move |s| (k.path.clone(), s))
        })
        .collect()
}

/// 在快照内选段：内核逆加载序、段逆文件内顺序，首个命中者胜出
/// （`spksfs` 后加载者生效）。BPC 段（`center: None`）不参与。
fn find_segment(
    kernels: &[Arc<LoadedKernel>],
    node: i32,
    et: f64,
) -> Option<(&LoadedKernel, &daf::DafSegment)> {
    for kernel in kernels.iter().rev() {
        for seg in kernel.file.segments.iter().rev() {
            if seg.body == node && seg.center.is_some() && seg.dc0 <= et && et <= seg.dc1 {
                return Some((kernel, seg));
            }
        }
    }
    None
}

/// 求一段（命中段）在 `et` 的状态；Type/Frame 不支持即硬报错。
fn evaluate_leg(
    kernel: &LoadedKernel,
    seg: &daf::DafSegment,
    et: f64,
) -> Result<[f64; 6], DafSpkError> {
    if seg.dtype != 2 {
        return Err(DafSpkError::UnsupportedType(seg.dtype));
    }
    if seg.frame != 1 {
        return Err(DafSpkError::UnsupportedFrame(seg.frame));
    }
    spk2::evaluate(&kernel.bytes, seg, et)
}

/// 求链式状态：`target` 相对 `observer` 在 J2000 下的几何状态
/// （km, km/s），语义与 CSPICE `spkgeo`（abcorr=NONE）逐位一致。
pub fn state(target: i32, et: f64, observer: i32) -> Result<[f64; 6], DafSpkError> {
    if target == observer {
        return Ok([0.0; 6]);
    }
    let kernels: Vec<Arc<LoadedKernel>> = store().to_vec(); // 读锁下克隆 Arc 快照，随即释放

    // ── target 链（spkgeo.c：ctarg/starg，至多 20 个节点）──
    // 链深 ≥20 时 CSPICE 走 `if (i__ == 20)` 分支继续上溯并把新腿折入
    // STARG(20)（"settle for the last common node"），本实现到此截断（真实
    // 内核链深 ≤3，不可达；ADR 0051 记为已接受简化）。
    let mut nodes: Vec<i32> = vec![target];
    let mut starg: Vec<[f64; 6]> = vec![[0.0; 6]]; // STARG(1) 恒为零向量
    let mut found = true;
    while found
        && nodes.len() < 20
        && *nodes.last().unwrap() != observer
        && *nodes.last().unwrap() != 0
    {
        let node = *nodes.last().unwrap();
        match find_segment(&kernels, node, et) {
            Some((kernel, seg)) => {
                let leg = evaluate_leg(kernel, seg, et)?;
                nodes.push(seg.center.unwrap());
                starg.push(leg);
            }
            None => found = false,
        }
    }
    let nct = nodes.len();
    // 首个公共节点：若 target 链末端恰为 observer，公共节点位置 = nct。
    let mut ctpos = if nodes[nct - 1] == observer { nct } else { 0 };

    // ── observer 链（spkgeo.c：cobs/sobs）──
    // spkgeo.c 在观察者循环前无条件 `found = TRUE_`（spkgeo.c:978）：target 链
    // 中途断链（无覆盖段）不影响观察者侧继续上溯找公共节点。
    found = true;
    let mut cobs = observer;
    let mut sobs = [0.0f64; 6];
    let mut legs = 0usize;
    while found && cobs != 0 && ctpos == 0 {
        match find_segment(&kernels, cobs, et) {
            Some((kernel, seg)) => {
                let leg = evaluate_leg(kernel, seg, et)?;
                // legs == 0：直接赋值（spkgeo.c 原样）；之后逐腿累加。
                if legs == 0 {
                    sobs = leg;
                } else {
                    for (s, l) in sobs.iter_mut().zip(leg) {
                        *s += l;
                    }
                }
                legs += 1;
                // spkgeo.c：cobs 先更新为该段 center，再查它是否已在
                // target 链上（isrchi 顺序查找，命中即首个公共节点）。
                cobs = seg.center.unwrap();
                ctpos = nodes.iter().position(|&n| n == cobs).map_or(0, |i| i + 1);
            }
            None => break,
        }
    }

    if ctpos == 0 {
        return Err(DafSpkError::NoSegment {
            target,
            observer,
            et,
        });
    }

    // ── 折叠 target 链：STARG(I+1) += STARG(I)，I = 2..=CTPOS-1（spkgeo.c
    // 原样，加法顺序即逐位一致的关键）──
    for k in 2..ctpos {
        let prev = starg[k - 1];
        for (acc, p) in starg[k].iter_mut().zip(prev) {
            *acc += p;
        }
    }
    // state = STARG(CTPOS) - SOBS（vsubg）。
    let mut state = [0.0f64; 6];
    for (out, (acc, obs)) in state.iter_mut().zip(starg[ctpos - 1].iter().zip(sobs)) {
        *out = acc - obs;
    }
    Ok(state)
}
