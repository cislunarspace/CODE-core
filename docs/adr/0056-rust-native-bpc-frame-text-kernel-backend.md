# ADR 0056: 纯 Rust BPC Type 2 帧旋转 + FK/文本 PCK/LSK 内核后端（#685 Phase B）

**状态**：已采纳（已实现）
**日期**：2026-09-27
**相关 issue**：#685（本决策的直接动机）、#684/ADR 0051（Phase A 前置）
**相关**：ADR 0020（失败显式化）、ADR 0039（共享内核叶）、ADR 0051（Phase A：DAF +
SPK Type 2 + spkezr 零 FFI）

## 背景

Phase A（ADR 0051）把 `spkezr` 换成纯 Rust 后端后，Rust 侧仍跨 CSPICE FFI 的入口剩
`pxform`/`sxform`（帧旋转）、`et2utc`（时间转换）与 `ktotal`（内核计数）。它们依赖
CSPICE 的进程级内核池与帧图，是并行区 StrictGuard 之外最后的 cspice 面本任务把这一
面也换成纯 Rust，`spice_ffi` 生产入口自此零 FFI。

行为地面真值是 vendored CSPICE（mice_windows）的 C 源：`pckr02.c`/`pcke02.c`/
`pckmat.c`/`tisbod.c`（文本 PCK 求值在其中，不经 bodeul）/`tkfram.c`/`chgirf.c`/
`refchg.c`/`frmchg.c`/`zzrxr.c`/`zzmsxf.c`/`invstm.c`/`deltet.c`/`unitim.c`/
`et2utc.c`/`ttrans.c`/`d_mod.c`。全部**逐操作移植**，操作顺序不得重排。

## 决策

### 1. 新模块与复用面

- `crates/e2m2e-spice/src/native_frame/`：`euler.rs`（rotate/rotmat/eul2m/eul2xf/
  pos_mod）、`bpc2.rs`（BPC Type 2 求值）、`text.rs`（文本内核解析器 + FK/TPCK/LSK
  解释器）、`mod.rs`（池、帧图、pxform/sxform）。
- `crates/e2m2e-spice/src/native_time/mod.rs`：deltet + et2utc（ISOC）。
- `spk2.rs::chbint` 改 `pub(crate)` 供 bpc2 复用（pcke02 与 spke02 同构）；DAF 段
  直接读 `native_spk::bpc_segments()` 快照，**BPC 不建新池**——卸载/重载语义天然
  与 Phase A 同步。
- `spice_ffi` 原 FFI 包装保留为 `#[doc(hidden)] pub mod ffi_oracle`，仅供对拍测试
  当 oracle（先例：`daf::parse_with`）；不入 `FFI_CALLS` 计数。

### 2. 帧图与组合语义（逐位一致的关键）

- `pxform` 走 3×3 链（`refchg.c`），`sxform` 走 6×6 链（`frmchg.c`）——两者是
  CSPICE 内的独立组合路径，数值不保证相同。
- 组合照搬 refchg/frmchg：`from` 链上行到 J2000 或命中 `to`；否则 `to` 链上行到
  首个公共节点；末端 3×3 转置（`xpose`）/ 6×6 块转置（`invstm`）折入 `from` 链，
  链乘顺序 `zzrxr`/`zzmsxf` 原样（后项·前项右折叠；块乘的 3 项和/6 项和顺序保留）。
- class 2（PCK）：`tisbod` 语义——BPC 段优先（`pckmat`：pcke02 角序重排
  `[w,δ,φ,dw,dδ,dφ]` + `eul2xf(3,1,3)`；段参考系 17 时按 pcref 调整把常值阵折进
  tsipm），无段回退文本 PCK（IAU 多项式 + NUT_PREC 级数）。
- class 4（TK）：`tkfram` 语义——MATRIX 原样直取，ANGLES 经单位换算（换算因子为
  **预乘常量一次乘**：π/648000、π/180，与 `convrt` 逐位一致）后 `eul2m` 因子化
  （angles[0] 为最左因子）。
- 内置帧：`J2000`（根）、`ECLIPJ2000`（`irfrot(17,1) = trans(17)ᵀ` = 转置的
  `rotate(1, +84381.448″)`，et=0 oracle 实测钉死方向与 `convrt` 乘法口径）、
  `ITRF93`（→ class 2 / class_id 3000）。FK 重定义内置名 → 硬报错。
- `IAU_<BODY>` 的 class_id 取**天体本体号**（IAU_MERCURY→199 而非质心 1），与
  `spice_ffi::name_to_id` 的质心映射（第三体摄动用）刻意分表。

### 3. 实现契约（易错处，全部实测钉死）

- **`d_mod` 必须 ports f2c 兜底分支**（mice 构建未定义 `IEEE_drem`）：
  `q = trunc(x/y)`（商来自舍入后的除法），`d_mod = x − y·q`，符号随 x。IEEE fmod
  （精确余数）与它在 W 角 ~2830 rad 时差 4.3e-14——IAU_MOON/BPC 第三角全部受影响。
- `eul2xf` 的 `eul2m` 调用：C 的 `eul2m(angle3, angle2, angle1, …)` 把首参作最左
  因子；本仓 `eul2m(a1, …)` 首参最右，故实参按位反转。tkfram 的 ANGLES 路径同理。
- `eul2xf` 的 `drdt`：`drdtrt·r` 逐行 3 项和；`solutn` 矩阵按行乘（`mxv` 顺序）。
- 文本 PCK（tisbod）：RA/DEC 多项式用世纪数 tc、PM 用日数 td（Horner 一次式）；
  导数除以 t（或 d）换每秒；相位角 `θ = (a + tc·b)·rpd`（nphsco=2 公共分支）、
  `dθ = b/t·rpd`；RA/PM 级数配 sin、DEC 配 cos；`vdotg` 整段点积后一次叠加；
  `w = d_mod(w, twopi)`；`φ = RA + π/2`、`δ = π/2 − DEC`；
  `eul2xf([w,δ,φ,dw,dδ,dφ], 3,1,3)`。
- `NUT_PREC_ANGLES` 按 `zzbodbry`（百位收缩到系统质心：399→3、301→3、501→5）查
  标签体；级数系数按天体本键。`MAX_PHASE_DEGREE` /
  `BODY#_CONSTANTS_JED_EPOCH` / `BODY#_CONSTANTS_REF_FRAME` 本仓内核不含，解析层
  遇到即硬报错（不支持静默降级）。
- 文本内核解析：`\begindata`/`\begintext` 必须**独占一行**（pck00010 注释区有行内
  提及）；值支持引号串（`''` 转义）、Fortran `D` 指数、`@date`（→
  `(JD − 2451545)·86400`，CSPICE 内核池装载实测钉死）；括号跨行；逗号分隔；
  未知关键字忽略；解析失败硬错误。文件按内容分类（`DAF/` 头 → 二进制；首个非空行
  `KPL/FK|PCK|LSK` → 文本池；其余文本 native 跳过、cspice 照常 furnsh）。
- 池语义：`FK_STORE`/`TPCK_STORE`/`LSK_STORE` 与 `native_spk::STORE` 同款（load
  幂等 = 移除旧同路径再追加；查询克隆 `Arc` 快照不持锁求值；**逆序扫描 =
  后加载者生效**）。naif0011/naif0012 并存时最后 furnsh 者生效（被
  `tests/data/kernels/test_spice_manager.py` 常量钉死）。
- 时间：`deltet`（含 ET/UTC 双向闰秒查表口径）与 `et2utc` ISOC 路径逐操作移植，
  含 `unitim` 的 TDB→TAI 三次不动点、`ttrans` 的 TAI→YMD 闰秒日拆分（23:59:60
  显示）与秒位进位链（exsecs）、prec=0 的整秒进位。只实现 ISOC（仓库调用面）；
  `prec` ∉ 0..=9 硬报错（CSPICE 是钳位，本仓调用面只用 0/3，超界按错误面拒绝）。
- `ktotal`：native 计数——`ALL` = DAF + FK + TPCK + LSK 文件数；`SPK` = 含 SPK 段
  （`center: Some`）的 DAF 文件数；`PCK` = 含 BPC 段的 DAF 文件数；`FK`/`LSK`/
  `TEXT` = 对应池计数；其他 kind → `Err`。生产唯一调用点
  `nbody_stm.rs::spk_kernels_loaded` 用 `"SPK"`，语义保持。
- `furnish_kernel`/`spice_unload`：文本内核分派到文本池；cspice 侧失败的回滚对
  文本池对称扩展（此前已登记 → 重载，否则卸载）；`spice_unload` 三池齐卸。
- ABI 面：`spice_ext` 四符号、`_RUST_SYMBOLS`、ABI 戳（24）、`abi-version.txt` 不动
  （无新 PyO3 符号）。cspice 双登记保留（oracle 与 `easier_reader` 依赖 cspice 池）。

### 4. 已知残差（实测）

- 生产帧对（ITRF93/MOON_PA/MOON_ME/ECLIPJ2000/IAU_EARTH → J2000 的 pxform 与
  sxform，MOON 帧图互转）：与 mice CSPICE **逐位一致**（1186 点 ET 网格 ×
  11 帧对，逐位全等）。
- IAU_MOON：1186 点中 9 点存在 1–2 ULP 元素差（≤8.9e-16，≈2e-10 角秒），其余
  IAU 帧逐位一致。已核对多项式/相位角/级数的全部 op 顺序与 C 一致，残差在
  编译器算术（乘加结合的浮点实现差）层面；对拍阈值元素 ≤1e-14、角 ≤1e-2 角秒
  （acos 角差度量对 1-ULP 元素差的噪声地板 ~4.35e-3 角秒）。#638 的 36 角秒是
  anise 的口径，不适用本实现。

## 边界

- 仅 BPC Type 2（其他 PCK 段类型 → `NATIVE_FRAME_UNSUPPORTED_PCK_TYPE`）；段参考
  帧仅 1（J2000）/ 17（ECLIPJ2000）；DAS 容器不支持（本仓无）。
- 时间仅 ISOC；`str2et` 全集、TDB 精细转换不在范围。
- 动态帧（class 5/6）、CK 帧（class 3）→ 显式 `UnsupportedClass`。
- `EphemCache`/`StrictGuard` 不变（其查询面走 `spkezr`/`pxform`，自动受益）。
- Python 侧 `SPICEManager` 继续走 spiceypy（issue 明确边界）。

## 验证

`crates/e2m2e-spice/tests/native_frame_time_parity.rs`（11 用例，oracle =
`ffi_oracle`，内核经 `furnish_kernel` 双登记同池装载）：

1. `pxform_sxform_production_pairs`：5 帧对 pxform + 3 帧对 sxform × 1186 ET 网格
   逐位一致（目标逐位；上限 1e-12，实测 0）。
2. `euler_313_negative_control`：3-1-3 正序逐位一致；1-2-3 误序旋转角差 > 1 角秒
   （负控可检出）。
3. `frame_graph_equivalence`：MOON 帧图 6 帧对逐位一致。
4. `itrf93_load_order_later_wins`：先预测后历史装载，native == oracle（同池）；
   oracle 侧卸预测后 native == 只装历史的 oracle（后加载者生效）。
5. `text_pck_iau_ten_frames`：10 IAU 帧 × 1186 点，元素 ≤1e-14、角 ≤1e-2 角秒
   （IAU_MOON 单列常量；实测 1177/1186 逐位、9 点 ≤2 ULP）。
6. `et2utc_string_equality`：prec∈{0,3} × 网格 + 28 闰秒历元 ±1 s + 舍入边，
   字符串全等。
7. `deltet_reversibility`：网格 + 闰秒边界，D1==D2 逐位。
8. `hifitime_fixed_points`：9 组 (UTC, ET) 常量前缀全等（naif0011 先装、naif0012
   后装钉死）。
9. `ktotal_native_semantics`：装齐计数断言 + 未知 kind 报错。
10. `text_fk_negative_controls`：class 3 / 环 / 非法 UNITS / 悬空 relative 全部
    硬报错。
11. `furnish_unload_text_idempotent`：双 furnsh 幂等 + 逐池 unload 幂等。
