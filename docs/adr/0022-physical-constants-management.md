# ADR 0022: Independent physical constants management / 物理常数独立管理

[English](#adr-0022-independent-physical-constants-management) | [简体中文](#中文)

## English

**Status**: Adopted
**Date**: 2026-08-10
**Related**: ADR 0011 (five-layer architecture; constants live at
data/templates/), ADR 0014 (config.py covers runtime environment only;
constants belong to data/templates/), ADR 0013 (correctness by definition)

### Context

e2m2e's physical constants today: a constants layer exists nominally while
multiple numeric sets coexist in fact. `data/templates/systems.py` manages a
few Earth-Moon constants, `data/kernels/manager.py`'s `_GM_VALUES` manages body
GMs; radii, SRP, light speed, rotation rates scatter across the algorithm layer
and Rust crates — with reproducible inconsistencies already observed.

Three sources were surveyed: e2m2e's own code, GMAT R2026a sources, and a
literature corpus (Folta 2022, Vallado 2022, Soffel 2015, Topputo 2013,
Szebehely 1967, 16 works total). All three converge: **an independent,
source-traceable physical constants management is needed, and system default
baselines must be distinguished from model-carried constants.**

#### Confirmed inconsistencies (9 classes)

Ordered by impact; all real in current code:

1. **Earth-Moon mass ratio μ forks across languages (worst)**. Python
   production default `data/templates/seed.py` = `0.0121506683`
   (Szebehely's 1965 value); Rust `cr3bp.rs` = `0.0121505856`; Python
   `normal_form/constants.py` = `0.012150585609624` (qiao convention). Within
   one Rust crate, `cr3bp.rs` uses 0.0121505856 while `bcr4bp.rs` reverts to
   0.0121506683.
2. **Five values for Earth GM**: `398600.4418` (systems.py, GMAT/WGS-84),
   `398600.435507` (DE440, `_GM_VALUES`), `398600.4415` (gravity_file
   default), `398600.435436` (DE430, nbody_stm test), `398600.436` (truncated
   test value).
3. **Six values for Moon GM**: `4902.800118` (DE440), `4902.8001`, `4902.8`,
   `4902.800066` (DE430), `4902.800122` (test), `4902.799967088639`
   (GRGM900C, frozen_orbit).
4. **Four sets of EM distance/characteristic length**: `384405` (systems.py,
   Cui 2025), `384400` (seed.py characteristic length), `389703`
   (transfer_optimization), `384747.981` (normal_form qiao). System-default
   scale and orbit-family scale out of sync.
5. **Four values for lunar radius**: `1737.4` (IAU mean), `1738.0` (GRGM900C
   reference), `1738.1` (transfer_optimization), `1737.1` (harmonics test).
6. **Two values for Earth radius**: `6378.1363` (systems.py), `6378.137`
   (transfer_optimization and many tests).
7. **Two values for solar radius**: `695700` (shadow/srp/Rust), `696000`
   (relativistic_correction.py). Rust `relativistic.rs` and Python
   `relativistic_correction.py` are both relativistic corrections yet each uses
   its own.
8. **Solar constant/SRP pressure only in comments**: 1367 W/m² appears only in
   `srp.py`/`ecom_srp.py` comments; light speed `2.998e8` is a comment
   approximation, true values only inside Rust `relativistic.rs` and parameter
   defaults.
9. **Time constants repeatedly defined**: `86400` ≥9 places, `365.25` ≥7,
   `36525` ≥4, all independent.

Also: `systems.py` defines `MU_EARTH` but `data/templates/__init__.py` doesn't
export it; `hohmann.py` bypasses the top-level package to import directly.

#### Key fact: none of μ's three values is computed

- `0.0121506683`: recorded in Szebehely (1967)'s appendix as the 1965 value;
  historical inheritance.
- `0.0121505856`: close to DE421's exact ratio
  (`4902.8005821478 / (398600.4415 + 4902.8005821478)` = 0.01215058535) — a
  hand-copied truncated approximation from literature, **not** computed from
  DE440 GMs. DE440-computed is 0.01215058439.
- `0.012150585609624`: qiao normalization convention value.

I.e., Rust's seemingly more modern value is also just another copied
approximation. This is precisely the direct consequence of lacking independent
constants management: every author copies from whichever source is at hand and
nobody can say which set is the baseline.

### Decision

1. **Establish an independent physical constants layer `data/constants/`,
   peer to `data/templates/`.** The template layer owns mission/algorithm
   defaults (orbit family seeds, perturbation switches); the constants layer
   owns physical truth tables. Separate concerns.

2. **Constants split into two groups by mutability**:
   - **Universal physical constants** (light speed, gravitational constant,
     AU, time constants): physically unique, one set repo-wide.
   - **Body parameter catalog** (per-body GM, equatorial radius, reference
     radius, flattening, NAIF ID, rotation rate, solar radiation parameters):
     same quantity has multiple authoritative sources (DE versions / WGS-84 /
     IAU / gravity models) — **organized as coexisting baselines (datums)**.

3. **Multiple baselines coexist; programs choose per scenario.** This is the
   fundamental difference from hard unification: don't anoint a single value;
   provide several **self-consistent baseline sets** (within each set GM, μ,
   characteristic length are mutually consistent and share provenance);
   callers pick per mission. Baselines include at least:
   - `DE440`: current SPICE ephemeris — the default for ephemeris dynamics.
   - `DE421`: CR3BP/BCR4BP literature baseline (Folta 2022, Szebehely-system
     EM parameters) — default for orbit families/libration point studies.
   - `WGS84`: GNSS baseline for Earth shape & GM (radius, flattening, J2) —
     default for near-Earth missions.
   - Gravity-model self-carried constants (e.g. GRGM900C's lunar GM/radius)
     are **not merged into baselines**; they stay with their force models.

4. **Default baseline selection**:
   - EM mass ratio μ, characteristic length/time: **unify on DE ephemeris
     values**, retiring the 1965 value `0.0121506683`. CR3BP orbit families
     default to `DE421` (μ = 0.012150585350562453), matching mainstream
     literature (Folta 2022, Topputo 2013, revised Szebehely); ephemeris
     dynamics defaults to `DE440`.
   - Earth geometric parameters (radius, flattening, J2): near-Earth scenes
     default `WGS84`.
   - Solar radiation/SRP: defaults to traditional engineering value
     1367 W/m² (GMAT-compatible), with modern TSI 1361 W/m² as alternative.

5. **Single source + generated alignment; Python/Rust never hand-copy their
   own.**
   - Python side: constants defined once in `data/constants/`; algorithm layer
     and tests import exclusively; no hardcoding.
   - Rust side: shares origin with Python. Guarantee no drift via
     **build-time generation** (generating the Rust constants module from the
     single-source file) or **alignment tests** (Rust tests assert equality
     with Python values). Pick one; see Open Questions.
   - Every constant annotated with provenance (`# DE440, TDB` /
     `# IAU 2015` / `# WGS-84`) — the annotation *is* documentation.

6. **Absorb scattered sites, remove duplicate definitions.** SRP/solar
   constant, light speed, body radius tables, time constants all absorbed here;
   defined-but-unexported constants like `MU_EARTH` get exported;
   `normal_form/constants.py`'s qiao-normalized constants labeled model-carried
   (qiao convention), kept out of baselines.

7. **Verification**: new constants-consistency tests: same quantity asserted
   equal across Python/Rust; within-baseline self-consistency assertions
   (μ == GM_moon/(GM_earth+GM_moon); t* == sqrt(l³/GM_total)); rewrite
   `systems.py`'s "already unified" comments to this layer's true state.

### Recommended value choices

The table below gives recommended baseline values per quantity plus retained
alternatives. Units uniformly km/s/kg (GM in km³/s²).

#### Universal physical constants (one set repo-wide)

| Constant | Value | Unit | Source |
|---|---|---|---|
| Light speed c | 299792.458 | km/s | defined (SI) |
| Gravitational constant G | 6.67430e-20 | km³/(kg·s²) | CODATA 2018 (already used in systems.py; keep) |
| Astronomical unit AU | 149597870.7 | km | IAU 2012 Resolution B2 (already used; keep) |
| Seconds per day | 86400 | s | defined |
| Julian year | 365.25 | day | defined |
| Julian century | 36525 | day | defined |
| Solar constant (flux) | 1367 (default) / 1361 (modern TSI alt.) | W/m² | GMAT IERS 1996 / Pesce 2023 |
| SRP pressure at 1 AU | derived flux/c | N/m² | derived; not separately defined |

#### Earth-Moon system baseline (CR3BP/BCR4BP characteristic scales)

| Baseline | μ | Characteristic length l* (km) | Characteristic time t* (s) | Source |
|---|---|---|---|---|
| `DE421` (orbit-family default) | 0.012150585350562453 | 384400.0 | 375190.2588926273 | Folta 2022 Table 2 |
| `DE440` (ephemeris default) | 0.012150584394709708 | derived from GMs & ephemeris | derived from GMs | JPL DE440 |
| ~~1965 legacy~~ | ~~0.0121506683~~ | — | — | **retired** (Szebehely 1965) |

Note: t*, l*, GM satisfy `t* = sqrt(l*³ / GM_total)`; in-set consistency via
assertions, never filled independently. DE421's t* equals Folta 2022's
3.751902588926273e5 s.

#### Body GMs (km³/s², by baseline)

| Body | DE440 | DE421 | WGS-84 |
|---|---|---|---|
| Earth | 398600.435507 | 398600.4415 | 398600.4418 |
| Moon | 4902.800118 | 4902.8005821478 | — |
| Sun | 1.32712440018e11 | 1.32712428e11 | — |
| EMB | 403503.235502 | 403503.242083 | — |

Earth's three values correspond respectively to DE440 ephemeris, DE421
ephemeris, WGS-84/EGM96 shape model. GNSS near-Earth uses WGS-84; ephemeris
dynamics uses the matching DE version. Model-carried (GRGM900C lunar GM
4902.799967088639) stays out of this table, remaining with its gravity model.

#### Body radii (km; distinguishing mean/equatorial vs gravity-field reference)

| Body | Mean/equatorial | Source | Field reference radius | Source |
|---|---|---|---|---|
| Earth | 6378.137 | WGS-84 (near-Earth default); 6378.1363 is GMAT PCK historical, kept for GMAT alignment | 6378.1363 | EGM96/JGM coefficient file header |
| Moon | 1737.4 | IAU 2015 (shadow/cartography/default datum); LOLA 1737.151 alternative | 1738.0 | GRGM900C file header |
| Sun | 696000 | Vallado D-5 (unify here; retire 695700) | — | — |

**Choice rationale**: shadow/SRP/relativistic occultation and physical radii
use mean/equatorial radii; gravity fields (harmonics, solid tides) read their
reference radius from coefficient-file headers, never mixed with mean radii.
This mirrors GMAT exactly (`GmatDefaults.hpp` default radius vs gravity file
`radius` override) — e2m2e currently conflates the two.

#### Earth rotation rate

| Value | Unit | Source |
|---|---|---|
| 7.292115146706979e-5 | rad/s | IERS (with LOD; used by ITRF frames; already in gmat_itrf.py) |
| 7.29211585530e-5 | rad/s | GMAT `CelestialBody` default (atmosphere/rotation models; alternative) |

GMAT itself carries both for different purposes; e2m2e follows suit per scene.

### Proposed structure

```
e2m2e/data/constants/
├── __init__.py          # unified export entry
├── universal.py         # universal constants (c/G/AU/time/solar constant)
├── datums.py            # baseline definitions (DE421/DE440/WGS84: GM/μ/l*/t*)
├── bodies.py            # body catalog (radii/flattening/NAIF ID/rotation)
└── sources.py           # datum/source enums & metadata (provenance per value)
```

- `datums.py` makes baselines first-class: `Datum.DE421.mu`,
  `Datum.DE440.earth_gm`, each internally consistent.
- `bodies.py`'s catalog replaces & extends `manager.py`'s `_GM_VALUES`
  (adding radii/flattening/rotation beyond GM); `get_gm()` queries by baseline.
- Constants migrate from `data/templates/systems.py`, which keeps re-exports
  for compatibility (or dies outright; see Open Questions).
- Rust constants module generated from single source or guarded by alignment
  tests.

### Rationale

1. **Coexisting baselines over hard unification**: literature and GMAT both
   show lunar mean radius 1737.4 vs field reference 1738.0, or Earth GM's DE-vs-
   WGS-84 split, are **each correct in their own scenario**; hard unification
   breaks model self-consistency. Today's mess isn't too many values but
   nobody saying which set a value belongs to and where it applies.
2. **Baseline sets over loose values**: μ, l*, t*, GM must share provenance to
   be self-consistent (mixing CR3BP's μ and l* across sources desyncs orbit
   families from literature). Packaging one source's values as a baseline beats
   managing constants one-by-one for internal consistency.
3. **Single source + generated alignment**: GMAT's lesson — centralization is
   necessary but insufficient: it centralized yet still grew three AU values.
   e2m2e spans Python/Rust; it needs an added cross-language single-origin
   enforcement (generation or alignment tests), or post-merge it forks again.
4. **Provenance is documentation**: μ's three-value lesson shows constants
   without stated sources equal no constants. Every value annotated with origin;
   later disputes have paper trails.

### Consequences

- New `data/constants/` layer, peer to `data/templates/`.
- μ defaults switch from 1965 legacy to DE values. **This changes CR3BP orbit
  family/libration point numerical results** — flag prominently in CHANGELOG
  and update affected test baselines uniformly (conftest, examples, normal-form).
- μ fork between Rust `cr3bp.rs` and Python `seed.py` eliminated; solar-radius
  fork between `relativistic.rs` and `relativistic_correction.py` eliminated.
- `manager.py`'s GM queries route through baselines.

### Open questions

1. **Python/Rust single-origin mechanism**: build-time generation of Rust
   constants from one source file (thorough but adds build complexity), or
   cross-language alignment tests (lightweight but test-guarded)? Leaning
   toward starting with tests.
2. **Fate of `systems.py`**: keep re-export compatibility after migration, or
   repoint all imports directly? Leaning direct repoint, no double layer.
3. **μ switch compatibility**: retain an explicit `datum="legacy1965"` option
   on CR3BP_System to reproduce old results, or clean cut? Leaning clean cut +
   CHANGELOG note; old value unsupported.

## 中文

**状态**：已采纳
**日期**：2026-08-10
**关联**：ADR 0011（五层架构，物理常量归 data/templates/）、ADR 0014（config.py 只管运行环境，物理常量归 data/templates/）、ADR 0013（正确性由物理定义裁决）

### 背景

e2m2e 的物理常数现在的状态是：名义上有常量层，实际上多套数值并存。`data/templates/systems.py` 管地月系统的少数几个常量，`data/kernels/manager.py` 的 `_GM_VALUES` 管天体 GM，其余半径、光压、光速、自转角速度散落在算法层与 Rust 各 crate，且已出现可复现的不一致。

调研了三个来源：e2m2e 自身代码、GMAT R2026a 源码、文献库（Folta 2022、Vallado 2022、Soffel 2015、Topputo 2013、Szebehely 1967 等 16 篇）。三个来源共同指向一个结论：**需要一套独立的、可追溯来源的物理常数管理，而且必须区分系统默认基准与模型自带常量**。

### 已确认的不一致（9 类）

按影响排序，均为当前代码中真实存在：

1. **地月质量比 μ 跨语言分叉（最严重）**。Python 生产默认 `data/templates/seed.py` = `0.0121506683`（Szebehely 1965 年值）；Rust `cr3bp.rs` = `0.0121505856`；Python `normal_form/constants.py` = `0.012150585609624`（qiao 约定）。同一个 Rust crate 里 `cr3bp.rs` 用 0.0121505856、`bcr4bp.rs` 又用回 0.0121506683。
2. **地球 GM 五个值**：`398600.4418`（systems.py，GMAT/WGS-84）、`398600.435507`（DE440，`_GM_VALUES`）、`398600.4415`（gravity_file 默认）、`398600.435436`（DE430，nbody_stm 测试）、`398600.436`（测试截断）。
3. **月球 GM 六个值**：`4902.800118`（DE440）、`4902.8001`、`4902.8`、`4902.800066`（DE430）、`4902.800122`（测试）、`4902.799967088639`（GRGM900C，frozen_orbit）。
4. **地月距离/特征长度四套**：`384405`（systems.py，Cui 2025）、`384400`（seed.py 特征长度）、`389703`（transfer_optimization）、`384747.981`（normal_form qiao）。系统默认尺度与轨道族尺度不同步。
5. **月球半径四个值**：`1737.4`（IAU 平均）、`1738.0`（GRGM900C 参考）、`1738.1`（transfer_optimization）、`1737.1`（球谐测试）。
6. **地球半径两个值**：`6378.1363`（systems.py）、`6378.137`（transfer_optimization 及大量测试）。
7. **太阳半径两个值**：`695700`（shadow/srp/Rust）、`696000`（relativistic_correction.py）。Rust `relativistic.rs` 与 Python `relativistic_correction.py` 同为相对论修正却各用各的。
8. **太阳常数/光压只在注释里**：1367 W/m² 仅出现在 `srp.py`、`ecom_srp.py` 注释；光速 `2.998e8` 是注释近似，真实值只在 Rust `relativistic.rs` 和参数默认值里。
9. **时间常量重复定义**：`86400` 至少 9 处、`365.25` 至少 7 处、`36525` 至少 4 处，各自独立。

另外：`systems.py` 定义了 `MU_EARTH` 但 `data/templates/__init__.py` 未导出，`hohmann.py` 绕过顶层包直接导入。

### 关键事实：μ 的三套值都不是算出来的

- `0.0121506683`：Szebehely (1967) 附录记载的 1965 年值，历史沿用。
- `0.0121505856`：接近 DE421 的精确比（`4902.8005821478 / (398600.4415 + 4902.8005821478)` = 0.01215058535），是文献抄来的截断近似，**不是**从 DE440 GM 现算。DE440 现算为 0.01215058439。
- `0.012150585609624`：qiao 归一化约定值。

也就是说，Rust 那个看起来更现代的值其实也只是另一份手抄近似。这正是没有独立常数管理的直接后果：每个作者从自己手头的文献抄一个值，没人能说清哪套是基准。

### 决策

1. **建立独立物理常数层 `data/constants/`，与 `data/templates/` 平级**。模板层管任务/算法默认参数（轨道族种子、摄动开关），常数层管物理量真值表，职责分开。

2. **常数分两组，按可变性划界**：
   - **通用物理常量**（光速、万有引力常数、天文单位、时间常量）：物理上唯一，全库一套。
   - **天体参数目录**（各天体 GM、赤道半径、参考半径、扁率、NAIF ID、自转角速度、太阳辐射参数）：同一物理量有多个权威来源（DE 各版本 / WGS-84 / IAU / 重力场模型），**按基准（datum）组织多套并存**。

3. **多套基准并存，程序按场景选择**。这是与硬统一的根本区别：不为全库钦定唯一数值，而是提供若干套**自洽的基准集**（每套内部 GM、μ、特征长度互相一致、来自同一来源），调用方按任务选定。基准集至少包含：
   - `DE440`：e2m2e SPICE 现行星历，星历力学（ephemeris dynamics）的默认基准。
   - `DE421`：CR3BP/BCR4BP 文献基准（Folta 2022、Szebehely 体系的地月参数），轨道族/平动点研究的默认基准。
   - `WGS84`：地球形状与 GM 的 GNSS 基准（地球半径、扁率、J2），近地任务的默认基准。
   - 重力场模型自带常量（如 GRGM900C 的月球 GM/半径）**不并入基准集**，作为模型自带常量保留在对应力模型处，与本层分家。

4. **默认基准的选取**：
   - 地月系统质量比 μ、特征长度、特征时间：**统一到 DE 星历值**，废弃 1965 旧值 `0.0121506683`。CR3BP 轨道族默认走 `DE421`（μ = 0.012150585350562453），与主流文献（Folta 2022、Topputo 2013、Szebehely 修订）一致；星历力学默认走 `DE440`。
   - 地球几何参数（半径、扁率、J2）：近地场景默认 `WGS84`。
   - 太阳辐射/光压：默认传统工程值 1367 W/m²（GMAT 兼容），同时提供现代 TSI 1361 W/m² 备选。

5. **单一来源 + 生成对齐，Python/Rust 不各自手抄**。
   - Python 侧：常数在 `data/constants/` 定义一次，算法层、测试一律 import，不再硬编码。
   - Rust 侧：Rust 常量与 Python 同源。用**构建期生成**（从单一来源文件生成 Rust 常量模块）或**对齐测试**（Rust 测试断言与 Python 值一致）保证两边不漂移。二选一，见开放问题一节。
   - 每个常量标注来源（如 `# DE440, TDB` / `# IAU 2015` / `# WGS-84`），来源即文档。

6. **收编散落点，消除重复定义**。光压/太阳常数、光速、天体半径表、时间常量全部收编进本层；`MU_EARTH` 等已定义未导出的补齐导出；`normal_form/constants.py` 的 qiao 归一化常量标注为模型自带（qiao 约定），不并入基准集。

7. **验证**：新增常量一致性测试：同一物理量跨 Python/Rust 断言相等；同一基准集内部自洽断言（如 μ == GM_moon/(GM_earth+GM_moon)，特征时间 == sqrt(l³/GM_total)）；`systems.py` 中声称已统一的注释改写为本层的真实状态。

### 各常量取值取舍建议

下表给出每个物理量的推荐基准值与保留的多套值。单位统一为 km / s / kg 系（GM 为 km³/s²）。

### 通用物理常量（全库一套）

| 常量 | 取值 | 单位 | 来源 |
|---|---|---|---|
| 光速 c | 299792.458 | km/s | 定义值（SI） |
| 万有引力常数 G | 6.67430e-20 | km³/(kg·s²) | CODATA 2018（现 systems.py 已用，保留） |
| 天文单位 AU | 149597870.7 | km | IAU 2012 决议 B2（现 systems.py 已用，保留） |
| 每日秒数 | 86400 | s | 定义值 |
| 儒略年 | 365.25 | day | 定义值 |
| 儒略世纪 | 36525 | day | 定义值 |
| 太阳常数（光通量） | 1367（默认）/ 1361（现代 TSI 备选） | W/m² | GMAT IERS 1996 / Pesce 2023 |
| 1 AU 光压 | 由 flux/c 派生 | N/m² | 派生量，不单独定义 |

### 地月系统基准（CR3BP/BCR4BP 特征尺度）

| 基准 | μ | 特征长度 l* (km) | 特征时间 t* (s) | 来源 |
|---|---|---|---|---|
| `DE421`（轨道族默认） | 0.012150585350562453 | 384400.0 | 375190.2588926273 | Folta 2022 Table 2 |
| `DE440`（星历默认） | 0.012150584394709708 | 由 GM 与历表推 | 由 GM 推 | JPL DE440 |
| ~~1965 旧值~~ | ~~0.0121506683~~ | 无 | 无 | **废弃**（Szebehely 1965） |

说明：t* 与 l*、GM 满足 `t* = sqrt(l*³ / GM_total)`，基准集内部用断言保证自洽，不各填各的。`DE421` 的 t* 即 Folta 2022 给出的 3.751902588926273e5 s。

### 天体 GM（km³/s²，按基准集）

| 天体 | DE440 | DE421 | WGS-84 |
|---|---|---|---|
| 地球 | 398600.435507 | 398600.4415 | 398600.4418 |
| 月球 | 4902.800118 | 4902.8005821478 | 无 |
| 太阳 | 1.32712440018e11 | 1.32712428e11 | 无 |
| EMB | 403503.235502 | 403503.242083 | 无 |

地球三个值分别对应：DE440 星历、DE421 星历、WGS-84/EGM96 地球形状模型。GNSS 近地用 WGS-84，星历力学用对应 DE 版本。模型自带（GRGM900C 月球 GM 4902.799967088639）不进此表，留在重力场模型处。

### 天体半径（km，区分平均/赤道半径与重力场参考半径）

| 天体 | 平均/赤道半径 | 来源 | 重力场参考半径 | 来源 |
|---|---|---|---|---|
| 地球 | 6378.137 | WGS-84（近地默认）；6378.1363 为 GMAT PCK 历史值，保留供 GMAT 对齐 | 6378.1363 | EGM96/JGM 系数文件头 |
| 月球 | 1737.4 | IAU 2015（阴影/制图/起算面默认）；1737.151 为 LOLA 备选 | 1738.0 | GRGM900C 系数文件头 |
| 太阳 | 696000 | Vallado D-5（统一此值，废弃 695700） | 无 | 无 |

**取舍**：阴影、SRP、相对论修正的遮挡/物理半径统一用平均/赤道半径；重力场（球谐、固潮）的参考半径从系数文件头读，不与平均半径混用。这一区分正是 GMAT 的做法（`GmatDefaults.hpp` 默认半径 vs 重力文件 `radius` 覆盖），e2m2e 现在把两套混在一起了。

### 地球自转角速度

| 取值 | 单位 | 来源 |
|---|---|---|
| 7.292115146706979e-5 | rad/s | IERS（含 LOD，坐标系 ITRF 用，现 gmat_itrf.py 已用） |
| 7.29211585530e-5 | rad/s | GMAT `CelestialBody` 默认值（大气/自转模型用，备选） |

两个值在 GMAT 里就并存且用途不同，e2m2e 照此分场景。

### 落地结构（建议）

```
e2m2e/data/constants/
├── __init__.py          # 统一导出入口
├── universal.py         # 通用物理常量（c/G/AU/时间/太阳常数）
├── datums.py            # 基准集定义（DE421/DE440/WGS84 各自的 GM/μ/l*/t*）
├── bodies.py            # 天体参数目录（半径/扁率/NAIF ID/自转角速度）
└── sources.py           # 基准/来源枚举与元数据（每个值的出处）
```

- `datums.py` 以基准为一等概念：`Datum.DE421.mu`、`Datum.DE440.earth_gm`，每套内部自洽。
- `bodies.py` 的天体目录替代并扩展 `manager.py` 的 `_GM_VALUES`（GM 之外补半径/扁率/自转），`get_gm()` 改为按基准查询。
- 现有 `data/templates/systems.py` 的常量迁移进本层，`systems.py` 保留 re-export 以兼容（或直接废弃，见开放问题）。
- Rust 侧常量模块由单一来源生成或加对齐测试。

### 理由

1. **多套并存而非硬统一**：文献与 GMAT 都表明，月球平均半径 1737.4 与重力场参考半径 1738.0、地球 GM 的 DE 各版与 WGS-84 值，是**不同场景各自正确的值**，硬统一会破坏模型自洽。e2m2e 当前的混乱不是值太多，而是没说清每个值属于哪套、用在哪。
2. **基准集而非散值**：μ、l*、t*、GM 必须来自同一来源才自洽（CR3BP 的 μ 和特征长度用不同来源拼，轨道族就对不上文献）。把一套来源的所有值打包成基准，比逐个常量管理更能保证内部一致。
3. **单一来源 + 生成对齐**：GMAT 的教训是，集中只是必要非充分：它集中了却仍让 AU 出现三个值。e2m2e 跨 Python/Rust，必须再加一道两边同源的强制（生成或对齐测试），否则合并后还会再分叉。
4. **来源即文档**：μ 三套值的教训说明，不标来源的常量等于没有常量。每个值注明出处，后续争议有据可查。

### 结果

- 新增 `data/constants/` 层，与 `data/templates/` 平级。
- μ 默认值从 1965 旧值切到 DE 星历值。**这会改变 CR3BP 轨道族/平动点的数值结果**，需在 CHANGELOG 显著标注，并对受影响的测试基准（conftest、examples、normal-form）统一更新。
- Rust `cr3bp.rs` 与 Python `seed.py` 的 μ 分叉消除；`relativistic.rs` 与 `relativistic_correction.py` 的太阳半径分叉消除。
- `manager.py` 的 GM 查询接入基准集。

### 开放问题

1. **Rust/Python 同源机制**：构建期从单一来源文件生成 Rust 常量（彻底但加构建复杂度），还是加跨语言对齐测试（轻量但靠测试兜底）？倾向后者起步。
2. **`systems.py` 去留**：常量迁走后保留 re-export 兼容层，还是直接改所有 import 指向新层？倾向直接改，不留双层。
3. **μ 切换的兼容**：是否需要给 CR3BP_System 保留一个显式 `datum="legacy1965"` 选项以复现旧结果，还是一刀切？倾向一刀切 + CHANGELOG 说明，旧值不再支持。
