# e2m2e Architecture Design / e2m2e 架构设计

[English](#english) | [简体中文](#简体中文)

## English

### Starting from one NRHO orbit

Take the design of an Earth–Moon L2 NRHO as an example to see where each of
the five modules participates.

**Step one: the user inputs a time instant — 2024-01-01 00:00:00 UTC.** This
is human convention, but it is not the independent variable of ephemerides:
DE ephemerides use barycentric dynamical time TDB. At that moment UTC and TDB
differ by about 69 seconds (37 seconds of leap seconds + 32.184 seconds fixed
offset). Query the Moon's position with UTC taken directly as TDB: the Moon
orbits Earth at roughly 1 km/s, so the position error is about 70 km — the
lunar braking window is completely lost. Moreover UTC leaps irregularly and is
discontinuous, unfit as a dynamics argument. So e2m2e standardizes on TDB
internually, converting to UTC only at interface boundaries. STK does the
same: UTC only for input/output, TAI/TDT internally. This is why time scales
exist in the **spacetime systems and constants module**.

**Next: which coordinate frame to design in.** The CR3BP initial guess comes
in the Earth–Moon synodic frame (libration points are stationary there), the
mission specification's parameters are in the J2000 inertial frame, and
ground-station coordinates are in the ITRF93 Earth-fixed frame. The same set
of (x,y,z) numbers without conversion means another physical location: Earth
rotates 15° per hour; states in two frames differ by a rotation growing with
time. In 1999 NASA's Mars Climate Orbiter was lost to unit confusion (pound-
force vs newton); frame confusion is the same class of error, only stealthier,
and the compiler will not flag it. Frame conversion makes every state vector's
frame traceable and convertible — this is why reference-frame conversion lives
in the same module.

**Then: which constants.** Earth GM has DE440 values, DE421 values, WGS-84
values; the Earth–Moon mass ratio μ has 1965-era values, DE421 values, and
normalized-convention values. e2m2e itself stumbled here: Python and Rust each
copied a μ, two values coexisted in one crate, and computed orbit families
mismatched the literature. The trouble with constants is not having many
values but that nobody says clearly which value belongs to which baseline and
where it applies. Constant baselines (multiple sets coexisting, self-consistent
within each) exist for exactly this — the constants part of the same module.

**Initial guess in hand, shooting begins.** Take a 2-year 50-rev ephemeris
correction: 50 revs × 8 nodes per rev × 50 iterations ≈ twenty thousand
trajectory segments, thousands of steps per segment, force models evaluated
and ephemeris queried at every step — scalar operations in the billions. This
layer must be fast and parallelizable, while cspice kernels are global state
and cannot be used concurrently — so the **Rust computation module** takes no
SPICE handles, only pre-sampled injected ephemeris cache tables.

**The whole design chain is orchestration, not formulas.** DRO or Halo, how
large an initial-guess amplitude, how tight correction tolerance, propagate
one year or ten — these are domain decisions changing daily, kept in Python.
That is the **orchestration module**'s share.

**After computing, deliver.** Design engineers sit on Windows, simulation
centers run Kylin, servers are Kunpeng ARM — three-platform wheels are the
deployment reality; NAIF's site is unstable from domestic networks, so CSPICE
must ship via GitHub Release. That is the **CI module**.

**Finally, data must be kept.** Ephemeris files are tens to hundreds of MB,
updated every few years; Halo family seeds are two floats whose every change
needs review. Things differing a million-fold in size with entirely different
lifecycles cannot be put in one basket. That is the **data management module**.

The software has five architectural designs:

### 1. Spacetime systems and constants module

(1) Time scales (UTC/TDB/TAI/TT; dynamics standardized on TDB);
(2) Reference-frame conversion (J2000/ITRF93/IAU 2006, GCRS-EBCRS, dynamic
axes);
(3) Physical constant baselines (DE421/DE440/WGS84 coexisting, chosen per
scenario, single-source across Python/Rust against drift).

Together they form a self-consistent spacetime basis, actually maintained
separately. Code locations: conversion **algorithms** live in
`e2m2e/algorithm/coordinate` (per ADR 0011, formerly `core/coordinate/`);
coordinate **data** (EOP/leap seconds/ephemeris handles) in
`e2m2e/data/frames` (ADR 0015). Functional markers are assigned by verified
content under `data` — an axis independent of code layering (ADR 0026).

### 2. Rust computation module

Six crates:

(1) spice (CSPICE FFI + caching);
(2) propagation (pure math integrators);
(3) forces (force models + STM);
(4) integrators (pyo3 bindings + iterative solvers such as shooting +
parallelism);
(5) levelset (level-set / HJB solver kernel, ToolboxLS port);
(6) hjb-dynamics (Hamiltonian dynamics implementations for the HJB solver,
ADR 0032).

Iterative problems are constructed by Python and passed in; Rust only iterates
to convergence; Rust takes no SPICE handles, only pre-sampled injected
ephemeris cache tables.

Each algorithm module's migration progress (which numerics have sunk, which
remain in Python, and why) is itemized in
[numerics-migration-status](numerics-migration-status.md); consult the ledger
before drawing audit conclusions.

### 3. Python orchestration module

Only three things, no numerical iteration:

(1) construct the problem;
(2) call Rust iterators;
(3) interpret results.

Three orchestration tiers: task-level Facade (MCP/CLI derived from the same
source), mission orbit design algorithm chains (family initial guess → Rust
correction → high-fidelity propagation), and subproblem construction (family
strategies, control laws, manifold seeds).

Beyond executing input validation, the task-level API model also exposes
parameter metadata for the current task context, for GUIs, CLIs, and MCP to
generate widgets and range hints; conditional value domains and validators
share one rule definition, so external consumers keep no local copies of
parameter rules.

### 4. CI module

Builds software for Linux x64 AMD (Ubuntu, Kylin, etc.), Linux ARM (aarch64,
Kunpeng/Phytium), and Windows. Three-platform wheel matrix + sdist + GitHub
Release + PyPI; the CSPICE build package (x86_64 re-uploaded from NAIF's
official package, aarch64 self-built) ships via GitHub Release; domestic
systems like Kylin are covered by the manylinux 2_28 glibc baseline.

### 5. Data management module

(1) GitHub Release manages some build libraries, ephemeris data, and the
CSPICE build library (released by version, fetched by script);
(2) Git tracks initial values of some orbits (family seed parameters);
(3) Local computation artifacts enter the orbit catalog (ADR 0031): records =
JSON metadata + NPZ array segments (with schema version), SQLite only as a
derived index.

This module will later extend to support intelligent game-theoretic research
as its data foundation. Ephemeris serialization uses EphemerisTable as the
unified intermediate format; converters to/from typical ephemeris formats
(CCSDS OEM/ODM, SPICE BSP/PCK, etc.) will follow.

Some computational functionality still runs in Python (NLP optimization
etc.) and is migrating stepwise to the Rust core. Continuation, NSGA-II
evolutionary operators, and low-thrust direct-method numerical evaluation
kernels have sunk; Python keeps outer orchestration and reference paths.
The spacetime and constants module still needs further review (three
coexisting frame paths, time-conversion responsibility chain). The Python
orchestration module is not yet stable enough (dual shooting paths, transfer
directory numerical residue); further review and system design are required.

## 简体中文

## 从一条 NRHO 轨道说起

以设计一条地月 L2 NRHO 轨道为例，看五个模块各自在哪个环节发挥作用。

**第一步，用户输入一个时刻：2024-01-01 00:00:00 UTC。** 这是人们约定的时间，但历表的自变量不是它：DE 历表用质心力学时 TDB。此刻 UTC 与 TDB 相差约 69 秒（37 秒闰秒 + 32.184 秒固定差）。若拿 UTC 直接当 TDB 查月球位置：月球以约 1 km/s 绕地球公转，位置错出约 70 km，近月制动窗口完全作废。何况 UTC 不定期跳秒、不连续，不能做动力学的自变量。所以 e2m2e 内部统一用 TDB，只在接口边界转 UTC。STK 也是这个做法：UTC 只用于输入输出，内部用 TAI/TDT。这是**时空系统与常量模块**里时间尺度存在的理由。

**接着要回答在哪个坐标系里设计。** CR3BP 初猜在地月会合系里给（会合系中平动点是定点），任务书参数在 J2000 惯性系，测控站坐标在 ITRF93 地固系。同一组 (x,y,z) 数字，换系不换数，就是另一个物理位置：地球一小时自转 15°，两个系的状态差一个随时间增长的旋转。1999 年 NASA 火星气候轨道器因单位混淆（磅力与牛顿）坠毁；坐标系混淆是同一类错误，且更隐蔽，编译器不会报错。参考系转换让每一个状态向量的系可追溯、可转换，这是同一模块里参考系转换存在的理由。

**然后要定用哪套常数。** 地球 GM 有 DE440 值、DE421 值、WGS-84 值；地月质量比 μ 有 1965 年值、DE421 值、归一化约定值。e2m2e 自己就栽在这里：Python 与 Rust 各抄一个 μ，同一个 crate 里两个值并存，轨道族算出来对不上文献。常量的麻烦不是值太多，而是没人说清哪个值属于哪套、用在哪。常量基准集（多套并存、每套内部自洽）就是为此而设，这是同一模块里常量的部分。

**初猜到手，开始打靶。** 以 2 年 50 圈的星历修正为例：50 圈 × 每圈 8 节点 × 50 次迭代，约两万段轨道积分，每段数千步，每步算力模型、查星历，标量运算以十亿计。这一层必须快、必须能并行，而 cspice 内核是全局状态、不可并发，所以 **Rust 计算模块**不吃 SPICE 句柄，吃预采样注入的星历缓存表。

**整条设计链是编排，不是算式。** 选 DRO 还是 Halo、初猜振幅给多大、修正容差多严、预报一年还是十年，这些是领域决策，天天在变，留 Python。这是**编排模块**的分工。

**算完要交付。** 设计工程师的桌面是 Windows，仿真中心是麒麟，服务器是鲲鹏 ARM，三平台 wheel 是部署现实；国内访问 NAIF 官网不稳定，CSPICE 只能走 GitHub Release。这是**CI 模块**。

**最后，数据要留下。** 星历文件几十到上百 MB，几年才更新一次；Halo 族种子是两个浮点数，每次改动都要审阅。体积差上百万倍、生命周期完全不同的东西，不能放在一个篮子里。这是**数据管理模块**。

软件有五个架构设计：

### 1. 时空系统与常量模块

（1）时间尺度（UTC/TDB/TAI/TT，动力学统一用 TDB）；
（2）参考系转换（J2000/ITRF93/IAU 2006、GCRS-EBCRS、动态轴）；
（3）物理常量基准集（DE421/DE440/WGS84 多套并存，按场景选用，Python/Rust 同源防漂移）。

三者合起来构成自洽的时空基准，实际分开维护。代码位置：转换**算法**在
`e2m2e/algorithm/coordinate`（ADR 0011 既定，源 `core/coordinate/`），坐标**数据**
（EOP/闰秒/历表句柄）在 `e2m2e/data/frames`（ADR 0015）。功能类标记按验证内容
归 `data`，与代码层级是两个独立轴（ADR 0026）。

### 2. Rust 计算模块

六个 crate：

（1）spice（CSPICE FFI + 缓存）；
（2）propagation（纯数学积分器）；
（3）forces（力模型 + STM）；
（4）integrators（pyo3 绑定 + 打靶等迭代求解器 + 并行）；
（5）levelset（水平集 / HJB 求解内核，ToolboxLS 移植）；
（6）hjb-dynamics（HJB 求解器的动力学 Hamiltonian 实现，ADR 0032）。

迭代的问题由 Python 构造传入，Rust 只管迭代到收敛；Rust 不吃 SPICE 句柄，吃预采样注入的星历缓存表。

各算法模块的迁移进度（哪些数值已下沉、哪些还在 Python、各自理由）逐项登记在
[numerics-migration-status](numerics-migration-status.md)，审计时先查清单再下结论。

### 3. Python 编排模块

只做三件事，不做数值迭代：

（1）构造问题；
（2）调 Rust 迭代器；
（3）解释结果。

任务级 Facade（MCP/CLI 同源派生）、任务轨道设计的算法链（族初猜 → Rust 修正 → 高精度预报三段）、子问题构造（族策略、控制律、流形种子）三层编排。

任务级 API 模型除执行输入校验外，还公开当前任务上下文下的参数元数据，供 GUI、CLI、MCP 生成控件及范围提示；条件取值域与校验器共用同一份规则定义，外部消费者不维护本地参数规则副本。

### 4. CI 模块

支持 Linux x64 AMD（Ubuntu、麒麟等操作系统）、Linux ARM（aarch64，鲲鹏/飞腾）、Windows 平台的软件编译。三平台 wheel 矩阵 + sdist + GitHub Release + PyPI；CSPICE 编译包（x86_64 转存 NAIF 官方包，aarch64 自建）走 GitHub Release；麒麟等国产系统靠 manylinux 2_28 的 glibc 基线覆盖。

### 5. 数据管理模块

（1）GitHub Release 管理一些编译库、星历数据、CSPICE 编译库（按版本发布，脚本拉取）；
（2）Git 跟踪一些轨道的初值（族种子参数）；
（3）本地计算产物入轨道库 catalog（ADR 0031）：记录 = JSON 元数据 + NPZ 数组段（带 schema 版本号），SQLite 只做派生索引。

后续这一模块还会扩展支持智能博弈，作为大数据研究的数据基础。星历序列化以 EphemerisTable 为统一中间格式，后续引入与各种典型星历数据格式的转换（CCSDS OEM/ODM、SPICE BSP/PCK 等）。

其中，部分计算功能还由 Python 执行（NLP 优化等），正在逐步迁移至 Rust 计算核心。延拓、NSGA-II 演化算子与低推力直接法的数值评估内核已下沉，Python 保留外层编排和参照路径。时空系统和常量模块还需要进一步审查（三条帧转换路径并存、时间转换责任链）。Python 编排模块架构还不够稳定（打靶双路径、transfer 目录数值残留），需要进一步审查和进行系统设计。
