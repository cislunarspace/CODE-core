# ADR 0049: NRLMSISE-00 大气密度模型与大气接口的历元/地理位置扩展

**状态**：已采纳（已实现）
**日期**：2026-09-27
**相关 issue**：#637（本决策的直接动机）
**相关**：ADR 0030（数值只在 Rust，Python 只做配置与编排）、ADR 0038（按发表规格自行实现、外部实现仅作数值 oracle 的先例）、ADR 0011（Python 力模型类 ↔ Rust `CompiledForce`）、ADR 0018（drag 力元组格式）、ADR 0019（ITRF93 pxFrame 旋转）、ADR 0022/0048（基准配对）

## 背景

USSA76 分段指数模型（`ExponentialAtmosphere`，`crates/e2m2e-forces/src/atmosphere.rs`）
是仓库唯一的大气密度模型，接口是**纯高度**：`density(altitude) -> float`。它对
F10.7/Ap 只做一阶线性乘法修正，不含纬度、地方时、季节与历元依赖——用于地球停泊
弧段的阻力建模过于粗糙。

issue #637 要求在地球停泊弧段支持 **NRLMSISE-00**（Picone et al. 2002）。该模型
的输入面比指数模型宽得多，且这些输入不是"可选修饰"：

| 输入 | 指数模型 | NRLMSISE-00 |
|---|---|---|
| 高度 | 球面几何高度 | WGS84 椭球面以上大地高度 |
| 纬度/经度 | — | 大地纬经度（球谐展开的地理依赖） |
| 历元 | — | 年积日 + UT 秒（季节、地方时、UT 谐波） |
| 空间天气 | 标量 F10.7 + 标量 Ap | 前一日 F10.7、81 日平均 F10.7、3 小时分辨率 Ap 史 |

因此大气密度模型接口必须从"纯高度"扩为"历元 + 大地坐标 + 空间天气"，Python 与
Rust 两侧同步。硬约束：默认模型与既有数值路径保持 `ExponentialAtmosphere` 且
**逐位不变**（既有 drag golden 测试即证据）。

另一个必须显式处理的现实：模型系数是官方 NRL 发布的**数据**，不是可推导的公式。
仓库不允许把参考实现代码搬进来（AGPL 的 nyx 等），因此需要一条"规格 + 公有领域
数据"的实现路线，以及一条独立于自身实现的验证路线。

## 决策

### 1. 大气密度接口扩为 `(高度, 历元, 大地坐标)`，两模型统一签名

`ExponentialAtmosphere.density` 增加**仅关键字**参数 `epoch_et` / `geodetic_lat_deg`
/ `geodetic_lon_deg`，本模型忽略；“高度”由其自行按球面高度 `|r| − R_EARTH`
解释。`NRLMSISE00Atmosphere.density` 要求同样三个参数（必填）。

统一签名让 `DragModel` 与 Rust drag 管线可以按大气模型类型分派而不必分别适配接口，
也避免调用方在两条路径间写两套调用代码。

Rust 侧的落点在 `crates/e2m2e-forces/src/forces/drag.rs`：新增
`DragAtmosphere` 枚举（`Exponential { f107, ap }` / `Nrlmsise00 { f107_daily, f107_avg, ap[7] }`），
`CompiledForce::Drag` 的 `f107`/`ap` 两字段收进该枚举。阻力公式本身提取为
`drag_from_density(rho, v_bf, area, mass, cd)` 由两条密度来源共用，**算术顺序原样
保留**，故指数路径逐位不变。

### 2. NRLMSISE-00 数值只在 Rust，Python 经绑定查询

与 ADR 0030 一致：领域数值在 `crates/e2m2e-forces/src/nrlmsise00/`，
`e2m2e/algorithm/forces/atmosphere.py` 只做参数校验与编排，经
`nrlmsise00_density_py(epoch_et, alt, lat, lon, f107_daily, f107_avg, ap[7]) -> (ρ, T)`
查询。

该 PyO3 接口属 ABI 表面，`crates/e2m2e-integrators/abi-version.txt` 随之 +1（24 → 25），
既有安装的扩展与 Python 侧版本不一致时按 ADR 0039 的 ABI 门显式报错并提示
`make dev`，不静默降级。

### 3. 按发表规格自行实现，参考实现只作对拍 oracle

系数表转录自 NRL 官方模型数据（U.S. Government work，公有领域；见
`nrlmsise00/coefficients.rs` 文件头），数值路径（`spline`/`densu`/`densm`/`globe7`/
`glob7s`/`gts7`）按 Picone et al. (2002) 的规格与本模块结构逐一对应自行实现，
含运算顺序。**不参考、不拷贝** nyx（AGPL）等 GPL/AGPL 实现代码，仅在开发期把
nyx 公开的 `nrlmsise00_validation.json`（72 工况）当作数值 oracle（ADR 0038 先例）。

实现期用独立 oracle 复核：以 NRL 公有领域 C 参考实现（Brodowski 转写，release
20041227）在同一 72 工况上逐点比对，本实现与其 f64 结果**逐位相同**（密度与温度
相对误差恰为 0）。nyx 夹具本身是 f32 精度产物，故 Rust 集成测试
（`crates/e2m2e-forces/tests/nrlmsise00_validation.rs`）的容差取 1e-5（实测最大
偏差：密度 1.4e-6、温度 1.8e-7），并已在测试文件头记录该口径。

### 4. 静态空间天气的输入语义与 Ap 史启用规则

- `ap` 传标量（或 7 元全等序列）= 静态空间天气：Rust 侧用标量日 Ap（`ap[0]`），
  即参考实现的 `switch 9 = +1` 路径。
- `ap` 传 7 元非全等序列 = 暴时历史：Rust 侧用 3 小时分辨率的 Ap 史先验
  （`gtd7d` + `switch 9 = −1` 路径，`sg0` 加权的 `apt[0]`）。
- 判定规则统一为"`ap[1..7]` 是否与 `ap[0]` 不同"，Python 绑定、Python 类与 Rust
  drag 管线共用同一条规则，避免两侧分歧。
- 本模型**不接任何外部空间天气数据源**：F10.7 与 Ap 由调用方显式注入。

总质量密度一律取参考实现 `gtd7d` 的"drag 有效总质量密度"口径（含 500 km 以上
不可忽略的异常氧贡献）；`gtd7` 的不含异常氧口径不对外暴露——阻力建模只关心前者。
这一条是实测校准的结果：对拍夹具在静态 Ap 工况下同样按 `gtd7d` 口径给出期望值
（若按 `gtd7` 口径，483 km 工况系统性偏差 ~2e-4，恰为异常氧量级）。

### 5. 默认模型不变，指数路径逐位不变

`ForceModel` 的摄动开关 `atmosphere=1`（`force_mapping.py`）仍构造
`ExponentialAtmosphere`；阻力力元组 `("drag", ...)` 的格式与逐位数值不变。
NRLMSISE-00 走独立力元组 `("drag_nrlmsise00", area, mass, cd, frame, f107_daily, f107_avg, ap[7])`，
由配置里的 `{"type": "NRLMSISE00Atmosphere", "params": {...}}` 触发。既有 drag
golden 测试（Rust 单元测试与 Python 契约测试）未改动且全绿，即逐位证据。

`epoch_et → (年积日, UT 秒)` 走 SPICE `et2utc`（依赖 leap second 内核），因此
NRLMSISE-00 路径**需要已装载 SPICE 内核**；内核池为空时 `et2utc` 返回明确错误并
透传，不静默回退。

### 6. 大地坐标转换用 WGS84 + Bowring，独立成模块

`crates/e2m2e-forces/src/geodesy.rs` 提供 `ecef_to_geodetic`（纯数学、无 SPICE）：
WGS84 椭球（a = 6378.137 km、1/f = 298.257223563），Bowring (1985) 单步闭式解。
近地空间（0–1000 km）往返残差 ≤ 1e-7 deg 与 ≤ 1e-4 km，远小于大气密度标高，
对阻力流场无可见影响。

## 影响

- 大气密度能力面从"纯高度"扩为"历元 + 大地坐标 + 空间天气"，Python/Rust 接口同步；
  新增 ABI 版本 25。
- `DragAtmosphere` 枚举成为 drag 力的大气配置载体，`CompiledForce::Drag` 的字段
  形态变化（内部 API，非调用方契约）。
- 与 GMAT 的对拍口径：GMAT 侧最近的大气模型是 `MSISE90`（MSISE-1990），与
  NRLMSISE-00 不是同一版本；且 GMAT 对空间天气做连续样条平滑，而本模型按其原始
  标定使用离散 3 小时台阶。这两点是**模型口径差异**，不是实现缺陷——`scripts/generate_gmat_leo_script.py
  --drag-model MSISE90` 与 `scripts/compare_with_gmat.py --atmosphere nrlmsise00`
  用于生成对照脚本与记录本方输出，但两者不可期待亚米级一致。
- 未引入新的 Python 依赖；Rust 侧未新增 crate 依赖（系数表是常量数据）。

## 备选方案

- **在 Python 实现 NRLMSISE-00**：违背 ADR 0030，且阻力每步调用会跨越 GIL；
  否决。
- **把 3 小时 Ap 史压成标量以复用现有 `("drag", ...)` 元组**：丢失暴时响应，
  且会把两种空间天气语义混在一处；否决。
- **引入第三方 NRLMSISE-00 crate**：会引入未审计的许可与 ABI 依赖，且遮蔽
  "按规格自行实现"的可验证性；否决。
- **把异常氧贡献做成可开关**：调用方多一个无物理动机的旋钮，且默认值无论如何
  都应当是 drag 口径；否决，只暴露 `gtd7d` 口径。
