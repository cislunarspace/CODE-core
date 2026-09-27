# ADR 0052: Lambert 内核迁移 pykep-core（评估后否决）

**状态**：已否决（性能门未过，保留自维护实现）
**日期**：2026-09-27
**相关 issue**：#715
**相关**：ADR 0017（转移网格搜索下沉 Rust）、ADR 0020（失败显式化）、ADR 0051（纯 Rust 后端替换 FFI 的先例）

## 背景

`crates/e2m2e-propagation/src/lambert.rs` 自维护一份 Izzo (2015) Lambert 实现
（代码结构移植自 Fortran-Astrodynamics-Toolkit，BSD-3-Clause，文件头与 NOTICE
携带署名）。#715 提议替换为 crates.io 依赖 `pykep-core`（0.1.4，MPL-2.0，
上游血统 ESA pykep/kep3 v3.0.1，pinned commit 53b1ca3ce5f8c223f96819b2ea9
ba16c3719e63e），对外契约（FFI/Python/porkchop）保持，并设定评估门：
对拍解一致、性能回归 ≤2×、许可兼容——任一失败即按 issue 终止条件保留自实现。

## 评估过程（全部实测）

### 对拍（通过，硬判据为解一致）

同一工况分别调用旧 FAT 内核与 pykep-core 适配原型（方向映射
`clockwise = (cross_z > 0) == (direction == LongWay)`，按（圈数, Right）挑右
分支）：Vallado 短程/76min/长程、Hohmann 型、多圈 revs=1/2、cross_z<0 双方向
几何、批量 4 几何 × 5 tof 网格——v0/vf 逐分量差 ≤ 1.8e-15 km/s（机器精度级），
Householder 迭代次数逐格相同；tof 不足时两侧错误文案逐字一致。多圈右分支与
旧实现一致。方向语义评估确认：本仓库 `TransferDirection` 是弧长参照（θ<π /
θ>π），pykep `clockwise` 是 +z 参照（先按转移面法向 z 分量自动翻转、再按
`clockwise` 翻转），两参照在 cross_z < 0 的几何上相差一次额外翻转，朴素映射
会静默给出相反弧。

### 许可（通过）

MPL-2.0 为文件级 copyleft，以未修改的 crates.io 依赖（库引用）方式使用，适用
MPL §3.3「Larger Work」，与 Apache-2.0 主项目共存可行；署名落 NOTICE 即可。

### 性能（未过门，否决主因）

基准：`scripts/benchmark_lambert_porkchop.py`（`porkchop_grid_states_py` 纯
Lambert+ΔV 热路径，确定性合成圆轨道网格 2400×30 = 72 000 格，串行
`parallel=False`，多次重复取中位；debug 与 release 两档构建各测一轮，数据与
复现方式见 `docs/plans/lambert-pykep-core-benchmark.md`）。

| 档位 | FAT 基线 | pykep | 比值 | 门限 2× |
| --- | --- | --- | --- | --- |
| revs=0（release） | 0.035 s | 0.048 s | 1.37× | 通过 |
| revs=1（release） | 0.037 s | 0.093 s | **2.51×** | **未过** |
| revs=0（debug） | 0.116 s | 0.246 s | 2.12× | 未过 |
| revs=1（debug） | 0.097 s | 0.312 s | 3.22× | 未过 |

revs=1 的回归为**架构性**：pykep 的 `LambertProblem::new` 对每个可行圈数固定
解零圈 + 左支 + 右支三组 Householder（API 无右支专用入口，左支对以右分支为
契约的本仓库是纯浪费），叠加可行性裁剪迭代；旧实现每个 tof 只解一次右支。
release 档重复测量离散度小（pykep 0.075–0.096 s、FAT 0.034–0.042 s），非
测量噪声。revs=0 的 1.37×（release）源于 pykep 每操作的 `hypot` 范数与有限性
校验层，属可接受范围，但救不了 revs=1。

## 决策

**否决迁移，保留自维护 FAT 移植实现**（issue 终止条件 #1）。代码、ABI、
NOTICE、CHANGELOG 全部回退；保留的产出：

- `scripts/benchmark_lambert_porkchop.py` 与基准报告——后续任何内核变更的
  性能锚（复测命令见脚本 docstring；性能基准用 release 构建，即
  `make dev-release` 档）。
- 本 ADR 与基准数据——未来 pykep-core 提供右支专用求解入口、或本仓库多圈
  契约放宽时，可据此重新评估，不必重跑对拍与许可审查。

## 理由

- 2× 否决线是 issue 原案设定的硬门；revs=1 在最能代表生产构建的 release 档
  仍 2.51×，且根因（多圈三解）是上游 API 结构，适配层无法消除。
- 评估门先行（先对拍、基准、许可，后动内核）的设计使本次否决零生产损失：
  内核重写、ABI bump、调用方适配均停在集成测试层面，全部回退。

## 影响

- `crates/e2m2e-propagation/src/lambert.rs` 保持 FAT 移植版；ABI 保持 25；
  NOTICE 与 docs 署名保持 BSD-3-Clause FAT 条目。
- 对拍中确认的三条 pykep 行为差异（NaN/Inf、转移平面含 z 轴、端点共线含
  180° 转移会显式报错）不生效，旧内核行为（任选法向 / 空转）保持。
- HMN 霍曼编排扫描几何与 `tests/algorithm/transfer/test_hohmann.py` 的精确
  对径构造保持原样（旧内核任选法向可解）。
- CONTEXT.md「转移设计」节新增四条术语（Lambert 求解器、转移方向、多圈
  分支、最小时间）——描述领域概念，与内核实现无关，保留。

## 备选方案

- **混合内核（revs=0 用 pykep、多圈保留 FAT）**：多圈恰是维护负担最重的
  部分，保留即未达成 #715 的去自维护目标，且 BSD-3 署名面照旧；否决。
- **等待上游**：pykep-core 若未来暴露右支专用入口（跳过左支），按本 ADR
  基线重测；届时对拍与许可结论可直接复用。
- **上调门限**：2× 门限来自 issue 原案（多圈网格是 porkchop 热路径，
  2.5× 直接吃掉并行收益的余量），无上调依据；否决。
