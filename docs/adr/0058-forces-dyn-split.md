# ADR 0058：e2m2e-forces 拆出 e2m2e-dyn——非力任务设计模块迁出

- 状态：已采纳（已实现）
- 日期：2026-09-28
- 关联 issue：#752
- 相关：ADR 0020（PropagateError 语义）

## 背景

`e2m2e-forces` 除力模型与动力学内核外，还收纳了 10 个非力模块，crate 名不副实：

| 模块 | 职责 |
|---|---|
| `cartography` | 地心 EM/EMS 六域制图单格传播（命运 + MEGNO） |
| `low_energy_patch` | 低能转移流形截面态配对 |
| `manifold` | 不变流形种子生成与批量弧传播 |
| `megno` | CR3BP/BCR4BP MEGNO 传播 |
| `pal_continuation` | 伪弧长延拓（F/dF/切向量/牛顿迭代） |
| `porkchop` | porkchop 网格扫描（终端传播 + Lambert + ΔV） |
| `qlaw` | Q-law 低推力反馈积分 |
| `transfer_geometry` | 转移搜索几何核（最近距离/相交/碰撞） |
| `transfer_grid_search` | 转移网格搜索（串行/Rayon 并行） |
| `wsb` | WSB 三维网格搜索 |

同批 issue #752 还把 `e2m2e-integrators` 的 lib.rs 巨石按主题拆为 `bindings/` 绑定模块（注册仍集中在 `#[pymodule]`），本 ADR 只记录 crate 边界决策。

## 决策

1. 新建 crate `e2m2e-dyn`，上述 10 个模块**纯搬家**迁入：算法、签名与测试零改动，仅 `use crate::cr3bp/bcr4bp/PropagateError` 改为 `use e2m2e_forces::...`。
2. `geodesy` 留守 forces：`forces/drag.rs` 的 ITRF93 大气阻力依赖 `ecef_to_geodetic`。
3. `cr3bp`/`bcr4bp` 内核与 `PropagateError` 留守 forces；dyn 以 `default-features = false` 依赖 forces 取用，自身无 CSPICE 构建链，可独立测试。rayon/crossbeam-channel/nalgebra 三依赖随之从 forces 移入 dyn。
4. pyo3 绑定与 Python 符号面全部留在 e2m2e-integrators（绑定改指 `e2m2e_dyn::`），`abi-version.txt` 与 `_py_abi_version()` 不动，无 ABI 变化。
5. `e2m2e-hjb-dynamics` 不动：其 `e2m2e_forces` 引用（`forces::compiled::CompiledForce` 与 dev-dep `cr3bp`）均为留守符号。

## 后果

- 正面：forces 职责纯化（只剩力模型 + 动力学内核）；dyn 无 CSPICE 构建链、可独立快速测试；依赖方向与命名一致（dyn→forces 单向）。
- 负面：workspace +1 crate、+2 依赖边（dyn→forces、integrators→dyn），跨 crate 跳转增加。
- 风险控制：纯搬家迁移（数值行为不变）、`git mv` 保留文件历史、逐 crate 测试门禁（dyn/forces/integrators 分别 `--test-threads=1` 后再跑全 workspace）。

## 修订记录

- 2026-09-28：首次记录。
