# ADR 0050: Sims-Flanagan 工具链双档保真度——问题级 backend，conic 快筛档先交付

**状态**：已采纳（决策落地：conic 档随 #725 实施，星历档另开切片）
**日期**：2026-09-27
**相关 issue**：#719（本决策 issue）、#718（迁移总纲地图）、#725（Sims-Flanagan 实施）、#726（segment 框架）
**相关**：ADR 0034（沿保真度轴升级的裁决先例）、ADR 0020（无隐式降级）、
ADR 0025（显式后端选择）、ADR 0030（数值只在 Rust）、ADR 0039（共享内核
叶模块与 ABI 门）

## 背景

Issue #719 提出迁移后的 Sims-Flanagan/MGA 工具链按什么保真度档位建，三个
选项：

- **A. 仅 conic 快筛单档**：二体内核（太阳中心体 + 段中冲量），MALTO 论文
  原文即此档，纯广泛搜索预设计。
- **B. 双档参数化**：同一问题结构（leg / 控制节点 / 匹配点），传播后端
  可选——conic 档先交付，星历 n 体档后交付。
- **C. 直接高保真档**：不建 conic 快筛档。

代码事实（#718 scout 盘点 + 本裁决源码核实，2026-09-27）：

- `e2m2e/algorithm/transfer/` 无任何 Sims-Flanagan / MGA / matchpoint 代码，
  无既有结构可沿用，也无冲突需要收敛。
- Rust 侧无通用二体 Kepler / universal-variable 正向传播暴露。既有
  `multi_impulse.propagate_two_body` 是 SciPy DOP853 数值积分（其 docstring
  自称"最小版本"），非封闭解，不满足快筛档在广泛搜索中百万次量级调用的
  性能要求——不改不删，仅不作为 conic 档内核。
- 星历 n 体档的执行内核现成：`EphemerisDynamics` +
  `propagate_with_stm_py`（含 STM 输出），缺的只是问题层接口参数化。

## 决策

1. **选 B：双档保真度结构。** SF 问题结构（leg / 节点 / 匹配点 / 匹配点
   连续性约束）一次建对，传播后端可替换；保真度档与问题结构正交。
2. **后端接口为问题级 `backend` 参数**：构造时取 `"conic"` 或
   `"ephemeris"`，全问题统一档位。逐 leg 混档不在 #725 预建抽象，留给
   #726 segment 框架裁决。
3. **conic 内核 Rust 新建**：universal-variable 封闭解传播器 + 解析 STM
   （f & g 系列偏导数封闭式）。Python 绑定为单符号
   `propagate_kepler_py(state0, t_eval, mu, *, with_stm=False)`，签名风格
   对齐 `propagate_with_stm_py` 的 `t_eval` 形状语义；不做批量入口，
   porkchop 式批量后置到有真实消费方时再议。该符号属 pyfunction 边界
   新增，实施时按 ADR 0039 的 ABI 门 bump `abi-version.txt` 并同步
   `e2m2e/integrators.py` 符号面（`TYPE_CHECKING` / `_RUST_SYMBOLS` /
   `__all__`）。
4. **MGA / flyby 转角模型后置**：#725 范围仅纯 SEP（太阳中心二体 +
   段中冲量）；flyby 节点的 V∞ 旋转模型另开切片。
5. **交付顺序**：#719 只锁结构双档，不写代码；conic 档随 #725 交付；
   星历 n 体档另开新切片 issue（复用决策所述现成内核），依赖 #725 的
   backend 接口交付。

## 理由

1. **B 优于 A**：星历档执行内核现成，B 的增量成本仅在接口参数化；A 省下
   的接口设计在日后加档时返工（改问题构造与全部调用点）。
2. **B 优于 C**：直接高保真慢且验证难，与迁移总纲 broad search 的动机
   相悖；快筛档与精化档各司其职。
3. **Copernicus 的"每段可选拉普拉斯传播帧"**（IAC-06-C1.4.04）证明
   leg/matchpoint 结构天然支持多档传播后端。本决策把可变性收敛到问题级，
   逐 leg 混档的收益（同一段问题里 conic 与星历混合）没有已知的任务场景，
   属 #726 segment 框架的抽象范围。
4. **conic 内核不数改 `propagate_two_body`**：数值积分器每步一次右端求值
   加自适应步控，封闭解 O(1) 次迭代；改它反而动摇既有 multi-impulse
   路径的已验证行为，新建符号零回归风险。
5. **与 ADR 0034 的关系**：0034 的裁决思路是"沿时间保真度轴升级而非撞
   空间保真度墙"；本决策是同一思想在传播后端上的应用——结构先立，保真度
   档作为可替换维度逐档补齐。
6. **与 ADR 0020 的关系**：`backend` 取值缺失或不支持时必须显式报错并
   提示可用档位，不得静默回退到另一档——保真度差异是结果语义差异，静默
   降级会让调用方把星历档结论误当 conic 档快筛（或反之）。

## 影响

- #725 实施时：`crates/e2m2e-integrators` 新增 `propagate_kepler_py`，
  ABI 版本自当前 25 起自增；`e2m2e/integrators.py` 三处符号面同步。
- `backend` 取值集合固定为 `"conic" | "ephemeris"`；后续新增档位走 ADR
  修订，不静默扩集合。
- 星历 n 体档切片 issue 挂 #718 地图并表达对 #725 的 blocked_by 依赖。
- 本决策无面向调用方的行为变化，`CHANGELOG.md` 不更新；#725 进代码时再
  记。

## 备选方案

- **A 仅 conic 单档**：接口最省，但加档时问题构造与调用点全面返工；
  否决。
- **C 直接高保真单档**：与 broad search 动机相悖且验证链路长；否决。
- **逐 leg 混档参数（leg 级 backend）**：无已知任务场景，#725 预建抽象
  违背"不为假想需求设计"；移交 #726 segment 框架统一考虑；本决策不采纳。
- **复用 `propagate_two_body` 作为 conic 内核**：性能不达快筛档要求，
  且改动有回归风险；否决，保留原样服务既有 multi-impulse 路径。
