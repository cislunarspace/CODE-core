# ADR 0026：测试套件层级澄清——coordinate 归属、forces 测试合并与死引用清理

**状态**：已采纳
**日期**：2026-08-14
**关联 Issue**：#429（forces 层级归属）、#430（dynamics 拆分）、#431（Python 数值迁移标注）
**关联**：ADR 0011（五层架构）、ADR 0012（依赖方向）、ADR 0021（测试套件功能类目）、ADR 0025（测试套件收敛）

## 背景

审计 `tests/algorithm`（14 子目录）时发现三处结构债：

1. **coordinate 标记与目录不一致**。`tests/algorithm/coordinate` 的 15 个文件全标 `data` 功能类，但目录在 `algorithm` 层。初判依据 architecture.md（"参考系转换"属"时空系统与常量模块"）与 ADR 0021（data 类含"坐标转换"），提议把 coordinate 源码与测试整体迁 data 层。
2. **forces 测试分裂**。源码 `e2m2e/algorithm/forces`（20 文件）的测试散在 `tests/numerical/forces`（约 45 个 force 标记测试，container/contract/physics 三层）与 `tests/algorithm/forces`（仅 `test_atmosphere.py` 一个）。
3. **normal_form 死引用**。`corrector.py` 有活 import `from e2m2e.core.dynamics import CR3BP_System`，而 `e2m2e.core`、`e2m2e.algorithms` 模块已在 ADR 0011 激进重命名中删除——该分支一执行即 `ImportError`。

深挖后推翻第 1 项的初判（见决策 1 理由），本 ADR 记录修正后的结论。

## 决策

1. **coordinate 留在 algorithm 层，不迁 data 层。** 尊重 ADR 0011 落地的既定决策——`e2m2e/algorithm/coordinate/__init__.py` 明确"转换算法归这里，data/frames 只留数据"。功能类标记（`data`）与代码层级（`algorithm`）是 ADR 0021 决策 2 明定的两个独立轴：目录按源码镜像，标记按"验证什么"，二者并存不矛盾。
2. **forces 测试合并。** `tests/algorithm/forces/test_atmosphere.py` 并入 `tests/numerical/forces/contract/`（与其他单模型契约测试并列），删除空的 `tests/algorithm/forces/` 目录。forces 源码层级归属（数值层配置面还是算法层力模型定义）留待独立 ADR，不在本篇裁决。
3. **清理 normal_form 及相邻模块的死引用。** 活 import 重映射到现行路径；docstring 交叉引用（`e2m2e.algorithms.*`、`e2m2e.core.*`）重映射到新路径；`data/__init__.py`、`data/templates/enums.py`、`exceptions.py` 三处描述 shim 历史兼容的说明是有意保留，不动。

## 理由

1. **coordinate 迁 data 会制造新的违规依赖。** coordinate 内部有反向依赖 algorithm 层的相对导入：`rho_bridge.py → ..dynamics.cr3bp_system.LibrationPoint`、`__init__.py 的 spacetime_convert → ..design.design_orbit / ..dynamics.cr3bp_system`。迁到 data 层后这些变成 data → algorithm，直接违反 ADR 0012"数据层不 import algorithm/"。而现状的 coordinate 反向依赖全部合规：api/facade → algorithm.coordinate（api→algorithm），design/station_keeping/forces/dynamics → coordinate（algorithm 层内部）。此前误报的"data/frames/gmat_fixture.py import coordinate"实为 docstring 文字，非 import。
2. **功能类与层级是两个轴，不该混淆。** ADR 0021 决策 2 原文："目录镜像源结构（导航用），标记标功能类（验证什么）"。data 类括注"内核/帧/类型/IO/模板 + 坐标转换"描述的是验证内容，不是代码位置。把"坐标转换属 data 类"读成"坐标转换代码应在 data 层"，是把标记轴当成了层级轴。
3. **forces 测试分裂违反目录镜像，合并是收尾而非重构。** ADR 0021 迁移时按"力=数值"的直觉把力模型测试放进 `tests/numerical/forces`，却留下 `test_atmosphere.py` 在 `tests/algorithm/forces`，同一源码包测试散两处。合并只动测试文件、不动源码 import，风险最小。
4. **死引用是 ADR 0011 的遗留，其中一处是真 bug。** docstring 交叉引用不影响运行，但 `corrector.py` 的活 import 指向已删模块，属必须修的缺陷；shim 说明（"旧路径经 shim 保持兼容"）描述现状，删改反而失真。

## 结果

### 变更

- `tests/algorithm/forces/` 删除；`test_atmosphere.py` 移入 `tests/numerical/forces/contract/`。
- normal_form：`corrector.py` 活 import 重映射为 `e2m2e.algorithm.dynamics.CR3BP_System`；`_ephemeris.py`、`fft.py`、`legendre.py`、`catalog.py`、`hamiltonian.py`、`coord_trans/`、`dynamical_substitution.py`、`multiple_shooting.py` 的旧路径交叉引用重映射（合计 12 处 `e2m2e.algorithms` → `e2m2e.algorithm`、1 处 `e2m2e.core.SPICEManager` → `e2m2e.data.kernels.manager.SPICEManager`）。
- forces/transfer/proximity：`thrust.py`、`exceptions.py`、`lowthrust_shooting.py`、`relative_dynamics.py` 的 `e2m2e.core.*` 交叉引用重映射到现行路径。

### 不变

- `e2m2e/algorithm/coordinate` 源码与 `tests/algorithm/coordinate` 测试位置。
- coordinate 测试的 `data` 功能类标记、forces 测试的 `force` 功能类标记。
- ADR 0011 的五层架构与 ADR 0012 的依赖方向规则本文。

### 后续工作（各自独立，不随本篇执行）

- forces 源码层级归属：`e2m2e/algorithm/forces` 是 Rust `e2m2e-forces` 的 Python 配置面（自述"参数验证 + to_rust_spec 序列化"），与 `e2m2e/integrators.py` 地位相近，是否移出 algorithm 层需单独 ADR，且受"数值层 Python 代码形态"这个未决问题约束。见 #429。
- dynamics 的 System/Dynamics 拆分：`CR3BP_System`（物理系统定义）与 `CR3BP_Dynamics`（构造积分问题）层级不同，属高风险大迁移。见 #430。
- normal_form、solver、transfer、family、manifold 中的 Python 数值是 ADR 0011 明示的"正在逐步迁移 Rust"过渡状态，需在文档标注迁移进度，避免后续审计再次误判为放错层。见 #431，进度清单落地于 `docs/architecture/numerics-migration-status.md`。
