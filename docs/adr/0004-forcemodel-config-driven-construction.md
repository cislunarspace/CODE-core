# ADR 0004：ForceModel 配置驱动

**状态**：已采纳
**日期**：2026-06-15
**关联 Issue**：#69

## 背景

Issue #69 想让用户写一份配置（JSON 或 dict）就能建出一套力模型（J2 + 阻力 + 光压 + 有限推力），还能存盘再读回，读回再建出的力与原先一样。现有容器已经能聚合多个 `PhysicalModel` 并通过 Rust `rk_step` 步进器传播（ADR 0002），每个力模型也已经各自完成坐标变换（ADR 0003）。还差两件事：一是按名字找到单个力：`get_force`/`remove_force`/`enable`/`disable` 与配置往返都要用它；二是把每种力的参数写成数据，包括那些本身持有别的实例、或持有 Python 函数的力。

有两处比较难办。`DragModel` 收一个 `AtmosphereModel`，`SolarRadiationPressure` 收一个 `ShadowModel`，这俩都是实例，不是几个数。`FiniteBurn` 收一个 `thrust_profile` 函数，它的 `direction` 也可能是个函数，一般情况下这俩都没法存进 JSON 再原样读回来。

## 决策

1. **力的身份（`name`）与开关状态（`enabled`）放在容器里，不放在 `PhysicalModel` 实例上。** `ForceModel` 内部维护一个有序的 `ForceEntry(name, force, enabled)` 注册表。`add_force(force, name=None)` 在省略名字时自动取 `type(force).__name__`，遇同类自动消歧（`Foo`、`Foo_2`、`Foo_3`…）；显式给出且与已有名字冲突时抛 `ValueError`。`get_force`/`remove_force`/`enable`/`disable` 按名字操作；`list_forces()` 返回这些条目。现有的 `forces` 属性不变（仍返回 `tuple[PhysicalModel, ...]`）以保持向后兼容。禁用一个力会让传播时跳过它，但仍把它留在容器里、留在 `to_config` 输出里（`enabled: false`）。

2. **配置 schema 是一个带版本号、由带名字的条目组成的清单。** 顶层是 `{"version": 1, "forces": [...]}`。每条是 `{"name", "type", "enabled", "params"}`，其中 `type` 是 Python 类名（也是注册表的 key），`params` 存该力的构造参数。传入的依赖（`DragModel` 的 `atmosphere`、`SolarRadiationPressure` 的 `shadow`）是嵌套条目，形状同为 `{type, params}`，递归处理；`null` 表示未注入（如完全日照下的光压）。`to_config` 输出解析后的实际值，即构造器默认值生效之后的值，因此 `GravityField(degree=2)` 与 `GravityField(degree=2, order=2)` 产生相同的配置。

3. **`FiniteBurn` 只能通过一套封闭 DSL（一组固定写法）从配置构造。** `thrust_profile` 接受 `{"kind": "constant", "thrust": N}` 或 `{"kind": "pulse", "t_start", "t_end", "thrust"}`；`direction` 接受 `{"kind": "fixed", "vector": [x, y, z]}`。`from_config` 据此构造闭包并打上标记（`_e2m2e_config_kind`），使 `to_config` 能反向还原。若一个 `FiniteBurn` 的可调用对象不是来自这套 DSL（比如用户手写的 `lambda t: ...`），`to_config` 时抛 `NotSerializableError`。它照常传播，只是无法序列化。

4. **往返验收标准 = 配置字典相等。** `to_config(from_config(config)) == config` 是 round-trip 的验收性质：Python 字典完全相等，没有容差。轨迹相等是另一项物理检查，交给 LEO 端到端测试。

只动了一个现有类：`GravityField.__init__` 把原始的 `gravity_file` 参数存下来（`self._gravity_file_arg`），使自定义的 `.gfc` 路径能往返。这是对 `PhysicalModel` 子类的唯一侵入；其余全部放进新模块 `e2m2e/algorithm/forces/force_config.py`。

## 理由

1. **为什么 `name`/`enabled` 放在容器而不是实例上。** `PhysicalModel` 实例会流经多个模块（`FiniteBurn` 被推力处理引用，`GravityField` 被引力路径引用）。在基类上放一个 `name` 属性，会把容器专用的标签带进每一个模块；而 `enabled` 会暗示关掉是力本身的属性，并非如此：一个力永远能算加速度，只是容器决定是否调用它。注册表把标签留在它被使用的地方。

2. **为什么用封闭 DSL，而不是排除 `FiniteBurn` 或接受任意可调用对象。** 排除 `FiniteBurn` 会让验收标准（覆盖推力）失去对应物：`ImpulsiveBurn` 不是 `PhysicalModel`，不在容器里。一个具名可调用对象的注册表会把用户代码泄进配置，破坏跨会话往返。封闭 DSL 覆盖现实情形（常开推力、开关脉冲、固定方向），对其余情形大声报错，而不是静默产出一个无法重新加载的配置。

3. **为什么用类名作为类型判别字段。** 目前只有四种力类型，还没有公开稳定标识符的需求，类名直接作为注册表 key，不必维护一层翻译。日后需要时，别名可以作为备用 key 加入，不破坏现有配置。

4. **为什么序列化解析后的实际值。** 用不同写法构造、但实际参数相同的两个实例，是同一个力。序列化解析后的值，让往返与用户恰巧怎么调用构造器无关；而现有的只读属性已经把它们暴露出来。

5. **为什么用配置字典相等作为往返验收标准。** 它精确、断言成本低。轨迹相等依赖积分器与浮点，只能带容差检查，作为单独的合理性核对有用，但不适合作为往返通过的定义。

## 结果

### 新增

- `e2m2e/algorithm/forces/force_config.py`：type→builder 与 type→serializer 分派、`FiniteBurn` 的 DSL 构造器、递归的 atmosphere/shadow 构造器、JSON 的 `load_force_config`/`dump_force_config`。
- `ForceModel`：`ForceEntry` 注册表、`add_force(name=)`、`remove_force(name | index)`、`get_force`、`list_forces`、`enable`、`disable`、`from_config` 类方法、`to_config` 方法。
- `GravityField._gravity_file_arg`（存原始路径）。

### 变更

- `ForceModel._forces` 改为 `ForceEntry` 元组 `_entries`；`forces` 属性仍返回 `PhysicalModel` 以兼容。
- 传播路径（`_propagate_via_rust` 等）跳过 `enabled=False` 的条目。

### 不变

- 四个 `PhysicalModel` 子类的物理与 `compute_acceleration` 签名（除 `GravityField` 存原始 `gravity_file` 参数外）。
- `propagate`/`propagate_maneuvers` 行为（传播时只取启用的条目，故禁用的力自然被排除）。
- 坐标变换职责：每个力模型自行变换；见 ADR 0003。

### 后续工作

- 通过新增 `kind` 值扩展 `FiniteBurn` 的 DSL（VNB/LVLH 方向对准、时变推力曲线），向后兼容。
- 若日后需要公开稳定标识符，把力类型别名作为备用注册表 key 加入。
- 当 `version` 超过 1 时，做带版本的 schema 迁移。
