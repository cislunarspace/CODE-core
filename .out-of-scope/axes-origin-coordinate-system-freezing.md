# Axes / Origin / CoordinateSystem 运行时冻结

本项目不对 `Axes`、`Origin`、`CoordinateSystem` 的具体子类做运行时冻结
(``@dataclass(frozen=True)`` 或等价机制)。这些类的实例构造后即视为只读约定:
调用方不应在运行时换 ``axes`` / ``origin`` 组件,但应用层不强制。

## 为什么不在范围内

冻结机制(包括 `CoordinateSystem` 的 `@dataclass(frozen=True)`、`Axes` /
`Origin` 基类的 `__init_subclass__` 钩子豁免 `DynamicAxes`、所有具体子类
改 dataclass 形态)最初由 #76 引入 `CoordinateSystem` 冻结,#121/#122
跟进 `Axes` / `Origin` 全量冻结。落地后重新讨论,判定冻结不必要,理由:

1. **没有真实 bug 驱动**。冻结要解决的"运行时偷换组件导致变换结果不一致"
   没有在实际代码中出现过。`CoordinateSystem` 冻结上线后,`cs.axes = X` 这类
   写法既不在 core 也不在 forces / transfer / algorithms 层出现。问题是被
   想象出来的,不是被观察到的。

2. **篡改防护不在应用层职责**。如果有人在运行时把 `cs.axes` 换成另一个实例,
   这是程序本身被篡改的场景——用户从 GitHub 获取经验证的代码副本即可信任
   运行环境。应用层冻结无法防住"程序被改"的根因,只是把检查点挪到运行时。

3. **冻结的工程成本不低**。`IAU2000EqAxes` 等具体子类有派生字段(如
   `time_step` 派生的内部 `IAU2000EqAxes`),dataclass 形态需要
   `field(default_factory=...)` + `__post_init__` + `object.__setattr__`
   等样板;`DynamicAxes`(故意需要可变状态,ADR 0007 锁定)与冻结基类的
   边界要专门设计(组合 / 豁免钩子 / 平行继承树三选一,#122 全是为此)。
   这些复杂度为一个不存在的 bug 服务,得不偿失。

4. **YAGNI**。代码风格层面的"作者不应主动写 mutate"已经由
   `tests/core/coordinate/test_coordinate_immutability.py` 的 grep 守门员
   覆盖(扫描 `e2m2e/core/` 内的 `cs.axes = X` 这类赋值)。运行时偷换
   防护与之不同,本就不在应用层。

## 已做的反转

`CoordinateSystem` 在 #76 中改为 `@dataclass(frozen=True)`,本次反转回退到
普通类(commit `62f7308`)。`TestCoordinateSystemFrozen` 三个测试删除;
`TestCoordinateSystemOrthogonality` 与 `TestCoordinateSystemTransformVector`
(零向量)保留,与冻结无关。

## 保留

`tests/core/coordinate/test_coordinate_immutability.py` 的 grep 守门员保留。
它防的是"代码作者主动写 mutate"(静态代码风格),不是"运行时偷换"(动态防护)。
两者职责不同,前者与冻结无关,值得保留。

## 相关 issue

- #76 — `CoordinateSystem` 冻结(已 CLOSED,本次回退破坏其原决策)
- #121 — 静态 `Axes` / `Origin` 及其具体子类冻结(wontfix)
- #122 — `DynamicAxes` 与冻结基类的边界设计(wontfix,依赖 #121)
