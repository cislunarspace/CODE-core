# ADR 0006：星历修正统一接缝与注册表分发

**状态**：已采纳
**日期**：2026-06-15
**关联 Issue**：#5（架构深化候选）

## 背景

`e2m2e/algorithms/ephemeris_correction.py` 中的 `correct_ephemeris_patch_points()` 用 `method: str` 参数和 `if/elif` 分支分发三种修正方法（standard、two_level、homotopy）。每条分支各自构造求解器、传入不同参数名、把不同结果类型重包成 `EphemerisCorrectionResult`。homotopy 分支还需要延迟导入以避免循环依赖。

这种分发模式的问题：
1. **接口几乎与实现一样复杂**：`method` 字符串是唯一的抽象，分支体是全部实现。
2. **新增方法需改分发函数**：每加一种修正方法，就要在 `if/elif` 链里加一条分支、一组参数翻译、一段结果重包。
3. **循环依赖**：`homotopy_correction.py` 从 `ephemeris_correction.py` 导入 `EphemerisCorrectionResult`，导致后者不能顶层导入前者。

ADR-0005 把 `TwoLevelMultipleShooting` 作为独立算法加入，但分发层仍是字符串分发。

## 决策

1. **定义 `PatchPointCorrector` 接缝。**
   - 在新模块 `ephemeris_correction_types.py` 中定义 `PatchPointCorrector` Protocol：
     ```python
     def correct(t_patch, state_patch, *, max_iter, tolerance, velocity_tolerance, verbose) -> EphemerisCorrectionResult
     ```
   - 构造参数（`n_workers`、`kernel_dir`、`base_bodies`、`lambda_steps` 等）通过构造器注入，不在统一接口中。
   - `EphemerisCorrectionResult` 也迁移到此模块。

2. **用注册表取代 `if/elif` 分发。**
   - `_REGISTRY: dict[str, Callable[..., PatchPointCorrector]]` 映射方法名到工厂函数。
   - 每个工厂创建一个私有的 `PatchPointCorrector` 实现（`_StandardPatchPointCorrector`、`_TwoLevelPatchPointCorrector`、`_HomotopyPatchPointCorrector`）。
   - 这些实现包装现有求解器，将统一参数翻译为求解器特定参数，并将结果重包为 `EphemerisCorrectionResult`。
   - `correct_ephemeris_patch_points()` 变为：查注册表 → 构造 corrector → 调用 `corrector.correct()`。

3. **解开循环依赖。**
   - `homotopy_correction.py` 改为从 `ephemeris_correction_types` 导入 `EphemerisCorrectionResult`。
   - `_HomotopyPatchPointCorrector` 在 `correct()` 方法内延迟导入 `correct_with_homotopy`（保持原有延迟导入语义）。

4. **错误类型明确化。**
   - 新增 `UnsupportedCorrectorMethodError(ValueError)` 替代原来的 `ValueError`。
   - 错误信息包含请求的方法名和可用方法列表。

5. **现有求解器接口不变。**
   - `MultipleShooting.correct()` 和 `TwoLevelMultipleShooting.correct()` 的签名、返回类型、行为均不改变。
   - 现有直接调用这些求解器的代码不受影响。

## 理由

1. **为什么用包装器而非直接改求解器签名。** 现有求解器有大量直接调用方（测试、DRO 端到端修正、CLI），改变其 `.correct()` 签名影响面太大。包装器把"接缝语言"和"求解器语言"分开，各自稳定演化。

2. **为什么用注册表而非 Protocol 注册。** Python 的 `Protocol` 不提供自动注册机制。一个显式的 `_REGISTRY` 字典足够简单、足够明确，新增方法只需加一行 lambda。

3. **为什么保留延迟导入。** `_HomotopyPatchPointCorrector` 在 `correct()` 内延迟导入 `correct_with_homotopy`，保持了原有的延迟加载语义：不使用同伦修正时不会触发 spiceypy 的加载。

4. **为什么不推翻 ADR-0005。** ADR-0005 决定 `TwoLevelMultipleShooting` 作为独立算法。本 ADR 强化了这一决定：两层求解器通过统一接缝被消费，内部实现保持独立。

## 结果

### 新增

- `e2m2e/algorithms/ephemeris_correction_types.py`：`EphemerisCorrectionResult`、`PatchPointCorrector` Protocol、`UnsupportedCorrectorMethodError`。
- `e2m2e/algorithms/ephemeris_correction.py`：三个私有 `PatchPointCorrector` 实现（`_StandardPatchPointCorrector`、`_TwoLevelPatchPointCorrector`、`_HomotopyPatchPointCorrector`）、`_REGISTRY` 注册表。
- `tests/algorithm/correction/test_patch_point_corrector.py`：接缝协议、注册表、分发、错误处理测试（20 个）。
- `docs/adr/0006-ephemeris-corrector-seam.md`。

### 变更

- `e2m2e/algorithms/ephemeris_correction.py`：`EphemerisCorrectionResult` 改为从 `ephemeris_correction_types` 导入；`correct_ephemeris_patch_points()` 改为注册表分发。
- `e2m2e/algorithms/homotopy_correction.py`：`EphemerisCorrectionResult` 改为从 `ephemeris_correction_types` 导入。
- `e2m2e/algorithms/__init__.py`：新增 `PatchPointCorrector`、`UnsupportedCorrectorMethodError` 延迟导出。

### 不变

- `MultipleShooting.correct()` 的签名、返回类型、行为。
- `TwoLevelMultipleShooting.correct()` 的签名、返回类型、行为。
- `correct_with_homotopy()` 的签名、行为。
- `correct_ephemeris_patch_points()` 的公开签名（向后兼容）。
- 所有现有测试（74 个算法测试 + 20 个新测试 = 94 个全部通过）。

### 后续工作

- Issue #8（MultipleShooting 并行内联）可复用同一 `PatchPointCorrector` 接缝。
- 新增修正方法只需：写一个 `PatchPointCorrector` 实现 + 加一行 `_REGISTRY` 注册。

## 修订（2026-08-13）

`ephemeris_correction` 子包（standard/two_level/homotopy 三个 `PatchPointCorrector`
实现 + 注册表分发）已整体删除：设计链路统一走 Rust 多重打靶
（``multiple_shooting_correct_py``，segmented 与稳定轨道默认路径），不再有
多个修正方法需要分发；`EphemerisCorrectionResult` 迁入
``e2m2e/algorithm/results.py`` 作为 Rust 打靶结果的领域重包。本 ADR 的决策 1/2/3/4
（接缝、注册表、延迟导入、错误类型）随之废止；决策 5 的
`MultipleShooting.correct()` 接口不变（transfer/hohmann 等非设计链路仍使用）。
关联 ADR 0005（同批删除 `TwoLevelMultipleShooting`）。
