# 第 5 批清理计划：io/ 全量迁出与旧包删除

> 状态：待审查。本计划接续 `docs/architecture/migration-to-five-layer.md`，
> 是五层架构迁移的最后一批。旧包 `core/`/`algorithms/`/`transfer/`/`dfh/`/
> `visualization/`/`proximity/` 的 `__init__.py` 已删除，只剩 `io/`。

## 现状

`e2m2e/io/` 含 7 个源文件 + 1 个 golden 夹具（1251 行 Python），仅 2 个算法模块引用：

```
e2m2e/algorithm/design/design_orbit.py
  ├── from e2m2e.io import EphemerisTable          ← 已迁至 data.types.trajectory
  └── from e2m2e.io import write_ephemeris          ← DFH 格式写出

e2m2e/algorithm/station_keeping/controller.py
  ├── from e2m2e.io.ephemeris import read_ephemeris ← DFH 格式读入
  ├── from e2m2e.io.inputs_dac import DEFAULT_DYB, DEFAULT_PERTURBATION
  │                                                   ← 已迁至 data.templates.perturbations
  ├── from e2m2e.io.ephemeris import write_ephemeris ← DFH 格式写出
  ├── from e2m2e.io.maneuvers import write_maneuvers ← DFH 格式写出
  └── from e2m2e.io.sk_statistic import write_sk_statistic ← DFH 格式写出
```

测试引用 11 个文件（`tests/io/` 6 个 + `tests/dfh/` 4 个 + `tests/data/` 2 个）。

## io/ 各模块的归属判断

按内容性质分三类：

| 模块 | 行数 | 性质 | 去向 |
|------|------|------|------|
| `__init__.py` | 66 | 纯 re-export | **删除** |
| `force_mapping.py` | 18 | 纯 shim → `algorithm.forces.force_mapping` | **删除** |
| `ephemeris.py` | 97 | `EphemerisTable`（已迁）+ DFH 格式 parse/read/write | parse/read/write 迁入 `data.types.trajectory`，旧文件删 |
| `maneuvers.py` | 64 | `ManeuverTable`（已迁）+ DFH 格式 parse/read/write | parse/read/write 迁入 `data.types.maneuver`，旧文件删 |
| `sk_statistic.py` | 73 | `SKStatistic`（已迁）+ DFH 格式 parse/read/write | parse/read/write 迁入 `data.types.sk_statistic`，旧文件删 |
| `results.py` | 349 | `HmnResult`/`MultiOrbitResult`/`OrbitSegment` dataclass + DFH 格式解析 | `scripts/dfh_results.py`（不进包，仅 dfh 测试用） |
| `inputs_dac.py` | 584 | DFH inputs-dac 生成器 + re-export `DEFAULT_DYB`/`DEFAULT_PERTURBATION` | `scripts/dfh_inputs_dac.py`（含 golden 文件） |

**原则**：
- 类型容器（`EphemerisTable` 等）已迁 `data/types/`——不回头。
- DFH 文本格式读写（`parse_*/read_*/write_*`）是**通用容器** ↔ 文本的序列化，放在容器所在模块内——与容器同生命周期，不给算法层添新依赖。
- `results.py` 和 `inputs_dac.py` 是**纯 DFH 专有格式**——不归入 e2m2e 包，移 `scripts/` 作开发期脚本。
- `force_mapping.py` 已经是 shim——直接删。
- `e2m2e.io` 包整体删除，顶层不导出。

## 分步执行

### 第 1 步：迁 DFH 格式读写到 `data/types/`

把 `io/ephemeris.py`、`io/maneuvers.py`、`io/sk_statistic.py` 中的 `parse_*`/`read_*`/`write_*` 函数迁入各自类型模块：

**`data/types/trajectory.py`** 新增：
- `parse_ephemeris(raw: str) -> EphemerisTable`
- `read_ephemeris(path) -> EphemerisTable`
- `write_ephemeris(table, path) -> Path`

**`data/types/maneuver.py`** 新增：
- `parse_maneuvers(raw: str) -> ManeuverTable`
- `read_maneuvers(path) -> ManeuverTable`
- `write_maneuvers(table, path) -> Path`

**`data/types/sk_statistic.py`** 新增：
- `parse_sk_statistic(raw: str) -> SKStatistic`
- `read_sk_statistic(path) -> SKStatistic`
- `write_sk_statistic(table, path) -> Path`

这些函数实现直接从旧模块复制（它们已从 data/types 导入容器类型，无循环依赖）。

### 第 2 步：修正算法层引用

**`algorithm/design/design_orbit.py`**：
- L41 `from e2m2e.io import EphemerisTable` → `from ...data.types.trajectory import EphemerisTable`
- L169 `from e2m2e.io import write_ephemeris` → `from ...data.types.trajectory import write_ephemeris`
- L748-749 docstring 中的 `io.` 前缀去掉

**`algorithm/station_keeping/controller.py`**：
- L25 `from e2m2e.io.ephemeris import read_ephemeris` → `from ...data.types.trajectory import read_ephemeris`
- L26 `from e2m2e.io.inputs_dac import DEFAULT_DYB, DEFAULT_PERTURBATION` → `from ...data.templates.perturbations import DEFAULT_DYB, DEFAULT_PERTURBATION`
- L75 `from e2m2e.io.ephemeris import write_ephemeris` → `from ...data.types.trajectory import write_ephemeris`
- L76 `from e2m2e.io.maneuvers import write_maneuvers` → `from ...data.types.maneuver import write_maneuvers`
- L77 `from e2m2e.io.sk_statistic import write_sk_statistic` → `from ...data.types.sk_statistic import write_sk_statistic`

### 第 3 步：移纯 DFH 脚本到 `scripts/`

- `e2m2e/io/results.py` → `scripts/dfh_results.py`（仅 `tests/dfh/` 和 `tests/io/test_results.py` 引用）
- `e2m2e/io/inputs_dac.py` → `scripts/dfh_inputs_dac.py`
- `e2m2e/io/data/inputs-dac.golden` → `scripts/data/inputs-dac.golden`（更新脚本内 `_GOLDEN_PATH`）
- 从 `e2m2e/io/__init__.py` 中删除 `results` 和 `inputs_dac` 的导出

### 第 4 步：更新测试引用

**`tests/io/` → `tests/dfh_format/`**：
- `test_ephemeris.py`：`from e2m2e.io` → `from e2m2e.data.types.trajectory`
- `test_maneuvers_sk.py`：分拆导入——容器从 `data.types.maneuver`/`data.types.sk_statistic`，R/W 从 `data.types.trajectory` 等
- `test_force_mapping.py`：`from e2m2e.io` → `from e2m2e.algorithm.forces.force_mapping`
- `test_results.py`：`from e2m2e.io` → `from scripts.dfh_results`（或在测试内 sys.path 加 scripts/）
- `test_inputs_dac.py`：`from e2m2e.io` → `from scripts.dfh_inputs_dac`

**`tests/data/test_templates.py`** L72-73：
- `from e2m2e.io import DEFAULT_DYB as io_dyb` → `from e2m2e.data.templates.perturbations import DEFAULT_DYB as io_dyb`

**`tests/data/test_types_trajectory.py`** L36：
- `from e2m2e.io.ephemeris import read_ephemeris, write_ephemeris` → `from e2m2e.data.types.trajectory import read_ephemeris, write_ephemeris`

**`tests/dfh/`** 4 文件：导入路径 `e2m2e.io` → `scripts.dfh_*` 或在测试 conftest 加 path。

### 第 5 步：删 `e2m2e/io/`

全部文件删除（含 `__init__.py`、`force_mapping.py`、`ephemeris.py`、`maneuvers.py`、`sk_statistic.py`、`data/`）。`e2m2e/__init__.py` 已不导出 io。

### 第 6 步：全量验证

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy e2m2e/ --ignore-missing-imports
uv run pytest -n auto -q
```

## 风险与边界

- **不重构**：只迁文件位置 + 改 import 路径，不改函数签名或实现逻辑。
- **`DEFAULT_DYB`/`DEFAULT_PERTURBATION`**：唯一来源是 `data.templates.perturbations`（`io/inputs_dac.py` 也是从它 re-export）。
- **`inputs-dac.golden`**：随 `inputs_dac.py` 一起移到 `scripts/data/`，脚本内 `_GOLDEN_PATH` 由 `Path(__file__).resolve().parent / "data" / ...` 自动适配新位置。
- **`scripts/` 目录**：已存在 20+ 个脚本。新增 `scripts/dfh_results.py`、`scripts/dfh_inputs_dac.py`、`scripts/data/` 子目录。

## 完成标准

1. `e2m2e/io/` 目录不存在。
2. 算法层无一引用 `e2m2e.io`。
3. DFH 格式读写函数可通过 `e2m2e.data.types.trajectory`（等）正常导入。
4. `tests/io/` 改名 `tests/dfh_format/`，全部测试通过。
5. lint + typecheck 全过。
6. 全部测试通过，无回归。
