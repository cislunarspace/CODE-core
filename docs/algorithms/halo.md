# Halo 轨道生成与轨道族延拓

> Halo 轨道是围绕 L1/L2 平动点的三维周期轨道。本页涵盖 Richardson 初值、微分修正、伪弧长延拓和命令行脚本。

## 快速开始：从零生成一条 Halo 轨道

```python
import numpy as np
from e2m2e.core import CR3BP_System, CR3BP_Dynamics
from e2m2e.algorithms import DifferentialCorrection

system = CR3BP_System.from_known_system("earth_moon")
system.set_characteristic_scales(distance=384400, period=27.32*86400)
system.compute_libration_points()
dynamics = CR3BP_Dynamics(system)

# Richardson 三阶近似提供初值
from e2m2e.algorithms import compute_halo_initial_guess
initial_state, t_half = compute_halo_initial_guess(
    system=system, libration_point=1, amplitude_z=0.1
)

# 微分修正
dc = DifferentialCorrection(dynamic=dynamics)
dc.setup_halo_orbit_fixed_z0(z0=initial_state[2], libration_point=1)
orbit, result = dc.iterate_correction(initial_state, t_half=t_half)

print(f"周期: {orbit.period:.4f}, Jacobi: {orbit.jacobi_constant:.6f}")
```

## 快速开始：生成 Halo 轨道族

```python
from e2m2e.algorithms import Continuation

dc = DifferentialCorrection(dynamic=dynamics)
cont = Continuation(corrector=dc)

# 生成种子轨道
seed = cont.generate_halo_seed_orbit(
    libration_point=1, amplitude_z=0.23, halo_class=0,
)

# 伪弧长延拓
family = cont.halo_pseudo_arclength_continuation(
    seed_orbit=seed,
    n_orbits=10,
    direction="both",
    step_size=0.0045,
    verbose=True,
)

family.save_to_file("output/halo_family.json")
print(f"生成了 {len(family)} 条 Halo 轨道")
```

---

**相关源码**：

| 模块 | 路径 |
|------|------|
| 延拓与 Halo 族 | `e2m2e/algorithms/continuation.py` |
| Richardson 初值与 Halo 微分修正 | `e2m2e/algorithms/differential_correction.py` |
| 单轨道 / 族生成脚本 | `scripts/generate/generate_halo_orbit.py`、`scripts/generate/generate_halo_family.py` |
| 绘图脚本 | `scripts/plot/plot_halo_orbit.py`、`scripts/plot/plot_halo_family.py` |
| 解析初值测试 | `tests/algorithms/test_analytical_halo.py` |

---

## 功能概览

1. **单条 Halo 周期轨道**：Richardson 三阶近似初值 + `DifferentialCorrection`（固定 `z0` 或 `x0`）。
2. **种子轨道**：`Continuation.generate_halo_seed_orbit` — 与单轨流程一致，并写入 `parameters`（`libration_point`、`amplitude_z`、`halo_class`）。
3. **伪弧长延拓（XZ 对称）**：`Continuation.pseudo_arclength_continuation` — 自由变量 \(\mathbf{X}=[r_x, r_z, \dot{y}, T/2]\)，与 `CR3BP_MATLAB_Library` 中 `continuation_PAL_CR3BP`（`plane=13`）同构。
4. **Halo 轨道族**：`Continuation.halo_pseudo_arclength_continuation` — 在 PAL 基础上按脚本 `FAMILY_L1Halo_North.m` 配置正负支步长、`DirectionalIncrement` 与目标分量；微分修正策略可选（见下文）。

---

## 核心 API

### `DifferentialCorrection`

| 方法 | 作用 |
|------|------|
| `compute_halo_initial_guess(mu, z_amplitude, L, halo_class)` | Richardson/MATLAB 标度初值（`x0`, `vy0`, `T_half` 等） |
| `setup_halo_orbit_fixed_z0(z0, libration_point)` | 固定初始 \(z_0\)，自由变量 \((x_0, \dot{y}_0, T/2)\) |
| `setup_halo_orbit_fixed_x0(x0, libration_point)` | 固定初始 \(x_0\)，自由变量 \((z_0, \dot{y}_0, T/2)\) |

Halo 收敛结果会校验完整周期 \(T\) 的下限（避免 \(T\to 0\) 的寄生根），见 `iterate_correction` 中对 `halo_orbit_fixed_z0` / `halo_orbit_fixed_x0` 的处理。

### `Continuation`

| 方法 | 作用 |
|------|------|
| `generate_halo_seed_orbit(libration_point, amplitude_z, halo_class, ...)` | 生成并修正单条种子 Halo |
| `generate_halo_family(seed_orbit, ...)` | 按 `amplitude_z` 步长独立 Richardson 初值（自然参数式，非 PAL） |
| `pseudo_arclength_continuation(seed_orbit, ...)` | 通用 XZ 对称伪弧长延拓（单方向 `positive` / `negative`） |
| `halo_pseudo_arclength_continuation(seed_orbit, ...)` | Halo 专用：双向支、默认步长与 MATLAB 脚本对齐的可选参数 |

**`pseudo_arclength_continuation` 主要参数**：

- `step_size`：\(|\Delta S|\)（正数；方向由 `direction` 决定）。
- `dc_scheme`：`adaptive`（按 \(\Delta x\)、\(\Delta z\) 在 3D 对称 fixed-x / fixed-z 间切换）、`matlab_halo_type1`（始终 `setup_halo_orbit_fixed_x0`）、`matlab_halo_type2`（在 fixed-x / fixed-z 间按幅值切换）。
- `directional_increment`、`target_vector`（0 基：\(0=r_x,1=r_z,2=\dot y,3=T/2\)）、`target_direction`：与 MATLAB `DirectionalIncrement` / `TargetVector` / `TargetDirection` 一致。

**`halo_pseudo_arclength_continuation` 要点**：

- `step_size` / `step_size_negative`：对应脚本中正支 `DeltaS≈0.0045`、负支 `|DeltaS|≈0.009`。
- 默认 `dc_scheme='adaptive'`：PAL 给出的初值在 Python STM 牛顿下与 MATLAB 固定 `x0` 行为不完全一致时更稳健；若需对齐 MATLAB `type=1`，可设 `matlab_halo_type1`（实现中可在 fixed-x 失败后再试 fixed-z）。

---

## PAL 实现说明（与 MATLAB 的差异与防护）

- **内层牛顿顺序**：与 `continuation_PAL_CR3BP.m` 一致，在 \(\|F\|\) 已小于容差时**不再**对 \(\mathbf{X}_{new}\) 多走一步，避免把点推离物理解。
- **牛顿步限幅**：对 \(\Delta\mathbf{X}\) 分量施加上限，减轻跳入 \(F=0\) 另一支（如 \(|r_x|\gg 1\)）的风险。
- **物理解筛选**：若 PAL 终点明显偏离 L1 Halo 常见范围，则**回退**为欧拉预测 \(\mathbf{X}+\Delta S\,\dot{\mathbf{X}}\) 再微分修正。
- **MATLAB 内层用固定 `X` 算 `F`**：本库在 PAL 内层用当前迭代的 \(\mathbf{X}_{new}\) 计算 \(F\)，理论上更自洽；若需逐行复现 MATLAB 数值，需在单独分支中实现。

---

## 命令行脚本

| 脚本 | 说明 |
|------|------|
| `scripts/generate/generate_halo_orbit.py` | 单条 Halo，可调 `libration_point`、`amplitude_z`、`halo_class` |
| `scripts/generate/generate_halo_family.py` | 由种子出发调用 `halo_pseudo_arclength_continuation`，输出 `output/halo/*.json` |
| `scripts/plot/plot_halo_orbit.py` | 单轨或 JSON 中多条轨道绘图 |
| `scripts/plot/plot_halo_family.py` | 轨道族 JSON：2D/3D、Jacobi、稳定性等 |

公共常数（如 \(\mu\)）见 `scripts/utils/common.py`。

---

## 参考文献与对照

- Richardson, D. L. (1980). Analytic construction of periodic orbits about the collinear points. *Celestial Mechanics*.
- 本地对照实现：`CR3BP_MATLAB_Library` — `continuation_PAL_CR3BP.m`、`examples/FAMILY_L1Halo_North.m`。

---

## 另见

- [轨道生成指南](../guides/orbit-generation.md) — 教程入口
- [延拓模块总述](continuation.md) — `Continuation` 类索引
- [微分修正](differential_correction.md) — 对称性配置详解
