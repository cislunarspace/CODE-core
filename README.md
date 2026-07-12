# e2m2e — Earth to Moon, Moon to Earth

**地月空间转移轨道设计库**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/e2m2e)](https://pypi.org/project/e2m2e/)
[![CI](https://github.com/cislunarspace/e2m2e/actions/workflows/ci.yml/badge.svg)](https://github.com/cislunarspace/e2m2e/actions/workflows/ci.yml)

e2m2e 是一个用于设计地月空间运行轨道和转移轨道的 Python 库，基于[圆型限制性三体问题 (CR3BP)](https://en.wikipedia.org/wiki/Three-body_problem#Restricted_three-body_problem) 和[星历](https://naif.jpl.nasa.gov/naif/) N 体动力学建模。

> **[PyPI](https://pypi.org/project/e2m2e/)**

e2m2e 围绕“建模—生成—转移—检查”组织工作流。你可以先建立地月空间的动力学模型，再生成或修正一条轨道，必要时把它转换到更精确的星历模型，最后通过可视化检查设计结果。

核心能力覆盖：
- **建模**：CR3BP 系统、星历系统、力模型组合、坐标系与积分器族
- **生成**：周期轨道族、微分修正、多重打靶、延拓、稳定性分析
- **正规化**：Hamiltonian 正规化流水线，把平动点附近轨道化简为少数表征参数
- **转移**：网格搜索 + NLP 优化的转移轨道设计
- **检查**：2D/3D 轨道绘图、Jacobi 常数图、稳定性分析图

更详细的能力按模块列出：

**动力学建模**
- CR3BP 系统（地月、日地、日木等）、平动点与 Jacobi 常数
- 星历系统：基于 SPICE 内核的 N 体引力，可配置力模型
- 力模型组合（`ForceModel`）：聚合多个 `PhysicalModel`，通过 [Rust 积分器](#积分器族)自适应步长传播，支持按名注册、启用/禁用、序列化到 JSON。内置力模型如下：

  - **`PointMassGravity`** — 中心天体点质量引力。适用于参考系原点天体自身的二体引力：

    $$\mathbf{a} = -\frac{\mu}{|\mathbf{r}|^3} \, \mathbf{r}$$

    其中 $\mu$ 为引力参数（km³/s²），$\mathbf{r}$ 为航天器相对中心天体的位置。

  - **`ThirdBodyGravity`** — 参考系原点之外的天体引力摄动。一个实例对应一个摄动天体（如 `ThirdBodyGravity("MOON")`）。加速度由直接项与间接项合成：

    $$\mathbf{a} = -\mu_i \left[ \frac{\mathbf{r} - \mathbf{r}_i}{|\mathbf{r} - \mathbf{r}_i|^3} + \frac{\mathbf{r}_i}{|\mathbf{r}_i|^3} \right]$$

    其中 $\mathbf{r}$ 为航天器相对原点的位置，$\mathbf{r}_i$ 为摄动天体相对原点的位置（由 SPICE 查询）。间接项扣除摄动天体对原点的引力，保持坐标原点固定。

  - **`GravityField`** — 完全正规化球谐重力场（Cnm/Snm），用 Pines 递推计算非球形引力加速度。天体无关：地球（EGM96）、月球（GRGM900C）等共用同一个类，按 `body` 参数自动切换 body-fixed 轴与系数文件。位势展开为：

    $$U = \frac{\mu}{r} \sum_{n=0}^{N} \left(\frac{R}{r}\right)^n \sum_{m=0}^{n} \left(C_{nm}\cos m\lambda + S_{nm}\sin m\lambda\right) \bar{P}_{nm}(\sin\phi)$$

    加速度由 Pines 方法直接递推位势梯度得到，不经过球谐系数的解析微分。内置固体潮修正：地球支持 Step1（天体无关）+ Step2（频率相关）+ 极潮 + 永久潮；月球支持 k₂ = 0.024116 Love 数的固体潮。

  - **`DragModel`** — 大气阻力。在 ITRF（地固系）中计算密度与相对速度，自动完成参考系↔ITRF 坐标变换。大气在 ITRF 中静止，相对速度等于航天器 ITRF 速度。阻力加速度为：

    $$\mathbf{a}_{\text{drag}} = -\frac{1}{2} \, \rho \, \frac{C_d A}{m} \, |\mathbf{v}_{\text{rel}}| \, \mathbf{v}_{\text{rel}}$$

    其中 $\rho$ 为大气密度（由 `ExponentialAtmosphere` 提供，US Standard Atmosphere 1976 分段指数模型），$C_d$ 为阻力系数（默认 2.2），$A/m$ 为面积质量比。

  - **`SolarRadiationPressure`** — 太阳光压（cannonball 模型），基于 Montenbruck & Gill eq. 3.75：

    $$\mathbf{a}_{\text{SRP}} = f \cdot P \cdot \left(\frac{1\,\text{AU}}{r}\right)^2 \cdot \frac{C_r A}{m} \, \hat{\mathbf{u}}$$

    其中 $P = 4.56 \times 10^{-6}$ N/m² 为 1 AU 处太阳光压常数，$f \in [0,1]$ 为光照因子（由 `ConicalShadowModel` 给出：全光照=1，本影=0），$\hat{\mathbf{u}}$ 为太阳→航天器单位向量。阴影模型实现圆锥算法（本影/半影/环形食），多遮挡体合成遵循 GMAT 规范。

  - **`FiniteBurn`** — 连续推力加速度力模型。推力大小（`thrust_profile(t)` → 标量 N）与方向（`direction`）解耦，方向支持传播惯性系、VNB、LVLH 三种坐标系：

    $$\mathbf{a}_{\text{thrust}} = \frac{T(t)}{m} \, \hat{\mathbf{d}}$$

    其中 $T(t)$ 为标量推力函数，$\hat{\mathbf{d}}$ 为归一化方向向量。配置往返支持固定推力/脉冲剖面与固定方向的封闭 DSL；任意 Python callable 可传播但无法序列化。

  - **`ImpulsiveBurn`** — 瞬时 Δv 机动事件，在指定 epoch 处直接修改状态速度：$\mathbf{v} \leftarrow \mathbf{v} + \Delta\mathbf{v}$。

  - **`RelativisticCorrection`** — 后牛顿相对论修正，含三项，公式与 GMAT 对齐：

    - **Schwarzschild 项**（质量引起的时空弯曲）：

      $$\mathbf{a}_S = \frac{\gamma \mu}{c^2 r^3} \left[ \left(\frac{4\mu}{r} - v^2\right) \mathbf{r} + 4(\mathbf{r} \cdot \mathbf{v})\mathbf{v} \right]$$

    - **Lense-Thirring 项**（参考系拖曳）：

      $$\mathbf{a}_{LT} = \frac{2\mu}{c^2 r^3} \left[ \frac{3}{r^2}(\mathbf{r} \cdot \mathbf{J})(\mathbf{r} \times \mathbf{v}) + \mathbf{v} \times \mathbf{J} \right]$$

    - **de Sitter 项**（测地进动，geodetic precession）：

      $$\mathbf{a}_{dS} = 2 \, \boldsymbol{\omega} \times \mathbf{v}$$

    其中 $\gamma = 1.0$ 为后牛顿参数，$c$ 为光速，$\mathbf{J}$ 为天体角动量参数，$\boldsymbol{\omega}$ 为 de Sitter 进动角速度。
- 积分器族（[ADR-0002](docs/adr/0002-rust-integrator-core.md)）：RK（PD45/PD78/RK89）、Adams-Bashforth-Moulton、Cowell 8 阶，自适应与固定步长，底层由 Rust crate `e2m2e-integrators` 实现

**坐标系**
- J2000/ICRS/ITRF 层级转换、GAST/极移，标准轴/原点定义，与 [GMAT](https://opensource.gsfc.nasa.gov/projects/GMAT/) 兼容（[ADR-0003](docs/adr/0003-coordinate-itrf93-gmat-compatibility.md)）

**轨道生成与分析**
- 周期轨道族：DRO、ARO、RO、Halo、Lyapunov、Lissajous、Butterfly、Dragonfly
- 微分修正（对称 2D/3D/Halo）、多重打靶（含两级）、自然/伪弧长延拓、稳定性分析
- LEO/GEO 参考轨道快速创建

**转移设计**
- 转移轨道搜索与优化（网格搜索 + NLP），如 DRO-RO

**Hamiltonian 正规化**
- 一键式流水线 `NormalFormPipeline`：动力学替代 → quasi-Floquet 变换 → 中心流形化简 → 表征参数 `(q1, p1, I2, θ2, I3, θ3)`
- 把 CR3BP 平动点附近的复杂非线性动力学化简为少数几乎不变的参数，用于轨道识别与高保真外推。可选依赖 `pip install e2m2e[normal-form]`，示例见 [`examples/normal_form_example.py`](examples/normal_form_example.py)

**可视化**
- 2D/3D 轨道绘图、Jacobi 常数图、稳定性分析图

## 安装

```bash
pip install e2m2e
```

从源码安装（需要 [Rust 工具链](https://www.rust-lang.org/tools/install) 和 [uv](https://docs.astral.sh/uv/)）:

```bash
git clone https://github.com/cislunarspace/e2m2e.git
cd e2m2e
uv sync
```

开发依赖:

```bash
uv sync --group dev
```

### SPICE 内核

星历动力学需要 NASA SPICE 内核文件，放置在 `kernels/` 目录或 `$SPICE_KERNEL_DIR` 指定的路径。

常用内核: [`de440.bsp`](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/) (行星星历)、`moon_pa_de440_200625.bsp` (月球姿态)、[`pck00011.tpc`](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/) (行星常数)。

内核下载: [NASA NAIF](https://naif.jpl.nasa.gov/naif/data.html)

## 快速开始

### 创建 CR3BP 系统

```python
from e2m2e.core import CR3BP_System

system = CR3BP_System(mu=0.01215, primary="earth", secondary="moon")
system.compute_libration_points()
system.info()
```

### 星历动力学

```python
from e2m2e.core import EphemerisSystem, EphemerisDynamics, SPICEManager

spice = SPICEManager()
kernel = spice.find_ephemeris_kernel("./kernels/")
spice.load_kernel(kernel)

ephemeris_system = EphemerisSystem(
    bodies=["EARTH", "MOON", "SUN"],
    spice=spice,
    origin="EARTH",
    frame="J2000",
)
dynamics = EphemerisDynamics(system=ephemeris_system)
```

### 生成 DRO 轨道族

```python
from e2m2e.core import CR3BP_System, Orbit, CR3BP_Dynamics
from e2m2e.algorithms import DifferentialCorrection, Continuation

system = CR3BP_System(mu=0.01215, primary="earth", secondary="moon")
dynamics = CR3BP_Dynamics(system=system)

# 种子轨道
initial_state = [0.79188556619742, 0.0, 0.0, 0.0, 0.53682, 0.0]
seed_orbit = Orbit(states=[initial_state], times=[0])

# 微分修正
corrector = DifferentialCorrection(dynamic=dynamics)
corrector.setup_2D_symmetric_x_fixed_x0(x0=initial_state[0])
seed_dro = corrector.iterate_correction(initial_guess=seed_orbit)

# 延拓生成轨道族
continuation = Continuation(corrector=corrector)
family = continuation.natural_continuation(
    seed_orbit=seed_dro,
    param_range=(0.14, 0.9),
    step_size=0.005,
)
```

### 多重打靶法

```python
from e2m2e.algorithms import MultipleShooting, sample_patch_points

ms = MultipleShooting(dynamics=dynamics)
t_patch, state_patch = sample_patch_points(seed_dro, n_points=5)

result = ms.correct(
    t_patch=t_patch,
    state_patch=state_patch,
    max_iter=50,
    tolerance=1e-10,
    var_time=True,
)

if result.converged:
    print(f"收敛，最大残差 {result.max_residual:.2e}")
```

### 转移轨道设计

```python
from e2m2e.transfer import Transfer

transfer = Transfer(dynamics)
result = transfer.set_orbit(start=dro_orbit, end=ro_orbit).optimize(
    initial_guess={"alpha": 1.0, "transfer_time": 15.0, "t_ins": 5.0},
    alpha_range=(0.5, 2.5),
)
```

底层搜索 + NLP 两步法:

```python
from e2m2e.transfer import TransferSearch, DROTRONLPOptimizer, NLPOptimizationVariables

# 搜索
searcher = TransferSearch(dynamics=dynamics)
results = searcher.search(
    alpha_min=0.5, alpha_max=2.5,
    n_alpha=101, n_departure=200,
    max_transfer_time=200.0,
    intersection_threshold=0.05,
    min_distance_threshold=0.02,
    collision_earth_radius=6378.0 / 384400.0,
    collision_moon_radius=1737.0 / 384400.0,
    integration_dt=0.01,
    departure_orbit=dro_orbit, arrival_orbit=ro_orbit,
)

# NLP 优化
optimizer = DROTRONLPOptimizer(
    system=system, dynamics=dynamics,
    departure_orbit=dro_orbit, arrival_orbit=ro_orbit,
    departure_state=dro_orbit.states[0]
)
result = optimizer.optimize(
    initial_guess=NLPOptimizationVariables(alpha=1.0, transfer_time=5.0, t_ins=3.0),
)
```

### 可视化

```python
from e2m2e.visualization import PlotConfig, FamilyPlotter

config = PlotConfig(title=32, label=28)
config.apply_rcparams()

plotter = FamilyPlotter(system, config)
plotter.plot_family_2d(family, jacobi_values, title="DRO Family")
```

## 项目结构

```text
e2m2e/
├── core/                       # 系统、动力学、轨道、坐标系、星历
│   ├── system.py               # System 抽象基类
│   ├── cr3bp_system.py         # CR3BP_System - 系统定义、平动点
│   ├── dynamics.py             # CR3BP_Dynamics - 运动方程、STM
│   ├── orbit.py                # Orbit, OrbitFamily - 轨道数据结构
│   ├── coordinate/             # 坐标系子包（坐标轴、原点、坐标系、rho 桥接）
│   ├── forces/                 # 力模型子包（见上文力模型详细介绍）
│   ├── atmosphere/             # 大气密度模型（ExponentialAtmosphere）
│   ├── ephemeris_system.py     # EphemerisSystem - 星历系统
│   ├── ephemeris_dynamics.py   # EphemerisDynamics - N 体动力学
│   └── spice.py                # SPICE 内核管理
├── algorithms/                 # 微分修正、延拓、打靶、稳定性分析、Hamiltonian 正规化
│   ├── differential_correction.py  # 微分修正（对称 2D/3D/Halo）
│   ├── continuation.py         # 自然/伪弧长延拓
│   ├── multiple_shooting.py    # 多重打靶（含两级）
│   ├── strategies/             # 微分修正策略函数
│   ├── normal_form/            # 标准形化简流水线
│   └── ephemeris_correction/   # 星历修正分发器与实现
├── transfer/                   # 转移轨道搜索与优化
├── mbse/                       # 基于模型的系统工程（文档产物，非运行时）
├── visualization/              # 2D/3D 绘图
└── integrators.py              # Rust 积分器 Python 薄封装
crates/e2m2e-integrators/       # Rust 积分器内核（PD45/PD78/RK89/ABM/Cowell）
```

## 文档

**入门**
- [快速开始](docs/getting-started/quickstart.rst) — 从零到第一条轨道
- [安装指南](docs/getting-started/installation.rst) — pip / 源码 / SPICE 内核
- [可视化教程](docs/getting-started/visualization.rst) — 2D/3D 绘图

**核心模块**
- [系统与轨道](docs/core/system.rst) — `System`、`CR3BP_System`、平动点
- [轨道数据结构](docs/core/orbit.rst) — `Orbit`、`OrbitFamily`
- [动力学](docs/core/dynamics.rst) — `CR3BP_Dynamics`、传播与 STM
- [坐标系](docs/core/coordinate.rst) — 坐标轴、原点、坐标系定义与变换
- [力模型](docs/core/forces.rst) — `PhysicalModel` 子类与 `ForceModel` 容器
- [积分器](docs/core/integrators.rst) — Rust 积分器族（RK / Adams / Cowell）
- [星历系统](docs/core/ephemeris.rst) — `EphemerisSystem`、SPICE 集成
- [大气模型](docs/core/atmosphere.rst) — `ExponentialAtmosphere`
- [可视化](docs/core/visualization.rst) — `FamilyPlotter`、`TransferPlotter`

**算法**
- [微分修正](docs/algorithms/differential-correction.rst) — 对称 2D/3D/Halo 策略
- [微分修正策略](docs/algorithms/strategies.rst) — 策略函数与 `CorrectionConfig`
- [延拓](docs/algorithms/continuation.rst) — 自然延拓、伪弧长延拓
- [Halo 轨道](docs/algorithms/halo.rst) — Halo 初始猜测与微分修正
- [Halo 轨道族](docs/algorithms/halo-family.rst) — Halo 种子生成与族延拓编排
- [Halo 初始猜测](docs/algorithms/halo-initial-guess.rst) — Richardson 三阶解析近似
- [多重打靶](docs/algorithms/multiple-shooting.rst) — 标准多重打靶与星历修正
- [两层多重打靶](docs/algorithms/two-level-multiple-shooting.rst) — 交替求解位置/速度连续
- [稳定性分析](docs/algorithms/stability.rst) — 单值矩阵特征值谱
- [标准形](docs/algorithms/normal-form.rst) — Hamiltonian 正规化流水线
- [同伦修正](docs/algorithms/homotopy-correction.rst) — 同伦延拓修正

**转移设计**
- [转移设计总览](docs/transfer/overview.rst) — 搜索-优化两步法
- [网格搜索](docs/transfer/search.rst) — alpha-t_departure 网格搜索
- [NLP 优化](docs/transfer/optimization.rst) — DRO-RO 非线性规划
- [端点条件](docs/transfer/terminal.rst) — `OrbitTerminal`、`StateTerminal`
- [脉冲推进](docs/transfer/propulsion.rst) — `ImpulsivePropulsion`、Δv 分解

**参考**
- [术语表](docs/reference/glossary.rst)
- [CONTEXT.md](CONTEXT.md) — 领域术语与关键约定
- [MBSE 模型](docs/reference/mbse/index.md) — SysML 组件、需求追溯、状态机、活动图

**架构决策记录**
- [ADR-0001](docs/adr/0001-protocol-seams.md) — 撤回 Protocol 接缝
- [ADR-0002](docs/adr/0002-rust-integrator-core.md) — Rust 积分器内核
- [ADR-0003](docs/adr/0003-coordinate-itrf93-gmat-compatibility.md) — ITRF93 坐标系与 GMAT 兼容
- [ADR-0004](docs/adr/0004-forcemodel-config-driven-construction.md) — ForceModel 配置驱动
- [ADR-0005](docs/adr/0005-two-level-multiple-shooting-dedicated.md) — 两级多重打靶
- [ADR-0006](docs/adr/0006-ephemeris-corrector-seam.md) — 星历修正接缝
- [ADR-0007](docs/adr/0007-dynamic-axes-state-injection.md) — 动态坐标轴状态注入

## 测试

```bash
uv run pytest tests/
```

## 代码规范

```bash
uv run ruff check .          # 检查
uv run ruff check --fix .    # 自动修复
uv run ruff format .         # 格式化
```

## 贡献

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

## 更新日志

见 [CHANGELOG.md](CHANGELOG.md)。

## 引用

```bibtex
@software{e2m2e,
  title = {e2m2e: Earth to Moon, Moon to Earth Transfer Orbit Design Library},
  author = {ouyangjiahong},
  email = {ouyangjiahong22@nudt.edu.cn},
  url = {https://github.com/cislunarspace/e2m2e},
  version = {5.2.0},
  year = {2026},
}
```
