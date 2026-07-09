# e2m2e — Earth to Moon, Moon to Earth

**地月空间转移轨道设计库**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/e2m2e)](https://pypi.org/project/e2m2e/)
[![CI](https://github.com/cislunarspace/e2m2e/actions/workflows/ci.yml/badge.svg)](https://github.com/cislunarspace/e2m2e/actions/workflows/ci.yml)

e2m2e 是一个用于设计地月空间运行轨道和转移轨道的 Python 库，基于圆型限制性三体问题 (CR3BP) 和星历 N 体动力学建模。

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
- 力模型组合：重力场、大气阻力、太阳光压（cannonball + 圆锥地影/月影）、脉冲/有限推力；按名注册、启用/禁用、序列化到 JSON
- 积分器族：RK（PD45/PD78/RK89）、Adams-Bashforth-Moulton、Cowell 8 阶，自适应与固定步长

**坐标系**
- J2000/ICRS/ITRF 层级转换、GAST/极移，标准轴/原点定义，与 GMAT 兼容

**轨道生成与分析**
- 周期轨道族：DRO、ARO、RO、Halo、Lyapunov、Lissajous、Butterfly、Dragonfly
- 微分修正（对称 2D/3D/Halo）、多重打靶（含两级）、自然/伪弧长延拓、稳定性分析
- LEO/GEO 参考轨道快速创建

**转移设计**
- 转移轨道搜索与优化（网格搜索 + NLP），如 DRO-RO

**Hamiltonian 正规化**
- 一键式流水线 `NormalFormPipeline`：动力学替代 → quasi-Floquet 变换 → 中心流形化简 → 表征参数 `(q1, p1, I2, θ2, I3, θ3)`
- 把 CR3BP 平动点附近的复杂非线性动力学化简为少数几乎不变的参数，用于轨道识别与高保真外推。可选依赖 `pip install e2m2e[normal-form]`，示例见 `examples/normal_form_example.py`

**可视化**
- 2D/3D 轨道绘图、Jacobi 常数图、稳定性分析图

## 安装

```bash
pip install e2m2e
```

从源码安装:

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

常用内核: `de440.bsp` (行星星历)、`moon_pa_de440_200625.bsp` (月球姿态)、`pck00011.tpc` (行星常数)。

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
├── core/                 # 系统、动力学、轨道、坐标系、星历
│   ├── system.py         # System 抽象基类
│   ├── cr3bp_system.py   # CR3BP_System - 系统定义、平动点
│   ├── dynamics.py       # CR3BP_Dynamics - 运动方程、STM
│   ├── orbit.py          # Orbit, OrbitFamily - 轨道数据结构
│   ├── coordinate_system.py  # CoordinateSystem - 坐标系定义
│   ├── synodic_j2000.py  # SynodicJ2000System - synodic ↔ J2000 转换器
│   ├── ephemeris_system.py  # EphemerisSystem - 星历系统
│   ├── ephemeris_dynamics.py # EphemerisDynamics - N 体动力学
│   └── spice.py          # SPICE 内核管理
├── algorithms/           # 微分修正、延拓、打靶、稳定性分析、Hamiltonian 正规化
├── transfer/             # 转移轨道搜索与优化
├── mbse/                 # 基于模型的系统工程
└── visualization/        # 2D/3D 绘图
```

## 文档

- [快速开始与安装](docs/getting-started/quickstart.rst)
- [API 参考](docs/api/e2m2e.rst)
- [MBSE 模型](docs/reference/mbse/index.md) — 组件登记、需求追溯与图表生成；多态接缝以 `Dynamics` 基类为准，见 [ADR-0001](docs/adr/0001-protocol-seams.md)

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
