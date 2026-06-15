# e2m2e — Earth to Moon, Moon to Earth

**地月空间转移轨道设计库**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/e2m2e)](https://pypi.org/project/e2m2e/)
[![CI](https://github.com/cislunarspace/e2m2e/actions/workflows/ci.yml/badge.svg)](https://github.com/cislunarspace/e2m2e/actions/workflows/ci.yml)

e2m2e 是一个用于设计地月空间运行轨道和转移轨道的 Python 库，基于圆型限制性三体问题 (CR3BP) 和星历 N 体动力学建模。

> **[PyPI](https://pypi.org/project/e2m2e/)**

## 功能

- **CR3BP 系统建模**：地月、日地、日木等天体系统，拉格朗日点计算，Jacobi 常数
- **星历动力学**：基于 SPICE 内核的 N 体引力计算，支持多天体摄动与可配置力模型
- **积分器族**：RK（PD45/PD78/RK89）、Adams-Bashforth-Moulton、Cowell 8 阶，支持自适应与固定步长
- **力模型容器**：配置驱动 ForceModel，按名注册、启用/禁用、序列化到 JSON
- **大气阻力**：指数大气模型 + 阻力系数模型，适用于 LEO 轨道衰减分析
- **太阳辐射压 (SRP)**：cannonball SRP + 圆锥地影/月影，支持多遮挡体光照份额合成
- **推力模型**：脉冲推力与有限推力，固定/速度/轨道根数方向配置
- **坐标系与 GMAT 兼容**：J2000/ICRS/ITRF 层级转换、GAST/极移、标准轴/原点定义
- **LEO/GEO 参考轨道**：快速创建近地/地球同步轨道，支持与 GMAT 参考轨道对比
- **周期轨道**：DRO、ARO、RO、Halo、Lyapunov、Lissajous、Butterfly、Dragonfly
- **设计算法**：微分修正（对称 2D/3D/Halo 策略）、多重打靶法、自然延拓、伪弧长延拓、稳定性分析
- **转移轨道**：DRO-RO 转移搜索 (网格搜索 + NLP 优化)
- **可视化**：2D/3D 轨道绘图、Jacobi 常数图、稳定性分析图

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

system = CR3BP_System.from_known_system("earth_moon")
system.compute_libration_points()
system.info()
```

### 星历动力学

```python
from e2m2e.core import EphemerisSystem, EphemerisDynamics, SPICEManager

spice = SPICEManager()
spice.load_kernels_from_directory("./kernels/")

ephemeris_system = EphemerisSystem(
    bodies=["EARTH", "MOON", "SUN"],
    reference_epoch="2025-06-21T11:00:06"
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
from e2m2e.algorithms import MultipleShooting, sample_patch_points, convert_to_j2000

ms = MultipleShooting(dynamics=dynamics)
t_patch, state_patch = sample_patch_points(orbit=seed_dro, n_segments=5)

result = ms.correct(
    t_patch=t_patch,
    state_patch=state_patch,
    max_iter=50,
    tol=1e-10,
    var_time=True
)

if result.converged:
    state_j2000 = convert_to_j2000(result.state_patch, system)
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
│   ├── system.py         # CR3BP_System - 系统定义、平动点
│   ├── dynamics.py       # CR3BP_Dynamics - 运动方程、STM
│   ├── orbit.py          # Orbit, OrbitFamily - 轨道数据结构
│   ├── coordinate.py     # 坐标变换
│   ├── ephemeris_system.py      # EphemerisSystem - 星历系统
│   ├── ephemeris_dynamics.py    # EphemerisDynamics - N 体动力学
│   └── spice.py                 # SPICE 内核管理
├── algorithms/           # 微分修正、延拓、打靶、稳定性分析
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

### v4.2.0

- **两级多重打靶法** — `TwoLevelMultipleShooting` 求解器，支持分层约束结构
- **Halo 轨道族生成** — `generate_halo_family` 新增 `z_range` 参数，支持按 z 轴范围筛选
- **星历修正分发** — `ephemeris_correction_dispatch` 自动选择修正策略
- **3D 天体图标** — 地球/月球 PNG Billboard 渲染，动态深度排序
- **环境变量配置** — `PlotConfig.from_env()` 从环境变量加载绘图配置
- **迭代回调** — `iterate_correction` 支持 `callback` 参数，实时监控收敛过程
- **统一 delta-v 计算** — `compute_transfer_cost` 提取为独立接口
- **搜索首次可行解计时** — `TransferSearch` 结果记录首次可行解时间
- **完整 docstring 审计** — 覆盖 algorithms、core、transfer、visualization 模块

### v4.1.0

- 转移轨道搜索与 NLP 优化两步法
- SRP 动力学建模
- 稳定性分析模块
- MBSE 需求追踪

## 引用

```bibtex
@software{e2m2e,
  title = {e2m2e: Earth to Moon, Moon to Earth Transfer Orbit Design Library},
  author = {ouyangjiahong},
  email = {ouyangjiahong22@nudt.edu.cn},
  url = {https://github.com/cislunarspace/e2m2e},
  version = {4.2.0},
  year = {2026},
}
```
