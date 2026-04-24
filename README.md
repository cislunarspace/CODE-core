# e2m2e — Earth to Moon, Moon to Earth

**地月空间转移轨道设计库**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/e2m2e)](https://pypi.org/project/e2m2e/)
[![CI](https://github.com/cislunarspace/e2m2e/actions/workflows/ci.yml/badge.svg)](https://github.com/cislunarspace/e2m2e/actions/workflows/ci.yml)
[![Docs](https://github.com/cislunarspace/e2m2e/actions/workflows/deploy-docs.yml/badge.svg)](https://cislunarspace.github.io/e2m2e/)

e2m2e 是一个用于设计地月空间运行轨道和转移轨道的 Python 库，基于圆型限制性三体问题 (CR3BP) 和星历 N 体动力学建模。

> **[在线文档](https://cislunarspace.github.io/e2m2e/)** | **[PyPI](https://pypi.org/project/e2m2e/)**

## 功能

- **CR3BP 系统建模**：地月、日地、日木等天体系统，拉格朗日点计算，Jacobi 常数
- **星历动力学**：基于 SPICE 内核的 N 体引力计算，支持多天体摄动
- **太阳辐射压 (SRP)**：CR3BP_SRP_Dynamics 子类，支持光学系数参数化
- **周期轨道**：DRO、ARO、RO、Halo、Lyapunov、Lissajous、Butterfly、Dragonfly
- **设计算法**：微分修正、多重打靶法、自然延拓、伪弧长延拓、稳定性分析
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
pip install -e .
```

开发依赖:

```bash
pip install -e ".[dev]"
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

```
e2m2e/
├── core/                 # 系统、动力学、轨道、坐标系、星历
│   ├── system.py         # CR3BP_System - 系统定义、平动点
│   ├── dynamics.py       # CR3BP_Dynamics - 运动方程、STM
│   ├── srp_dynamics.py   # CR3BP_SRP_Dynamics - 太阳辐射压扰动
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

## 测试

```bash
pytest tests/
```

## 代码规范

```bash
ruff check .          # 检查
ruff check --fix .    # 自动修复
ruff format .         # 格式化
```

## 文档

文档使用 Sphinx 构建，源文件位于 `docs/` 目录。

```bash
pip install -e ".[docs]"
cd docs && make html
# 用浏览器打开 docs/_build/html/index.html
```

在线文档: https://cislunarspace.github.io/e2m2e/

### 文档结构

```
docs/
├── conf.py              # Sphinx 配置
├── index.rst            # 首页与导航
├── getting-started/     # 安装、快速入门、可视化
├── core/                # 系统、动力学、星历、轨道、坐标系
├── algorithms/          # 微分修正、延拓法、稳定性
├── transfer/            # 转移轨道搜索与优化
├── api/                 # 从 docstring 自动生成的 API 文档
├── reference/           # 算法参考、术语表、MBSE 文档
└── _static/             # 图片等静态资源
```

API 文档通过 `sphinx.ext.autodoc` 从代码中的 docstring 自动提取。

## 贡献

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

## 引用

```bibtex
@software{e2m2e,
  title = {e2m2e: Earth to Moon, Moon to Earth Transfer Orbit Design Library},
  author = {ouyangjiahong},
  email = {ouyangjiahong22@nudt.edu.cn},
  url = {https://github.com/cislunarspace/e2m2e},
  version = {4.1.0},
  year = {2026},
}
```
