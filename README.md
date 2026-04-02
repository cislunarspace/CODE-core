# e2m2e — Earth to Moon, Moon to Earth

**地月空间转移轨道设计库**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Alpha-orange)](https://www.python.org/)

`e2m2e` 是一个用于设计地月空间**运行轨道**和**转移轨道**的 Python 库，基于圆型限制性三体问题 (CR3BP) 的轨道动力学建模。

## 核心功能

- **CR3BP 系统建模**：支持地月、日地、日木等常见天体系统
- **多种轨道类型**：DRO、ARO、RO、Halo、Lyapunov、Lissajous、Butterfly 等
- **轨道设计算法**：微分修正、自然延拓、伪弧长延拓、稳定性分析
- **转移轨道搜索**：网格搜索、NLP 优化、脉冲转移设计
- **可视化工具**：2D/3D 轨道绘图、Jacobi 常数图、稳定性分析图

## 支持的轨道类型

| 轨道类型 | 描述 |
|---------|------|
| **DRO** | 远距离逆行轨道 (Distant Retrograde Orbit) |
| **RO** | 共振轨道 (Resonant Orbit)，支持 3:2、4:3 等多种共振 |
| **ARO** | 轴向共振轨道 (Axial Resonant Orbit) |
| **Halo** | Halo 轨道，周期轨道的一种 |
| **Lyapunov** | Lyapunov 轨道，平面周期轨道 |
| **Lissajous** | Lissajous 轨道，拟周期轨道 |
| **Butterfly** | Butterfly 轨道，关于 xy 面对称 |
| **Dragonfly** | Dragonfly 轨道，多重对称性 |

## 安装

### 从源码安装

```bash
git clone https://github.com/cislunarspace/e2m2e.git
cd e2m2e
python -m pip install -e .
```

### 开发依赖

```bash
pip install -e ".[dev]"
```

## 快速开始

### 1. 创建系统并计算平动点

```python
import e2m2e
from e2m2e.core import CR3BP_System

# 创建地月系统
system = CR3BP_System.from_known_system("earth_moon")
system.set_characteristic_scales(distance=384400, period=27.32 * 86400)
system.compute_libration_points()
system.info()
```

### 2. 生成 DRO 轨道族

```python
import e2m2e
from e2m2e.core import CR3BP_System, Orbit
from e2m2e.algorithms import DifferentialCorrection, Continuation

# 初始化
system = CR3BP_System(mu=0.01215, primary="earth", secondary="moon")
dynamics = e2m2e.core.dynamics.CR3BP_Dynamics(system=system)

# 种子轨道
x0 = 0.79188556619742
vy0 = 0.53682
initial_state = [x0, 0.0, 0.0, 0.0, vy0, 0.0]
seed_orbit = Orbit(states=[initial_state], times=[0])
seed_orbit.period = 3.472526005624708

# 微分修正
corrector = DifferentialCorrection(dynamic=dynamics)
corrector.setup_2D_symmetric_x_fixed_x0(x0=x0)
seed_dro = corrector.iterate_correction(initial_guess=seed_orbit)

# 自然延拓生成轨道族
continuation = Continuation(corrector=corrector)
family = continuation.natural_continuation(
    seed_orbit=seed_dro,
    param_range=(0.14, 0.9),
    step_size=0.005,
)
```

### 3. 转移轨道设计

```python
from e2m2e.transfer import Transfer

# 简化链式 API
transfer = Transfer(dynamics)
result = transfer.set_orbit(start=dro_orbit, end=ro_orbit).optimize(
    initial_guess={"alpha": 1.0, "transfer_time": 15.0, "t_ins": 5.0},
    alpha_range=(0.5, 2.5),
)
```

或使用底层搜索 + NLP 两步法：

```python
from e2m2e.transfer import TransferSearch, DROTRONLPOptimizer, NLPOptimizationVariables

# 网格搜索
searcher = TransferSearch(dynamics=dynamics)
results = searcher.search(
    alpha_min=0.5, alpha_max=2.5,
    n_alpha=101, n_departure=200,
    max_transfer_time=200.0,
    departure_orbit=dro_orbit, arrival_orbit=ro_orbit,
)

# NLP 优化
optimizer = DROTRONLPOptimizer(system=system, dynamics=dynamics,
                                departure_orbit=dro_orbit, arrival_orbit=ro_orbit,
                                departure_state=dro_orbit.states[0])
result = optimizer.optimize(
    initial_guess=NLPOptimizationVariables(alpha=1.0, transfer_time=5.0, t_ins=3.0),
)
```

### 4. 可视化

```python
from e2m2e.visualization import PlotConfig, FamilyPlotter

config = PlotConfig(title=32, label=28)
config.apply_rcparams()

plotter = FamilyPlotter(system, config)
plotter.plot_family_2d(family, jacobi_values, title="DRO Family")
plotter.plot_jacobi_period_stability(jacobi_values, periods, stability_values)
```

## 项目结构

```
e2m2e/
├── core/                 # 核心模块
│   ├── system.py         # CR3BP_System - 系统定义、平动点、Jacobi 常数
│   ├── dynamics.py       # CR3BP_Dynamics - 运动方程、STM、数值积分
│   ├── orbit.py          # Orbit, OrbitFamily - 轨道数据结构与序列化
│   └── coordinate.py     # CoordinateTransformation - 坐标变换
├── algorithms/           # 算法模块
│   ├── differential_correction.py  # DifferentialCorrection - 微分修正
│   ├── continuation.py             # Continuation - 自然/伪弧长延拓
│   └── stability.py                # StabilityAnalysis - 稳定性分析
├── transfer/             # 转移轨道设计
│   ├── transfer.py                 # Transfer - 简化链式 API
│   ├── transfer_search.py          # TransferSearch - 网格搜索（并行）
│   └── transfer_optimization.py    # DROTRONLPOptimizer - NLP 优化
└── visualization/        # 可视化
    ├── config.py                    # PlotConfig - 字体/颜色/尺寸等全局配置
    ├── base.py                      # OrbitVisualizer - 2D/3D 绘图基类
    ├── family.py                    # FamilyPlotter - 轨道族可视化（高层 API）
    ├── transfer.py                  # TransferPlotter - 转移轨道可视化
    └── stability.py                 # compute_stability_for_family - 并行稳定性计算
```

## 算法介绍

### 微分修正 (Differential Correction)

通过迭代修正轨道初始状态，使轨道满足周期性边界条件：

- 2D X 对称固定 x0
- 3D XZ 对称固定 z0
- 垂直轨道修正

### 轨道延拓 (Continuation)

从种子轨道出发，参数化延拓生成完整轨道族：

- **自然延拓**：逐步改变参数值
- **伪弧长延拓**：跨越分岔点

### 稳定性分析 (Stability Analysis)

计算 Floquet 乘子，分析轨道稳定性：

- 特征值计算
- 分岔点检测
- 稳定性指标

## 开发与贡献

### 运行测试

```bash
pytest tests/
```

### 代码规范

本项目使用 [Ruff](https://github.com/astral-sh/ruff) 进行代码格式化：

```bash
ruff check .          # 检查
ruff check --fix .    # 自动修复
ruff format .         # 格式化
```

### 提交贡献

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

## 文档

更多文档请参考 [`docs/`](docs/) 目录：

- [系统架构](docs/guides/system-overview.md)
- [轨道生成教程](docs/guides/orbit-generation.md)
- [可视化指南](docs/guides/visualization-guide.md)
- [发布指南](docs/guides/release.md)

## 致谢

- 感谢所有三体问题研究者的开创性工作
- 感谢开源社区提供的优秀工具和库

## 引用

如果您在学术工作中使用了 e2m2e，请引用：

```bibtex
@software{e2m2e,
  title = {e2m2e: Earth to Moon, Moon to Earth Transfer Orbit Design Library},
  author = {ouyangjiahong},
  email = {ouyangjiahong22@nudt.edu.cn},
  url = {https://github.com/cislunarspace/e2m2e},
  version = {3.1.11},
  year = {2026},
}
```
