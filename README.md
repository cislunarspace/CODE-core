# e2m2e — Earth to Moon, Moon to Earth

**地月空间转移轨道设计库**

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/e2m2e.svg)](https://pypi.org/project/e2m2e/)

`e2m2e` 是一个基于圆型限制性三体问题（CR3BP）的Python库，专注于设计和分析地月空间转移轨道。该库提供了完整的工具链，用于设计、分析和可视化地月空间中的低能转移轨道。

## ✨ 主要特性

- **完整的CR3BP实现** - 支持多种天体系统（地月、日地、日木等）
- **平动点轨道设计** - Halo、Lyapunov、Vertical等轨道类型
- **微分修正算法** - 多种对称性配置支持
- **轨道族延拓** - 自然参数和伪弧长延拓方法
- **转移轨道设计** - 地球到月球、月球到地球的低能转移
- **可视化工具** - 3D/2D轨道绘制，稳定性分析图
- **模块化设计** - 易于扩展和集成到现有工作流

## 📦 安装

### 从PyPI安装（推荐）

```bash
pip install e2m2e
```

### 从源码安装（开发版）

```bash
# 克隆仓库
git clone https://github.com/yourusername/e2m2e.git
cd e2m2e

# 安装依赖
pip install -e .
```

### 依赖要求
- Python 3.10+
- NumPy >= 1.24
- SciPy >= 1.10
- Matplotlib >= 3.7

## 🚀 快速开始

### 基本使用示例

```python
import e2m2e
from e2m2e.core.system import CR3BP_System

# 1. 创建地月系统
system = CR3BP_System.from_known_system("earth_moon")
system.set_characteristic_scales(distance=384400, period=27.32 * 86400)
system.compute_libration_points()

# 使用 info() 方法查看系统信息
system.info()

# 2. 获取平动点信息
print(f"L1点位置: {system.L1}")
print(f"L2点位置: {system.L2}")

# 3. 计算Jacobi常数
state = [0.8, 0.1, 0.0, 0.0, 0.2, 0.0]
jacobi_constant = system.get_jacobi_constant(state)
print(f"Jacobi常数: {jacobi_constant:.4f}")
```

### 完整示例：设计平动点轨道

```python
import numpy as np
import e2m2e
from e2m2e.core.system import CR3BP_System
from e2m2e.core.dynamics import CR3BP_Dynamics
from e2m2e.algorithms.differential_correction import DifferentialCorrection

# 1. 初始化系统
system = CR3BP_System.from_known_system("earth_moon")
system.set_characteristic_scales(384400, 27.32 * 86400)
system.compute_libration_points()

# 2. 创建动力学对象
dynamics = CR3BP_Dynamics(system)

# 3. 设置微分修正器
dc = DifferentialCorrection(dynamics)
dc.setup_2D_symmetric_x_fixed_x0(x0=system.L1[0] + 0.01)

# 4. 设计Lyapunov轨道
initial_guess = [system.L1[0] + 0.01, 0, 0, 0, 0.1, 0]
orbit, result = dc.correct_orbit(initial_guess, t_half=1.5)

if orbit is not None:
    print(f"成功设计Lyapunov轨道!")
    print(f"轨道周期: {orbit.period:.4f} 无量纲时间")
    print(f"收敛迭代次数: {result['iterations']}")
    
    # 5. 可视化轨道
    from e2m2e.visualization.plotting import OrbitVisualizer
    viz = OrbitVisualizer(system)
    viz.create_overview_plot(orbit)
    viz.show()
```

## 📚 文档

### 核心模块

#### `e2m2e.core.system`
- `CR3BP_System` - CR3BP系统参数管理
  - `from_known_system()` - 从已知系统创建
  - `compute_libration_points()` - 计算平动点
  - `set_characteristic_scales()` - 设置特征尺度
  - `info()` - 输出系统信息（新增功能）
  - `get_jacobi_constant()` - 计算Jacobi常数

#### `e2m2e.core.dynamics`
- `CR3BP_Dynamics` - CR3BP动力学方程

#### `e2m2e.core.orbit`
- `Orbit` - 轨道数据结构

### 算法模块

#### `e2m2e.algorithms.differential_correction`
- `DifferentialCorrection` - 微分修正算法

#### `e2m2e.algorithms.continuation`
- `Continuation` - 轨道族延拓算法

#### `e2m2e.algorithms.stability`
- 稳定性分析工具

### 转移轨道模块

#### `e2m2e.transfer.earth_moon`
- `EarthMoonTransfer` - 地球到月球转移

#### `e2m2e.transfer.moon_earth`
- `MoonEarthTransfer` - 月球到地球转移

#### `e2m2e.transfer.inter_orbit`
- 轨道间转移

### 可视化模块

#### `e2m2e.visualization.plotting`
- `OrbitVisualizer` - 轨道可视化工具

## 🏗️ 项目结构

```
e2m2e/
├── __init__.py              # 包主入口
├── core/                    # 核心模块
│   ├── __init__.py
│   ├── system.py            # CR3BP系统定义和参数管理
│   ├── dynamics.py          # 动力学方程
│   ├── orbit.py             # 轨道数据结构
│   └── coordinate.py        # 坐标转换
├── algorithms/              # 算法模块
│   ├── __init__.py
│   ├── differential_correction.py  # 微分修正
│   ├── continuation.py      # 轨道族延拓
│   └── stability.py         # 稳定性分析
├── transfer/                # 转移轨道模块
│   ├── __init__.py
│   ├── earth_moon.py        # 地球到月球转移
│   ├── moon_earth.py        # 月球到地球转移
│   └── inter_orbit.py       # 轨道间转移
├── visualization/           # 可视化模块
│   ├── __init__.py
│   └── plotting.py          # 绘图工具
└── tests/                   # 测试文件
    └── test_basic.py        # 基础测试
```

## 🔧 开发与贡献

### 运行测试

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/

# 运行测试并生成覆盖率报告
pytest tests/ --cov=e2m2e --cov-report=html
```

### 代码规范

本项目使用 [Ruff](https://github.com/astral-sh/ruff) 进行代码格式化：

```bash
# 检查代码格式
ruff check .

# 自动修复代码格式
ruff check --fix .

# 格式化代码
ruff format .
```

### 提交贡献

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

## 📖 理论背景

本库基于**圆型限制性三体问题（CR3BP）**框架，在旋转坐标系中建立运动方程：

$$
\begin{aligned}
\ddot{x} - 2\dot{y} &= \frac{\partial \Omega}{\partial x} \\
\ddot{y} + 2\dot{x} &= \frac{\partial \Omega}{\partial y} \\
\ddot{z} &= \frac{\partial \Omega}{\partial z}
\end{aligned}
$$

其中有效势函数为：

$$
\Omega(x,y,z) = \frac{1}{2}(x^2 + y^2) + \frac{1-\mu}{r_1} + \frac{\mu}{r_2}
$$

Jacobi积分为：

$$
C = 2\Omega - (\dot{x}^2 + \dot{y}^2 + \dot{z}^2)
$$

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- 感谢所有三体问题研究者的开创性工作
- 感谢开源社区提供的优秀工具和库
- 特别感谢 [CR3BP](https://en.wikipedia.org/wiki/Circular_restricted_three-body_problem) 理论的发展者

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- GitHub Issues: [https://github.com/yourusername/e2m2e/issues](https://github.com/yourusername/e2m2e/issues)
- 邮箱: your.email@example.com

---

**e2m2e** - 让地月空间轨道设计变得更简单！ 🚀

$$\ddot{z} = \frac{\partial \Omega}{\partial z}$$

其中等效势能为：

$$\Omega = \frac{1}{2}(x^2+y^2) + \frac{1-\mu}{r_1} + \frac{\mu}{r_2}$$

## 依赖

- Python ≥ 3.10
- NumPy ≥ 1.24
- SciPy ≥ 1.10
- Matplotlib ≥ 3.7

## 许可证

MIT License