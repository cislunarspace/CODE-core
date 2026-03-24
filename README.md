# e2m2e — Earth to Moon, Moon to Earth

**地月空间转移轨道设计库**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

`e2m2e` 是一个设计地月空间**运行轨道**和**转移轨道**的Python库。该库使用了面向对象编程，提供了模块化的设计。

## 安装

### 从源码安装

```bash
# 克隆仓库
git clone https://gitee.com/cislunarspace/e2m2e.git
cd e2m2e

# 安装依赖
python -m pip install -e .
```

## 快速开始

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

## 可视化功能

`e2m2e` 提供了强大的轨道可视化功能：

```python
from e2m2e.visualization.plotting import OrbitVisualizer

# 创建可视化器
viz = OrbitVisualizer(system)

# 绘制2D投影（假设orbit是轨道数据）
viz.plot_2d_projection(orbit, plane='xy', color='blue', label='My Orbit')
viz.plot_primary_bodies()      # 添加天体
viz.plot_libration_points()    # 添加平动点
viz.show()                     # 显示图形

# 创建综合概览图
viz.create_overview_plot(orbit)
viz.show()
```

更多可视化功能和使用示例，请参考 [可视化模块使用指南](docs/guides/visualization-guide.md)。

## 开发与贡献

### 运行测试

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/
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

## 🙏 致谢

- 感谢所有三体问题研究者的开创性工作
- 感谢开源社区提供的优秀工具和库