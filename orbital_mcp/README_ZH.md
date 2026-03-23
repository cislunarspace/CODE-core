# Orbital MCP Server

基于 e2m2e 库构建的轨道力学计算模型上下文协议 (Model Context Protocol) 服务器。

## 功能特性

- **CR3BP 动力学**: 在圆形限制性三体问题中传播轨迹
- **轨道分析**: 计算周期、振幅、稳定指数和单值矩阵
- **状态传播**: 具有可选状态转移矩阵 (STM) 计算的前向积分
- **Jacobi 常数**: 计算基于能量的轨道分类的 Jacobi 常数
- **截面检测**: 检测 Poincaré 截面穿越

## 安装

```bash
# 安装依赖
uv sync

# 以开发模式运行
uv run mcp dev orbital_mcp/server.py

# 安装到 Claude Desktop
uv run mcp install orbital_mcp/server.py
```

## 可用工具

| 工具 | 描述 |
|------|------|
| `propagate_trajectory` | 在 CR3BP 中传播轨迹 |
| `compute_stm` | 计算两个时刻之间的状态转移矩阵 |
| `compute_jacobi` | 计算给定状态的 Jacobi 常数 |
| `get_orbit_period` | 计算轨道周期 |
| `get_orbit_amplitude` | 计算 x、y 或 z 方向的振幅 |
| `check_crossing` | 检查轨迹是否穿越 Poincaré 截面 |

## 配置

服务器默认使用 stdio 传输方式进行本地操作。

## 许可证

MIT
