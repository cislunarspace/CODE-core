# e2m2e: Earth to Moon, Moon to Earth

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/e2m2e)](https://pypi.org/project/e2m2e/)
[![CI](https://github.com/cislunarspace/e2m2e/actions/workflows/ci.yml/badge.svg)](https://github.com/cislunarspace/e2m2e/actions/workflows/ci.yml)
[![GitHub stars](https://img.shields.io/github/stars/cislunarspace/e2m2e.svg)](https://github.com/cislunarspace/e2m2e/stargazers)
[![Rust: 1.98.0](https://img.shields.io/badge/rust-1.98.0-orange.svg)](https://www.rust-lang.org/)

e2m2e 是地月空间**算法工具集**。

## 安装

用 [uv](https://docs.astral.sh/uv/) 安装：

```bash
uv pip install e2m2e
```

从源码开发：

```bash
git clone https://github.com/cislunarspace/e2m2e.git
cd e2m2e
make dev
```

<details>
  <summary>Windows 安装 make</summary>
  Windows 默认不提供 `make`。可使用 [Scoop](https://scoop.sh/) 安装；在 PowerShell 中依次执行：

  ```powershell
  Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
  irm get.scoop.sh | iex
  scoop install make
  ```
</details>

e2m2e 所需的全部星历数据已打包在 [GitHub Release](https://github.com/cislunarspace/e2m2e/releases) 的 `kernels-v1` 中，`make dev` 会自动下载到 `kernels/`；也可手动下载解压到该目录。

## 快速开始

设计一条地月 L2 Halo 轨道：

```python
from e2m2e.api import Facade

facade = Facade()

result = facade.design_orbit(
    orbit_type="Halo",
    collinear_point=2,
    amplitude=30000.0,
    epoch=[2024, 1, 1, 0, 0, 0.0],
    duration=365.25 * 86400.0,
)

print(result.orbit_type)
print(result.initial_state)
```

## 功能介绍

### MCP

e2m2e 为 18 个工具设计了 MCP 接口。

安装 MCP：

```bash
uv pip install "e2m2e[mcp]"
```

在 MCP 客户端配置中注册服务器（`command` 指向安装了 e2m2e 的环境里的可执行文件，多数 MCP 客户端都采用这一 `mcpServers` 格式）：

```json
{
  "mcpServers": {
    "e2m2e": {
      "command": "/path/to/venv/bin/e2m2e",
      "args": ["mcp-serve"],
      "cwd": "/path/to/e2m2e-repo"
    }
  }
}
```

配置完成后在客户端直接用自然语言驱动，例如：

> 设计一条 L2 南族 NRHO，近月点高度 3000 km。

### 时空系统

- 坐标系转换：J2000 / ITRF93（SPICE 高精度）/ IAU 2006，GMAT 兼容的原生 ITRF，动态坐标轴 VNB / LVLH。
- 时空联合转换：TDT+GCRS ↔ TDB+EBCRS（r2s2 后端，含相对论项）。
- SPICE 星历与时间管理：内核加载、UTC / TDB / TAI 时间尺度、天体状态与帧旋转查询。

### 积分器与动力学

- Rust 积分器内核：单步 RK（PD45 / PD78 / RK89）、Adams 多步、Störmer–Cowell 二阶积分；状态转移矩阵（STM）传播；事件检测（terminal / direction 语义）。
- 动力学模型：CR3BP（快速设计）、星历 N 体（SPICE，精确外推）、含太阳解析摄动的 BCR4BP，以及三者之间的转换。
- 高精度力模型：点质量与第三体引力、球谐重力场（含固体潮）、ECOM 9 系数光压、大气阻力、太阳光压、连续推力。

### 任务轨道设计

- 周期轨道族：DRO、Halo、Lyapunov、Lissajous、共振轨道（RO）、DPO、Axial、三角平动点 SPO / LPO、Horseshoe。
- 数值算法：微分修正、多重打靶、延拓；全链路 CR3BP 初猜 → 星历修正 → 高精度预报。
- 名义轨道契约（NominalOrbit）：等间距状态表 + Floquet 基 + 投影因子，供轨道保持直接消费。

### 转移轨道设计

- 脉冲转移：Lambert 求解与 porkchop 扫描、多脉冲优化（Lawden 主矢量检验）、霍曼直接转移（HMN）。
- 低能量转移：月球引力辅助（LGA）、WSB 太阳引力辅助弹道捕获、不变流形与庞加莱截面拼接。
- 低推力转移：Q-law 初猜 + 打靶 / 配点。
- 网格搜索 + 非线性规划两步法（Rust Rayon 并行）。

### 轨道控制

- 三种控制律：特征点、目标点严格、目标点宽松；蒙特卡洛测定轨与推力误差仿真。
- 角动量管理：姿态发动机联合控制。

## 文档

设计决策记录（ADR）见 [docs/adr/](docs/adr/)，ADR 0043 起以中文书写，更早条目为英文历史存档。接口字段的权威描述在请求/响应模型的字段描述中，CLI `--help` 与 MCP schema 与之同源。

## 测试与代码规范

```bash
make test
make check
```

## 贡献

见 [CONTRIBUTING.md](CONTRIBUTING.md)

## 引用

```bibtex
@software{e2m2e,
  title = {e2m2e: Earth to Moon, Moon to Earth Transfer Orbit Design Library},
  author = {ouyangjiahong},
  email = {ouyangjiahong22@nudt.edu.cn},
  url = {https://github.com/cislunarspace/e2m2e},
  version = {5.8.7},
  year = {2026},
}
```

## 许可证

[Apache 2.0](LICENSE)
