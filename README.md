# e2m2e: Earth to Moon, Moon to Earth

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/e2m2e)](https://pypi.org/project/e2m2e/)
[![CI](https://github.com/cislunarspace/CODE-core/actions/workflows/ci.yml/badge.svg)](https://github.com/cislunarspace/CODE-core/actions/workflows/ci.yml)
[![GitHub stars](https://img.shields.io/github/stars/cislunarspace/CODE-core.svg)](https://github.com/cislunarspace/CODE-core/stargazers)
[![Rust: 1.98.0](https://img.shields.io/badge/rust-1.98.0-orange.svg)](https://www.rust-lang.org/)
[![Docs](https://img.shields.io/badge/docs-在线文档-blue)](https://cislunarspace.github.io/CODE-core/)

e2m2e 是地月空间算法工具集。

## 安装

用 [uv](https://docs.astral.sh/uv/) 安装：

```bash
uv pip install e2m2e
```

从源码开发：

```bash
git clone https://github.com/cislunarspace/CODE-core.git
cd e2m2e
make dev
```

<details>
  <summary>Windows 安装 make</summary>
  Windows 默认不提供 `make`。可使用 [Scoop](https://scoop.sh/) 安装，在 PowerShell 中依次执行：

  ```powershell
  Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
  irm get.scoop.sh | iex
  scoop install make
  ```
</details>

e2m2e 所需的全部星历数据已打包在 [GitHub Release](https://github.com/cislunarspace/CODE-core/releases) 的 `kernels-v1` 中，`make dev` 会自动下载到 `kernels/`。也可手动下载解压到该目录。

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

e2m2e 的调用面共 21 个工具，由三个暴露类承载：任务级入口 Facade、轨道库 Catalog、分区分析 Spatiography。进程内 API、CLI 与 MCP 共用这一工具面，清单由 `Facade().tool_inventory()` 派生，是唯一来源。

### MCP

e2m2e 为上述全部工具设计了 MCP 接口。

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

### 任务级工具（Facade，8 个）

- `design_orbit`：任务轨道设计。`orbit_type` 共 17 种：DRO、DPO、NRHO、HALO、LYAPUNOV、LISSAJOUS、AXIAL、RO、ELFO 冻结轨道，以及 L4 / L5 三角平动点的 SPO / LPO / HORSESHOE 细分。含微分修正、多重打靶与延拓，全链路从 CR3BP 初猜经星历修正到高精度预报。
- `control_orbit`：轨道保持蒙特卡洛仿真。三种控制律（特征点、目标点严格、目标点宽松）、蒙特卡洛测定轨与推力误差仿真、角动量管理（姿态发动机联合控制）。消费名义轨道契约（NominalOrbit 等间距状态表，Floquet 预计算不在本仓库范围，见 `.out-of-scope/nominal-orbit-floquet-precompute.md`）。
- `transfer_design`：转移轨道设计。脉冲（Lambert 求解与 porkchop 扫描、Lawden 主矢量多脉冲优化、霍曼直接转移 HMN）、低能量（月球引力辅助 LGA、WSB 太阳引力辅助弹道捕获、不变流形与庞加莱截面拼接）、低推力（Q-law 初猜 + 打靶 / 配点）、网格搜索 + 非线性规划两步法（Rust Rayon 并行）。
- `mission_architecture_search`：行星际多借力（MGA）链网格搜索（Lambert leg + 无动力 flyby 等模 / 近心点剔除 + 日心 Tisserand 诊断，返回 top-N）。
- `low_thrust_preliminary`：低推力转移预设计。Sims-Flanagan 多 leg 段中冲量直接法 NLP（SLSQP，全解析雅可比），二体 conic 与星历 n 体双档后端。
- `orbit_propagation`：轨道预报（Rust 积分内核，含 STM 传播与事件检测）。
- `spacetime_transform`：时空坐标转换。坐标系 J2000 / ITRF93（SPICE 高精度）/ IAU 2006、GMAT 兼容的原生 ITRF、动态坐标轴 VNB / LVLH。支持时空联合转换 TDT+GCRS 与 TDB+EBCRS 互转（r2s2 后端，含相对论项）。
- `valid_ranges`：请求侧条件值域清单。design_orbit 与族生成的合法参数区间及离散选项，包版本即值域版本。

### 轨道库（Catalog，8 个）

- `orbit_family_generation`：轨道族生成，九族（HALO / NRHO / AXIAL / LISSAJOUS / SPO / LPO / HORSESHOE / DRO / RO）。库开启时逐成员入库，可按 `family_id` 整族取回。
- `catalog_query` / `catalog_get`：多维过滤查询与完整记录取回。`records/*.json + *.npz` 是事实来源，`catalog.db` 只是可重建索引。
- `catalog_tag` / `catalog_export` / `catalog_terminology` / `catalog_delete`：教学标签写入、查询子集打包导出（导出包可直接作为库打开）、术语图例与闭值集、按 record_id 删除（不可撤销）。
- `catalog_sweep`：参数空间批量扫描入库（jacobi 能量窗口、振幅 / 近月点高度网格）。

### 分区分析（Spatiography，5 个）

- `spatiography_scales`：分区解析尺度计算。
- `spatiography_classify`：分区区域分类。
- `spatiography_boundaries`：分区边界几何。
- `spatiography_resonance_atlas`：共振图集。
- `spatiography_dynamical_map`：六域两层天图。

### 数值内核与 SPICE（Rust 底座，供全部工具共用）

- Rust 积分器内核：单步 RK（PD45 / PD78 / RK89）、Adams 多步、Störmer–Cowell 二阶积分。含状态转移矩阵（STM）传播与事件检测（terminal / direction 语义）。
- 动力学模型：CR3BP（快速设计）、星历 N 体（SPICE，精确外推）、含太阳解析摄动的 BCR4BP，以及三者之间的转换。
- 高精度力模型：点质量与第三体引力、球谐重力场（含固体潮）、ECOM 9 系数光压、大气阻力、太阳光压、连续推力。
- SPICE 星历与时间管理：内核加载、UTC / TDB / TAI 时间尺度、天体状态与帧旋转查询。

## 文档

在线文档（教程 / 示例 / API 参考）见 <https://cislunarspace.github.io/CODE-core/>。本地构建：

```bash
uv pip install sphinx myst-parser sphinx-autoapi shibuya
make docs   # 产物在 docs/_build/html/
```

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
  url = {https://github.com/cislunarspace/CODE-core},
  version = {5.9.10},
  year = {2026},
}
```

## 许可证

[Apache 2.0](LICENSE)
