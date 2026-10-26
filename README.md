# e2m2e: Earth to Moon, Moon to Earth

![e2m2e](assets/banner.png)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/e2m2e)](https://pypi.org/project/e2m2e/)
[![CI](https://github.com/cislunarspace/CODE-core/actions/workflows/ci.yml/badge.svg)](https://github.com/cislunarspace/CODE-core/actions/workflows/ci.yml)
[![GitHub stars](https://img.shields.io/github/stars/cislunarspace/CODE-core.svg)](https://github.com/cislunarspace/CODE-core/stargazers)
[![Rust: 1.98.0](https://img.shields.io/badge/rust-1.98.0-orange.svg)](https://www.rust-lang.org/)
[![Docs](https://img.shields.io/badge/docs-在线文档-blue)](https://cislunarspace.github.io/CODE-core/)

e2m2e 是地月空间算法工具集，覆盖 CR3BP 任务轨道与轨道族、星历传播、转移设计、轨道保持、时空坐标转换、轨道库和地月空间分区分析。数值密集路径由 Rust 与 PyO3 扩展 `e2m2e._integrators` 承担，Python 侧负责问题构造、算法编排和接口边界。

调用方可以在进程内用 `from e2m2e.api import Facade` 直接调用，也可以用 `e2m2e <工具名>` 走 CLI，或经 `e2m2e mcp-serve` 在 MCP 客户端里使用同一套工具。GUI 通过 `e2m2e serve-stdio` 的 sidecar 协议接入。四条路径共用同一份工具清单。工具名与参数模型以 `tool_inventory(Facade())` 为准，MCP schema 和 CLI 参数都从请求模型生成。各路径的配置见[调用方式](#调用方式)。

## 安装

用 [uv](https://docs.astral.sh/uv/) 从 PyPI 安装：

```bash
uv pip install e2m2e
```

从源码开发：

```bash
git clone https://github.com/cislunarspace/CODE-core.git
cd CODE-core
make dev
```

`make dev` 同步 Python 依赖，下载 CSPICE 与星历内核，并用 maturin 构建 Rust 扩展。

<details>
  <summary>Windows 安装 make</summary>
  Windows 默认不提供 `make`。可使用 [Scoop](https://scoop.sh/) 安装，在 PowerShell 中依次执行：

  ```powershell
  Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
  irm get.scoop.sh | iex
  scoop install make
  ```
</details>

SPICE 星历内核不随包分发，默认内核在 [GitHub Release](https://github.com/cislunarspace/CODE-core/releases) 的 `kernels-v1` 资产中。`make dev` 会自动下载到 `kernels/`。从 PyPI 安装时，手动下载 `kernels-v1` 资产并解压到任意目录，再用 `SPICE_KERNEL_DIR` 指向该目录。纯 CR3BP 的族生成与分区分析不需要内核，涉及星历的设计、传播与坐标转换需要。

## 快速开始

设计一条地月 L2 Halo 轨道。振幅 30000 km，预报 7 天：

```python
from e2m2e.api import Facade

facade = Facade()

result = facade.design_orbit(
    orbit_type="Halo",
    collinear_point=2,
    amplitude=30000.0,
    epoch=[2024, 1, 1, 0, 0, 0.0],
    duration=7.0 * 86400.0,
)

print(result.status, result.orbit_type)
print(result.initial_state)
```

`design_orbit` 返回 `DesignOrbitResponse`。`status` 是收敛状态，`initial_state` 是修正后的无量纲会合系状态，`ephemeris` 是标称星历数据，含 GCRS 位置、速度与会合系字段。需要更长弧段时增大 `duration`。这个例子需要 SPICE 内核，获取方式见安装一节。

## 能力范围

e2m2e 对外提供一套工具，按职责分给三个类：任务级入口 `Facade`、轨道库 `Catalog`、分区分析 `Spatiography`。进程内 API、CLI、MCP 与 GUI sidecar 共用这份工具面。以下按类介绍，完整清单与参数以 `tool_inventory(Facade())` 为准。

### 任务级工具（Facade）

- `design_orbit`：任务轨道设计。`orbit_type` 共 17 种：DRO、DPO、NRHO、HALO、LYAPUNOV、LISSAJOUS、AXIAL、RO、ELFO 冻结轨道，以及 L4、L5 三角平动点的 SPO、LPO、HORSESHOE 细分。含微分修正、多重打靶与延拓，全链路从 CR3BP 初猜经星历修正到高精度预报。
- `control_orbit`：轨道保持蒙特卡洛仿真。支持特征点、目标点严格、目标点宽松三种控制律，以及测定轨与推力误差仿真、角动量管理（姿态发动机联合控制）。名义轨道可由轨道库记录或星历对象接入，契约只保证 `NominalOrbit` 等间距状态表。Floquet 基与投影因子预计算不在本仓库范围内，原因见 [.out-of-scope/nominal-orbit-floquet-precompute.md](.out-of-scope/nominal-orbit-floquet-precompute.md)。
- `transfer_design`：转移轨道设计，覆盖脉冲、低能量、低推力和网格搜索四类方法。脉冲类含 Lambert 求解与 porkchop 扫描、Lawden 主矢量多脉冲优化、霍曼直接转移 HMN。低能量类含月球引力辅助 LGA、WSB 太阳引力辅助弹道捕获、不变流形与庞加莱截面拼接。低推力类用 Q-law 初猜加打靶或配点。网格搜索加非线性规划的两步法在 Rust 侧用 Rayon 并行。
- `mission_architecture_search`：行星际多借力链（MGA）网格搜索。每个转移段（leg）用 Lambert 求解，借力点按无动力飞越等模与近心点约束筛选，再用日心 Tisserand 参数诊断，返回前 N 个候选。
- `low_thrust_preliminary`：低推力转移预设计。用 Sims-Flanagan 多段中冲量直接法构造 NLP，SLSQP 求解并给出全解析雅可比。后端分二体 conic 与星历 n 体两档。
- `orbit_propagation`：轨道预报。Rust 积分内核支持状态转移矩阵传播与终端事件检测。
- `spacetime_transform`：时空坐标转换。支持 J2000、ITRF93（SPICE 高精度）、IAU 2006、GMAT 兼容的原生 ITRF，以及 VNB、LVLH 动态坐标轴。还支持 TDT+GCRS 与 TDB+EBCRS 的时空联合转换，后端为 r2s2，含相对论项。
- `valid_ranges`：查询 `design_orbit` 与族生成在各条件下的合法参数区间和离散选项。值域随包版本冻结。

### 轨道库（Catalog）

轨道库以 `records/*.json + *.npz` 为事实来源，`catalog.db` 只是可重建索引。

- `orbit_family_generation`：轨道族生成，覆盖 HALO、NRHO、AXIAL、LISSAJOUS、SPO、LPO、HORSESHOE、DRO、RO 以及周期倍化的 FIGURE_EIGHT、RING 共十一族。库开启时逐成员入库，可按 `family_id` 整族取回。
- `catalog_query`：多维过滤查询，返回摘要列表。
- `catalog_get`：按 `record_id` 取完整记录，含数组段与请求快照。
- `catalog_tag`：写入教学标签，整体替换原有标签。
- `catalog_export`：把查询子集打包导出，导出包可直接作为库打开。
- `catalog_terminology`：返回分类学标签图例、`orbit_family` 闭值集与 `transfer_type` 闭值集，无参数。
- `catalog_delete`：按 `record_id` 删除记录，不可撤销。
- `catalog_sweep`：参数空间批量扫描入库，支持 Jacobi 能量窗口以及振幅、近月点高度网格。

### 分区分析（Spatiography）

- `spatiography_scales`：分区解析尺度计算。
- `spatiography_classify`：分区区域分类。
- `spatiography_interval_classify`：带不确定度的区间化分区分类，输出可能区域与确定区域双标签。
- `spatiography_boundaries`：分区边界几何。
- `spatiography_resonance_atlas`：共振图集。
- `spatiography_dynamical_map`：六域两层天图。

### 数值内核与 SPICE

这一层供全部工具共用。

- Rust 积分器内核：单步 RK（PD45、PD78、RK89）、Adams 多步、Störmer–Cowell 二阶积分。支持状态转移矩阵传播与事件检测，事件语义含 terminal 与 direction。
- 动力学模型：CR3BP（快速设计）、星历 N 体（SPICE，精确外推）、含太阳解析摄动的 BCR4BP，以及三者之间的转换。
- 高精度力模型：点质量与第三体引力、球谐重力场（含固体潮）、ECOM 9 系数光压、大气阻力、太阳光压、连续推力。
- SPICE 星历与时间管理：内核加载、UTC、TDB、TAI 时间尺度、天体状态与帧旋转查询。

## 调用方式

进程内调用见快速开始。CLI 的每个工具对应一个子命令，工具名的下划线转连字符，例如 `e2m2e design-orbit --help`。

MCP 需要安装可选依赖：

```bash
uv pip install "e2m2e[mcp]"
```

在 MCP 客户端的 `mcpServers` 配置中注册服务器，下面是一个例子。`command` 指向安装了 e2m2e 的环境里的 `e2m2e` 可执行文件。`cwd` 指向仓库根时，服务器可以找到 `kernels/` 下的星历内核。内核在别处时，用 `env` 设置 `SPICE_KERNEL_DIR`。

```json
{
  "mcpServers": {
    "e2m2e": {
      "command": "/path/to/venv/bin/e2m2e",
      "args": ["mcp-serve"],
      "cwd": "/path/to/CODE-core"
    }
  }
}
```

配置完成后，可以在客户端用自然语言发起任务，例如：

> 设计一条 L2 南族 NRHO，近月点高度 3000 km。

## 文档

在线文档包含教程、示例与 API 参考：<https://cislunarspace.github.io/CODE-core/>。教程走接口层暴露类，示例脚本走 `algorithm` 层细粒度 API。本地构建：

```bash
uv pip install sphinx myst-parser sphinx-autoapi shibuya
make docs
```

产物在 `docs/_build/html/`。文档构建不需要安装项目本体、Rust 或 SPICE。

设计决策记录见 [docs/adr/](docs/adr/)，ADR 0043 起以中文书写，更早条目为英文历史存档。接口字段的权威描述在请求与响应模型的字段描述中，CLI `--help` 与 MCP schema 与之同源。

## 开发与验证

开发环境用 `make dev` 构建。交互式探索或性能基准可用 `make dev-release`。

```bash
make test    # Rust workspace 与 Python 全量测试
make check   # 格式、lint、类型与架构门禁
```

## 贡献

见 [CONTRIBUTING.md](CONTRIBUTING.md)。Issue 与 PR 的模板、标签、Project 流水线和正文写作约定都在那里。

## 引用

如果 e2m2e 对你的研究有帮助，请引用：

```bibtex
@software{e2m2e,
  title = {e2m2e: Earth to Moon, Moon to Earth Transfer Orbit Design Library},
  author = {ouyangjiahong},
  email = {ouyangjiahong22@nudt.edu.cn},
  url = {https://github.com/cislunarspace/CODE-core},
  version = {5.9.11},
  year = {2026},
}
```

## 许可证

[Apache 2.0](LICENSE)
