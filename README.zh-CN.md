# e2m2e: Earth to Moon, Moon to Earth（地月往返）

[English](README.md) | **简体中文**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/e2m2e)](https://pypi.org/project/e2m2e/)
[![CI](https://github.com/cislunarspace/e2m2e/actions/workflows/ci.yml/badge.svg)](https://github.com/cislunarspace/e2m2e/actions/workflows/ci.yml)
[![GitHub stars](https://img.shields.io/github/stars/cislunarspace/e2m2e.svg)](https://github.com/cislunarspace/e2m2e/stargazers)
[![Rust: 1.98.0](https://img.shields.io/badge/rust-1.98.0-orange.svg)](https://www.rust-lang.org/)

e2m2e 是面向地月空间任务规划的**算法工具集基础设施**。在 LLM+Agent 式自主任务规划系统中，大模型负责理解任务意图、分解与编排子任务，e2m2e 负责提供精确可靠的轨道计算工具：建立地月空间的动力学模型，生成周期轨道族，设计轨道之间的转移路径，并把结果画出来检查。

## 仓库怎么读

运行时架构只有四块：`e2m2e/api/` 是唯一对外入口（Facade，派生 CLI 与 MCP）；`e2m2e/algorithm/` 用领域知识构造问题（选轨道族、定约束、配初猜）；`crates/` 是 Rust 数值层，打靶迭代等重计算在这里收敛；`e2m2e/data/` 供星历缓存、坐标数据与常数基准。`e2m2e/tools/` 为日志/可视化辅助，`e2m2e/mbse/` 在依赖链之外。

一条任务轨道的旅程：`api` 收到请求 → `algorithm/family` 选族给初猜（种子来自 `catalog/records` 或 `algorithm/normal_form`）→ 打靶下沉 `crates/e2m2e-integrators`，每步算力在 `crates/e2m2e-forces`（星历不吃 SPICE 句柄，吃 `data/frames` 预采样缓存表；常数出自 `data/constants`）→ 结果落 `catalog/`，经 `api/cli` 与 `api/mcp` 交付。

仓库其余顶层目录（`tests/`、`examples/`、`docs/`、`scripts/`、`kernels/` 等）是测试、文档、脚本与数据资产，不在运行时依赖链上。完整设计叙事见 [docs/architecture/architecture.md](docs/architecture/architecture.md)。

## 安装

用 [uv](https://docs.astral.sh/uv/) 安装：

```bash
uv pip install e2m2e
```

发布物覆盖 Windows x86_64、Linux x86_64 与 Linux aarch64（arm64，如鲲鹏/飞腾/树莓派）的 wheel；其余平台从源码构建。

在自己的项目中使用：

```bash
uv init my-project && cd my-project
uv add e2m2e
```

从源码开发（需要 [Rust 1.98.0 工具链](https://www.rust-lang.org/tools/install)，仓库通过 `rust-toolchain.toml` 固定版本，用于构建积分器内核）：

```bash
git clone https://github.com/cislunarspace/e2m2e.git
cd e2m2e
make dev     # 唯一入口：同步依赖 + 拉取 CSPICE 编译包与 SPICE 内核 + 构建安装 Rust 扩展（spice 默认开启）
```

#### Windows 安装 make

Windows 默认不提供 `make`。可使用 [Scoop](https://scoop.sh/) 安装；在 PowerShell 中依次执行：

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
irm get.scoop.sh | iex
scoop install make
```

安装完成后重开 PowerShell，回到仓库目录执行 `make dev`（唯一入口：同步依赖 + 拉取数据 + 构建安装，见上方）。若已安装 Scoop，只需执行 `scoop install make`。

Windows 上运行 Rust 测试请使用 `make test-rust`：测试二进制依赖 Python 的 `python3.dll`，该 DLL 在 Python 安装根目录而不在虚拟环境 `Scripts/`；Makefile 会自动探测并加入测试进程 PATH。若自动探测失败，可显式执行 `make test-rust PYTHON_DLL_DIR=<含 python*.dll 的目录>`。手工排查时先对失败的测试 EXE 执行 `dumpbin /DEPENDENTS` 或 `dumpbin /IMPORTS`，确认实际缺失的 DLL，不要把 CSPICE 的 `lib/`（静态库目录）加入 PATH。

### SPICE 内核

星历动力学需要 NASA SPICE 内核文件。本项目测试所需的全部内核（行星星历、地球自转、月球姿态、闰秒与行星常数）已打包在 [GitHub Release](https://github.com/cislunarspace/e2m2e/releases) 的 `kernels-v1` 中。三种配置方式：

- **自动配置（推荐）**：`make setup` 下载并放入 `kernels/`（见 `scripts/download_kernels.py`）。
- **手动下载**：从 Release 下载 `kernels-v1`，解压到 `kernels/` 目录。
- **自备数据**：使用自己的内核文件，放入 `kernels/` 目录，或将 `$SPICE_KERNEL_DIR` 指向其所在路径。

官方来源：[NASA NAIF](https://naif.jpl.nasa.gov/naif/data.html)。

## 快速开始

设计一条地月 L2 Halo 轨道（需先完成上方 SPICE 内核配置）：

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

参数含义、返回字段与其他轨道类型见[在线文档](https://cislunarspace.github.io/e2m2e/)；可运行示例见 [`examples/`](examples/) 目录。

## MCP

e2m2e 可作为 [MCP](https://modelcontextprotocol.io/) 服务器把 18 个任务级工具（轨道设计、站保仿真、转移设计、轨道预报、时空转换、轨道族生成、5 个分区解析工具与 7 个轨道库工具）暴露给 LLM Agent，stdio 传输，不监听端口。工具清单由 Facade 方法元数据派生，产物自动入库、`record_id` 跨工具链式调用；用法与工具速查见文档 [通过 MCP 使用 e2m2e](https://cislunarspace.github.io/e2m2e/getting-started/mcp.html)。

安装 MCP extra（在已有 e2m2e 的环境里）：

```bash
uv pip install "e2m2e[mcp]"   # 或 pip install "e2m2e[mcp]"
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

`cwd` 建议钉在含 `kernels/`（SPICE 内核）与 `catalog/`（轨道库）的目录；两目录也可分别用环境变量 `SPICE_KERNEL_DIR` / `E2M2E_CATALOG_DIR` 指定绝对路径。

配置完成后在客户端直接用自然语言驱动，例如：

> 设计一条 L2 南族 NRHO，近月点高度 3000 km，然后对它做 100 次蒙特卡洛站保仿真，结果打上候选标签。

## 能力

已建成与未建成的部分按领域列出。详细的能力清单与 API 文档见[在线文档](https://cislunarspace.github.io/e2m2e/)；逐版本变更见 [CHANGELOG.md](CHANGELOG.md)。

**时空系统**

- 坐标系转换：J2000 / ITRF93（SPICE 高精度）/ IAU 2006，GMAT 兼容的原生 ITRF，动态坐标轴 VNB / LVLH。
- 时空联合转换：TDT+GCRS ↔ TDB+EBCRS（r2s2 后端，含相对论项）。
- SPICE 星历与时间管理：内核加载、UTC / TDB / TAI 时间尺度、天体状态与帧旋转查询。

**积分器与动力学**

- Rust 积分器内核：单步 RK（PD45 / PD78 / RK89）、Adams 多步、Störmer–Cowell 二阶积分；状态转移矩阵（STM）传播；事件检测（terminal / direction 语义）。
- 动力学模型：CR3BP（快速设计）、星历 N 体（SPICE，精确外推）、含太阳解析摄动的 BCR4BP，以及三者之间的转换。
- 高精度力模型：点质量与第三体引力、球谐重力场（含固体潮）、ECOM 9 系数光压、大气阻力、太阳光压、连续推力。

**任务轨道设计**

- 周期轨道族：DRO、Halo、Lyapunov、Lissajous、共振轨道（RO）、DPO、Axial、三角平动点 SPO / LPO、Horseshoe。
- 数值算法：微分修正、多重打靶、延拓；全链路 CR3BP 初猜 → 星历修正 → 高精度预报。
- 名义轨道契约（NominalOrbit）：等间距状态表 + Floquet 基 + 投影因子，供轨道保持直接消费。

**转移轨道设计**

- 脉冲转移：Lambert 求解与 porkchop 扫描、多脉冲优化（Lawden 主矢量检验）、霍曼直接转移（HMN）。
- 低能量转移：月球引力辅助（LGA）、WSB 太阳引力辅助弹道捕获、不变流形与庞加莱截面拼接。
- 低推力转移：Q-law 初猜 + 打靶 / 配点。
- 网格搜索 + 非线性规划两步法（Rust Rayon 并行）。

**轨道控制**

- 三种控制律：特征点、目标点严格、目标点宽松；蒙特卡洛测定轨与推力误差仿真。
- 角动量管理：姿态发动机联合控制。

**接口与工具**

- Facade 任务级入口，统一对外调用面。
- MCP 服务化封装：`create_server` 进程内服务器与 `e2m2e mcp-serve` 子命令（stdio 传输，`[mcp]` extra），由 Facade 方法元数据派生工具清单——轨道设计、站保仿真、转移设计、轨道预报、时空转换、轨道族生成、5 个分区解析工具与 7 个轨道库工具，产物自动入库、`record_id` 跨工具链式调用。接入配置与工具用法见文档 [通过 MCP 使用 e2m2e](https://cislunarspace.github.io/e2m2e/getting-started/mcp.html)。

## 文档

在线文档：<https://cislunarspace.github.io/e2m2e/>

本地构建：

```bash
uv sync --group docs
uv run sphinx-build -b html docs docs/_build/html
```

## 测试与代码规范

```bash
make test     # Rust 测试 + Python xdist 并行测试（需先 make setup 拉内核）
make check    # cargo fmt/clippy + ruff
```

测试按验证对象分六类，目录镜像源码结构：

- `algorithm`：坐标、动力学、设计、修正、转移等算法编排链路
- `api`：Facade 与 MCP 接口的校验、响应与错误翻译
- `data`：内核、参考帧、物理常数等数据层
- `mbse`：MBSE 数据模型、需求注册与图生成
- `numerical`：积分器对解析轨道的精度、力模型的加速度与雅可比
- `tools`：日志、格式化、可视化等辅助工具

另有 `_meta` 目录约束测试基础设施自身。断言以解析解、守恒量与文献公式为主；坐标与时间链路另有 GMAT 参考数据对照。Rust 侧数值方法在 `crates/*/tests/` 对解析解。

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
  version = {5.8.7},
  year = {2026},
}
```

## 许可证

[Apache 2.0](LICENSE)
