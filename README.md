# e2m2e — Earth to Moon, Moon to Earth

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/e2m2e)](https://pypi.org/project/e2m2e/)
[![CI](https://github.com/cislunarspace/e2m2e/actions/workflows/ci.yml/badge.svg)](https://github.com/cislunarspace/e2m2e/actions/workflows/ci.yml)
[![GitHub stars](https://img.shields.io/github/stars/cislunarspace/e2m2e.svg)](https://github.com/cislunarspace/e2m2e/stargazers)
[![Rust: stable](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org/)

e2m2e 是面向地月空间任务规划的**算法工具集基础设施**。在“LLM+Agent”式自主任务规划系统中，大模型负责理解任务意图、分解与编排子任务，e2m2e 负责提供精确可靠的轨道计算工具：建立地月空间的动力学模型，生成周期轨道族，设计轨道之间的转移路径，并把结果画出来检查。

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

从源码开发（需要 [Rust 工具链](https://www.rust-lang.org/tools/install)，用于构建积分器内核）：

```bash
git clone https://github.com/cislunarspace/e2m2e.git
cd e2m2e
uv sync
make setup   # 拉取 CSPICE 编译包 + SPICE 内核（cspice-v1 / kernels-v1 release，首次必跑）
make dev     # maturin develop 构建并安装 Rust 扩展（spice 默认开启）
```

### SPICE 内核

星历动力学需要 NASA SPICE 内核文件。本项目测试所需的全部内核（行星星历、地球自转、月球姿态、闰秒与行星常数）已打包在 [GitHub Release](https://github.com/cislunarspace/e2m2e/releases) 的 `kernels-v1` 中。三种配置方式：

- **自动配置（推荐）**：`make setup` 下载并放入 `kernels/`（见 `scripts/download_kernels.py`）。
- **手动下载**：从 Release 下载 `kernels-v1`，解压到 `kernels/` 目录。
- **自备数据**：使用自己的内核文件，放入 `kernels/` 目录，或将 `$SPICE_KERNEL_DIR` 指向其所在路径。

官方来源：[NASA NAIF](https://naif.jpl.nasa.gov/naif/data.html)。

## 快速开始

设计一条地月 L2 Halo 轨道（需先完成上方「SPICE 内核」配置）：

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

## 能力

已建成与未建成的部分按领域列出。详细的能力清单与 API 文档见[在线文档](https://cislunarspace.github.io/e2m2e/)；逐版本变更见 [CHANGELOG.md](CHANGELOG.md)。

**时空系统**

- 坐标系转换：J2000 / ITRF93（SPICE 高精度）/ IAU 2006，GMAT 兼容的原生 ITRF，动态坐标轴 VNB / LVLH。
- 时空联合转换：TDT+GCRS ↔ TDB+EBCRS（r2s2 后端，含相对论项）。
- SPICE 星历与时间管理：内核加载、UTC / TDB / TAI 时间尺度、天体状态与帧旋转查询。

**积分器与动力学**

- Rust 积分器内核：单步 RK（PD45 / PD78 / RK89）、Adams 多步、Störmer–Cowell 二阶积分；状态转移矩阵（STM）传播；事件检测（terminal / direction 语义）。
- 动力学模型：CR3BP（快速设计）、星历 N 体（SPICE，精确外推）、含太阳解析摄动的 BCR4BP，以及三者之间的转换。
- 高精度力模型：点质量与第三体引力、球谐重力场（含固体潮）、ECOM 9 系数光压、大气阻力、太阳光压、连续推力，传播精度与 GMAT、DFH 对齐到亚百米级。

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
- 未建成：MCP 服务化封装——Facade 方法的元数据与派生机制已就位，`create_server` / `e2m2e mcp-serve` 尚为占位（`[mcp]` extra）。

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

测试按“验证什么”分七类，目录镜像源码结构：

- `theory`：数学公式与物理理论（解析解、Jacobi 常数等守恒量、文献公式）
- `integrator`：积分器对解析轨道的精度
- `force`：力模型的加速度与雅可比
- `data`：内核、参考帧、物理常数等数据层
- `orchestration`：设计、修正、转移等算法编排链路
- `interface`：Facade API 的校验、响应与错误翻译
- `aux`：日志等辅助工具

断言来自解析解、守恒量与文献公式，不与其他软件对照。Rust 侧数值方法在 `crates/*/tests/` 对解析解。

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
  version = {5.6.10},
  year = {2026},
}
```

## License

[Apache 2.0](LICENSE)
