# ADR 0010：r2s2 接入与 TDT+GCRS ↔ TDB+EBCRS 时空坐标转换

**状态**：已接受（已实施）
**日期**：2026-07-30
**关联**：issue #252、`archive/plans/dfh-parity-prd.md` FR5、ADR 0003（坐标三层抽象）、ADR 0007（动态轴）

## 背景

DFH 对齐需求 FR5 要求实现 TDT+GCRS ↔ TDB+EBCRS 双向时空坐标转换，统一走
r2s2（中科院地月空间时空坐标系库，`https://github.com/r2s2-astro/r2s2`，
需求方确定的必装依赖）。TDT 是 TT（地球时）的旧称；EBCRS 是地月质心天球
参考系，轴向与 BCRS/ICRS 一致，原点在地月质心。

## r2s2 调研结论

r2s2（PyPI 包名 `r2s2`，版本 0.1.0）提供 BCRS、GCRS、LCRS 三个参考架下
六种时空坐标（TCB/TDB 配 BCRS 位置、TCG/TT 配 GCRS 位置、TCL/LRT 配
LCRS 位置）的两两互转，转换含 IAU 决议要求的相对论项，API 为
`TT2TDB`、`TDB2TT` 等成对函数，内部经 calcephpy 读历表。

对本任务的覆盖面：

- **直接覆盖**：(TT, GCRS 地心位置) ↔ (TDB, BCRS 太阳系质心位置)，即
  `TT2TDB` / `TDB2TT`。时间尺度转换（TDT↔TDB）内嵌其中，不需要
  e2m2e 现有 SPICE 时间链路另补。
- **缺口一：EBCRS 原点**。r2s2 的 TDB 侧位置以太阳系质心为原点，没有
  地月质心架。EBCRS 与 BCRS 只差原点平移，用同一历表中的地月质心位置
  补齐：`x_ebcrs = xs − x_emb(t)`。
- **缺口二：速度**。r2s2 只转换位置三元组，不转换速度。本次封装同样
  只覆盖位置；速度转换待有明确需求再议。
- **历表要求**：必须有内置时间星历（TT−TDB），推荐 JPL `de440t.bsp`
  （带 `t` 后缀变体）。仓库现有的 `de440s.bsp`、`de430.bsp` 均不含时间
  星历，实测调用时报 calceph 数据缺失错误。INPOP21a 的 spice 格式历表
  对（主文件 + 时间星历 `*_time.bsp`）经实测可用，作为无法访问 JPL
  时的备选（二者均为历表数据，转换算法不变，差别在历表本身的模型
  差异）。注意 INPOP 原生格式（.dat）不可用：其主文件与时间星历分属
  两个文件，而 calceph 不允许同时打开多个 INPOP 原生文件；spice 格式
  无此限制。

## 安装与依赖

`r2s2>=0.1.0` 加入 `pyproject.toml` 必装依赖（PyPI 源，无需 git 依赖）。
其依赖 calcephpy 在 Windows 下从 PyPI 源码构建可行（本机 Python 3.13
实测通过）；r2s2 项目 release 页另提供 Windows 预编译 wheel 备用。

## 封装决策

新模块 `e2m2e/algorithm/coordinate/gcrs_ebcrs.py`，公开类 `GCRSEBCRSSystem`：

```python
system = GCRSEBCRSSystem("kernels/de440t.bsp")
jd_tdb, r_ebcrs = system.gcrs_to_ebcrs(jd_tt, r_gcrs)
jd_tt, r_gcrs = system.ebcrs_to_gcrs(jd_tdb, r_ebcrs)
```

决策要点：

1. **形状仿 `SynodicJ2000System`**（同包内的转换器类先例），不继承
   `Axes`。理由：`Axes` 抽象是"给定 et 返回旋转矩阵"，装不下时空联合
   转换——本转换同时切换时间尺度并含相对论项，不是任一固定时刻的
   纯旋转。EBCRS 的空间部分（ICRS 轴 + 地月质心原点）本就可以用现有
   `CoordinateSystem(ICRSAxes, CelestialBodyOrigin("EARTH MOON BARYCENTER"))`
   表达，新类的价值恰在 Axes/Origin 模型之外的部分。
2. **EMB 平移走同一历表**：经 calcephpy 直接查地月质心状态，与 r2s2
   内部所用历表保持同一份数据，不混用 SPICE 链路里的 de440s，避免
   跨历表误差混入转换结果。calcephpy 是 r2s2 的硬依赖，此用法不引入
   新依赖。
3. **构造时校验时间星历**：历表不含 TT−TDB 时间星历时立即报
   `CoordinateDataError` 并指明需要 de440t 变体，不把 calceph 底层错误
   留给用户猜。
4. **已知限制**：`R2S2.init_E` 是进程级全局状态，用不同历表构造多个
   实例会互相覆盖（后构造者生效），已在类 docstring 标明。

## 验证

`tests/algorithm/coordinate/test_gcrs_ebcrs.py`：

- 双向往返一致性（GCRS→EBCRS→GCRS 与反向），位置容差 1 cm、时间容差
  1 ms（容差下界由 r2s2 的 1 ns/1 mm 迭代精度与单段儒略日 float 的
  约 40 µs 分辨率共同决定，留一个量级余量）；
- 与同历表牛顿式平移参照（即 DFH CoordinateTransform 的做法：平移
  地心-地月质心偏移、不区分 TT/TDB）的差分量化：空间差即相对论修正，
  地月距离量级下应在毫米到百米之间，时间差应在 TDB−TT 的 ±1.7 ms
  包络内；
- 与 e2m2e 现有 SPICE 链路（SPICEManager + de440s）的差分，容差 1 km，
  用于抓接线性错误（轴向、原点、单位），不用于评定精度；
- 地心处纯时间尺度转换与 ERFA `dtdb`（Fairhead & Bretagnon 解析模型）
  对比，亚毫秒一致；
- 历表缺时间星历（de440s.bsp）与文件缺失两条报错路径。

## 备注

本机实施期间 JPL 站点（ssd.jpl.nasa.gov / naif.jpl.nasa.gov）网络不可达，
`de440t.bsp` 未能下载，功能验证改用 IMCCE 镜像的 INPOP21a spice 历表对
完成；测试代码优先使用 `de440t.bsp`，缺省时退到 INPOP21a，两者皆无则
跳过（与仓库既有 SPICE 测试的内核缺失跳过惯例一致）。

INPOP21a 验证实测数值（DFH main.cpp 演示位置，LEO）：双向往返位置差
< 0.001 mm、时间差 < 1 µs；与同历表牛顿式参照的空间差（即相对论修正）
0.20 m，月球距离量级位置（~36.5 万 km）为 10.1 m；TDB−TT 为 ±0.36 ms，
地心处与 ERFA `dtdb` 独立模型差 < 0.03 µs。
