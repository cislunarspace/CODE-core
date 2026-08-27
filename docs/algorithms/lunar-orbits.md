# Lunar Orbit Design / 环月轨道设计

[English](#english) | [简体中文](#简体中文)

## English

This page explains how e2m2e designs lunar orbits. The companion example
`examples/main_lunar_orbits.py` demonstrates the whole process with four
orbits; this page clarifies what stands behind the example: what lunar orbits
are, which spacetime systems e2m2e uses, what steps computation takes, and how
data flows.

Start from a concrete task: give e2m2e four numbers — semi-major axis
7000 km, perilune altitude 200 km, inclination 75 degrees, argument of
perilune 270 degrees — and ask for this orbit's positions over the coming
days. In which coordinate frame do these four numbers make sense? Through
which conversions and computations must they pass to become a trustworthy
ephemeris? Run this task through and lunar orbit design is understood.

### 1. What lunar orbits are

A lunar orbit revolves around the Moon as its central body. By lunar distance
there are four layers; each layer has different dominant dynamics and a
different design language.

| Layer | Orbit | Scale & shape | Dominant dynamics | Stability & period |
|---|---|---|---|---|
| ① | Low lunar orbit LLO | 100–2000 km above the surface, near-circular | Lunar gravity dominant, approximates two-body Kepler | Stable, period ≈ 2 h |
| ② | Elliptical lunar frozen orbit ELFO | Low perilune (≈ 200 km), apolune thousands to tens of thousands of km, high eccentricity | Lunar gravity plus non-spherical perturbations | Period ≈ 1 day, line of apsides can freeze |
| ③ | Distant retrograde orbit DRO | 10–50 thousand km from the Moon, retrograde | Earth–Moon three-body dynamics significant, yet the orbit still closes around the Moon | Neutrally stable, period days to weeks |
| ④ | Halo | Around Earth–Moon L1 or L2, ≈ 60 thousand km from the Moon | Three-body dynamics dominant | Unstable, period ≈ half a month |

Scale is the main thread. The farther from the Moon, the larger Earth's
gravity share, with dynamics transitioning from two-body Kepler to the
three-body problem and orbits from stable to unstable.

Layer ① LLO speaks pure two-body language. Keplerian elements fully describe
the orbit; period on the order of 2 h, near-circular shape; main uses are
remote sensing and landing intermediate orbits. e2m2e has no separate LLO
type: taking the ELFO pipeline's eccentricity toward zero yields near-circular
orbits.

Layer ② ELFO is the most subtle class of lunar orbit. As a large-ellipse
orbit flies around the Moon, lunar non-spherical perturbations (mainly J2)
rotate the line of apsides secularly, perilune altitude drifts, and the orbit
stops repeating. Choose suitable inclination and argument of perilune so that
perturbations' secular effects cancel and apsidal drift approaches zero — the
orbit freezes. The e2m2e example uses inclination 75°, argument of perilune
270° (apolune pointing at Earth). Be clear: freezing is a verified property,
not a constructed one. The ELFO pipeline does not solve; the user supplies
parameters, the pipeline propagates under full perturbed dynamics and
statistics drift, answering with data whether those parameters freeze.

Layer ③ DRO flies against the Moon's orbital direction. Three-body dynamics
is already significant but the orbit still closes around the Moon. DRO is
neutrally stable — neither diverging nor converging — cheap to maintain, and a
candidate orbit for lunar orbital stations. It cannot be described by
Keplerian elements; it needs three-body language.

Layer ④ Halo moves around a libration point without directly encircling the
Moon; perilune sits thousands to tens of thousands of km up depending on
amplitude. Three-body dynamics dominates, the orbit is unstable and needs
continuous station keeping. NRHO is the near-rectilinear form of large-amplitude
Halos with perilune only a few thousand km up — NASA's Gateway station uses
one.

One phrase sums it up: lunar orbits are a family, not a single orbit. Design
must first ask which layer the orbit sits in, then pick the matching model
language.

### 2. Spacetime systems and frame conversion

#### Time

UTC appears only at input/output boundaries. Dynamics use TDB as independent
variable, called ephemeris time ET in SPICE. Why: UTC has leap seconds and is
discontinuous, unfit as dynamics' variable; TDB is the unified time scale of
ephemerides and dynamics. They currently differ by about 69 s; using UTC as
TDB directly shifts the Moon's position by ~70 km (Moon orbits Earth at
~1 km/s). e2m2e interfaces accept UTC strings and convert everything internal
to ET seconds.

#### Frames

Lunar orbit design passes through four frames, each with its own role:

1. Moon-centered inertial frame. Axes aligned with J2000, origin at the Moon.
   Home of Keplerian elements. ELFO's four inputs (semi-major axis, perilune
   altitude, inclination, argument of perilune) mean something only here;
   geometrical quantities like perilune/apolune/period are computed here too.

2. Earth-centered inertial frame (GCRS). J2000 axes, origin at Earth. Home of
   propagation. e2m2e's ephemeris force model is built Earth-centered;
   spacecraft and lunar states both live in this frame. The Moon-centered
   state equals spacecraft-Earth-centered state minus the Moon's Earth-
   centered state — one subtraction converts frames. Moon-centered quantities
   such as perilune altitude are extracted from Earth-centered ephemeris after
   propagation.

3. Earth–Moon synodic frame. Rotating frame, x-axis toward the Moon,
   Earth–Moon stationary, libration points fixed. Home of CR3BP. The CR3BP
   model is simplest here; initial guesses and corrections happen in it. It is
   nondimensional: lengths divided by characteristic length, times by
   characteristic time. Two origin conventions coexist: barycentric
   normalization (Moon at x = 1-mu) and geocentric normalization (Moon at
   x = 1); internal computation uses barycentric, while output ephemeris
   synodic_position uses geocentric.

4. Moon-fixed frame (MOON_PA principal axes). Expansion frame of the lunar
   gravity field's spherical harmonics. Non-spherical perturbation
   accelerations are computed here, then rotated back to inertial.

Characteristic values come from the DE421 baseline: mu = 0.01215058535
(Earth–Moon mass ratio), characteristic length 384400 km (mean Earth–Moon
distance), characteristic time 375190 s (≈ 4.34 days). The lunar gravity field
uses GRGM900C, reference radius 1738 km.

#### Implementation and one pitfall

Synodic↔J2000 conversion is done by `SynodicJ2000System`, with the batched
version sunk into Rust. The synodic x-axis takes the instantaneous Moon–Earth
unit vector, z-axis the normal of the Moon–Earth orbital plane, y completing
right-handed. Synodic coordinates divide by the instantaneous Earth–Moon
distance, which varies between 360 thousand and 405 thousand km.

Hence a pitfall you must know: `synodic_position` divides each point by the
instantaneous distance, so converting nondimensional coordinates back to km
with constant 384400 introduces ~5% scale error — an 1838 km orbit computes as
1746 km in the example. For exact Moon-centered distance, subtract the Moon's
position from Earth-centered inertial position; both ephemeris position_km and
lunar ephemeris are available without touching the synodic frame. The example's
`_moon_centric_position_km` does exactly this.

### 3. Computation flow

By orbit type e2m2e dispatches to two pipelines, corresponding to two
dynamics languages.

#### ELFO pipeline

`orbit_type=ELFO`; input is Keplerian elements. Six steps:

1. Convert eccentricity: perilune altitude plus lunar radius gives perilune
   distance; e = 1 - rp/a.
2. Elements to Moon-centered Cartesian. Classical two-body formulas; true
   anomaly 0, i.e., start at perilune.
3. Superpose the Moon's Earth-centered state: Moon-centered Cartesian plus
   Moon state → Earth-centered inertial initial value.
4. Full-perturbation propagation in Earth-centered frame. Rust integrator,
   relative tolerance 1e-12. Force models include Earth/Moon/Sun/major-planet
   gravity, Earth and Moon nonspherical to degree 10, cannonball SRP.
5. Extract Moon-centered elements pointwise: each point's Earth-centered state
   minus the Moon, batched conversion to Keplerian elements.
6. Drift statistics: first-last differences of eccentricity, argument of
   perilune, perilune distance over the arc; linear fit of argument of
   perilune for annual drift rate.

This pipeline performs no correction. Whether freezing happens depends on the
input parameters; the pipeline validates under real dynamics and reports
drift. The example's 75° + 270° combination shows ≈ 2° argument-of-perilune
drift over 4 days while eccentricity and perilune altitude barely move —
freezing showing itself. Annual drift rates need a longer window (default 60
days) fits to be reliable.

#### CR3BP pipeline

`orbit_type=DRO`, `HALO`, etc.; input is shape parameters. Initial values are
not computed, they are generated:

1. Generate initial guess. Family generators produce periodic orbits inside
   CR3BP. DRO starts from a seed orbit with differential correction +
   continuation fixing x-axis crossings; amplitude = mean of min/max
   Moon-centered distance. Halo starts from Richardson's third-order analytic
   approximation; after differential correction it walks the family; amplitude
   is the z-coordinate at y=0 crossings, positive north, negative south.
2. Phase location: place the initial point per the phase parameter on the
   periodic orbit; DRO additionally offsets half a period.
3. Sample nodes along the orbit (patch points). Strategies differ per family:
   Halo densifies near perilune (high speed, ill-conditioned STMs); NRHO and
   others sample uniformly in time. Deleting nodes near perilune remains as a
   utility function for comparison (forcing inclusion of epoch t=0) but is no
   longer the production default for NRHO since #473.
4. Batch conversion synodic→J2000.
5. Ephemeris correction. CR3BP differs from real ephemeris: the lunar orbit is
   not circular; solar and planetary perturbations exist; an ideal periodic
   orbit no longer closes under real dynamics. Rust multiple shooting anchors
   nodes onto real ephemeris. Stable orbits (DRO) take the velocity-weighted
   two_level path — correct one rev then extrapolate freely, bounded; unstable
   orbits (Halo/NRHO) take segmented full-arc shooting, integrating segment by
   segment to fill the ephemeris grid. NRHO fixes 1 rev/segment at step 1;
   Halo allows at most 3 revs/segment.
6. Assemble the ephemeris table.

NRHO and Halo share segmented shooting; discretization defaults are
independent: uniform time + 1 rev/segment. Defaults L2 southern family,
perilune altitude 5000 km, phase 0.5, about one-month arc converge to a
nominal ephemeris as long as the time grid (#473; 5.7.2's delete-near-perilune
default still had epoch holes or non-converging merge tiers at this scale).
Closer-in (~2000 km perilune) short arcs work too.

#### Where computation happens

Numerical heavy lifting of both pipelines runs in Rust: propagation, shooting,
batched frame conversion, batched ephemeris queries. Python only orchestrates:
construct request, dispatch pipeline, interpret results. All four example
orbits take well under a minute (release build); most of it is the ELFO
pipeline's strict-tolerance integration, not Python.

### 4. Data flow

Data enters as request and exits as result, passing dispatch and two pipelines
in between.

Input is `DesignOrbitRequest`, Pydantic-validated. orbit_type decides
dispatch; shape parameters validated per type with defaults filled; ELFO
requires semi-major axis, defaults inclination 75°, argument of perilune 270°,
perilune altitude 200 km, propagating 60 days; propagation parameters in
seconds, default output step 3600 s; perturbation switches and harmonic degree
overridable.

On receiving the request, `design_orbit` dispatches by orbit_type and produces
an `OrbitDesignResult`:

- `ephemeris`: nominal ephemeris in an `EphemerisTable` container. Each row
  holds UTC calendar time, GCRS position (km) and velocity (m/s), synodic
  nondimensional position. GCRS is mission data; synodic serves three-body
  perspective plots.
- `initial_state`: Earth-centered inertial 6-dim state at epoch, feedable
  directly to prediction and control chains.
- `cr3bp_orbit`, `cr3bp_jacobi`, `correction`: CR3BP reference periodic orbit,
  Jacobi constant, correction result. None/nan for ELFO scenarios.
- `drift_e`, `drift_aop_deg`, `drift_rp_km`,
  `secular_aop_rate_deg_per_year`: ELFO freeze diagnostics.
- `moon_centric_elements`: ELFO's Moon-centered element series, every point.

Restating the frame discipline when consuming ephemeris: Moon-centered
geometric quantities (perilune altitude, apolune altitude, Moon distance)
come from GCRS position minus Moon position — never reconstruct via synodic ×
constant. The synodic series exists to view the orbit's three-body perspective
shape.

### 5. Companion example

`examples/main_lunar_orbits.py` turns all of the above into four runnable
orbits matching the four layers:

- LLO: ELFO pipeline, semi-major axis 1838 km, perilune altitude 100 km,
  eccentricity near zero, propagating 2 days ≈ 24 revs.
- ELFO: semi-major axis 7000 km, perilune altitude 200 km, propagating 4 days
  ≈ 4 revs.
- DRO: amplitude 50 thousand km, period 8.4 days, propagating 9 days for over
  a full rev.
- Halo: L2, amplitude 30 thousand km, period 14.6 days, propagating 15 days
  for one full rev.

The example draws three figures. Fig 1: Moon-centric distance vs time for all
four orbits, log axis — the scale hierarchy from 1800 to 70 thousand km at a
glance. Fig 2: Moon-centered X-Y plane shapes, two subplots splitting
near-Moon and distant regions. Fig 3: ELFO freeze verification — argument of
perilune, eccentricity, perilune altitude evolving in time, watching whether
the apsidal line stays near 270°.

To run:

```bash
python examples/main_lunar_orbits.py --save          # headless: save PNGs
python examples/main_lunar_orbits.py                 # interactive plots
python examples/main_lunar_orbits.py --skip DRO      # skip one orbit
```

SPICE kernels required (`kernels/` at repo root). Each orbit prints design
parameters, elapsed time, peri/apolume, and freeze diagnostics. To dig into
single-Halo design details see `examples/main_design.py`.

## 简体中文

这篇文档讲 e2m2e 怎么设计环月轨道。配套示例 `examples/main_lunar_orbits.py` 用四条轨道演示了全过程，本文讲清示例背后的东西：环月轨道是什么、e2m2e 用哪些时空系统、计算经过哪些步骤、数据怎么流动。

先从一个具体的任务说起。给 e2m2e 四个数：半长轴 7000 km、近月点高度 200 km、倾角 75 度、近月点幅角 270 度，要它算出这条轨道未来几天的位置。这四个数在哪个坐标系里有意义？它们要经过哪些转换、哪些计算，才能变成一份可信的星历？把这个任务跑通，环月轨道设计就懂了。

## 一、环月轨道是什么

环月轨道是以月球为中心天体的环绕轨道。按离月距离分四层，每层的主导动力学不同，设计语言也不同。

| 层 | 轨道 | 尺度与形态 | 主导动力学 | 稳定性与周期 |
|---|---|---|---|---|
| ① | 近月轨道 LLO | 离月面 100 到 2000 km，近圆 | 月球引力主导，近似二体开普勒 | 稳定，周期约 2 小时 |
| ② | 大椭圆冻结轨道 ELFO | 近月点低（约 200 km），远月点数千到上万 km，高偏心 | 月球引力加非球形摄动 | 周期约 1 天，拱线可冻结 |
| ③ | 远距轨道 DRO | 离月 1 到 5 万 km，逆行 | 地月三体动力学显著，但轨道仍绕月闭合 | 中性稳定，周期数天到十几天 |
| ④ | 平动点轨道 Halo | 绕地月 L1 或 L2，月心距离约 6 万 km | 三体动力学主导 | 不稳定，周期约半个月 |

主线是尺度。离月越远，地球引力占比越高，动力学从二体开普勒过渡到三体问题，轨道从稳定变为不稳定。

第①层 LLO 是纯二体语言。开普勒根数完整描述轨道，周期 2 小时量级，形状近圆，主要用途是遥感与着陆中间轨道。e2m2e 没有单独的 LLO 类型，把 ELFO 管线的离心率取到趋近 0 就是近圆轨道。

第②层 ELFO 是环月轨道里最有讲头的一类。大椭圆轨道绕月飞行时，月球非球形摄动（主要是 J2 项）使拱线长期旋转，近月点高度随之变化，轨道不再重复。选合适的倾角与近月点幅角，让摄动的长期效应相互抵消，拱线漂移趋近于零，轨道就冻结了。e2m2e 示例采用倾角 75 度、近月点幅角 270 度的构型，即远月点朝地球。需要分清：冻结是验证出来的性质，不是构造出来的。ELFO 管线不做求解，用户给一组参数，管线在全摄动动力学里传播后统计漂移量，用数据回答这组参数冻不冻结。

第③层 DRO 逆着月球公转方向飞行。三体动力学已显著，但轨道仍绕月闭合。DRO 中性稳定，既不发散也不收敛，维持成本低，是月轨空间站的候选轨道之一。它不能用开普勒根数描述，得用三体语言。

第④层 Halo 绕平动点运动，不直接绕月，近月点离月面几千到几万 km，随振幅变化。三体动力学主导，轨道不稳定，需要连续保持。NRHO 是大振幅 Halo 的近直线形态，近月点离月面只有几千公里，NASA 的月球门户空间站采用。

一个词概括：环月轨道是一个家族，不是一条轨道。设计时必须先问轨道落在哪一层，再选对应的模型语言。

## 二、时空系统与坐标变换

### 时间

UTC 只出现在输入输出边界。动力学自变量用 TDB，SPICE 里称星历时间 ET。原因：UTC 有闰秒，不连续，不能做动力学的自变量；TDB 是历表与动力学统一的时间尺度。二者当前相差约 69 秒，若拿 UTC 直接当 TDB 查月球位置，月球以约 1 km/s 绕地球公转，位置错出约 70 km。e2m2e 接口收 UTC 字符串，内部一律转 ET 秒。

### 坐标系

环月轨道设计经过四套坐标系，各有各的用途：

1. 月心惯性系。轴与 J2000 对齐，原点在月球。开普勒根数的家。ELFO 输入的四个数（半长轴、近月点高度、倾角、近月点幅角）只有在这个系里有意义；近月点、远月点、周期这些几何量也在此计算。

2. 地心惯性系（GCRS）。轴为 J2000，原点在地球。传播的家。e2m2e 的星历力模型以地球为中心构建，航天器状态与月球状态都在这个系里描述。月心状态等于航天器地心状态减月球地心状态，一条减法完成两个系的换算。近月点高度这类月心量，传播后从地心星历里提取。

3. 地月会合系。旋转系，x 轴指向月球，地月静止，平动点定点。CR3BP 的家。CR3BP 模型在这个系里最简单，初猜与修正都在此完成。它是无量纲的：长度除以特征长度，时间除以特征时间。两个原点约定并存：质心归一（月球在 x = 1-mu）与地心归一（月球在 x = 1），e2m2e 内部计算用质心归一，输出星历的 synodic_position 用地心归一。

4. 月固系（MOON_PA 主轴系）。月球引力场球谐的展开系。非球形摄动加速度在月固系算，再旋转回惯性系。

特征量取自 DE421 基准集：mu = 0.01215058535（地月质量比）、特征长度 384400 km（地月平均距离）、特征时间 375190 s（约 4.34 天）。月球引力场用 GRGM900C 模型，参考半径 1738 km。

### 变换的实现与一个坑

会合系与 J2000 的转换由 `SynodicJ2000System` 完成，批量版本下沉 Rust。会合系 x 轴取瞬时月地单位矢量，z 轴取月地轨道面法向，y 轴右手补齐。会合系坐标除以的是当时的地月距离，这个距离在 36 万到 40.5 万 km 之间变化。

由此产生一个必须知道的坑：`synodic_position` 每点除以的是瞬时地月距离，把无量纲坐标乘常数 384400 还原成 km 会引入约 5% 的尺度误差。示例里 1838 km 的轨道这样算会得到 1746 km。要拿精确月心距离，用地心惯性位置减月球位置，星历表的 position_km 与月球星历都可得，不经过会合系。示例里的 `_moon_centric_position_km` 正是这么做的。

## 三、计算流程

e2m2e 按轨道类型分派到两条管线，对应两种动力学语言。

### ELFO 管线

`orbit_type=ELFO`，输入是开普勒根数。六步：

1. 换算离心率。近月点高度加月球半径得近月点距离，e = 1 - rp/a。
2. 根数转月心笛卡尔。经典二体公式，真近点角取 0，即从近月点出发。
3. 叠加月球地心状态。月心笛卡尔加月球当时的地心状态，得到地心惯性系初值。
4. 地心系全摄动传播。Rust 积分器，相对容差 1e-12。力模型含地球、月球、太阳与各大行星引力，地球与月球非球形 10 阶，炮弹模型光压。
5. 逐点提取月心根数。每点地心状态减月球位置，批量换算开普勒根数。
6. 漂移统计。计算传播弧段内离心率、近月点幅角、近月点距离的首末差，再对近月点幅角做线性拟合得年漂移率。

这条管线不做修正。冻结与否由输入参数决定，管线负责用真实动力学验证并报告漂移量。示例的 75 度倾角加 270 度近月点幅角组合，4 天传播的近月点幅角漂移约 2 度，离心率与近月点高度基本不动，就是冻结的表现。年漂移率用更长的窗口（默认 60 天）拟合才可靠。

### CR3BP 管线

`orbit_type=DRO`、`HALO` 等，输入是形状参数。初值不是算出来的，是生成的：

1. 生成初猜。族生成器在 CR3BP 内生成周期轨道。DRO 从种子轨道出发，固定 x 轴穿越点做微分修正加延拓，振幅定义为月心距离最小最大值的均值。Halo 从 Richardson 三阶解析近似出发，微分修正后沿族行走，振幅是 y=0 穿越点的 z 坐标，正北负南。
2. 相位定位。按相位参数在周期轨道上定位初值，DRO 额外偏移半个周期。
3. 采样节点。沿轨道采 patch points。策略按族分开：Halo 在近月点加密（近月点速度大、状态转移矩阵病态）；NRHO 与其余族等时间采样。删近月点附近节点仍可作为工具函数对照（强制含历元 `t=0`），但自 #473 起不再作 NRHO 生产默认。
4. 批量转会合系到 J2000。
5. 星历修正。CR3BP 模型与真实星历有差：月球轨道非圆、有太阳与其他行星摄动，理想周期轨道拿到真实动力学里不再闭合。修正用 Rust 多重打靶把节点锚到真实星历上。稳定轨道（DRO）走速度加权的 two_level 路径，修正一圈后自由外推有界；不稳定轨道（Halo/NRHO）走 segmented 全程分段打靶，逐段积分填满星历网格。NRHO 第 1 步固定 1 圈/段，Halo 最长 3 圈/段。
6. 组装星历表。

NRHO 与 Halo 同走 segmented，离散默认独立：等时间 + 1 圈/段。默认 L2 南族、近月点高度 5000 km、相位 0.5、约一个月弧长可收敛得到与时间网格等长的标称星历（#473；5.7.2 的删近月点默认在该量级仍有历元空洞或合并层不收敛）。更贴月（近月高约 2000 km）短弧同样可用。

### 计算在哪发生

两条管线的数值重活都在 Rust：传播、打靶、批量坐标转换、批量星历查询。Python 只做编排：构造请求、分派管线、解释结果。示例四条轨道全量不到 1 分钟（release 构建），大头是 ELFO 管线的严格容差积分，不是 Python。

## 四、数据流

数据从请求进，从结果出，中间经过分派与两条管线。

输入是 `DesignOrbitRequest`，Pydantic 校验。orbit_type 决定分派；形状参数按类型校验并填默认值；ELFO 必填半长轴，默认倾角 75 度、近月点幅角 270 度、近月点高度 200 km、传播 60 天；传播参数统一用秒，输出步长默认 3600 秒；摄动开关与球谐阶数可覆盖。

`design_orbit` 收到请求后按 orbit_type 分派，产出一个 `OrbitDesignResult`：

- `ephemeris`：标称星历，`EphemerisTable` 容器。每行含 UTC 日历时间、GCRS 位置（km）与速度（m/s）、会合系无量纲位置。GCRS 是任务数据，会合系是给三体视角绘图用的。
- `initial_state`：历元时刻地心惯性系 6 维状态，可直接喂给预报与控制链路。
- `cr3bp_orbit`、`cr3bp_jacobi`、`correction`：CR3BP 参考周期轨道、Jacobi 常数与修正结果。ELFO 场景为 None 或 nan。
- `drift_e`、`drift_aop_deg`、`drift_rp_km`、`secular_aop_rate_deg_per_year`：ELFO 的冻结诊断。
- `moon_centric_elements`：ELFO 的月心根数序列，每点都有。

消费星历时的坐标系纪律重申一遍：月心几何量（近月点高度、远月点高度、月心距离）从 GCRS 位置减月球位置得到，不要用会合系乘常数还原。会合系列用于观察轨道在三体视角下的形态。

## 五、配套示例

`examples/main_lunar_orbits.py` 把上面的内容变成可跑的四条轨道，对应四个分层：

- LLO：ELFO 管线，半长轴 1838 km、近月点高度 100 km，离心率趋近 0，传播 2 天约 24 圈。
- ELFO：半长轴 7000 km、近月点高度 200 km，传播 4 天约 4 圈。
- DRO：振幅 5 万 km，周期 8.4 天，传播 9 天取一整圈多。
- Halo：L2、振幅 3 万 km，周期 14.6 天，传播 15 天取一整圈。

示例画三张图。图 1 四条轨道的月心距离随时间演化，对数轴，一眼看清从 1800 km 到 7 万 km 的尺度层次。图 2 月心 X-Y 平面形态，分近月区与远距区两个子图。图 3 ELFO 冻结性验证，近月点幅角、离心率、近月点高度三个量随时间的演化，看拱线是否稳定在 270 度附近。

运行方式：

```bash
python examples/main_lunar_orbits.py --save          # 无头环境，存 PNG
python examples/main_lunar_orbits.py                 # 交互式出图
python examples/main_lunar_orbits.py --skip DRO      # 跳过某条轨道
```

需要 SPICE 内核（仓库根 `kernels/`）。每条轨道的设计参数、耗时、近远月点与冻结诊断都打印在终端。想深挖单条 Halo 的设计细节，看 `examples/main_design.py`。
