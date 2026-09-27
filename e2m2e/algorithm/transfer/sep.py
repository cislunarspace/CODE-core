"""SEP/NEP 电推进功率模型、Edelbaum 解析近似与螺旋初猜（ADR 0055 决策 5）。

本模块是 #724 切片：行星际低推力预设计的解析前置件。全部为标量闭式，
不依赖星历、SPICE 或 Rust 扩展；消费方是低推力打靶与 Sims-Flanagan
预设计（#725）的初猜生成。

单位约定：距离 km、速度 km/s、质量 kg、推力 N、时间 s、功率 W。

口径声明：

- 标准重力 ``g₀`` 复用 :data:`~e2m2e.algorithm.transfer.thrust_arcs.G0_MPS2`
  （9.81 m/s²），与 :meth:`~e2m2e.algorithm.transfer.thrust_arcs.ThrustArcSequence.fuel_kg`
  及 Rust ``e2m2e-forces`` 增广态质量流率同源，不另立第二常数；
- 1 AU 取 :data:`~e2m2e.data.constants.AU_KM`；
- 推力映射 ``T = 2·η·P/c_e`` 里的 ``c`` 是**有效排气速度**
  ``c_e = Isp·g₀``，不是真空光速：喷流功率守恒 ``ηP = ½·ṁ·c_e²`` 与
  ``ṁ = T/c_e`` 联立唯一解出 ``T = 2ηP/c_e``；按光速取值时 2300 W / η=0.65 /
  Isp=3100 s 的 NSTAR 级推力器只得约 1e-5 N（光子火箭量级），而量级正确的
  NSTAR 推力约 0.09 N。本模块因此只出现 ``c_e``，不出现光速。

⑦ 类人工对照（非断言，ADR 0055 决策 3）：ARM 螺旋 ΔV ≈ 4.6 km/s——#722 裁决
与 ADR 0055 决策 2 引用的 ARM 任务分析量级；编写时未能定位论文原文出处，
待补“作者 年份 §节”。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from e2m2e.data.constants import AU_KM

from .hohmann import MU_EARTH
from .thrust_arcs import G0_MPS2, ThrustArc, ThrustArcSequence

__all__ = [
    "SpiralGuess",
    "constant_thrust_final_mass",
    "constant_thrust_transfer_time",
    "edelbaum_delta_v",
    "edelbaum_delta_v_inclined",
    "mass_flow_rate",
    "nep_available_power",
    "sep_available_power",
    "spiral_arc_guess",
    "thrust_from_power",
]


def _finite(value: float, name: str) -> float:
    """取有限标量，非有限即报错。"""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} 必须为有限数，当前 {value!r}")
    return numeric


def _finite_nonnegative(value: float, name: str) -> float:
    """取有限非负标量。"""
    numeric = _finite(value, name)
    if numeric < 0.0:
        raise ValueError(f"{name} 不得为负，当前 {numeric!r}")
    return numeric


def _finite_positive(value: float, name: str) -> float:
    """取有限正标量。"""
    numeric = _finite(value, name)
    if numeric <= 0.0:
        raise ValueError(f"{name} 必须为正，当前 {numeric!r}")
    return numeric


def sep_available_power(p0_w: float, p_bus_w: float, r_helio_km: float) -> float:
    """太阳电推进（SEP）可用推进功率（论文模型，ADR 0055 ⑤）。

    ``P(r) = max(0, P₀·(1AU/r)² − P_bus)``：太阳阵列功率按日心距离平方反比
    衰减，扣除载荷/平台常耗 ``P_bus`` 后的余额为推进可用功率；余额为负表示
    供电不足以点火，截为 0。

    Args:
        p0_w: 1 AU 处的太阳阵列功率（W）。
        p_bus_w: 平台常耗功率（W）。
        r_helio_km: 日心距离（km）。

    Returns:
        可用推进功率（W），非负。
    """
    p0 = _finite_nonnegative(p0_w, "p0_w")
    p_bus = _finite_nonnegative(p_bus_w, "p_bus_w")
    r_helio = _finite_positive(r_helio_km, "r_helio_km")
    return max(0.0, p0 * (AU_KM / r_helio) ** 2 - p_bus)


def nep_available_power(p0_w: float, p_bus_w: float) -> float:
    """核电推进（NEP）可用推进功率。

    ``P = max(0, P₀ − P_bus)``：反应堆功率与日心距离无关，功率为常数。

    Args:
        p0_w: 反应堆输出功率（W）。
        p_bus_w: 平台常耗功率（W）。

    Returns:
        可用推进功率（W），非负。
    """
    p0 = _finite_nonnegative(p0_w, "p0_w")
    p_bus = _finite_nonnegative(p_bus_w, "p_bus_w")
    return max(0.0, p0 - p_bus)


def thrust_from_power(power_w: float, efficiency: float, isp_s: float) -> float:
    """功率限制推力映射 ``T = 2·η·P/c_e``（ADR 0055 ①）。

    推导：喷流功率守恒 ``ηP = ½·ṁ·c_e²``，质量流 ``ṁ = T/c_e``，联立得
    ``ηP = ½·T·c_e``，即 ``T = 2ηP/c_e``。这里 ``c_e = Isp·g₀`` 是**有效排气
    速度**，不是真空光速（见模块 docstring）。

    Args:
        power_w: 推进可用电功率（W）。
        efficiency: 推力器总效率 ``η``，``0 < η ≤ 1``。
        isp_s: 比冲（s）。

    Returns:
        推力（N），非负。
    """
    power = _finite_nonnegative(power_w, "power_w")
    eta = _finite(efficiency, "efficiency")
    if eta <= 0.0 or eta > 1.0:
        raise ValueError(f"efficiency 必须落在 (0, 1] 内，当前 {eta!r}")
    isp = _finite_positive(isp_s, "isp_s")
    return 2.0 * eta * power / (isp * G0_MPS2)


def mass_flow_rate(thrust_n: float, isp_s: float) -> float:
    """质量流率 ``ṁ = T/(Isp·g₀)``（ADR 0055 ②）。

    与 :meth:`~e2m2e.algorithm.transfer.thrust_arcs.ThrustArcSequence.fuel_kg`
    的 ``ṁ = δ·T_max/(Isp·g₀)`` 以及 Rust ``e2m2e-forces`` 增广态质量流率
    同口径（同一 ``g₀``）。

    Args:
        thrust_n: 推力（N）。
        isp_s: 比冲（s）。

    Returns:
        质量流率（kg/s），非负。
    """
    thrust = _finite_nonnegative(thrust_n, "thrust_n")
    isp = _finite_positive(isp_s, "isp_s")
    return thrust / (isp * G0_MPS2)


def edelbaum_delta_v(v1_km_s: float, v2_km_s: float) -> float:
    """Edelbaum 共面圆-圆低推力螺旋 ΔV 闭式：``ΔV = |v₁ − v₂|``（ADR 0055 ①）。

    Edelbaum (1961), "Propulsion Requirements for Controllable Satellites",
    *ARS Journal* 31(8)。共面圆轨道间多圈螺旋的速度增量只取决于两端圆速度
    之差，与推力大小和转移圈数无关。

    Args:
        v1_km_s: 初始圆轨道速度（km/s）。
        v2_km_s: 目标圆轨道速度（km/s）。

    Returns:
        ΔV（km/s），非负。
    """
    v1 = _finite_positive(v1_km_s, "v1_km_s")
    v2 = _finite_positive(v2_km_s, "v2_km_s")
    return abs(v1 - v2)


def edelbaum_delta_v_inclined(v1_km_s: float, v2_km_s: float, inc_change_deg: float) -> float:
    """Edelbaum 含倾角变化的低推力螺旋 ΔV 闭式（ADR 0055 ①）。

    ``ΔV = sqrt(v₁² + v₂² − 2·v₁·v₂·cos(Δi))``：把倾角变化折算为两端速度矢量
    的夹角。``Δi = 0`` 退化回共面形式 :func:`edelbaum_delta_v`；正负倾角变化
    对称（仅经余弦依赖 ``Δi``）。Edelbaum (1961)。

    Args:
        v1_km_s: 初始圆轨道速度（km/s）。
        v2_km_s: 目标圆轨道速度（km/s）。
        inc_change_deg: 倾角变化量（deg，带符号，仅幅值起作用）。

    Returns:
        ΔV（km/s），非负。
    """
    v1 = _finite_positive(v1_km_s, "v1_km_s")
    v2 = _finite_positive(v2_km_s, "v2_km_s")
    inc = _finite(inc_change_deg, "inc_change_deg")
    return math.sqrt(v1 * v1 + v2 * v2 - 2.0 * v1 * v2 * math.cos(math.radians(inc)))


def constant_thrust_transfer_time(
    m0_kg: float, thrust_n: float, delta_v_km_s: float, isp_s: float
) -> float:
    """常推力弧转移时间闭式（ADR 0055 ①）。

    ``t = (m₀·c_e/T)·(1 − e^(−ΔV/c_e))``，其中 ``c_e = Isp·g₀``（m/s）、
    ``ΔV`` 换算为 m/s。用 ``math.expm1`` 计算指数部分，保证小 ΔV 时的相对精度。

    Args:
        m0_kg: 初始质量（kg）。
        thrust_n: 推力（N）。
        delta_v_km_s: 速度增量（km/s）。
        isp_s: 比冲（s）。

    Returns:
        转移时间（s），非负。
    """
    m0 = _finite_positive(m0_kg, "m0_kg")
    thrust = _finite_positive(thrust_n, "thrust_n")
    delta_v = _finite_nonnegative(delta_v_km_s, "delta_v_km_s")
    isp = _finite_positive(isp_s, "isp_s")
    exhaust_m_s = isp * G0_MPS2
    return (m0 * exhaust_m_s / thrust) * -math.expm1(-delta_v * 1000.0 / exhaust_m_s)


def constant_thrust_final_mass(m0_kg: float, delta_v_km_s: float, isp_s: float) -> float:
    """常推力弧末质量闭式 ``m_f = m₀·e^(−ΔV/c_e)``（火箭方程，ADR 0055 ①）。

    末质量只依赖初始质量、ΔV 与 ``c_e = Isp·g₀``，**与推力无关**（推力决定
    耗时而非耗量），因此本函数不接收 ``thrust`` 形参。

    Args:
        m0_kg: 初始质量（kg）。
        delta_v_km_s: 速度增量（km/s）。
        isp_s: 比冲（s）。

    Returns:
        末质量（kg），``(0, m₀]``。
    """
    m0 = _finite_positive(m0_kg, "m0_kg")
    delta_v = _finite_nonnegative(delta_v_km_s, "delta_v_km_s")
    isp = _finite_positive(isp_s, "isp_s")
    return m0 * math.exp(-delta_v * 1000.0 / (isp * G0_MPS2))


@dataclass(frozen=True)
class SpiralGuess:
    """共面圆-圆低推力螺旋初猜（常推力单弧）。

    Attributes:
        arcs: 单条满油门常推力弧（惯性系常方向 ``±v̂``）。
        delta_v_km_s: Edelbaum ΔV（km/s），即两端圆速度之差。
        power_w: 可用推进功率（W）。
        thrust_n: 由功率映射得到的推力（N）。
        duration_s: 常推力弧时长（s）。
        final_mass_kg: 末质量（kg）。

    解析恒等式：``arcs.fuel_kg(EngineConfig(t_max=thrust_n, isp=Isp))`` 等于
    ``m₀ − final_mass_kg``（同一 ``g₀`` 口径下的质量流与火箭方程互洽）。
    """

    arcs: ThrustArcSequence
    delta_v_km_s: float
    power_w: float
    thrust_n: float
    duration_s: float
    final_mass_kg: float


def spiral_arc_guess(
    initial_state: npt.ArrayLike,
    target_radius_km: float,
    m0_kg: float,
    *,
    p0_w: float,
    p_bus_w: float,
    efficiency: float,
    isp_s: float,
    r_helio_km: float = AU_KM,
    mu_km3_s2: float = MU_EARTH,
    t_start_s: float = 0.0,
) -> SpiralGuess:
    """共面圆-圆低推力螺旋初猜：±v̂ 常推力单弧（MALTO spiral-guess 同款）。

    按 Edelbaum 共面解析 ΔV 生成一条满油门常推力弧：抬升轨道（目标速度更小）
    取 ``+v̂`` 沿初速方向，降低轨道取 ``−v̂``。方向为惯性系常向量——MALTO 式
    近似，真实螺旋中 ``v̂`` 缓慢转动；该语义与
    ``e2m2e.algorithm.forces`` 的 ``direction_frame="VNB"`` 速度方向定义对齐。

    消费方：低推力打靶取 ``tf = t0 + duration_s``、初值控制默认满油门沿初速
    （与本弧方向一致）；Sims-Flanagan 预设计（#725）直接消费 ``arcs``。

    Args:
        initial_state: 初始状态 ``(6,)``（位置 km + 速度 km/s）。
        target_radius_km: 目标圆轨道半径（km）。
        m0_kg: 初始质量（kg）。
        p0_w: 1 AU 处的太阳阵列功率（W，SEP 档）。
        p_bus_w: 平台常耗功率（W）。
        efficiency: 推力器总效率 ``η``。
        isp_s: 比冲（s）。
        r_helio_km: 日心距离（km），决定 SEP 功率档。
        mu_km3_s2: 中心天体引力参数（km³/s²）。
        t_start_s: 弧起始时刻（s）。

    Returns:
        :class:`SpiralGuess`，单条满油门常推力弧。

    Raises:
        ValueError: 状态形状/有限性/初速、目标半径、质量、引力参数或起始时刻
            非法；或可用功率为零；或目标圆轨道速度与初速相同（无需螺旋弧）。
    """
    state = np.asarray(initial_state, dtype=float)
    if state.shape != (6,):
        raise ValueError(f"初态必须为 (6,) 位置-速度向量，当前形状 {state.shape}")
    if not np.all(np.isfinite(state)):
        raise ValueError("初态分量必须全部有限")
    v1 = float(np.linalg.norm(state[3:6]))
    if v1 <= 0.0:
        raise ValueError("初速为零，无法定义 ±v̂ 推力方向")

    target_radius = _finite_positive(target_radius_km, "target_radius_km")
    m0 = _finite_positive(m0_kg, "m0_kg")
    mu = _finite_positive(mu_km3_s2, "mu_km3_s2")
    t_start = _finite(t_start_s, "t_start_s")

    v2 = math.sqrt(mu / target_radius)
    delta_v = edelbaum_delta_v(v1, v2)
    if delta_v == 0.0:
        raise ValueError("目标圆轨道速度与初速相同，无需螺旋弧")

    power = sep_available_power(p0_w, p_bus_w, r_helio_km)
    thrust = thrust_from_power(power, efficiency, isp_s)
    if thrust == 0.0:
        raise ValueError("可用推进功率为 0，无法构造常推力弧")

    duration = constant_thrust_transfer_time(m0, thrust, delta_v, isp_s)
    final_mass = constant_thrust_final_mass(m0, delta_v, isp_s)

    direction = state[3:6] / v1
    if v2 > v1:
        direction = -direction

    arc = ThrustArc(
        t_start=t_start,
        t_end=t_start + duration,
        throttle=1.0,
        direction=direction,
    )
    return SpiralGuess(
        arcs=ThrustArcSequence((arc,)),
        delta_v_km_s=delta_v,
        power_w=power,
        thrust_n=thrust,
        duration_s=duration,
        final_mass_kg=final_mass,
    )
