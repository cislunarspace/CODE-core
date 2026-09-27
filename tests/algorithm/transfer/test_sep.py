"""SEP 电推进功率模型、Edelbaum 解析近似与螺旋初猜测试（ADR 0055 决策 5）。

oracle 口径：① 闭式公式与 ② 守恒量/不变量用研究级紧容差（实现内复算 rel=1e-12
与交叉恒等式 rel=1e-9）；⑤ 功率模型为论文模型定义；⑥ 定义性数值仅 NSTAR 公开
工况点（NASA Glenn 公布的 2.3 kW / 3100 s / 92 mN），逐值给带宽与陷阱注记；
⑦ 结果性数值（ARM 螺旋 ΔV ≈ 4.6 km/s）不进断言，登记在
``e2m2e.algorithm.transfer.sep`` 模块 docstring。本模块纯解析，无星历/SPICE 依赖。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from e2m2e.algorithm.transfer import (
    EngineConfig,
    SpiralGuess,
    _equivalent_delta_v,
    constant_thrust_final_mass,
    constant_thrust_transfer_time,
    edelbaum_delta_v,
    edelbaum_delta_v_inclined,
    mass_flow_rate,
    nep_available_power,
    sep_available_power,
    spiral_arc_guess,
    thrust_from_power,
)
from e2m2e.algorithm.transfer.hohmann import MU_EARTH
from e2m2e.algorithm.transfer.thrust_arcs import G0_MPS2, ThrustArcSequence
from e2m2e.data.constants import AU_KM

pytestmark = pytest.mark.low_thrust

#: 螺旋初猜场景：LEO 6678 km（300 km 高度附近）→ GEO 42164 km。
R_LEO_KM = 6678.0
R_GEO_KM = 42164.0
V_LEO_KM_S = math.sqrt(MU_EARTH / R_LEO_KM)
V_GEO_KM_S = math.sqrt(MU_EARTH / R_GEO_KM)
M0_KG = 1000.0
P0_W = 5000.0
P_BUS_W = 500.0
ETA = 0.6
ISP_S = 3000.0


def _leo_state() -> list[float]:
    """LEO 圆轨道状态（位置 km + 速度 km/s），速度沿 +y。"""
    return [R_LEO_KM, 0.0, 0.0, 0.0, V_LEO_KM_S, 0.0]


def _geo_state() -> list[float]:
    """GEO 圆轨道状态，速度沿 +y。"""
    return [R_GEO_KM, 0.0, 0.0, 0.0, V_GEO_KM_S, 0.0]


@pytest.mark.theory
class TestPowerModel:
    """⑤ 论文功率模型：SEP 平方反比衰减 + 平台常耗，NEP 常数功率。"""

    def test_sep_power_at_1au(self):
        """1 AU 处可用功率 = P₀ − P_bus。"""
        assert sep_available_power(5000.0, 800.0, AU_KM) == pytest.approx(4200.0, rel=1e-12)

    def test_sep_power_inverse_square_scaling(self):
        """日心距离减半功率增至 4 倍，加倍降至 1/4（P_bus = 0 隔离衰减项）。"""
        p_at_1au = sep_available_power(5000.0, 0.0, AU_KM)
        at_half = sep_available_power(5000.0, 0.0, 0.5 * AU_KM)
        at_double = sep_available_power(5000.0, 0.0, 2.0 * AU_KM)
        assert at_half == pytest.approx(4.0 * p_at_1au, rel=1e-12)
        assert at_double == pytest.approx(5000.0 / 4.0, rel=1e-12)

    def test_sep_power_clamps_to_zero(self):
        """阵列余额为负时截为 0（供电不足即不可点火）。"""
        # 2 AU 处 P₀/4 = 250 W < P_bus = 900 W
        assert sep_available_power(1000.0, 900.0, 2.0 * AU_KM) == 0.0

    def test_nep_power_constant(self):
        """NEP 功率与日心距离无关，负余额截 0。"""
        assert nep_available_power(5000.0, 800.0) == pytest.approx(4200.0, rel=1e-12)
        assert nep_available_power(500.0, 800.0) == 0.0

    @pytest.mark.parametrize(
        "args",
        [
            (-1.0, 500.0, AU_KM),
            (5000.0, -1.0, AU_KM),
            (5000.0, 500.0, 0.0),
            (5000.0, 500.0, -1.0),
            (math.inf, 500.0, AU_KM),
            (5000.0, 500.0, math.nan),
        ],
    )
    def test_sep_power_validation_errors(self, args):
        """功率档输入非法（负值、零距离、非有限）即报错。"""
        with pytest.raises(ValueError):
            sep_available_power(*args)

    @pytest.mark.parametrize("args", [(-1.0, 500.0), (5000.0, -1.0), (math.nan, 0.0)])
    def test_nep_power_validation_errors(self, args):
        """NEP 功率档输入非法即报错。"""
        with pytest.raises(ValueError):
            nep_available_power(*args)


@pytest.mark.theory
class TestThrustMapping:
    """①/② 功率限制推力映射与质量流口径。"""

    @pytest.mark.parametrize(
        ("power_w", "efficiency", "isp_s"),
        [
            (2300.0, 0.61, 3100.0),
            (4500.0, 0.6, 3000.0),
            (1000.0, 0.75, 4500.0),
            (15000.0, 0.9, 8000.0),
        ],
    )
    def test_thrust_from_power_formula(self, power_w, efficiency, isp_s):
        """T = 2ηP/(Isp·g₀)（① 实现内复算）且喷流功率守恒 ½ṁc_e² = ηP（②）。"""
        expected = 2.0 * efficiency * power_w / (isp_s * G0_MPS2)
        thrust = thrust_from_power(power_w, efficiency, isp_s)
        assert thrust == pytest.approx(expected, rel=1e-12)
        exhaust_m_s = isp_s * G0_MPS2
        jet_power = 0.5 * mass_flow_rate(thrust, isp_s) * exhaust_m_s**2
        assert jet_power == pytest.approx(efficiency * power_w, rel=1e-12)

    def test_thrust_from_power_nstar_scenario(self):
        """⑥ NSTAR 满功率工况点：92 mN 推力落进 ±10% 带宽。

        数据出处（NASA Glenn 公开工况，经 Wikipedia 条目 “NASA Solar
        Technology Application Readiness” 摘引，2026-09-27 直抓核对）：
        “The 30-cm ion thruster operates over a 0.5 kW to 2.3 kW input power
        range providing thrust from 19 mN to 92 mN. The specific impulse ranges
        from 1900 s at 0.5 kW to 3100 s at 2.3 kW.”

        陷阱注记：η ≈ 0.61 系由该满功率工况点经本公式反推的总效率
        （T·Isp·g₀/(2P) = 0.092·3100·9.81/4600 ≈ 0.608）；η 的“总效率 vs 推进
        效率”与“输入功率进 PPU 还是进推力器”口径随文献而异，±10% 带宽覆盖之。
        """
        thrust = thrust_from_power(2300.0, 0.61, 3100.0)
        assert 0.085 <= thrust <= 0.100

    @pytest.mark.parametrize(
        ("thrust_n", "isp_s"),
        [(0.1835, 3000.0), (1.0, 2000.0), (0.02, 4500.0)],
    )
    def test_mass_flow_consistency(self, thrust_n, isp_s):
        """② ṁ = T/(Isp·g₀)，与 thrust_arcs/Rust 力模型同口径。"""
        assert mass_flow_rate(thrust_n, isp_s) == pytest.approx(
            thrust_n / (isp_s * G0_MPS2), rel=1e-12
        )

    def test_mass_flow_accepts_zero_thrust(self):
        """零推力无推进剂消耗（滑行弧边界）。"""
        assert mass_flow_rate(0.0, 3000.0) == 0.0

    @pytest.mark.parametrize(
        ("power_w", "efficiency", "isp_s"),
        [
            (-1.0, 0.6, 3000.0),
            (1000.0, 0.0, 3000.0),
            (1000.0, -0.1, 3000.0),
            (1000.0, 1.2, 3000.0),
            (1000.0, 0.6, 0.0),
            (1000.0, 0.6, -5.0),
            (math.nan, 0.6, 3000.0),
            (1000.0, math.inf, 3000.0),
        ],
    )
    def test_thrust_mapping_validation(self, power_w, efficiency, isp_s):
        """非物理参数（η ∉ (0,1]、非正 Isp、负/非有限功率）即报错。"""
        with pytest.raises(ValueError):
            thrust_from_power(power_w, efficiency, isp_s)

    @pytest.mark.parametrize(("thrust_n", "isp_s"), [(-1.0, 3000.0), (1.0, 0.0), (1.0, math.nan)])
    def test_mass_flow_validation(self, thrust_n, isp_s):
        """负推力或非正/非有限 Isp 即报错。"""
        with pytest.raises(ValueError):
            mass_flow_rate(thrust_n, isp_s)


@pytest.mark.theory
class TestEdelbaum:
    """① Edelbaum 共面与含倾角 ΔV 闭式。"""

    @pytest.mark.parametrize(
        ("r1_km", "r2_km"),
        [
            (R_LEO_KM, R_GEO_KM),
            (R_GEO_KM, R_LEO_KM),
            (7000.0, 7100.0),
            (10000.0, 300000.0),
        ],
    )
    def test_coplanar_circle_to_circle(self, r1_km, r2_km):
        """共面圆-圆 ΔV = |v₁ − v₂|（速度由 √(μ/r) 独立算出）。"""
        v1 = math.sqrt(MU_EARTH / r1_km)
        v2 = math.sqrt(MU_EARTH / r2_km)
        assert edelbaum_delta_v(v1, v2) == pytest.approx(abs(v1 - v2), rel=1e-12)

    def test_coplanar_leo_to_geo_magnitude(self):
        """LEO→GEO 低推力螺旋 ΔV 约 4.65 km/s（Edelbaum 共面文献量级，带宽 0.02）。"""
        delta_v = edelbaum_delta_v(V_LEO_KM_S, V_GEO_KM_S)
        assert abs(V_LEO_KM_S - 7.7258) < 1e-3
        assert abs(V_GEO_KM_S - 3.0747) < 1e-3
        assert abs(delta_v - 4.651) < 0.02

    def test_inclined_form(self):
        """含倾角形式复算、量级带、零倾角退化与正负对称。"""
        inc_deg = 28.5
        expected = math.sqrt(
            V_LEO_KM_S**2
            + V_GEO_KM_S**2
            - 2.0 * V_LEO_KM_S * V_GEO_KM_S * math.cos(math.radians(inc_deg))
        )
        delta_v = edelbaum_delta_v_inclined(V_LEO_KM_S, V_GEO_KM_S, inc_deg)
        assert delta_v == pytest.approx(expected, rel=1e-12)
        # LEO→GEO 含 28.5° 倾角变化的低推力文献量级约 5.2–5.3 km/s
        assert abs(delta_v - 5.233) < 0.1
        # Δi = 0 退化为共面形式
        assert edelbaum_delta_v_inclined(V_LEO_KM_S, V_GEO_KM_S, 0.0) == pytest.approx(
            edelbaum_delta_v(V_LEO_KM_S, V_GEO_KM_S), rel=1e-12
        )
        # 倾角变化仅经余弦依赖，正负对称
        assert edelbaum_delta_v_inclined(V_LEO_KM_S, V_GEO_KM_S, -inc_deg) == pytest.approx(
            delta_v, rel=1e-12
        )

    @pytest.mark.parametrize(
        "args",
        [
            (0.0, 3.0),
            (7.7, 0.0),
            (-7.7, 3.0),
            (math.nan, 3.0),
            (7.7, math.inf),
        ],
    )
    def test_edelbaum_validation(self, args):
        """零/负/非有限圆轨道速度即报错。"""
        with pytest.raises(ValueError):
            edelbaum_delta_v(*args)

    def test_edelbaum_inclined_validation(self):
        """含倾角形式沿用同一速度校验，并要求倾角有限。"""
        with pytest.raises(ValueError):
            edelbaum_delta_v_inclined(0.0, 3.0, 10.0)
        with pytest.raises(ValueError):
            edelbaum_delta_v_inclined(7.7, 3.0, math.nan)


@pytest.mark.theory
class TestConstantThrustClosedForm:
    """①/② 常推力时间与末质量闭式，及与火箭方程的交叉恒等式。"""

    @pytest.mark.parametrize(
        ("m0_kg", "thrust_n", "delta_v_km_s", "isp_s"),
        [
            (1000.0, 0.1835, 4.651, 3000.0),
            (2000.0, 1.0, 3.0, 2000.0),
            (500.0, 0.05, 1.5, 4000.0),
            (1500.0, 2.5, 6.0, 3000.0),
        ],
    )
    def test_transfer_time_closed_form(self, m0_kg, thrust_n, delta_v_km_s, isp_s):
        """t = (m₀c_e/T)(1 − e^(−ΔV/c_e))（① 复算），且 (m₀ − m_f)/ṁ 同值（②）。"""
        exhaust_m_s = isp_s * G0_MPS2
        expected = (m0_kg * exhaust_m_s / thrust_n) * (
            1.0 - math.exp(-delta_v_km_s * 1000.0 / exhaust_m_s)
        )
        duration = constant_thrust_transfer_time(m0_kg, thrust_n, delta_v_km_s, isp_s)
        assert duration == pytest.approx(expected, rel=1e-12)

        final_mass = constant_thrust_final_mass(m0_kg, delta_v_km_s, isp_s)
        mass_flow = mass_flow_rate(thrust_n, isp_s)
        assert duration == pytest.approx((m0_kg - final_mass) / mass_flow, rel=1e-9)

    @pytest.mark.parametrize(
        ("m0_kg", "mf_kg", "isp_s"),
        [
            (1000.0, 854.0, 3000.0),
            (2000.0, 1200.0, 5000.0),
            (500.0, 480.0, 1500.0),
        ],
    )
    def test_final_mass_roundtrip_with_tsiolkovsky(self, m0_kg, mf_kg, isp_s):
        """② 与 _equivalent_delta_v 正逆互洽：m_f → ΔV → m_f 恒等。"""
        delta_v = _equivalent_delta_v(m0_kg, mf_kg, isp_s)
        assert constant_thrust_final_mass(m0_kg, delta_v, isp_s) == pytest.approx(mf_kg, rel=1e-12)

    def test_zero_delta_v_degenerate(self):
        """ΔV = 0：零耗时、末质量 = 初质量。"""
        assert constant_thrust_transfer_time(1000.0, 0.2, 0.0, 3000.0) == 0.0
        assert constant_thrust_final_mass(1000.0, 0.0, 3000.0) == 1000.0

    @pytest.mark.parametrize(
        ("m0_kg", "thrust_n", "delta_v_km_s", "isp_s"),
        [
            (0.0, 1.0, 3.0, 3000.0),
            (-1.0, 1.0, 3.0, 3000.0),
            (1000.0, 0.0, 3.0, 3000.0),
            (1000.0, -1.0, 3.0, 3000.0),
            (1000.0, 1.0, -1.0, 3000.0),
            (1000.0, 1.0, 3.0, 0.0),
            (math.nan, 1.0, 3.0, 3000.0),
        ],
    )
    def test_closed_form_validation(self, m0_kg, thrust_n, delta_v_km_s, isp_s):
        """时间闭式的非法输入即报错。"""
        with pytest.raises(ValueError):
            constant_thrust_transfer_time(m0_kg, thrust_n, delta_v_km_s, isp_s)

    @pytest.mark.parametrize(
        ("m0_kg", "delta_v_km_s", "isp_s"),
        [
            (0.0, 3.0, 3000.0),
            (1000.0, -1.0, 3000.0),
            (1000.0, 3.0, -3000.0),
            (math.inf, 3.0, 3000.0),
        ],
    )
    def test_final_mass_validation(self, m0_kg, delta_v_km_s, isp_s):
        """末质量闭式的非法输入即报错（无 thrust 形参：末质量与推力无关）。"""
        with pytest.raises(ValueError):
            constant_thrust_final_mass(m0_kg, delta_v_km_s, isp_s)


@pytest.mark.orchestration
class TestSpiralArcGuess:
    """螺旋初猜编排：LEO→GEO 场景量级。

    场景算得：P = 4500 W、T ≈ 0.1835 N、ΔV ≈ 4.651 km/s、t ≈ 2.34e7 s ≈ 271 天、
    燃料 ≈ 146 kg（末质量 ≈ 854 kg）。仅作量级 sanity，不逐一断言。
    """

    def test_raise_guess_shape_and_direction(self):
        """抬升取 +v̂ 满油门单弧，映射链（功率→推力→时长）自洽。"""
        guess = spiral_arc_guess(
            _leo_state(),
            R_GEO_KM,
            M0_KG,
            p0_w=P0_W,
            p_bus_w=P_BUS_W,
            efficiency=ETA,
            isp_s=ISP_S,
        )
        assert isinstance(guess, SpiralGuess)
        assert isinstance(guess.arcs, ThrustArcSequence)
        assert len(guess.arcs.arcs) == 1
        arc = guess.arcs.arcs[0]
        assert arc.throttle == 1.0
        assert arc.t_start == 0.0
        assert arc.t_end - arc.t_start == pytest.approx(guess.duration_s, rel=1e-12)
        v_hat = np.array(_leo_state()[3:6]) / V_LEO_KM_S
        np.testing.assert_allclose(arc.direction, v_hat, atol=1e-12)

        assert guess.power_w == pytest.approx(4500.0, rel=1e-12)
        assert guess.thrust_n == pytest.approx(2.0 * ETA * 4500.0 / (ISP_S * G0_MPS2), rel=1e-12)
        assert guess.delta_v_km_s == pytest.approx(abs(V_LEO_KM_S - V_GEO_KM_S), rel=1e-12)
        # 初猜 ΔV 即 Edelbaum 共面值，带宽防口径漂移
        assert abs(guess.delta_v_km_s - 4.651) < 0.05

    def test_guess_magnitudes(self):
        """时长与末质量落进场景量级（≈271 天、≈854 kg）。"""
        guess = spiral_arc_guess(
            _leo_state(),
            R_GEO_KM,
            M0_KG,
            p0_w=P0_W,
            p_bus_w=P_BUS_W,
            efficiency=ETA,
            isp_s=ISP_S,
        )
        assert guess.duration_s / 86400.0 == pytest.approx(271.3, rel=0.01)
        assert guess.final_mass_kg == pytest.approx(853.9, rel=0.01)
        assert guess.final_mass_kg < M0_KG

    def test_guess_fuel_matches_thrust_arcs(self):
        """② 跨实现交叉：arcs.fuel_kg 等于 m₀ − 末质量（同一 g₀ 口径）。"""
        guess = spiral_arc_guess(
            _leo_state(),
            R_GEO_KM,
            M0_KG,
            p0_w=P0_W,
            p_bus_w=P_BUS_W,
            efficiency=ETA,
            isp_s=ISP_S,
        )
        engine = EngineConfig(t_max=guess.thrust_n, isp=ISP_S)
        assert guess.arcs.fuel_kg(engine) == pytest.approx(M0_KG - guess.final_mass_kg, rel=1e-9)

    def test_lowering_guess_reverses_direction(self):
        """降轨（目标速度更大）取 −v̂ 反向推力。"""
        guess = spiral_arc_guess(
            _geo_state(),
            R_LEO_KM,
            M0_KG,
            p0_w=P0_W,
            p_bus_w=P_BUS_W,
            efficiency=ETA,
            isp_s=ISP_S,
        )
        v_hat = np.array(_geo_state()[3:6]) / V_GEO_KM_S
        np.testing.assert_allclose(guess.arcs.arcs[0].direction, -v_hat, atol=1e-12)

    def test_t_start_offset(self):
        """t_start_s 同时偏移弧与序列时间轴。"""
        guess = spiral_arc_guess(
            _leo_state(),
            R_GEO_KM,
            M0_KG,
            p0_w=P0_W,
            p_bus_w=P_BUS_W,
            efficiency=ETA,
            isp_s=ISP_S,
            t_start_s=1000.0,
        )
        assert guess.arcs.arcs[0].t_start == 1000.0
        assert guess.arcs.t_start == 1000.0

    @pytest.mark.parametrize(
        ("state", "target_radius_km", "p0_w", "p_bus_w"),
        [
            ([R_LEO_KM, 0.0, 0.0, 0.0, 0.0, 0.0], R_GEO_KM, P0_W, P_BUS_W),
            ([R_LEO_KM, 0.0, 0.0, 0.0, math.nan, 0.0], R_GEO_KM, P0_W, P_BUS_W),
            ([R_LEO_KM, 0.0, 0.0, 0.0, V_LEO_KM_S], R_GEO_KM, P0_W, P_BUS_W),
            (_leo_state(), R_LEO_KM, P0_W, P_BUS_W),
            (_leo_state(), math.nan, P0_W, P_BUS_W),
            (_leo_state(), 0.0, P0_W, P_BUS_W),
            (_leo_state(), R_GEO_KM, 100.0, 1000.0),
        ],
    )
    def test_guess_validation_errors(self, state, target_radius_km, p0_w, p_bus_w):
        """零初速/非有限状态/形状错/同半径目标/非法目标半径/零可用功率即报错。"""
        with pytest.raises(ValueError):
            spiral_arc_guess(
                state,
                target_radius_km,
                M0_KG,
                p0_w=p0_w,
                p_bus_w=p_bus_w,
                efficiency=ETA,
                isp_s=ISP_S,
            )

    @pytest.mark.parametrize("m0_kg", [0.0, -10.0, math.inf])
    def test_guess_validation_mass(self, m0_kg):
        """初质量非正/非有限即报错。"""
        with pytest.raises(ValueError):
            spiral_arc_guess(
                _leo_state(),
                R_GEO_KM,
                m0_kg,
                p0_w=P0_W,
                p_bus_w=P_BUS_W,
                efficiency=ETA,
                isp_s=ISP_S,
            )
