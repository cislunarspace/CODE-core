"""标准指数大气密度模型。"""

from __future__ import annotations

import numpy as np

# US Standard Atmosphere 1976 断点密度（kg/m³）。
# 数据来源：USSA76 标准大气表，覆盖 0-1000 km。
_USS76_BREAKPOINTS: tuple[tuple[float, float], ...] = (
    (0.0, 1.225e0),
    (25.0, 4.048e-2),
    (50.0, 1.057e-3),
    (75.0, 3.313e-5),
    (100.0, 5.604e-7),
    (150.0, 2.384e-9),
    (200.0, 2.541e-10),
    (300.0, 1.916e-11),
    (400.0, 2.803e-12),
    (500.0, 5.215e-13),
    (600.0, 1.189e-13),
    (700.0, 3.381e-14),
    (800.0, 1.137e-14),
    (900.0, 4.390e-15),
    (1000.0, 1.879e-15),
)


def _build_layers(
    breakpoints: tuple[tuple[float, float], ...],
) -> list[tuple[float, float, float]]:
    """从断点密度推导自洽标高，构造连续分段层表。

    每层标高 ``H = Δh / ln(ρ₀/ρ₁)`` 确保层间密度连续且单调递减。
    """
    layers: list[tuple[float, float, float]] = []
    for i in range(len(breakpoints) - 1):
        h0, rho0 = breakpoints[i]
        h1, rho1 = breakpoints[i + 1]
        scale_height = (h1 - h0) / np.log(rho0 / rho1)
        layers.append((h0, rho0, scale_height))
    return layers


# 自洽分段层表：(基准高度 km, 基准密度 kg/m³, 标高 km)。
_LAYERS: list[tuple[float, float, float]] = _build_layers(_USS76_BREAKPOINTS)

_CEILING_ALTITUDE_KM = _USS76_BREAKPOINTS[-1][0]
_DEFAULT_F107 = 150.0
_DEFAULT_AP = 15.0
_F107_SENSITIVITY = 0.5
_AP_SENSITIVITY = 0.1


class ExponentialAtmosphere:
    """US Standard Atmosphere 1976 分段指数大气密度模型。

    在每个高度层内使用 ``ρ(h) = ρ₀ · exp(-(h - h₀) / H)`` 计算密度。
    层间标高由相邻断点密度比推导，确保密度连续且单调递减。
    F10.7 太阳射电通量和 Ap 地磁指数通过线性乘法因子对基准密度做一阶修正。

    Args:
        f107: F10.7 太阳射电通量（sfu），默认 150（中等太阳活动）。
        ap: Ap 地磁指数，默认 15（中等地磁活动）。
    """

    def __init__(self, f107: float = _DEFAULT_F107, ap: float = _DEFAULT_AP) -> None:
        self._f107 = float(f107)
        self._ap = float(ap)

    @property
    def f107(self) -> float:
        """F10.7 太阳射电通量（sfu）。"""
        return self._f107

    @property
    def ap(self) -> float:
        """Ap 地磁指数。"""
        return self._ap

    def density(self, altitude: float) -> float:
        """返回指定高度处的大气密度。

        高度超出模型范围时：高于 1000 km 返回 0（阻力可忽略），
        低于 0 km 钳到 0 km（用地表密度，避免负高度导致 exp 爆炸）。

        Args:
            altitude: 几何高度，单位 km。

        Returns:
            大气密度，单位 kg/m³。
        """
        h = max(0.0, float(altitude))
        if h >= _CEILING_ALTITUDE_KM:
            return 0.0

        h0, rho0, scale_height = _lookup_layer(h)
        rho_ref = rho0 * np.exp(-(h - h0) / scale_height)
        return rho_ref * _solar_activity_factor(self._f107, self._ap)


def _lookup_layer(altitude: float) -> tuple[float, float, float]:
    """查找包含给定高度的层参数。低于 0 km 钳到海平面层。"""
    h = max(0.0, altitude)
    h0, rho0, scale_height = _LAYERS[0]
    for layer_h0, layer_rho0, layer_h in _LAYERS:
        if h >= layer_h0:
            h0, rho0, scale_height = layer_h0, layer_rho0, layer_h
        else:
            break
    return h0, rho0, scale_height


def _solar_activity_factor(f107: float, ap: float) -> float:
    """计算 F10.7 和 Ap 的一阶线性密度修正因子。"""
    f_factor = 1.0 + _F107_SENSITIVITY * (f107 - _DEFAULT_F107) / _DEFAULT_F107
    a_factor = 1.0 + _AP_SENSITIVITY * (ap - _DEFAULT_AP) / _DEFAULT_AP
    return f_factor * a_factor
