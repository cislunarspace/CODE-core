"""标准指数大气密度模型。"""

from __future__ import annotations

from collections.abc import Sequence

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

    def density(
        self,
        altitude: float,
        *,
        epoch_et: float | None = None,
        geodetic_lat_deg: float | None = None,
        geodetic_lon_deg: float | None = None,
    ) -> float:
        """返回指定高度处的大气密度。

        高度超出模型范围时：高于 1000 km 返回 0（阻力可忽略），
        低于 0 km 钳到 0 km（用地表密度，避免负高度导致 exp 爆炸）。

        Args:
            altitude: 几何高度，单位 km。
            epoch_et: 历元（SPICE et 秒）。**本模型忽略**，仅为与
                :class:`NRLMSISE00Atmosphere` 统一调用签名而保留。
            geodetic_lat_deg: 大地纬度（deg）。**本模型忽略**，同上。
            geodetic_lon_deg: 大地经度（deg）。**本模型忽略**，同上。

        Returns:
            大气密度，单位 kg/m³。
        """
        del epoch_et, geodetic_lat_deg, geodetic_lon_deg
        h = max(0.0, float(altitude))
        if h >= _CEILING_ALTITUDE_KM:
            return 0.0

        h0, rho0, scale_height = _lookup_layer(h)
        rho_ref = rho0 * np.exp(-(h - h0) / scale_height)
        return rho_ref * _solar_activity_factor(self._f107, self._ap)


# NRLMSISE-00 默认空间天气（中等太阳活动 / 中等地磁活动）。
_NRLMSISE00_DEFAULT_F107 = 150.0
_NRLMSISE00_DEFAULT_AP = 15.0

# Ap 史固定长度：3 小时分辨率（当前、−3h、−6h、−9h、−12..−33h 平均、
# −36..−57h 平均），第 0 元是当日 Ap 标量通道。
_NRLMSISE00_AP_HISTORY_LEN = 7


def _normalize_ap(ap: float | Sequence[float]) -> tuple[float, ...]:
    """把标量或 7 元序列规范化为 7 元 Ap 史（标量广播为全时同值）。"""
    arr = np.asarray(ap, dtype=float)
    if arr.ndim == 0:
        values = (float(arr),) * _NRLMSISE00_AP_HISTORY_LEN
    elif arr.ndim == 1 and arr.size == _NRLMSISE00_AP_HISTORY_LEN:
        values = tuple(float(v) for v in arr)
    else:
        raise ValueError(
            f"ap must be a scalar or a sequence of {_NRLMSISE00_AP_HISTORY_LEN} elements"
        )
    if not bool(np.all(np.isfinite(arr))) or float(np.min(arr)) < 0.0:
        raise ValueError("ap values must be finite and non-negative")
    return values


class NRLMSISE00Atmosphere:
    """NRLMSISE-00 经验大气密度模型（Picone et al. 2002）。

    与 :class:`ExponentialAtmosphere` 的分段指数模型相比，本模型是空间天气
    驱动的经验模型：密度不仅随高度变化，还随历元（年积日 + UT）、WGS84 大地
    经纬度与太阳/地磁活动变化，适用于地球停泊弧段的高保真阻力建模。

    数值实现只在 Rust 侧（``crates/e2m2e-forces/src/nrlmsise00/``）；本类只做
    参数校验与调用编排，经 ``nrlmsise00_density_py`` 查询（ADR 0030）。

    空间天气语义（**调用方自行提供，本模型不接任何外部数据源**）：

    - ``f107_daily``：前一日 10.7 cm 太阳射电通量，单位 sfu（10⁻²² W·m⁻²·Hz⁻¹）。
    - ``f107_avg``：以目标日为心的 81 日滑动平均 F10.7，单位 sfu；
      ``None`` 时取 ``f107_daily``。
    - ``ap``：3 小时分辨率地磁 Ap 指数史，7 元。第 0 元是当日 Ap 标量通道，
      第 1–6 元依次为当前、−3 h、−6 h、−9 h 的 3 小时 Ap，以及 −12..−33 h、
      −36..−57 h 的平均。传标量（或 7 元全等序列）即"全时同值"的静态空间天气，
      模型只用当日 Ap；传 7 元非全等序列则启用 3 小时分辨率的暴时先验。

    高度域与 :class:`ExponentialAtmosphere` 一致：0–1000 km，负高度钳到 0，
    1000 km 及以上返回 0。大地坐标按 WGS84 椭球解释（高度是椭球面以上高度）。
    密度口径是模型 ``gtd7d`` 的"drag 有效总质量密度"（含 500 km 以上不可忽略的
    异常氧贡献）。

    Note:
        查询需要已装载 SPICE 内核。历元 → 年积日/UTC 的换算优先走星历预采样
        缓存（传播打靶/分段积分的并行区因此不跨 cspice，见 ADR 0016/0049）；缓存
        未启用时回退 ``et2utc``，依赖 leap second 内核（``spice_furnsh`` 装载
        LSK）。内核池为空时抛出明确错误，不静默回退。

    Note:
        与 GMAT/参考实现的对拍差异主要来自模型版本与输入映射：GMAT 的
        ``MSISE90`` 是 MSISE-1990 版本，而本模型是 NRLMSISE-00（2000 版本）；
        两者的 F10.7/Ap 预处理（离散 3 小时台阶 vs 连续样条）也不同。
        这些差异属于模型标定口径，不是实现缺陷。

    Args:
        f107_daily: 前一日 F10.7 通量（sfu），默认 150（中等太阳活动）。
        f107_avg: 81 日滑动平均 F10.7（sfu），默认与 ``f107_daily`` 相同。
        ap: 地磁 Ap 指数，标量或 7 元序列，默认 15（中等地磁活动）。

    Raises:
        ValueError: ``f107_daily``/``f107_avg`` 非正，或 ``ap`` 形状错误、
            含非有限值或负值。
    """

    def __init__(
        self,
        f107_daily: float = _NRLMSISE00_DEFAULT_F107,
        f107_avg: float | None = None,
        ap: float | Sequence[float] = _NRLMSISE00_DEFAULT_AP,
    ) -> None:
        self._f107_daily = float(f107_daily)
        self._f107_avg = float(self._f107_daily if f107_avg is None else f107_avg)
        if not np.isfinite(self._f107_daily) or self._f107_daily <= 0.0:
            raise ValueError("f107_daily must be finite and positive")
        if not np.isfinite(self._f107_avg) or self._f107_avg <= 0.0:
            raise ValueError("f107_avg must be finite and positive")
        self._ap = _normalize_ap(ap)

    @property
    def f107_daily(self) -> float:
        """前一日 F10.7 太阳射电通量（sfu）。"""
        return self._f107_daily

    @property
    def f107_avg(self) -> float:
        """81 日滑动平均 F10.7（sfu）。"""
        return self._f107_avg

    @property
    def ap(self) -> tuple[float, ...]:
        """地磁 Ap 指数史，恒为 7 元。"""
        return self._ap

    def density(
        self,
        altitude_km: float,
        *,
        epoch_et: float,
        geodetic_lat_deg: float,
        geodetic_lon_deg: float,
    ) -> float:
        """返回指定历元与大地坐标处的大气密度。

        Args:
            altitude_km: WGS84 椭球面以上大地高度，单位 km。
            epoch_et: 历元（SPICE et 秒），用于定年积日与 UT。
            geodetic_lat_deg: WGS84 大地纬度，单位 deg。
            geodetic_lon_deg: 大地经度（东经为正），单位 deg。

        Returns:
            大气密度，单位 kg/m³。1000 km 及以上返回 0。

        Raises:
            RustExtensionUnavailableError: Rust 扩展或 ``nrlmsise00_density_py``
                符号缺失。
        """
        from e2m2e.integrators import nrlmsise00_density_py, require_rust_extension

        require_rust_extension("nrlmsise00_density_py")
        rho, _temperature_k = nrlmsise00_density_py(
            float(epoch_et),
            float(altitude_km),
            float(geodetic_lat_deg),
            float(geodetic_lon_deg),
            self._f107_daily,
            self._f107_avg,
            list(self._ap),
        )
        return float(rho)


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
