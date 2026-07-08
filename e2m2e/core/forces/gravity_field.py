"""球谐重力场力模型。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from e2m2e.core.coordinate_system import CoordinateSystem
from e2m2e.core.standard_axes import ITRFSpiceAxes
from e2m2e.core.standard_origins import CelestialBodyOrigin

from .earth_tide import (
    _K_EARTH,
    _K_PLUS_EARTH,
    load_love_number_file,
    permanent_tide_correction,
    pole_tide,
    solid_tide_step1,
    solid_tide_step2,
)
from .exceptions import CoordinateTransformError
from .gravity_file import extrapolate_coefficients, load_gravity_file
from .physical_model import PhysicalModel

# 按中心天体的默认 body-fixed SPICE frame。
# 地球用 ITRF93(由 earth_*.bpc 定义),月球用 MOON_PA(DE421 principal axes,
# 由 SPICELunaFrameKernel.tf + SPICELunaCurrentKernel.bpc 定义)。
_DEFAULT_FRAME_BY_BODY: dict[str, str] = {
    "EARTH": "ITRF93",
    "MOON": "MOON_PA",
}
# 按中心天体的默认重力场文件名(相对于 e2m2e.core.forces.data)。
_DEFAULT_FILE_BY_BODY: dict[str, str] = {
    "EARTH": "egm96_to10.gfc",
    "MOON": "grgm900c.cof",
}
# 按中心天体的默认 Love 数文件名(None 表示用硬编码常量,如地球)。
_DEFAULT_TIDE_FILE_BY_BODY: dict[str, str | None] = {
    "EARTH": None,
    "MOON": "grgm900c.tide",
}
# Sun/Moon 平均半长轴(永久潮汐修正用,近似)
_A_SUN_KM = 1.495978707e8
_A_MOON_KM = 384400.0

# 每个中心天体做固体潮时的扰动体列表(body 名)。地球受 Sun+Moon 扰动,
# 月球受 Earth 扰动(GMAT 行为)。
_PERTURBERS_BY_BODY: dict[str, list[str]] = {
    "EARTH": ["SUN", "MOON"],
    "MOON": ["EARTH"],
}
# 查扰动体位置时用到的观察者(中心天体)映射,与 _PERTURBERS_BY_BODY 对齐。
# get_body_position(target, et, frame, observer) 的 observer 即中心天体。


class GravityField(PhysicalModel):
    """球谐重力场模型。

    在指定的固连坐标系（默认 ITRF93）中展开球谐级数，计算引力加速度。
    力模型接口约定：输入状态与输出加速度均在 ``system.coordinate_system``
    下；本类内部负责转换到输入坐标系并转回。
    """

    def __init__(
        self,
        body: str,
        degree: int = 2,
        order: int | None = None,
        gravity_file: str | Path | None = None,
        input_frame: str | None = None,
        tide_mode: str = "none",
        tide_convention: str = "tide_free",
        epoch: float | None = None,
        polar_motion_provider: Callable[[float], tuple[float, float]] | None = None,
    ) -> None:
        """初始化 GravityField。

        Args:
            body: 中心天体名称,如 ``'EARTH'``、``'MOON'``。
            degree: 最大 degree,默认 2。
            order: 最大 order,默认等于 degree。
            gravity_file: 自定义重力场文件路径（.gfc 或 .cof）；``None`` 时按
                ``body`` 取包内默认文件（地球 EGM96-to10，月球 GRGM900C）。
            input_frame: 球谐展开坐标系的 SPICE frame 名。``None`` 时按 ``body``
                推导：地球 ``ITRF93``、月球 ``MOON_PA``；其它天体需显式提供。
            tide_mode: 潮汐档位，对齐 GMAT ``ETide`` 三档：
                ``"none"`` （无潮汐）、``"solid"`` （固体潮 Step1+Step2）、
                ``"solid_and_pole"`` （固体潮 + 极潮）。
            tide_convention: 系数约定，``"tide_free"`` 或 ``"zero_tide"`` 。
                zero_tide 模式减去永久潮汐（系数已含永久分量）。
            epoch: dot 项（系数长期变化率）外推的参考历元（SPICE et 秒）。
                与 .gfc 的 dot 行配合；``None`` 表示不外推。
            polar_motion_provider: 极潮 xp/yp 提供者，签名 ``(et) -> (xp, yp)``
                （arcsec）。solid_and_pole 档必需；由调用方从 ``gmat_eop`` 注入。
        """
        self._body = body.upper()
        self._input_frame = self._resolve_input_frame(self._body, input_frame)
        self._gravity_file_arg = gravity_file
        self._degree = int(degree)
        tide_mode_normalized = tide_mode.lower()
        if tide_mode_normalized not in ("none", "solid", "solid_and_pole"):
            raise ValueError(f"tide_mode must be none/solid/solid_and_pole, got {tide_mode!r}")
        self._tide_mode = tide_mode_normalized
        tide_convention_normalized = tide_convention.lower()
        if tide_convention_normalized not in ("tide_free", "zero_tide"):
            raise ValueError(
                f"tide_convention must be tide_free/zero_tide, got {tide_convention!r}"
            )
        self._tide_convention = tide_convention_normalized
        self._epoch = epoch
        self._polar_motion_provider = polar_motion_provider
        if self._degree < 0:
            raise ValueError("degree must be non-negative")

        if order is None:
            self._order = self._degree
        else:
            self._order = int(order)
        if self._order < 0 or self._order > self._degree:
            raise ValueError("order must satisfy 0 <= order <= degree")

        if gravity_file is None:
            self._load_default_file(requested_degree=self._degree)
        else:
            self._data = load_gravity_file(gravity_file, requested_degree=self._degree)

        # Trim to requested order
        for n in range(self._degree + 1):
            max_m = min(n, self._order)
            self._data.C[n, max_m + 1 :] = 0.0
            self._data.S[n, max_m + 1 :] = 0.0

    @staticmethod
    def _resolve_input_frame(body: str, input_frame: str | None) -> str:
        """按 body 推导默认 input_frame,显式传入则覆盖。"""
        if input_frame is not None:
            return input_frame
        if body in _DEFAULT_FRAME_BY_BODY:
            return _DEFAULT_FRAME_BY_BODY[body]
        raise ValueError(
            f"No default body-fixed frame for body={body!r}; pass input_frame explicitly"
        )

    def _load_default_file(self, requested_degree: int) -> None:
        """按 ``body`` 加载包内默认重力场文件(地球 EGM96-to10，月球 GRGM900C)。"""
        filename = _DEFAULT_FILE_BY_BODY.get(self._body)
        if filename is None:
            raise ValueError(
                f"No default gravity file for body={self._body!r}; "
                "pass gravity_file explicitly"
            )
        from importlib import resources

        ref = resources.files("e2m2e.core.forces.data").joinpath(filename)
        suffix = Path(filename).suffix
        with ref.open("r", encoding="utf-8") as f:
            # load_gravity_file expects a path; write to a temporary file
            from tempfile import NamedTemporaryFile

            content = f.read()
        with NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            self._data = load_gravity_file(tmp_path, requested_degree=requested_degree)
        finally:
            Path(tmp_path).unlink()

    @property
    def body(self) -> str:
        """中心天体名称。"""
        return self._body

    @property
    def degree(self) -> int:
        """最大 degree。"""
        return self._degree

    @property
    def order(self) -> int:
        """最大 order。"""
        return self._order

    @property
    def input_frame(self) -> str:
        """球谐展开坐标系的 SPICE frame 名。"""
        return self._input_frame

    @property
    def gravity_file(self) -> str | Path | None:
        """用户传入的自定义 .gfc 路径；``None`` 表示用包内默认 EGM96。"""
        return self._gravity_file_arg

    @property
    def gravitational_parameter(self) -> float:
        """引力参数 GM。"""
        return self._data.mu

    @property
    def reference_radius(self) -> float:
        """参考半径 R_e。"""
        return self._data.radius

    @property
    def coefficients(self) -> dict[str, npt.NDArray[np.floating]]:
        """正规化系数副本。"""
        return {"C": self._data.C.copy(), "S": self._data.S.copy()}

    @property
    def tide_mode(self) -> str:
        """潮汐档位。"""
        return self._tide_mode

    @property
    def tide_convention(self) -> str:
        """系数约定。"""
        return self._tide_convention

    def _effective_coefficients(
        self, t: float, system: Any
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """返回 t 时刻的有效 C/S(含 dot 外推 + 潮汐修正)。

        潮汐路径按中心天体分流:
        - ``EARTH``:扰动体 [Sun, Moon],Love 数取硬编码 ``_K_EARTH``/
          ``_K_PLUS_EARTH``;Step1 后追加地球专用 Step2/极潮/永久潮。与重构前
          行为逐字一致(回归验证 atol=1e-12)。
        - ``MOON``:扰动体 [Earth](地球相对月球的位置),Love 数从
          ``grgm900c.tide`` 读(k₂=0.024116);只做 Step1,不做 Step2/极潮。

        Args:
            t: SPICE et 秒。
            system: 动力学系统;tide_mode 非 none 时需暴露 ``spice`` 属性以查
                扰动体位置与 GM。

        Returns:
            (C_eff, S_eff),形状与 ``self._data.C`` 一致。
        """
        C = self._data.C.copy()
        S = self._data.S.copy()

        # dot 项历元外推
        if self._epoch is not None and np.any(self._data.dotC):
            C, S = extrapolate_coefficients(C, S, self._data.dotC, self._data.dotS, t, self._epoch)

        if self._tide_mode == "none":
            return C, S

        spice = getattr(system, "spice", None) if system is not None else None
        if spice is None:
            raise CoordinateTransformError(
                "tide_mode != 'none' requires system.spice for perturber ephemeris"
            )

        # 查扰动体位置/GM(observer 为中心天体)。位置在中心天体 body-fixed 系下。
        perturber_names = _PERTURBERS_BY_BODY.get(self._body)
        if perturber_names is None:
            raise CoordinateTransformError(
                f"solid tide not configured for body={self._body!r}"
            )
        perturbers = [
            (
                spice.get_body_position(name, t, self._input_frame, self._body),
                spice.get_gm(name),
            )
            for name in perturber_names
        ]

        mu_central = self._data.mu
        r_central = self._data.radius

        # 固体潮 Step 1(天体无关:扰动体列表 + 该天体的 Love 数表)
        k_love, k_plus = self._resolve_love_numbers()
        dC, dS = solid_tide_step1(
            perturbers,
            k_love=k_love,
            k_plus=k_plus,
            mu_central=mu_central,
            r_central=r_central,
        )

        # 地球专用:Step2(频率相关)+ 极潮 + 永久潮汐修正
        if self._body == "EARTH":
            # 固体潮 Step 2(频率相关,Delaunay 幅角)
            dC2, dS2 = solid_tide_step2(t)
            dC = dC + dC2
            dS = dS + dS2

            # zero-tide:减去永久潮汐(系数已含永久分量)
            if self._tide_convention == "zero_tide":
                mu_sun = spice.get_gm("SUN")
                mu_moon = spice.get_gm("MOON")
                dC_perm, dS_perm = permanent_tide_correction(
                    mu_sun, mu_moon, mu_central, r_central, _A_SUN_KM, _A_MOON_KM
                )
                dC = dC - dC_perm
                dS = dS - dS_perm

            # 极档:叠加极潮
            if self._tide_mode == "solid_and_pole":
                if self._polar_motion_provider is None:
                    raise ValueError(
                        "tide_mode='solid_and_pole' requires polar_motion_provider"
                    )
                xp, yp = self._polar_motion_provider(t)
                dC_pole, dS_pole = pole_tide(t, xp, yp)
                dC = dC + dC_pole
                dS = dS + dS_pole

        # pad ΔC/ΔS(5×5)到 C/S 形状(degree 可能 > 4)
        n = min(5, C.shape[0])
        dC_padded = np.zeros_like(C)
        dS_padded = np.zeros_like(S)
        dC_padded[:n, :n] = dC[:n, :n]
        dS_padded[:n, :n] = dS[:n, :n]
        return C + dC_padded, S + dS_padded

    def _resolve_love_numbers(
        self,
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating] | None]:
        """按 ``body`` 解析固体潮 Love 数表 (k_love, k_plus)。

        地球用硬编码常量(``_K_EARTH`` / ``_K_PLUS_EARTH``);月球从包内
        ``grgm900c.tide`` 读取(仅 k₂=0.024116,无弹性 3 阶位移 → k_plus=None)。
        """
        if self._body == "EARTH":
            return _K_EARTH, _K_PLUS_EARTH
        tide_file = _DEFAULT_TIDE_FILE_BY_BODY.get(self._body)
        if tide_file is None:
            raise CoordinateTransformError(
                f"no Love number source for body={self._body!r}"
            )
        from importlib import resources

        ref = resources.files("e2m2e.core.forces.data").joinpath(tide_file)
        with ref.open("r", encoding="utf-8") as f:
            content = f.read()
        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(mode="w", suffix=".tide", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            k_love = load_love_number_file(tmp_path)
        finally:
            Path(tmp_path).unlink()
        # 月球等无弹性 3 阶位移贡献。
        return k_love, None

    def compute_acceleration(
        self,
        t: float,
        state: npt.ArrayLike,
        system: Any,
    ) -> npt.NDArray[np.floating]:
        """计算引力加速度。

        Args:
            t: SPICE et 时间。
            state: 状态向量,在 system.coordinate_system 下。
            system: 动力学系统;若传入 None,则假设状态已在 input_frame 下
                (仅用于隔离测试,tide_mode 必须为 none)。

        Returns:
            加速度向量,在 system.coordinate_system 下。
        """
        state_arr = np.asarray(state, dtype=float)
        if state_arr.shape[0] < 3:
            raise ValueError("state must have at least 3 elements")

        if system is None:
            r_input = state_arr[:3].copy()
        else:
            r_input = self._transform_position_to_input_frame(t, state_arr, system)

        # 潮汐/dot 外推是时变的,每步重算有效系数
        needs_effective = self._tide_mode != "none" or (
            self._epoch is not None and np.any(self._data.dotC)
        )
        if needs_effective:
            C_eff, S_eff = self._effective_coefficients(t, system)
        else:
            C_eff, S_eff = None, None

        acc_input = self._compute_acceleration_in_input_frame(r_input, C_eff, S_eff)

        if system is None:
            return acc_input

        return self._transform_vector_from_input_frame(t, acc_input, system)

    def _compute_acceleration_in_input_frame(
        self,
        r: npt.NDArray[np.floating],
        C: npt.NDArray[np.floating] | None = None,
        S: npt.NDArray[np.floating] | None = None,
    ) -> npt.NDArray[np.floating]:
        """在输入坐标系(固连系)中计算引力加速度。

        Args:
            r: 位置,在 input_frame 下。
            C, S: 可选的有效球谐系数(含潮汐/dot);``None`` 用文件原始系数。
        """
        if C is None:
            C = self._data.C
        if S is None:
            S = self._data.S
        x, y, z = r
        r_norm = np.linalg.norm(r)
        if r_norm == 0:
            raise ValueError("Cannot compute gravity at the origin")

        rho = self._data.radius / r_norm
        s = z / r_norm  # sin(phi), phi = geocentric latitude
        u = np.sqrt(max(0.0, 1.0 - s * s))  # cos(phi)
        lon = np.arctan2(y, x)

        n_max = self._degree
        m_max = self._order

        # Associated Legendre functions, fully normalized (physics convention)
        P = np.zeros((n_max + 1, n_max + 1))
        if n_max >= 0:
            P[0, 0] = 1.0
        if n_max >= 1:
            P[1, 0] = np.sqrt(3.0) * s
            P[1, 1] = -np.sqrt(3.0) * u
        for n in range(2, n_max + 1):
            # Vertical recurrence for m = 0
            alpha = np.sqrt((2.0 * n + 1.0) * (2.0 * n - 1.0) / (n * n))
            beta = np.sqrt((2.0 * n + 1.0) * (n - 1.0) * (n - 1.0) / (n * n * (2.0 * n - 3.0)))
            P[n, 0] = alpha * s * P[n - 1, 0] - beta * P[n - 2, 0]
        for m in range(1, n_max + 1):
            # Diagonal recurrence (n == m, n >= 2)
            if m + 1 <= n_max:
                # Sub-diagonal
                P[m + 1, m] = s * np.sqrt(2.0 * m + 3.0) * P[m, m]
            for n in range(m + 2, n_max + 1):
                alpha = np.sqrt((2.0 * n + 1.0) * (2.0 * n - 1.0) / ((n + m) * (n - m)))
                beta = np.sqrt(
                    (2.0 * n + 1.0)
                    * (n + m - 1.0)
                    * (n - m - 1.0)
                    / ((n + m) * (n - m) * (2.0 * n - 3.0))
                )
                P[n, m] = alpha * s * P[n - 1, m] - beta * P[n - 2, m]
            # Next diagonal
            if m + 1 <= n_max:
                P[m + 1, m + 1] = -u * np.sqrt((2.0 * m + 3.0) / (2.0 * m + 2.0)) * P[m, m]

        # Derivatives with respect to latitude phi
        dP = np.zeros((n_max + 1, n_max + 1))
        for n in range(1, n_max + 1):
            dP[n, 0] = -np.sqrt(n * (n + 1.0) / 2.0) * P[n, 1]
        for m in range(1, n_max + 1):
            for n in range(m + 1, n_max + 1):
                term1 = np.sqrt((n + m) * (n - m + 1.0)) * P[n, m - 1]
                term2 = np.sqrt((n + m + 1.0) * (n - m)) * P[n, m + 1]
                dP[n, m] = 0.5 * (term1 - term2)

        # Spherical coordinate acceleration components
        dUdr = 0.0
        dUdphi = 0.0
        dUdlambda = 0.0

        for n in range(0, n_max + 1):
            rho_n = rho**n
            for m in range(0, min(n, m_max) + 1):
                c_val = C[n, m]
                s_val = S[n, m]
                if c_val == 0.0 and s_val == 0.0:
                    continue
                cm = np.cos(m * lon)
                sm = np.sin(m * lon)
                cs = c_val * cm + s_val * sm
                dUdr += rho_n * (n + 1.0) * P[n, m] * cs
                dUdphi += rho_n * dP[n, m] * cs
                dUdlambda += rho_n * m * P[n, m] * (-c_val * sm + s_val * cm)

        mu = self._data.mu
        dUdr = float(-mu / (r_norm * r_norm) * dUdr)
        dUdphi = float(mu / r_norm * dUdphi)
        dUdlambda = float(mu / r_norm * dUdlambda)

        # Convert to Cartesian (ITRF)
        cos_lon = np.cos(lon)
        sin_lon = np.sin(lon)
        cos_phi = u
        sin_phi = s

        a_r = dUdr
        a_phi = dUdphi / r_norm
        a_lambda = dUdlambda / (r_norm * cos_phi)

        ax = a_r * cos_phi * cos_lon - a_phi * sin_phi * cos_lon - a_lambda * sin_lon
        ay = a_r * cos_phi * sin_lon - a_phi * sin_phi * sin_lon + a_lambda * cos_lon
        az = a_r * sin_phi + a_phi * cos_phi

        return np.array([ax, ay, az])

    def _transform_position_to_input_frame(
        self, t: float, state: npt.NDArray[np.floating], system: Any
    ) -> npt.NDArray[np.floating]:
        """把状态位置转换到输入坐标系。"""
        input_cs = self._get_input_coordinate_system(system)
        try:
            state_input = system.coordinate_system.transform_state(
                state, from_cs=system.coordinate_system, to_cs=input_cs, et=t
            )
        except Exception as exc:
            raise CoordinateTransformError(
                f"Failed to transform state to {self._input_frame}"
            ) from exc
        return state_input[:3]

    def _transform_vector_from_input_frame(
        self,
        t: float,
        vector: npt.NDArray[np.floating],
        system: Any,
    ) -> npt.NDArray[np.floating]:
        """把输入坐标系中的矢量转换回参考系。"""
        input_cs = self._get_input_coordinate_system(system)
        try:
            return system.coordinate_system.transform_vector(
                vector, from_cs=input_cs, to_cs=system.coordinate_system, et=t
            )
        except Exception as exc:
            raise CoordinateTransformError(
                f"Failed to transform acceleration from {self._input_frame}"
            ) from exc

    def _get_input_coordinate_system(self, system: Any) -> CoordinateSystem:
        """构造输入坐标系（固连系）。

        按 ``self._input_frame``（默认按 ``body`` 推导）构造 SPICE-backed 坐标轴:
        地球 ``ITRF93``、月球 ``MOON_PA``。
        """
        spice = getattr(system, "spice", None)
        if spice is None:
            raise CoordinateTransformError(
                "system must expose a 'spice' attribute for body-fixed transforms"
            )
        axes = ITRFSpiceAxes(frame=self._input_frame)
        origin = CelestialBodyOrigin(body=self._body, spice=spice)
        return CoordinateSystem(axes=axes, origin=origin)
