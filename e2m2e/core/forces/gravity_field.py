"""球谐重力场力模型。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from e2m2e.core.coordinate_system import CoordinateSystem
from e2m2e.core.standard_axes import ITRFApproxAxes
from e2m2e.core.standard_origins import CelestialBodyOrigin

from .exceptions import CoordinateTransformError
from .gravity_file import load_gfc_file
from .physical_model import PhysicalModel


_DEFAULT_SPICE_FRAME = "ITRF93"


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
        input_frame: str = _DEFAULT_SPICE_FRAME,
    ) -> None:
        """初始化 GravityField。

        Args:
            body: 中心天体名称，如 ``'EARTH'``。
            degree: 最大 degree，默认 2。
            order: 最大 order，默认等于 degree。
            gravity_file: 自定义 .gfc 文件路径，默认使用包内 EGM96-to10。
            input_frame: 球谐展开坐标系的 SPICE frame 名，默认 ITRF93。
        """
        self._body = body.upper()
        self._input_frame = input_frame
        self._degree = int(degree)
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
            self._data = load_gfc_file(
                gravity_file, requested_degree=self._degree
            )

        # Trim to requested order
        for n in range(self._degree + 1):
            max_m = min(n, self._order)
            self._data.C[n, max_m + 1 :] = 0.0
            self._data.S[n, max_m + 1 :] = 0.0

    def _load_default_file(self, requested_degree: int) -> None:
        """加载包内默认 EGM96-to10 .gfc 文件。"""
        from importlib import resources

        ref = resources.files("e2m2e.core.forces.data").joinpath("egm96_to10.gfc")
        with ref.open("r", encoding="utf-8") as f:
            # load_gfc_file expects a path; write to a temporary file
            from tempfile import NamedTemporaryFile

            content = f.read()
        with NamedTemporaryFile(mode="w", suffix=".gfc", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            self._data = load_gfc_file(tmp_path, requested_degree=requested_degree)
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

    def compute_acceleration(
        self,
        t: float,
        state: npt.ArrayLike,
        system: Any,
    ) -> npt.NDArray[np.floating]:
        """计算引力加速度。

        Args:
            t: SPICE et 时间。
            state: 状态向量，在 system.coordinate_system 下。
            system: 动力学系统；若传入 None，则假设状态已在 input_frame 下
                （仅用于隔离测试）。

        Returns:
            加速度向量，在 system.coordinate_system 下。
        """
        state_arr = np.asarray(state, dtype=float)
        if state_arr.shape[0] < 3:
            raise ValueError("state must have at least 3 elements")

        if system is None:
            r_input = state_arr[:3].copy()
        else:
            r_input = self._transform_position_to_input_frame(t, state_arr, system)

        acc_input = self._compute_acceleration_in_input_frame(r_input)

        if system is None:
            return acc_input

        return self._transform_vector_from_input_frame(t, acc_input, system)

    def _compute_acceleration_in_input_frame(
        self, r: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """在输入坐标系（固连系）中计算引力加速度。"""
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
            alpha = np.sqrt(
                (2.0 * n + 1.0) * (2.0 * n - 1.0) / (n * n)
            )
            beta = np.sqrt(
                (2.0 * n + 1.0) * (n - 1.0) * (n - 1.0)
                / (n * n * (2.0 * n - 3.0))
            )
            P[n, 0] = alpha * s * P[n - 1, 0] - beta * P[n - 2, 0]
        for m in range(1, n_max + 1):
            # Diagonal recurrence (n == m, n >= 2)
            if m + 1 <= n_max:
                # Sub-diagonal
                P[m + 1, m] = s * np.sqrt(2.0 * m + 3.0) * P[m, m]
            for n in range(m + 2, n_max + 1):
                alpha = np.sqrt(
                    (2.0 * n + 1.0) * (2.0 * n - 1.0)
                    / ((n + m) * (n - m))
                )
                beta = np.sqrt(
                    (2.0 * n + 1.0) * (n + m - 1.0) * (n - m - 1.0)
                    / ((n + m) * (n - m) * (2.0 * n - 3.0))
                )
                P[n, m] = alpha * s * P[n - 1, m] - beta * P[n - 2, m]
            # Next diagonal
            if m + 1 <= n_max:
                P[m + 1, m + 1] = (
                    -u
                    * np.sqrt((2.0 * m + 3.0) / (2.0 * m + 2.0))
                    * P[m, m]
                )

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
                c_val = self._data.C[n, m]
                s_val = self._data.S[n, m]
                if c_val == 0.0 and s_val == 0.0:
                    continue
                cm = np.cos(m * lon)
                sm = np.sin(m * lon)
                cs = c_val * cm + s_val * sm
                dUdr += rho_n * (n + 1.0) * P[n, m] * cs
                dUdphi += rho_n * dP[n, m] * cs
                dUdlambda += rho_n * m * P[n, m] * (-c_val * sm + s_val * cm)

        mu = self._data.mu
        dUdr = -mu / (r_norm * r_norm) * dUdr
        dUdphi = mu / r_norm * dUdphi
        dUdlambda = mu / r_norm * dUdlambda

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
        """把输入坐标系中的矢量转换回传播坐标系。"""
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
        """构造输入坐标系（固连系）。"""
        spice = getattr(system, "spice", None)
        if spice is None:
            raise CoordinateTransformError(
                "system must expose a 'spice' attribute for ITRF transforms"
            )
        axes = ITRFApproxAxes()
        origin = CelestialBodyOrigin(body=self._body, spice=spice)
        return CoordinateSystem(axes=axes, origin=origin)
