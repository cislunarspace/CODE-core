"""球谐重力场力模型。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ..coordinate.coordinate_system import CoordinateSystem
from ..coordinate.standard_axes import ITRFSpiceAxes
from ..coordinate.standard_origins import CelestialBodyOrigin
from .earth_tide import _K_EARTH, _K_PLUS_EARTH, load_love_number_file
from .exceptions import CoordinateTransformError
from .gravity_file import load_gravity_file
from .physical_model import PhysicalModel

# 按中心天体的默认 body-fixed SPICE frame。
# 地球用 ITRF93(由 earth_*.bpc 定义),月球用 MOON_PA(DE421 principal axes,
# 由 SPICELunaFrameKernel.tf + SPICELunaCurrentKernel.bpc 定义)。
_DEFAULT_FRAME_BY_BODY: dict[str, str] = {
    "EARTH": "ITRF93",
    "MOON": "MOON_PA",
}
# 按中心天体的默认重力场文件名(相对于 e2m2e.algorithm.forces.data)。
_DEFAULT_FILE_BY_BODY: dict[str, str] = {
    "EARTH": "egm96_to10.gfc",
    "MOON": "grgm900c.cof",
}
# 按中心天体的默认 Love 数文件名(None 表示用硬编码常量,如地球)。
_DEFAULT_TIDE_FILE_BY_BODY: dict[str, str | None] = {
    "EARTH": None,
    "MOON": "grgm900c.tide",
}


class GravityField(PhysicalModel):
    """球谐重力场模型。

    在指定的固连坐标系（默认 ITRF93）中展开球谐级数，计算引力加速度。
    加速度计算全部由 Rust 编译路径承载（``("gravity", ...)`` 力元组，
    ``crates/e2m2e-forces/src/forces/gravity_field.rs``，含潮汐），Python 侧
    不保留参考实现（issue #378）。
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

        # Love 数是常数，构造时一次性解析缓存。先前 _resolve_love_numbers 每步
        # 重读文件 + 写临时文件 + 解析，是满配直推 22% 耗时的根源（profile 定位）。
        self._love_cache: (
            tuple[npt.NDArray[np.floating], npt.NDArray[np.floating] | None] | None
        ) = None
        if self._tide_mode != "none":
            self._love_cache = self._load_love_numbers()

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
                f"No default gravity file for body={self._body!r}; pass gravity_file explicitly"
            )
        from importlib import resources

        ref = resources.files("e2m2e.algorithm.forces.data").joinpath(filename)
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

    def to_rust_spec(self, system: Any) -> tuple | None:
        """序列化为 Rust propagate_compiled 的 ``("gravity", ...)`` 元组。

        SolidAndPole 档暂不支持（需外部 xp/yp provider），返回 ``None``
        让 ForceModel 回退 Python 路径。
        """
        if self._tide_mode == "solid_and_pole":
            return None
        # tide_mode → 整数（与 Rust TideMode enum 一致）
        tide_int = {"none": 0, "solid": 1}[self._tide_mode]
        # Love 数（tide_mode=none 时仍传占位 zeros）
        if self._love_cache is not None:
            k_love, k_plus = self._love_cache
            k_love_flat = k_love.ravel().tolist()
            k_plus_flat = k_plus.ravel().tolist() if k_plus is not None else None
        else:
            k_love_flat = [0.0] * 25
            k_plus_flat = None
        # 传播系 origin（通常 "EARTH"）
        propagation_origin = getattr(system, "origin", "EARTH")
        return (
            "gravity",
            self._data.C.ravel(order="C").tolist(),
            self._data.S.ravel(order="C").tolist(),
            float(self._data.mu),
            float(self._data.radius),
            int(self._degree),
            int(self._order),
            self._input_frame,
            "J2000",
            self._body,
            propagation_origin,
            tide_int,
            k_love_flat,
            k_plus_flat,
        )

    def _load_love_numbers(
        self,
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating] | None]:
        """按中心天体加载固体潮 Love 数表。"""
        if self._body == "EARTH":
            return _K_EARTH, _K_PLUS_EARTH
        tide_file = _DEFAULT_TIDE_FILE_BY_BODY.get(self._body)
        if tide_file is None:
            raise CoordinateTransformError(f"no Love number source for body={self._body!r}")
        from importlib import resources
        from tempfile import NamedTemporaryFile

        ref = resources.files("e2m2e.algorithm.forces.data").joinpath(tide_file)
        with ref.open("r", encoding="utf-8") as f:
            content = f.read()
        with NamedTemporaryFile(mode="w", suffix=".tide", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            k_love = load_love_number_file(tmp_path)
        finally:
            Path(tmp_path).unlink()
        return k_love, None

    @property
    def tide_convention(self) -> str:
        """系数约定。"""
        return self._tide_convention

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
