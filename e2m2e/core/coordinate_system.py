"""坐标系组合与转换。"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .axes import Axes
from .origin import Origin


class CoordinateSystem:
    """由坐标轴和原点组成的参考系。"""

    def __init__(self, axes: Axes, origin: Origin) -> None:
        self.axes = axes
        self.origin = origin

    def transform_vector(
        self,
        vec: npt.ArrayLike,
        from_cs: CoordinateSystem,
        to_cs: CoordinateSystem,
        et: float,
    ) -> npt.NDArray[np.floating]:
        """将三维向量从 ``from_cs`` 转换到 ``to_cs``。"""
        vector = np.asarray(vec, dtype=float)
        from_rotation = from_cs.axes.rotation_matrix(et)
        to_rotation = to_cs.axes.rotation_matrix(et)
        return to_rotation.T @ from_rotation @ vector

    def transform_state(
        self,
        state: npt.ArrayLike,
        from_cs: CoordinateSystem,
        to_cs: CoordinateSystem,
        et: float,
    ) -> npt.NDArray[np.floating]:
        """将六维状态从 ``from_cs`` 转换到 ``to_cs``。

        坐标轴约定为 ``r_icrf = R @ r_axes``，速度使用
        ``v_icrf = R @ v_axes + Rdot @ r_axes``。
        """
        state_array = np.asarray(state, dtype=float)
        position = state_array[:3]
        velocity = state_array[3:]

        from_rotation, from_rate = from_cs.axes.rotation_and_rate(et)
        to_rotation, to_rate = to_cs.axes.rotation_and_rate(et)
        from_origin = from_cs.origin.state(et)
        to_origin = to_cs.origin.state(et)

        position_icrf = from_rotation @ position + from_origin[:3]
        velocity_icrf = from_rotation @ velocity + from_rate @ position + from_origin[3:]

        relative_position = position_icrf - to_origin[:3]
        position_out = to_rotation.T @ relative_position
        velocity_out = to_rotation.T @ (velocity_icrf - to_origin[3:] - to_rate @ position_out)

        return np.concatenate([position_out, velocity_out])
