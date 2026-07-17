"""力模型抽象基类。"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np
import numpy.typing as npt

from e2m2e.core.system import System


class PhysicalModel(abc.ABC):
    """物理力模型抽象基类。

    力模型以纯函数接口提供加速度。所有坐标约定都在
    ``system.coordinate_system`` 下完成；需要非默认坐标系计算的子类
    应通过 ``system.coordinate_system.transform_state()`` /
    ``transform_vector()`` 自行完成转换。
    """

    @abc.abstractmethod
    def compute_acceleration(
        self,
        t: float,
        state: npt.ArrayLike,
        system: System,
    ) -> npt.NDArray[np.floating]:
        """返回状态在 ``system.coordinate_system`` 下的加速度。

        Args:
            t: 时间，单位为 SPICE et（秒 past J2000）。
            state: 状态向量，形状至少为 ``(6,)``，前三个元素为位置。
            system: 当前动力学系统，提供坐标系与天体参数。

        Returns:
            加速度向量，形状 ``(3,)``，单位与 ``system.unit_system`` 一致。
        """
        raise NotImplementedError

    def _resolve_mu(self, system: System | None) -> float:
        """返回该力模型的引力参数 μ。

        显式设过 ``self._mu`` 就用它；否则从 ``system.gravitational_parameter``
        查。``system`` 在隔离测试时可为 ``None``（只给 ``mu``、不查星历）；
        当 ``mu`` 与 ``system`` 都缺失时抛 ``ValueError``，这是被
        ``test_*_no_system_no_mu_raises`` 固化的契约。
        """
        if self._mu is not None:
            return self._mu
        if system is None:
            raise ValueError(
                "mu is None and system is None; cannot resolve gravitational_parameter"
            )
        return system.gravitational_parameter(self._body)

    def compute_jacobian(
        self,
        t: float,
        state: npt.ArrayLike,
        system: System,
    ) -> npt.NDArray[np.floating] | None:
        """返回加速度对位置的偏导 ∂a/∂r（3×3）。

        供 STM 变分方程组装雅可比矩阵 A = [[0, I], [∂a/∂r, 0]] 用。
        N 体问题中无速度相关力，故 ∂a/∂v = 0，只需位置块。

        默认返回 ``None``，表示该力不提供解析雅可比，由 ``ForceModel``
        走有限差分兜底（三点中心差分调 ``compute_acceleration``）。

        Args:
            t: 时间，单位为 SPICE et（秒 past J2000）。
            state: 状态向量，形状至少为 ``(6,)``，前三个元素为位置。
            system: 当前动力学系统。

        Returns:
            3×3 雅可比矩阵，或 ``None`` 表示无解析实现。
        """
        return None


def require_inertial_frame(system: Any, t: float) -> tuple[Any, Any, str]:
    """校验参考系为惯性系，返回 (coordinate_system, spice, origin_body)。

    供在传播惯性系（ICRF，轴旋转矩阵为单位阵）中直接计算的力模型调用。
    非惯性系（如 ITRFApproxAxes）抛 ``NotImplementedError``。
    """
    cs = getattr(system, "coordinate_system", None)
    if cs is None:
        raise ValueError("system.coordinate_system is required")
    rotation = np.asarray(cs.axes.rotation_matrix(t), dtype=float)
    if not np.allclose(rotation, np.eye(3), atol=1e-9):
        raise NotImplementedError(
            "force model requires an inertial propagation frame (ICRF); "
            f"got non-identity axes {type(cs.axes).__name__}."
        )
    return cs, system.spice, cs.origin.body
