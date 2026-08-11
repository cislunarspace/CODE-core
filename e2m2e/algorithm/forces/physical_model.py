"""力模型抽象基类。"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..dynamics import System


class PhysicalModel:
    """物理力模型基类。

    力模型在 Python 侧只承担"配置定义"职责：参数验证、``to_rust_spec``
    序列化、``to_config``/``from_config``。加速度与雅可比计算全部由 Rust
    编译路径（``ForceModel.propagate`` → ``propagate_compiled``/
    ``propagate_compiled_stm_py``）承载，不保留 Python 参考实现（issue #378）：
    需要 Rust 的场景扩展不可用即显式报错，不静默回退到 Python。

    所有坐标约定都在 ``system.coordinate_system`` 下完成；需要非默认坐标系
    计算的子类应通过 ``system.coordinate_system.transform_state()`` /
    ``transform_vector()`` 自行完成转换。
    """

    _mu: float | None
    _body: str

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

    def to_rust_spec(self, system: System) -> tuple | None:
        """序列化该 force 为 Rust ``propagate_compiled`` 接受的元组。

        返回 ``None`` 表示该 force 不支持 Rust 编译，``ForceModel.propagate``
        检测到任一 force 返回 ``None`` 时抛能力错误（显式报错，不静默回退到
        Python eom 路径）。子类按需覆盖。元组协议见 ``parse_force_tuple``
        （Rust lib.rs）：

        - GravityField: ``("gravity", c_flat, s_flat, mu, radius, degree, order,
          input_frame, propagation_frame, body, propagation_origin, tide_mode,
          k_love_flat, k_plus_flat_or_none)``
        - ThirdBody: ``("third_body", naif_id_str, mu)``
        - Indirect: ``("indirect", naif_id_str, mu)``
        - SRP: ``("srp", area, mass, cr, shadow_bodies_list)``

        Args:
            system: 当前动力学系统（用于查 origin / frame 等运行时参数）。

        Returns:
            力元组，或 ``None``。
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
