"""角动量管理联合控制（《控制方案.md》§1.6）。

太阳光压力矩导致姿态角动量累积，需周期性卸载。姿态发动机布局由安装位置
``rᵢ``（相对质心，m）与喷气方向 ``eᵢ``（单位矢量）描述，构造常数矩阵：

- ``E``（3×N）：力臂矩阵，``E[:,i] = rᵢ × eᵢ``
- ``E_r``（3×N）：力矩方向矩阵，``E_r[:,i] = eᵢ``

两种求解场景：

1. **纯角动量卸载**（mode 4-6 的角动量事件）：``min ‖V‖²  s.t.  m·E_r·V = ΔM``
2. **联合控制**（轨道控制 + 角动量卸载一次开机）：``min ‖V‖²  s.t.
   E·V = Δv_orbital,  m·E_r·V = ΔM``

发动机数 N ≥ 6（6 维约束需 N 维自由度保证可解性）。

用户输入参考：MATLAB ``fmt_inputs_control.m`` 角动量块、控制方案 §1.6 图 5-38
发动机安装示意（仅示意，布局必须作为用户输入）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import linalg as splinalg

__all__ = [
    "EngineLayout",
    "compute_delta_m",
    "compute_srp_torque",
    "solve_joint_control",
    "solve_momentum_unload",
    "validate_engine_layout",
]


@dataclass(frozen=True)
class EngineLayout:
    """姿态发动机布局。

    Attributes:
        positions_m: 安装位置（N×3，m，航天器本体坐标系质心为原点）
        directions: 喷气方向（N×3，单位矢量，构造时自动归一化）
        E: 力臂矩阵（3×N），``E[:,i] = rᵢ × eᵢ``
        E_r: 力矩方向矩阵（3×N），``E_r[:,i] = eᵢ``
    """

    positions_m: npt.NDArray[np.floating]
    directions: npt.NDArray[np.floating]
    E: npt.NDArray[np.floating] = None  # type: ignore[assignment]
    E_r: npt.NDArray[np.floating] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        pos = np.asarray(self.positions_m, dtype=float)
        direc = np.asarray(self.directions, dtype=float)
        if pos.ndim != 2 or direc.ndim != 2:
            raise ValueError("positions_m 与 directions 必须为二维数组")
        if pos.shape != direc.shape:
            raise ValueError(
                f"positions_m 形状 {pos.shape} 与 directions 形状 {direc.shape} 不匹配"
            )
        n = pos.shape[0]
        if n < 6:
            raise ValueError(f"发动机数 N={n} 不足 6，方程组欠定")
        # 归一化方向
        norms = np.linalg.norm(direc, axis=1, keepdims=True)
        if np.any(norms < 1e-12):
            raise ValueError("directions 存在零矢量（无法归一化）")
        direc = direc / norms
        # E[:,i] = rᵢ × eᵢ（力臂叉积）；E_r[:,i] = eᵢ
        E = np.cross(pos, direc).T  # (3, N)
        E_r = direc.T  # (3, N)
        # frozen=True 需要 object.__setattr__
        object.__setattr__(self, "positions_m", pos)
        object.__setattr__(self, "directions", direc)
        object.__setattr__(self, "E", E)
        object.__setattr__(self, "E_r", E_r)

    @property
    def num_engines(self) -> int:
        return self.positions_m.shape[0]


def validate_engine_layout(layout: EngineLayout) -> None:
    """校验发动机布局的可解性。

    检查项（全部抛 ``ValueError``，附中文信息）：

    1. E_r 秩 < 3：角动量管理无解（三维角动量变化需 E_r 行满秩）
    2. 增广 [E; E_r] 秩 < 6：联合控制不可解（存在 Δv_o + ΔM 组合无法同时满足）

    发动机数 N < 6 的检查在 ``EngineLayout.__post_init__`` 中完成。
    """
    E_r = layout.E_r
    E = layout.E
    if np.linalg.matrix_rank(E_r, tol=1e-10) < 3:
        raise ValueError(
            "E_r 秩不足 3（角动量管理无解）：发动机喷气方向共面或共线，"
            "无法产生三轴独立力矩。请检查安装方向是否线性无关。"
        )
    aug = np.vstack([E, E_r])
    if np.linalg.matrix_rank(aug, tol=1e-10) < 6:
        raise ValueError(
            "增广矩阵 [E; E_r] 秩不足 6（联合控制不可解）：存在 Δv_o 与 ΔM 的"
            "组合无法同时满足。请检查发动机布局的力臂与方向配置。"
        )


def compute_srp_torque(
    srp_force_n: npt.ArrayLike,
    offset_m: npt.ArrayLike,
) -> npt.NDArray[np.floating]:
    """计算 SRP 力矩（压心偏移 × SRP 力）。

    Args:
        srp_force_n: SRP 力矢量（N），方向为 Sun→SC
        offset_m: SRP 压心相对质心偏移（m），用户输入常数

    Returns:
        3 矢量（N·m）
    """
    f = np.asarray(srp_force_n, dtype=float)
    r = np.asarray(offset_m, dtype=float)
    return np.cross(r, f)


def compute_delta_m(
    torque_nm: npt.ArrayLike,
    dt_sec: float,
) -> npt.NDArray[np.floating]:
    """角动量卸载需求：ΔM = τ × Δt。

    Args:
        torque_nm: 3 矢量（N·m），SRP 力矩
        dt_sec: 自上次卸载以来的时间间隔（秒）

    Returns:
        3 矢量（kg·m²/s）
    """
    tau = np.asarray(torque_nm, dtype=float)
    return tau * float(dt_sec)


def solve_momentum_unload(
    E_r: npt.ArrayLike,
    delta_m: npt.ArrayLike,
    mass_kg: float,
) -> npt.NDArray[np.floating]:
    """纯角动量卸载求解。

    ``min ‖V‖²  s.t.  m · E_r · V = ΔM``

    Args:
        E_r: 力矩方向矩阵（3×N）
        delta_m: 角动量需求（3 矢量，kg·m²/s）
        mass_kg: 航天器质量（kg）

    Returns:
        N 矢量（m/s），各发动机速度增量
    """
    E_r = np.asarray(E_r, dtype=float)
    d = np.asarray(delta_m, dtype=float) / float(mass_kg)
    # min-norm 解：V = E_r^T (E_r E_r^T)^{-1} d
    return E_r.T @ splinalg.solve(E_r @ E_r.T, d, assume_a="pos")


def solve_joint_control(
    E: npt.ArrayLike,
    E_r: npt.ArrayLike,
    dv_orbital_mps: npt.ArrayLike,
    delta_m: npt.ArrayLike,
    mass_kg: float,
) -> npt.NDArray[np.floating]:
    """联合控制求解（轨道控制 + 角动量卸载一次开机）。

    ``min ‖V‖²  s.t.  E·V = Δv_orbital,  m·E_r·V = ΔM``

    增广线性方程组 ``A·V = b``，``A = [E; m·E_r]``（6×N，N≥6）。

    Args:
        E: 力臂矩阵（3×N）
        E_r: 力矩方向矩阵（3×N）
        dv_orbital_mps: 轨道控制速度增量（3 矢量，m/s）
        delta_m: 角动量需求（3 矢量，kg·m²/s）
        mass_kg: 航天器质量（kg）

    Returns:
        N 矢量（m/s），各发动机速度增量
    """
    E = np.asarray(E, dtype=float)
    E_r = np.asarray(E_r, dtype=float)
    dv = np.asarray(dv_orbital_mps, dtype=float)
    dm = np.asarray(delta_m, dtype=float) / float(mass_kg)
    A = np.vstack([E, E_r])
    b = np.concatenate([dv, dm])
    V, *_ = splinalg.lstsq(A, b)
    return V
