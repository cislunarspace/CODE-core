"""状态向量与轨道根数类型。

``State`` 是 6 维状态向量 ``[x, y, z, vx, vy, vz]`` 的**类型别名** （numpy 数组），
分量顺序不可变更（CLAUDE.md 约定）。单值 → 别名，不强制包装类。

实现状态：骨架。类型别名已可用，轨道根数转换待实现。
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

__all__ = ["State", "OrbitState"]

#: 航天器瞬时状态（CR3BP 无量纲或物理单位），形状 (6,)。
State: TypeAlias = npt.NDArray[np.float64]

#: 同 ``State`` （轨道状态别名）。
OrbitState: TypeAlias = State
