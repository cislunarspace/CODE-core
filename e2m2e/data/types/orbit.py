"""Orbit 数据容器：单条轨道的状态序列与时间序列。

纯数据容器（ADR 0011 迁移，源：``core/orbit.py``）：states（形状
(n_points, 6)）+ times + metadata + 可选 ``system`` 绑定（解释单位/坐标
系）+ 手动设 ``period``（不自动算，需要时由调用方设）。``OrbitFamily``
是轨道族容器，一并归数据层（多轨道集合也是数据）。

依赖方向：数据层不依赖算法层。``system`` 是鸭子类型绑定（解释单位/坐标
系的算法层 System 实例）；``get_jacobi_constants`` 用 ``hasattr`` 判断
能力，不 import 算法层类。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .state import State


class Orbit:
    """单条轨道的数据容器与处理工具

    Orbit 是 e2m2e 中最基本的轨道数据结构，用于存储、计算和持久化
    一条 CR3BP 轨道的全部信息。

    v4.0 重构：采用组合模式组织属性。

    Attributes:
        states: 状态序列 ``[x, y, z, vx, vy, vz]``，形状为 ``(n, 6)``
        times: 时间序列，形状为 ``(n,)``
        system: 关联的 System 对象（算法层，鸭子类型绑定）
        family_type: 轨道族类型
        parameters: 轨道参数字典
        metadata: 轨道元数据
    """

    # 状态向量分量名称
    VALID_COMPONENTS = ["x", "y", "z", "vx", "vy", "vz"]

    def __init__(
        self,
        states: npt.ArrayLike,
        times: npt.ArrayLike,
        system: Any = None,
    ) -> None:
        """初始化轨道对象

        Args:
            states: 状态序列，形状 ``(n, 6)`` 或 ``(6,)``
            times: 时间序列，形状 ``(n,)``
            system: System 对象（可选，鸭子类型绑定）

        Raises:
            ValueError: 状态分量数不等于 6 或时间序列长度不一致
        """
        self.states = np.array(states)
        self.times = np.array(times)
        self.system = system

        if self.states.ndim == 1:
            self.states = self.states.reshape(1, -1)
        if self.states.shape[1] != 6:
            raise ValueError(f"状态序列必须包含6个分量，当前为{self.states.shape[1]}个")
        if len(self.times) != self.states.shape[0]:
            raise ValueError("时间序列长度必须与状态序列长度一致")

        self.family_type: str | None = None
        self.parameters: dict = {}

        self._period: float | None = None
        self._amplitudes: dict = {}
        self._extrema: dict = {}
        self._mean_state: np.ndarray | None = None
        self._center: np.ndarray | None = None
        self._is_periodic: bool = False
        self._periodicity_error: float | None = None

        self.metadata: dict = {
            "created": datetime.now().isoformat(),
            "source": "e2m2e library",
            "description": "",
            "tags": [],
        }

        self.correction_success: bool | None = None
        self.correction_iterations: int | None = None
        self.correction_error: float | None = None
        self.correction_termination_reason: str | None = None
        self.closure_error: float | None = None

        self.compute_basic_properties()

    @property
    def period(self) -> float | None:
        """轨道周期（无量纲时间），由 x 方向零交叉检测估计或外部设置。"""
        return self._period

    @period.setter
    def period(self, value: float | None) -> None:
        self._period = value

    @property
    def amplitudes(self) -> dict:
        """各方向振幅字典（"x"/"y"/"z"，半极差 (max - min) / 2）。"""
        return self._amplitudes

    @amplitudes.setter
    def amplitudes(self, value: dict) -> None:
        self._amplitudes = value

    @property
    def extrema(self) -> dict:
        """位置极值字典（"x_max"/"x_min"/"y_max" …）。"""
        return self._extrema

    @extrema.setter
    def extrema(self, value: dict) -> None:
        self._extrema = value

    @property
    def mean_state(self) -> np.ndarray | None:
        """状态向量均值，形状 ``(6,)``"""
        return self._mean_state

    @mean_state.setter
    def mean_state(self, value: np.ndarray | None) -> None:
        self._mean_state = value

    @property
    def center(self) -> np.ndarray | None:
        """轨道几何中心（位置均值），形状 ``(3,)``"""
        return self._center

    @center.setter
    def center(self, value: np.ndarray | None) -> None:
        self._center = value

    @property
    def is_periodic(self) -> bool:
        """轨道是否被判定为周期轨道"""
        return self._is_periodic

    @is_periodic.setter
    def is_periodic(self, value: bool) -> None:
        self._is_periodic = value

    @property
    def periodicity_error(self) -> float | None:
        """周期性闭合误差范数（首尾状态差的无穷范数）"""
        return self._periodicity_error

    @periodicity_error.setter
    def periodicity_error(self, value: float | None) -> None:
        self._periodicity_error = value

    @property
    def state0(self) -> State:
        """首状态。"""
        return self.states[0]

    # ---- 核心方法 ----

    def compute_basic_properties(self) -> None:
        """计算轨道的基本几何属性（均值、极值、振幅、中心、周期估计）。"""
        self._mean_state = np.mean(self.states, axis=0)

        for i, component in enumerate(self.VALID_COMPONENTS[:3]):
            values = self.states[:, i]
            self._extrema[f"{component}_max"] = np.max(values)
            self._extrema[f"{component}_min"] = np.min(values)
            self._amplitudes[component] = (np.max(values) - np.min(values)) / 2

        self._center = self._mean_state[:3]
        self._estimate_period()

    def _estimate_period(self) -> None:
        """通过 x 方向零交叉检测估计轨道周期"""
        if len(self.times) < 2:
            return

        x_values = self.states[:, 0]
        if self._center is None:
            return
        zero_crossings = np.where(np.diff(np.sign(x_values - self._center[0])))[0]

        if len(zero_crossings) >= 2:
            t1 = self.times[zero_crossings[0]]
            t2 = self.times[zero_crossings[1]]
            self._period = 2 * (t2 - t1)
            self._check_periodicity()

    def _check_periodicity(self) -> None:
        """验证轨道的周期性（首尾闭合误差，必要时回退到 _period 查找）。"""
        if self.states is None or len(self.states) < 2:
            return

        start_state = self.states[0]
        end_state = self.states[-1]
        closure_error = float(np.linalg.norm(start_state - end_state))

        if closure_error < 1e-4:
            self._periodicity_error = closure_error
            self._is_periodic = closure_error < 1e-5
        elif self._period is not None:
            target_t = float(self.times[0]) + float(self._period)
            idx = int(np.argmin(np.abs(self.times - target_t)))
            end_state = self.states[idx]
            self._periodicity_error = float(np.linalg.norm(start_state - end_state))
            self._is_periodic = self._periodicity_error < 1e-6
        else:
            self._periodicity_error = closure_error
            self._is_periodic = False

        if self._is_periodic:
            self.metadata["description"] = "Periodic orbit"
        else:
            self.metadata["description"] = "Non-periodic trajectory"

    def save_to_file(self, filename: str | Path) -> None:
        """将轨道数据序列化保存到 JSON 文件（v3 格式兼容）"""
        filepath = Path(filename)
        dirpath = filepath.parent
        if not dirpath.exists():
            dirpath.mkdir(parents=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metadata["saved_timestamp"] = timestamp

        data = {
            "states": self.states.tolist(),
            "times": self.times.tolist(),
            "metadata": self.metadata,
            "properties": {
                "period": self._period,
                "amplitudes": self._amplitudes,
                "extrema": self._extrema,
                "mean_state": self._mean_state.tolist() if self._mean_state is not None else None,
                "family_type": self.family_type,
                "is_periodic": bool(self._is_periodic),
                "periodicity_error": self._periodicity_error,
            },
            "timestamp": timestamp,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_file(
        cls,
        filename: str | Path,
        system: Any = None,
        orbit_index: int | None = None,
    ) -> Orbit:
        """从 JSON 文件反序列化加载轨道数据（v3 格式兼容）"""
        filepath = Path(filename)
        with open(filepath) as f:
            data = json.load(f)

        if "orbits" in data:
            if orbit_index is None:
                raise ValueError(f"文件 '{filename}' 是轨道族格式，需要提供 orbit_index 参数")
            orbits_data = data["orbits"]
            if orbit_index < 0 or orbit_index >= len(orbits_data):
                raise IndexError(
                    f"orbit_index={orbit_index} 超出范围，轨道族共有 {len(orbits_data)} 条轨道"
                )
            orbit_data = orbits_data[orbit_index]
        else:
            orbit_data = data

        states = np.array(orbit_data["states"])
        times = np.array(orbit_data["times"])
        orbit = cls(states, times, system)

        properties = orbit_data.get("properties", {})
        orbit.period = properties.get("period", orbit_data.get("period"))
        orbit.amplitudes = properties.get("amplitudes", orbit_data.get("amplitudes", {}))
        orbit.extrema = properties.get("extrema", orbit_data.get("extrema", {}))
        mean_state = properties.get("mean_state", orbit_data.get("mean_state"))
        orbit.mean_state = np.array(mean_state) if mean_state else None
        orbit.family_type = properties.get("family_type", orbit_data.get("family_type"))
        orbit._check_periodicity()
        saved_error = properties.get("periodicity_error")
        if saved_error is not None:
            orbit._periodicity_error = saved_error
        orbit.metadata = orbit_data.get("metadata", data.get("metadata", {}))

        return orbit

    def __str__(self):
        if self._is_periodic:
            return f"Orbit(period={self._period:.4f}, amplitudes={self._amplitudes}, periodic=True)"
        return f"Orbit(length={len(self.times)}, amplitudes={self._amplitudes})"

    def __repr__(self):
        return (
            f"Orbit(states_shape={self.states.shape}, times_length={len(self.times)}, "
            f"period={self._period}, system={self.system})"
        )

    def copy(self) -> Orbit:
        """创建轨道的深拷贝"""
        new_orbit = Orbit(
            states=self.states.copy(),
            times=self.times.copy(),
            system=self.system,
        )

        new_orbit.family_type = self.family_type
        new_orbit.parameters = self.parameters.copy()

        new_orbit.period = self._period
        new_orbit.amplitudes = self._amplitudes.copy()
        new_orbit.extrema = self._extrema.copy()
        new_orbit.mean_state = self._mean_state.copy() if self._mean_state is not None else None
        new_orbit.center = self._center.copy() if self._center is not None else None

        new_orbit.is_periodic = self._is_periodic
        new_orbit.periodicity_error = self._periodicity_error

        new_orbit.metadata = self.metadata.copy()

        new_orbit.correction_success = self.correction_success
        new_orbit.correction_iterations = self.correction_iterations
        new_orbit.correction_error = self.correction_error
        new_orbit.correction_termination_reason = self.correction_termination_reason
        new_orbit.closure_error = self.closure_error

        return new_orbit


class OrbitFamily:
    """轨道族容器

    用于存储和管理多个 Orbit 对象组成的轨道族。

    Attributes:
        orbits: Orbit 对象列表
        family_type: 轨道族类型
        system: 关联的 System 对象
        metadata: 轨道族元数据
    """

    def __init__(
        self,
        orbits: list[Orbit] | None = None,
        family_type: str | None = None,
        system: Any = None,
    ) -> None:
        """初始化轨道族

        Args:
            orbits: 初始轨道列表，也可传入单条 Orbit 对象
            family_type: 轨道族类型（如 ``"halo"``、``"lyapunov"``）
            system: 关联的 System 对象（用于计算 Jacobi 常数等）

        Raises:
            TypeError: orbits 列表中包含非 Orbit 对象
        """
        self.orbits: list[Orbit] = []
        if orbits is not None:
            if isinstance(orbits, Orbit):
                self.orbits = [orbits]
            elif isinstance(orbits, list):
                if len(orbits) > 0 and not all(isinstance(o, Orbit) for o in orbits):
                    raise TypeError("All elements in orbits list must be Orbit instances")
                self.orbits = list(orbits)
        self.family_type = family_type
        self.system = system
        self.metadata = {
            "created": datetime.now().isoformat(),
            "source": "e2m2e library",
            "description": "",
            "tags": [],
        }

    @property
    def states(self) -> npt.NDArray[np.floating]:
        return self.get_states()

    @property
    def periods(self) -> npt.NDArray[np.floating]:
        return self.get_periods()

    def __len__(self) -> int:
        return len(self.orbits)

    def __getitem__(self, index: int) -> Orbit:
        return self.orbits[index]

    def __iter__(self):
        return iter(self.orbits)

    def add_orbit(self, orbit: Orbit) -> None:
        """向轨道族添加一条轨道"""
        self.orbits.append(orbit)

    def get_states(self) -> npt.NDArray[np.floating]:
        """获取所有轨道的初始状态，形状 ``(n_orbits, 6)``"""
        return np.array([orbit.states[0] for orbit in self.orbits])

    def get_periods(self) -> npt.NDArray[np.floating]:
        """获取所有已知周期（period 不为 None 的轨道）"""
        return np.array([orbit.period for orbit in self.orbits if orbit.period is not None])

    def get_jacobi_constants(self) -> npt.NDArray[np.floating]:
        """计算所有轨道初始状态的 Jacobi 常数。

        Jacobi 常数是 CR3BP 专有概念；非 CR3BP 系统（无
        ``get_jacobi_constant`` 方法）返回空数组。数据层不依赖算法层类，
        用鸭子类型判断能力。
        """
        if self.system is None or not hasattr(self.system, "get_jacobi_constant"):
            return np.array([])
        return np.array([self.system.get_jacobi_constant(orbit.states[0]) for orbit in self.orbits])

    def save_to_file(self, filename: str | Path) -> None:
        """将轨道族序列化保存到 JSON 文件"""
        filepath = Path(filename)
        dirpath = filepath.parent
        if not dirpath.exists():
            dirpath.mkdir(parents=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metadata["saved_timestamp"] = timestamp

        data = {
            "n_orbits": len(self.orbits),
            "family_type": self.family_type,
            "metadata": self.metadata,
            "orbits": [
                {
                    "states": orbit.states.tolist(),
                    "times": orbit.times.tolist(),
                    "period": orbit.period,
                    "amplitudes": orbit.amplitudes,
                    "family_type": orbit.family_type,
                    "is_periodic": orbit.is_periodic,
                    "closure_error": getattr(orbit, "closure_error", None),
                    "metadata": getattr(orbit, "metadata", {}),
                }
                for orbit in self.orbits
            ],
            "timestamp": timestamp,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_file(cls, filename: str | Path, system: Any = None) -> OrbitFamily:
        """从 JSON 文件反序列化加载轨道族"""
        filepath = Path(filename)
        with open(filepath) as f:
            data = json.load(f)

        orbits = []
        for orbit_data in data["orbits"]:
            states = np.array(orbit_data["states"])
            times = np.array(orbit_data["times"])
            orbit = Orbit(states, times, system)
            orbit.period = orbit_data.get("period")
            orbit.amplitudes = orbit_data.get("amplitudes", {})
            orbit.family_type = orbit_data.get("family_type")
            orbit.is_periodic = orbit_data.get("is_periodic", False)
            orbit.metadata = orbit_data.get("metadata", {})
            orbits.append(orbit)

        family = cls(orbits, data.get("family_type"), system)
        family.metadata = data.get("metadata", {})
        return family

    def __str__(self):
        return f"OrbitFamily(n_orbits={len(self.orbits)}, family_type={self.family_type})"

    def __repr__(self):
        return f"OrbitFamily(orbits={len(self.orbits)}, system={self.system})"
