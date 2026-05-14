"""轨道数据模块

本模块定义了圆型限制性三体问题（CR3BP）中轨道数据的表示与处理逻辑，
是 e2m2e 四层架构中 core 层的核心组件之一。

重构说明 (v4.0 MBSE)
--------------------
采用组合模式重构 Orbit 类：
- 核心数据（states, times, system）直接持有
- 计算属性（period, amplitudes, extrema, center 等）通过 property 代理
- 稳定性属性（monodromy_matrix, eigenvalues, stability 等）通过 property 代理
- 删除所有未使用的保留字段
  （radius, shape, orientation, is_quasi_periodic, is_chaotic, segments, segment_indices）
- 保持 v3 JSON 格式向后兼容

主要类：
    Orbit: 单条轨道的数据容器，支持属性计算、序列化/反序列化和稳定性分析。
    OrbitFamily: 轨道族容器，用于存储和管理多条同族轨道。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .dynamics import CR3BP_Dynamics
from .system import CR3BP_System


class Orbit:
    """单条轨道的数据容器与处理工具

    Orbit 是 e2m2e 中最基本的轨道数据结构，用于存储、计算和持久化
    一条 CR3BP 轨道的全部信息。

    v4.0 重构：采用组合模式组织属性。

    Attributes:
        states: 状态序列 ``[x, y, z, vx, vy, vz]``，形状为 ``(n, 6)``
        times: 时间序列，形状为 ``(n,)``
        system: 关联的 CR3BP_System 对象
        family_type: 轨道族类型
        parameters: 轨道参数字典
        metadata: 轨道元数据
        jacobi_constants: Jacobi 常数序列
        stability_indices: 稳定性指标（由外部算法填充）
    """

    # 支持的轨道族类型
    VALID_FAMILY_TYPES = [
        "halo",
        "lyapunov",
        "vertical",
        "axial",
        "butterfly",
        "dragonfly",
    ]
    # 状态向量分量名称
    VALID_COMPONENTS = ["x", "y", "z", "vx", "vy", "vz"]

    def __init__(
        self,
        states: npt.ArrayLike,
        times: npt.ArrayLike,
        system: CR3BP_System | None = None,
    ) -> None:
        """初始化轨道对象

        Args:
            states: 状态序列，形状 ``(n, 6)`` 或 ``(6,)``
            times: 时间序列，形状 ``(n,)``
            system: CR3BP_System 对象（可选）

        Raises:
            ValueError: 状态分量数不等于 6 或时间序列长度不一致
        """
        # 核心数据
        self.states = np.array(states)
        self.times = np.array(times)
        self.system = system

        # 数据验证
        if self.states.ndim == 1:
            self.states = self.states.reshape(1, -1)
        if self.states.shape[1] != 6:
            raise ValueError(f"状态序列必须包含6个分量，当前为{self.states.shape[1]}个")
        if len(self.times) != self.states.shape[0]:
            raise ValueError("时间序列长度必须与状态序列长度一致")

        # 外部填充属性
        self.jacobi_constants: np.ndarray | None = None
        self.stability_indices: dict | None = None
        self.family_type: str | None = None
        self.parameters: dict = {}

        # ---- 内部计算属性（通过 property 代理） ----
        self._period: float | None = None
        self._amplitudes: dict = {}
        self._extrema: dict = {}
        self._mean_state: np.ndarray | None = None
        self._center: np.ndarray | None = None
        self._is_periodic: bool = False
        self._periodicity_error: float | None = None

        # 稳定性属性
        self._monodromy_matrix: np.ndarray | None = None
        self._eigenvalues: np.ndarray | None = None
        self._stability: str | None = None
        self._lyapunov_exponents: np.ndarray | None = None

        # 元数据
        self.metadata: dict = {
            "created": datetime.now().isoformat(),
            "source": "e2m2e library",
            "description": "",
            "tags": [],
        }

        # 计算基本属性
        self.compute_basic_properties()

    # ---- Property 代理（保持向后兼容） ----

    @property
    def period(self) -> float | None:
        """轨道周期（无量纲时间）

        由 x 方向零交叉检测估计，或由外部算法设置。
        """
        return self._period

    @period.setter
    def period(self, value: float | None) -> None:
        self._period = value

    @property
    def amplitudes(self) -> dict:
        """各方向振幅字典

        键为 ``"x"``/``"y"``/``"z"``，值为半极差 ``(max - min) / 2``。
        """
        return self._amplitudes

    @amplitudes.setter
    def amplitudes(self, value: dict) -> None:
        self._amplitudes = value

    @property
    def extrema(self) -> dict:
        """位置极值字典

        键为 ``"x_max"``/``"x_min"``/``"y_max"`` … 等格式。
        """
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
    def monodromy_matrix(self) -> np.ndarray | None:
        """单值矩阵（Monodromy Matrix），形状 ``(6, 6)``

        通过 ``compute_monodromy_matrix`` 计算并缓存。
        """
        return self._monodromy_matrix

    @monodromy_matrix.setter
    def monodromy_matrix(self, value: np.ndarray | None) -> None:
        self._monodromy_matrix = value

    @property
    def eigenvalues(self) -> np.ndarray | None:
        """单值矩阵的特征值，形状 ``(6,)``

        特征值在单位圆上 → 稳定；|λ| > 1 → 不稳定方向。
        """
        return self._eigenvalues

    @eigenvalues.setter
    def eigenvalues(self, value: np.ndarray | None) -> None:
        self._eigenvalues = value

    @property
    def stability(self) -> str | None:
        """稳定性标签：``"stable"`` / ``"unstable"`` / ``"marginally_stable"``"""
        return self._stability

    @stability.setter
    def stability(self, value: str | None) -> None:
        self._stability = value

    @property
    def lyapunov_exponents(self) -> np.ndarray | None:
        """Lyapunov 指数，形状 ``(6,)``

        由 ``ln|λ_i| / T`` 计算，其中 T 为轨道周期。
        """
        return self._lyapunov_exponents

    @lyapunov_exponents.setter
    def lyapunov_exponents(self, value: np.ndarray | None) -> None:
        self._lyapunov_exponents = value

    # ---- 核心方法 ----

    def compute_basic_properties(self) -> None:
        """计算轨道的基本几何与物理属性

        自动计算：
        1. Jacobi 常数序列（当 system 不为 None）
        2. 平均状态向量
        3. 位置极值与振幅
        4. 轨道中心
        5. 周期估计（零交叉检测）
        """
        if self.system is not None and hasattr(self.system, "get_jacobi_constant"):
            self.jacobi_constants = np.array(
                [self.system.get_jacobi_constant(state) for state in self.states]
            )

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
        """验证轨道的周期性

        优先使用轨迹数据自身的闭合误差（第一个状态与最后一个状态的差异），
        因为轨迹通常为一个完整周期的积分结果。仅当轨迹闭合误差不理想时，
        才回退到基于 _period 的查找。
        """
        if self.states is None or len(self.states) < 2:
            return

        start_state = self.states[0]
        end_state = self.states[-1]
        closure_error = float(np.linalg.norm(start_state - end_state))

        # 若轨迹闭合误差 < 1e-4，认为轨迹是一个完整周期的积分结果
        # 直接用闭合误差作为周期性误差
        # 阈值说明：1e-4 用于判断「轨迹大致闭合」（来自差分修正的典型残差量级），
        # 而非严格周期性判定；严格判定使用 1e-5（见下方赋值）。
        # 两级阈值的设计：粗阈值允许将 DC 残差级别的近周期轨道纳入「可分析」范畴，
        # 细阈值则用于最终标记 is_periodic 供后续稳定性分析使用。
        if closure_error < 1e-4:
            self._periodicity_error = closure_error
            self._is_periodic = closure_error < 1e-5
        elif self._period is not None:
            # 回退：用 _period 查找最近的时刻
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

    def compute_monodromy_matrix(self, dynamics: CR3BP_Dynamics) -> npt.NDArray[np.floating]:
        """计算轨道的单值矩阵（Monodromy Matrix）

        Args:
            dynamics: CR3BP_Dynamics 对象

        Returns:
            单值矩阵 (6, 6)

        Raises:
            ValueError: 轨道周期未知
        """
        if self._period is None:
            raise ValueError("无法计算单值矩阵：轨道周期未知")

        initial_state = self.states[0]
        self._monodromy_matrix = dynamics.compute_state_transition_matrix(
            initial_state, self._period
        )
        self._eigenvalues = np.linalg.eigvals(self._monodromy_matrix)
        return self._monodromy_matrix

    def compute_stability(self, dynamics: CR3BP_Dynamics) -> dict[str, Any]:
        """计算轨道的稳定性指标

        Args:
            dynamics: CR3BP_Dynamics 对象

        Returns:
            稳定性分析结果字典
        """
        if self._period is None or self._period <= 0:
            return {
                "stability": "unknown",
                "eigenvalues": None,
                "max_deviation": None,
                "lyapunov_exponents": None,
            }
        if self._monodromy_matrix is None:
            self.compute_monodromy_matrix(dynamics)

        if self._eigenvalues is None:
            raise ValueError("Eigenvalues not computed. Call compute_monodromy_matrix first.")
        eigenvalues = self._eigenvalues
        magnitudes = np.abs(eigenvalues)
        max_deviation = np.max(np.abs(magnitudes - 1.0))

        if max_deviation < 1e-6:
            self._stability = "stable"
        elif np.any(magnitudes > 1.0 + 1e-6):
            self._stability = "unstable"
        else:
            self._stability = "marginally_stable"

        self._lyapunov_exponents = np.log(magnitudes) / self._period

        return {
            "stability": self._stability,
            "eigenvalues": eigenvalues,
            "max_deviation": max_deviation,
            "lyapunov_exponents": self._lyapunov_exponents,
        }

    def get_period(self) -> float | None:
        """获取轨道周期

        Returns:
            无量纲周期；若未估计到则返回 None
        """
        return self._period

    def get_amplitude(self, direction: str) -> float:
        """获取指定方向的轨道振幅"""
        if direction not in self._amplitudes:
            raise ValueError(f"无效的方向: {direction}。可用方向: {list(self._amplitudes.keys())}")
        return self._amplitudes[direction]

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
        system: CR3BP_System | None = None,
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

        orbit.metadata = orbit_data.get("metadata", data.get("metadata", {}))

        properties = orbit_data.get("properties", {})
        orbit.period = properties.get("period", orbit_data.get("period"))
        orbit.amplitudes = properties.get("amplitudes", orbit_data.get("amplitudes", {}))
        orbit.extrema = properties.get("extrema", orbit_data.get("extrema", {}))
        mean_state = properties.get("mean_state", orbit_data.get("mean_state"))
        orbit.mean_state = np.array(mean_state) if mean_state else None
        orbit.family_type = properties.get("family_type", orbit_data.get("family_type"))
        # 重新检查周期性（使用正确的 period），不再从 JSON 恢复 is_periodic
        orbit._check_periodicity()

        return orbit

    def __str__(self):
        if self._is_periodic:
            return f"Orbit(period={self._period:.4f}, amplitudes={self._amplitudes}, periodic=True)"
        else:
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

        new_orbit.jacobi_constants = (
            self.jacobi_constants.copy() if self.jacobi_constants is not None else None
        )
        new_orbit.stability_indices = (
            self.stability_indices.copy() if self.stability_indices is not None else None
        )
        new_orbit.family_type = self.family_type
        new_orbit.parameters = self.parameters.copy()

        new_orbit.period = self._period
        new_orbit.amplitudes = self._amplitudes.copy()
        new_orbit.extrema = self._extrema.copy()
        new_orbit.mean_state = self._mean_state.copy() if self._mean_state is not None else None
        new_orbit.center = self._center.copy() if self._center is not None else None

        new_orbit.monodromy_matrix = (
            self._monodromy_matrix.copy() if self._monodromy_matrix is not None else None
        )
        new_orbit.eigenvalues = self._eigenvalues.copy() if self._eigenvalues is not None else None
        new_orbit.stability = self._stability
        new_orbit.lyapunov_exponents = (
            self._lyapunov_exponents.copy() if self._lyapunov_exponents is not None else None
        )

        new_orbit.is_periodic = self._is_periodic
        new_orbit.periodicity_error = self._periodicity_error

        new_orbit.metadata = self.metadata.copy()

        # 复制动态添加的属性
        for attr in (
            "correction_success",
            "correction_iterations",
            "correction_error",
            "correction_termination_reason",
            "closure_error",
        ):
            if hasattr(self, attr):
                setattr(new_orbit, attr, getattr(self, attr))

        return new_orbit


class OrbitFamily:
    """轨道族容器

    用于存储和管理多个 Orbit 对象组成的轨道族。

    Attributes:
        orbits: Orbit 对象列表
        family_type: 轨道族类型
        system: 关联的 CR3BP_System 对象
        metadata: 轨道族元数据
    """

    def __init__(
        self,
        orbits: list[Orbit] | None = None,
        family_type: str | None = None,
        system: CR3BP_System | None = None,
    ) -> None:
        """初始化轨道族

        Args:
            orbits: 初始轨道列表，也可传入单条 Orbit 对象
            family_type: 轨道族类型（如 ``"halo"``、``"lyapunov"``）
            system: 关联的 CR3BP_System 对象（用于计算 Jacobi 常数等）

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
        """向轨道族添加一条轨道

        Args:
            orbit: 要添加的 Orbit 对象
        """
        self.orbits.append(orbit)

    def get_states(self) -> npt.NDArray[np.floating]:
        """获取所有轨道的初始状态

        Returns:
            初始状态数组，形状 ``(n_orbits, 6)``
        """
        return np.array([orbit.states[0] for orbit in self.orbits])

    def get_periods(self) -> npt.NDArray[np.floating]:
        """获取所有已知周期

        Returns:
            周期数组，仅包含 period 不为 None 的轨道
        """
        return np.array([orbit.period for orbit in self.orbits if orbit.period is not None])

    def get_jacobi_constants(self) -> npt.NDArray[np.floating]:
        """计算所有轨道初始状态的 Jacobi 常数

        需要关联 CR3BP_System；若 system 为 None 则返回空数组。

        Returns:
            Jacobi 常数数组，形状 ``(n_orbits,)``
        """
        if self.system is None:
            return np.array([])
        return np.array([self.system.get_jacobi_constant(orbit.states[0]) for orbit in self.orbits])

    def save_to_file(self, filename: str | Path) -> None:
        """将轨道族序列化保存到 JSON 文件

        Args:
            filename: 输出文件路径，父目录不存在时自动创建
        """
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
    def load_from_file(
        cls, filename: str | Path, system: CR3BP_System | None = None
    ) -> OrbitFamily:
        """从 JSON 文件反序列化加载轨道族

        Args:
            filename: 输入文件路径
            system: 关联的 CR3BP_System 对象（可选）

        Returns:
            加载的 OrbitFamily 实例
        """
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
