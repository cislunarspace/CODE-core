"""
轨道模块

包含Orbit类，用于表示和处理三体问题中的轨道数据。
"""

from __future__ import annotations

import os
import numpy as np
from scipy import interpolate
from scipy.interpolate import interp1d
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from pathlib import Path

import numpy.typing as npt

from .system import System, CR3BP_System
from .dynamics import CR3BP_Dynamics


class Orbit:
    """轨道数据和处理

    属性：
    - states: 状态序列[x, y, z, vx, vy, vz]
    - times: 时间序列
    - jacobi_constants: Jacobi常数序列
    - stability_indices: 稳定性指标
    - family_type: 轨道族类型(halo, lyapunov, etc.)
    - parameters: 轨道参数

    方法：
    - __init__(states, times): 初始化轨道
    - compute_monodromy_matrix(): 计算单值矩阵
    - compute_stability(): 计算稳定性
    - get_period(): 获取轨道周期
    - get_amplitude(direction): 获取指定方向振幅
    - interpolate_at_time(t): 时间插值
    - save_to_file(filename): 保存轨道数据
    - load_from_file(filename): 加载轨道数据
    """

    # 类属性
    VALID_FAMILY_TYPES = [
        "halo",
        "lyapunov",
        "vertical",
        "axial",
        "butterfly",
        "dragonfly",
    ]
    VALID_COMPONENTS = ["x", "y", "z", "vx", "vy", "vz"]
    DEFAULT_INTERPOLATION_KIND = "cubic"  # 插值类型

    def __init__(
        self,
        states: npt.ArrayLike,
        times: npt.ArrayLike,
        system: Optional[CR3BP_System] = None,
    ) -> None:
        """初始化轨道

        参数：
        - states: 状态序列，形状为 (n, 6) 的数组
        - times: 时间序列，形状为 (n,) 的数组
        - system: CR3BP_System对象（可选）
        """
        # 基本轨道数据
        self.states = np.array(states)  # 状态序列
        self.times = np.array(times)  # 时间序列
        self.system = system  # 关联的系统

        # 验证输入维度
        if self.states.ndim == 1:
            self.states = self.states.reshape(1, -1)
        if self.states.shape[1] != 6:
            raise ValueError(f"状态序列必须包含6个分量，当前为{self.states.shape[1]}个")
        if len(self.times) != self.states.shape[0]:
            raise ValueError("时间序列长度必须与状态序列长度一致")

        # 轨道特征
        self.jacobi_constants = None  # Jacobi常数序列
        self.stability_indices = None  # 稳定性指标
        self.family_type = None  # 轨道族类型
        self.parameters = {}  # 轨道参数字典

        # 轨道属性
        self.period: float | None = None  # 轨道周期
        self.amplitudes: dict = {}  # 各方向振幅 {'x': amp_x, 'y': amp_y, 'z': amp_z}
        self.extrema: dict = {}  # 极值点 {'x_max': xmax, 'x_min': xmin, ...}
        self.mean_state = None  # 平均状态

        # 单值矩阵和稳定性
        self.monodromy_matrix = None  # 单值矩阵
        self.eigenvalues = None  # 特征值
        self.stability = None  # 稳定性标签 ('stable', 'unstable', 'marginally_stable')
        self.lyapunov_exponents = None  # Lyapunov指数

        # 插值对象
        self.interpolators = {}  # 各分量的插值函数 {'x': interp_func, ...}
        self.interpolation_kind = self.DEFAULT_INTERPOLATION_KIND  # 插值类型

        # 轨道几何特征
        self.center = None  # 轨道中心点
        self.radius = None  # 轨道半径（如果是圆形）
        self.shape = None  # 轨道形状特征
        self.orientation = None  # 轨道取向

        # 轨道分类
        self.is_periodic = False  # 是否为周期轨道
        self.is_quasi_periodic = False  # 是否为拟周期轨道
        self.is_chaotic = False  # 是否为混沌轨道
        self.periodicity_error = None  # 周期性误差

        # 轨道段（用于长轨迹分段）
        self.segments = []  # 轨道段列表
        self.segment_indices = []  # 分段索引

        # 元数据
        self.metadata = {  # 轨道元数据
            "created": datetime.now().isoformat(),  # 创建时间
            "source": "e2m2e library",  # 数据来源
            "description": "",  # 轨道描述
            "tags": [],  # 标签
        }

        # 初始化计算
        self.compute_basic_properties()

    def compute_basic_properties(self) -> None:
        """计算基本轨道属性"""
        # 计算Jacobi常数
        if self.system is not None:
            self.jacobi_constants = np.array(
                [self.system.get_jacobi_constant(state) for state in self.states]
            )

        # 计算平均值
        self.mean_state = np.mean(self.states, axis=0)

        # 计算极值
        for i, component in enumerate(self.VALID_COMPONENTS[:3]):  # 只计算位置分量
            values = self.states[:, i]
            self.extrema[f"{component}_max"] = np.max(values)
            self.extrema[f"{component}_min"] = np.min(values)
            self.amplitudes[component] = (np.max(values) - np.min(values)) / 2

        # 计算轨道中心（位置分量的平均值）
        self.center = self.mean_state[:3]

        # 构建插值函数（需要至少2个点）
        if len(self.times) >= 2:
            for i, component in enumerate(self.VALID_COMPONENTS):
                self.interpolators[component] = interpolate.interp1d(
                    self.times,
                    self.states[:, i],
                    kind=self.interpolation_kind if len(self.times) >= 4 else "linear",
                    fill_value="extrapolate",
                )

        # 估计周期（如果轨道是周期的）
        self._estimate_period()

    def _estimate_period(self) -> None:
        """估计轨道周期"""
        if len(self.times) < 2:
            return

        # 尝试通过x分量的过零点检测周期
        x_values = self.states[:, 0]
        zero_crossings = np.where(np.diff(np.sign(x_values - self.center[0])))[0]

        if len(zero_crossings) >= 2:
            # 使用前两个过零点的时间差作为周期估计
            t1 = self.times[zero_crossings[0]]
            t2 = self.times[zero_crossings[1]]
            self.period = 2 * (t2 - t1)  # 假设对称性

            # 检查周期性
            self._check_periodicity()

    def _check_periodicity(self) -> None:
        """检查轨道周期性"""
        if self.period is None:
            return

        # 计算轨道起点和终点状态
        start_state = self.states[0]
        end_state = self.interpolate_at_time(self.period)

        # 计算周期性误差
        self.periodicity_error = np.linalg.norm(start_state - end_state)

        # 判断是否为周期轨道
        tolerance = 1e-6
        self.is_periodic = self.periodicity_error < tolerance

        if self.is_periodic:
            self.metadata["description"] = "Periodic orbit"
        else:
            self.metadata["description"] = "Non-periodic trajectory"

    def interpolate_at_time(self, t: float) -> npt.NDArray[np.floating]:
        """在指定时间插值轨道状态

        参数：
        - t: 时间值

        返回：
        - 插值后的状态向量
        """
        state = np.zeros(6)
        for i, component in enumerate(self.VALID_COMPONENTS):
            state[i] = self.interpolators[component](t)
        return state

    def compute_monodromy_matrix(self, dynamics: CR3BP_Dynamics) -> npt.NDArray[np.floating]:
        """计算单值矩阵

        参数：
        - dynamic: CR3BP_Dynamics对象

        返回：
        - 单值矩阵 (6x6)
        """
        if self.period is None:
            raise ValueError("无法计算单值矩阵：轨道周期未知")

        # 使用动力学系统计算状态转移矩阵
        initial_state = self.states[0]
        self.monodromy_matrix = dynamics.compute_state_transition_matrix(initial_state, self.period)

        # 计算特征值
        self.eigenvalues = np.linalg.eigvals(self.monodromy_matrix)

        return self.monodromy_matrix

    def compute_stability(self, dynamics: CR3BP_Dynamics) -> Dict[str, Any]:
        """计算轨道稳定性

        参数：
        - dynamic: CR3BP_Dynamics对象

        返回：
        - 稳定性分析结果字典
        """
        # 计算单值矩阵（如果尚未计算）
        if self.monodromy_matrix is None:
            self.compute_monodromy_matrix(dynamics)

        # 分析特征值
        eigenvalues = self.eigenvalues
        magnitudes = np.abs(eigenvalues)

        # 检查稳定性条件
        # 对于周期轨道，稳定性要求所有特征值的模为1
        max_deviation = np.max(np.abs(magnitudes - 1.0))

        if max_deviation < 1e-6:
            self.stability = "stable"
        elif np.any(magnitudes > 1.0 + 1e-6):
            self.stability = "unstable"
        else:
            self.stability = "marginally_stable"

        # 计算Lyapunov指数
        self.lyapunov_exponents = np.log(magnitudes) / self.period

        return {
            "stability": self.stability,
            "eigenvalues": eigenvalues,
            "max_deviation": max_deviation,
            "lyapunov_exponents": self.lyapunov_exponents,
        }

    def get_period(self) -> Optional[float]:
        """获取轨道周期

        返回：
        - 轨道周期（如果已知），否则返回None
        """
        return self.period

    def get_amplitude(self, direction: str) -> float:
        """获取指定方向振幅

        参数：
        - direction: 方向 ('x', 'y', 'z')

        返回：
        - 振幅值
        """
        if direction not in self.amplitudes:
            raise ValueError(f"无效的方向: {direction}。可用方向: {list(self.amplitudes.keys())}")
        return self.amplitudes[direction]

    def save_to_file(self, filename: Union[str, Path]) -> None:
        """保存轨道数据到文件

        参数：
        - filename: 文件名（必须是相对路径，且指向当前工作目录内）

        异常：
        - ValueError: 当路径指向当前工作目录之外时抛出
        """
        filepath = Path(filename)

        # 自动创建目录
        dirpath = filepath.parent
        if not dirpath.exists():
            dirpath.mkdir(parents=True)

        # 加入时间戳字段
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metadata["saved_timestamp"] = timestamp

        data = {
            "states": self.states.tolist(),
            "times": self.times.tolist(),
            "metadata": self.metadata,
            "properties": {
                "period": self.period,
                "amplitudes": self.amplitudes,
                "extrema": self.extrema,
                "mean_state": self.mean_state.tolist() if self.mean_state is not None else None,
                "family_type": self.family_type,
                "is_periodic": bool(self.is_periodic),
                "periodicity_error": self.periodicity_error,
            },
            "timestamp": timestamp,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_file(
        cls,
        filename: Union[str, Path],
        system: Optional[CR3BP_System] = None,
        orbit_index: Optional[int] = None
    ) -> "Orbit":
        """从文件加载轨道数据

        参数：
        - filename: 文件名（必须是相对路径，且指向当前工作目录内）
        - system: CR3BP_System对象（可选）
        - orbit_index: 轨道索引（可选），当文件为轨道族格式时有效

        返回：
        - 如果提供 orbit_index：返回指定单条 Orbit
        - 如果不提供 orbit_index 且文件为单轨道格式：返回 Orbit
        - 如果不提供 orbit_index 且文件为轨道族格式：返回 OrbitFamily

        注意：
        - 单轨道格式包含 "states", "times", "metadata", "properties" 字段
        - 轨道族格式包含 "orbits" 数组字段

        异常：
        - ValueError: 当路径指向当前工作目录之外时抛出
        """
        filepath = Path(filename)
        with open(filepath, "r") as f:
            data = json.load(f)

        # 判断是轨道族格式还是单轨道格式
        if "orbits" in data:
            # 轨道族格式
            if orbit_index is None:
                raise ValueError(
                    f"文件 '{filename}' 是轨道族格式，需要提供 orbit_index 参数指定要加载的轨道"
                )
            orbits_data = data["orbits"]
            if orbit_index < 0 or orbit_index >= len(orbits_data):
                raise IndexError(
                    f"orbit_index={orbit_index} 超出范围，轨道族共有 {len(orbits_data)} 条轨道"
                )
            orbit_data = orbits_data[orbit_index]
        else:
            # 单轨道格式
            orbit_data = data

        # 创建轨道对象
        states = np.array(orbit_data["states"])
        times = np.array(orbit_data["times"])
        orbit = cls(states, times, system)

        # 恢复元数据
        orbit.metadata = orbit_data.get("metadata", data.get("metadata", {}))

        # 恢复属性
        properties = orbit_data.get("properties", {})
        orbit.period = properties.get("period", orbit_data.get("period"))
        orbit.amplitudes = properties.get("amplitudes", orbit_data.get("amplitudes", {}))
        orbit.extrema = properties.get("extrema", orbit_data.get("extrema", {}))
        mean_state = properties.get("mean_state", orbit_data.get("mean_state"))
        orbit.mean_state = np.array(mean_state) if mean_state else None
        orbit.family_type = properties.get("family_type", orbit_data.get("family_type"))
        orbit.is_periodic = properties.get("is_periodic", orbit_data.get("is_periodic", False))
        orbit.periodicity_error = properties.get(
            "periodicity_error", orbit_data.get("periodicity_error")
        )

        return orbit

    def __str__(self):
        """字符串表示"""
        if self.is_periodic:
            return f"Orbit(period={self.period:.4f}, amplitudes={self.amplitudes}, periodic=True)"
        else:
            return f"Orbit(length={len(self.times)}, amplitudes={self.amplitudes})"

    def __repr__(self):
        """详细表示"""
        return (
            f"Orbit(states_shape={self.states.shape}, times_length={len(self.times)}, "
            f"period={self.period}, system={self.system})"
        )

    def copy(self) -> "Orbit":
        """创建轨道的深拷贝

        返回:
            Orbit: 一个新的 Orbit 对象，包含相同的数据但独立引用
        """
        new_orbit = Orbit(
            states=self.states.copy(),
            times=self.times.copy(),
            system=self.system,
        )

        # 复制所有属性
        new_orbit.jacobi_constants = (
            self.jacobi_constants.copy() if self.jacobi_constants is not None else None
        )
        new_orbit.stability_indices = (
            self.stability_indices.copy() if self.stability_indices is not None else None
        )
        new_orbit.family_type = self.family_type
        new_orbit.parameters = self.parameters.copy()

        # 轨道属性
        new_orbit.period = self.period
        new_orbit.amplitudes = self.amplitudes.copy()
        new_orbit.extrema = self.extrema.copy()
        new_orbit.mean_state = self.mean_state.copy() if self.mean_state is not None else None

        # 单值矩阵和稳定性
        new_orbit.monodromy_matrix = (
            self.monodromy_matrix.copy() if self.monodromy_matrix is not None else None
        )
        new_orbit.eigenvalues = self.eigenvalues.copy() if self.eigenvalues is not None else None
        new_orbit.stability = self.stability
        new_orbit.lyapunov_exponents = (
            self.lyapunov_exponents.copy() if self.lyapunov_exponents is not None else None
        )

        # 插值对象（重新构建）
        new_orbit.interpolation_kind = self.interpolation_kind

        # 轨道几何特征
        new_orbit.center = self.center.copy() if self.center is not None else None
        new_orbit.radius = self.radius
        new_orbit.shape = self.shape
        new_orbit.orientation = self.orientation

        # 轨道分类
        new_orbit.is_periodic = self.is_periodic
        new_orbit.is_quasi_periodic = self.is_quasi_periodic
        new_orbit.is_chaotic = self.is_chaotic
        new_orbit.periodicity_error = self.periodicity_error

        # 轨道段
        new_orbit.segments = self.segments.copy()
        new_orbit.segment_indices = self.segment_indices.copy()

        # 元数据
        new_orbit.metadata = self.metadata.copy()

        return new_orbit


class OrbitFamily:
    """轨道族容器

    用于存储和管理多个Orbit对象组成的轨道族。

    属性：
    - orbits: Orbit对象列表
    - family_type: 轨道族类型 (halo, lyapunov, dro, ro, etc.)
    - system: 关联的CR3BP_System对象

    方法：
    - __init__(orbits, family_type, system): 初始化轨道族
    - __len__(): 返回轨道数量
    - __getitem__(i): 获取第i条轨道
    - __iter__(): 迭代轨道
    - append(orbit): 添加轨道
    - get_states(): 获取所有初始状态数组 (n, 6)
    - get_periods(): 获取所有周期数组 (n,)
    - get_jacobi_constants(): 获取所有Jacobi常数数组 (n,)
    - save_to_file(filename): 保存到文件
    - load_from_file(filename): 从文件加载
    """

    def __init__(
        self,
        orbits: Optional[List[Orbit]] = None,
        family_type: Optional[str] = None,
        system: Optional[CR3BP_System] = None,
    ) -> None:
        """初始化轨道族

        参数：
        - orbits: Orbit对象列表
        - family_type: 轨道族类型
        - system: CR3BP_System对象
        """
        self.orbits: List[Orbit] = []
        if orbits is not None:
            if type(orbits) is Orbit:
                self.orbits = [orbits]
            else:
                if type(orbits) is list and type(orbits[0]) is Orbit:
                    self.orbits = orbits
                else:
                    self.orbits = []
        else:
            self.orbits = []
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
        """获取所有轨道的初始状态数组 (property版本)

        返回：
        - np.ndarray, shape (n, 6)
        """
        return self.get_states()

    @property
    def periods(self) -> npt.NDArray[np.floating]:
        """获取所有轨道的周期数组 (property版本)

        返回：
        - np.ndarray, shape (n,)
        """
        return self.get_periods()

    def __len__(self) -> int:
        """返回轨道数量"""
        return len(self.orbits)

    def __getitem__(self, index: int) -> Orbit:
        """获取指定索引的轨道"""
        return self.orbits[index]

    def __iter__(self):
        """迭代轨道"""
        return iter(self.orbits)

    def add_orbit(self, orbit: Orbit) -> None:
        """添加轨道到族中

        参数：
        - orbit: Orbit对象
        """
        self.orbits.append(orbit)

    def get_states(self) -> npt.NDArray[np.floating]:
        """获取所有轨道的初始状态数组

        返回：
        - np.ndarray, shape (n, 6)
        """
        return np.array([orbit.states[0] for orbit in self.orbits])

    def get_periods(self) -> npt.NDArray[np.floating]:
        """获取所有轨道的周期数组

        返回：
        - np.ndarray, shape (n,)
        """
        return np.array([orbit.period for orbit in self.orbits if orbit.period is not None])

    def get_jacobi_constants(self) -> npt.NDArray[np.floating]:
        """获取所有轨道的Jacobi常数数组

        返回：
        - np.ndarray, shape (n,)
        """
        if self.system is None:
            return np.array([])
        return np.array([self.system.get_jacobi_constant(orbit.states[0]) for orbit in self.orbits])

    def save_to_file(self, filename: Union[str, Path]) -> None:
        """保存轨道族到文件

        参数：
        - filename: 文件名（必须是相对路径，且指向当前工作目录内）

        异常：
        - ValueError: 当路径指向当前工作目录之外时抛出
        """
        filepath = Path(filename)

        # 自动创建目录
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
        cls, filename: Union[str, Path], system: Optional[CR3BP_System] = None
    ) -> "OrbitFamily":
        """从文件加载轨道族

        参数：
        - filename: 文件名（必须是相对路径，且指向当前工作目录内）
        - system: CR3BP_System对象

        返回：
        - OrbitFamily对象

        异常：
        - ValueError: 当路径指向当前工作目录之外时抛出
        """
        filepath = Path(filename)
        with open(filepath, "r") as f:
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
