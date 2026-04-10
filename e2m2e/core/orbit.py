"""轨道数据模块

本模块定义了圆型限制性三体问题（CR3BP）中轨道数据的表示与处理逻辑，
是 e2m2e 四层架构中 core 层的核心组件之一。

主要类：
    Orbit: 单条轨道的数据容器，支持属性计算、序列化/反序列化和稳定性分析。
    OrbitFamily: 轨道族容器，用于存储和管理多条同族轨道（如 halo 族、Lyapunov 族等）。

依赖关系：
    - CR3BP_System: 提供质量参数、Jacobi 常数计算等物理模型
    - CR3BP_Dynamics: 提供状态转移矩阵（STM）的计算能力，用于单值矩阵求解

典型用法::

    system = CR3BP_System.from_known_system("earth_moon")
    orbit = Orbit(states=np.array([[...]]), times=np.array([...]), system=system)
    orbit.compute_stability(CR3BP_Dynamics(system))
    orbit.save_to_file("orbit.json")
"""

from __future__ import annotations

import os
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from pathlib import Path

import numpy.typing as npt

from .system import CR3BP_System
from .dynamics import CR3BP_Dynamics


class Orbit:
    """单条轨道的数据容器与处理工具

    Orbit 是 e2m2e 中最基本的轨道数据结构，用于存储、计算和持久化
    一条 CR3BP 轨道的全部信息。每个 Orbit 对象包含：

    - **状态序列** ``states``: 形状 ``(n, 6)`` 的数组，每行为
      ``[x, y, z, vx, vy, vz]``，采用 CR3BP 旋转坐标系下的无量纲单位。
    - **时间序列** ``times``: 形状 ``(n,)`` 的数组，对应每个状态的时间戳。
    - **物理属性**: 周期、振幅、极值、Jacobi 常数等，由 ``compute_basic_properties()``
      在初始化时自动计算。
    - **稳定性信息**: 单值矩阵、特征值、Lyapunov 指数等，需通过
      ``compute_monodromy_matrix()`` / ``compute_stability()`` 手动触发。
    - **元数据**: 创建时间、来源、描述标签等，随序列化一起保存。

    Attributes:
        states: 状态序列 ``[x, y, z, vx, vy, vz]``，形状为 ``(n, 6)``
        times: 时间序列，形状为 ``(n,)``
        system: 关联的 CR3BP_System 对象，用于 Jacobi 常数计算
        jacobi_constants: Jacobi 常数序列，形状为 ``(n,)``，需 system 不为 None
        stability_indices: 稳定性指标（由外部算法填充）
        family_type: 轨道族类型，如 ``'halo'``、``'lyapunov'`` 等
        parameters: 轨道参数字典，供外部算法存储附加信息
        period: 轨道周期（无量纲时间），由零交叉检测自动估计
        amplitudes: 各方向振幅，如 ``{'x': amp_x, 'y': amp_y, 'z': amp_z}``
        extrema: 极值点，如 ``{'x_max': xmax, 'x_min': xmin, ...}``
        mean_state: 平均状态向量，形状 ``(6,)``
        monodromy_matrix: 单值矩阵（6x6），通过 ``compute_monodromy_matrix()`` 计算
        eigenvalues: 单值矩阵的特征值，复数数组
        stability: 稳定性标签，取值为 ``'stable'``、``'unstable'`` 或 ``'marginally_stable'``
        lyapunov_exponents: Lyapunov 指数数组
        center: 轨道中心点，取位置分量均值，形状 ``(3,)``
        radius: 轨道半径（保留字段，用于近圆形轨道）
        shape: 轨道形状特征（保留字段）
        orientation: 轨道取向（保留字段）
        is_periodic: 是否为周期轨道，由 ``_check_periodicity()`` 判定
        is_quasi_periodic: 是否为拟周期轨道（保留字段，需外部设置）
        is_chaotic: 是否为混沌轨道（保留字段，需外部设置）
        periodicity_error: 周期性误差，即首末状态的欧氏距离
        segments: 轨道段列表，用于长轨迹分段存储
        segment_indices: 分段索引列表，记录每段在 states 数组中的位置
        metadata: 轨道元数据字典，含创建时间、来源、描述等
    """

    # 支持的轨道族类型枚举，用于序列化时验证 family_type 字段
    VALID_FAMILY_TYPES = [
        "halo",
        "lyapunov",
        "vertical",
        "axial",
        "butterfly",
        "dragonfly",
    ]
    # 状态向量的六个分量名称，按 [x, y, z, vx, vy, vz] 顺序排列
    VALID_COMPONENTS = ["x", "y", "z", "vx", "vy", "vz"]

    def __init__(
        self,
        states: npt.ArrayLike,
        times: npt.ArrayLike,
        system: Optional[CR3BP_System] = None,
    ) -> None:
        """初始化轨道对象

        根据给定的状态序列和时间序列创建轨道实例，并自动计算基本属性
        （振幅、极值、中心点、周期估计等）。

        Args:
            states: 状态序列，形状为 ``(n, 6)`` 或 ``(6,)`` 的数组，
                每行格式为 ``[x, y, z, vx, vy, vz]``。
            times: 时间序列，形状为 ``(n,)`` 的数组，需与 states 行数一致。
            system: CR3BP_System 对象。提供后可自动计算 Jacobi 常数。

        Raises:
            ValueError: 当状态分量数不等于 6，或时间序列与状态序列长度不一致时抛出。
        """
        # 将输入转换为 numpy 数组并存储
        self.states = np.array(states)
        self.times = np.array(times)
        self.system = system

        # 数据验证：单行状态自动升维为 (1, 6)
        if self.states.ndim == 1:
            self.states = self.states.reshape(1, -1)
        # 验证状态向量必须包含 6 个分量 [x, y, z, vx, vy, vz]
        if self.states.shape[1] != 6:
            raise ValueError(f"状态序列必须包含6个分量，当前为{self.states.shape[1]}个")
        # 验证时间序列长度与状态序列行数一致
        if len(self.times) != self.states.shape[0]:
            raise ValueError("时间序列长度必须与状态序列长度一致")

        # ---- 以下为轨道属性初始化，均设为合理的默认值 ----

        # 物理属性
        self.jacobi_constants = None  # Jacobi 常数序列，需 system 不为 None
        self.stability_indices = None  # 稳定性指标，由 StabilityAnalysis 填充
        self.family_type: Optional[str] = None  # 轨道族类型标识
        self.parameters = {}  # 附加参数字典，供外部算法使用

        # 几何属性
        self.period: float | None = None  # 轨道周期（无量纲时间）
        self.amplitudes: dict = {}  # 各方向振幅 {'x': ..., 'y': ..., 'z': ...}
        self.extrema: dict = {}  # 各方向极值 {'x_max': ..., 'x_min': ..., ...}
        self.mean_state = None  # 平均状态向量

        # 稳定性属性
        self.monodromy_matrix = None  # 单值矩阵 (6x6)
        self.eigenvalues = None  # 单值矩阵特征值（复数）
        self.stability = None  # 稳定性标签
        self.lyapunov_exponents = None  # Lyapunov 指数

        # 几何特征（保留字段，用于近圆形轨道分析）
        self.center = None  # 轨道中心点（位置分量均值）
        self.radius = None  # 轨道半径
        self.shape = None  # 轨道形状特征
        self.orientation = None  # 轨道取向

        # 轨道类型标志
        self.is_periodic = False  # 周期轨道标志
        self.is_quasi_periodic = False  # 拟周期轨道标志
        self.is_chaotic = False  # 混沌轨道标志
        self.periodicity_error = None  # 周期性误差（首末状态距离）

        # 分段存储（用于多条拼接的长轨迹）
        self.segments = []  # 轨道段列表
        self.segment_indices = []  # 分段索引

        # 元数据：记录创建时间、来源等信息，随序列化一起保存
        self.metadata = {
            "created": datetime.now().isoformat(),
            "source": "e2m2e library",
            "description": "",
            "tags": [],
        }

        # 初始化完成后立即计算基本属性
        self.compute_basic_properties()

    def compute_basic_properties(self) -> None:
        """计算轨道的基本几何与物理属性

        自动执行以下计算：
        1. **Jacobi 常数序列**: 当 system 不为 None 时，对每个状态点计算 Jacobi 常数。
        2. **平均状态**: 对所有状态沿时间轴求均值。
        3. **位置极值与振幅**: 分别对 x、y、z 三个位置分量计算最大值、最小值和半振幅。
        4. **轨道中心**: 取平均状态的位置分量作为中心点。
        5. **周期估计**: 通过 x 方向的零交叉检测估计轨道周期。

        Note:
            本方法在 ``__init__`` 中自动调用，一般无需手动执行。
        """
        # 仅在关联了 CR3BP_System 时才能计算 Jacobi 常数
        if self.system is not None:
            self.jacobi_constants = np.array(
                [self.system.get_jacobi_constant(state) for state in self.states]
            )

        # 计算状态沿时间轴的算术平均值
        self.mean_state = np.mean(self.states, axis=0)

        # 只对前三个位置分量 (x, y, z) 计算极值和振幅
        for i, component in enumerate(self.VALID_COMPONENTS[:3]):
            values = self.states[:, i]
            self.extrema[f"{component}_max"] = np.max(values)
            self.extrema[f"{component}_min"] = np.min(values)
            # 振幅定义为 (最大值 - 最小值) / 2，即半峰全幅
            self.amplitudes[component] = (np.max(values) - np.min(values)) / 2

        # 轨道中心取平均状态的位置分量
        self.center = self.mean_state[:3]

        # 尝试通过零交叉检测估计轨道周期
        self._estimate_period()

    def _estimate_period(self) -> None:
        """通过 x 方向零交叉检测估计轨道周期

        检测 ``x(t) - x_center`` 的符号变化来定位零交叉点，
        取前两个零交叉点之间的时间差作为半周期，乘以 2 得到完整周期估计。
        此方法适用于大致对称的周期轨道（如 Lyapunov、Halo 轨道），
        对非周期或高度非对称轨迹可能给出不准确的结果。

        Note:
            如果检测到至少 2 个零交叉点，还会自动调用 ``_check_periodicity()``
            验证首末状态的吻合程度。
        """
        # 至少需要 2 个时间点才能进行零交叉检测
        if len(self.times) < 2:
            return

        # 计算 x(t) 相对于中心点的偏移序列
        x_values = self.states[:, 0]
        # np.diff(np.sign(...)) 检测符号变化，非零位置即为零交叉点
        zero_crossings = np.where(np.diff(np.sign(x_values - self.center[0])))[0]

        # 至少需要 2 个零交叉点才能估计半周期
        if len(zero_crossings) >= 2:
            t1 = self.times[zero_crossings[0]]
            t2 = self.times[zero_crossings[1]]
            # 半周期 = 相邻两个零交叉点的时间差，完整周期 = 2 × 半周期
            self.period = 2 * (t2 - t1)

            # 周期估计完成后，验证轨道的周期性
            self._check_periodicity()

    def _check_periodicity(self) -> None:
        """验证轨道的周期性

        通过比较初始状态与经过一个周期后的状态之间的欧氏距离来判断周期性：
        - 在时间序列中找到最接近 ``t0 + period`` 时刻的状态点
        - 计算该点与初始状态的欧氏距离作为 ``periodicity_error``
        - 当误差小于 ``1e-6`` 时标记为周期轨道

        Note:
            此方法由 ``_estimate_period()`` 在成功估计周期后自动调用。
        """
        if self.period is None:
            return

        # 取初始状态
        start_state = self.states[0]
        # 计算经过一个周期后的目标时间
        target_t = float(self.times[0]) + float(self.period)
        # 在时间序列中找到最接近目标时间的索引
        idx = int(np.argmin(np.abs(self.times - target_t)))
        # 取该时间点对应的状态
        end_state = self.states[idx]

        # 周期性误差 = 首末状态的欧氏距离
        self.periodicity_error = np.linalg.norm(start_state - end_state)

        # 以 1e-6 为阈值判定周期性
        tolerance = 1e-6
        self.is_periodic = self.periodicity_error < tolerance

        # 更新元数据中的描述信息
        if self.is_periodic:
            self.metadata["description"] = "Periodic orbit"
        else:
            self.metadata["description"] = "Non-periodic trajectory"

    def compute_monodromy_matrix(self, dynamics: CR3BP_Dynamics) -> npt.NDArray[np.floating]:
        """计算轨道的单值矩阵（Monodromy Matrix）

        单值矩阵是状态转移矩阵（STM）沿一个完整周期的积分结果，
        反映了周期轨道附近小偏差的线性演化特性。其特征值（Floquet 乘子）
        可用于判断轨道的稳定性。

        Args:
            dynamics: CR3BP_Dynamics 对象，提供 STM 积分能力。

        Returns:
            单值矩阵，形状为 ``(6, 6)`` 的 numpy 数组。

        Raises:
            ValueError: 当轨道周期未知（``self.period is None``）时抛出。
        """
        if self.period is None:
            raise ValueError("无法计算单值矩阵：轨道周期未知")

        # 以初始状态和轨道周期积分 STM，得到单值矩阵
        initial_state = self.states[0]
        self.monodromy_matrix = dynamics.compute_state_transition_matrix(initial_state, self.period)

        # 同时计算特征值（Floquet 乘子），供后续稳定性分析使用
        self.eigenvalues = np.linalg.eigvals(self.monodromy_matrix)

        return self.monodromy_matrix

    def compute_stability(self, dynamics: CR3BP_Dynamics) -> Dict[str, Any]:
        """计算轨道的稳定性指标

        基于单值矩阵的特征值（Floquet 乘子）判断轨道稳定性：
        - 所有乘子模长接近 1 → 稳定（stable）
        - 存在乘子模长 > 1 → 不稳定（unstable）
        - 其余情况 → 临界稳定（marginally_stable）

        同时计算 Lyapunov 指数，定义为 ``ln(|eigenvalue|) / period``。

        Args:
            dynamics: CR3BP_Dynamics 对象，用于计算单值矩阵（如果尚未计算）。

        Returns:
            稳定性分析结果字典，包含以下键:
            - ``stability`` (str): 稳定性标签
            - ``eigenvalues`` (ndarray): Floquet 乘子（复数数组）
            - ``max_deviation`` (float): 最大乘子模长偏差
            - ``lyapunov_exponents`` (ndarray): Lyapunov 指数数组
        """
        # 如果尚未计算单值矩阵，先计算
        if self.monodromy_matrix is None:
            self.compute_monodromy_matrix(dynamics)

        # 提取特征值及其模长
        eigenvalues = self.eigenvalues
        magnitudes = np.abs(eigenvalues)

        # 计算最大模长偏差：所有 Floquet 乘子的模长与 1 的最大偏差
        max_deviation = np.max(np.abs(magnitudes - 1.0))

        # 稳定性判定逻辑
        if max_deviation < 1e-6:
            # 所有乘子模长都接近 1 → 稳定
            self.stability = "stable"
        elif np.any(magnitudes > 1.0 + 1e-6):
            # 存在模长显著大于 1 的乘子 → 不稳定
            self.stability = "unstable"
        else:
            # 其余情况 → 临界稳定
            self.stability = "marginally_stable"

        # Lyapunov 指数 = ln(|eigenvalue|) / period
        self.lyapunov_exponents = np.log(magnitudes) / self.period

        return {
            "stability": self.stability,
            "eigenvalues": eigenvalues,
            "max_deviation": max_deviation,
            "lyapunov_exponents": self.lyapunov_exponents,
        }

    def get_period(self) -> Optional[float]:
        """获取轨道周期

        Returns:
            轨道周期（无量纲时间），如果尚未估计则返回 ``None``。
        """
        return self.period

    def get_amplitude(self, direction: str) -> float:
        """获取指定方向的轨道振幅

        Args:
            direction: 方向标识，取值为 ``'x'``、``'y'`` 或 ``'z'``。

        Returns:
            该方向的半振幅值，定义为 ``(max - min) / 2``。

        Raises:
            ValueError: 当 ``direction`` 不在已计算的振幅字典中时抛出。
        """
        if direction not in self.amplitudes:
            raise ValueError(f"无效的方向: {direction}。可用方向: {list(self.amplitudes.keys())}")
        return self.amplitudes[direction]

    def save_to_file(self, filename: Union[str, Path]) -> None:
        """将轨道数据序列化保存到 JSON 文件

        采用 JSON 格式保存，包含以下顶层字段：
        - ``states``: 状态序列（列表的列表）
        - ``times``: 时间序列（列表）
        - ``metadata``: 元数据字典
        - ``properties``: 物理属性（周期、振幅、极值等）
        - ``timestamp``: 保存时间戳

        文件格式为"单轨道格式"，与轨道族格式（含 ``"orbits"`` 数组）互不兼容。
        可通过 ``load_from_file()`` 反序列化。

        Args:
            filename: 目标文件路径，支持 ``str`` 或 ``Path`` 对象。
                如果父目录不存在，会自动递归创建。

        Raises:
            ValueError: 当路径指向当前工作目录之外时抛出。
        """
        filepath = Path(filename)

        # 如果父目录不存在，自动创建
        dirpath = filepath.parent
        if not dirpath.exists():
            dirpath.mkdir(parents=True)

        # 记录保存时间戳到元数据
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metadata["saved_timestamp"] = timestamp

        # 构建序列化数据字典
        data = {
            "states": self.states.tolist(),  # numpy 数组 → 嵌套列表
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

        # 以 JSON 格式写入文件，indent=2 保证可读性
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_file(
        cls,
        filename: Union[str, Path],
        system: Optional[CR3BP_System] = None,
        orbit_index: Optional[int] = None,
    ) -> "Orbit":
        """从 JSON 文件反序列化加载轨道数据

        支持两种文件格式的自动识别：
        1. **单轨道格式**: 包含 ``"states"``、``"times"``、``"metadata"``、``"properties"``
           顶层字段，直接返回 ``Orbit`` 对象。
        2. **轨道族格式**: 包含 ``"orbits"`` 数组字段，需指定 ``orbit_index``
           以加载其中的某一条轨道。

        反序列化时，已保存的物理属性（周期、振幅、极值等）会从文件中恢复，
        覆盖 ``__init__`` 中 ``compute_basic_properties()`` 的自动计算结果。

        Args:
            filename: JSON 文件路径，支持 ``str`` 或 ``Path`` 对象。
            system: CR3BP_System 对象（可选）。提供后可在反序列化时重新计算
                Jacobi 常数等系统相关属性。
            orbit_index: 轨道索引（可选）。当文件为轨道族格式时，
                用于指定加载第几条轨道（从 0 开始）。

        Returns:
            加载得到的 Orbit 对象，已恢复保存时的物理属性和元数据。

        Raises:
            ValueError: 当文件为轨道族格式但未提供 ``orbit_index`` 时抛出。
            IndexError: 当 ``orbit_index`` 超出轨道族中轨道数量范围时抛出。

        Note:
            - 单轨道格式与轨道族格式通过 ``"orbits"`` 键的存在与否来区分。
            - 反序列化后，稳定性相关属性（单值矩阵、特征值等）不会自动恢复，
              需要重新调用 ``compute_stability()``。
        """
        filepath = Path(filename)
        # 读取并解析 JSON 文件
        with open(filepath, "r") as f:
            data = json.load(f)

        # 格式自动识别：检查是否存在 "orbits" 键（轨道族格式标志）
        if "orbits" in data:
            # 轨道族格式：必须指定 orbit_index
            if orbit_index is None:
                raise ValueError(
                    f"文件 '{filename}' 是轨道族格式，需要提供 orbit_index 参数指定要加载的轨道"
                )
            # 验证 orbit_index 范围
            orbits_data = data["orbits"]
            if orbit_index < 0 or orbit_index >= len(orbits_data):
                raise IndexError(
                    f"orbit_index={orbit_index} 超出范围，轨道族共有 {len(orbits_data)} 条轨道"
                )
            # 从轨道族中提取指定索引的轨道数据
            orbit_data = orbits_data[orbit_index]
        else:
            # 单轨道格式：直接使用整个 data 字典
            orbit_data = data

        # 从 JSON 列表恢复 numpy 数组
        states = np.array(orbit_data["states"])
        times = np.array(orbit_data["times"])
        # 创建 Orbit 实例（__init__ 会调用 compute_basic_properties）
        orbit = cls(states, times, system)

        # 恢复元数据，优先使用轨道级 metadata，回退到文件级 metadata
        orbit.metadata = orbit_data.get("metadata", data.get("metadata", {}))

        # 恢复保存时的物理属性，覆盖 compute_basic_properties() 的自动计算结果
        # 兼容新旧两种存储格式：新格式属性在 "properties" 子字典中，旧格式在顶层
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
        """返回轨道的简短字符串描述

        周期轨道显示周期和振幅，非周期轨道显示时间序列长度和振幅。

        Returns:
            轨道描述字符串。
        """
        if self.is_periodic:
            return f"Orbit(period={self.period:.4f}, amplitudes={self.amplitudes}, periodic=True)"
        else:
            return f"Orbit(length={len(self.times)}, amplitudes={self.amplitudes})"

    def __repr__(self):
        """返回轨道的详细字符串表示，用于调试

        Returns:
            包含状态矩阵形状、时间序列长度、周期和系统信息的字符串。
        """
        return (
            f"Orbit(states_shape={self.states.shape}, times_length={len(self.times)}, "
            f"period={self.period}, system={self.system})"
        )

    def copy(self) -> "Orbit":
        """创建轨道的深拷贝

        Returns:
            新的 Orbit 对象，包含相同的数据但独立引用
        """
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

        new_orbit.period = self.period
        new_orbit.amplitudes = self.amplitudes.copy()
        new_orbit.extrema = self.extrema.copy()
        new_orbit.mean_state = self.mean_state.copy() if self.mean_state is not None else None

        new_orbit.monodromy_matrix = (
            self.monodromy_matrix.copy() if self.monodromy_matrix is not None else None
        )
        new_orbit.eigenvalues = self.eigenvalues.copy() if self.eigenvalues is not None else None
        new_orbit.stability = self.stability
        new_orbit.lyapunov_exponents = (
            self.lyapunov_exponents.copy() if self.lyapunov_exponents is not None else None
        )

        new_orbit.center = self.center.copy() if self.center is not None else None
        new_orbit.radius = self.radius
        new_orbit.shape = self.shape
        new_orbit.orientation = self.orientation

        new_orbit.is_periodic = self.is_periodic
        new_orbit.is_quasi_periodic = self.is_quasi_periodic
        new_orbit.is_chaotic = self.is_chaotic
        new_orbit.periodicity_error = self.periodicity_error

        new_orbit.segments = self.segments.copy()
        new_orbit.segment_indices = self.segment_indices.copy()

        new_orbit.metadata = self.metadata.copy()

        return new_orbit


class OrbitFamily:
    """轨道族容器

    用于存储和管理多个 Orbit 对象组成的轨道族。

    Attributes:
        orbits: Orbit 对象列表
        family_type: 轨道族类型 (halo, lyapunov, dro, ro, etc.)
        system: 关联的 CR3BP_System 对象
        metadata: 轨道族元数据
    """

    def __init__(
        self,
        orbits: Optional[List[Orbit]] = None,
        family_type: Optional[str] = None,
        system: Optional[CR3BP_System] = None,
    ) -> None:
        """初始化轨道族

        Args:
            orbits: Orbit 对象列表
            family_type: 轨道族类型
            system: CR3BP_System 对象
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
        """获取所有轨道的初始状态数组

        Returns:
            初始状态数组，形状为 (n, 6)
        """
        return self.get_states()

    @property
    def periods(self) -> npt.NDArray[np.floating]:
        """获取所有轨道的周期数组

        Returns:
            周期数组，形状为 (n,)
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

        Args:
            orbit: Orbit 对象
        """
        self.orbits.append(orbit)

    def get_states(self) -> npt.NDArray[np.floating]:
        """获取所有轨道的初始状态数组

        Returns:
            初始状态数组，形状为 (n, 6)
        """
        return np.array([orbit.states[0] for orbit in self.orbits])

    def get_periods(self) -> npt.NDArray[np.floating]:
        """获取所有轨道的周期数组

        Returns:
            周期数组，形状为 (n,)
        """
        return np.array([orbit.period for orbit in self.orbits if orbit.period is not None])

    def get_jacobi_constants(self) -> npt.NDArray[np.floating]:
        """获取所有轨道的 Jacobi 常数数组

        Returns:
            Jacobi 常数数组，形状为 (n,)
        """
        if self.system is None:
            return np.array([])
        return np.array([self.system.get_jacobi_constant(orbit.states[0]) for orbit in self.orbits])

    def save_to_file(self, filename: Union[str, Path]) -> None:
        """保存轨道族到文件

        Args:
            filename: 文件名（必须是相对路径，且指向当前工作目录内）

        Raises:
            ValueError: 当路径指向当前工作目录之外时抛出
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
        cls, filename: Union[str, Path], system: Optional[CR3BP_System] = None
    ) -> "OrbitFamily":
        """从文件加载轨道族

        Args:
            filename: 文件名（必须是相对路径，且指向当前工作目录内）
            system: CR3BP_System 对象

        Returns:
            OrbitFamily 对象

        Raises:
            ValueError: 当路径指向当前工作目录之外时抛出
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
