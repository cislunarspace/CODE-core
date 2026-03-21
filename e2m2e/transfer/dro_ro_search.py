"""
DRO到RO转移轨道搜索模块

实现论文Cui et al. (2025)中的"搜索-优化"两步法的搜索阶段。
搜索变量: 出发点位置、α(切向速度比)、β(法向速度比)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

from ..core.orbit import Orbit
from ..core.dynamics import CR3BP_Dynamics
from ..core.system import CR3BP_System


@dataclass
class TransferSearchVariables:
    """转移搜索变量
    
    用于网格搜索阶段的变量集合。
    
    属性:
        departure_orbit: 出发点所在轨道
        departure_time_index: 出发点时间索引 (0到n_departure-1)
        alpha: 切向速度比 (0.5-2.5)
        beta: 法向速度比 (-0.5-0.5), 平面转移固定为0
    """
    departure_orbit: Orbit
    departure_time_index: int
    alpha: float
    beta: float = 0.0
    
    def __post_init__(self):
        """验证变量范围"""
        if not 0 <= self.departure_time_index:
            raise ValueError(f"departure_time_index必须为非负整数，当前为{self.departure_time_index}")
        if not 0.5 <= self.alpha <= 2.5:
            warnings.warn(f"alpha={self.alpha}超出推荐范围[0.5, 2.5]")
        if not -0.5 <= self.beta <= 0.5:
            warnings.warn(f"beta={self.beta}超出推荐范围[-0.5, 0.5]")
    
    @property
    def departure_state(self) -> np.ndarray:
        """获取出发点状态向量 [x, y, z, vx, vy, vz]"""
        n_states = len(self.departure_orbit.times)
        t_dep = (self.departure_time_index / n_states) * self.departure_orbit.period
        return self.departure_orbit.interpolate_at_time(t_dep)
    
    @property
    def departure_time(self) -> float:
        """获取出发点对应的时间"""
        n_states = len(self.departure_orbit.times)
        return (self.departure_time_index / n_states) * self.departure_orbit.period
    
    @property
    def velocity_ratios(self) -> Tuple[float, float]:
        """获取速度比例(α, β)"""
        return (self.alpha, self.beta)


@dataclass
class TransferSearchResult:
    """转移搜索结果
    
    存储单次搜索尝试的结果。
    
    属性:
        search_vars: 搜索变量
        transfer_trajectory: 转移轨迹状态序列 [n_steps, 6]
        transfer_times: 转移轨迹时间序列 [n_steps]
        transfer_time: 总转移时间
        min_distance: 与目标轨道的最小距离
        intersection_found: 是否找到相交
        intersection_point: 相交点状态(如果有)
        intersection_idx: 相交点索引
    """
    search_vars: TransferSearchVariables
    transfer_trajectory: Optional[np.ndarray] = None
    transfer_times: Optional[np.ndarray] = None
    transfer_time: float = 0.0
    min_distance: float = np.inf
    intersection_found: bool = False
    intersection_point: Optional[np.ndarray] = None
    intersection_idx: int = -1
    status: str = "pending"  # pending, success, failed, no_solution
    
    @property
    def is_feasible(self) -> bool:
        """判断是否为可行候选解"""
        return self.intersection_found or self.min_distance < 0.05


class DROROTransferSearch:
    """DRO到RO转移轨道搜索算法
    
    实现论文Section III.A的搜索阶段算法:
    1. 从出发点轨道等时间间隔采样
    2. 对每个出发点，网格化搜索α, β
    3. 前向积分获取转移轨迹
    4. 筛选与目标轨道相交或距离局部最小的候选解
    
    属性:
        system: CR3BP系统对象
        dynamics: CR3BP动力学对象
        max_transfer_time: 最大转移时间(无量纲)
        intersection_threshold: 相交检测阈值
        min_distance_threshold: 最小距离阈值
    """
    
    # 搜索参数默认值
    DEFAULT_ALPHA_RANGE = (0.5, 2.5)
    DEFAULT_BETA_RANGE = (-0.5, 0.5)
    DEFAULT_N_ALPHA = 101  # 切向速度比网格数
    DEFAULT_N_BETA = 21   # 法向速度比网格数
    DEFAULT_N_DEPARTURE = 200  # 出发点采样数
    DEFAULT_MAX_TRANSFER_TIME = 15.0  # 最大转移时间(无量纲CR3BP时间)
    
    def __init__(
        self,
        system: CR3BP_System,
        dynamics: CR3BP_Dynamics,
        max_transfer_time: float = DEFAULT_MAX_TRANSFER_TIME,
        intersection_threshold: float = 0.001,
        min_distance_threshold: float = 0.05,
    ):
        """初始化搜索算法
        
        参数:
            system: CR3BP系统对象
            dynamics: CR3BP动力学对象
            max_transfer_time: 最大转移时间(无量纲)
            intersection_threshold: 相交检测阈值
            min_distance_threshold: 最小距离阈值
        """
        self.system = system
        self.dynamics = dynamics
        self.mu = system.mu
        
        # 搜索参数
        self.max_transfer_time = max_transfer_time
        self.intersection_threshold = intersection_threshold
        self.min_distance_threshold = min_distance_threshold
        
        # 缓存
        self._departure_cache: Dict[int, np.ndarray] = {}
        self._velocity_cache: Dict[Tuple, np.ndarray] = {}
    
    def sample_departure_points(
        self,
        orbit: Orbit,
        n_points: int = DEFAULT_N_DEPARTURE
    ) -> List[np.ndarray]:
        """从轨道中等时间间隔采样出发点
        
        参数:
            orbit: 采样轨道
            n_points: 采样点数量
        
        返回:
            状态向量列表 [n_points, 6]
        """
        if orbit.period is None:
            raise ValueError("轨道必须具有周期属性")
        
        states = []
        times = np.linspace(0, orbit.period, n_points, endpoint=False)
        
        for t in times:
            state = orbit.interpolate_at_time(t)
            states.append(state)
        
        return states
    
    def compute_departure_velocity(
        self,
        state: np.ndarray,
        alpha: float,
        beta: float = 0.0
    ) -> np.ndarray:
        """根据α, β计算出发速度
        
        速度计算公式(基于论文Eq.11-12):
        v_injection = alpha * v_tangential + beta * v_normal
        
        参数:
            state: 出发点状态 [x, y, z, vx, vy, vz]
            alpha: 切向速度比
            beta: 法向速度比(平面转移为0)
        
        返回:
            注入速度向量 [vx, vy, vz]
        """
        pos = state[:3]
        vel = state[3:]
        
        # 轨道面法向(CR3BP中为z轴)
        normal = np.array([0.0, 0.0, 1.0])
        
        # 速度大小
        v_mag = np.linalg.norm(vel)
        if v_mag < 1e-10:
            warnings.warn("出发点速度接近零")
            return vel
        
        # 切向单位向量(沿速度方向)
        tangential = vel / v_mag
        
        # 法向单位向量(垂直于速度和法向组成的平面)
        normal_dir = np.cross(tangential, normal)
        norm_nd = np.linalg.norm(normal_dir)
        if norm_nd < 1e-10:
            # 速度平行于z轴，使用x轴作为备份
            normal_dir = np.array([1.0, 0.0, 0.0])
        else:
            normal_dir = normal_dir / norm_nd
        
        # 注入速度 = alpha * 切向分量 + beta * 法向分量
        v_injection = alpha * v_mag * tangential + beta * v_mag * normal_dir
        
        return v_injection
    
    def forward_integrate(
        self,
        initial_state: np.ndarray,
        t_span: Tuple[float, float],
        t_eval: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """前向积分转移弧
        
        参数:
            initial_state: 初始状态 [x, y, z, vx, vy, vz]
            t_span: 积分时间范围 (t0, tf)
            t_eval: 评估时间点
        
        返回:
            (times, states): 时间序列和状态序列
        """
        if t_eval is None:
            # 自动生成评估点
            n_steps = int((t_span[1] - t_span[0]) / 0.01) + 1
            t_eval = np.linspace(t_span[0], t_span[1], n_steps)
        
        # 扩展初始状态(如果需要STM)
        result = self.dynamics.propagate(
            initial_state=initial_state,
            t_span=t_span,
            t_eval=t_eval,
            with_stm=False
        )
        
        times = result.times
        states = result.states
        
        return times, states
    
    def compute_min_distance_to_orbit(
        self,
        trajectory_states: np.ndarray,
        target_orbit: Orbit
    ) -> Tuple[float, int]:
        """计算轨迹与目标轨道的最小距离
        
        参数:
            trajectory_states: 转移轨迹状态 [n_steps, 6]
            target_orbit: 目标轨道
        
        返回:
            (min_distance, min_idx): 最小距离和对应的轨迹索引
        """
        min_dist = np.inf
        min_idx = -1
        
        target_states = target_orbit.states
        
        for i, traj_state in enumerate(trajectory_states):
            pos_traj = traj_state[:3]
            for orb_state in target_states:
                pos_orb = orb_state[:3]
                dist = np.linalg.norm(pos_traj - pos_orb)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = i
        
        return min_dist, min_idx
    
    def find_intersection(
        self,
        trajectory_states: np.ndarray,
        target_orbit: Orbit
    ) -> Tuple[bool, np.ndarray, int]:
        """检测转移轨迹与目标轨道是否相交
        
        参数:
            trajectory_states: 转移轨迹状态 [n_steps, 6]
            target_orbit: 目标轨道
        
        返回:
            (found, intersection_state, idx): 是否找到相交、相交点状态、轨迹索引
        """
        target_states = target_orbit.states
        
        for i, traj_state in enumerate(trajectory_states):
            pos_traj = traj_state[:3]
            for j, orb_state in enumerate(target_states):
                pos_orb = orb_state[:3]
                dist = np.linalg.norm(pos_traj - pos_orb)
                if dist < self.intersection_threshold:
                    return True, traj_state, i
        
        return False, np.zeros(6), -1
    
    def search_single_departure(
        self,
        departure_state: np.ndarray,
        arrival_orbit: Orbit,
        alpha_range: Tuple[float, float] = DEFAULT_ALPHA_RANGE,
        beta_range: Tuple[float, float] = DEFAULT_BETA_RANGE,
        n_alpha: int = DEFAULT_N_ALPHA,
        n_beta: int = DEFAULT_N_BETA
    ) -> List[TransferSearchResult]:
        """对单个出发点进行网格搜索
        
        参数:
            departure_state: 出发点状态
            arrival_orbit: 目标轨道
            alpha_range: α搜索范围
            beta_range: β搜索范围
            n_alpha: α方向网格数
            n_beta: β方向网格数
        
        返回:
            可行候选解列表
        """
        results = []
        
        # 生成网格
        alphas = np.linspace(alpha_range[0], alpha_range[1], n_alpha)
        betas = np.linspace(beta_range[0], beta_range[1], n_beta)
        
        for alpha in alphas:
            for beta in betas:
                # 计算出发速度
                v_injection = self.compute_departure_velocity(departure_state, alpha, beta)
                
                # 构建完整状态
                full_state = np.concatenate([departure_state[:3], v_injection])
                
                # 前向积分
                try:
                    times, states = self.forward_integrate(
                        initial_state=full_state,
                        t_span=(0.0, self.max_transfer_time)
                    )
                    
                    # 检测相交
                    intersection_found, int_state, int_idx = self.find_intersection(
                        states, arrival_orbit
                    )
                    
                    # 计算最小距离
                    min_dist, min_idx = self.compute_min_distance_to_orbit(
                        states, arrival_orbit
                    )
                    
                    # 创建搜索变量
                    search_vars = TransferSearchVariables(
                        departure_orbit=arrival_orbit,  # 临时借用，实际需要传入真实orbit
                        departure_time_index=0,  # 临时值
                        alpha=alpha,
                        beta=beta
                    )
                    
                    # 创建结果
                    result = TransferSearchResult(
                        search_vars=search_vars,
                        transfer_trajectory=states,
                        transfer_times=times,
                        transfer_time=times[-1] if len(times) > 0 else 0.0,
                        min_distance=min_dist,
                        intersection_found=intersection_found,
                        intersection_point=int_state if intersection_found else None,
                        intersection_idx=int_idx if intersection_found else -1,
                        status="success" if (intersection_found or min_dist < self.min_distance_threshold) else "no_solution"
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    # 积分失败，跳过
                    continue
        
        return results
    
    def grid_search(
        self,
        departure_orbit: Orbit,
        arrival_orbit: Orbit,
        alpha_range: Tuple[float, float] = DEFAULT_ALPHA_RANGE,
        beta_range: Tuple[float, float] = DEFAULT_BETA_RANGE,
        n_alpha: int = DEFAULT_N_ALPHA,
        n_beta: int = DEFAULT_N_BETA,
        n_departure: int = DEFAULT_N_DEPARTURE,
        parallel: bool = True,
        n_workers: int = 4
    ) -> List[TransferSearchResult]:
        """网格搜索DRO到RO的所有可能转移
        
        参数:
            departure_orbit: 出发点轨道
            arrival_orbit: 目标轨道
            alpha_range: α搜索范围
            beta_range: β搜索范围
            n_alpha: α方向网格数
            n_beta: β方向网格数
            n_departure: 出发点采样数
            parallel: 是否并行搜索
            n_workers: 并行worker数
        
        返回:
            可行候选解列表
        """
        print(f"\n开始网格搜索:")
        print(f"  出发点采样数: {n_departure}")
        print(f"  α范围: [{alpha_range[0]}, {alpha_range[1]}], 网格数: {n_alpha}")
        print(f"  β范围: [{beta_range[0]}, {beta_range[1]}], 网格数: {n_beta}")
        print(f"  总搜索点数: {n_departure * n_alpha * n_beta}")
        
        # 采样出发点
        departure_states = self.sample_departure_points(departure_orbit, n_departure)
        
        all_results = []
        total_combinations = len(departure_states) * n_alpha * n_beta
        processed = 0
        
        if parallel:
            # 并行搜索
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = []
                for dep_state in departure_states:
                    future = executor.submit(
                        self.search_single_departure,
                        dep_state,
                        arrival_orbit,
                        alpha_range,
                        beta_range,
                        n_alpha,
                        n_beta
                    )
                    futures.append(future)
                
                for future in as_completed(futures):
                    try:
                        results = future.result()
                        all_results.extend(results)
                        processed += 1
                        if processed % 20 == 0:
                            print(f"  进度: {processed}/{len(departure_states)} ({100*processed/len(departure_states):.1f}%)")
                    except Exception as e:
                        print(f"  搜索失败: {e}")
        else:
            # 串行搜索
            for i, dep_state in enumerate(departure_states):
                results = self.search_single_departure(
                    dep_state,
                    arrival_orbit,
                    alpha_range,
                    beta_range,
                    n_alpha,
                    n_beta
                )
                all_results.extend(results)
                processed += 1
                if processed % 20 == 0:
                    print(f"  进度: {processed}/{len(departure_states)} ({100*processed/len(departure_states):.1f}%)")
        
        # 筛选可行解
        feasible_results = [r for r in all_results if r.is_feasible]
        
        print(f"\n搜索完成:")
        print(f"  总搜索点数: {total_combinations}")
        print(f"  可行候选解: {len(feasible_results)}")
        
        return feasible_results
    
    def filter_candidates_by_local_minimum(
        self,
        results: List[TransferSearchResult]
    ) -> List[TransferSearchResult]:
        """筛选距离局部最小的候选解
        
        参数:
            results: 搜索结果列表
        
        返回:
            满足局部最小条件的候选解
        """
        # 按alpha值分组
        alpha_groups: Dict[float, List[TransferSearchResult]] = {}
        for r in results:
            alpha = round(r.search_vars.alpha, 2)  # 离散化
            if alpha not in alpha_groups:
                alpha_groups[alpha] = []
            alpha_groups[alpha].append(r)
        
        filtered = []
        
        for alpha, group in alpha_groups.items():
            # 按转移时间排序
            group.sort(key=lambda x: x.transfer_time)
            
            # 检查相邻点的距离梯度
            for i in range(1, len(group) - 1):
                prev_dist = group[i-1].min_distance
                curr_dist = group[i].min_distance
                next_dist = group[i+1].min_distance
                
                # 局部最小: 前一个和后一个都比当前大
                if curr_dist < prev_dist and curr_dist < next_dist:
                    if curr_dist < self.min_distance_threshold:
                        filtered.append(group[i])
        
        return filtered
