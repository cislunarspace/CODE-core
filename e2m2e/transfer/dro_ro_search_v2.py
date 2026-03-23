"""
DRO到RO转移轨道搜索模块 V2

实现论文Cui et al. (2025)中的"搜索-优化"两步法的搜索阶段。
相比V1版本，修复了以下bug:
- BUG-001: departure_orbit引用错误
- BUG-002: α,β速度扰动计算错误
- BUG-003: 距离计算使用嵌套循环，效率低
- BUG-004: 缺少碰撞检测

搜索变量: 出发点位置、α(切向速度比)、β(法向速度比)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any
import warnings

from ..core.orbit import Orbit
from ..core.dynamics import CR3BP_Dynamics
from ..core.system import CR3BP_System


@dataclass
class TransferSearchConfig:
    """转移搜索配置
    
    搜索参数配置类，包含搜索范围、网格密度、阈值等参数。
    按照论文Table 3设置默认值。
    """
    # α (切向速度比) 搜索范围
    alpha_min: float = 0.5
    alpha_max: float = 2.5
    n_alpha: int = 101
    
    # β (法向速度比) 搜索范围
    beta_min: float = -0.5
    beta_max: float = 0.5
    n_beta: int = 21
    
    # 出发点采样数量
    n_departure: int = 200
    
    # 最大转移时间 (CR3BP无量纲时间)
    max_transfer_time: float = 15.0
    
    # 检测阈值
    intersection_threshold: float = 0.001  # 相交检测阈值
    min_distance_threshold: float = 0.05   # 最小距离阈值
    
    # 碰撞检测半径 (无量纲)
    collision_earth_radius: float = 0.999    # Earth exclusion radius
    collision_moon_radius: float = 0.999     # Moon exclusion radius
    
    # 积分参数
    integration_dt: float = 0.001  # 积分时间步长
    
    @property
    def alpha_grid(self) -> np.ndarray:
        """α网格点"""
        return np.linspace(self.alpha_min, self.alpha_max, self.n_alpha)
    
    @property
    def beta_grid(self) -> np.ndarray:
        """β网格点"""
        return np.linspace(self.beta_min, self.beta_max, self.n_beta)


@dataclass
class TransferSearchResultV2:
    """转移搜索结果 V2
    
    存储单次搜索尝试的完整结果。
    """
    # 搜索变量标识
    departure_orbit_name: str = ""
    arrival_orbit_name: str = ""
    departure_time_index: int = 0
    alpha: float = 0.0
    beta: float = 0.0
    
    # 出发点状态
    departure_state: Optional[np.ndarray] = None  # [x, y, z, vx, vy, vz]
    departure_time: float = 0.0  # CR3BP时间
    
    # 转移轨迹
    transfer_trajectory: Optional[np.ndarray] = None  # [n_steps, 6]
    transfer_times: Optional[np.ndarray] = None     # [n_steps]
    transfer_time: float = 0.0
    
    # 与目标轨道相交信息
    intersection_found: bool = False
    intersection_point: Optional[np.ndarray] = None
    intersection_idx: int = -1
    
    # 距离信息
    min_distance: float = np.inf
    min_distance_idx: int = -1
    
    # 局部最小检测
    local_minimum_found: bool = False
    local_minimum_distance: float = np.inf
    local_minimum_idx: int = -1
    
    # 碰撞信息
    collision_found: bool = False
    collision_body: Optional[str] = None  # 'earth' or 'moon'
    collision_idx: int = -1
    
    # 状态
    status: str = "pending"  # pending, success, no_intersection, collision, integration_failed
    
    @property
    def is_feasible(self) -> bool:
        """判断是否为可行候选解"""
        has_approach = (self.intersection_found or 
                       self.min_distance < 0.05 or
                       self.local_minimum_found)
        no_collision = not self.collision_found
        return has_approach and no_collision
    
    @property
    def dv_departure(self) -> float:
        """计算departure impulse (如果已知transfer trajectory)"""
        if self.departure_state is None or self.transfer_trajectory is None:
            return 0.0
        dv = self.transfer_trajectory[0, 3:] - self.departure_state[3:]
        return np.linalg.norm(dv)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'departure_orbit_name': self.departure_orbit_name,
            'arrival_orbit_name': self.arrival_orbit_name,
            'departure_time_index': self.departure_time_index,
            'alpha': self.alpha,
            'beta': self.beta,
            'departure_time': self.departure_time,
            'transfer_time': self.transfer_time,
            'intersection_found': self.intersection_found,
            'min_distance': self.min_distance,
            'local_minimum_found': self.local_minimum_found,
            'collision_found': self.collision_found,
            'status': self.status,
            'is_feasible': self.is_feasible,
        }


class DROROTransferSearchV2:
    """DRO到RO转移轨道搜索算法 V2
    
    实现论文Section III.A的搜索阶段算法:
    1. 从出发点轨道等时间间隔采样
    2. 对每个出发点，网格化搜索α, β
    3. 前向积分获取转移轨迹
    4. 筛选与目标轨道相交或距离局部最小的候选解
    
    修复的bug:
    - BUG-001: departure_orbit=arrival_orbit 引用错误
    - BUG-002: α,β速度扰动应用错误的velocity component
    - BUG-003: O(n²)嵌套循环计算距离，效率低
    - BUG-004: 缺少Earth/Moon碰撞检测
    
    属性:
        system: CR3BP系统对象
        dynamics: CR3BP动力学对象
        config: 搜索配置
    """
    
    def __init__(
        self,
        system: CR3BP_System,
        dynamics: CR3BP_Dynamics,
        config: Optional[TransferSearchConfig] = None,
    ):
        """初始化搜索算法
        
        参数:
            system: CR3BP系统对象
            dynamics: CR3BP动力学对象
            config: 搜索配置 (默认使用论文Table 3)
        """
        self.system = system
        self.dynamics = dynamics
        self.mu = system.mu
        self.config = config or TransferSearchConfig()
    
    def sample_departure_points(
        self, 
        departure_orbit: Orbit
    ) -> Tuple[np.ndarray, np.ndarray]:
        """从轨道等时间间隔采样出发点
        
        参数:
            departure_orbit: 出发点轨道 (DRO)
            
        返回:
            departure_states: 出发点状态序列 [n_departure, 6]
            departure_times: 出发点时间序列 [n_departure]
        """
        n = self.config.n_departure
        times = np.linspace(0, departure_orbit.period, n, endpoint=False)
        
        # 使用Orbit的插值方法获取各时间点的状态
        states = np.array([
            departure_orbit.interpolate_at_time(t) for t in times
        ])
        
        return states, times
    
    def compute_departure_velocity(
        self, 
        orbit_state: np.ndarray, 
        alpha: float, 
        beta: float = 0.0
    ) -> np.ndarray:
        """计算出发点速度扰动
        
        论文中的定义为:
        - α: 切向速度比 (tangential velocity ratio)
        - β: 法向速度比 (normal velocity ratio)
        
        速度扰动在轨道的切向和法向方向进行。
        
        参数:
            orbit_state: 轨道状态 [x, y, z, vx, vy, vz]
            alpha: 切向速度比
            beta: 法向速度比 (平面转移时为0)
            
        返回:
            扰动后的速度向量 [vx, vy, vz]
        """
        pos = orbit_state[:3]
        vel = orbit_state[3:]
        
        # 计算轨道面内的位置平面投影
        r_xy = np.sqrt(pos[0]**2 + pos[1]**2)
        if r_xy < 1e-10:
            # 靠近原点，使用原始速度
            warnings.warn("位置靠近原点，使用原始速度")
            return vel.copy()
        
        # 计算切向方向 (垂直于位置矢量的轨道面内方向)
        # 对于xy平面内的轨道，切向方向为 [-y, x, 0] / r
        tangential = np.array([-pos[1], pos[0], 0.0]) / r_xy
        
        # 计算径向方向 (位置矢量方向)
        radial = pos / np.linalg.norm(pos)
        
        # 计算法向方向 (轨道面法向，out of plane)
        # 使用叉乘: normal = radial × tangential
        normal = np.cross(radial, tangential)
        
        # 分解速度到切向和法向分量
        v_radial_comp = np.dot(vel, radial)
        v_tangential_comp = np.dot(vel, tangential)
        v_normal_comp = np.dot(vel, normal)
        
        # 构造新的速度向量
        # α控制切向速度的缩放
        # β控制法向速度的缩放
        new_vel = (
            v_radial_comp * radial +                    # 保持径向分量
            alpha * v_tangential_comp * tangential +    # 切向缩放α倍
            beta * v_normal_comp * normal                # 法向缩放β倍
        )
        
        return new_vel
    
    def forward_integrate(
        self,
        initial_state: np.ndarray,
        transfer_time: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """前向积分转移轨迹
        
        参数:
            initial_state: 初始状态 [x, y, z, vx, vy, vz]
            transfer_time: 转移时间 (CR3BP无量纲时间)
            
        返回:
            states: 轨迹状态序列 [n_steps, 6]
            times: 轨迹时间序列 [n_steps]
        """
        # 创建时间评估点
        n_steps = max(int(transfer_time / self.config.integration_dt) + 1, 100)
        t_eval = np.linspace(0, transfer_time, n_steps)
        
        # 使用dynamics.propagate进行积分
        result = self.dynamics.propagate(
            initial_state=initial_state,
            t_span=[0, transfer_time],
            t_eval=t_eval,
            with_stm=False,
        )
        
        states = result['states']
        times = result['time']
        
        return states, times
    
    def compute_min_distance(
        self,
        trajectory_states: np.ndarray,
        arrival_orbit: Orbit
    ) -> Tuple[float, int]:
        """计算轨迹到目标轨道的最小距离 (向量化实现)
        
        使用广播计算，效率远高于嵌套循环。
        
        参数:
            trajectory_states: 转移轨迹状态 [n_steps, 6]
            arrival_orbit: 目标轨道
            
        返回:
            min_distance: 最小距离
            min_idx: 最小距离对应的轨迹索引
        """
        # 提取轨迹位置: shape (n_steps, 3)
        traj_positions = trajectory_states[:, :3]
        
        # 提取目标轨道位置: shape (n_orbit, 3)
        orbit_positions = arrival_orbit.states[:, :3]
        
        # 使用广播计算所有距离对: shape (n_steps, n_orbit)
        # diff[i,j,k] = traj_pos[i,k] - orbit_pos[j,k]
        diff = traj_positions[:, np.newaxis, :] - orbit_positions[np.newaxis, :, :]
        
        # 计算欧氏距离: shape (n_steps, n_orbit)
        distances = np.sqrt(np.sum(diff**2, axis=2))
        
        # 找到全局最小值
        flat_distances = distances.flatten()
        min_flat_idx = np.argmin(flat_distances)
        min_distance = flat_distances[min_flat_idx]
        
        # 转换为(step_idx, orbit_idx)
        n_orbit = len(orbit_positions)
        step_idx = min_flat_idx // n_orbit
        orbit_idx = min_flat_idx % n_orbit
        
        return min_distance, step_idx
    
    def detect_intersection(
        self,
        trajectory_states: np.ndarray,
        arrival_orbit: Orbit,
    ) -> Tuple[bool, Optional[np.ndarray], int]:
        """检测轨迹是否与目标轨道相交
        
        参数:
            trajectory_states: 转移轨迹状态 [n_steps, 6]
            arrival_orbit: 目标轨道
            
        返回:
            intersection_found: 是否找到相交
            intersection_point: 相交点状态 (如果找到)
            intersection_idx: 相交点在轨迹中的索引
        """
        min_dist, step_idx = self.compute_min_distance(
            trajectory_states, arrival_orbit
        )
        
        if min_dist < self.config.intersection_threshold:
            return True, trajectory_states[step_idx], step_idx
        
        return False, None, -1
    
    def detect_local_minimum(
        self,
        trajectory_states: np.ndarray,
        arrival_orbit: Orbit,
    ) -> Tuple[bool, float, int]:
        """检测轨迹到目标轨道的距离是否出现局部最小
        
        使用有限差分检测: d'dt = 0, d²/dt² > 0
        
        参数:
            trajectory_states: 转移轨迹状态 [n_steps, 6]
            arrival_orbit: 目标轨道
            
        返回:
            local_min_found: 是否找到局部最小
            local_min_distance: 局部最小距离
            local_min_idx: 局部最小点对应的轨迹索引
        """
        # 计算轨迹上每个点到目标轨道的距离
        traj_positions = trajectory_states[:, :3]
        orbit_positions = arrival_orbit.states[:, :3]
        
        # 向量化距离计算
        diff = traj_positions[:, np.newaxis, :] - orbit_positions[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        min_distances = np.min(distances, axis=1)  # shape: (n_steps,)
        
        # 检测局部最小: 前一点和后一点都比当前点距离大
        local_mins = []
        for i in range(1, len(min_distances) - 1):
            # 局部最小条件:
            # dist[i+1] > dist[i] AND dist[i-1] > dist[i]
            if (min_distances[i+1] > min_distances[i] and 
                min_distances[i-1] > min_distances[i]):
                local_mins.append((i, min_distances[i]))
        
        if local_mins:
            # 返回距离最小的局部最小
            best = min(local_mins, key=lambda x: x[1])
            return True, best[1], best[0]
        
        return False, np.inf, -1
    
    def check_collision(
        self,
        trajectory_states: np.ndarray,
    ) -> Tuple[bool, Optional[str], int]:
        """检测轨迹是否与地球或月球碰撞
        
        参数:
            trajectory_states: 转移轨迹状态 [n_steps, 6]
            
        返回:
            collision_found: 是否发生碰撞
            collision_body: 碰撞天体 ('earth' 或 'moon')
            collision_idx: 碰撞点索引
        """
        positions = trajectory_states[:, :3]
        
        # 地球中心位置 (CR3BP无量纲)
        earth_center = np.array([-self.mu, 0.0, 0.0])
        
        # 月球中心位置 (CR3BP无量纲)
        moon_center = np.array([1.0 - self.mu, 0.0, 0.0])
        
        # 计算到各天体中心的距离
        dist_earth = np.linalg.norm(positions - earth_center, axis=1)
        dist_moon = np.linalg.norm(positions - moon_center, axis=1)
        
        # 检测碰撞 (距离小于阈值)
        earth_collision_idx = np.where(dist_earth < self.config.collision_earth_radius)[0]
        moon_collision_idx = np.where(dist_moon < self.config.collision_moon_radius)[0]
        
        if len(earth_collision_idx) > 0:
            return True, 'earth', int(earth_collision_idx[0])
        if len(moon_collision_idx) > 0:
            return True, 'moon', int(moon_collision_idx[0])
        
        return False, None, -1
    
    def search_single_departure(
        self,
        departure_state: np.ndarray,
        departure_time: float,
        arrival_orbit: Orbit,
    ) -> List[TransferSearchResultV2]:
        """对单个出发点搜索α,β网格
        
        参数:
            departure_state: 出发点状态 [x, y, z, vx, vy, vz]
            departure_time: 出发点对应的时间
            arrival_orbit: 目标轨道 (RO)
            
        返回:
            results: 搜索结果列表
        """
        results = []
        
        # 构建初始状态 (位置用出发点，速度用扰动后速度)
        # 注意: 这里v_dep_new已经是扰动后的速度
        # 转移轨道的初始速度 = departure velocity + Δv
        
        # 遍历α,β网格
        for alpha in self.config.alpha_grid:
            for beta in self.config.beta_grid:
                # 计算扰动后的速度
                new_vel = self.compute_departure_velocity(
                    departure_state, alpha, beta
                )
                
                # 构建转移轨道初始状态
                initial_state = np.concatenate([
                    departure_state[:3],  # 位置
                    new_vel              # 速度
                ])
                
                # 前向积分
                try:
                    traj_states, traj_times = self.forward_integrate(
                        initial_state,
                        self.config.max_transfer_time,
                    )
                except Exception as e:
                    # 积分失败，跳过
                    result = TransferSearchResultV2(
                        status='integration_failed',
                        departure_state=departure_state,
                        departure_time=departure_time,
                        alpha=alpha,
                        beta=beta,
                    )
                    results.append(result)
                    continue
                
                # 检查碰撞
                collision, body, col_idx = self.check_collision(traj_states)
                
                # 计算到目标轨道的最小距离
                min_dist, min_idx = self.compute_min_distance(traj_states, arrival_orbit)
                
                # 检测相交
                intersection, int_point, int_idx = self.detect_intersection(
                    traj_states, arrival_orbit
                )
                
                # 检测局部最小
                local_min, local_min_dist, local_min_idx = self.detect_local_minimum(
                    traj_states, arrival_orbit
                )
                
                # 构建结果
                result = TransferSearchResultV2(
                    departure_state=departure_state,
                    departure_time=departure_time,
                    alpha=alpha,
                    beta=beta,
                    transfer_trajectory=traj_states,
                    transfer_times=traj_times,
                    transfer_time=traj_times[-1],
                    intersection_found=intersection,
                    intersection_point=int_point,
                    intersection_idx=int_idx,
                    min_distance=min_dist,
                    min_distance_idx=min_idx,
                    local_minimum_found=local_min,
                    local_minimum_distance=local_min_dist,
                    local_minimum_idx=local_min_idx,
                    collision_found=collision,
                    collision_body=body,
                    collision_idx=col_idx,
                )
                
                # 设置状态
                if collision:
                    result.status = 'collision'
                elif intersection:
                    result.status = 'success'
                elif min_dist < self.config.min_distance_threshold:
                    result.status = 'success'  # 足够接近也算成功
                else:
                    result.status = 'no_intersection'
                
                results.append(result)
        
        return results
    
    def grid_search(
        self,
        departure_orbit: Orbit,
        arrival_orbit: Orbit,
        verbose: bool = True,
    ) -> List[TransferSearchResultV2]:
        """网格搜索主函数
        
        在出发点轨道上采样，在α,β网格上搜索，
        寻找与目标轨道相交或距离局部最小的转移轨道。
        
        参数:
            departure_orbit: 出发点轨道 (DRO)
            arrival_orbit: 目标轨道 (RO)
            verbose: 是否输出进度信息
            
        返回:
            results: 所有搜索结果列表
        """
        # 获取轨道名称 (如果可用)
        dep_name = getattr(departure_orbit, 'name', 'unknown')
        arr_name = getattr(arrival_orbit, 'name', 'unknown')
        
        # 采样出发点
        if verbose:
            print(f"采样出发点: n_departure={self.config.n_departure}")
        
        departure_states, departure_times = self.sample_departure_points(departure_orbit)
        
        all_results = []
        total_departures = len(departure_states)
        
        if verbose:
            print(f"开始网格搜索: {total_departures}出发点 × {self.config.n_alpha}α × {self.config.n_beta}β")
        
        # 遍历每个出发点
        for i, (dep_state, dep_time) in enumerate(zip(departure_states, departure_times)):
            if verbose and (i + 1) % 20 == 0:
                print(f"  进度: {i+1}/{total_departures}出发点 ({(i+1)/total_departures*100:.1f}%)")
            
            # 对单个出发点搜索α,β网格
            results = self.search_single_departure(
                dep_state, dep_time, arrival_orbit
            )
            
            # 设置轨道名称
            for r in results:
                r.departure_orbit_name = dep_name
                r.arrival_orbit_name = arr_name
                r.departure_time_index = i
            
            all_results.extend(results)
        
        if verbose:
            # 统计结果
            feasible = [r for r in all_results if r.is_feasible]
            print(f"搜索完成: {len(all_results)}个候选解, {len(feasible)}个可行")
        
        return all_results
    
    def filter_local_minima(
        self,
        results: List[TransferSearchResultV2],
    ) -> List[TransferSearchResultV2]:
        """从结果中筛选满足局部最小条件的候选解
        
        按α值分组，在每组内检测局部最小。
        
        参数:
            results: 所有搜索结果
            
        返回:
            filtered: 筛选后的候选解列表
        """
        # 按alpha值分组
        alpha_groups: Dict[float, List[TransferSearchResultV2]] = {}
        for r in results:
            alpha = round(r.alpha, 2)  # 离散化
            if alpha not in alpha_groups:
                alpha_groups[alpha] = []
            alpha_groups[alpha].append(r)
        
        filtered = []
        
        for alpha, group in alpha_groups.items():
            # 按转移时间排序
            group.sort(key=lambda x: x.transfer_time)
            
            # 检测相邻点的距离梯度
            for i in range(1, len(group) - 1):
                prev_dist = group[i-1].min_distance
                curr_dist = group[i].min_distance
                next_dist = group[i+1].min_distance
                
                # 局部最小: 前一个和后一个都比当前大
                if curr_dist < prev_dist and curr_dist < next_dist:
                    if curr_dist < self.config.min_distance_threshold:
                        filtered.append(group[i])
        
        return filtered


def load_orbit_from_json(filepath: str) -> Orbit:
    """从JSON文件加载轨道数据
    
    参数:
        filepath: JSON文件路径
        
    返回:
        Orbit对象
    """
    import json
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # 提取状态和时间
    states = np.array(data['states'])
    times = np.array(data['times'])
    
    # 创建Orbit对象
    orbit = Orbit(states=states, times=times)
    
    # 设置元数据
    if 'orbit_type' in data:
        orbit.metadata['orbit_type'] = data['orbit_type']
    if 'period_ratio' in data:
        orbit.metadata['period_ratio'] = data['period_ratio']
    
    return orbit


def save_search_results(
    results: List[TransferSearchResultV2],
    filepath: str,
) -> None:
    """保存搜索结果到JSON文件
    
    参数:
        results: 搜索结果列表
        filepath: 输出文件路径
    """
    import json
    
    # 转换为可序列化的字典
    output = []
    for r in results:
        result_dict = {
            'departure_orbit_name': r.departure_orbit_name,
            'arrival_orbit_name': r.arrival_orbit_name,
            'departure_time_index': r.departure_time_index,
            'departure_time': float(r.departure_time),
            'alpha': float(r.alpha),
            'beta': float(r.beta),
            'transfer_time': float(r.transfer_time),
            'intersection_found': r.intersection_found,
            'min_distance': float(r.min_distance),
            'local_minimum_found': r.local_minimum_found,
            'collision_found': r.collision_found,
            'status': r.status,
            'is_feasible': r.is_feasible,
        }
        
        # 如果有转移轨迹数据，不保存完整轨迹(太大)
        # 只保存元数据
        output.append(result_dict)
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
