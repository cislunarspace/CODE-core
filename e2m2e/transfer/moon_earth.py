"""
月球到地球转移轨道设计模块

提供从月球轨道或月球附近平动点轨道返回地球的转移轨道设计。
支持策略：
- 直接返回转移
- 经平动点的低能返回转移
- 利用不变流形的返回转移
"""

import numpy as np
from scipy.integrate import solve_ivp


class MoonEarthTransfer:
    """月球到地球转移轨道设计

    设计从月球附近目标轨道返回地球的转移轨道。

    核心方法：
    - design_direct_return: 设计直接返回轨道
    - design_low_energy_return: 设计低能返回轨道
    - design_manifold_return: 利用不变流形返回
    - compute_reentry_conditions: 计算再入条件
    """

    def __init__(self, system, dynamics):
        """初始化

        参数：
        - system: CR3BP_System对象（地月系统）
        - dynamics: CR3BP_Dynamics对象
        """
        self.system = system
        self.dynamics = dynamics
        self.mu = system.mu

        # 天体位置
        self.earth_pos = np.array([-self.mu, 0, 0])
        self.moon_pos = np.array([1 - self.mu, 0, 0])

        # 转移结果缓存
        self.transfer_trajectory = None
        self.delta_v_total = None
        self.transfer_time = None

    def design_direct_return(
        self, departure_orbit, r_reentry=None, n_search_points=20, verbose=True
    ):
        """设计直接返回地球轨道

        从月球附近的目标轨道出发，直接返回地球。

        参数：
            departure_orbit: 出发轨道（Orbit对象）
            r_reentry: 再入轨道半径（无量纲，相对地球），默认LEO
            n_search_points: 出发点搜索数量
            verbose: 是否打印信息

        返回：
            dict: 返回轨道设计结果
        """
        if verbose:
            print("\n设计直接返回地球轨道")

        # 再入轨道参数
        if r_reentry is None:
            if self.system.characteristic_length:
                r_reentry = (6378.0 + 200.0) / self.system.characteristic_length
            else:
                r_reentry = 0.017  # 默认LEO

        best_dv = np.inf
        best_result = None

        # 在出发轨道上搜索最优出发点
        period = departure_orbit.period or 6.0
        departure_times = np.linspace(0, period, n_search_points, endpoint=False)

        for t_dep in departure_times:
            dep_state = departure_orbit.interpolate_at_time(t_dep)

            # 尝试不同的转移时间
            for t_transfer in np.linspace(2.0, 15.0, 10):
                try:
                    result = self._propagate_to_earth(dep_state, t_transfer, r_reentry)

                    if result is not None and result["delta_v"] < best_dv:
                        best_dv = result["delta_v"]
                        best_result = result
                        best_result["departure_time_on_orbit"] = t_dep

                except Exception:
                    continue

        if best_result is not None:
            self.transfer_trajectory = best_result.get("trajectory")
            self.delta_v_total = best_dv
            self.transfer_time = best_result.get("transfer_time")

            if verbose:
                print(f"  最优ΔV: {best_dv:.6f}")
                print(f"  转移时间: {self.transfer_time:.4f}")

        return best_result

    def design_low_energy_return(self, departure_orbit, target_altitude=200, verbose=True):
        """设计低能返回地球轨道

        利用三体动力学结构实现低能量的月球-地球转移。

        参数：
            departure_orbit: 出发轨道
            target_altitude: 目标LEO高度（km）
            verbose: 是否打印信息

        返回：
            dict: 低能返回设计结果
        """
        if verbose:
            print("\n设计低能返回轨道")

        # 利用不变流形
        from ..algorithms.stability import StabilityAnalysis

        stability = StabilityAnalysis(departure_orbit, self.dynamics)

        try:
            stability.compute_floquet_multipliers()
        except Exception as e:
            if verbose:
                print(f"  稳定性分析失败: {e}")
            return None

        if stability.eigenvectors is None:
            return None

        # 选择不稳定方向
        eigenvalues = stability.eigenvalues
        magnitudes = np.abs(eigenvalues)
        unstable_idx = np.argmax(magnitudes)
        unstable_direction = np.real(stability.eigenvectors[:, unstable_idx])
        unstable_direction = unstable_direction / np.linalg.norm(unstable_direction)

        # 在出发轨道上搜索最优出发点
        eps = 1e-6
        best_result = None
        best_dv = np.inf

        n_points = min(50, len(departure_orbit.states))
        indices = np.linspace(0, len(departure_orbit.states) - 1, n_points, dtype=int)

        for i in indices:
            state = departure_orbit.states[i]

            # 两个方向的扰动
            for sign in [1, -1]:
                perturbed = state + sign * eps * unstable_direction

                try:
                    result = solve_ivp(
                        self.dynamics.equations_of_motion,
                        (0, 20),
                        perturbed,
                        method="DOP853",
                        t_eval=np.linspace(0, 20, 5000),
                        rtol=1e-12,
                        atol=1e-12,
                    )

                    if result.success:
                        positions = result.y[:3, :].T
                        distances = np.linalg.norm(positions - self.earth_pos, axis=1)
                        min_idx = np.argmin(distances)
                        min_dist = distances[min_idx]

                        if min_dist < 0.05:  # 接近地球
                            earth_state = result.y[:, min_idx]
                            r = np.linalg.norm(earth_state[:3] - self.earth_pos)
                            v_circ = np.sqrt((1 - self.mu) / r)
                            v_actual = np.linalg.norm(earth_state[3:])
                            dv = abs(v_actual - v_circ)

                            if dv < best_dv:
                                best_dv = dv
                                best_result = {
                                    "trajectory": result.y.T,
                                    "times": result.t,
                                    "departure_state": perturbed,
                                    "arrival_state": earth_state,
                                    "transfer_time": result.t[min_idx],
                                    "delta_v": dv,
                                    "min_earth_distance": min_dist,
                                    "orbit_index": int(i),
                                }
                except Exception:
                    continue

        if best_result is not None and verbose:
            print("  找到低能返回轨迹")
            print(f"  ΔV: {best_dv:.6f}")
            print(f"  转移时间: {best_result['transfer_time']:.4f}")

        return best_result

    def design_manifold_return(self, departure_orbit, n_trajectories=100, verbose=True):
        """利用不变流形设计返回轨道

        计算出发轨道的不稳定流形，找到接近地球的流形臂。

        参数：
            departure_orbit: 出发轨道
            n_trajectories: 流形轨迹数量
            verbose: 是否打印信息

        返回：
            dict: 流形返回设计结果
        """
        # 复用EarthMoonTransfer中的流形计算，但使用不稳定流形
        from .earth_moon import EarthMoonTransfer

        e2m = EarthMoonTransfer(self.system, self.dynamics)
        result = e2m.design_manifold_transfer(
            departure_orbit,
            manifold_type="unstable",
            n_trajectories=n_trajectories,
            verbose=verbose,
        )

        return result

    def compute_reentry_conditions(self, arrival_state, reentry_angle=-6.0):
        """计算再入条件

        参数：
            arrival_state: 到达地球附近的状态向量
            reentry_angle: 再入角度（度）

        返回：
            dict: 再入条件
        """
        # 相对地球的位置和速度
        r_rel = arrival_state[:3] - self.earth_pos
        v_rel = arrival_state[3:]

        # 距地球距离
        r_mag = np.linalg.norm(r_rel)

        # 速度大小
        v_mag = np.linalg.norm(v_rel)

        # 飞行路径角
        flight_path_angle = np.arcsin(np.dot(r_rel, v_rel) / (r_mag * v_mag))

        # 再入速度估计（转换为物理单位）
        if self.system.characteristic_velocity:
            v_reentry_km_s = v_mag * self.system.characteristic_velocity
        else:
            v_reentry_km_s = None

        return {
            "relative_position": r_rel,
            "relative_velocity": v_rel,
            "distance_to_earth": r_mag,
            "velocity_magnitude": v_mag,
            "velocity_km_s": v_reentry_km_s,
            "flight_path_angle_deg": np.degrees(flight_path_angle),
            "target_reentry_angle_deg": reentry_angle,
        }

    def _propagate_to_earth(self, departure_state, t_transfer, r_target):
        """传播轨道到地球附近

        参数：
            departure_state: 出发状态
            t_transfer: 转移时间
            r_target: 目标距地球距离

        返回：
            dict: 传播结果
        """
        result = solve_ivp(
            self.dynamics.equations_of_motion,
            (0, t_transfer),
            departure_state,
            method="DOP853",
            t_eval=np.linspace(0, t_transfer, 2000),
            rtol=1e-12,
            atol=1e-12,
        )

        if not result.success:
            return None

        # 检查是否到达地球附近
        positions = result.y[:3, :].T
        distances = np.linalg.norm(positions - self.earth_pos, axis=1)
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]

        if min_dist < r_target * 3:
            arrival_state = result.y[:, min_idx]
            r = np.linalg.norm(arrival_state[:3] - self.earth_pos)
            v_circ = np.sqrt((1 - self.mu) / r) if r > 0 else 0
            v_actual = np.linalg.norm(arrival_state[3:])
            dv = abs(v_actual - v_circ)

            return {
                "trajectory": result.y.T,
                "times": result.t,
                "departure_state": departure_state,
                "arrival_state": arrival_state,
                "transfer_time": result.t[min_idx],
                "delta_v": dv,
                "min_earth_distance": min_dist,
            }

        return None

    def __str__(self):
        return f"MoonEarthTransfer(mu={self.mu})"
