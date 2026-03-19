"""
地球到月球转移轨道设计模块

提供从地球低轨道（LEO）到月球轨道或平动点轨道的转移轨道设计。
支持多种转移策略：
- 直接转移（Hohmann-like）
- 经平动点L1的低能转移
- 利用不变流形的低能转移
- 弱稳定边界（WSB）转移
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


class EarthMoonTransfer:
    """地球到月球转移轨道设计

    设计从地球出发到月球附近目标轨道的转移轨道。

    核心方法：
    - design_direct_transfer: 设计直接转移轨道
    - design_low_energy_transfer: 设计低能转移轨道
    - design_manifold_transfer: 利用不变流形设计转移
    - compute_delta_v: 计算转移所需的速度增量
    - optimize_transfer: 优化转移参数
    """

    def __init__(self, system, dynamics):
        """初始化

        参数：
        - system: CR3BP_System对象（地月系统）
        - dynamic: CR3BP_Dynamics对象
        """
        self.system = system
        self.dynamics = dynamics
        self.mu = system.mu

        # 地球和月球参数（无量纲）
        self.earth_pos = np.array([-self.mu, 0, 0])
        self.moon_pos = np.array([1 - self.mu, 0, 0])

        # 转移结果缓存
        self.transfer_trajectory = None
        self.delta_v_total = None
        self.transfer_time = None
        self.departure_state = None
        self.arrival_state = None

    def design_direct_transfer(
        self, r_departure, r_arrival, v_departure_guess=None, t_transfer_guess=None, n_revolutions=0
    ):
        """设计直接转移轨道

        从地球附近的圆轨道出发，直接转移到月球附近。

        参数：
            r_departure: 出发轨道半径（无量纲，相对地球）
            r_arrival: 到达轨道半径（无量纲，相对月球）
            v_departure_guess: 出发速度猜测值
            t_transfer_guess: 转移时间猜测值
            n_revolutions: 转移过程中的圈数

        返回：
            dict: 转移轨道设计结果
        """
        # 出发点（地球表面附近圆轨道）
        # 在旋转系中，地球位于(-mu, 0, 0)
        departure_pos = self.earth_pos + np.array([r_departure, 0, 0])

        # 出发速度（圆轨道速度 + 转移增量）
        if v_departure_guess is None:
            # 粗略估计Hohmann转移速度增量
            a_transfer = (r_departure + (1.0)) / 2  # 近似转移半长轴
            v_departure_guess = np.sqrt(
                2 * (1 - self.mu) * (1 / r_departure - 1 / (2 * a_transfer))
            )

        # 初始状态（旋转系中）
        # 从地球附近圆轨道出发，垂直于连线方向
        initial_state = np.array(
            [
                departure_pos[0],
                departure_pos[1],
                0,
                0,
                v_departure_guess + departure_pos[0],  # 包含旋转系修正
                0,
            ]
        )

        # 转移时间
        if t_transfer_guess is None:
            # 估计转移时间（Hohmann转移时间的一半到一倍）
            t_transfer_guess = np.pi  # 约半个周期

        # 使用打靶法优化转移参数
        result = self._shooting_method(initial_state, t_transfer_guess, r_arrival)

        return result

    def design_low_energy_transfer(
        self, target_orbit, libration_point="L1", departure_altitude=200, verbose=True
    ):
        """设计低能转移轨道

        利用平动点附近的动力学结构实现低能转移。

        参数：
            target_orbit: 目标周期轨道（Orbit对象）
            libration_point: 经过的平动点（"L1"或"L2"）
            departure_altitude: 出发轨道高度（km，转为无量纲）
            verbose: 是否打印信息

        返回：
            dict: 转移设计结果
        """
        if verbose:
            print(f"\n设计低能转移轨道 (经{libration_point})")

        # 获取目标轨道参数
        target_state = target_orbit.states[0]
        target_jacobi = self.system.get_jacobi_constant(target_state)

        if verbose:
            print(f"  目标轨道Jacobi常数: {target_jacobi:.6f}")

        # 出发轨道（地球低轨道）
        if self.system.characteristic_length is not None:
            r_earth = 6378.0 / self.system.characteristic_length  # 地球半径（无量纲）
            r_departure = r_earth + departure_altitude / self.system.characteristic_length
        else:
            r_departure = 0.017  # 默认LEO半径（无量纲）

        # 在目标轨道周围搜索切入点
        # 沿目标轨道寻找最优切入时刻
        best_dv = np.inf
        best_transfer = None

        # 离散搜索目标轨道上的切入点
        n_search = 20
        target_times = np.linspace(0, target_orbit.period or 6.0, n_search, endpoint=False)

        for t_arrival in target_times:
            arrival_state = target_orbit.interpolate_at_time(t_arrival)

            # 反向积分从目标轨道到达地球附近
            try:
                result = self._backward_propagate(arrival_state, r_departure, max_time=20.0)

                if result is not None:
                    dv = result["delta_v"]
                    if dv < best_dv:
                        best_dv = dv
                        best_transfer = result
                        best_transfer["arrival_time_on_orbit"] = t_arrival

            except Exception:
                continue

        if best_transfer is not None:
            self.transfer_trajectory = best_transfer.get("trajectory")
            self.delta_v_total = best_dv
            self.transfer_time = best_transfer.get("transfer_time")

            if verbose:
                print(f"  最优ΔV: {best_dv:.6f}")
                print(f"  转移时间: {self.transfer_time:.4f}")

        return best_transfer

    def design_manifold_transfer(
        self, target_orbit, manifold_type="stable", n_trajectories=100, verbose=True
    ):
        """利用不变流形设计转移轨道

        计算目标轨道的稳定/不稳定流形，找到与地球低轨道相交的流形臂。

        参数：
            target_orbit: 目标周期轨道
            manifold_type: 流形类型（"stable"或"unstable"）
            n_trajectories: 流形轨迹数量
            verbose: 是否打印信息

        返回：
            dict: 流形转移设计结果
        """
        if verbose:
            print(f"\n计算{manifold_type}不变流形转移")

        # 计算目标轨道的单值矩阵和特征向量
        from ..algorithms.stability import StabilityAnalysis

        stability = StabilityAnalysis(target_orbit, self.dynamics)
        stability.compute_floquet_multipliers()

        if stability.eigenvectors is None:
            if verbose:
                print("  无法计算特征向量")
            return None

        # 选择稳定/不稳定特征方向
        eigenvalues = stability.eigenvalues
        eigenvectors = stability.eigenvectors
        magnitudes = np.abs(eigenvalues)

        if manifold_type == "stable":
            # 选择模长最小的特征值对应方向
            idx = np.argmin(magnitudes)
        else:
            # 选择模长最大的特征值对应方向
            idx = np.argmax(magnitudes)

        manifold_direction = np.real(eigenvectors[:, idx])
        manifold_direction = manifold_direction / np.linalg.norm(manifold_direction)

        # 沿轨道离散点生成流形
        eps = 1e-6  # 流形偏移量
        manifold_trajectories = []
        departure_states = []

        n_points = min(n_trajectories, len(target_orbit.states))
        indices = np.linspace(0, len(target_orbit.states) - 1, n_points, dtype=int)

        for i in indices:
            state = target_orbit.states[i]

            # 偏移到流形方向
            perturbed_state = state + eps * manifold_direction

            # 积分（稳定流形反向，不稳定流形正向）
            if manifold_type == "stable":
                t_span = (0, -15)  # 反向积分
            else:
                t_span = (0, 15)  # 正向积分

            try:
                result = solve_ivp(
                    self.dynamics.equations_of_motion,
                    t_span,
                    perturbed_state,
                    method="DOP853",
                    t_eval=np.linspace(t_span[0], t_span[1], 2000),
                    rtol=1e-12,
                    atol=1e-12,
                )
                if result.success:
                    manifold_trajectories.append(result.y.T)

                    # 检查是否经过地球附近
                    positions = result.y[:3, :].T
                    distances_to_earth = np.linalg.norm(positions - self.earth_pos, axis=1)
                    min_dist = np.min(distances_to_earth)

                    if min_dist < 0.05:  # 接近地球
                        min_idx = np.argmin(distances_to_earth)
                        departure_states.append(
                            {
                                "state": result.y[:, min_idx],
                                "time": result.t[min_idx],
                                "distance_to_earth": min_dist,
                                "orbit_index": i,
                            }
                        )
            except Exception:
                continue

        if verbose:
            print(f"  计算了 {len(manifold_trajectories)} 条流形轨迹")
            print(f"  其中 {len(departure_states)} 条经过地球附近")

        # 找到最优转移
        best_transfer = None
        if departure_states:
            departure_states.sort(key=lambda x: x["distance_to_earth"])
            best_transfer = departure_states[0]

        return {
            "manifold_trajectories": manifold_trajectories,
            "departure_candidates": departure_states,
            "best_transfer": best_transfer,
            "manifold_type": manifold_type,
        }

    def compute_delta_v(self, departure_state, arrival_state):
        """计算两个状态之间的速度增量

        参数：
            departure_state: 出发状态 [x, y, z, vx, vy, vz]
            arrival_state: 到达状态 [x, y, z, vx, vy, vz]

        返回：
            float: ΔV大小
        """
        dv = arrival_state[3:] - departure_state[3:]
        return np.linalg.norm(dv)

    def _shooting_method(self, initial_state, t_transfer, r_target):
        """打靶法求解转移轨道

        参数：
            initial_state: 初始状态
            t_transfer: 转移时间
            r_target: 目标距月球的距离

        返回：
            dict: 打靶法结果
        """

        def objective(params):
            vy0, tf = params
            state = initial_state.copy()
            state[4] = vy0

            try:
                result = solve_ivp(
                    self.dynamics.equations_of_motion,
                    (0, tf),
                    state,
                    method="DOP853",
                    rtol=1e-12,
                    atol=1e-12,
                )
                if result.success:
                    final_pos = result.y[:3, -1]
                    dist_to_moon = np.linalg.norm(final_pos - self.moon_pos)
                    return (dist_to_moon - r_target) ** 2
                else:
                    return 1e10
            except Exception:
                return 1e10

        # 优化
        result = minimize(
            objective,
            x0=[initial_state[4], t_transfer],
            method="Nelder-Mead",
            options={"maxiter": 1000, "xatol": 1e-10},
        )

        if result.success:
            vy0_opt, tf_opt = result.x
            state = initial_state.copy()
            state[4] = vy0_opt

            # 积分最优轨迹
            prop = solve_ivp(
                self.dynamics.equations_of_motion,
                (0, tf_opt),
                state,
                method="DOP853",
                t_eval=np.linspace(0, tf_opt, 2000),
                rtol=1e-12,
                atol=1e-12,
            )

            return {
                "trajectory": prop.y.T,
                "times": prop.t,
                "departure_state": state,
                "arrival_state": prop.y[:, -1],
                "transfer_time": tf_opt,
                "delta_v": abs(vy0_opt - initial_state[4]),
                "success": True,
            }

        return {"success": False, "message": result.message}

    def _backward_propagate(self, arrival_state, r_departure, max_time=20.0):
        """从到达状态反向传播到地球附近

        参数：
            arrival_state: 到达状态
            r_departure: 出发轨道半径
            max_time: 最大传播时间

        返回：
            dict: 反向传播结果
        """
        result = solve_ivp(
            self.dynamics.equations_of_motion,
            (0, -max_time),
            arrival_state,
            method="DOP853",
            t_eval=np.linspace(0, -max_time, 5000),
            rtol=1e-12,
            atol=1e-12,
        )

        if not result.success:
            return None

        # 检查是否经过地球附近
        positions = result.y[:3, :].T
        distances = np.linalg.norm(positions - self.earth_pos, axis=1)
        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]

        if min_dist < r_departure * 5:  # 接近出发轨道
            departure_state = result.y[:, min_dist_idx]

            # 计算从圆轨道到转移轨道的ΔV
            r = np.linalg.norm(departure_state[:3] - self.earth_pos)
            v_circular = np.sqrt((1 - self.mu) / r)
            v_actual = np.linalg.norm(departure_state[3:])
            dv = abs(v_actual - v_circular)

            return {
                "trajectory": result.y.T,
                "times": result.t,
                "departure_state": departure_state,
                "arrival_state": arrival_state,
                "transfer_time": abs(result.t[min_dist_idx]),
                "delta_v": dv,
                "min_earth_distance": min_dist,
            }

        return None

    def __str__(self):
        return f"EarthMoonTransfer(mu={self.mu})"
