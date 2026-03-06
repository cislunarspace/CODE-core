"""
轨道间转移设计模块

提供地月空间中不同三体轨道之间的转移轨道设计。
支持策略：
- 同族轨道之间的转移（如不同Halo轨道之间）
- 不同族轨道之间的转移（如Lyapunov到Halo）
- 不同平动点轨道之间的转移（如L1 Halo到L2 Halo）
- 异宿/同宿连接
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


class InterOrbitTransfer:
    """轨道间转移设计

    设计地月空间中不同周期轨道之间的转移轨道。

    核心方法：
    - design_homoclinic_transfer: 同宿转移设计
    - design_heteroclinic_transfer: 异宿转移设计
    - design_direct_transfer: 直接轨道间转移
    - design_manifold_intersection: 流形交叉转移
    - compute_transfer_cost: 计算转移代价
    """

    def __init__(self, system, dynamics):
        """初始化

        参数：
        - system: CR3BP_System对象
        - dynamics: CR3BP_Dynamics对象
        """
        self.system = system
        self.dynamics = dynamics
        self.mu = system.mu

        # 转移结果缓存
        self.transfer_trajectory = None
        self.delta_v_total = None
        self.transfer_time = None
        self.transfer_arcs = []

    def design_direct_transfer(self, orbit_departure, orbit_arrival, n_search=20, verbose=True):
        """设计直接轨道间转移

        通过搜索出发和到达轨道上的最优切入/切出点实现转移。

        参数：
            orbit_departure: 出发轨道（Orbit对象）
            orbit_arrival: 到达轨道（Orbit对象）
            n_search: 每条轨道上的搜索点数
            verbose: 是否打印信息

        返回：
            dict: 转移设计结果
        """
        if verbose:
            print("\n设计直接轨道间转移")

        best_dv = np.inf
        best_result = None

        # 出发和到达轨道上的时间点
        T_dep = orbit_departure.period or 6.0
        T_arr = orbit_arrival.period or 6.0

        dep_times = np.linspace(0, T_dep, n_search, endpoint=False)
        arr_times = np.linspace(0, T_arr, n_search, endpoint=False)

        for t_dep in dep_times:
            dep_state = orbit_departure.interpolate_at_time(t_dep)

            for t_arr in arr_times:
                arr_state = orbit_arrival.interpolate_at_time(t_arr)

                # 尝试Lambert求解（在CR3BP中用打靶法近似）
                for tof in np.linspace(1.0, 10.0, 5):
                    try:
                        result = self._solve_transfer_arc(dep_state, arr_state, tof)

                        if result is not None and result["total_dv"] < best_dv:
                            best_dv = result["total_dv"]
                            best_result = result
                            best_result["t_departure"] = t_dep
                            best_result["t_arrival"] = t_arr

                    except Exception:
                        continue

        if best_result is not None:
            self.transfer_trajectory = best_result.get("trajectory")
            self.delta_v_total = best_dv
            self.transfer_time = best_result.get("transfer_time")

            if verbose:
                print(f"  最优总ΔV: {best_dv:.6f}")
                print(f"  转移时间: {self.transfer_time:.4f}")

        return best_result

    def design_manifold_intersection(
        self,
        orbit_departure,
        orbit_arrival,
        n_manifold=50,
        poincare_plane="y",
        poincare_value=0.0,
        verbose=True,
    ):
        """利用不变流形交叉设计转移

        计算出发轨道的不稳定流形和到达轨道的稳定流形，
        在庞加莱截面上寻找交叉点。

        参数：
            orbit_departure: 出发轨道
            orbit_arrival: 到达轨道
            n_manifold: 流形轨迹数量
            poincare_plane: 庞加莱截面平面
            poincare_value: 截面位置
            verbose: 是否打印信息

        返回：
            dict: 流形交叉转移结果
        """
        if verbose:
            print("\n设计流形交叉转移")
            print(f"  庞加莱截面: {poincare_plane}={poincare_value}")

        # 计算出发轨道的不稳定流形截面交叉
        unstable_crossings = self._compute_manifold_crossings(
            orbit_departure,
            manifold_type="unstable",
            n_trajectories=n_manifold,
            plane=poincare_plane,
            value=poincare_value,
            direction="forward",
        )

        # 计算到达轨道的稳定流形截面交叉
        stable_crossings = self._compute_manifold_crossings(
            orbit_arrival,
            manifold_type="stable",
            n_trajectories=n_manifold,
            plane=poincare_plane,
            value=poincare_value,
            direction="backward",
        )

        if verbose:
            print(f"  不稳定流形穿越点: {len(unstable_crossings)}")
            print(f"  稳定流形穿越点: {len(stable_crossings)}")

        # 在截面上寻找最近的交叉对
        best_dv = np.inf
        best_pair = None

        for uc in unstable_crossings:
            for sc in stable_crossings:
                # 在截面上的状态差异
                dv = np.linalg.norm(uc["state"][3:] - sc["state"][3:])

                if dv < best_dv:
                    best_dv = dv
                    best_pair = {
                        "unstable_crossing": uc,
                        "stable_crossing": sc,
                        "delta_v_at_section": dv,
                    }

        if best_pair is not None and verbose:
            print(f"  最优截面ΔV: {best_dv:.6f}")

        return {
            "unstable_crossings": unstable_crossings,
            "stable_crossings": stable_crossings,
            "best_pair": best_pair,
            "delta_v": best_dv if best_pair else None,
        }

    def design_heteroclinic_transfer(self, orbit_L1, orbit_L2, verbose=True):
        """设计异宿转移（L1 ↔ L2）

        利用L1轨道的不稳定流形和L2轨道的稳定流形的交叉实现
        低能转移连接。

        参数：
            orbit_L1: L1平动点附近的周期轨道
            orbit_L2: L2平动点附近的周期轨道
            verbose: 是否打印信息

        返回：
            dict: 异宿转移结果
        """
        if verbose:
            print("\n设计L1-L2异宿转移")

        # 使用流形交叉方法
        result = self.design_manifold_intersection(
            orbit_departure=orbit_L1,
            orbit_arrival=orbit_L2,
            n_manifold=100,
            poincare_plane="y",
            poincare_value=0.0,
            verbose=verbose,
        )

        if result and result.get("best_pair"):
            if verbose:
                print(f"  异宿转移ΔV: {result['delta_v']:.6f}")

        return result

    def design_homoclinic_transfer(self, orbit, verbose=True):
        """设计同宿转移

        同一周期轨道的不稳定流形和稳定流形的交叉，
        用于实现轨道能量变化。

        参数：
            orbit: 周期轨道
            verbose: 是否打印信息

        返回：
            dict: 同宿转移结果
        """
        if verbose:
            print("\n设计同宿转移")

        result = self.design_manifold_intersection(
            orbit_departure=orbit, orbit_arrival=orbit, n_manifold=100, verbose=verbose
        )

        return result

    def compute_transfer_cost(self, transfer_result):
        """计算转移代价

        参数：
            transfer_result: 转移设计结果字典

        返回：
            dict: 代价分析
        """
        if transfer_result is None:
            return None

        dv = transfer_result.get("delta_v", transfer_result.get("total_dv", None))
        tof = transfer_result.get("transfer_time", None)

        cost = {
            "delta_v": dv,
            "transfer_time": tof,
        }

        # 如果有物理单位
        if dv is not None and self.system.characteristic_velocity:
            cost["delta_v_km_s"] = dv * self.system.characteristic_velocity
        if tof is not None and self.system.characteristic_time:
            cost["transfer_time_days"] = tof * self.system.characteristic_time / 86400

        return cost

    def _solve_transfer_arc(self, dep_state, arr_state, tof):
        """求解转移弧

        使用优化方法求解从出发状态到到达状态的转移弧。

        参数：
            dep_state: 出发状态
            arr_state: 到达状态
            tof: 飞行时间

        返回：
            dict: 转移弧结果
        """
        dep_pos = dep_state[:3]
        arr_pos = arr_state[:3]

        # 初始猜测：连接两点的直线速度
        v_guess = (arr_pos - dep_pos) / tof

        def objective(v0):
            state = np.concatenate([dep_pos, v0])
            try:
                result = solve_ivp(
                    self.dynamics.equations_of_motion,
                    (0, tof),
                    state,
                    method="DOP853",
                    rtol=1e-10,
                    atol=1e-10,
                )
                if result.success:
                    final_pos = result.y[:3, -1]
                    return np.linalg.norm(final_pos - arr_pos) ** 2
                return 1e10
            except Exception:
                return 1e10

        result = minimize(
            objective, v_guess, method="Nelder-Mead", options={"maxiter": 500, "xatol": 1e-8}
        )

        if result.fun < 1e-4:  # 位置精度足够
            v0_opt = result.x
            state = np.concatenate([dep_pos, v0_opt])

            # 积分最优轨迹
            prop = solve_ivp(
                self.dynamics.equations_of_motion,
                (0, tof),
                state,
                method="DOP853",
                t_eval=np.linspace(0, tof, 1000),
                rtol=1e-12,
                atol=1e-12,
            )

            vf = prop.y[3:, -1]

            # 计算ΔV
            dv_departure = np.linalg.norm(v0_opt - dep_state[3:])
            dv_arrival = np.linalg.norm(arr_state[3:] - vf)

            return {
                "trajectory": prop.y.T,
                "times": prop.t,
                "v_departure": v0_opt,
                "v_arrival": vf,
                "dv_departure": dv_departure,
                "dv_arrival": dv_arrival,
                "total_dv": dv_departure + dv_arrival,
                "transfer_time": tof,
            }

        return None

    def _compute_manifold_crossings(
        self, orbit, manifold_type, n_trajectories, plane, value, direction
    ):
        """计算流形与庞加莱截面的交叉点

        参数：
            orbit: 周期轨道
            manifold_type: "stable" 或 "unstable"
            n_trajectories: 流形轨迹数量
            plane: 截面平面
            value: 截面位置
            direction: 积分方向

        返回：
            list: 交叉点列表
        """
        from ..algorithms.stability import StabilityAnalysis

        try:
            stability = StabilityAnalysis(orbit, self.dynamics)
            stability.compute_floquet_multipliers()
        except Exception:
            return []

        if stability.eigenvectors is None:
            return []

        # 选择特征方向
        magnitudes = np.abs(stability.eigenvalues)
        if manifold_type == "stable":
            idx = np.argmin(magnitudes)
        else:
            idx = np.argmax(magnitudes)

        manifold_dir = np.real(stability.eigenvectors[:, idx])
        manifold_dir = manifold_dir / np.linalg.norm(manifold_dir)

        eps = 1e-6
        plane_map = {"x": 0, "y": 1, "z": 2}
        plane_idx = plane_map.get(plane, 1)

        crossings = []
        n_points = min(n_trajectories, len(orbit.states))
        indices = np.linspace(0, len(orbit.states) - 1, n_points, dtype=int)

        t_max = 15.0
        if direction == "backward":
            t_span = (0, -t_max)
        else:
            t_span = (0, t_max)

        for i in indices:
            state = orbit.states[i]
            perturbed = state + eps * manifold_dir

            try:
                result = solve_ivp(
                    self.dynamics.equations_of_motion,
                    t_span,
                    perturbed,
                    method="DOP853",
                    t_eval=np.linspace(t_span[0], t_span[1], 3000),
                    rtol=1e-12,
                    atol=1e-12,
                )

                if result.success:
                    vals = result.y[plane_idx, :]
                    for j in range(len(vals) - 1):
                        if (vals[j] - value) * (vals[j + 1] - value) < 0:
                            # 插值求交叉状态
                            frac = (value - vals[j]) / (vals[j + 1] - vals[j])
                            cross_state = result.y[:, j] + frac * (
                                result.y[:, j + 1] - result.y[:, j]
                            )
                            crossings.append(
                                {
                                    "state": cross_state,
                                    "time": result.t[j] + frac * (result.t[j + 1] - result.t[j]),
                                    "orbit_index": int(i),
                                }
                            )
                            break  # 取第一个穿越点

            except Exception:
                continue

        return crossings

    def __str__(self):
        return f"InterOrbitTransfer(mu={self.mu})"
