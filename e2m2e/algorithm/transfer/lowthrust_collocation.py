"""低推力 Hermite-Simpson 配点求解器。

与 :class:`~e2m2e.transfer.lowthrust_shooting.LowThrustShooting`（直接打靶）
并列的直接法求解器。把节点状态与控制都作为决策变量，用 Hermite-Simpson 缺陷
约束保证段间动力学连续——比单弧打靶更鲁棒、初猜要求低（Q-law 输出直接可用）。

## Hermite-Simpson 配点

把 ``[t0,tf]`` 分 N 段（N+1 节点），决策变量 = 节点状态 ``{x_i}``（7D）+
节点控制 ``{p_i=(throttle,θ₁,θ₂)}``（3D），共 ``10(N+1)``。缺陷约束（每段 7 维）：

```text
x_c = (x_i + x_{i+1})/2 + (dt/8)(f_i − f_{i+1})
缺陷_i = x_{i+1} − x_i − (dt/6)(f_i + 4·f(x_c, p_c) + f_{i+1})
```

缺陷为零 ⟺ 节点间动力学连续（Simpson 三阶积分与状态差一致）。

详见 ``docs/plans/lowthrust-collocation-prd.md``。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, minimize

from ..forces import PhysicalModel
from .lowthrust_shooting import EngineConfig, LowThrustSegment, LowThrustShootingSolution

if TYPE_CHECKING:
    from ..dynamics import System


class LowThrustCollocation:
    """低推力 Hermite-Simpson 配点求解器（min-fuel）。

    与 :class:`LowThrustShooting`（直接打靶）并列：打靶适合小规模高精度，配点
    适合大规模鲁棒。两者复用 ``EngineConfig``/``LowThrustSegment``/Q-law 初猜/
    ``LowThrustShootingSolution``。

    Args:
        system: 动力学系统（提供 origin）。
        forces: 非推力力模型，须支持 ``to_rust_spec``。
        engine: 推进配置。
        initial_state: 出发状态 ``[r,v]``，``(6,)``。
        initial_mass: 初始质量 kg。
        target_state: 目标末态 ``[r,v]``，``(6,)``。
        t0, tf: 起止时刻。
    """

    def __init__(
        self,
        system: System,
        forces: Sequence[PhysicalModel],
        engine: EngineConfig,
        initial_state: npt.ArrayLike,
        initial_mass: float,
        target_state: npt.ArrayLike,
        t0: float,
        tf: float,
    ) -> None:
        self._system = system
        self._engine = engine
        self._t0 = float(t0)
        self._tf = float(tf)
        if self._tf <= self._t0:
            raise ValueError(f"tf ({self._tf}) must be > t0 ({self._t0})")

        self._initial_state = np.asarray(initial_state, dtype=float)
        self._initial_mass = float(initial_mass)
        self._target_state = np.asarray(target_state, dtype=float)

        self._forces_py: list[tuple] = []
        for f in forces:
            spec = f.to_rust_spec(system)
            if spec is None:
                raise NotImplementedError(
                    f"force {f.__class__.__name__} lacks to_rust_spec; "
                    "collocation requires the Rust EOM path"
                )
            self._forces_py.append(spec)
        self._observer = getattr(system, "origin", "EARTH")

    # ---- 求解 ----

    def solve(
        self,
        n_segments: int,
        *,
        z0: npt.ArrayLike | None = None,
        ftol: float = 1e-9,
        maxiter: int = 300,
        verbose: bool = False,
    ) -> LowThrustShootingSolution:
        """求解 min-fuel 配点 NLP。

        Args:
            n_segments: 段数 N（节点数 N+1，决策变量 10(N+1)）。
            z0: 初猜决策向量；None 时节点状态沿初末线性插值、控制满推沿初速。
            ftol: SLSQP 目标容差。
            maxiter: SLSQP 最大迭代。
            verbose: 打印 SLSQP 进度。

        Returns:
            :class:`LowThrustShootingSolution`（节点状态 + 各段控制）。
        """
        if n_segments < 1:
            raise ValueError(f"n_segments must be >= 1, got {n_segments}")
        n_nodes = n_segments + 1
        n_var = 10 * n_nodes

        z_init = self._default_z0(n_segments) if z0 is None else np.asarray(z0, dtype=float)
        if z_init.shape != (n_var,):
            raise ValueError(f"z0 must have shape ({n_var},), got {z_init.shape}")

        lb, ub = self._bounds(n_segments)

        # 缺陷约束 + 端点约束
        constraints = [
            {"type": "eq", "fun": self._defect_constraints, "args": (n_segments,)},
            {"type": "eq", "fun": self._endpoint_constraints, "args": (n_segments,)},
        ]
        result = minimize(
            self._fuel_objective,
            z_init,
            args=(n_nodes,),
            method="SLSQP",
            bounds=Bounds(lb, ub),
            constraints=constraints,
            options={"ftol": ftol, "maxiter": maxiter, "disp": verbose},
        )
        return self._build_solution(
            result.x,
            n_segments,
            converged=bool(result.success),
            n_iter=int(result.nit),
            message=str(result.message),
        )

    def solve_from_qlaw(
        self,
        n_segments: int,
        target_oe: tuple[float, float, float],
        forces: Sequence[PhysicalModel],
        *,
        step: float = 120.0,
        maxiter: int = 200,
        verbose: bool = False,
    ) -> LowThrustShootingSolution:
        """用 Q-law 生成初猜，再配点打磨。"""
        from .qlaw import qlaw_guess

        # Q-law 产出 n_segments 段常量控制（n_segments 个节点控制，配点要 n_segments+1）
        y_seg, _segs, _qh, final = qlaw_guess(
            self._system,  # type: ignore[arg-type]
            forces,
            self._engine,
            self._initial_state,
            self._initial_mass,
            target_oe,
            self._t0,
            self._tf,
            n_segments,
            step=step,
            verbose=verbose,
        )
        z0 = self._qlaw_to_z0(y_seg, final, n_segments)
        return self.solve(n_segments, z0=z0, maxiter=maxiter, verbose=verbose)

    # ---- 决策向量编解码 ----

    def _default_z0(self, n_segments: int) -> np.ndarray:
        """默认初猜：节点状态初末线性插值，控制满推沿初速方向。"""
        n_nodes = n_segments + 1
        # 节点状态：x_0..x_N，位置速度线性插值到目标，质量保持初值
        x0 = np.concatenate([self._initial_state, [self._initial_mass]])
        xN = np.concatenate([self._target_state, [self._initial_mass * 0.95]])
        states = np.array([x0 + (xN - x0) * i / n_segments for i in range(n_nodes)])
        # 控制初猜：满推沿初速方向
        v0 = self._initial_state[3:6]
        theta1 = float(np.arctan2(v0[1], v0[0]))
        theta2 = float(np.arcsin(v0[2] / np.linalg.norm(v0))) if np.linalg.norm(v0) > 1e-15 else 0.0
        controls = np.tile([1.0, theta1, theta2], n_nodes)
        return np.concatenate([states.ravel(), controls])

    def _qlaw_to_z0(
        self, y_seg: np.ndarray, final_state: np.ndarray, n_segments: int
    ) -> np.ndarray:
        """Q-law 产出（n_segments 段控制）转配点初猜 z0。

        节点状态用 Q-law 末态线性回插 + 控制从 Q-law 的 n_segments 段扩到 n_segments+1 节点。
        """
        n_nodes = n_segments + 1
        x0 = np.concatenate([self._initial_state, [self._initial_mass]])
        states = np.array([x0 + (final_state - x0) * i / n_segments for i in range(n_nodes)])
        # Q-law 给了 n_segments 段控制（每段 3 维），配点要 n_nodes 个节点控制
        seg_ctrl = y_seg.reshape(n_segments, 3)
        controls = np.zeros((n_nodes, 3))
        controls[:n_segments] = seg_ctrl
        controls[-1] = seg_ctrl[-1]  # 末节点复用最后一段控制
        return np.concatenate([states.ravel(), controls.ravel()])

    def _bounds(self, n_segments: int) -> tuple[np.ndarray, np.ndarray]:
        """决策变量上下界：状态自由（大开区间）、throttle∈[0,1]、θ 自由。"""
        n_nodes = n_segments + 1
        # 状态：位置速度大开区间，质量 > 0
        big = 1e7
        state_lb = np.tile([-big, -big, -big, -big, -big, -big, 1.0], n_nodes)
        state_ub = np.tile([big, big, big, big, big, big, self._initial_mass * 1.01], n_nodes)
        # 控制：throttle∈[0,1]，θ 自由
        ctrl_lb = np.tile([0.0, -np.pi, -np.pi / 2], n_nodes)
        ctrl_ub = np.tile([1.0, np.pi, np.pi / 2], n_nodes)
        return np.concatenate([state_lb, ctrl_lb]), np.concatenate([state_ub, ctrl_ub])

    def _unpack(self, z: npt.NDArray[np.floating], n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
        """z → 节点状态 (n_nodes,7) + 节点控制 (n_nodes,3)。"""
        states = z[: 7 * n_nodes].reshape(n_nodes, 7)
        controls = z[7 * n_nodes :].reshape(n_nodes, 3)
        return states, controls

    def _eom(
        self,
        state7: npt.NDArray[np.floating],
        throttle: float,
        theta1: float,
        theta2: float,
        et: float,
    ) -> np.ndarray:
        """单点受控 EOM 求值（调 Rust）。"""
        from e2m2e._integrators import augmented_eom_7d_py

        return np.asarray(
            augmented_eom_7d_py(
                self._forces_py,
                self._observer,
                et,
                list(state7),
                (
                    self._engine.t_max,
                    self._engine.isp,
                    float(throttle),
                    float(theta1),
                    float(theta2),
                ),
            ),
            dtype=float,
        )

    # ---- 约束与目标 ----

    def _defect_constraints(
        self, z: npt.NDArray[np.floating], n_segments: int
    ) -> npt.NDArray[np.floating]:
        """Hermite-Simpson 缺陷约束，形状 (7*n_segments,)。"""
        n_nodes = n_segments + 1
        states, controls = self._unpack(z, n_nodes)
        dt = (self._tf - self._t0) / n_segments
        defects = np.zeros(7 * n_segments)
        for i in range(n_segments):
            xi = states[i]
            xip1 = states[i + 1]
            pi = controls[i]
            pip1 = controls[i + 1]
            et_i = self._t0 + i * dt
            fi = self._eom(xi, pi[0], pi[1], pi[2], et_i)
            fip1 = self._eom(xip1, pip1[0], pip1[1], pip1[2], et_i + dt)
            # Hermite 插值中点状态 + 线性插值中点控制
            xc = (xi + xip1) / 2 + (dt / 8) * (fi - fip1)
            pc = (pi + pip1) / 2
            fc = self._eom(xc, pc[0], pc[1], pc[2], et_i + dt / 2)
            # Simpson 缺陷
            defects[i * 7 : (i + 1) * 7] = xip1 - xi - (dt / 6) * (fi + 4 * fc + fip1)
        return defects

    def _endpoint_constraints(
        self, z: npt.NDArray[np.floating], n_segments: int
    ) -> npt.NDArray[np.floating]:
        """端点约束：x_0 固定（7维）+ x_N [r,v] 匹配目标（6维），共 13 维。"""
        n_nodes = n_segments + 1
        states, _ = self._unpack(z, n_nodes)
        x0_fixed = states[0] - np.concatenate([self._initial_state, [self._initial_mass]])
        xN_rv = states[-1][:6] - self._target_state
        return np.concatenate([x0_fixed, xN_rv])

    def _fuel_objective(self, z: npt.NDArray[np.floating], n_nodes: int) -> float:
        """目标：最小化 -m_N（最大化末态质量）。"""
        states, _ = self._unpack(z, n_nodes)
        return -float(states[-1][6])

    def _build_solution(
        self,
        z: npt.NDArray[np.floating],
        n_segments: int,
        *,
        converged: bool,
        n_iter: int,
        message: str,
    ) -> LowThrustShootingSolution:
        """从决策向量构造解。"""
        n_nodes = n_segments + 1
        states, controls = self._unpack(z, n_nodes)
        node_times = self._t0 + np.arange(n_nodes) * (self._tf - self._t0) / n_segments
        segments = tuple(
            LowThrustSegment(
                throttle=float(c[0]),
                direction=np.array(
                    [np.cos(c[1]) * np.cos(c[2]), np.sin(c[1]) * np.cos(c[2]), np.sin(c[2])]
                ),
            )
            for c in controls
        )
        final_mass = float(states[-1][6])
        return LowThrustShootingSolution(
            time=node_times,
            states=states,
            segments=segments,
            final_mass=final_mass,
            fuel_consumed=self._initial_mass - final_mass,
            converged=converged,
            n_iter=n_iter,
            message=message,
        )
