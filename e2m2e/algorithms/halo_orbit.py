"""
Halo轨道生成器模块

提供Halo轨道生成的高级接口，结合解析近似和微分修正生成精确周期轨道。
"""

from __future__ import annotations

import numpy as np
from typing import List


from ..core.system import CR3BP_System
from ..core.dynamics import CR3BP_Dynamics
from ..core.orbit import Orbit
from .differential_correction import DifferentialCorrection
from .analytical import compute_halo_initial_guess


class HaloOrbitGenerator:
    """Halo轨道生成器

    使用Richardson三阶近似生成初始猜测，结合微分修正生成精确的Halo周期轨道。

    属性：
        system: CR3BP_System对象
        dynamics: CR3BP_Dynamics对象
        corrector: 微分修正器

    示例：
        >>> system = CR3BP_System.from_known_system("earth_moon")
        >>> generator = HaloOrbitGenerator(system)
        >>> halo = generator.generate_seed_orbit(
        ...     libration_point=1,
        ...     amplitude_z=0.1,
        ...     halo_class=0
        ... )
    """

    def __init__(self, system: CR3BP_System) -> None:
        """初始化Halo轨道生成器

        参数：
            system: CR3BP_System对象
        """
        self.system = system
        self.dynamics = CR3BP_Dynamics(system)
        self.corrector = DifferentialCorrection(self.dynamics)

    def generate_seed_orbit(
        self,
        libration_point: int,
        amplitude_z: float,
        halo_class: int = 0,
        verbose: bool = False,
    ) -> Orbit:
        """生成Halo轨道初始猜测并修正

        使用Richardson三阶近似生成初始猜测，然后通过微分修正
        得到精确的周期轨道。

        参数：
            libration_point: 拉格朗日点 (1=L1, 2=L2)
            amplitude_z: Z方向振幅
            halo_class: 0=北Halo, 1=南Halo
            verbose: 是否打印详细信息

        返回：
            Orbit: Halo周期轨道
        """
        if libration_point not in [1, 2]:
            raise ValueError(f"libration_point必须是1或2，当前为{libration_point}")
        if amplitude_z <= 0:
            raise ValueError(f"amplitude_z必须为正数，当前为{amplitude_z}")
        if halo_class not in [0, 1]:
            raise ValueError(f"halo_class必须是0或1，当前为{halo_class}")

        mu = self.system.mu

        guess = compute_halo_initial_guess(
            mu=mu,
            z_amplitude=amplitude_z,
            L=libration_point,
            halo_class=halo_class,
        )

        initial_state = np.array(
            [
                guess["x0"],
                guess["y0"],
                guess["z0"],
                guess["vx0"],
                guess["vy0"],
                guess["vz0"],
            ]
        )

        if halo_class == 0:
            self.corrector.setup_halo_orbit_fixed_z0(
                z0=0.0,
                libration_point=libration_point,
            )
        else:
            self.corrector.setup_halo_orbit_fixed_z0(
                z0=0.0,
                libration_point=libration_point,
            )

        orbit = self.corrector.iterate_correction(
            initial_guess=initial_state,
            verbose=verbose,
        )

        if orbit is not None:
            orbit.family_type = "halo"
            orbit.parameters["libration_point"] = libration_point
            orbit.parameters["amplitude_z"] = amplitude_z
            orbit.parameters["halo_class"] = halo_class

        return orbit

    def generate_family(
        self,
        seed_orbit: Orbit,
        n_orbits: int = 50,
        direction: str = "positive",
        step_size: float = 0.001,
    ) -> List[Orbit]:
        """生成Halo轨道族

        使用自然参数延拓法生成Halo轨道族。

        参数：
            seed_orbit: 种子轨道
            n_orbits: 目标轨道数量
            direction: 延拓方向 ("positive", "negative", "both")
            step_size: 步长

        返回：
            List[Orbit]: Halo轨道族
        """
        if n_orbits < 1:
            raise ValueError(f"n_orbits必须大于0，当前为{n_orbits}")
        if direction not in ["positive", "negative", "both"]:
            raise ValueError(f"direction必须是positive/negative/both，当前为{direction}")

        family = [seed_orbit]

        directions = ["positive", "negative"] if direction == "both" else [direction]

        for direction in directions:
            z_amplitude = seed_orbit.parameters.get("amplitude_z", 0.1)
            step = step_size if direction == "positive" else -step_size

            for i in range(n_orbits - 1):
                new_z = z_amplitude + step * (i + 1)
                if new_z <= 0:
                    break

                try:
                    new_orbit = self.generate_seed_orbit(
                        libration_point=seed_orbit.parameters.get("libration_point", 1),
                        amplitude_z=new_z,
                        halo_class=seed_orbit.parameters.get("halo_class", 0),
                        verbose=False,
                    )
                    if new_orbit is not None:
                        family.append(new_orbit)
                except Exception:
                    break

        return family


__all__ = [
    "HaloOrbitGenerator",
]
