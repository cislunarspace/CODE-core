"""Halo 轨道族编排模块

从 ``continuation.py`` 拆出的 Halo 专用编排：种子生成、自然参数族延拓、
伪弧长（PAL）延拓。``Continuation`` 实例上同名方法仍可用 — 在
``continuation.py`` 文件末尾以方法重绑定的形式保留调用语法。
"""

from __future__ import annotations

import logging

import numpy as np

from ..core.orbit import Orbit, OrbitFamily

logger = logging.getLogger(__name__)


def _tag_halo_family(orb: Orbit, libration_point: int, halo_class: int) -> None:
    """标记 Halo 族类型与参数"""
    orb.family_type = "halo"
    orb.parameters["libration_point"] = libration_point
    orb.parameters["halo_class"] = halo_class
    z0 = float(orb.states[0, 2])
    orb.parameters["amplitude_z"] = abs(z0)


def generate_halo_seed_orbit(
    continuation,
    libration_point: int,
    amplitude_z: float,
    halo_class: int = 0,
    verbose: bool = False,
) -> Orbit:
    """生成 Halo 种子轨道（作为 ``Continuation`` 实例方法使用）"""
    if libration_point not in [1, 2]:
        raise ValueError(f"libration_point必须是1或2，当前为{libration_point}")
    if amplitude_z <= 0:
        raise ValueError(f"amplitude_z必须为正数，当前为{amplitude_z}")
    if halo_class not in [0, 1]:
        raise ValueError(f"halo_class必须是0或1，当前为{halo_class}")

    if verbose:
        halo_label = "北" if halo_class == 0 else "南"
        logger.info("生成Halo轨道: L%d %s Halo", libration_point, halo_label)
        logger.info("  Z振幅: %s", amplitude_z)

    mu = continuation.correction.dynamics.system.mu

    from .halo_initial_guess import compute_halo_initial_guess

    guess = compute_halo_initial_guess(
        mu=mu,
        z_amplitude=amplitude_z,
        L=libration_point,
        halo_class=halo_class,
    )

    initial_z = amplitude_z if halo_class == 0 else -amplitude_z

    initial_state = np.array(
        [
            guess["x0"],
            0.0,
            initial_z,
            guess["vx0"],
            guess["vy0"],
            guess["vz0"],
        ]
    )

    if halo_class == 0:
        continuation.correction.setup_halo_orbit_fixed_z0(
            z0=amplitude_z,
            libration_point=libration_point,
        )
    else:
        continuation.correction.setup_halo_orbit_fixed_z0(
            z0=-amplitude_z,
            libration_point=libration_point,
        )

    initial_orbit = Orbit(
        states=initial_state.reshape(1, -1),
        times=np.array([0.0]),
        system=continuation.correction.dynamics.system,
    )
    initial_orbit.period = 2.0 * guess["T_half"]

    continuation.correction.max_iterations = 150
    continuation.correction.tolerance = 1e-5

    if verbose:
        logger.info("  初始猜测: x0=%.6f, vy0=%.6f", guess["x0"], guess["vy0"])
        logger.info("  预估周期: %.4f TU", initial_orbit.period)

    orbit = continuation.correction.iterate_correction(
        initial_guess=initial_orbit,
        verbose=verbose,
    )

    if orbit is not None:
        _tag_halo_family(orbit, libration_point, halo_class)
        if verbose:
            logger.info("[ok] Halo轨道生成成功: 周期=%.6f TU", orbit.period)

    return orbit


def generate_halo_family(
    continuation,
    seed_orbit: Orbit,
    n_orbits: int = 50,
    direction: str = "positive",
    step_size: float = 0.001,
    z_range=None,
    verbose: bool = False,
    progress_callback=None,
) -> list[Orbit]:
    """Halo 自然参数延拓族（作为 ``Continuation`` 实例方法使用）"""
    if n_orbits < 1:
        raise ValueError(f"n_orbits必须大于0，当前为{n_orbits}")
    if direction not in ["positive", "negative", "both"]:
        raise ValueError(f"direction必须是positive/negative/both，当前为{direction}")

    family = [seed_orbit]
    libration_point = int(seed_orbit.parameters.get("libration_point", 1))
    halo_class = int(seed_orbit.parameters.get("halo_class", 0))
    seed_z = float(seed_orbit.states[0, 2])

    default_z_limit = 0.5 if halo_class == 0 else -0.5
    z_threshold = 1e-4 if halo_class == 0 else -1e-4

    if z_range is not None:
        z_min, z_max = z_range
        if z_min >= z_max:
            raise ValueError(f"z_range必须满足z_min < z_max，当前为({z_min}, {z_max})")
        forward = z_max > seed_z
        backward = z_min < seed_z
        dirs: list[str] = []
        if forward:
            dirs.append("positive")
        if backward:
            dirs.append("negative")
        if not dirs:
            logger.warning("z_range不包含种子轨道z0，不延拓")
            return family
        directions = dirs
        logger.info(
            "开始生成Halo轨道族: z范围=[%.4f, %.4f], 方向=%s, 最大数量=%d",
            z_min,
            z_max,
            "/".join(directions),
            n_orbits,
        )
    else:
        directions = ["positive", "negative"] if direction == "both" else [direction]
        logger.info("开始生成Halo轨道族: 目标数量=%d, 方向=%s", n_orbits, direction)

    logger.info(
        "  种子轨道: L%d %s Halo, z0=%.6f",
        libration_point,
        "北" if halo_class == 0 else "南",
        seed_z,
    )

    min_step = 1e-4
    max_step = 0.05
    growth = continuation.step_increase_factor
    shrink = continuation.step_reduction_factor

    for dir_name in directions:
        current_orbit = seed_orbit
        current_step = float(step_size)
        current_z = float(current_orbit.states[0, 2])

        if z_range is not None:
            z_min_val, z_max_val = z_range
            z_limit = z_max_val if dir_name == "positive" else z_min_val
        else:
            z_limit = default_z_limit

        if verbose:
            dir_label = "正向" if dir_name == "positive" else "反向"
            logger.info("--- %s延拓 (边界=%.4f) ---", dir_label, z_limit)

        for i in range(n_orbits - 1):
            if len(family) >= n_orbits:
                if verbose:
                    logger.info("  达到全局轨道数上限 %d, 终止", n_orbits)
                break

            dz = current_step if dir_name == "positive" else -current_step
            target_z = current_z + dz

            if halo_class == 0:
                if target_z <= z_threshold or target_z >= z_limit:
                    if verbose:
                        logger.info("  达到z边界 %.4f, 终止", z_limit)
                    break
            else:
                if target_z >= z_threshold or target_z <= z_limit:
                    if verbose:
                        logger.info("  达到z边界 %.4f, 终止", z_limit)
                    break

            continuation.correction.setup_halo_orbit_fixed_z0(
                z0=target_z,
                libration_point=libration_point,
            )
            continuation.correction.max_iterations = 150
            continuation.correction.tolerance = 1e-6

            guess_state = current_orbit.states[0].copy()
            guess_state[2] = target_z
            guess = Orbit(
                states=guess_state.reshape(1, -1),
                times=np.array([0.0]),
                system=continuation.correction.dynamics.system,
            )
            guess.period = current_orbit.period

            orbit = continuation.correction.iterate_correction(guess, verbose=False)

            if orbit is not None and orbit.correction_success:
                _tag_halo_family(orbit, libration_point, halo_class)
                family.append(orbit)
                current_orbit = orbit
                current_z = target_z

                if continuation.step_size_adaptation:
                    if orbit.correction_iterations < 5:
                        current_step = min(current_step * growth, max_step)
                    elif orbit.correction_iterations > 20:
                        current_step = max(current_step * shrink, min_step)

                if progress_callback is not None:
                    progress_callback(i + 1, n_orbits - 1, orbit, dir_name)

                if verbose and (i + 1) % 5 == 0:
                    logger.info(
                        "  第%d条: z=%.5f, x=%.6f, T=%.4f",
                        i + 1,
                        target_z,
                        orbit.states[0, 0],
                        orbit.period,
                    )
            else:
                current_step = max(current_step * shrink, min_step)
                if current_step <= min_step:
                    if verbose:
                        logger.warning("  第%d步修正失败且步长已达最小, 终止", i + 1)
                    break
                if verbose:
                    logger.info(
                        "  第%d步修正失败, 缩小步长至%.6f后重试",
                        i + 1,
                        current_step,
                    )
                continue

    logger.info("[ok] 轨道族生成完成: 共%d条轨道", len(family))
    return family


def halo_pseudo_arclength_continuation(
    continuation,
    seed_orbit: Orbit,
    n_orbits: int = 50,
    direction: str = "both",
    step_size: float = 0.0045,
    step_size_negative: float | None = None,
    verbose: bool = True,
    TolPAL: float = 1e-6,
    TolDiffCorr: float = 1e-6,
    IterMax: int = 100,
    dc_scheme: str = "adaptive",
    directional_increment: bool = True,
    progress_callback=None,
) -> OrbitFamily:
    """Halo 轨道族伪弧长延拓（作为 ``Continuation`` 实例方法使用）"""
    libration_point = int(seed_orbit.parameters.get("libration_point", 1))
    halo_class = int(seed_orbit.parameters.get("halo_class", 0))
    seed_z_amplitude = seed_orbit.parameters.get("amplitude_z", 0.1)

    if direction not in ("positive", "negative", "both"):
        raise ValueError("direction 须为 positive / negative / both")

    if step_size_negative is None:
        step_size_negative = step_size

    continuation.correction.max_iterations = 150

    if verbose:
        logger.info("=" * 30)
        logger.info("Halo 伪弧长延拓（对齐 continuation_PAL_CR3BP + FAMILY_L1Halo_North）")
        logger.info("  种子: L%d %s Halo", libration_point, "北" if halo_class == 0 else "南")
        logger.info("  z_amplitude(参数): %.4f", seed_z_amplitude)
        logger.info("  每支新轨道数 N = %d", n_orbits)
        logger.info(
            "  正向 |DeltaS| = %s, 负向 |DeltaS| = %s",
            step_size,
            step_size_negative,
        )
        logger.info(
            "  dc_scheme = %s, DirectionalIncrement = %s",
            dc_scheme,
            directional_increment,
        )
        logger.info("=" * 30)

    orbit_family = OrbitFamily([seed_orbit])

    def tag(orb: Orbit) -> None:
        _tag_halo_family(orb, libration_point, halo_class)

    branches: list[tuple[str, float, int, int]] = []
    if direction in ("positive", "both"):
        td_pos = -1 if halo_class == 1 else 1
        branches.append(("positive", step_size, 1, td_pos))
    if direction in ("negative", "both"):
        branches.append(("negative", step_size_negative, 0, -1))

    for br_name, ds_mag, tv, td in branches:
        if verbose:
            logger.info("--- Halo 延拓支: %s (|DeltaS|=%s) ---", br_name, ds_mag)

        sub = continuation.pseudo_arclength_continuation(
            seed_orbit,
            n_orbits=n_orbits,
            step_size=ds_mag,
            direction="positive" if br_name == "positive" else "negative",
            verbose=verbose,
            TolPAL=TolPAL,
            TolDiffCorr=TolDiffCorr,
            IterMax=IterMax,
            dc_scheme=dc_scheme,
            libration_point=libration_point,
            directional_increment=directional_increment,
            target_vector=tv,
            target_direction=td,
            progress_callback=progress_callback,
        )
        for o in sub.orbits[1:]:
            tag(o)
            orbit_family.add_orbit(o)

    if verbose:
        logger.info("延拓完成：共 %d 条轨道", len(orbit_family))
        z_values = [o.parameters.get("amplitude_z", 0) for o in orbit_family]
        if z_values:
            logger.info("  z_amplitude 范围: [%.4f, %.4f]", min(z_values), max(z_values))

    return orbit_family
