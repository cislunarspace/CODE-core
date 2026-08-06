"""tests/orbit_design 标准种子矩阵。

集中登记各轨道族「类型 × 平动点 → 标准初猜参数」，供 correction/
continuation/multiple_shooting/ephemeris 等后续阶段测试统一取用，避免
标准参数散落各测试。阶段 1 只填初猜四族（Halo/Lissajous/Axial/
Triangular）；DRO/DPO/SPO/LPO/Horseshoe 等星历族种子留待阶段 2-3 填入。
"""

# 地月 CR3BP 标准质量比（与 conftest.earth_moon_system 一致）
EARTH_MOON_MU = 0.01215058560962404

# --- Halo: compute_halo_initial_guess(mu, z_amplitude, L, halo_class) ---
# 北族（halo_class=0）/南族（halo_class=1）在 L1/L2 共用 z_amplitude。
HALO_POINTS = (1, 2)  # L1、L2
HALO_CLASSES = (0, 1)  # 北、南
HALO_Z_AMPLITUDE = 0.001

# --- Lissajous: compute_lissajous_initial_guess(
#         system, point, amplitude_in, amplitude_out, phase_in, phase_out) ---
# L1/L2 标准面内/面外振幅 (km) 与相位。
LISSAJOUS: dict[int, dict[str, float]] = {
    1: {"amplitude_in": 2500.0, "amplitude_out": 7500.0, "phase_in": 0.01, "phase_out": 0.55},
    2: {"amplitude_in": 2500.0, "amplitude_out": 7500.0, "phase_in": 0.01, "phase_out": 0.55},
}

# --- Axial: compute_axial_initial_guess(dynamics, collinear_point, vz0) ---
# 分岔种子由内部 Lyapunov 垂直临界扫描决定，此处只登记平动点；vz0 由调用方给。
AXIAL_POINTS = (1, 2)

# --- Triangular: compute_triangular_initial_guess(
#         system, point, amplitude_in, amplitude_out, phase_in, phase_out) ---
# L4/L5 标准面内/面外振幅 (km)。
TRIANGULAR: dict[int, dict[str, float]] = {
    4: {"amplitude_in": 8000.0, "amplitude_out": 6000.0},
    5: {"amplitude_in": 8000.0, "amplitude_out": 6000.0},
}

# --- 阶段 2-3 待填（星历族种子，来源标注于各条）---
# DRO: x0=0.79188556619742（Cui 2025, 单圈 DRO）
# DPO: x0=0.90（distant prograde orbit）
# SPO / LPO / Horseshoe: 待补，见 survey
