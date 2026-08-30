"""tests/orbit_design 标准种子矩阵。

集中登记各轨道族的类型 × 平动点 → 标准初猜参数映射，供 correction/
continuation/multiple_shooting/ephemeris 等后续阶段测试统一取用，避免
标准参数散落各测试。

- 初猜四族（Halo/Lissajous/Axial/Triangular）：阶段 1 已填，见上方。
- 修正/延拓种子（DRO/Halo/Axial/DPO/Lyapunov/SPO）：阶段 2 已填，见下方。
- Horseshoe 等大振幅星历族种子留待阶段 3。
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

# --- 阶段 2：修正/延拓标准种子（来源标注于各条）---

# DRO 地月（Cui et al. 2025 单圈 DRO，与 tests/algorithm/conftest 一致）
DRO_X0 = 0.79188556619742
DRO_VY0 = 0.573665890385585
DRO_PERIOD = 6.307498

# Halo 族种子面外振幅（无量纲，Richardson 三阶近似在小振幅下精度高；
# 与 data/templates/seed._HALO_SEED_Z0 一致）。北族 halo_class=0。
HALO_SEED_Z0 = 0.001
HALO_SEED_CLASS = 0
HALO_SEED_POINTS = (1, 2)  # L1、L2

# Axial 族种子面外速度（无量纲 DU/TU；与 _AXIAL_SEED_VZ0 一致）
# axial 修正/延拓测试已按 ADR 0037 移出默认套件（单次修正 ~2 min 超预算），
# 种子登记保留供回归测试取用。
AXIAL_SEED_VZ0 = 0.001
AXIAL_SEED_POINT = 1  # L1

# DPO 族标准种子（data/templates/seed._DPO_SEED_*；顺行 vy0<0）
DPO_X0 = 0.90
DPO_VY0 = -0.247645
DPO_PERIOD = 2.5022

# Lyapunov L1 平面族：种子由共线点面内线性模态构造（x0 = x_L1 − 偏移，
# vy0 由特征向量相位定出使 y(0)=ẋ(0)=0；构造方式与
# axial_initial_guess._correct_lyapunov_fixed_x0 的 guess=None 分支一致）。
LYAPUNOV_POINT = 1
LYAPUNOV_OFFSET = 0.01  # x0 = x_L1 − offset（DU）

# 三角平动点区域的周期族：三模态 Triangular 初猜（TRIANGULAR 的 amplitude_in/out）
# 是拟周期的，无周期修正 setup；修正/延拓测试改用 SPO（L4/L5 短周期族，
# 该区域唯一可用全周期闭合修正的周期族）。
SPO_SEED_POINT = 4  # L4
SPO_SEED_AMPLITUDE_KM = 1000.0
