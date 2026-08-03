"""参数校验测试（移植 MATLAB ``tests/validation/TestParameterValidation.m``）。

无效参数应触发预期的 ``ValueError``。范围取值以说明文档与 MATLAB
``design_orbit.m``/``control_orbit.m`` 的 ``arguments`` 块为准；Python 侧
校验发生在入口函数（``design_orbit``/``control_orbit``/``format_inputs_propagate``），
不触碰 SPICE/力模型构造，快测。
"""

import pytest

from e2m2e.algorithm.design import design_orbit
from e2m2e.algorithm.station_keeping import control_orbit
from e2m2e.io import format_inputs_propagate


# =============================================================================
# design_orbit（MATLAB design_orbit arguments 块范围校验）
# =============================================================================
class TestDesignOrbitValidation:
    def test_invalid_orbit_type(self):
        with pytest.raises(ValueError, match="DRO/NRHO/Halo"):
            design_orbit("Lyapunov")

    def test_duration_zero_rejected(self):
        with pytest.raises(ValueError, match="20"):
            design_orbit("Halo", duration=0.0)

    def test_duration_over_20_rejected(self):
        with pytest.raises(ValueError, match="20"):
            design_orbit("Halo", duration=20.1)

    def test_perilune_height_too_low_rejected(self):
        with pytest.raises(ValueError, match="100~10000"):
            design_orbit("NRHO", perilune_height=50.0)

    def test_perilune_height_too_high_rejected(self):
        with pytest.raises(ValueError, match="100~10000"):
            design_orbit("NRHO", perilune_height=20000.0)

    def test_phase_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="0~1"):
            design_orbit("Halo", phase=1.5)

    def test_halo_amplitude_too_large_rejected(self):
        with pytest.raises(ValueError, match="73000"):
            design_orbit("Halo", amplitude=80000.0)

    def test_dro_amplitude_too_small_rejected(self):
        with pytest.raises(ValueError, match="1737~110000"):
            design_orbit("DRO", amplitude=1000.0)

    def test_lissajous_amplitude_too_large_rejected(self):
        """Lissajous L2 面内/外振幅超 7600 km 应拒绝。"""
        with pytest.raises(ValueError, match="7600"):
            design_orbit("Lissajous", collinear_point=2, amplitude_in=10000.0)
        with pytest.raises(ValueError, match="7600"):
            design_orbit("Lissajous", collinear_point=2, amplitude_out=10000.0)

    def test_l4_amplitude_in_too_large_rejected(self):
        with pytest.raises(ValueError, match="10000"):
            design_orbit("L4", amplitude_in=20000.0)

    def test_l4_amplitude_out_too_large_rejected(self):
        with pytest.raises(ValueError, match="76000"):
            design_orbit("L4", amplitude_out=80000.0)

    def test_l5_amplitude_out_too_large_rejected(self):
        with pytest.raises(ValueError, match="76000"):
            design_orbit("L5", amplitude_out=80000.0)

    def test_lissajous_phase_out_of_range(self):
        with pytest.raises(ValueError, match="phase_in"):
            design_orbit("Lissajous", phase_in=1.5)
        with pytest.raises(ValueError, match="phase_out"):
            design_orbit("L4", phase_out=-0.5)

    def test_unknown_perturbation_key_rejected(self):
        """摄动字典含未知开关时立即拒绝（未知 key 在序列化层校验）。"""
        with pytest.raises(ValueError, match="未知摄动开关"):
            format_inputs_propagate(
                epoch=[2024, 1, 1, 0, 0, 0],
                duration=1.0,
                initial_state=[1, 2, 3, 4, 5, 6],
                perturbation={"bogus": 1},
            )


# =============================================================================
# control_orbit（MATLAB control_orbit arguments 块范围校验）
# =============================================================================
class TestControlOrbitValidation:
    def test_control_interval_zero_rejected(self):
        with pytest.raises(ValueError, match="control_mode"):
            control_orbit("dummy.txt", control_mode=0)

    def test_num_controls_zero_rejected(self):
        with pytest.raises(ValueError, match="num_controls"):
            control_orbit("dummy.txt", num_controls=0)

    def test_control_mode_out_of_range(self):
        """control_mode 超出 1-6 范围应拒绝。"""
        with pytest.raises(ValueError, match="control_mode"):
            control_orbit("dummy.txt", control_mode=7)

    def test_momentum_mode_requires_engine_layout(self):
        """角动量管理模式（4-6）需提供 engine_layout。"""
        with pytest.raises(ValueError, match="engine_layout"):
            control_orbit("dummy.txt", control_mode=4)

    def test_thrust_min_nonpositive_rejected(self):
        with pytest.raises(ValueError, match="thrust_min"):
            control_orbit("dummy.txt", thrust_min=0.0)


# =============================================================================
# propagate（MATLAB propagate_orbit arguments 块范围校验）
# =============================================================================
class TestPropagateValidation:
    def test_bad_state_size(self):
        with pytest.raises(ValueError, match="initial_state"):
            format_inputs_propagate(
                epoch=[2024, 1, 1, 0, 0, 0], duration=1.0, initial_state=[1, 2, 3]
            )


# =============================================================================
# Lissajous / L4 / L5 补充边界校验（Phase 2 #255）
# =============================================================================
class TestLissajousTriangularValidation:
    """Lissajous 与 L4/L5 轨道额外边界条件校验。"""

    # --- Lissajous L3（amplitude 限制 100000，远大于 L1/L2 的 7600） ---
    def test_lissajous_l3_amplitude_in_too_large(self):
        with pytest.raises(ValueError, match="100000"):
            design_orbit("Lissajous", collinear_point=3, amplitude_in=150000.0)

    def test_lissajous_l3_amplitude_out_too_large(self):
        with pytest.raises(ValueError, match="100000"):
            design_orbit("Lissajous", collinear_point=3, amplitude_out=150000.0)

    # --- Lissajous amplitude 为零或负 ---
    def test_lissajous_amplitude_in_zero_rejected(self):
        with pytest.raises(ValueError, match="0"):
            design_orbit("Lissajous", collinear_point=2, amplitude_in=0.0)

    def test_lissajous_amplitude_in_negative_rejected(self):
        with pytest.raises(ValueError, match="0"):
            design_orbit("Lissajous", collinear_point=2, amplitude_in=-500.0)

    def test_lissajous_amplitude_out_zero_rejected(self):
        with pytest.raises(ValueError, match="0"):
            design_orbit("Lissajous", collinear_point=2, amplitude_out=0.0)

    def test_lissajous_amplitude_out_negative_rejected(self):
        with pytest.raises(ValueError, match="0"):
            design_orbit("Lissajous", collinear_point=2, amplitude_out=-1000.0)

    # --- Lissajous 无效 collinear_point ---
    def test_lissajous_invalid_collinear_point_rejected(self):
        with pytest.raises(ValueError, match="1/2/3"):
            design_orbit("Lissajous", collinear_point=4)

    # --- Lissajous L1 amplitude 限制（同 L2：7600 km） ---
    def test_lissajous_l1_amplitude_in_too_large(self):
        with pytest.raises(ValueError, match="7600"):
            design_orbit("Lissajous", collinear_point=1, amplitude_in=8000.0)

    # --- L4/L5 amplitude 为零或负 ---
    def test_l4_amplitude_in_zero_rejected(self):
        with pytest.raises(ValueError, match="0"):
            design_orbit("L4", amplitude_in=0.0)

    def test_l4_amplitude_in_negative_rejected(self):
        with pytest.raises(ValueError, match="0"):
            design_orbit("L4", amplitude_in=-500.0)

    def test_l4_amplitude_out_zero_rejected(self):
        with pytest.raises(ValueError, match="0"):
            design_orbit("L4", amplitude_out=0.0)

    def test_l5_amplitude_in_zero_rejected(self):
        with pytest.raises(ValueError, match="0"):
            design_orbit("L5", amplitude_in=0.0)

    def test_l5_amplitude_out_zero_rejected(self):
        with pytest.raises(ValueError, match="0"):
            design_orbit("L5", amplitude_out=0.0)

    # --- L4/L5 amplitude 上界 ---
    def test_l5_amplitude_in_too_large_rejected(self):
        with pytest.raises(ValueError, match="10000"):
            design_orbit("L5", amplitude_in=15000.0)

    # --- L4/L5 phase 越界 ---
    def test_l4_phase_in_out_of_range(self):
        with pytest.raises(ValueError, match="phase_in"):
            design_orbit("L4", phase_in=1.5)

    def test_l5_phase_in_out_of_range(self):
        with pytest.raises(ValueError, match="phase_in"):
            design_orbit("L5", phase_in=-0.1)
