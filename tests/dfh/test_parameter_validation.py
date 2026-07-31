"""参数校验测试（移植 MATLAB ``tests/validation/TestParameterValidation.m``）。

无效参数应触发预期的 ``ValueError``。范围取值以说明文档与 MATLAB
``design_orbit.m``/``control_orbit.m`` 的 ``arguments`` 块为准；Python 侧
校验发生在入口函数（``design_orbit``/``control_orbit``/``format_inputs_propagate``），
不触碰 SPICE/力模型构造，快测。
"""

import pytest

from e2m2e.dfh import control_orbit, design_orbit
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
        """角动量管理模式 4-6 本期未实现（#261），应明确拒绝。"""
        with pytest.raises(ValueError, match="control_mode"):
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
