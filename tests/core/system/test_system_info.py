"""
CR3BP_System.info() 方法测试模块

测试 info() 方法在不同模式和系统状态下的输出。
"""

import numpy as np
import pytest
from io import StringIO
from contextlib import redirect_stdout

from e2m2e import CR3BP_System


@pytest.fixture
def earth_moon_system():
    """创建地月系统实例"""
    return CR3BP_System.from_known_system("earth_moon")


@pytest.fixture
def initialized_system(earth_moon_system):
    """创建已初始化特征尺度的地月系统"""
    earth_moon_system.set_characteristic_scales(distance=384400, period=27.32 * 86400)
    return earth_moon_system


@pytest.fixture
def full_system(initialized_system):
    """创建已计算平动点并设置质量的完整系统"""
    initialized_system.compute_libration_points()
    initialized_system.mass_primary = 5.972e24
    initialized_system.mass_secondary = 7.342e22
    initialized_system.total_mass = 5.972e24 + 7.342e22
    return initialized_system


def _capture_info(system, mode="default"):
    """捕获 info() 方法的打印输出"""
    buf = StringIO()
    with redirect_stdout(buf):
        system.info(mode=mode)
    return buf.getvalue()


class TestInfoDefault:
    """测试 info() 默认模式"""

    def test_default_header_footer(self, earth_moon_system):
        """默认模式包含头部和尾部分隔线"""
        output = _capture_info(earth_moon_system)
        lines = output.strip().split("\n")
        assert lines[0] == "=" * 60
        assert lines[1] == "CR3BP 系统信息"
        assert lines[2] == "=" * 60
        assert lines[-1] == "=" * 60

    def test_default_basic_params(self, earth_moon_system):
        """默认模式包含基础参数"""
        output = _capture_info(earth_moon_system)
        assert "Earth-Moon" in output
        assert "1.215000e-02" in output
        assert "主天体：Earth" in output
        assert "次天体：Moon" in output

    def test_default_no_extended_info(self, earth_moon_system):
        """默认模式不包含扩展信息"""
        output = _capture_info(earth_moon_system)
        assert "系统状态:" not in output
        assert "特征尺度:" not in output
        assert "平动点位置" not in output
        assert "质量信息:" not in output


class TestInfoAll:
    """测试 info(mode='all') 模式"""

    def test_all_includes_system_state(self, earth_moon_system):
        """all 模式包含系统状态信息"""
        output = _capture_info(earth_moon_system, mode="all")
        assert "系统状态:" in output
        assert "是否初始化：False" in output
        assert "是否已计算平动点：False" in output

    def test_all_uninitialized_scales(self, earth_moon_system):
        """all 模式 - 未设置特征尺度时的提示"""
        output = _capture_info(earth_moon_system, mode="all")
        assert "特征尺度：未设置" in output

    def test_all_initialized_scales(self, initialized_system):
        """all 模式 - 已设置特征尺度时显示具体数值"""
        output = _capture_info(initialized_system, mode="all")
        assert "特征尺度:" in output
        assert "特征长度：384400.00 km" in output
        assert "特征速度" in output
        assert "平均角速度" in output
        assert "轨道周期" in output
        assert "半长轴：384400.00 km" in output

    def test_all_no_libration_points(self, earth_moon_system):
        """all 模式 - 未计算平动点时的提示"""
        output = _capture_info(earth_moon_system, mode="all")
        assert "平动点：未计算" in output

    def test_all_with_libration_points(self, full_system):
        """all 模式 - 已计算平动点时显示坐标"""
        output = _capture_info(full_system, mode="all")
        assert "平动点位置 (无量纲坐标):" in output
        assert "L1:" in output
        assert "L2:" in output
        assert "L3:" in output
        assert "L4:" in output
        assert "L5:" in output

    def test_all_no_mass_info(self, earth_moon_system):
        """all 模式 - 未设置质量时的提示"""
        output = _capture_info(earth_moon_system, mode="all")
        assert "质量信息：未设置" in output

    def test_all_with_mass_info(self, full_system):
        """all 模式 - 已设置质量时显示具体数值"""
        output = _capture_info(full_system, mode="all")
        assert "主天体质量" in output
        assert "次天体质量" in output
        assert "总质量" in output

    def test_all_state_flags_after_init(self, full_system):
        """all 模式 - 完全初始化后的状态标志"""
        output = _capture_info(full_system, mode="all")
        assert "是否初始化：True" in output
        assert "是否已计算平动点：True" in output


class TestInfoDifferentSystems:
    """测试不同系统的 info() 输出"""

    def test_sun_earth_system(self):
        system = CR3BP_System.from_known_system("sun_earth")
        output = _capture_info(system)
        assert "Sun-Earth" in output

    def test_sun_jupiter_system(self):
        system = CR3BP_System.from_known_system("sun_jupiter")
        output = _capture_info(system)
        assert "Sun-Jupiter" in output

    def test_custom_system(self):
        system = CR3BP_System(mu=0.001, primary="Star", secondary="Planet")
        output = _capture_info(system)
        assert "Star-Planet" in output
        assert "1.000000e-03" in output

