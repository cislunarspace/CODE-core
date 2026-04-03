"""
需求: SPICE 星历内核集成 (Layer 1a)

e2m2e 需要封装 SPICE Toolkit (spiceypy) 的核心功能，以支持高精度星历模型。
SPICEManager 负责:
  1. 加载 JPL DE440/DE438 星历内核 (.bsp)
  2. UTC ↔ ET (历书时) 时间转换
  3. 查询天体在指定参考系下的位置和速度
  4. 查询天体间距离

参考实现:
  SEMpy 使用 spiceypy 库封装 CSPICE 函数。
  SEMpy 中 Primary 类定义了各天体的 NAIF ID (Earth=399, Moon=301, Sun=10)。
  SEMpy 中 Ephemeris 类管理天体列表和 GM 值。

天体 NAIF ID:
  Sun = 10, Earth = 399, Moon = 301
  Earth-Moon Barycenter = 3

依赖:
  pip install spiceypy
  需要下载 JPL DE440 或 DE438 内核文件 (.bsp)
"""

import os
import pytest
import numpy as np
from numpy.testing import assert_allclose

from e2m2e.core import SPICEManager


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def spice_manager():
    """创建 SPICEManager 实例"""
    return SPICEManager()


@pytest.fixture
def loaded_spice(spice_manager, spice_kernel_path):
    """加载了 DE440 内核的 SPICEManager"""
    spice_manager.load_kernel(spice_kernel_path)
    yield spice_manager
    spice_manager.unload_kernel(spice_kernel_path)


# =============================================================================
# Test SPICEManager 初始化
# =============================================================================
class TestSPICEManagerInit:
    """测试 SPICEManager 创建和基本属性"""

    def test_create_instance(self):
        """应能创建 SPICEManager 实例"""
        manager = SPICEManager()
        assert manager is not None

    def test_has_load_kernel_method(self, spice_manager):
        """SPICEManager 应有 load_kernel 方法"""
        assert hasattr(spice_manager, "load_kernel")
        assert callable(spice_manager.load_kernel)

    def test_has_unload_kernel_method(self, spice_manager):
        """SPICEManager 应有 unload_kernel 方法"""
        assert hasattr(spice_manager, "unload_kernel")

    def test_has_utc_to_et_method(self, spice_manager):
        """SPICEManager 应有 utc_to_et 方法"""
        assert hasattr(spice_manager, "utc_to_et")

    def test_has_get_body_state_method(self, spice_manager):
        """SPICEManager 应有 get_body_state 方法"""
        assert hasattr(spice_manager, "get_body_state")


# =============================================================================
# Test 内核加载
# =============================================================================
class TestSPICEKernelLoading:
    """测试 SPICE 内核文件的加载和卸载"""

    def test_load_de440_kernel(self, spice_manager, spice_kernel_path):
        """应能成功加载 DE440 内核"""
        spice_manager.load_kernel(spice_kernel_path)
        spice_manager.unload_kernel(spice_kernel_path)

    def test_load_nonexistent_kernel_raises(self, spice_manager):
        """加载不存在的文件应抛出异常"""
        with pytest.raises((FileNotFoundError, OSError, RuntimeError)):
            spice_manager.load_kernel("/nonexistent/path/de440.bsp")

    def test_load_and_unload(self, spice_manager, spice_kernel_path):
        """应能加载后卸载内核"""
        spice_manager.load_kernel(spice_kernel_path)
        spice_manager.unload_kernel(spice_kernel_path)
        # 卸载后再次卸载不应崩溃
        spice_manager.unload_kernel(spice_kernel_path)


# =============================================================================
# Test 时间转换
# =============================================================================
class TestSPICETimeConversion:
    """测试 SPICE 时间转换功能"""

    def test_utc_to_et_returns_float(self, loaded_spice):
        """utc_to_et 应返回浮点数"""
        et = loaded_spice.utc_to_et("2025-06-21T11:00:00")
        assert isinstance(et, float)

    def test_j2000_epoch_et_zero(self, loaded_spice):
        """J2000 历元 (2000-01-01 12:00:00) 的 ET 应接近 0"""
        et = loaded_spice.utc_to_et("2000-01-01T12:00:00")
        assert_allclose(et, 0.0, atol=100.0)

    def test_et_increases_with_time(self, loaded_spice):
        """ET 应随时间递增"""
        et1 = loaded_spice.utc_to_et("2025-01-01T00:00:00")
        et2 = loaded_spice.utc_to_et("2025-06-01T00:00:00")
        assert et2 > et1

    def test_et_seconds_per_year(self, loaded_spice):
        """ET 每年约增加 3.156e7 秒"""
        et_2024 = loaded_spice.utc_to_et("2024-01-01T12:00:00")
        et_2025 = loaded_spice.utc_to_et("2025-01-01T12:00:00")
        delta_et = et_2025 - et_2024
        assert_allclose(delta_et, 365.25 * 86400, rtol=0.005)

    def test_et_to_utc_round_trip(self, loaded_spice):
        """UTC → ET → UTC 应恢复原始时间"""
        utc_in = "2025-06-21T11:00:00"
        et = loaded_spice.utc_to_et(utc_in)
        utc_out = loaded_spice.et_to_utc(et)
        assert utc_in == utc_out


# =============================================================================
# Test 天体状态查询
# =============================================================================
class TestSPICEBodyStateQuery:
    """测试天体状态查询功能"""

    def test_moon_state_shape(self, loaded_spice, reference_epoch):
        """月球状态向量应为 6 维 [x, y, z, vx, vy, vz]"""
        et = loaded_spice.utc_to_et(reference_epoch)
        state = loaded_spice.get_body_state(target="MOON", et=et, frame="J2000", observer="EARTH")
        assert state.shape == (6,)

    def test_sun_state_shape(self, loaded_spice, reference_epoch):
        """太阳状态向量应为 6 维"""
        et = loaded_spice.utc_to_et(reference_epoch)
        state = loaded_spice.get_body_state(target="SUN", et=et, frame="J2000", observer="EARTH")
        assert state.shape == (6,)

    def test_moon_position_physical_range(self, loaded_spice, reference_epoch):
        """月球位置应在物理合理范围内 (距地球约 384,400 km)"""
        et = loaded_spice.utc_to_et(reference_epoch)
        state = loaded_spice.get_body_state(target="MOON", et=et, frame="J2000", observer="EARTH")
        r = np.linalg.norm(state[:3])
        assert 350000 < r < 420000, f"月球距地球 {r:.0f} km，超出合理范围"

    def test_sun_distance_physical_range(self, loaded_spice, reference_epoch):
        """太阳距地球约 1 AU ≈ 1.496e8 km"""
        et = loaded_spice.utc_to_et(reference_epoch)
        state = loaded_spice.get_body_state(target="SUN", et=et, frame="J2000", observer="EARTH")
        r = np.linalg.norm(state[:3])
        assert 1.46e8 < r < 1.53e8, f"太阳距地球 {r:.0f} km，超出合理范围"

    def test_moon_velocity_physical_range(self, loaded_spice, reference_epoch):
        """月球速度应在物理合理范围内 (约 1 km/s)"""
        et = loaded_spice.utc_to_et(reference_epoch)
        state = loaded_spice.get_body_state(target="MOON", et=et, frame="J2000", observer="EARTH")
        v = np.linalg.norm(state[3:])
        assert 0.9 < v < 1.2, f"月球速度 {v:.3f} km/s，超出合理范围"

    def test_body_state_at_different_times(self, loaded_spice):
        """不同时刻天体位置应不同"""
        et1 = loaded_spice.utc_to_et("2025-01-01T00:00:00")
        et2 = loaded_spice.utc_to_et("2025-06-01T00:00:00")
        state1 = loaded_spice.get_body_state("MOON", et1, "J2000", "EARTH")
        state2 = loaded_spice.get_body_state("MOON", et2, "J2000", "EARTH")
        assert not np.allclose(state1, state2)

    def test_get_body_position(self, loaded_spice, reference_epoch):
        """应有单独获取位置的方法"""
        et = loaded_spice.utc_to_et(reference_epoch)
        pos = loaded_spice.get_body_position(target="MOON", et=et, frame="J2000", observer="EARTH")
        assert pos.shape == (3,)
        r = np.linalg.norm(pos)
        assert 350000 < r < 420000


# =============================================================================
# Test 天体物理参数
# =============================================================================
class TestSPICEBodyParameters:
    """测试天体物理参数查询"""

    def test_get_gm_values(self, loaded_spice):
        """应能获取天体 GM 值"""
        gm_earth = loaded_spice.get_gm("EARTH")
        gm_moon = loaded_spice.get_gm("MOON")
        gm_sun = loaded_spice.get_gm("SUN")

        assert gm_earth > 0
        assert gm_moon > 0
        assert gm_sun > 0
        assert gm_sun > gm_earth > gm_moon

    def test_gm_earth_value(self, loaded_spice):
        """地球 GM 值应接近 398600 km^3/s^2"""
        gm = loaded_spice.get_gm("EARTH")
        assert_allclose(gm, 398600.0, rtol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
