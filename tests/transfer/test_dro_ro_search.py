"""
DRO-RO转移搜索模块测试

测试TASK-009的实现:
- TransferSearchVariables数据结构
- DROROTransferSearch类的采样和速度计算
"""

import numpy as np
import pytest
import os
import json
from pathlib import Path

from e2m2e.core import Orbit, CR3BP_System, CR3BP_Dynamics
from e2m2e.transfer import DROROTransferSearch, TransferSearchVariables, TransferSearchResult


# 测试数据路径
TEST_DATA_DIR = Path(__file__).parent.parent.parent / "transfer-orbit-design" / "output"


class TestTransferSearchVariables:
    """测试TransferSearchVariables数据结构"""
    
    @pytest.fixture
    def sample_orbit(self):
        """创建测试轨道"""
        # 创建简单的圆形轨道近似
        t = np.linspace(0, 6.5, 200)
        x = 0.9 + 0.1 * np.cos(t)
        y = 0.1 * np.sin(t)
        z = np.zeros_like(t)
        vx = -0.1 * np.sin(t)
        vy = 0.1 * np.cos(t)
        vz = np.zeros_like(t)
        
        states = np.column_stack([x, y, z, vx, vy, vz])
        orbit = Orbit(states, t)
        orbit.period = 6.5
        return orbit
    
    def test_search_variables_creation(self, sample_orbit):
        """测试搜索变量创建"""
        vars = TransferSearchVariables(
            departure_orbit=sample_orbit,
            departure_time_index=50,
            alpha=1.5,
            beta=0.0
        )
        
        assert vars.departure_time_index == 50
        assert vars.alpha == 1.5
        assert vars.beta == 0.0
    
    def test_departure_state_property(self, sample_orbit):
        """测试departure_state属性"""
        vars = TransferSearchVariables(
            departure_orbit=sample_orbit,
            departure_time_index=100,
            alpha=1.0,
            beta=0.0
        )
        
        state = vars.departure_state
        assert state.shape == (6,)
        assert isinstance(state, np.ndarray)
    
    def test_velocity_ratios_property(self, sample_orbit):
        """测试velocity_ratios属性"""
        vars = TransferSearchVariables(
            departure_orbit=sample_orbit,
            departure_time_index=0,
            alpha=2.0,
            beta=0.1
        )
        
        ratios = vars.velocity_ratios
        assert ratios == (2.0, 0.1)
    
    def test_invalid_alpha_warning(self, sample_orbit):
        """测试alpha超出范围警告"""
        import warnings
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            vars = TransferSearchVariables(
                departure_orbit=sample_orbit,
                departure_time_index=0,
                alpha=5.0,  # 超出范围
                beta=0.0
            )
            # 应该有警告但不会抛异常
            assert len(w) == 1


class TestDROROTransferSearch:
    """测试DROROTransferSearch类"""
    
    @pytest.fixture
    def system(self):
        """创建CR3BP系统"""
        return CR3BP_System(mu=0.01215, primary="earth", secondary="moon")
    
    @pytest.fixture
    def dynamics(self, system):
        """创建动力学模型"""
        return CR3BP_Dynamics(system)
    
    @pytest.fixture
    def searcher(self, system, dynamics):
        """创建搜索器"""
        return DROROTransferSearch(system, dynamics)
    
    @pytest.fixture
    def dro_orbit(self):
        """加载DRO测试数据"""
        # 查找DRO数据文件
        dro_dir = TEST_DATA_DIR / "dro"
        if not dro_dir.exists():
            pytest.skip("DRO数据不存在，跳过测试")
        
        dro_files = list(dro_dir.glob("dro_family_*.json"))
        if not dro_files:
            pytest.skip("DRO数据文件不存在")
        
        # 加载第一个文件
        with open(dro_files[0], 'r') as f:
            data = json.load(f)
        
        states = np.array(data.get('states', []))
        if len(states) == 0:
            pytest.skip("DRO数据为空")
        
        # 估计周期
        times = np.arange(len(states)) * 0.01
        orbit = Orbit(states, times)
        orbit.period = times[-1] if len(times) > 0 else 6.5
        
        return orbit
    
    @pytest.fixture
    def ro_orbit(self):
        """加载RO测试数据"""
        ro_dir = TEST_DATA_DIR / "ro"
        if not ro_dir.exists():
            pytest.skip("RO数据不存在，跳过测试")
        
        ro_files = list(ro_dir.glob("ro_31_family_*.json"))
        if not ro_files:
            pytest.skip("RO数据文件不存在")
        
        with open(ro_files[0], 'r') as f:
            data = json.load(f)
        
        states = np.array(data.get('states', []))
        if len(states) == 0:
            pytest.skip("RO数据为空")
        
        times = np.arange(len(states)) * 0.01
        orbit = Orbit(states, times)
        orbit.period = times[-1] if len(times) > 0 else 6.5
        
        return orbit
    
    def test_searcher_creation(self, searcher):
        """测试搜索器创建"""
        assert searcher is not None
        assert searcher.mu == 0.01215
    
    def test_sample_departure_points(self, searcher, dro_orbit):
        """测试出发点采样"""
        n_points = 50
        states = searcher.sample_departure_points(dro_orbit, n_points)
        
        assert len(states) == n_points
        assert states[0].shape == (6,)
    
    def test_compute_departure_velocity(self, searcher):
        """测试速度计算"""
        # 测试状态 [x, y, z, vx, vy, vz]
        state = np.array([0.9, 0.0, 0.0, 0.0, 1.0, 0.0])
        
        # alpha=1.0, beta=0.0 应该保持原始速度方向
        v_new = searcher.compute_departure_velocity(state, alpha=1.0, beta=0.0)
        
        # 速度大小应该相同
        assert np.abs(np.linalg.norm(v_new) - np.linalg.norm(state[3:])) < 1e-10
        
        # 方向应该相同
        original_dir = state[3:] / np.linalg.norm(state[3:])
        new_dir = v_new / np.linalg.norm(v_new)
        assert np.allclose(original_dir, new_dir)
    
    def test_compute_departure_velocity_scaling(self, searcher):
        """测试速度缩放"""
        state = np.array([0.9, 0.0, 0.0, 0.0, 1.0, 0.0])
        original_v_mag = np.linalg.norm(state[3:])
        
        # alpha=2.0应该使速度翻倍
        v_new = searcher.compute_departure_velocity(state, alpha=2.0, beta=0.0)
        new_v_mag = np.linalg.norm(v_new)
        
        assert np.abs(new_v_mag - 2.0 * original_v_mag) < 1e-10
    
    def test_forward_integrate(self, searcher, dro_orbit):
        """测试前向积分"""
        # 获取一个初始状态
        state = dro_orbit.states[0]
        
        # 积分短时间
        times, states = searcher.forward_integrate(
            state,
            t_span=(0.0, 0.1)
        )
        
        assert len(times) > 0
        assert states.shape[1] == 6
    
    def test_compute_min_distance(self, searcher, dro_orbit, ro_orbit):
        """测试最小距离计算"""
        # 创建一段测试轨迹
        trajectory = dro_orbit.states[:50]
        
        min_dist, min_idx = searcher.compute_min_distance_to_orbit(
            trajectory, ro_orbit
        )
        
        assert min_dist >= 0
        assert min_idx >= 0
        assert min_idx < len(trajectory)
    
    def test_find_intersection(self, searcher, dro_orbit, ro_orbit):
        """测试相交检测"""
        # 轨迹应该不与RO相交
        trajectory = dro_orbit.states[:50]
        
        found, int_state, int_idx = searcher.find_intersection(
            trajectory, ro_orbit
        )
        
        # DRO和RO一般不相交，所以这个测试只是检查函数运行正常
        assert isinstance(found, bool)
        assert int_state.shape == (6,)
    
    def test_search_single_departure(self, searcher, dro_orbit, ro_orbit):
        """测试单出发点搜索"""
        # 使用较小的网格加快测试
        dep_state = dro_orbit.states[0]
        
        results = searcher.search_single_departure(
            dep_state,
            ro_orbit,
            alpha_range=(1.0, 1.5),
            beta_range=(0.0, 0.0),
            n_alpha=5,
            n_beta=1
        )
        
        assert isinstance(results, list)
        # 结果数量
        assert len(results) == 5
    
    def test_grid_search_small(self, searcher, dro_orbit, ro_orbit):
        """测试小规模网格搜索"""
        # 使用非常小的网格进行快速测试
        results = searcher.grid_search(
            departure_orbit=dro_orbit,
            arrival_orbit=ro_orbit,
            alpha_range=(1.0, 1.2),
            beta_range=(0.0, 0.0),
            n_alpha=3,
            n_beta=1,
            n_departure=5,
            parallel=False  # 使用串行便于测试
        )
        
        assert isinstance(results, list)


class TestTransferSearchResult:
    """测试TransferSearchResult数据结构"""
    
    @pytest.fixture
    def sample_orbit(self):
        """创建测试轨道"""
        t = np.linspace(0, 6.5, 100)
        x = 0.9 + 0.1 * np.cos(t)
        y = 0.1 * np.sin(t)
        z = np.zeros_like(t)
        vx = -0.1 * np.sin(t)
        vy = 0.1 * np.cos(t)
        vz = np.zeros_like(t)
        
        states = np.column_stack([x, y, z, vx, vy, vz])
        orbit = Orbit(states, t)
        orbit.period = 6.5
        return orbit
    
    def test_result_creation(self, sample_orbit):
        """测试结果创建"""
        vars = TransferSearchVariables(
            departure_orbit=sample_orbit,
            departure_time_index=10,
            alpha=1.5,
            beta=0.0
        )
        
        result = TransferSearchResult(
            search_vars=vars,
            transfer_time=5.0,
            min_distance=0.1,
            intersection_found=False,
            status="success"
        )
        
        assert result.transfer_time == 5.0
        assert result.min_distance == 0.1
        assert result.status == "success"
    
    def test_is_feasible(self, sample_orbit):
        """测试is_feasible属性"""
        vars = TransferSearchVariables(
            departure_orbit=sample_orbit,
            departure_time_index=0,
            alpha=1.0,
            beta=0.0
        )
        
        # 相交的情况
        result1 = TransferSearchResult(
            search_vars=vars,
            intersection_found=True,
            min_distance=0.1
        )
        assert result1.is_feasible
        
        # 距离较小的情况
        result2 = TransferSearchResult(
            search_vars=vars,
            intersection_found=False,
            min_distance=0.03  # 小于阈值0.05
        )
        assert result2.is_feasible
        
        # 距离较大的情况
        result3 = TransferSearchResult(
            search_vars=vars,
            intersection_found=False,
            min_distance=0.1  # 大于阈值0.05
        )
        assert not result3.is_feasible
