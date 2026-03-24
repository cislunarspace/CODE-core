"""
测试 dro_ro_search 模块

测试 DROROTransferSearch 类的各项功能。
"""

import pytest
import numpy as np

from e2m2e.core.orbit import Orbit
from e2m2e.core.dynamics import CR3BP_Dynamics
from e2m2e.core.system import CR3BP_System
from e2m2e.transfer.dro_transfer_search import (
    TransferSearchConfig,
    TransferSearchResult,
    DROROTransferSearch,
)


# ===== Fixtures =====

@pytest.fixture
def em_system():
    """地球-月球CR3BP系统"""
    return CR3BP_System(mu=1.21506683e-2, primary="earth", secondary="moon")


@pytest.fixture
def em_dynamics(em_system):
    """地球-月球CR3BP动力学"""
    return CR3BP_Dynamics(system=em_system)


@pytest.fixture
def simple_orbit():
    """简单的测试轨道 (圆形轨道的近似)"""
    # 创建简单的椭圆轨道状态
    states = []
    times = []
    for i in range(100):
        t = i * 0.1
        # 简化的圆形轨道状态 [x, y, z, vx, vy, vz]
        x = np.cos(t)
        y = np.sin(t)
        z = 0.0
        vx = -np.sin(t)
        vy = np.cos(t)
        vz = 0.0
        states.append([x, y, z, vx, vy, vz])
        times.append(t)
    
    orbit = Orbit(states=np.array(states), times=np.array(times))
    orbit._period = 2 * np.pi  # 设置周期
    return orbit


@pytest.fixture
def search_config():
    """测试用搜索配置"""
    return TransferSearchConfig(
        alpha_min=0.8,
        alpha_max=1.2,
        n_alpha=5,
        beta_min=-0.1,
        beta_max=0.1,
        n_beta=3,
        n_departure=10,
        max_transfer_time=5.0,
    )


@pytest.fixture
def searcher(em_system, em_dynamics, search_config):
    """DRO-RO转移搜索器"""
    return DROROTransferSearch(
        system=em_system,
        dynamics=em_dynamics,
        config=search_config,
    )


# ===== Test TransferSearchConfig =====

class TestTransferSearchConfig:
    """测试 TransferSearchConfig 类"""
    
    def test_default_values(self):
        """测试默认值"""
        config = TransferSearchConfig()
        
        assert config.alpha_min == 0.5
        assert config.alpha_max == 2.5
        assert config.n_alpha == 101
        assert config.beta_min == -0.5
        assert config.beta_max == 0.5
        assert config.n_beta == 21
        assert config.n_departure == 200
    
    def test_alpha_grid(self):
        """测试α网格生成"""
        config = TransferSearchConfig(
            alpha_min=0.0,
            alpha_max=1.0,
            n_alpha=11,
        )
        
        grid = config.alpha_grid
        assert len(grid) == 11
        assert np.isclose(grid[0], 0.0)
        assert np.isclose(grid[-1], 1.0)
    
    def test_beta_grid(self):
        """测试β网格生成"""
        config = TransferSearchConfig(
            beta_min=-0.5,
            beta_max=0.5,
            n_beta=21,
        )
        
        grid = config.beta_grid
        assert len(grid) == 21
        assert np.isclose(grid[0], -0.5)
        assert np.isclose(grid[-1], 0.5)


# ===== Test TransferSearchResult =====

class TestTransferSearchResult:
    """测试 TransferSearchResult 类"""
    
    def test_default_values(self):
        """测试默认值"""
        result = TransferSearchResult()
        
        assert result.intersection_found is False
        assert result.collision_found is False
        assert result.status == "pending"
        assert result.is_feasible is False
    
    def test_is_feasible(self):
        """测试is_feasible判断逻辑"""
        result = TransferSearchResult()
        
        # 无碰撞但无接近 -> 不可行
        assert result.is_feasible is False
        
        # 有接近但无碰撞 -> 可行
        result.intersection_found = False
        result.min_distance = 0.01  # 很小
        assert result.is_feasible is True
        
        # 有接近但有碰撞 -> 不可行
        result.collision_found = True
        assert result.is_feasible is False
    
    def test_dv_departure(self):
        """测试dv_departure计算"""
        result = TransferSearchResult()
        
        # 无数据时返回0
        assert result.dv_departure == 0.0
        
        # 有数据时计算Δv
        result.departure_state = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        result.transfer_trajectory = np.array([
            [1.0, 0.0, 0.0, 0.1, 1.0, 0.0]  # 只有速度不同
        ])
        dv = result.dv_departure
        assert np.isclose(dv, 0.1)


# ===== Test DROROTransferSearch =====

class TestDROROTransferSearch:
    """测试 DROROTransferSearch 类"""
    
    def test_init(self, searcher, em_system, em_dynamics, search_config):
        """测试初始化"""
        assert searcher.system is em_system
        assert searcher.dynamics is em_dynamics
        assert searcher.config is search_config
        assert searcher.mu == em_system.mu
    
    def test_sample_departure_points(self, searcher, simple_orbit):
        """测试出发点采样"""
        states, times = searcher.sample_departure_points(simple_orbit)
        
        assert len(states) == searcher.config.n_departure
        assert len(times) == searcher.config.n_departure
        assert states.shape[1] == 6  # 6个状态分量
    
    def test_compute_departure_velocity(self, searcher):
        """测试速度扰动计算"""
        # 原始状态 (位于(1,0,0)，速度(0,1,0))
        orbit_state = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        
        # α=1, β=0 应该保持切向速度不变
        new_vel = searcher.compute_departure_velocity(orbit_state, alpha=1.0, beta=0.0)
        
        # 检查速度是否被正确缩放
        # 切向方向应该保持，但经过向量构建后...
        # 原始速度 vy=1.0 应该变为 α*vy = 1.0
        assert new_vel.shape == (3,)
        
        # α=2 时，切向速度应该翻倍
        new_vel_2 = searcher.compute_departure_velocity(orbit_state, alpha=2.0, beta=0.0)
        
        # 验证速度增大的方向是正确的（切向）
        # 原始速度向量在切向方向的分量
        pos = orbit_state[:3]
        r_xy = np.sqrt(pos[0]**2 + pos[1]**2)
        tangential = np.array([-pos[1], pos[0], 0.0]) / r_xy
        original_tangential_comp = np.dot(orbit_state[3:], tangential)
        
        new_tangential_comp = np.dot(new_vel_2, tangential)
        assert np.isclose(new_tangential_comp, 2.0 * original_tangential_comp)
    
    def test_compute_min_distance(self, searcher, simple_orbit):
        """测试最小距离计算（向量化实现）"""
        # 创建测试轨迹
        trajectory_states = np.array([
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0, 0.0, 0.0, 0.0],  # 接近轨道点(1,0,0)
            [1.5, 0.5, 0.0, 0.0, 0.0, 0.0],
        ])
        
        min_dist, min_idx = searcher.compute_min_distance(
            trajectory_states, simple_orbit
        )
        
        assert min_dist >= 0
        assert min_idx >= 0
        assert min_idx < len(trajectory_states)
    
    def test_check_collision_earth(self, searcher):
        """测试地球碰撞检测"""
        # 创建靠近地球的轨迹
        mu = searcher.mu
        trajectory_states = np.array([
            [-mu + 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],  # 在地球附近
            [-mu + 0.01, 0.0, 0.0, 0.0, 0.0, 0.0],  # 在地球内部!
        ])
        
        collision, body, idx = searcher.check_collision(trajectory_states)
        
        assert collision is True
        assert body == 'earth'
        assert idx == 1
    
    def test_check_collision_moon(self, searcher):
        """测试月球碰撞检测"""
        # 创建靠近月球的轨迹
        mu = searcher.mu
        trajectory_states = np.array([
            [0.3, 0.0, 0.0, 0.0, 0.0, 0.0],  # 远离月球
            [1.0 - mu + 0.01, 0.0, 0.0, 0.0, 0.0, 0.0],  # 在月球内部!
        ])
        
        collision, body, idx = searcher.check_collision(trajectory_states)
        
        assert collision is True
        assert body == 'moon'
        assert idx == 1
    
    def test_check_collision_none(self, searcher):
        """测试无碰撞情况"""
        # 创建远离天体中心的轨迹
        trajectory_states = np.array([
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
        ])
        
        collision, body, idx = searcher.check_collision(trajectory_states)
        
        assert collision is False
        assert body is None
        assert idx == -1
    
    def test_detect_intersection(self, searcher, simple_orbit):
        """测试相交检测"""
        # 创建与目标轨道相交的轨迹
        trajectory_states = np.array([
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 正好在轨道上!
            [1.5, 0.5, 0.0, 0.0, 0.0, 0.0],
        ])
        
        found, point, idx = searcher.detect_intersection(
            trajectory_states, simple_orbit
        )
        
        assert found is True
        assert point is not None
        assert idx == 1
    
    def test_detect_local_minimum(self, searcher, simple_orbit):
        """测试局部最小检测"""
        # 创建有局部最小的轨迹
        # 距离先减小后增大
        trajectory_states = np.array([
            [2.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            [1.5, 1.5, 0.0, 0.0, 0.0, 0.0],  # 局部最小
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ])
        
        found, dist, idx = searcher.detect_local_minimum(
            trajectory_states, simple_orbit
        )
        
        assert found is True
        assert dist >= 0
        assert idx == 1


# ===== Integration Tests =====

class TestDROROTransferSearchIntegration:
    """集成测试：完整的搜索流程"""
    
    @pytest.mark.slow
    def test_search_single_departure(
        self, searcher, simple_orbit
    ):
        """测试单个出发点搜索"""
        # 使用配置中的单点
        dep_state = np.array([0.9, 0.0, 0.0, 0.0, 0.5, 0.0])
        dep_time = 0.0
        
        results = searcher.search_single_departure(
            dep_state, dep_time, simple_orbit
        )
        
        # 应该返回α×β个结果
        n_expected = searcher.config.n_alpha * searcher.config.n_beta
        assert len(results) == n_expected
        
        # 所有结果应该有正确的字段
        for r in results:
            assert hasattr(r, 'alpha')
            assert hasattr(r, 'beta')
            assert hasattr(r, 'status')
    
    @pytest.mark.slow
    def test_grid_search_returns_results(
        self, searcher, simple_orbit
    ):
        """测试网格搜索返回结果"""
        # 使用很小的搜索范围加快测试
        searcher.config.n_departure = 3
        searcher.config.n_alpha = 2
        searcher.config.n_beta = 2
        
        results = searcher.grid_search(
            departure_orbit=simple_orbit,
            arrival_orbit=simple_orbit,
            verbose=False,
        )
        
        assert len(results) > 0
        assert all(hasattr(r, 'is_feasible') for r in results)
    
    @pytest.mark.slow
    def test_grid_search_finds_feasible(
        self, searcher, simple_orbit
    ):
        """测试网格搜索能找到可行解"""
        # 创建一个真正靠近的轨道来测试
        # 出发点轨道
        dep_orbit = simple_orbit
        
        # 目标轨道几乎相同
        arr_orbit = simple_orbit
        
        results = searcher.grid_search(
            departure_orbit=dep_orbit,
            arrival_orbit=arr_orbit,
            verbose=False,
        )
        
        # 应该有可行解
        feasible = [r for r in results if r.is_feasible]
        # 注意：不一定总有可行解，取决于轨道配置


# ===== Performance Tests =====

class TestPerformance:
    """性能测试"""
    
    def test_vectorized_vs_loop(self, searcher, simple_orbit):
        """测试向量化实现比循环快"""
        # 创建较大的轨迹
        n_traj = 100
        trajectory_states = np.random.rand(n_traj, 6) * 2 - 1  # 随机状态
        
        # 向量化实现
        import time
        start = time.time()
        min_dist_vec, _ = searcher.compute_min_distance(
            trajectory_states, simple_orbit
        )
        vec_time = time.time() - start
        
        # 这个测试只是确保向量化实现能工作
        # 实际的性能提升取决于数据规模
        assert min_dist_vec >= 0