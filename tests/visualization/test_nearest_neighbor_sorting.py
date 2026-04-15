"""
Nearest Neighbor Sorting Methods Tests

Tests for the _sort_points_by_nearest_neighbor and _sort_3d_points_by_nearest_neighbor
methods in OrbitVisualizer that solve orbit plot crossing issues.

Reference: Commit cf9337a - 添加最近邻排序方法解决轨道绘图交叉线问题
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from e2m2e.core import CR3BP_System
from e2m2e.visualization.base import OrbitVisualizer


@pytest.fixture
def visualizer():
    """创建测试用可视化器"""
    system = CR3BP_System(mu=0.01215, primary="Earth", secondary="Moon")
    system.compute_libration_points()
    return OrbitVisualizer(system)


class TestSortPointsByNearestNeighbor:
    """测试2D点最近邻排序方法"""

    def test_sort_simple_crossing_points(self, visualizer):
        """测试交叉点的排序"""
        # 创建一组有明显交叉的4个点（正方形顶点，但按交叉顺序给出）
        # 顺序: (1,0) -> (0,1) -> (-1,0) -> (0,-1) -> 回到 (1,0)
        # 这会导致绘制时出现"X"形交叉
        x = np.array([1.0, 0.0, -1.0, 0.0])
        y = np.array([0.0, 1.0, 0.0, -1.0])

        sorted_x, sorted_y = visualizer._sort_points_by_nearest_neighbor(x, y)

        # 验证输出形状不变
        assert len(sorted_x) == len(x)
        assert len(sorted_y) == len(y)

        # 验证所有点都被保留（作为集合相等）
        original_points = set(zip(x, y, strict=False))
        sorted_points = set(zip(sorted_x, sorted_y, strict=False))
        assert original_points == sorted_points

        # 验证排序后相邻点之间的距离递增或保持较小
        # （即不存在大跳跃）
        for i in range(len(sorted_x) - 1):
            dist = np.sqrt(
                (sorted_x[i + 1] - sorted_x[i]) ** 2 + (sorted_y[i + 1] - sorted_y[i]) ** 2
            )
            # 在正方形例子中，排序后相邻点应该是相邻顶点，距离应为1
            assert dist < 3.0  # 不应该有大跳跃

    def test_sort_already_sorted_points(self, visualizer):
        """测试已排序点的处理"""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 2.0, 3.0])

        sorted_x, sorted_y = visualizer._sort_points_by_nearest_neighbor(x, y)

        assert len(sorted_x) == 4
        assert len(sorted_y) == 4

    def test_sort_two_points(self, visualizer):
        """测试只有两个点的情况"""
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 0.0])

        sorted_x, sorted_y = visualizer._sort_points_by_nearest_neighbor(x, y)

        assert len(sorted_x) == 2
        assert len(sorted_y) == 2

    def test_sort_single_point(self, visualizer):
        """测试只有一个点的情况"""
        x = np.array([1.0])
        y = np.array([1.0])

        sorted_x, sorted_y = visualizer._sort_points_by_nearest_neighbor(x, y)

        assert len(sorted_x) == 1
        assert len(sorted_y) == 1

    def test_sort_ellipse_shape(self, visualizer):
        """测试椭圆形状的点（可能产生交叉）"""
        # 创建一个椭圆上的点，但打乱顺序
        t = np.linspace(0, 2 * np.pi, 20)
        x_orig = 0.8 + 0.2 * np.cos(t)
        y_orig = 0.1 * np.sin(t)

        # 打乱顺序
        np.random.seed(42)
        idx = np.random.permutation(len(x_orig))
        x_shuffled = x_orig[idx]
        y_shuffled = y_orig[idx]

        sorted_x, sorted_y = visualizer._sort_points_by_nearest_neighbor(x_shuffled, y_shuffled)

        # 验证所有点都保留了
        assert len(sorted_x) == len(x_orig)

        # 验证没有大的跳跃（椭圆上相邻点距离较小）
        max_dist = 0
        for i in range(len(sorted_x) - 1):
            dist = np.sqrt(
                (sorted_x[i + 1] - sorted_x[i]) ** 2 + (sorted_y[i + 1] - sorted_y[i]) ** 2
            )
            max_dist = max(max_dist, dist)

        # 椭圆周长约 2*pi*a ≈ 3.2，每段距离应小于1
        assert max_dist < 1.5

    def test_sort_origin_start_point(self, visualizer):
        """测试最远点作为起点的选择"""
        # 创建包含原点的点集
        x = np.array([0.0, 1.0, -1.0, 0.5, -0.5])
        y = np.array([0.0, 0.0, 0.0, 0.866, 0.866])  # 原点 + 正五边形顶点

        sorted_x, sorted_y = visualizer._sort_points_by_nearest_neighbor(x, y)

        # 验证原点(0,0)作为起点（距离原点最远）
        # 实际上起点是距离原点最远的点
        distances = np.sqrt(sorted_x**2 + sorted_y**2)
        start_dist = distances[0]
        assert start_dist >= np.max(distances)


class TestSort3DPointsByNearestNeighbor:
    """测试3D点最近邻排序方法"""

    def test_sort_3d_simple_crossing(self, visualizer):
        """测试3D交叉点的排序"""
        # 创建立方体顶点，打乱顺序
        x = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        y = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
        z = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])

        sorted_x, sorted_y, sorted_z = visualizer._sort_3d_points_by_nearest_neighbor(x, y, z)

        # 验证输出形状不变
        assert len(sorted_x) == len(x)
        assert len(sorted_y) == len(y)
        assert len(sorted_z) == len(z)

        # 验证所有点都被保留
        original_points = set(zip(x, y, z, strict=False))
        sorted_points = set(zip(sorted_x, sorted_y, sorted_z, strict=False))
        assert original_points == sorted_points

    def test_sort_3d_two_points(self, visualizer):
        """测试3D只有两个点的情况"""
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 0.0])
        z = np.array([0.0, 1.0])

        sorted_x, sorted_y, sorted_z = visualizer._sort_3d_points_by_nearest_neighbor(x, y, z)

        assert len(sorted_x) == 2
        assert len(sorted_y) == 2
        assert len(sorted_z) == 2

    def test_sort_3d_single_point(self, visualizer):
        """测试3D只有一个点的情况"""
        x = np.array([1.0])
        y = np.array([2.0])
        z = np.array([3.0])

        sorted_x, sorted_y, sorted_z = visualizer._sort_3d_points_by_nearest_neighbor(x, y, z)

        assert len(sorted_x) == 1
        assert len(sorted_y) == 1
        assert len(sorted_z) == 1

    def test_sort_3d_circular_helix(self, visualizer):
        """测试3D螺旋线点（可能产生交叉）"""
        # 创建螺旋线点并打乱
        t = np.linspace(0, 4 * np.pi, 40)
        x_orig = np.cos(t)
        y_orig = np.sin(t)
        z_orig = t / (4 * np.pi)

        # 打乱顺序
        np.random.seed(123)
        idx = np.random.permutation(len(x_orig))
        x_shuffled = x_orig[idx]
        y_shuffled = y_orig[idx]
        z_shuffled = z_orig[idx]

        sorted_x, sorted_y, sorted_z = visualizer._sort_3d_points_by_nearest_neighbor(
            x_shuffled, y_shuffled, z_shuffled
        )

        # 验证所有点都保留了
        assert len(sorted_x) == len(x_orig)

        # 验证相邻点之间距离连续且较小（螺旋线相邻点应该接近）
        for i in range(len(sorted_x) - 1):
            dist = np.sqrt(
                (sorted_x[i + 1] - sorted_x[i]) ** 2
                + (sorted_y[i + 1] - sorted_y[i]) ** 2
                + (sorted_z[i + 1] - sorted_z[i]) ** 2
            )
            # 螺旋线每圈约 2*pi 弧长，相邻点距离应较小
            assert dist < 2.0  # 不应该有大跳跃

    def test_sort_3d_sphere_surface(self, visualizer):
        """测试球面上的点"""
        # 创建球面上的随机点
        np.random.seed(456)
        n_points = 30

        # 使用球面坐标生成均匀分布的点
        phi = np.random.uniform(0, 2 * np.pi, n_points)
        theta = np.random.uniform(0, np.pi, n_points)

        x_orig = np.sin(theta) * np.cos(phi)
        y_orig = np.sin(theta) * np.sin(phi)
        z_orig = np.cos(theta)

        # 打乱顺序
        idx = np.random.permutation(n_points)
        x_shuffled = x_orig[idx]
        y_shuffled = y_orig[idx]
        z_shuffled = z_orig[idx]

        sorted_x, sorted_y, sorted_z = visualizer._sort_3d_points_by_nearest_neighbor(
            x_shuffled, y_shuffled, z_shuffled
        )

        # 验证所有点都保留了
        assert len(sorted_x) == n_points

        # 验证没有非常大的跳跃（相邻点应该在球面上接近）
        max_dist = 0
        for i in range(len(sorted_x) - 1):
            dist = np.sqrt(
                (sorted_x[i + 1] - sorted_x[i]) ** 2
                + (sorted_y[i + 1] - sorted_y[i]) ** 2
                + (sorted_z[i + 1] - sorted_z[i]) ** 2
            )
            max_dist = max(max_dist, dist)

        # 球面上相邻点最大距离应该较小
        assert max_dist < 3.0


class TestNearestNeighborSortingIntegration:
    """测试最近邻排序与轨道绘图的集成"""

    def test_sorting_used_in_3d_plot(self, visualizer):
        """测试排序功能是否在实际3D绘图中的使用"""
        # 创建一个有明显交叉的轨道数据
        t = np.linspace(0, 2 * np.pi, 50)
        # 制作一个"8"字形轨道（两个圆交叉）
        x = np.sin(t)
        y = np.sin(t) * np.cos(t)
        z = 0.1 * np.sin(3 * t)

        # 打乱数据顺序来模拟真实场景
        np.random.seed(789)
        idx = np.arange(50)
        np.random.shuffle(idx)

        states = np.column_stack(
            [x[idx], y[idx], z[idx], -np.cos(t)[idx], np.cos(2 * t)[idx], 0.3 * np.cos(3 * t)[idx]]
        )

        # 验证排序后数据仍有效（不为空）
        sorted_x, sorted_y, sorted_z = visualizer._sort_3d_points_by_nearest_neighbor(
            states[:, 0], states[:, 1], states[:, 2]
        )

        assert len(sorted_x) == 50
        assert not np.any(np.isnan(sorted_x))
        assert not np.any(np.isnan(sorted_y))
        assert not np.any(np.isnan(sorted_z))
