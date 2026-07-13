"""NRHO 轨道 CR3BP → 星历模型修正测试（Layer 3）。

覆盖 NRHO 加载、patch points 采样、synodic→J2000 转换、
Multiple Shooting 修正与结果验证。

NRHO（Near Rectilinear Halo Orbit）是三维轨道（z ≠ 0），
比 DRO 更难处理：对初始参数敏感、容易发散。

当前状态：标准多重打靶对 NRHO 不收敛（残差从 ~8000 km 降到 ~80 km 后停滞），
验证了 issue #212 中描述的敏感性问题。修正相关的测试标记为 xfail，
待实现同伦过渡或滚动时域方法后再启用。
"""

import os
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pytest
from numpy.testing import assert_allclose

from e2m2e.algorithms import (  # noqa: E402
    MultipleShooting,
)
from e2m2e.core import Orbit

pytestmark = pytest.mark.spice

# =============================================================================
# 物理参数
# =============================================================================
MU = 1.21506683e-2
TU_SECONDS = 4.34811305 * 86400  # 秒

N_PATCH_POINTS = 8
POSITION_CONTINUITY_TOL = 1e-6  # km

# xlsx 文件路径
XLSX_PATH = os.environ.get("NRHO_REFERENCE_DATA")
if not XLSX_PATH:
    pytest.skip(
        "未设置 NRHO_REFERENCE_DATA 环境变量（需指向 earth-moon_halo_L2_S.xlsx），跳过本文件",
        allow_module_level=True,
    )


# =============================================================================
# 辅助函数
# =============================================================================
def load_nrho_from_xlsx(xlsx_path: str, index: int = 0) -> dict:
    """从 xlsx 文件加载 NRHO 轨道数据。

    Parameters
    ----------
    xlsx_path : str
        xlsx 文件路径
    index : int
        轨道索引（从 0 开始）

    Returns
    -------
    dict
        包含 x0, period, jacobi 等字段
    """
    with zipfile.ZipFile(xlsx_path) as z:
        shared_strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            tree = ET.parse(z.open("xl/sharedStrings.xml"))
            root = tree.getroot()
            ns = {"ns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            for si in root.findall("ns:si", ns):
                t = si.find("ns:t", ns)
                if t is not None:
                    shared_strings.append(t.text)

        tree = ET.parse(z.open("xl/worksheets/sheet1.xml"))
        root = tree.getroot()
        ns = {"ns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

        rows = root.findall(".//ns:row", ns)
        # 跳过表头，取指定行
        row = rows[index + 1]
        cells = []
        for cell in row.findall("ns:c", ns):
            val = cell.find("ns:v", ns)
            if val is not None:
                v = val.text
                if cell.get("t") == "s" and v:
                    v = shared_strings[int(v)]
                cells.append(float(v))

        return {
            "x0": cells[0],
            "y0": cells[1],
            "z0": cells[2],
            "vx0": cells[3],
            "vy0": cells[4],
            "vz0": cells[5],
            "jacobi": cells[6],
            "period": cells[7],
            "stability": cells[8],
        }


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture(params=[0, 100, 500], ids=["small", "medium", "large"])
def nrho_orbit(cr3bp_dynamics, cr3bp_system, request):
    """
    从 xlsx 加载 NRHO 轨道。

    使用不同振幅的轨道进行测试：
    - index=0: 小振幅（z ≈ 0.0001）
    - index=100: 中振幅（z ≈ 0.013）
    - index=500: 大振幅（z ≈ 0.067）
    """
    index = request.param
    data = load_nrho_from_xlsx(XLSX_PATH, index)

    seed_state = np.array([data["x0"], data["y0"], data["z0"],
                           data["vx0"], data["vy0"], data["vz0"]])
    orbit = Orbit([seed_state], [0])
    orbit.period = data["period"]
    return orbit


# =============================================================================
# Test Step 1: 加载 NRHO 轨道
# =============================================================================
class TestStep1LoadNRHOOrbit:
    """测试从 xlsx 加载 NRHO 轨道"""

    def test_nrho_orbit_exists(self, nrho_orbit):
        """NRHO 轨道应成功加载"""
        assert nrho_orbit is not None

    def test_nrho_period_positive(self, nrho_orbit):
        """NRHO 周期应为正数"""
        assert nrho_orbit.period > 0

    def test_nrho_is_3d(self, nrho_orbit):
        """NRHO 应为三维轨道 (z ≠ 0)"""
        state0 = nrho_orbit.states[0]
        # z 分量不为零（允许小振幅接近零）
        assert abs(state0[2]) > 1e-10 or abs(state0[5]) > 1e-10, "NRHO 应有 z 或 vz 分量"

    def test_nrho_initial_state_finite(self, nrho_orbit):
        """NRHO 初始状态应为有限值"""
        state0 = nrho_orbit.states[0]
        assert np.all(np.isfinite(state0))


# =============================================================================
# Test Step 2: 采样 patch points
# =============================================================================
class TestStep2SamplePatchPoints:
    """测试从 NRHO 轨道采样 patch points"""

    def test_sample_uniform_time(self, nrho_orbit):
        """应在 NRHO 周期内均匀采样"""
        period = nrho_orbit.period
        n_points = N_PATCH_POINTS
        t_patch = np.linspace(0, period, n_points, endpoint=False)

        assert len(t_patch) == n_points
        assert_allclose(t_patch[0], 0.0)
        assert t_patch[-1] < period

    def test_sample_states_at_patch_times(self, nrho_orbit, cr3bp_dynamics):
        """应在采样时间点获取 NRHO 状态"""
        period = nrho_orbit.period
        n_points = N_PATCH_POINTS
        t_patch = np.linspace(0, period, n_points, endpoint=False)

        state_patch = np.zeros((n_points, 6))
        state_patch[0] = nrho_orbit.states[0]

        for i in range(1, n_points):
            state_patch[i] = cr3bp_dynamics.propagate_orbit_state_at_time(nrho_orbit, t_patch[i])

        assert state_patch.shape == (n_points, 6)
        assert np.all(np.isfinite(state_patch))

    def test_sampled_states_are_3d(self, nrho_orbit, cr3bp_dynamics):
        """采样点应保持三维特性"""
        period = nrho_orbit.period
        t_patch = np.linspace(0, period, N_PATCH_POINTS, endpoint=False)

        for t in t_patch[1:]:
            state = cr3bp_dynamics.propagate_orbit_state_at_time(nrho_orbit, t)
            # 至少有一个 z 或 vz 分量不为零
            assert abs(state[2]) > 1e-10 or abs(state[5]) > 1e-10, \
                "采样点 z 和 vz 均为零，不是三维轨道"


# =============================================================================
# Test Step 3: synodic → J2000 坐标转换
# =============================================================================
class TestStep3SynodicToJ2000:
    """测试将 NRHO patch points 从 synodic 转换到 J2000"""

    def test_convert_patch_points(self, nrho_orbit, cr3bp_dynamics, spice_syn_j2000, reference_et):
        """应能将所有 patch points 转换到 J2000"""
        period = nrho_orbit.period
        t_patch = np.linspace(0, period, N_PATCH_POINTS, endpoint=False)

        state_patch_syn = np.zeros((N_PATCH_POINTS, 6))
        state_patch_syn[0] = nrho_orbit.states[0]
        for i in range(1, N_PATCH_POINTS):
            state_patch_syn[i] = cr3bp_dynamics.propagate_orbit_state_at_time(
                nrho_orbit, t_patch[i]
            )

        state_patch_j2000 = spice_syn_j2000.batch_synodic_to_j2000(
            states_syn=state_patch_syn,
            t_syn_arr=t_patch,
            et0=reference_et,
        )

        assert state_patch_j2000.shape == (N_PATCH_POINTS, 6)
        assert np.all(np.isfinite(state_patch_j2000))

    def test_j2000_positions_near_moon(
        self, nrho_orbit, cr3bp_dynamics, spice_syn_j2000, reference_et
    ):
        """J2000 下的 NRHO 位置应在月球距离附近"""
        state0_j2000 = spice_syn_j2000.synodic_to_j2000(
            state_syn=nrho_orbit.states[0],
            t_syn=0.0,
            et0=reference_et,
        )
        r = np.linalg.norm(state0_j2000[:3])
        # NRHO 在地月系统内，距离应在合理范围
        assert 300000 < r < 500000, f"NRHO距地球 {r:.0f} km，超出合理范围"

    def test_j2000_time_is_et(self, spice_syn_j2000, reference_et, nrho_orbit):
        """J2000 时间应为 ET 秒"""
        tc = TU_SECONDS
        period = nrho_orbit.period
        t_patch_syn = np.linspace(0, period, N_PATCH_POINTS, endpoint=False)
        t_patch_j2000 = reference_et + t_patch_syn * tc

        assert_allclose(t_patch_j2000[0], reference_et)
        assert t_patch_j2000[-1] - t_patch_j2000[0] > 0


# =============================================================================
# Test Step 4: Multiple Shooting 修正（xfail：标准方法不收敛）
# =============================================================================
class TestStep4MultipleShootingCorrection:
    """测试在星历模型下进行 Multiple Shooting 修正。

    当前状态：标准多重打靶对 NRHO 不收敛，残差从 ~8000 km 降到 ~80 km 后停滞。
    这验证了 issue #212 中描述的 NRHO 敏感性问题。
    待实现同伦过渡或滚动时域方法后再启用这些测试。
    """

    @pytest.mark.xfail(
        reason="标准多重打靶对 NRHO 不收敛（issue #212），待实现同伦过渡方法",
        strict=False,
    )
    def test_correction_converges(
        self,
        nrho_orbit,
        cr3bp_dynamics,
        spice_syn_j2000,
        spice_eph_dynamics,
        reference_et,
    ):
        """Multiple Shooting 修正应收敛（当前不收敛，标记为 xfail）"""
        period = nrho_orbit.period
        tc = TU_SECONDS

        t_patch_syn = np.linspace(0, period, N_PATCH_POINTS, endpoint=False)
        state_patch_syn = np.zeros((N_PATCH_POINTS, 6))
        state_patch_syn[0] = nrho_orbit.states[0]
        for i in range(1, N_PATCH_POINTS):
            state_patch_syn[i] = cr3bp_dynamics.propagate_orbit_state_at_time(
                nrho_orbit, t_patch_syn[i]
            )

        state_patch_j2000 = spice_syn_j2000.batch_synodic_to_j2000(
            states_syn=state_patch_syn,
            t_syn_arr=t_patch_syn,
            et0=reference_et,
        )
        t_patch_j2000 = reference_et + t_patch_syn * tc

        ms = MultipleShooting(dynamics=spice_eph_dynamics)
        result = ms.correct(
            t_patch=t_patch_j2000,
            state_patch=state_patch_j2000,
            var_time=True,
            max_iter=50,
            tolerance=POSITION_CONTINUITY_TOL,
        )

        # 记录诊断信息
        print("\nNRHO 修正诊断:")
        print(f"  振幅索引: {nrho_orbit.period:.4f}")
        print(f"  收敛状态: {result.converged}")
        print(f"  迭代次数: {result.outer_iterations}")
        print(f"  最大残差: {result.max_residual:.2e} km")
        print(f"  残差历史: {[f'{r:.2e}' for r in result.residual_history[:5]]}...")

        assert result.converged, f"Multiple Shooting 未收敛，迭代 {result.outer_iterations} 次"


# =============================================================================
# Test Step 5: 验证修正结果（xfail：依赖 Step 4）
# =============================================================================
class TestStep5Validation:
    """测试修正后的轨道质量（当前不收敛，标记为 xfail）"""

    @pytest.fixture
    def correction_result(
        self,
        nrho_orbit,
        cr3bp_dynamics,
        spice_syn_j2000,
        spice_eph_dynamics,
        reference_et,
    ):
        """运行修正并返回结果"""
        period = nrho_orbit.period
        tc = TU_SECONDS

        t_patch_syn = np.linspace(0, period, N_PATCH_POINTS, endpoint=False)
        state_patch_syn = np.zeros((N_PATCH_POINTS, 6))
        state_patch_syn[0] = nrho_orbit.states[0]
        for i in range(1, N_PATCH_POINTS):
            state_patch_syn[i] = cr3bp_dynamics.propagate_orbit_state_at_time(
                nrho_orbit, t_patch_syn[i]
            )

        state_patch_j2000 = spice_syn_j2000.batch_synodic_to_j2000(
            states_syn=state_patch_syn,
            t_syn_arr=t_patch_syn,
            et0=reference_et,
        )
        t_patch_j2000 = reference_et + t_patch_syn * tc

        ms = MultipleShooting(dynamics=spice_eph_dynamics)
        result = ms.correct(
            t_patch=t_patch_j2000,
            state_patch=state_patch_j2000,
            var_time=True,
            max_iter=50,
            tolerance=POSITION_CONTINUITY_TOL,
        )
        return result

    @pytest.mark.xfail(
        reason="标准多重打靶对 NRHO 不收敛（issue #212），待实现同伦过渡方法",
        strict=False,
    )
    def test_position_continuity(self, correction_result):
        """修正后相邻段端点位置连续性误差应 < 1e-6 km"""
        result = correction_result
        assert result.converged, "修正应收敛"
        assert result.max_residual < POSITION_CONTINUITY_TOL, (
            f"最大残差 {result.max_residual:.2e} km > {POSITION_CONTINUITY_TOL}"
        )

    @pytest.mark.xfail(
        reason="标准多重打靶对 NRHO 不收敛（issue #212），待实现同伦过渡方法",
        strict=False,
    )
    def test_orbit_shape_preserved(self, correction_result):
        """修正后轨道形状应与 CR3BP NRHO 相似"""
        result = correction_result
        assert result.converged, f"修正未收敛 (residual={result.max_residual:.2e})"

        corrected_states = result.state_patch
        distances = np.linalg.norm(corrected_states[:, :3], axis=1)
        mean_dist = np.mean(distances)
        # NRHO 在地月系统内，距离应在合理范围
        assert 300000 < mean_dist < 500000, f"修正后平均距地球 {mean_dist:.0f} km，偏离 NRHO 范围"
        std_dist = np.std(distances)
        assert std_dist / mean_dist < 0.2, (
            f"修正后轨道形状变化过大: std/mean = {std_dist / mean_dist:.3f}"
        )


# =============================================================================
# Test 完整流程 (Integration)
# =============================================================================
class TestNRHOEphemerisPipeline:
    """测试完整的 NRHO CR3BP → 星历模型修正流程。

    当前状态：标准多重打靶对 NRHO 不收敛，标记为 xfail。
    待实现同伦过渡或滚动时域方法后再启用。
    """

    @pytest.fixture
    def correction_result(
        self,
        nrho_orbit,
        cr3bp_dynamics,
        spice_syn_j2000,
        spice_eph_dynamics,
        reference_et,
    ):
        """运行修正并返回结果"""
        period = nrho_orbit.period
        tc = TU_SECONDS

        t_patch_syn = np.linspace(0, period, N_PATCH_POINTS, endpoint=False)
        state_patch_syn = np.zeros((N_PATCH_POINTS, 6))
        state_patch_syn[0] = nrho_orbit.states[0]
        for i in range(1, N_PATCH_POINTS):
            state_patch_syn[i] = cr3bp_dynamics.propagate_orbit_state_at_time(
                nrho_orbit, t_patch_syn[i]
            )

        state_patch_j2000 = spice_syn_j2000.batch_synodic_to_j2000(
            states_syn=state_patch_syn,
            t_syn_arr=t_patch_syn,
            et0=reference_et,
        )
        t_patch_j2000 = reference_et + t_patch_syn * tc

        ms = MultipleShooting(dynamics=spice_eph_dynamics)
        result = ms.correct(
            t_patch=t_patch_j2000,
            state_patch=state_patch_j2000,
            var_time=True,
            max_iter=50,
            tolerance=POSITION_CONTINUITY_TOL,
        )
        return result

    @pytest.mark.xfail(
        reason="标准多重打靶对 NRHO 不收敛（issue #212），待实现同伦过渡方法",
        strict=False,
    )
    def test_full_pipeline(self, correction_result):
        """完整流程: NRHO加载 → 采样 → 坐标转换 → 星历修正 → 验证"""
        result = correction_result

        # 记录诊断信息
        print("\nNRHO 完整流程诊断:")
        print(f"  收敛状态: {result.converged}")
        print(f"  迭代次数: {result.outer_iterations}")
        print(f"  最大残差: {result.max_residual:.2e} km")

        assert result.converged, f"修正未收敛，迭代 {result.outer_iterations} 次"
        assert result.max_residual < POSITION_CONTINUITY_TOL, (
            f"最大位置连续性误差 {result.max_residual:.2e} km > {POSITION_CONTINUITY_TOL} km"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
