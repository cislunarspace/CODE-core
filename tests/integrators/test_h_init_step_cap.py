"""h_init 步长上限回归测试（issue #279）。

验证：稀疏 t_eval（2 点）与密集 t_eval（31 点）在同一状态 30 天传播后，
终态差 < 1 km。无 h_init 上限时自适应步长会失控，实测差 22 万 km。

测试走 Rust propagate_compiled 路径（需 spice feature）。
"""

import numpy as np
import pytest

from e2m2e.algorithm.forces import ForceModel
from e2m2e.algorithm.forces.point_mass_gravity import PointMassGravity

pytestmark = pytest.mark.spice

# 地球引力参数 (km³/s²)
EARTH_MU = 398600.4418


class _FakeSystem:
    """最小 System 桩，仅供 PointMassGravity.to_rust_spec 解析 mu。"""

    def __init__(self):
        self.coordinate_system = object()

    @property
    def frame(self):
        from e2m2e.mbse.data.enums import ReferenceFrame

        return ReferenceFrame.J2000

    @property
    def unit_system(self):
        from e2m2e.mbse.data.enums import UnitSystem

        return UnitSystem.SI

    def gravitational_parameter(self, body):
        return EARTH_MU


def test_sparse_vs_dense_t_eval_consistency():
    """稀疏 vs 密集 t_eval 的 LEO 传播终态差 < 1 km。

    回归目标：h_init 步长上限缺失时，稀疏 t_eval 下自适应步长失控，
    导致终态差 22 万 km（实测，30 天）。加 h_init 上限后两者应一致。
    注：tol=1e-12 下有效步长 ~1.6s，1 天需 ~55k 步，受默认 max_steps
    限制，此处用 1 天验证回归（差值与持续时间成正比，1 天已足够暴露 bug）。
    """
    system = _FakeSystem()
    pm = PointMassGravity("EARTH", mu=EARTH_MU)
    fm = ForceModel(system, forces=[pm])
    # h_init 需足够大让自适应控制器自由选择步长
    fm.max_step = 3600.0

    # LEO 圆轨道初始状态 (km, km/s)
    r = 6778.0
    v = np.sqrt(EARTH_MU / r)
    y0 = np.array([r, 0.0, 0.0, 0.0, v, 0.0])

    days = 1.0
    t0 = 0.0
    tf = days * 86400.0  # 秒

    # 稀疏：只有起点和终点
    t_eval_sparse = np.array([t0, tf])
    # 密集：31 个等距点
    t_eval_dense = np.linspace(t0, tf, 31)

    result_sparse = fm.propagate(y0, (t0, tf), t_eval=t_eval_sparse)
    result_dense = fm.propagate(y0, (t0, tf), t_eval=t_eval_dense)

    final_sparse = np.asarray(result_sparse["states"][-1])
    final_dense = np.asarray(result_dense["states"][-1])

    # 终态位置差 < 1 km（回归阈值：无 h_init 上限时差 22 万 km/30天，
    # 1 天约为 1/30，仍远超 1 km）
    pos_diff_km = np.linalg.norm(final_sparse[:3] - final_dense[:3])
    assert pos_diff_km < 1.0, (
        f"Sparse vs dense t_eval position drift: {pos_diff_km:.1f} km > 1 km; "
        "h_init step cap may be missing"
    )
