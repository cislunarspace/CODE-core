"""pytest 配置与共享 fixture：CR3BP 系统、SPICE 内核 fixture 与参考历元。"""

# xdist worker 内的线程池钉死为单线程：并行度已由 -n 决定，Rust rayon 全局池
# 与 BLAS 各自再开满核会相互踩踏（16 worker × 16 线程实测把隔离时 6.5s 的
# 用例拉到 19s，ADR 0037 门禁的 wall-clock 随之失真）。OpenBLAS 与 rayon 在
# 池首用时读环境，必须赶在下方 e2m2e/numpy 导入之前；setdefault 保留显式
# 覆盖入口（此时门禁读数反映真实线程争用）。
import os

for _var in (
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

# ADR 0037 时间预算门禁（tests/time_budget.py）：钩子在该模块中定义，此处
# 再导出注册。不能用 pytest_plugins 声明——本文件不是 rootdir 层 conftest。
import pytest  # noqa: E402
from kernel_helpers import SPICE_KERNEL_DIR  # noqa: E402
from time_budget import (  # noqa: F401,E402
    pytest_collection_modifyitems,
    pytest_configure,
    pytest_runtest_logreport,
    pytest_runtest_makereport,
    pytest_sessionfinish,
)

from e2m2e.algorithm.dynamics import CR3BP_Dynamics, CR3BP_System  # noqa: E402
from e2m2e.data.constants import Datum  # noqa: E402

# 单测的 catalog 目录是合成数据：关闭包内基线数据集的首用导入（ADR 0036），
# 否则查询计数类断言会被基线记录污染。基线导入自身的测试显式注入合成源。
os.environ.setdefault("E2M2E_CATALOG_BASELINE_IMPORT", "0")


@pytest.fixture
def earth_moon_system():
    """地月 CR3BP 系统（DE421 基准）。

    normal_form / design 切片以 qiao 文献参数（μ=1.215058560962404e-2、
    地月距 384405 km、周期 27.32 d）局部覆盖本 fixture，见各自 conftest。
    """
    return CR3BP_System(mu=Datum.DE421.mu, primary="Earth", secondary="Moon")._with_default_scales()


@pytest.fixture
def earth_moon_dynamics(earth_moon_system):
    """地月 CR3BP 动力学。"""
    return CR3BP_Dynamics(system=earth_moon_system)


# =============================================================================
# SPICE 内核与星历数据（内核缺失时跳过）
# =============================================================================


@pytest.fixture
def spice_kernel_path():
    """返回 DE440 内核文件路径，不存在则跳过。"""
    kernel_file = os.path.join(SPICE_KERNEL_DIR, "de440.bsp")
    if not os.path.exists(kernel_file):
        kernel_file = os.path.join(SPICE_KERNEL_DIR, "de440s.bsp")
    if not os.path.exists(kernel_file):
        kernel_file = os.path.join(SPICE_KERNEL_DIR, "de438.bsp")
    if not os.path.exists(kernel_file):
        kernel_file = os.path.join(SPICE_KERNEL_DIR, "de435.bsp")
    if not os.path.exists(kernel_file):
        pytest.skip("DE440/DE438/DE435 SPICE kernel not found, set SPICE_KERNEL_DIR")
    return kernel_file


@pytest.fixture
def reference_epoch():
    """参考历元：2025-06-21 11:00:06 UTC（J2000 后的 ET 秒数）。"""
    return "2025-06-21T11:00:06"


# =============================================================================
# SPICE 星历 fixture
#
# 统一经 spice_manager 的 yield-teardown 加载/卸载内核，保证用完即卸，
# 避免 unload_kernel 未被调用导致的内核泄漏。
# =============================================================================


@pytest.fixture
def spice_manager(spice_kernel_path):
    """加载 DE440/DE438/DE435 内核的 SPICEManager，测试结束后自动卸载。"""
    from e2m2e.data.kernels.manager import SPICEManager

    mgr = SPICEManager()
    mgr.load_kernel(spice_kernel_path)
    yield mgr
    mgr.unload_kernel(spice_kernel_path)


@pytest.fixture
def spice_eph_system(spice_manager):
    """J2000 下的地月日星历系统，原点在地球。"""
    from e2m2e.algorithm.dynamics.ephemeris_system import EphemerisSystem
    from e2m2e.data.templates.enums import ReferenceFrame

    return EphemerisSystem(
        bodies=["EARTH", "MOON", "SUN"],
        spice=spice_manager,
        origin="EARTH",
        frame=ReferenceFrame.J2000,
    )


@pytest.fixture
def spice_eph_dynamics(spice_eph_system):
    """放宽 rtol/atol/max_step 的星历 N 体动力学，加速测试。"""
    from e2m2e.algorithm.dynamics.ephemeris_dynamics import EphemerisDynamics

    d = EphemerisDynamics(system=spice_eph_system)
    # 这些宽松参数让测试中的星历传播比生产快 ~10×
    # 生产代码（_run_propagate 等）使用更严格的 1e-12。
    d.rtol = 1e-10
    d.atol = 1e-10
    d.max_step = 600.0
    return d


@pytest.fixture
def spice_syn_j2000(earth_moon_system, spice_manager):
    """同步系 ↔ J2000 坐标转换器，接入标准 CR3BP 系统。

    基于 ``CoordinateSystem`` 的 ``SynodicJ2000System`` 实现，接口包括
    ``synodic_to_j2000``、``j2000_to_synodic``、``batch_synodic_to_j2000``、
    ``batch_j2000_to_synodic``，以及 ``cr3bp_system``、``spice`` 属性。
    """
    from e2m2e.algorithm.coordinate import SynodicJ2000System

    return SynodicJ2000System(
        cr3bp_system=earth_moon_system,
        spice=spice_manager,
    )
