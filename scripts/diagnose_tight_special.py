"""诊断脚本：确认 TIGHT/SPECIAL 控制律修复后的量级对齐（Issue #280 Phase 0）。

用同一标称 NRHO 轨道，跑 LOOSE / TIGHT / SPECIAL 三种模式（无测量误差、
无推力误差、num_monte_carlo=1），打印每个控制节点的 Δv 量级与总 Δv，
确认 TIGHT/SPECIAL 不再出现 3×/16× 偏差。

用法：python scripts/diagnose_tight_special.py
前置：SPICE 内核（kernels/ 目录下 de440.bsp 等）

ADR 0013 对齐：本脚本为开发期诊断工具（§4），放 scripts/，不进 CI。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保项目根目录在 sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from e2m2e.algorithm.design.design_orbit import design_orbit  # noqa: E402
from e2m2e.algorithm.station_keeping.controller import control_orbit  # noqa: E402
from e2m2e.api.models import DesignOrbitRequest  # noqa: E402

# ─── 轨道与仿真参数 ───────────────────────────────────────────────────────────
EPOCH = [2024, 1, 1, 0, 0, 0.0]
ORBIT_DURATION_SEC = 0.05 * 365.25 * 86400  # ~18 天，足够 5 个控制节点（间隔 3 天）
CONTROL_INTERVAL_DAYS = 3.0  # 短间隔，快速验证
FEEDBACK_ARC_DAYS = 3.0  # 与 control_interval 对齐
NUM_CONTROLS = 5
NUM_MC = 1
OUTPUT_STEP = 3600.0

# 误差参数：导航误差用 DFH 默认值（TIGHT/SPECIAL 需要真实偏差才能校正），
# 推力误差归零（隔离控制律行为），推力阈值用 DFH 默认值
NAV_POS_SIGMA_M = 1500.0  # DFH 默认（#280：原值 0.0 导致 TIGHT Δv≈0）
NAV_VEL_SIGMA_MPS = 0.002  # DFH 默认
THRUST_MIN = 0.1  # DFH 默认（#280：原值 0.001 过小）
THRUST_MAX = 9999.0
THRUST_MEAN = 10.0  # DFH 默认（#280：原值 0.001 过小）
THRUST_ABS_ERR = 0.0
THRUST_REL_ERR = 0.0
THRUST_ANGLE_ERR = 0.0
SRP_ERROR = 0.10  # DFH 默认（#280：原值 0.0）


def _run_mode(name: str, control_mode: int, **kwargs) -> float:
    """跑一个控制模式，返回总 Δv（m/s）。"""
    result = control_orbit(
        NOMINAL_EPH,
        control_mode=control_mode,
        is_nrho=1,
        special_mode=2,  # NRHO → Halo 类型（ẋ=0 且 ż=0）
        control_interval=CONTROL_INTERVAL_DAYS,
        feedback_arc=FEEDBACK_ARC_DAYS,
        special_crossings=3,
        num_controls=NUM_CONTROLS,
        num_monte_carlo=NUM_MC,
        output_step=OUTPUT_STEP,
        position_accuracy=NAV_POS_SIGMA_M,
        velocity_accuracy=NAV_VEL_SIGMA_MPS,
        thrust_min=THRUST_MIN,
        thrust_max=THRUST_MAX,
        thrust_mean=THRUST_MEAN,
        thrust_abs_err=THRUST_ABS_ERR,
        thrust_rel_err=THRUST_REL_ERR,
        thrust_angle_err=THRUST_ANGLE_ERR,
        srp_error_level=SRP_ERROR,
        seed=42,
        **kwargs,
    )
    sk = result.sk_statistic
    print(f"\n{'=' * 60}")
    print(f"  {name}  (control_mode={control_mode})")
    print(f"{'=' * 60}")
    print(f"  控制节点数: {len(sk.rows)}")
    print(f"  失败样本数: {result.num_failed}")
    # SK_STATISTIC 每行：MJD  Δv_orb(m/s)  Δv_cum(m/s)
    if sk.rows.shape[1] >= 3:
        total_dv = float(sk.rows[-1, 2])  # 最后一行累计值
        print(f"  各节点 Δv (m/s): {[f'{v:.4f}' for v in sk.rows[:, 1]]}")
        print(f"  总 Δv (m/s): {total_dv:.4f}")
    else:
        total_dv = float(np.sum(sk.rows[:, 1]))
        print(f"  总 Δv (m/s): {total_dv:.4f}")
    return total_dv


if __name__ == "__main__":
    print("=" * 60)
    print("  Issue #280 诊断：TIGHT/SPECIAL 量级对齐")
    print("=" * 60)

    # Step 1: 生成标称 NRHO 轨道
    print("\n[1/4] 生成标称 NRHO 轨道 ...")
    orbit_result = design_orbit(
        DesignOrbitRequest(
            orbit_type="NRHO",
            collinear_point=2,
            north_south=1,
            perilune_height=3500.0,
            epoch=EPOCH,
            duration=ORBIT_DURATION_SEC,
            output_step=OUTPUT_STEP,
        )
    )
    NOMINAL_EPH = orbit_result.ephemeris
    print(f"  标称星历行数: {len(NOMINAL_EPH)}")
    pos_norm = np.linalg.norm(NOMINAL_EPH.position_km[0])
    print(f"  首行位置量级: {pos_norm:.1f} km")

    # Step 2-4: 三种模式
    dv_loose = _run_mode("LOOSE (模式 1)", 1)
    dv_tight = _run_mode("TIGHT (模式 2)", 2, tight_tolerance_km=0.1, tight_max_iter=6)
    dv_special = _run_mode(
        "SPECIAL (模式 3)",
        3,
        special_damping_factor=0.5,  # 启用阻尼防振荡
    )

    # 汇总
    print(f"\n{'=' * 60}")
    print("  汇总")
    print(f"{'=' * 60}")
    print(f"  LOOSE  总 Δv: {dv_loose:.4f} m/s")
    print(f"  TIGHT  总 Δv: {dv_tight:.4f} m/s")
    print(f"  SPECIAL 总 Δv: {dv_special:.4f} m/s")

    # 合理性检查
    issues = []
    if dv_tight < dv_loose * 0.1:
        issues.append(f"TIGHT 偏低 {dv_loose / max(dv_tight, 1e-12):.1f}×（预期与 LOOSE 同阶）")
    if dv_special > dv_loose * 10:
        issues.append(f"SPECIAL 偏高 {dv_special / max(dv_loose, 1e-12):.1f}×（预期与 LOOSE 同阶）")
    if issues:
        print("\n  ⚠️  发现量级异常：")
        for issue in issues:
            print(f"    - {issue}")
        sys.exit(1)
    else:
        print("\n  ✅ 三种模式量级合理，无显著偏差")
        sys.exit(0)
