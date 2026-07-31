"""力模型类：ForceModel/PhysicalModel 子类/推力。

Python 类是"力模型定义"（参数验证 + to_rust_spec 序列化 + 无 Rust 时的兜底
计算），与 Rust ``e2m2e-forces`` 的 CompiledForce 枚举对应（ADR 0011）。配置
schema 在 ``data/templates/force_config.py``（纯数据）。

实现状态：骨架。ForceModel/PointMassGravity/ThirdBodyGravity/GravityField/SRP/
DragModel/RelativisticCorrection/FiniteBurn/ImpulsiveBurn 待从 ``core/forces/``
迁入。

未实现（对外承诺能力）：ECOM 光压（原 #253），占位抛 ``NotImplementedError``。
"""

from __future__ import annotations

__all__: list[str] = []


def ecom_solar_radiation_pressure(*args, **kwargs):
    """ECOM 光压模型（原 #253）。

    实现状态：未实现（对外承诺能力，占位）。DFH 有炮弹/ECOM 两档，现有仅炮弹
    模型。ECOM 9 系数（首系数 = 等效面质比）。

    Raises:
        NotImplementedError: ECOM 光压未实现。
    """
    raise NotImplementedError("ECOM 光压模型未实现（原 #253）：9 系数模型待补")
