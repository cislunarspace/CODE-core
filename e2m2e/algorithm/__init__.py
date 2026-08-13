"""算法层：领域知识构造问题的编排。

第 3 层，依赖方向：data/ + _integrators（Rust）。不 import api/、不 import
tools/（ADR 0012）。

Python 算法层 = 一切"需要领域知识构造问题"的编排，不做数值迭代（迭代在 Rust
solver/）：①构造问题（选轨道族、定约束、选流形方向）②调 Rust 迭代器 ③解释结果。

子模块：
- ``design/``：任务轨道设计（三段编排：family → 星历修正[Rust 打靶] → propagation）。
- ``family/``：轨道族生成（种子/初猜/族行走/注册表）。
- ``station_keeping/``：轨道保持（controller + 三控制律 + 误差模型 + 蒙特卡洛）。
- ``transfer/``：转移设计（transfer_orbit 编排器 + 数学模块）。
- ``dynamics/``：System + Dynamics。
- ``forces/``：力模型类（ForceModel/PhysicalModel 子类/推力）。
- ``propagation.py``：轨道预报薄壳。
- ``coordinate/``：坐标转换算法。
- ``manifold/``：不变流形 + 庞加莱截面。
- ``proximity/``：相对运动。
- ``stability.py``：稳定性。
- ``normal_form/``：正规化（可选依赖）。
- ``nominal_orbit/``：名义轨道。

实现状态：骨架。模块逐个实现/迁入中，未实现能力占位函数抛 ``NotImplementedError``。
"""

__all__: list[str] = []
