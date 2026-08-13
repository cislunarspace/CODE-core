"""e2m2e.algorithm.forces 模块测试包。

按数据流接缝分四个子包，每个测试文件只断言一个接缝：

- ``contract/``：定义→spec 契约（构造校验、``to_rust_spec`` 序列化、
  能力边界显式报错、文件解析）——断言纯 Python 数据；
- ``physics/``：物理规律验证（解析对照、守恒律、独立路径交叉验证、
  几何）——断言 Rust 端到端传播/单点绑定的物理性质；
- ``config/``：配置保真（``from_config`` / ``to_config`` 往返字典相等、
  摄动开关映射、config 的传播物理验收）；
- ``container/``：ForceModel 编排（注册表、启停、传播边界、STM 组装、
  机动编排）——断言容器机制。

星历缓存透明性（接缝④）归 ``tests/numerical/integrators/bindings/``。
共享基准物理（Datum.WGS84 常量、LEO 场景 fixture、开普勒转换工具）
集中在 ``conftest.py``。
"""
