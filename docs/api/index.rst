API 参考
========

API 参考由 `sphinx-autoapi <https://sphinx-autoapi.readthedocs.io/>`_ 从源码
docstring **静态生成** ——构建时解析而不导入模块——覆盖 ``e2m2e`` 全部公开
Python 模块。

阅读顺序建议：

- **接口层** ：多数调用方只需要这一层：``e2m2e.api``——``Facade``
  （一档任务）、``Catalog``（轨道库）、``Spatiography``（分区分析）三个
  暴露类；``e2m2e.api.models``——每工具成对的 ``*Request``／``*Response``
  Pydantic 模型（字段描述的权威来源）；``e2m2e.api.config``——运行配置。
- **数值门面**：``e2m2e.integrators``——Rust 积分器扩展的公共适配层。
- **契约**：``e2m2e.status``——状态三元组与原因码；``e2m2e.exceptions``——
  异常层次。
- **编排层**：``e2m2e.algorithm``——动力学、轨道族、转移、站保等编排器
  （教程与示例用到时才需要）。
- **数据层**：``e2m2e.data``——常数、内核、坐标系、轨道库存储。

对 MCP / CLI 而言，**工具面的权威清单** 是
``tool_inventory(Facade())``，本站页面与它同源派生；以它为准。

.. toctree::
   :maxdepth: 2

   /autoapi/e2m2e/index
