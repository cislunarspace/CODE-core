集成方式：import / MCP / CLI / sidecar
======================================

目标
----

同一套能力有三条调用链，全部派生自同一份工具清单
``tool_inventory(Facade())``（``e2m2e/api/facade.py``），没有第二份清单文件。
本篇说明各链路的入口与适用场景。

进程内 import
-------------

Python 直接调用，最短路径，教程全部使用这种方式：

.. code-block:: python

   from e2m2e.api import Facade

   facade = Facade()
   result = facade.orbit_propagation(
       initial_state=[7000.0, 0.0, 0.0, 0.0, 7.5, 0.0],
       epoch="2027-08-02T00:00:00",
       duration=3600.0,
   )
   print(result.status, result.n_points)

三个暴露类：``Facade``（一档任务）、``facade.catalog``（轨道库）、
``facade.spatiography``（分区分析）。

MCP
---

在 MCP 客户端里注册服务器，用自然语言驱动：

.. code-block:: bash

   uv pip install "e2m2e[mcp]"
   e2m2e mcp-serve

客户端配置（``command`` 指向装了 e2m2e 的环境里的可执行文件）：

.. code-block:: json

   {
     "mcpServers": {
       "e2m2e": {
         "command": "/path/to/venv/bin/e2m2e",
         "args": ["mcp-serve"],
         "cwd": "/path/to/CODE-core"
       }
     }
   }

工具 schema 从 Request 模型生成，返回统一信封 ``{status, data, error, meta}``。
长任务（``transfer_design``、``orbit_family_generation``）自动跑 worker
子进程，取消即 kill（见 :doc:`transfer-design`）。

CLI
---

每个 ``mcp_exposed`` 工具有一个同名子命令（下划线转连字符），参数从同一份
Pydantic 模型生成，输出同一个信封 JSON（stdout）：

.. code-block:: bash

   e2m2e design-orbit --orbit-type HALO --collinear-point 2 --amplitude 30000.0
   e2m2e orbit-propagation --help
   e2m2e valid-ranges

进度与用法错误写 stderr，便于脚本管道区分。

sidecar（GUI）
--------------

Tauri 壳经 stdio JSON 行 + 二进制帧协议驱动（ADR 0035）：

.. code-block:: bash

   e2m2e serve-stdio

大数组（轨迹、场数据）走二进制帧：帧头 magic 为 ASCII ``"E2M2"`` 小端 u32
（``0x324D3245``），后接 dtype／ndim／shape，唯一实现在 ``e2m2e/api/frames.py``。

怎么选
------

=============== =========================== ===============================
方式             适用                        说明
=============== =========================== ===============================
import           Python 项目、批处理         直接拿 Response 对象
MCP              LLM Agent、交互式会话       自然语言驱动，schema 自动生成
CLI              脚本管道、cron              信封 JSON 进 stdout
sidecar          GUI 应用                    二进制帧传大数组
=============== =========================== ===============================

延伸
----

- 工具权威清单：``tool_inventory(Facade())``；
- MCP 部署细节见 README“快速开始”一节。
