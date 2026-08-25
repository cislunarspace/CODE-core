通过 MCP 使用 e2m2e
===================

e2m2e 把任务级能力（轨道设计、站保仿真、转移设计、轨道库等）封装为
MCP (Model Context Protocol) 工具，供 LLM Agent 直接调用（ADR 0014）。
在支持 MCP 的客户端里，你不需要写代码或填参数表——用自然语言描述任务，
模型会根据每个工具的 schema（含参数单位、默认值、取值域的中文描述）
自行选工具、拼参数。

接入
----

MCP 服务器随 ``[mcp]`` extra 分发，以 stdio 传输启动::

   pip install "e2m2e[mcp]"        # 或 uv sync --extra mcp
   e2m2e mcp-serve                 # stdio JSON-RPC，不监听端口

在 MCP 客户端配置中注册（以 Claude Desktop / Cursor 的 ``mcpServers``
格式为例）::

   {
     "mcpServers": {
       "e2m2e": {
         "command": "/path/to/.venv/Scripts/e2m2e.exe",
         "args": ["mcp-serve"],
         "cwd": "/path/to/e2m2e-repo"
       }
     }
   }

ZCode 工作区配置（``<repo>/.zcode/config.json``）同构，键为嵌套的
``mcp.servers``::

   {
     "mcp": {
       "servers": {
         "e2m2e": {
           "type": "stdio",
           "command": "C:\\path\\to\\.venv\\Scripts\\e2m2e.exe",
           "args": ["mcp-serve"],
           "cwd": "C:\\path\\to\\e2m2e-repo"
         }
       }
     }
   }

.. note::

   ``cwd`` 建议钉在仓库根：SPICE 内核目录（``kernels``）与轨道库目录
   （``catalog``）的默认值是相对路径。也可用环境变量
   ``SPICE_KERNEL_DIR`` / ``E2M2E_CATALOG_DIR`` 显式指定绝对路径。

统一信封
--------

所有工具的返回都是统一信封 ``{status, data, error, meta}``：成功时
``status="ok"``、``data`` 为该工具的响应模型；失败时 ``status="error"``、
``error`` 含结构化错误码与信息（如 ``INVALID_PARAMS``、
``RECORD_NOT_FOUND``、``PROPAGATION_FAILED``），不泄漏 traceback。
数值计算类工具的 ``data.status`` 另行报告算法收敛状态
（converged / diverged / stagnated / max_iterations / …，ADR 0020 软失败
语义：发散也是有效结果）。

工具清单
--------

清单由 Facade 方法元数据纯派生，``placeholder`` 状态的不注册
（实现后自动上线）。当前 13 个：

.. list-table::
   :header-rows: 1
   :widths: 28 14 58

   * - 工具
     - 分档
     - 用途与关键输入
   * - ``design_orbit``
     - 一档
     - 任务轨道设计。``orbit_type`` 取 DRO/DPO/HALO/NRHO/LISSAJOUS/L4/L5/
       AXIAL/SPO/LPO/HORSESHOE/ELFO 之一，另给各族形状参数（多数有默认值），
       产出星历修正后的标称轨道并自动入库。
   * - ``control_orbit``
     - 一档
     - 轨道保持（站保）蒙特卡洛仿真。输入二选一：``input_record_id``
       引用库中记录，或 ``input_ephemeris`` 直接给星历；控制模式
       1–6、测定轨/推力误差、力模型阶数等。
   * - ``transfer_design``
     - 一档
     - 转移轨道设计。``transfer_type`` 取 HMN/LGA/WSB/low_thrust，
       ``tli_epoch`` 为 TLI 历元；LGA/WSB 需 ``target_ephemeris``，
       坐标系契约见下文注意事项。
   * - ``orbit_propagation``
     - 一档
     - 轨道预报。GCRS 六维初态 + 历元 + 时长（秒）。
   * - ``spacetime_transform``
     - 一档
     - 时空坐标转换：synodic↔J2000、GCRS↔EBCRS。会合系转换的
       ``times`` 是无量纲会合时间 t_syn（0 = ``et0_jd`` 参考历元）；
       GCRS↔EBCRS 用 JD_TDB 且需 ``ephemeris_path``。
   * - ``orbit_family_generation``
     - 二档
     - 轨道族延拓生成（HALO/NRHO/AXIAL/LISSAJOUS/SPO/LPO/HORSESHOE/
       DRO 八族），按族有振幅/延拓方向参数，族记录自动入库。
   * - ``catalog_query``
     - 库
     - 多维过滤查询（族、平动点、Jacobi 区间、振幅区间、标签、收敛
       状态），返回摘要列表。
   * - ``catalog_get``
     - 库
     - 按 ``record_id`` 取完整记录（含 CR3BP 段与星历段数组）。
   * - ``catalog_tag``
     - 库
     - 写教学标注：``tags`` 整体替换，``note`` 为自由文本。
   * - ``catalog_promote``
     - 库
     - 把族成员（``member_index``）提升为独立记录，
       ``source_record_id`` 指向所属族。
   * - ``catalog_export``
     - 库
     - 按查询条件打包导出：``dest`` 以 ``.zip`` 结尾产出 zip 包，
       否则产出目录；包可直接作为库打开。
   * - ``catalog_sweep``
     - 库
     - 参数空间扫描批量生成并入库：族 × 平动点 × 振幅网格 / 能量
       窗口 / LISSAJOUS 二维振幅网格。
   * - ``catalog_delete``
     - 库
     - 按 ``record_id`` 删除记录（文件与索引条目），不可撤销。

每个产物型工具成功后都会自动写入轨道库（ADR 0031），响应里带
``record_id``——这是跨工具链式调用的句柄。

典型工作流
----------

设计 → 站保 → 归档::

   design_orbit(orbit_type="NRHO", north_south=2, perilune_height=3000)
     └─ 得 record_id
   control_orbit(input_record_id=<上一步>, control_mode=1)
     └─ 站保产物自动指向该记录，谱系不断
   catalog_tag(record_id=…, tags=["教学示例"], note="…")
   catalog_export(orbit_family="nrho", dest="nrho.zip")

族 → 挑成员 → 站保::

   orbit_family_generation(orbit_type="HALO", libration_point=2, n_orbits=10)
   catalog_query(orbit_family="halo")          # 浏览族记录
   catalog_promote(record_id=…, member_index=3)  # 挑成员独立成记录
   control_orbit(input_record_id=<提升产物>)

批量填充轨道库::

   catalog_sweep(orbit_types=["HALO","NRHO"], max_amplitudes_km=[…],
                 jacobi_windows=[[3.0,3.2]])

自然语言示例（在 MCP 客户端里直接说）::

   设计一条 L2 南族 NRHO，近月点高度 3000 km，起始历元 2026-01-01，
   然后对它做 100 次蒙特卡洛的站保仿真，最后打上"候选"标签。

注意事项
--------

1. **``transfer_design`` 的 ``target_ephemeris`` 坐标系契约**：LGA/WSB
   要求会合旋转系（synodic）物理单位（km, km/s）；``design_orbit`` /
   ``orbit_propagation`` 产出的惯性系星历必须先经
   ``spacetime_transform(j2000_to_synodic)`` 转换再传入，否则目标态
   几何全错。HMN / low_thrust 按地心惯性系解释。
2. **单位**：时长统一秒（``control_orbit`` 的间隔类参数是天）；角度用
   度；历元可用 UTC ISO 字符串或 ``[年,月,日,时,分,秒]``；
   ``spacetime_transform`` 的 JD_TDB 与无量纲 t_syn 按上文区分。
3. **``catalog_delete`` 不可撤销**，删除前先用 ``catalog_get`` 确认。
4. ``orbit_stability``（稳定性分析）需要带 period/system 绑定的
   ``Orbit`` 对象入参，暂无法经 JSON 信封表达，未注册为 MCP 工具；
   记录引用式入参落地后再开放。需要稳定性分析时用算法层 API
   （:doc:`../algorithms/stability`）。

下一步
------

- :doc:`installation`：安装与 ``[mcp]`` extra
- :doc:`../api/e2m2e`：Facade / 模型 / MCP 模块的 API 参考
- ``docs/adr/0014-api-facade-mcp-cli.md``：接口层设计决策（ADR，不入 Sphinx 构建）
