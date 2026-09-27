# Lambert 内核迁移 pykep-core 基准（issue #715 / ADR 0052）

porkchop 纯 Lambert+ΔV 热路径（`porkchop_grid_states_py`，串行
`parallel=False`）在自维护 FAT 移植内核与 pykep-core 0.1.4 适配层下的耗时
对照。构造与复现方式见 `scripts/benchmark_lambert_porkchop.py` docstring；
网格 2400×30 = 72 000 格/次，重复取中位。结论先行：**revs=1 回归超 2× 门限
（release 2.51×），迁移按 issue 终止条件 #1 否决**（决策记录见 ADR 0052）。

| 档位 | FAT（release） | pykep（release） | 比值 | FAT（debug） | pykep（debug） | 比值 |
| --- | --- | --- | --- | --- | --- | --- |
| revs=0 | 0.035 s | 0.048 s | 1.37× | 0.116 s | 0.246 s | 2.12× |
| revs=1 | 0.037 s | 0.093 s | **2.51×** | 0.097 s | 0.312 s | 3.22× |

- 环境：Windows 11（10.0.26200）｜Intel 13th Gen（Family 6 Model 170），
  2026-09-27。release 档以 `cargo rustc --release` 产出 cdylib（等价
  `make dev-release`），debug 档以 `make dev` 等价流程产出。
- revs=1 回归为架构性：pykep 对每个可行圈数固定解零圈 + 左支 + 右支三组
  Householder（API 无右支专用入口），本仓库契约只需右支；旧实现每个 tof
  只解一次右支。release 档重复测量离散度小（pykep 0.075–0.096 s、FAT
  0.034–0.042 s），非测量噪声。
- revs=0 的 1.37×（release）来自 pykep 每操作的 `hypot` 范数与有限性校验
  层，在门限内。
- debug 档数字仅供开发档参考；性能裁决以 release 档为准（Makefile：release
  构建用于性能基准）。

## 迁移前（pykep 评估原型未合入，FAT 内核，release 档实测）

- 时间：2026-09-27T19:20（release）/ 2026-09-27T18:24（debug）
- 各次耗时（release, reps=7）：revs=0 [0.035, 0.036, 0.035, 0.035, 0.034, 0.035, 0.035]；
  revs=1 [0.037, 0.038, 0.037, 0.037, 0.037, 0.040, 0.042]

## 迁移后（pykep-core 0.1.4 适配层，release 档实测）

- 时间：2026-09-27T19:16（release）/ 2026-09-27T19:14（debug）
- 各次耗时（release, reps=7）：revs=0 [0.048, 0.048, 0.047, 0.047, 0.050, 0.062, 0.058]；
  revs=1 [0.093, 0.094, 0.095, 0.096, 0.075, 0.083, 0.091]
