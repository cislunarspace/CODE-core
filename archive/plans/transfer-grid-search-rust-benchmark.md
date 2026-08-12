# 转移网格搜索三档并行后端基准报告

> 日期：2026-08-07 · 由 `scripts/benchmark_transfer_search.py` 生成

## 配置

- 机器 CPU 核数：48（基准固定并行度 n_workers=4，三档统一）
- 每档每 backend 跑 3 次，取中位 wall-time
- 系统：CR3BP 地月（mu=1.21506683e-2），积分器 DOP853，rtol=atol=1e-9，max_step=0.05
- 轨道：出发绕 (0.9,0) r=0.08 共 80 点；目标绕 (0.7,0) r=0.12 共 60 点
- 搜索：α∈[0.9,1.1]，integration_dt=0.02；积分发散候选走惩罚仍计时
- warm-up：rust 小网格 1 次（初始化 rayon 线程池），不计入计时

## 数据

| 规模 (dep×α) | backend | 中位时间(s) | 加速比(vs processes) | 候选数 |
|---|---|---|---|---|
| 小 (2×3) | processes | 0.028 | 1.00x | 6 |
| 小 (2×3) | threads | 0.029 | 0.94x | 6 |
| 小 (2×3) | rust | 0.005 | 5.50x | 6 |
| 中 (8×10) | processes | 0.112 | 1.00x | 80 |
| 中 (8×10) | threads | 0.202 | 0.55x | 80 |
| 中 (8×10) | rust | 0.015 | 7.52x | 80 |
| 大 (16×20) | processes | 0.438 | 1.00x | 320 |
| 大 (16×20) | threads | 0.913 | 0.48x | 320 |
| 大 (16×20) | rust | 0.042 | 10.35x | 320 |

## 结论

大网格（320 评估）下 rust 相对 processes 加速约 10.35× （0.44s → 0.04s）。
差距随网格规模增大——processes 的人进程 pickle + Python per-α 循环开销按点数线性增长，rust 的 rayon par_iter 零调度开销、无 Python 循环。
