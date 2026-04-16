---
format: md
title: Halo 轨道功能 — 后续开发路线图
---

# Halo 轨道功能 — 后续开发路线图

本文档在 [质量改进总计划](test-coverage-project-plan.md) 之外，单独跟踪 **Halo 单轨 / 轨道族 / PAL 延拓** 的演进，便于与 Continuation、微分修正、测试与文档工作对齐。

---

## 当前已交付（基线）

- Richardson 初值、`setup_halo_orbit_fixed_z0` / `fixed_x0`、单轨与种子生成脚本。
- `pseudo_arclength_continuation`（XZ 对称 PAL + 多种 `dc_scheme`）。
- `halo_pseudo_arclength_continuation`（双向支、步长与方向参数可对齐 MATLAB 示例脚本）。
- PAL 内层数值防护：收敛判定顺序、牛顿步限幅、非物理支回退、必要时 DC 策略回退。
- 绘图脚本：`plot_halo_orbit.py`、`plot_halo_family.py`。
- 文档：`algorithms/halo.md`（中英双语，英文版通过 Docusaurus i18n 提供）。

---

## 短期（1–2 个迭代）

| 优先级 | 项 | 说明 |
|--------|----|------|
| P0 | **PAL 与 MATLAB 逐行对齐开关** | 可选实现「内层用固定上一曲线点 \(\mathbf{X}\) 计算 \(F\)」分支，用于与 `continuation_PAL_CR3BP.m` 二进制级对照；默认仍用 \(\mathbf{X}_{new}\) 计算 \(F\)。 |
| P0 | **延拓回归测试** | 在 `tests/algorithms/` 增加小规模 PAL 步（固定随机种子或解析初值），断言 \(\mathbf{X}\) 落在合理区间、周期 \(T\in[0.5,5]\)（无量纲）。 |
| P1 | **`dc_scheme=matlab_halo_type1` 稳健性** | 调研 STM 牛顿与 MATLAB `newton_symPeriodicXZ_fixedX` 差异；或统一在 PAL 后先用 `adaptive` 再可选 refine。 |
| P1 | **性能** | `compute_F_and_dF_symmetric_xz_plane` 与动力学积分：评估减少 `t_eval` 点数、复用 STM 或粗网格预估 + 细校正。 |

---

## 中期

| 项 | 说明 |
|----|------|
| **L2 Halo 与南支 Halo** | 脚本参数与文档已部分支持 `libration_point` / `halo_class`；补充示例 JSON 与绘图默认范围。 |
| **自然参数族 `generate_halo_family`** | 明确与 PAL 族的适用场景（小幅振幅扫描 vs 跟踪族曲线）；文档化限制。 |
| **转折与分叉** | 真伪弧长在转折点附近的行为；可选接入更一般的 `pseudo_arclength` 增广方程（与现有 Lyapunov/DRO 延拓统一接口）。 |

---

## 长期

| 项 | 说明 |
|----|------|
| **与多体/星历接口** | 从无量纲 CR3BP 结果映射到有量纲、任务时间线。 |
| **GUI / Notebook** | 基于现有脚本封装交互式族生成与稳定性浏览。 |

---

## 与质量改进计划的关联

- **算法测试 Epic**：将「Continuation PAL + Halo」纳入 `continuation.py` 与 `differential_correction.py` 的覆盖率目标（参见总计划 **Feature: Algorithm 模块测试**）。
- **文档**：本路线图与 `docs/algorithms/halo.md` 同步更新；重大 API 变更时更新 `docs/reference/api-reference.md` 与 `docs/guides/orbit-generation.md`。

---

## 修订记录

| 日期 | 变更 |
|------|------|
| 2026-03 | 初版：Halo PAL、脚本与文档基线后的首份路线图 |
