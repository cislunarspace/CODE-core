# 修订后的 Issue #289 描述

以下是修订后的 issue body，可直接更新到 GitHub issue。

---

## 修订后描述

L4/L5 三角平动点的 **Short-Period Orbit (SPO)** 精确周期轨道族。

SPO 是 CR3BP 中围绕 L4/L5 的短周期族成员（$\mathcal{L}_s$），周期 ≈ 1
朔望月（~28 天），形状为近似椭圆，**近稳定**（特征值模 ≈ 1.001）。
与现有 `design_triangular`（拟周期近似）互补，SPO 提供精确的 CR3BP
周期解。

## 物理定义

- **Gómez 分类**：短周期族 $\mathcal{L}_s$，从 L4/L5 平衡点的线性化
  椭圆运动沿族延拓得到（Gómez vol II, §2.5）
- **对称性**：**无** x 轴或 xz 平面对称性（y₀≠0）
- **维度**：平面轨道（z₀=0, ż₀=0）
- **Jacobi 常数**：≈ 2.91（Capdevila & Howell 2018, Table 1）
- **周期**：≈ 27-31 天（1 朔望月量级）
- **稳定性**：近稳定（Gómez vol II 中间方程 λ=1.0011）
- **形状**：近似椭圆，主导谐波 n=1（Gómez vol II Fourier 分析）

## 种子数据（Capdevila & Howell 2018, JGCD Table 1）

| 轨道 | x₀ (nd) | y₀ (nd) | ẋ₀ (nd) | ẏ₀ (nd) | 周期 (天) | C |
|------|---------|---------|---------|---------|----------|---|
| L4 SPO | -0.2255 | 0.8660 | -0.2384 | 0.2494 | 28.3488 | 2.9132 |
| L5 SPO | -0.2255 | -0.8660 | 0.2384 | 0.2494 | 28.3488 | 2.9132 |

## 要构建

1. 通用平面周期轨道修正方法（`setup_spo_fixed_x0`，无对称性假设）
2. SPO 初猜策略（Capdevila 种子 + `_walk_family` 沿 x₀ 行走）
3. `design_spo` 设计函数（振幅 = 距 L4/L5 径向距离均值，km）
4. 注册到 registry（`"L4_SPO"` / `"L5_SPO"`）
5. `_validate_params` + `_cr3bp_orbit_for` 分发

## 验收（ADR 0013）

- 周期闭合 < 1e-8
- Jacobi 守恒 < 1e-10
- 平面约束 max|z| < 1e-8
- 周期范围 27-31 天
- Jacobi 范围 2.85-2.95
- Capdevila 种子直接复现
- L4/L5 镜像对称验证
- 注册到 design_orbit

## 与现有 L4/L5 的关系

现有 `design_triangular`（拟周期初猜，走 `"L4"` / `"L5"` 注册表条目）
**保持不变**。新增 `"L4_SPO"` / `"L5_SPO"` 走 `design_spo`（精确周期
轨道）。两者互补而非替代。

## 参考

- Gómez et al. (2001) Vol. II（L4/L5 周期轨道族结构 + 数值延拓方法）
- Capdevila & Howell (2018) JGCD（CR3BP SPO 显式初始条件 + DRO↔SPO
  转移设计）
- 详细实施方案：`archive/plans/issue289-spo-implementation-plan.md`
