# Issue #288 DPO 轨道族——文献调研报告

> **状态**：已完成
> **日期**：2026-08-04

---

## 1. DPO 的精确定义

### 1.1 文献来源

**Folta et al. (2015)** "An Earth–Moon system trajectory design reference catalog"（AIAA SciTech）：

> "direct prograde orbits about the Moon (DPOs)...display, in general,
> counterclockwise motion about the Moon"（旋转坐标系下）

> "Recently, the DPOs have been examined as transfer mechanisms between
> L₁ and L₂ Lyapunov orbits, and as lunar parking orbits."

**Guzzetti et al. (2016)** "Rapid trajectory design in the Earth–Moon
ephemeris system via an interactive catalog of periodic orbits"（JGCD）：

Table 1 分类：DPO 归 **Moon-centered (P2)** 类，tag = "DPO"。

### 1.2 物理定义

DPO（Distant Prograde Orbit）= xy 平面内围绕月球的**顺行**周期轨道：

| 特征 | DRO（逆行） | DPO（顺行） |
|------|------------|------------|
| 旋转坐标系下方向 | 顺时针 | 逆时针 |
| vy0 初始符号 | > 0 | < 0 |
| 月心距 | 较远（~75k–106k km） | 较近（~16k–47k km） |
| Jacobi 常数 | 较低（~2.92） | 较高（~2.99–3.19） |
| 周期 | 较长 | 较短 |
| 对称性 | x 轴对称（xy 平面内） | x 轴对称（xy 平面内） |
| 族参数 | x0（近侧穿越点） | x0（近侧穿越点） |
| 修正策略 | `setup_2D_symmetric_x_fixed_x0` | 同 DRO |

### 1.3 与其他族的关系

**Folta 2015 Figure 7**（Jacobi 常数对比）：

- DPO 与 DRO 的 Jacobi 范围**不重叠** → 低转移成本连接不太可能
- DPO 与 L₁ Lyapunov 的 Jacobi 范围**重叠** → 存在低转移成本通道

### 1.4 存在范围

DPO 族在所有共线平动点（L1–L5）的 Moon-centered 区域均存在，
但文献主要关注 L2 邻域的 DPO（与 L1/L2 Lyapunov 转移相关）。

---

## 2. 种子获取策略

### 2.1 方案 A（已验证）：DRO seed 反转 vy0

从 DRO 标准种子（x0=0.79, vy0=+0.537）反转 vy0 方向，在不同 x0
处微分修正收敛：

| x0 | period | C (Jacobi) | vy0 | 备注 |
|----|--------|------------|-----|------|
| 0.90 | 2.50 | 3.191 | -0.248 | ← DPO 标准种子 |
| 0.95 | 2.26 | 3.316 | -0.531 | |
| 1.05 | 2.36 | 3.094 | -0.510 | |
| 1.10 | 1.71 | 2.991 | -0.460 | |

**选定种子**：x0=0.90, vy0=-0.247645, period=2.5022。

### 2.2 方案 B（备选）：近月 prograde 圆轨道

若方案 A 在某些振幅范围不收敛，可从近月 prograde 圆轨道出发
（r ≈ MOON_RADIUS + 500km）逐步族行走拉远。本实施未用到。

---

## 3. 振幅定义

与 DRO 一致：**一个周期内距月心距离最小/最大值的均值（km）**。

实测 20000 km 振幅的 DPO：
- 近月距 ≈ 较小值 DU，远月距 ≈ 较大值 DU
- Jacobi ≈ 高于 DRO

---

## 4. 参考文献

1. **Folta, D.C. et al. (2015)**. An Earth–Moon system trajectory design
   reference catalog. AIAA SciTech. DPO 分类、Jacobi 对比、L1-Lyapunov
   转移通道。
2. **Guzzetti, D. et al. (2016)**. Rapid trajectory design in the
   Earth–Moon ephemeris system via an interactive catalog of periodic
   orbits. JGCD. Table 1 DPO 分类为 Moon-centered (P2) 类。
