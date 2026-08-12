# Issue #287 Axial 轨道族——文献调研报告（最终版 v3）

> **状态**：已完成
> **日期**：2026-08-03

---

## 1. Axial 轨道的精确定义

### 1.1 Gómez vol I Type A/B/C 分类

Gómez et al. (2001) Vol. I, lines 653-669：

> "Only what Henon calls **vertical critical orbits** (|a_v| = 1) can be embedded
> in three-dimensional families of periodic orbits."

> "**Type A**: Symmetric periodic orbits with respect to the (x₁, x₃)-plane and
> generated from planar periodic orbits for which a_v = 1, c_v = 0."

> "**Type B**: Symmetric with respect to the **x₁-axis** generating from planar
> ones for which **a_v = 1, b_v = 0**."

> "**Type C**: Symmetric with respect to both the (x₁, x₃)-plane and the x₁-axis,
> originating from planar periodic orbits with a_v = -1, b_v or/and c_v = 0."

> "Orbits of type A bifurcated from the families (a), (b) and (c) are usually
> called **halo orbits**."

→ **Axial = Type B**。Gómez 原文未命名，"Axial" 由 Grebow (2006) / Campbell (1999) 命名。

### 1.2 分岔机制

**分岔路径**：
```
Planar Lyapunov family (xy 平面周期轨道)
    │
    │ 在垂直临界轨道处 (|a_v| = 1)
    │
    ├── a_v=+1, c_v=0 → Type A pitchfork → Halo family (xz 平面对称)
    │
    ├── a_v=+1, b_v=0 → Type B pitchfork → Axial family (x 轴对称)
    │
    └── a_v=-1 → Type C → Vertical family
```

**Haapala & Howell (2016)**：

> "The halo and **axial** orbit families may be located via their **bifurcations
> from the planar Lyapunov orbits** [Grebow, 2006; Campbell, 1999; Haapala et al., 2014]."

> "The axial families exist for a **narrow range of Jacobi constant values, between
> the values associated with the bifurcations from the planar Lyapunov family and
> from the vertical family**."

### 1.3 对称性对比

| | Halo (Type A) | Axial (Type B) |
|--|---------------|----------------|
| 对称面 | (x₁, x₃) 平面 | x₁ 轴 |
| 分岔条件 | a_v=1, c_v=0 | a_v=1, b_v=0 |
| 初始条件 | (x₀, 0, z₀, 0, ẏ₀, 0) | (x₀, 0, z₀, 0, ẏ₀, ż₀) |
| 半周期后 | y=0, ẋ=0, ż=0 | y=0, ẋ=0, z→-z, ż→-ż |

**Axial 比 Halo 多一个自由度**：ż₀ 不为零（z 和 ż 在半周期时都反号）。

### 1.4 Jacobi 常数范围

**Haapala & Howell (2016)**：

| 平动点 | Axial Jacobi 范围 | 宽度 |
|--------|-------------------|------|
| L1 | 2.991 ≤ C ≤ 3.021 | 0.030 |
| L2 | 2.967 ≤ C ≤ 3.014 | 0.047 |

对比：Halo 族的 Jacobi 范围 ~0.2，Lyapunov 族 ~0.2。

### 1.5 稳定性特征

**Guzzetti (2016)**：

> "The L₁ axial orbits possess **no quasi-periodic motion** along the entire family."

> "The L₂ axial family does not possess any members with a **center subspace**."

→ Axial 轨道**全部不稳定**，无准周期邻域。

---

## 2. 存在范围

**Guzzetti (2016) Table 1**：Axial (Ai) 存在于 **所有 L_i（L1-L5）**。

**Haapala & Howell (2016)**：提供了 L1 和 L2 的 Jacobi 范围。L3/L4/L5 的 Axial Jacobi 范围待查。

---

## 3. 与其他轨道族的关系

### 3.1 同源分岔

```
Planar Lyapunov
    ├── Halo (Type A) ← 已实现
    ├── Axial (Type B) ← 本 issue
    └── Vertical ← 已有枚举
```

三者从同一个平面族分岔，是"兄弟"关系。

### 3.2 与 Tadpole/Horseshoe 的关系

**Axial 是 LPO（libration point orbit），Tadpole/Horseshoe 是独立分类。**

- Axial: 3D 周期轨道，从 Lyapunov 分岔，紧邻共线平动点
- Tadpole: 2D 准周期轨道，绕单个 L4/L5
- Horseshoe: 2D 准周期轨道，跨越 L4-L5

三者**不是连续过渡关系**。

---

## 4. 实施策略

### 4.1 初猜构造

**方法**：从 planar Lyapunov 轨道的 Type B 垂直临界点出发，施加 z 方向扰动。

1. 沿 Lyapunov 族计算垂直稳定性指数 a_v
2. 找到 a_v = +1 且 b_v = 0 的垂直临界轨道
3. 在该轨道上施加小 z 扰动（含 ż₀）作为 Axial 族的种子
4. 用 PAL 延拓沿族行走

**可复用的代码**：
- `halo_initial_guess.py` 的 Richardson 展开 → 修改为 Type B 版本
- `halo_pseudo_arclength_continuation` → 直接复用 PAL 框架
- `compute_stability_index` → 垂直稳定性分析

### 4.2 微分修正

**对称性**：关于 x 轴对称（Type B）

初始条件：(x₀, 0, z₀, 0, ẏ₀, ż₀)
半周期后：y=0, ẋ=0, z→-z, ż→-ż

**约束**：
- y(T/2) = 0
- ẋ(T/2) = 0

**自由变量**：x₀, z₀, ẏ₀, ż₀, T/2（5 个自由变量，2 个约束 → 3 维族）

**需要新的修正策略**：`setup_axial_fixed_z0` 等。

### 4.3 延拓

**可复用**：`halo_pseudo_arclength_continuation` 的 PAL 框架。

需要修改：修正策略（从 Halo 的 xz 对称改为 Axial 的 x 轴对称）。

---

## 5. 参考文献

1. **Gómez, G. et al. (2001)**. Vol. I, lines 653-669. Type A/B/C 分岔分类。
2. **Haapala, A. & Howell, K.C. (2016)**. Axial 分岔来源和 Jacobi 范围。
3. **Guzzetti, D. et al. (2016)**. Axial 分类表和稳定性特征。
4. **Folta, D.C. et al. (2015)**. Axial 作为标准 LPO 子类。
5. **Grebow, D. (2006)**. "Axial" 命名来源（待获取原文）。
6. **Campbell, E. (1999)**. Axial 轨道早期计算（待获取原文）。
7. **Oshima, K. (2019)**. 垂直不稳定性机制。
8. **Qi, Y. & Ruiter, A. (2020)**. L4/L5 Axial 轨道的 torus 结构（待获取）。
