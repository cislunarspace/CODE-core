# CR3BP 轨道力学算法技术文档

## 1. 概述

本文档描述了 Earth-to-Moon-to-Earth (E2M2E) 项目中实现的圆型限制性三体问题 (Circular Restricted Three-Body Problem, CR3BP) 轨道力学算法。这些算法用于生成月球远距离逆行轨道 (Distant Retrograde Orbit, DRO) 及其他周期轨道。

### 1.1 CR3BP 模型简介

CR3BP 描述一个小质量天体在两个大质量天体（主天体和次天体）引力作用下的运动。在地月系统中：
- 主天体：地球
- 次天体：月球
- 小天体：航天器

在旋转坐标系（会合坐标系）中，系统的无量纲运动方程为：

$$\ddot{x} - 2\dot{y} = \frac{\partial U}{\partial x}$$
$$\ddot{y} + 2\dot{x} = \frac{\partial U}{\partial y}$$
$$\ddot{z} = \frac{\partial U}{\partial z}$$

其中 $U = \frac{1}{2}(x^2 + y^2) + \frac{1-\mu}{r_1} + \frac{\mu}{r_2}$ 为人工势能函数，$r_1$ 和 $r_2$ 分别是航天器到主天体和次天体的距离。

### 1.2 关键参数（基于论文 Table 1）

| 参数 | 符号 | 值 | 单位 |
|------|------|-----|------|
| 地月质量比 | $\mu$ | $1.21506683 \times 10^{-2}$ | - |
| 距离单位 | $D_U$ | $3.84405 \times 10^5$ | km |
| 时间单位 | $T_U$ | $4.34811305$ | days |
| 速度单位 | $V_U$ | $1023.23281$ | m/s |

---

## 2. 系统参数模块

### 2.1 CR3BP_System 类

**文件**: `e2m2e/core/system.py`

**功能**: 定义 CR3BP 系统参数和平动点计算。

**主要属性**:
- `mu`: 质量参数 $\mu = m_{moon} / (m_{earth} + m_{moon})$
- `primary_body`: 主天体名称
- `secondary_body`: 次天体名称
- `L1-L5`: 五个平动点坐标

**主要方法**:

```python
class CR3BP_System:
    def compute_libration_points() -> List[Tuple[float, float]]
        """计算五个平动点位置 (L1-L5)"""
        
    def get_jacobi_constant(state: np.ndarray) -> float
        """计算Jacobi常数 C_J = 2U - v²"""
        
    def dimensionless_to_physical(state: np.ndarray) -> Dict[str, np.ndarray]
        """无量纲状态转物理单位"""
        
    def physical_to_dimensionless(state: Dict) -> np.ndarray
        """物理单位转无量纲状态"""
```

**平动点稳定性**: L1、L2、L3 点在动力学上不稳定（马蹄形轨道的基础），L4、L5 点是动力学稳定的（特洛伊天体）。

---

## 3. 动力学模型

### 3.1 CR3BP_Dynamics 类

**文件**: `e2m2e/core/dynamics.py`

**功能**: 实现 CR3BP 动力学方程的数值积分，支持状态转移矩阵 (STM) 计算。

#### 3.1.1 运动方程

```python
def equations_of_motion(self, t: float, state: np.ndarray) -> np.ndarray:
    """
    6维状态向量的运动方程
    
    状态向量: [x, y, z, vx, vy, vz]
    返回导数: [vx, vy, vz, ax, ay, az]
    """
    mu = self.system.mu
    x, y, z, vx, vy, vz = state
    
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)  # 到主天体距离
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)  # 到次天体距离
    
    ax = 2*vy + x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
    ay = -2*vx + y - (1-mu)*y/r1**3 - mu*y/r2**3
    az = -(1-mu)*z/r1**3 - mu*z/r2**3
    
    return np.array([vx, vy, vz, ax, ay, az])
```

#### 3.1.2 状态转移矩阵 (STM)

对于周期轨道搜索，需要线性化动力学并计算状态转移矩阵 $\Phi(t, t_0)$，满足：

$$\dot{\Phi}(t, t_0) = A(t) \cdot \Phi(t, t_0), \quad \Phi(t_0, t_0) = I$$

其中 $A(t)$ 是动力学方程的雅可比矩阵：

$$A = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix} 
0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 \\
U_{xx} & U_{xy} & U_{xz} & 0 & 2 & 0 \\
U_{xy} & U_{yy} & U_{yz} & -2 & 0 & 0 \\
U_{xz} & U_{yz} & U_{zz} & 0 & 0 & 0
\end{bmatrix}$$

```python
def equations_with_stm(self, t: float, augmented_state: np.ndarray) -> np.ndarray:
    """
    42维增广状态向量运动方程 (6状态 + 36 STM元素)
    """
    # ... 计算雅可比矩阵 A ...
    stm_dot = A @ stm  # STM传播
    return np.concatenate([state_derivative, stm_dot.flatten()])
```

#### 3.1.3 Jacobi 常数

Jacobi 常数是 CR3BP 的能量积分：

$$C_J = 2U - |\mathbf{v}|^2 = x^2 + y^2 + \frac{2(1-\mu)}{r_1} + \frac{2\mu}{r_2} - (vx^2 + vy^2 + vz^2)$$

对于有界轨道运动，必须满足 $C_J > 0$（有效势能大于动能）。

#### 3.1.4 轨迹传播

```python
def propagate(
    initial_state: np.ndarray,
    t_span: Tuple[float, float],
    t_eval: Optional[np.ndarray] = None,
    with_stm: bool = False
) -> Dict[str, Any]:
    """
    数值积分传播轨迹
    
    参数:
        initial_state: 初始状态向量 [x, y, z, vx, vy, vz]
        t_span: 时间区间 [t0, tf]
        with_stm: 是否同时计算状态转移矩阵
    
    返回:
        包含轨迹时间、状态、STM等的结果字典
    """
```

**积分器设置**:
- 默认积分器: RK45 (4(5)阶龙格-库塔)
- 相对容差: $1 \times 10^{-12}$
- 绝对容差: $1 \times 10^{-12}$

---

## 4. 微分修正算法

### 4.1 DifferentialCorrection 类

**文件**: `e2m2e/algorithms/differential_correction.py`

**功能**: 通过迭代修正初始条件，将近似周期轨道修正为精确周期轨道。

#### 4.1.1 算法原理

对于周期轨道问题，初始状态 $\mathbf{x}_0$ 和周期 $T$ 必须满足周期条件：

$$\mathbf{x}(T) - \mathbf{x}_0 = \mathbf{0}$$

这构成了一个非线性方程组 $\mathbf{F}(\mathbf{X}) = \mathbf{0}$，其中 $\mathbf{X}$ 是待求的初始参数。

使用牛顿-拉夫森迭代：

$$\mathbf{X}_{k+1} = \mathbf{X}_k - \mathbf{J}^{-1} \mathbf{F}(\mathbf{X}_k)$$

其中 $\mathbf{J} = \frac{\partial \mathbf{F}}{\partial \mathbf{X}}$ 是雅可比矩阵，利用 STM 计算：

$$\frac{\partial \mathbf{x}(T)}{\partial \mathbf{x}_0} = \Phi(T, 0)$$

#### 4.1.2 2D 对称轨道配置 (setup_2D_symmetric_x_fixed_x0)

对于关于 x 轴对称的 DRO（远距离逆行轨道），利用对称性减少待求解参数：

**对称性条件**:
- 初始状态: $[x_0, 0, 0, 0, \dot{y}_0, 0]$（从 x 轴垂直出发）
- 半周期条件: $y(T/2) = 0$, $\dot{x}(T/2) = 0$（再次垂直穿越 x 轴）

**自由变量**: $[\dot{y}_0, T/2]$（初始 y 方向速度和半周期）

**目标约束**: $[y(T/2), \dot{x}(T/2)] = [0, 0]$

```python
def setup_2D_symmetric_x_fixed_x0(self, x0: float):
    """
    配置2D对称轨道微分修正
    
    参数:
        x0: 固定的初始x坐标
    
    配置:
        - 自由变量: [y_dot0, T_half]
        - 约束条件: [y(T/2)=0, x_dot(T/2)=0]
        - 状态索引: y=1, x_dot=3
    """
    self.setup_type = "2D_symmetric_x_fixed_x0"
    self.free_variables = ["y_dot0", "T_half"]
    self.constraint_indices = [1, 3]  # y, x_dot
    self.target_conditions = {"y": 0.0, "x_dot": 0.0}
```

#### 4.1.3 迭代修正流程

```python
def iterate_correction(self, initial_guess: Orbit, verbose: bool = False) -> Optional[Orbit]:
    """
    执行微分修正迭代
    
    流程:
    1. 传播轨道到半周期
    2. 计算约束误差
    3. 构建雅可比矩阵 (利用STM)
    4. 求解校正量
    5. 更新状态
    6. 重复直至收敛
    
    收敛条件:
        - 所有约束误差 < tolerance (1e-12)
        - 或达到最大迭代次数 (50)
    """
```

**自适应阻尼**: 防止过冲和震荡

$$X_{k+1} = X_k + \alpha \cdot \delta X_k$$

其中 $\alpha \in [0.1, 2.0]$ 是自适应阻尼因子。

---

## 5. 轨道族延拓算法

### 5.1 Continuation 类

**文件**: `e2m2e/algorithms/continuation.py`

**功能**: 从已知种子轨道出发，通过参数连续变化生成完整轨道族。

#### 5.1.1 自然参数延拓

```python
def natural_continuation(
    seed_orbit: Orbit,
    param_range: Tuple[float, float],
    step_size: float,
    verbose: bool = False
) -> OrbitFamily:
    """
    自然参数延拓算法
    
    原理:
        从种子轨道出发，固定延拓参数方向，逐步改变参数值，
        每一歩使用前一条轨道作为初始猜测进行微分修正。
    
    参数:
        seed_orbit: 种子轨道（精确的周期轨道）
        param_range: 参数范围 (param_min, param_max)
        step_size: 延拓步长
    
    延拓方向:
        - 正向: param 增大方向
        - 反向: param 减小方向
        - 双向: param_min < seed < param_max
    """
```

#### 5.1.2 延拓参数

| 参数 | 符号 | 物理意义 |
|------|------|----------|
| x0 | $x_0$ | 初始 x 坐标 |
| z0 | $z_0$ | 初始 z 振幅 |
| period | $T$ | 轨道周期 |

#### 5.1.3 伪弧长延拓（备选）

当自然延拓遇到分支点或拐点时，伪弧长延拓可以稳定地穿过这些区域：

$$ds = \sqrt{dx_0^2 + dT^2}$$

---

## 6. 轨道类

### 6.1 Orbit 类

**文件**: `e2m2e/core/orbit.py`

**功能**: 表示单条周期轨道，包含状态序列和时间信息。

**属性**:
- `states`: 状态数组 (n × 6)
- `times`: 时间数组 (n,)
- `period`: 轨道周期
- `jacobi_constant`: Jacobi 常数
- `stability_index`: 稳定性指标

### 6.2 OrbitFamily 类

**文件**: `e2m2e/core/orbit.py`

**功能**: 表示轨道族，管理多条轨道的集合。

**方法**:
```python
class OrbitFamily:
    def add_orbit(self, orbit: Orbit) -> None:
        """添加轨道到族"""
        
    def save_to_file(self, filename: str) -> None:
        """保存轨道族到JSON文件"""
        
    def load_from_file(self, filename: str) -> "OrbitFamily":
        """从JSON文件加载轨道族"""
        
    def compute_stability_indices(self) -> np.ndarray:
        """计算族中所有轨道的稳定性指标"""
```

---

## 7. 稳定性分析

### 7.1 稳定性指标

通过单值矩阵 (Monodromy Matrix) $M = \Phi(T, 0)$ 的特征值判断轨道稳定性：

$$M = \begin{bmatrix} \Phi_{xx} & \Phi_{xy} \\ \Phi_{yx} & \Phi_{yy} \end{bmatrix}$$

**Floquet 乘子**: 特征值 $\lambda_i$，满足 $\lambda_1 \cdot \lambda_2 \cdot \lambda_3 \cdot \lambda_4 = 1$

**稳定性判定**:
- 稳定轨道: 所有特征值在单位圆上（纯虚数共轭对）
- 不稳定轨道: 至少一个特征值在单位圆外

---

## 8. DRO 生成完整流程

### 8.1 算法流程图

```
┌─────────────────────────────────────────┐
│         1. 初始化 CR3BP 系统             │
│    mu = 0.0121506683, 创建 system        │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│         2. 创建种子轨道初猜              │
│   x0 = 0.79188556619742                 │
│   vy0 = 0.53682                         │
│   T_guess = 3.472526005624708           │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│      3. 微分修正得到精确种子轨道          │
│   setup_2D_symmetric_x_fixed_x0(x0)     │
│   iterate_correction()                  │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│      4. 自然延拓生成轨道族               │
│   natural_continuation(seed,            │
│       param_range=(0.2, 0.9),          │
│       step_size=0.0005)                 │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│         5. 保存结果到 JSON               │
│   dro_family_{params}_{timestamp}.json  │
└─────────────────────────────────────────┘
```

### 8.2 代码示例

```python
import e2m2e
from e2m2e.core import Orbit, OrbitFamily

# 1. 创建系统
system = e2m2e.core.system.CR3BP_System(mu=0.0121506683, 
                                         primary="earth", 
                                         secondary="moon")

# 2. 创建动力学模型
dynamics = e2m2e.core.dynamics.CR3BP_Dynamics(system)

# 3. 种子轨道初猜
x0 = 0.79188556619742
vy0 = 0.53682
initial_state = [x0, 0.0, 0.0, 0.0, vy0, 0.0]
seed_orbit = Orbit(states=[initial_state], times=[0])
seed_orbit.period = 3.472526005624708

# 4. 微分修正
corrector = e2m2e.algorithms.DifferentialCorrection(dynamics)
corrector.setup_2D_symmetric_x_fixed_x0(x0)
seed_dro = corrector.iterate_correction(seed_orbit)

# 5. 延拓生成族
continuation = e2m2e.algorithms.Continuation(corrector)
family = continuation.natural_continuation(
    seed_dro,
    param_range=(0.2, 0.9),
    step_size=0.0005
)

# 6. 保存
family.save_to_file("output/dro/dro_family.json")
```

---

## 9. 参考文献

1. Broucke, R. A. (1968). Periodic orbits in the restricted three body problem with Earth-moon masses. NASA JPL.

2. Cui, P., et al. (2025). Two-Impulse Transfers from Lunar Distant Retrograde Orbits to Resonant Orbits. Journal of Guidance, Control, and Dynamics, Vol. 48, No. 6.

3. Koon, W. S., et al. (2011). Dynamical Systems, the Three-Body Problem and Space Mission Design. Springer.

---

## 10. 附录：文件结构

```
e2m2e/
├── e2m2e/
│   ├── core/
│   │   ├── system.py          # CR3BP 系统参数
│   │   ├── dynamics.py        # 动力学方程
│   │   ├── orbit.py          # Orbit 和 OrbitFamily 类
│   │   └── coordinate.py     # 坐标变换
│   └── algorithms/
│       ├── differential_correction.py  # 微分修正
│       ├── continuation.py            # 轨道族延拓
│       └── stability.py              # 稳定性分析
└── docs/
    └── cr3bp_algorithms.md   # 本文档
```
