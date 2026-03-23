# e2m2e 开发与修订指南

## 1. 库的目录结构与职责划分

```
e2m2e/e2m2e/
├── core/          ← 数据结构和基础物理模型（改动最需谨慎）
├── algorithms/    ← 数值算法（最常扩展的地方）
├── transfer/      ← 转移轨道设计方案
├── visualization/ ← 绘图工具
└── __init__.py    ← 公共API注册入口
```

**原则：`core/` 是地基，`algorithms/` 和 `transfer/` 是上层建筑。** 修改 core 会波及全库，扩展 algorithms/transfer 则相对独立。

---

## 2. 最常见的修订场景

### 场景A：给已有类添加字段/方法

直接在对应文件中添加即可。例如给 `Orbit` 加一个 `energy` 属性：

```python
# core/orbit.py 中添加
@property
def energy(self):
    """计算轨道能量（Jacobi常数的负半）"""
    if self.system is not None:
        return -0.5 * self.jacobi_constant
    return None
```

**注意**：如果新字段影响 `save_to_file()` / `load_from_file()`，记得同步更新序列化逻辑。

### 场景B：添加新算法

1. 在 `e2m2e/algorithms/` 下创建新文件，如 `multiple_shooting.py`
2. 在 `algorithms/__init__.py` 中导出
3. 在 `e2m2e/__init__.py` 中注册到公共API

```python
# algorithms/multiple_shooting.py
from ..core.dynamics import CR3BP_Dynamics
from ..core.orbit import Orbit

class MultipleShooting:
    def __init__(self, dynamics: CR3BP_Dynamics):
        self.dynamics = dynamics
    ...
```

```python
# __init__.py 中添加
from .algorithms.multiple_shooting import MultipleShooting
# 并加入 __all__
```

### 场景C：添加新的转移方案

同理在 `transfer/` 下加文件，模式与上面一致。

---

## 3. 关键注意事项

### 3.1 保持接口稳定（最重要）

在外部调用这个库时，**公共方法的签名不能随意改**：

```python
# ❌ 破坏性改动 — 外部调用会崩
def propagate(self, initial_state, t_span, with_stm=False):
# 改成了
def propagate(self, initial_state, t_span, stm_mode="none"):  # 参数名变了

# ✅ 向后兼容的改动
def propagate(self, initial_state, t_span, with_stm=False, **kwargs):
```

如果必须改接口，**通过添加新参数并给默认值**来保持兼容。

### 3.2 `editable install` 的优势

已经用 `pip install -e .` 安装，这意味着：
- 直接修改 `e2m2e/e2m2e/` 下的源码，**外部 `import e2m2e` 立即生效**，无需重新安装
- **唯一例外**：如果修改了 `pyproject.toml`（如加依赖），需要重新 `pip install -e .`

### 3.3 核心类之间的依赖关系

```
CR3BP_System  ←─ CR3BP_Dynamics  ←─ DifferentialCorrection
                      ↑                      ↑
                    Orbit           Continuation, StabilityAnalysis
                      ↑
              CoordinateTransformation
```

**改 `CR3BP_Dynamics` 时特别注意**：
- `equations_of_motion(t, state)` 的签名被微分修正、延拓、转移设计等所有算法调用
- `propagate()` 返回的字典 key（`'states'`, `'time'`, `'stm'`, `'jacobi_error'`）被多处依赖
- 如果要加新的动力学模型（如椭圆限制性三体问题 ER3BP），**建议新建子类而非修改基类**

### 3.4 数值敏感性

CR3BP 计算对精度非常敏感，修改时注意：
- 积分器始终用 `rtol=1e-12, atol=1e-12` 以上精度
- 有限差分步长（微分修正中的 `eps`）不要随意调大
- 修改状态向量顺序 `[x, y, z, vx, vy, vz]` 会导致**全局崩溃**

### 3.5 版本管理

每次有实质性改动时，更新 `__init__.py` 中的 `__version__`：

```
0.1.0 → 0.1.1  修bug/小调整
0.1.0 → 0.2.0  添加新功能模块
0.1.0 → 1.0.0  接口有破坏性变更
```

---

## 4. 推荐的开发工作流

```
1. 写/改代码  →  e2m2e/e2m2e/ 下对应模块
2. 跑测试验证 →  python tests/test_basic.py
3. 外部脚本调用 →  在项目根目录或其他项目中 import e2m2e
4. 发现问题   →  回到步骤1
```

如果后续需要更完善的测试，可以用 pytest：
```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## 5. 快速参考：添加新内容的 checklist

| 动作 | 需要改的文件 |
|------|------------|
| 给已有类加方法 | 对应模块文件 |
| 加新算法类 | 新文件 + 子包 `__init__.py` + 顶层 `__init__.py` |
| 加新依赖 | `pyproject.toml` → 重新 `pip install -e .` |
| 改公共接口 | 对应模块 + 测试 + 确认外部调用兼容 |
| 加新子包 | 新目录 + `__init__.py` + 顶层注册 |

**核心原则：改 core 要谨慎、扩展 algorithms/transfer 很自由、改完跑测试。**
