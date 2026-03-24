# e2m2e Development and Revision Guide

## 1. Library Directory Structure and Responsibilities

```
e2m2e/e2m2e/
├── core/          ← Data structures and basic physics models (changes require extra caution)
├── algorithms/    ← Numerical algorithms (most frequently extended)
├── transfer/      ← Transfer trajectory design schemes
├── visualization/ ← Plotting tools
└── __init__.py    ← Public API registration entry
```

**Principle: `core/` is the foundation, `algorithms/` and `transfer/` are the superstructure.** Modifying core affects the entire library, while extending algorithms/transfer is relatively independent.

---

## 2. Most Common Revision Scenarios

### Scenario A: Adding Fields/Methods to Existing Classes

Simply add them in the corresponding file. For example, adding an `energy` property to `Orbit`:

```python
# In core/orbit.py, add
@property
def energy(self):
    """Calculate orbital energy (negative half of Jacobi constant)"""
    if self.system is not None:
        return -0.5 * self.jacobi_constant
    return None
```

**Note**: If the new field affects `save_to_file()` / `load_from_file()`, remember to update the serialization logic accordingly.

### Scenario B: Adding New Algorithms

1. Create a new file under `e2m2e/algorithms/`, e.g., `multiple_shooting.py`
2. Export it in `algorithms/__init__.py`
3. Register it in the public API in `e2m2e/__init__.py`

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
# Add in __init__.py
from .algorithms.multiple_shooting import MultipleShooting
# And add to __all__
```

### Scenario C: Adding New Transfer Schemes

Similarly, add files under `transfer/`, with the same pattern as above.

---

## 3. Key Considerations

### 3.1 Maintain Interface Stability (Most Important)

When externally calling this library, **public method signatures cannot be arbitrarily changed**:

```python
# ❌ Breaking change — external calls will break
def propagate(self, initial_state, t_span, with_stm=False):
# changed to
def propagate(self, initial_state, t_span, stm_mode="none"):  # parameter name changed

# ✅ Backward-compatible change
def propagate(self, initial_state, t_span, with_stm=False, **kwargs):
```

If you must change an interface, **add new parameters with default values** to maintain compatibility.

### 3.2 Advantages of `editable install`

Since installed with `pip install -e .`, this means:
- Modifying source code in `e2m2e/e2m2e/` **takes effect immediately in external `import e2m2e`**, no reinstallation needed
- **Only exception**: If you modify `pyproject.toml` (e.g., adding dependencies), you need to re-run `pip install -e .`

### 3.3 Dependency Relationships Between Core Classes

```
CR3BP_System  ←─ CR3BP_Dynamics  ←─ DifferentialCorrection
                       ↑                      ↑
                     Orbit           Continuation, StabilityAnalysis
                       ↑
               CoordinateTransformation
```

**Special attention when modifying `CR3BP_Dynamics`**:
- The signature of `equations_of_motion(t, state)` is called by all algorithms including differential correction, continuation, and transfer design
- Dictionary keys returned by `propagate()` (`'states'`, `'time'`, `'stm'`, `'jacobi_error'`) are depended on by multiple places
- If adding new dynamics models (e.g., Elliptic Restricted Three-Body Problem ER3BP), **it is recommended to create a new subclass rather than modifying the base class**

### 3.4 Numerical Sensitivity

CR3BP calculations are very sensitive to precision. When modifying, pay attention:
- Always use integrator with `rtol=1e-12, atol=1e-12` or higher precision
- Do not arbitrarily increase finite difference step sizes (`eps` in differential correction)
- Changing the state vector order `[x, y, z, vx, vy, vz]` will cause **global crashes**

### 3.5 Version Management

When making substantial changes, update `__version__` in `__init__.py`:

```
0.1.0 → 0.1.1  Bug fixes/minor adjustments
0.1.0 → 0.2.0  New feature modules added
0.1.0 → 1.0.0  Breaking interface changes
```

---

## 4. Recommended Development Workflow

```
1. Write/modify code  →  Corresponding module under e2m2e/e2m2e/
2. Run tests to verify →  python tests/test_basic.py
3. External script calls →  import e2m2e from project root or other projects
4.发现问题   →  Go back to step 1
```

If you need more comprehensive testing later, you can use pytest:
```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## 5. Quick Reference: Checklist for Adding New Content

| Action | Files to Modify |
|--------|-----------------|
| Add method to existing class | Corresponding module file |
| Add new algorithm class | New file + subpackage `__init__.py` + top-level `__init__.py` |
| Add new dependency | `pyproject.toml` → re-run `pip install -e .` |
| Modify public interface | Corresponding module + tests + verify external call compatibility |
| Add new subpackage | New directory + `__init__.py` + top-level registration |

**Core Principle: Be cautious when modifying core, extending algorithms/transfer is flexible, run tests after changes.**
