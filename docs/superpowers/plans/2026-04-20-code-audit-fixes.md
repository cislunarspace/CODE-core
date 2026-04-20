# e2m2e Code Audit Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all HIGH and critical MEDIUM issues from the code audit report in priority order.

**Architecture:** Three phases: (1) numerical safety guards in core/, (2) code quality fixes across modules, (3) input validation and robustness. Each phase is independently committable and testable.

**Tech Stack:** Python 3.10+, NumPy, pytest, ruff, mypy

---

## Phase 1: Numerical Safety (H-SEC-01, H-SEC-03, M-SEC-01, M-SEC-04)

### Task 1: Add singularity protection to CR3BP equations_of_motion (H-SEC-01)

**Files:**
- Modify: `e2m2e/core/dynamics.py:356-404`
- Test: `tests/core/dynamics/test_dynamics.py`

- [ ] **Step 1: Write failing test for singularity protection**

Add to `tests/core/dynamics/test_dynamics.py` (or create the test file if needed):

```python
def test_cr3bp_eom_no_nan_at_primary_body_position(earth_moon_dynamics):
    """State exactly at the primary body position should not produce NaN."""
    mu = earth_moon_dynamics.system.mu
    state_at_primary = np.array([-mu, 0.0, 0.0, 0.0, 0.1, 0.0])
    derivative = earth_moon_dynamics.equations_of_motion(0.0, state_at_primary)
    assert np.all(np.isfinite(derivative)), f"Non-finite values in derivative: {derivative}"


def test_cr3bp_eom_no_nan_at_secondary_body_position(earth_moon_dynamics):
    """State exactly at the secondary body position should not produce NaN."""
    mu = earth_moon_dynamics.system.mu
    state_at_secondary = np.array([1 - mu, 0.0, 0.0, 0.0, 0.1, 0.0])
    derivative = earth_moon_dynamics.equations_of_motion(0.0, state_at_secondary)
    assert np.all(np.isfinite(derivative)), f"Non-finite values in derivative: {derivative}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/dynamics/test_dynamics.py::test_cr3bp_eom_no_nan_at_primary_body_position tests/core/dynamics/test_dynamics.py::test_cr3bp_eom_no_nan_at_secondary_body_position -v`
Expected: FAIL (NaN in derivative)

- [ ] **Step 3: Add MIN_DISTANCE constant and clamping to CR3BP_Dynamics**

In `e2m2e/core/dynamics.py`, add class constant after `STATE_DIM` line (around line 71):

```python
MIN_DISTANCE = 1e-10  # km (dimensionless), prevents division by zero at singularities
```

Then modify `equations_of_motion` (lines 392-395) to clamp r1 and r2:

```python
        # r₁：航天器到较大天体（质量 1-μ，位于 x=-μ）的距离
        r1 = max(np.sqrt((x + mu) ** 2 + y**2 + z**2), self.MIN_DISTANCE)
        # r₂：航天器到较小天体（质量 μ，位于 x=1-μ）的距离
        r2 = max(np.sqrt((x - 1 + mu) ** 2 + y**2 + z**2), self.MIN_DISTANCE)
```

Also modify `compute_jacobian_A` (lines 425-426) to use the same clamping:

```python
        r1 = max(np.sqrt((x + mu) ** 2 + y**2 + z**2), self.MIN_DISTANCE)
        r2 = max(np.sqrt((x - 1 + mu) ** 2 + y**2 + z**2), self.MIN_DISTANCE)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/dynamics/test_dynamics.py -v -k "no_nan"`
Expected: PASS

- [ ] **Step 5: Run full test suite to check no regressions**

Run: `pytest tests/core/ -v --tb=short`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add e2m2e/core/dynamics.py tests/core/dynamics/test_dynamics.py
git commit -m "fix(core): add MIN_DISTANCE clamping to CR3BP equations of motion

Prevents NaN/inf propagation when spacecraft state coincides with
a primary body position. Matches the pattern used in EphemerisDynamics."
```

---

### Task 2: Add singularity protection to get_jacobi_constant (H-SEC-03)

**Files:**
- Modify: `e2m2e/core/system.py:238-258`
- Test: `tests/core/system/test_system.py`

- [ ] **Step 1: Write failing test**

```python
import warnings


def test_jacobi_constant_no_inf_at_body_position(earth_moon_system):
    """Jacobi constant at a body position should return nan with warning, not inf."""
    mu = earth_moon_system.mu
    state_at_primary = np.array([-mu, 0.0, 0.0, 0.0, 0.0, 0.0])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = earth_moon_system.get_jacobi_constant(state_at_primary)
        assert np.isnan(result), f"Expected NaN at singularity, got {result}"
        assert len(w) == 1
        assert issubclass(w[0].category, RuntimeWarning)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/system/test_system.py::test_jacobi_constant_no_inf_at_body_position -v`
Expected: FAIL (returns inf, not nan)

- [ ] **Step 3: Add singularity guard to get_jacobi_constant**

In `e2m2e/core/system.py`, add `import warnings` at the top (if not already there), then modify `get_jacobi_constant` (around lines 249-252):

```python
        r1 = np.sqrt((x + self.mu) ** 2 + y**2 + z**2)
        r2 = np.sqrt((x - 1 + self.mu) ** 2 + y**2 + z**2)

        if r1 < 1e-12 or r2 < 1e-12:
            warnings.warn(
                "State at singularity in Jacobi constant calculation (r1={:.2e}, r2={:.2e})".format(r1, r2),
                RuntimeWarning,
                stacklevel=2,
            )
            return float("nan")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/system/test_system.py -v -k "jacobi"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add e2m2e/core/system.py tests/core/system/test_system.py
git commit -m "fix(core): add singularity guard to get_jacobi_constant

Returns NaN with RuntimeWarning instead of inf when state coincides
with a primary body position."
```

---

### Task 3: Add mu range validation to CR3BP_System (M-SEC-01)

**Files:**
- Modify: `e2m2e/core/system.py:107-117`
- Test: `tests/core/system/test_system.py`

- [ ] **Step 1: Write failing test**

```python
import pytest


def test_mu_validation_rejects_zero():
    with pytest.raises(ValueError, match="mu must be in"):
        CR3BP_System(mu=0.0, primary="A", secondary="B")


def test_mu_validation_rejects_negative():
    with pytest.raises(ValueError, match="mu must be in"):
        CR3BP_System(mu=-0.01, primary="A", secondary="B")


def test_mu_validation_rejects_half():
    with pytest.raises(ValueError, match="mu must be in"):
        CR3BP_System(mu=0.5, primary="A", secondary="B")


def test_mu_validation_accepts_valid():
    system = CR3BP_System(mu=0.012, primary="Earth", secondary="Moon")
    assert system.mu == 0.012
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/system/test_system.py -k "mu_validation" -v`
Expected: FAIL

- [ ] **Step 3: Add validation to CR3BP_System.__init__**

In `e2m2e/core/system.py`, after line `self.mu: float = mu` (line 117):

```python
        if not (0 < mu < 0.5):
            raise ValueError(
                f"mu must be in (0, 0.5), got {mu}. "
                f"mu = m2/(m1+m2) where m2 is the smaller body mass."
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/system/test_system.py -k "mu_validation" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add e2m2e/core/system.py tests/core/system/test_system.py
git commit -m "fix(core): validate mu parameter range in CR3BP_System

Rejects mu outside (0, 0.5) to prevent complex numbers in libration
point computation and division by zero in equations of motion."
```

---

### Task 4: Add period > 0 guard to Orbit.compute_stability (M-SEC-04)

**Files:**
- Modify: `e2m2e/core/orbit.py:304-335`
- Test: `tests/core/test_orbit.py`

- [ ] **Step 1: Write failing test**

```python
def test_compute_stability_with_zero_period(sample_orbit):
    """compute_stability should handle zero period gracefully."""
    sample_orbit.period = 0.0
    result = sample_orbit.compute_stability(dynamics_fixture)
    assert result["stability"] == "unknown"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_orbit.py::test_compute_stability_with_zero_period -v`
Expected: FAIL (ZeroDivisionError or inf)

- [ ] **Step 3: Add guard to compute_stability**

In `e2m2e/core/orbit.py`, at the beginning of `compute_stability` (around line 304):

```python
    def compute_stability(self, dynamics: CR3BP_Dynamics) -> dict[str, Any]:
        if self._period is None or self._period <= 0:
            return {
                "stability": "unknown",
                "eigenvalues": None,
                "max_deviation": None,
                "lyapunov_exponents": None,
            }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_orbit.py::test_compute_stability_with_zero_period -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add e2m2e/core/orbit.py tests/core/test_orbit.py
git commit -m "fix(core): guard Orbit.compute_stability against zero/None period

Returns 'unknown' stability dict instead of ZeroDivisionError or inf
Lyapunov exponents."
```

---

## Phase 2: Code Quality (H-03, H-07, H-04, H-01, H-02)

### Task 5: Replace print() with logging in continuation.py (H-03)

**Files:**
- Modify: `e2m2e/algorithms/continuation.py`

- [ ] **Step 1: Add logger import at top of continuation.py**

After the existing imports, add:

```python
import logging

logger = logging.getLogger(__name__)
```

- [ ] **Step 2: Replace all print() calls with logger calls**

Replace every `print(f"...")` with the appropriate logger level:
- Section headers / status: `logger.info(...)`
- Progress updates (every N orbits): `logger.info(...)`
- Warnings: `logger.warning(...)`
- Debug details (X values, tangents): `logger.debug(...)`

Keep the `verbose` parameter behavior: when `verbose=False`, info-level messages are suppressed by standard logging configuration. Remove explicit `if verbose:` guards where the logger level naturally controls output.

Pattern for each replacement:
```python
# Before:
if verbose:
    print(f"  第 {i + 1} 条轨道，参数值={param_val:.6f}")
# After:
logger.info("  轨道 %d，参数值=%.6f", i + 1, param_val)
```

For unconditional print statements (always shown regardless of verbose):
```python
# Before:
print(f"\n{'=' * 60}")
print(f"开始自然参数延拓")
# After:
logger.info("开始自然参数延拓")
```

- [ ] **Step 3: Run tests to verify no regressions**

Run: `pytest tests/algorithms/ -v --tb=short -k "continuation"`
Expected: All pass (logger output goes to stderr, not captured by default)

- [ ] **Step 4: Commit**

```bash
git add e2m2e/algorithms/continuation.py
git commit -m "refactor(algorithms): replace print() with logging in continuation module

Use logger.info/debug/warning instead of print() for all output.
Standard logging configuration now controls verbosity."
```

---

### Task 6: Replace assert statements with explicit exceptions (H-07)

**Files:**
- Modify: `e2m2e/core/orbit.py:316`
- Modify: `e2m2e/core/system.py:232,415`
- Modify: `e2m2e/visualization/base.py:252`

- [ ] **Step 1: Fix core/orbit.py:316**

Replace:
```python
        assert self._eigenvalues is not None
```
With:
```python
        if self._eigenvalues is None:
            raise ValueError("Eigenvalues not computed. Call compute_monodromy_matrix first.")
```

- [ ] **Step 2: Fix core/system.py:232**

Replace:
```python
        assert self.L_points is not None
```
With:
```python
        if self.L_points is None:
            raise ValueError("Libration points not computed.")
```

- [ ] **Step 3: Fix core/system.py:415**

Replace:
```python
                assert self.orbital_period is not None
```
With:
```python
                if self.orbital_period is None:
                    raise ValueError("Orbital period not set.")
```

- [ ] **Step 4: Fix visualization/base.py:252**

Replace:
```python
        assert self.system.L_points is not None
```
With:
```python
        if self.system.L_points is None:
            raise ValueError("Libration points not computed. Call compute_libration_points() first.")
```

- [ ] **Step 5: Run tests to verify no regressions**

Run: `pytest tests/core/ tests/visualization/ -v --tb=short`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add e2m2e/core/orbit.py e2m2e/core/system.py e2m2e/visualization/base.py
git commit -m "fix: replace assert statements with explicit ValueError exceptions

Assert statements are stripped by python -O, making them unreliable
for data validation. Converted to explicit checks with descriptive
error messages."
```

---

### Task 7: Unify coverage threshold (H-04)

**Files:**
- Modify: `pyproject.toml:81`

- [ ] **Step 1: Update fail_under in pyproject.toml**

Change line 81 from:
```toml
fail_under = 50
```
To:
```toml
fail_under = 80
```

- [ ] **Step 2: Verify CI workflow matches**

The CI workflow already uses `--cov-fail-under=80`, so no change needed there.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: unify coverage threshold to 80% in pyproject.toml

Previously pyproject.toml had 50% while CI required 80%, causing
confusion for local development."
```

---

### Task 8: Add logger.debug to silent exception catches in config.py (H-01)

**Files:**
- Modify: `e2m2e/visualization/config.py:39,48,103,105,169,189`

- [ ] **Step 1: Add logger calls to all 6 silent except blocks**

For each `except ... pass` or `except Exception: pass` block, add `logger.debug(...)`:

At line 39 (MPL_SCALE ValueError):
```python
        except ValueError:
            logger.debug("Invalid MPL_SCALE value: %s", env)
```

At line 48 (GDK_SCALE/QT_SCALE ValueError):
```python
            except ValueError:
                logger.debug("Invalid %s value: %s", var, val)
```

At line 103 (FileNotFoundError for xrandr):
```python
    except FileNotFoundError:
        logger.debug("xrandr not found, skipping DPI detection")
```

At line 105 (general xrandr exception):
```python
    except Exception:
        logger.debug("xrandr query failed", exc_info=True)
```

At lines 169 and 189 (zenity subprocess exceptions):
```python
            except Exception:
                logger.debug("zenity file dialog failed", exc_info=True)
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/visualization/ -v --tb=short`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add e2m2e/visualization/config.py
git commit -m "fix(visualization): add debug logging to silent exception catches in config

Previously, DPI detection and zenity failures were completely silent.
Now logged at DEBUG level for troubleshooting."
```

---

### Task 9: Add failure tracking to detect_bifurcation_in_family (H-02)

**Files:**
- Modify: `e2m2e/algorithms/stability.py:386-413`

- [ ] **Step 1: Modify detect_bifurcation_in_family to log failures**

Replace the `except Exception: continue` at line 410 with:

```python
            except Exception:
                logger.debug("Bifurcation analysis failed for orbit %d", i, exc_info=True)
                continue
```

Add `import logging` and `logger = logging.getLogger(__name__)` at the top of the file if not already present.

After the loop, add a summary:

```python
        n_failed = len(orbits) - len(bifurcation_points)
        if n_failed > 0:
            logger.warning(
                "Bifurcation analysis failed for %d/%d orbits",
                n_failed,
                len(orbits),
            )
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/algorithms/test_stability.py -v --tb=short`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add e2m2e/algorithms/stability.py
git commit -m "fix(algorithms): log failures in detect_bifurcation_in_family

Previously silent exception catches could hide data quality issues.
Now logs debug details per orbit and warning summary."
```

---

## Phase 3: Input Validation and Robustness (M-SEC-06, M-SEC-07, M-06, H-06)

### Task 10: Add max_orbits check in natural_continuation (M-SEC-07)

**Files:**
- Modify: `e2m2e/algorithms/continuation.py:277-336`

- [ ] **Step 1: Add guard inside forward while loop**

After line `i += 1` in the forward loop (around line 336), before the loop continues:

```python
                if len(orbit_family) >= self.max_orbits:
                    self.termination_reason = "达到最大轨道数限制"
                    break
```

- [ ] **Step 2: Add same guard inside backward while loop**

After the corresponding `i += 1` in the backward loop:

```python
                if len(orbit_family) >= self.max_orbits:
                    self.termination_reason = "达到最大轨道数限制"
                    break
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/algorithms/ -v --tb=short -k "continuation"`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add e2m2e/algorithms/continuation.py
git commit -m "fix(algorithms): enforce max_orbits limit in natural_continuation

The max_orbits attribute was set but never checked inside the while
loops, allowing unbounded iteration."
```

---

### Task 11: Add input validation to set_characteristic_scales (M-06)

**Files:**
- Modify: `e2m2e/core/system.py:141-157`

- [ ] **Step 1: Add validation**

At the start of `set_characteristic_scales` (after docstring, before line 148):

```python
        if distance <= 0:
            raise ValueError(f"distance must be positive, got {distance}")
        if period <= 0:
            raise ValueError(f"period must be positive, got {period}")
```

- [ ] **Step 2: Write test**

```python
def test_set_characteristic_scales_rejects_non_positive():
    system = CR3BP_System(mu=0.012, primary="Earth", secondary="Moon")
    with pytest.raises(ValueError, match="distance must be positive"):
        system.set_characteristic_scales(distance=0, period=100)
    with pytest.raises(ValueError, match="period must be positive"):
        system.set_characteristic_scales(distance=100, period=-1)
```

- [ ] **Step 3: Run test**

Run: `pytest tests/core/system/test_system.py::test_set_characteristic_scales_rejects_non_positive -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add e2m2e/core/system.py tests/core/system/test_system.py
git commit -m "fix(core): validate distance and period in set_characteristic_scales

Prevents NaN in unit conversion from zero or negative inputs."
```

---

### Task 12: Fix OrbitFamily.__init__ to use isinstance (H-06)

**Files:**
- Modify: `e2m2e/core/orbit.py:494-508`

- [ ] **Step 1: Replace type() checks with isinstance and add TypeError**

Replace the entire `__init__` body (lines 500-508) with:

```python
        self.orbits: list[Orbit] = []
        if orbits is not None:
            if isinstance(orbits, Orbit):
                self.orbits = [orbits]
            elif isinstance(orbits, list):
                if len(orbits) > 0 and not all(isinstance(o, Orbit) for o in orbits):
                    raise TypeError("All elements in orbits list must be Orbit instances")
                self.orbits = list(orbits)
        self.family_type = family_type
```

- [ ] **Step 2: Write test**

```python
def test_orbit_family_rejects_non_orbit_list():
    with pytest.raises(TypeError, match="Orbit instances"):
        OrbitFamily(orbits=[1, 2, 3])


def test_orbit_family_accepts_single_orbit(sample_orbit):
    family = OrbitFamily(orbits=sample_orbit)
    assert len(family) == 1
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/core/test_orbit.py -v -k "orbit_family"`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add e2m2e/core/orbit.py tests/core/test_orbit.py
git commit -m "fix(core): use isinstance in OrbitFamily.__init__ and reject invalid types

Previously, non-Orbit lists were silently discarded. Now raises
TypeError for invalid inputs."
```

---

### Task 13: Add memory budget check to _compute_min_distance (H-SEC-02)

**Files:**
- Modify: `e2m2e/transfer/transfer_search.py:1026-1044`

- [ ] **Step 1: Add size check before allocation**

Replace the method body:

```python
    def _compute_min_distance(
        self, trajectory_states: np.ndarray, arrival_orbit: Orbit
    ) -> tuple[float, int, int]:
        """计算轨迹到目标轨道的最小距离及最近点（轨迹步、目标轨道采样下标）。"""
        traj_positions = trajectory_states[:, :3]
        orbit_positions = arrival_orbit.states[:, :3]

        n_traj = len(traj_positions)
        n_orbit = len(orbit_positions)
        max_pairs = 10_000_000  # ~240 MB for float64 (n * 3 * 8 bytes)

        if n_traj * n_orbit > max_pairs:
            return self._compute_min_distance_chunked(traj_positions, orbit_positions)

        diff = traj_positions[:, np.newaxis, :] - orbit_positions[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))

        flat_distances = distances.flatten()
        min_flat_idx = np.argmin(flat_distances)
        min_distance = flat_distances[min_flat_idx]

        step_idx = int(min_flat_idx // n_orbit)
        orbit_idx = int(min_flat_idx % n_orbit)

        return min_distance, step_idx, orbit_idx
```

- [ ] **Step 2: Add chunked fallback method**

```python
    def _compute_min_distance_chunked(
        self, traj_positions: np.ndarray, orbit_positions: np.ndarray
    ) -> tuple[float, int, int]:
        """分块计算最小距离，避免内存溢出。"""
        n_traj = len(traj_positions)
        n_orbit = len(orbit_positions)
        chunk_size = max(1, 10_000_000 // n_orbit)

        global_min_dist = float("inf")
        global_step_idx = 0
        global_orbit_idx = 0

        for start in range(0, n_traj, chunk_size):
            end = min(start + chunk_size, n_traj)
            chunk = traj_positions[start:end]

            diff = chunk[:, np.newaxis, :] - orbit_positions[np.newaxis, :, :]
            distances = np.sqrt(np.sum(diff**2, axis=2))

            flat_distances = distances.flatten()
            min_flat_idx = np.argmin(flat_distances)
            min_distance = flat_distances[min_flat_idx]

            if min_distance < global_min_dist:
                global_min_dist = min_distance
                local_step = int(min_flat_idx // n_orbit)
                global_step_idx = start + local_step
                global_orbit_idx = int(min_flat_idx % n_orbit)

        return global_min_dist, global_step_idx, global_orbit_idx
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/transfer/ -v --tb=short`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add e2m2e/transfer/transfer_search.py
git commit -m "fix(transfer): add memory budget check to _compute_min_distance

Falls back to chunked computation when trajectory × orbit pairs
exceed 10 million, preventing out-of-memory crashes on large grids."
```

---

## Summary

| Phase | Tasks | Files Modified | Commits |
|-------|-------|---------------|---------|
| Phase 1: Numerical Safety | 1-4 | dynamics.py, system.py, orbit.py | 4 |
| Phase 2: Code Quality | 5-9 | continuation.py, orbit.py, system.py, config.py, stability.py, pyproject.toml | 5 |
| Phase 3: Validation & Robustness | 10-13 | continuation.py, system.py, orbit.py, transfer_search.py | 4 |

**Total: 13 tasks, ~14 files, 13 commits**

All tasks are independently committable and testable. Phase 1 fixes numerical correctness, Phase 2 fixes observability and maintainability, Phase 3 hardens input boundaries.
