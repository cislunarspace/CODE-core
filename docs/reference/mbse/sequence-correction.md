---
title: 修正序列
---

# 修正序列

```mermaid
sequenceDiagram
    participant Client
    participant DifferentialCorrection
    participant RustCore
    participant Dynamics
    Client->>DifferentialCorrection: setup_*(...)
    DifferentialCorrection->>DifferentialCorrection: _apply_config(config): free_variable_indices, constraint_indices
    Client->>DifferentialCorrection: iterate_correction(initial_guess)
    DifferentialCorrection->>RustCore: differential_correction_cr3bp_py(state, half_period, ...)
    Note over RustCore: 残差、STM 雅可比、Newton 修正与收敛判定
    RustCore->>DifferentialCorrection: solution_state, solution_time, error_history
    DifferentialCorrection->>Dynamics: propagate(solution_state, (0, period))
    Dynamics->>DifferentialCorrection: states
    DifferentialCorrection->>Client: DifferentialCorrectionResult(orbit)
```
