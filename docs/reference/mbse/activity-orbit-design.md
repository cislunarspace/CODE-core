---
title: Orbit Design Activity Diagram
---

# Orbit Design Activity Diagram

```mermaid
flowchart TD
    start([Start])
    sys[Create CR3BP_System]
    dyn[Create CR3BP_Dynamics]
    prop[Propagate initial guess]
    correct[Differential correction]
    cont[Orbit continuation]
    end([Obtain orbit family])
    start --> sys
    sys --> dyn
    dyn --> prop
    prop --> correct
    correct --> cont
    cont --> end
```
