```mermaid
flowchart TD
    start([开始])
    sys[创建 CR3BP_System]
    dyn[创建 CR3BP_Dynamics]
    prop[传播初始猜测]
    correct[微分修正]
    cont[轨道延续]
    end([获得轨道族])
    start --> sys
    sys --> dyn
    dyn --> prop
    prop --> correct
    correct --> cont
    cont --> end
```
