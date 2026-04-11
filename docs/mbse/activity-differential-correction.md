```mermaid
flowchart TD
    start([开始修正])
    config[加载 CorrectionConfig 策略]
    propagate[传播半周期 (with_stm=True)]
    error[计算约束误差向量]
    check[收敛?]
    update[Newton 更新自由变量]
    end([返回收敛轨道])
    start --> config
    config --> propagate
    propagate --> error
    error --> check
    check --> update
    update --> end
    check{收敛?}
    check -->|是 (error < tol)| end
    check -->|否| update
```
