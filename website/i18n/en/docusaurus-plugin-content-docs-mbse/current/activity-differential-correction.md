```mermaid
flowchart TD
    start([Start correction])
    config[Load CorrectionConfig strategy]
    propagate[Propagate half period (with_stm=True)]
    error[Compute constraint error vector]
    check[Converged?]
    update[Newton update free variables]
    end([Return converged orbit])
    start --> config
    config --> propagate
    propagate --> error
    error --> check
    check --> update
    update --> end
    check{Converged?}
    check -->|Yes (error < tol)| end
    check -->|No| update
```
