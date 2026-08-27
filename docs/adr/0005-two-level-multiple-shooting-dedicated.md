# ADR 0005: TwoLevelMultipleShooting as an independent algorithm

**Status**: Adopted
**Date**: 2026-05-13
**Numbering note**: back-filled historical decision; actual decision time falls
between ADR 0001 and 0002.

Add `TwoLevelMultipleShooting` to `e2m2e.algorithms` as an independent
algorithm rather than extending or subclassing the existing
`MultipleShooting`. The two-level correction differs from the existing
full-state multiple-shooting solver in free variables, residuals, Jacobian
structure, and result diagnostics; keeping it separate preserves the general
solver's simpler API while giving transfer-design code a stable set of APIs
carrying the original two-level ephemeris correction semantics.

## Revision (2026-08-13, related: ADR 0006 revision)

`TwoLevelMultipleShooting` along with the `ephemeris_correction` dispatch
subpackage was deleted: the design chain unified on Rust multiple shooting
(``multiple_shooting_correct_py``, the default path for segmented and stable
orbits), with velocity continuity converged Rust-side via ``vel_weight``
weighting. This ADR's independent-algorithm arrangement has no consumers left;
its decision object is withdrawn. `MultipleShooting` itself is retained
(still used by transfer/hohmann and other non-design chains).
