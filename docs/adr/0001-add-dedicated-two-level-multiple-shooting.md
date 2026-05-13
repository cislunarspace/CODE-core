# Add dedicated TwoLevelMultipleShooting

We will add `TwoLevelMultipleShooting` as a dedicated algorithm in `e2m2e.algorithms` instead of extending or inheriting from the existing `MultipleShooting`. The two-level correction has different free variables, residuals, Jacobian structure, and result diagnostics from the existing full-state multiple shooting solver; keeping it separate preserves the simpler generic API while giving transfer design code a stable API for the legacy two-level ephemeris correction semantics.
