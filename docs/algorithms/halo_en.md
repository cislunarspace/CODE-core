# Halo Orbit Generation and Family Continuation

**Source layout**

| Area | Path |
|------|------|
| Continuation & Halo family | `e2m2e/algorithms/continuation.py` |
| Richardson guess & Halo DC | `e2m2e/algorithms/differential_correction.py` |
| CLI: single / family | `scripts/generate/generate_halo_orbit.py`, `generate_halo_family.py` |
| CLI: plotting | `scripts/plot/plot_halo_orbit.py`, `plot_halo_family.py` |

## Features

1. **Single periodic Halo**: Richardson third-order guess + `DifferentialCorrection` (`setup_halo_orbit_fixed_z0` or fixed-x variants).
2. **Seed orbit**: `Continuation.generate_halo_seed_orbit` — same pipeline, fills `orbit.parameters` (`libration_point`, `amplitude_z`, `halo_class`).
3. **Pseudo-arclength (XZ symmetry)**: `Continuation.pseudo_arclength_continuation` — state vector \(\mathbf{X}=[r_x,r_z,\dot y,T/2]\), aligned with `continuation_PAL_CR3BP` (plane 13) in `CR3BP_MATLAB_Library`.
4. **Halo family wrapper**: `Continuation.halo_pseudo_arclength_continuation` — two branches (positive/negative step), optional `step_size_negative`, MATLAB-like directional increment flags; `dc_scheme` selects differential correction after each PAL step (`adaptive` default for robustness with the Python STM corrector).

## PAL safeguards

Newton step order matches MATLAB (no extra update after \(\|F\|\) is below tolerance); step clipping; optional fallback to Euler predictor if PAL lands on a non-physical root; optional retry of DC with fixed \(z_0\) when `matlab_halo_type1` is used.

## See also

- [Halo (Chinese)](halo.md)  
- [Halo roadmap (ZH)](../ways-of-work/plan/halo-roadmap_zh.md)
