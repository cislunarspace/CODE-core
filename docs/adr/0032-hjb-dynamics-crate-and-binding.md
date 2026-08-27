# ADR 0032: HJB dynamics in a new crate and binding-layer generic entry

**Status**: Adopted
**Date**: 2026-08-21
**Related Issue**: #497
**Related**: ADR 0011 (five-layer architecture), ADR 0012 (dependency
direction), ADR 0013 (verification by definition), ADR 0016 (EphemCache)

## Context

After completing the ToolboxLS port, e2m2e-levelset carried only example
Hamiltonians (constant advection, Burgers, double integrator). geo-nrho's two-
level dynamic programming route requires plugging Earth-Moon synodic CR3BP
dynamics into the HJB solver (#497), with an explicit later deepening sequence:
5D mass-carrying, solar third body, ephemeris force models. This is the first
original dynamics entering that subsystem — code ownership, license boundary,
binding shape, and dimension ceiling must be fixed before work starts, or every
deepening reworks. The full system shape & division live in
docs/architecture/hjb-subsystem.md; this entry only fixes decisions.

## Decision

1. Original HJB dynamics live in the new crate **e2m2e-hjb-dynamics**, licensed
   under workspace Apache-2.0. e2m2e-levelset keeps only ToolboxLS
   counterparts.
2. Dynamics enter the solver via the `Hamiltonian` trait; parameters are fields
   of implementing structs — no callbacks, no Python inside Rust hot loops.
3. Python bindings use a single generic entry: dynamics identifier + parameter
   table + grid definition + terminal conditions. No per-dynamics binding
   functions. The binding layer validates key existence and values; invalid
   input raises explicit errors.
4. Grid-layer state dimensions cap at five. Three-dimensional high-fidelity
   problems (six dims + time) belong to downstream two-level neural-network
   tiers, never solved on structured grids.
5. State semantics documented: standard nondimensional synodic frame; angular
   speed identically 1, not a parameter; μ fixed at construction; state order
   (x, y, vx, vy); four-dimensional states lift to three-dimensional as z = vz = 0
   sections.

## Rationale

1. e2m2e-levelset wholesale inherits ToolboxLS's ACM non-commercial license;
   merging original dynamics would sweep them under it, damaging availability as
   general infrastructure. An independent crate also preserves levelset's
   faithful-porting positioning — each module maintainable against MATLAB
   originals.
2. The Hamiltonian trait is ToolboxLS's function-handle protocol naturalized
   (see crate README's protocol mapping table), proven sufficient through four
   verification phases. Dimension-agnostic with explicit time parameter:
   mass-carrying and non-autonomous deepening don't touch the seam.
3. The counterexample is solve_planar_lowthrust_hjb_py: signature pinned to a
   double integrator. Following that pattern, every new family member bumps the
   ABI stamp again and maintains another dedicated signature. The generic entry
   pins ABI change to the single addition of the entry itself.
4. At 40 nodes/dim, six-dimensional grids need ~33 GB per array — the dimension
   cap is arithmetic fact. Two-tier division is the demand side's settled plan
   (geo-nrho architecture doc §1.2/3.1); this entry promotes it from downstream
   convention to e2m2e-side architectural constraint.
5. Level-two networks train on level-one solutions; both sides' interpretations
   of state order, nondimensionalization, and frames must match verbatim.
   Conventions living only in code will misalign when mission trajectory data
   arrives.

## Consequences

- Added: crate e2m2e-hjb-dynamics (landed with #497; first member four-
  dimensional planar CR3BP); docs/architecture/hjb-subsystem.md.
- Changed: e2m2e-integrators gains generic HJB solving bindings; ABI stamp
  increments.
- Unchanged: e2m2e-levelset module structure & licensing; ADR 0012/0016 rules.
- Costs: generic bindings' parameter tables are weakly typed at FFI boundaries;
  misspelled keys surface only at runtime, mitigated by binding-layer validation
  + explicit errors; one more crate scaffold to maintain.

## Revision (2026-08-21, #497 implementation review)

Consequences add distribution implications: from #497,
e2m2e-integrators depends on e2m2e-levelset, so released wheels contain ACM
non-commercial licensed code. Per e2m2e-levelset README's license section,
future release notes must state this difference; commercial use requires
contacting ToolboxLS's original author.
