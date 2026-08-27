# ADR 0023: SciPy propagation exception for explicit event inputs

**Status**: Adopted
**Date**: 2026-08-11
**Related Issue**: #378

## Context

CR3BP and BCR4BP routine propagation already routes through the Rust path. The
Rust propagation functions don't carry the SciPy event-interface semantics for
these dynamics classes: event function attributes, augmented-STM event states,
and termination results must preserve the existing SciPy contract. Meanwhile a
missing Rust extension must not be treated as ordinary runtime degradation,
which would mask build problems.

## Decision

CR3BP and BCR4BP use SciPy propagation only when callers explicitly pass
`events`. This is a deliberate exception triggered by the `events` input,
independent of Rust extension availability; availability changes never alter
this dispatch rule.

Without `events`, propagation must require the corresponding Rust extension
symbols. Missing extensions or symbols raise
`RustExtensionUnavailableError` explicitly — no SciPy fallback. ForceModel's
event interface remains explicitly unsupported for now, because compiled-forces
doesn't yet provide an event-propagation API; the underlying
`e2m2e.integrators.solve_ivp_events` is provided and tested independently for
Rust event refinement.

## Consequences

Explicit-event calls keep SciPy's event semantics and result fields; plain
propagation keeps the Rust numeric path with explicit failure on missing
extensions. Dispatch reasons in both scenarios are directly decidable from
input arguments; extension-missingness and event exceptions no longer get
conflated.

## Revision (2026-08-23)

This page's dispatch rule was revised by ADR 0020 decision 4: passing `events`
no longer auto-dispatches to SciPy — event paths require callers to pass
explicitly `backend='scipy'` or `backend='rust'` (the latter via Rust
`solve_ivp_events`); passing `events` without `backend` raises `ValueError`
directly. The SciPy event semantics described above hold only under
`backend='scipy'`.
