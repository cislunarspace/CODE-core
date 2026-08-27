---
title: MBSE Models
---

# MBSE Models

e2m2e's Model-Based Systems Engineering (MBSE) model is built around component
registration, requirement traceability, Pydantic data models, and Mermaid
diagram generation. ADR 0001 withdrew the decorative Protocol seams; the current
polymorphism seam rests on the `Dynamics` base class.

```{toctree}
:hidden:

system-overview
generated/bdd-data
generated/bdd-numerical
generated/bdd-algorithm
generated/bdd-api
generated/bdd-tools
generated/requirements
activity-orbit-design
activity-differential-correction
sequence-propagation
sequence-correction
state-orbit-lifecycle
state-convergence
generated/traceability-matrix
```

## Architecture

| Document | Description |
|------|------|
| [System overview](system-overview) | Block definition diagrams (BDD) & component architecture |
| [BDD: data layer](generated/bdd-data) | Data containers & kernel management components |
| [BDD: numerical layer](generated/bdd-numerical) | Rust numerical-computation facade |
| [BDD: algorithm layer](generated/bdd-algorithm) | Dynamics, correction & continuation components |
| [BDD: interface layer](generated/bdd-api) | Facade, CLI & MCP interfaces |
| [BDD: tools layer](generated/bdd-tools) | Auxiliary tools such as logging |
| [Functional requirements](generated/requirements) | Requirement registry with code traceability |

## Activity diagrams

| Document | Description |
|------|------|
| [Orbit design activity](activity-orbit-design) | End-to-end orbit design workflow |
| [Differential correction activity](activity-differential-correction) | Correction-iteration lifecycle |

## Sequence diagrams

| Document | Description |
|------|------|
| [Propagation sequence](sequence-propagation) | State-propagation call chain |
| [Correction sequence](sequence-correction) | Differential-correction message flow |

## State machines

| Document | Description |
|------|------|
| [Orbit lifecycle](state-orbit-lifecycle) | Orbit state transitions |
| [Convergence states](state-convergence) | Correction convergence states |

## Traceability

| Document | Description |
|------|------|
| [Traceability matrix](generated/traceability-matrix) | Requirements ↔ code & tests mapping |
