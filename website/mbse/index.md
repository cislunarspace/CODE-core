---
title: MBSE Models
---

# MBSE Models

Model-Based Systems Engineering (MBSE) models for e2m2e, built with SysML-style Protocol interfaces, Pydantic data models, and requirement traceability.

## Architecture

| Document | Description |
|----------|-------------|
| [System Overview](system-overview) | Block Definition Diagram (BDD) and component architecture |
| [BDD: Core](bdd-core) | Core module block definitions — SystemModel, EOMProvider, Propagator |
| [BDD: Algorithms](bdd-algorithms) | Algorithm module block definitions — CorrectorStrategy, Continuation, Stability |
| [Requirements](requirements) | Functional requirements registry with traceability |

## Activity Diagrams

| Document | Description |
|----------|-------------|
| [Orbit Design Activity](activity-orbit-design) | End-to-end orbit design workflow |
| [Differential Correction Activity](activity-differential-correction) | Correction iteration lifecycle |

## Sequence Diagrams

| Document | Description |
|----------|-------------|
| [Propagation Sequence](sequence-propagation) | State propagation call chain |
| [Correction Sequence](sequence-correction) | Differential correction message flow |

## State Machines

| Document | Description |
|----------|-------------|
| [Orbit Lifecycle](state-orbit-lifecycle) | Orbit state transitions |
| [Convergence State](state-convergence) | Correction convergence states |

## Traceability

| Document | Description |
|----------|-------------|
| [Traceability Matrix](traceability-matrix) | Requirements-to-components mapping |
