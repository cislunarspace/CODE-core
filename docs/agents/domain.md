# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`CONTEXT-MAP.md`** at the repo root — this points to the per-context `CONTEXT.md` files.
- Read the `CONTEXT.md` for the specific context you're working in.
- **`docs/adr/`** under each relevant context — read ADRs that touch the area you're about to work in.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The producer skill (`/grill-with-docs`) creates them lazily when terms or decisions actually get resolved.

## File structure

Multi-context repo:

```
/
├── CONTEXT-MAP.md          ← points to per-context docs
├── frontend/
│   ├── CONTEXT.md
│   └── docs/adr/
│       └── 0001-state-management.md
├── backend/
│   ├── CONTEXT.md
│   └── docs/adr/
│       └── 0002-api-versioning.md
└── shared/
    ├── CONTEXT.md
    └── docs/adr/
        └── 0003-error-handling.md
```

Use `CONTEXT-MAP.md` to discover which contexts exist and which one covers the code you're about to touch.

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in the relevant `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal — either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/grill-with-docs`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0001 (CR3BP frame choice) — but worth reopening because…_
