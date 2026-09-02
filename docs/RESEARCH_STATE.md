# MetaSTC-J Research State

Last updated: 2026-09-02

## Confirmed state
- The original MetaSTC experimental behavior has been reproduced closely enough to serve as the journal-development baseline.
- Intended evaluation datasets include Beijing, Shanghai, and LargeST.
- The journal extension focuses on dynamic spatio-temporal heterogeneity rather than stacking small independent modules.

## Current technical route
1. Dynamic soft task discovery to model context-dependent latent task structure.
2. Task-conditioned meta-parameter generation/adaptation using the inferred task/context representation.
3. Selective test-time adaptation as a supporting mechanism under distribution shift.
4. Evaluate standard MAE/RMSE together with shift/generalization/cross-city behavior.

## Workspace state
- Primary workspace: `/workspace/MetaSTC-J`.
- Local `.git` exists, but inspection on 2026-09-02 showed no commits and no `origin` remote.
- Actual code/results remain local to the 4090 workspace until a remote is explicitly configured.

## Cross-device source of truth
Use this file with `DECISIONS.md`, `EXPERIMENT_LOG.md`, and `NEXT_STEPS.md` to resume work without relying on a previous Codex session. Notion contains the human-facing journal plan and research conclusions.
