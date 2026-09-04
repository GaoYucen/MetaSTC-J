# MetaSTC-J Research State

Last updated: 2026-09-05

## Confirmed state
- The original MetaSTC experimental behavior has been reproduced closely enough to serve as the journal-development baseline.
- Intended evaluation datasets are Beijing, Shanghai, and LargeST.
- The journal extension targets **dynamic spatio-temporal heterogeneity**, but current evidence does not justify claiming discrete dynamic task discovery as the core mechanism.
- The official frozen P0 implementation/protocol is documented in `docs/JOURNAL_P0_FREEZE.md`.
- Frozen implementation tag: `metastc-j-p0-v14-freeze-20260905`.
- Frozen implementation commit: `2a50bc0645a589652e238af5ee133448f5d4e1da`.

## Current technical route
1. Preserve the conference MetaSTC model as a strong **Static MetaSTC Anchor**.
2. Represent the current state with a continuous context-conditioned latent task/state representation instead of forcing fixed K-way task discovery.
3. Use a **context-conditioned gated residual meta-adapter** to make conservative dynamic corrections:
   `y_hat = y_static + gate(context) * residual(context, task/state)`.
4. Treat selective/shift-aware adaptation as a mechanism whose value must be demonstrated under measurable temporal/distribution shift, not merely as an extra module.
5. Evaluate not only average MAE/MSE but also when adaptation helps: high-shift segments, hard temporal windows, cross-city/generalization, tail performance, and adaptation cost.

## Latest mechanism evidence
The Beijing five-seed attribution study (seeds 42–46) currently shows:
- V14: MAE `3.477888 ± 0.023204`, `-0.604%` vs static, beats static in `4/5` seeds.
- constant_gate: `3.494879 ± 0.028843`, `-0.118%`, beats static in `1/5`.
- shared_no_task: `3.560092 ± 0.011238`, `+1.746%`, beats static in `0/5`.
- linear_cluster: `3.469372 ± 0.000561`, `-0.847%`, beats static in `5/5`.

Therefore V14 remains a candidate, but **context-conditioned dynamic adaptation has not yet been proven superior to simple matched residual controls**. The key P0 question is whether V14 becomes meaningfully better in high-shift / hard-regime settings and whether gate/context statistics explain those gains.

## Workspace and repository state
- Primary workspace: `/workspace/MetaSTC-J`.
- Active branch: `dev-260902-dynamic`.
- Remote: `git@github.com:GaoYucen/MetaSTC-J.git`.
- The 2026-09-05 freeze commit and tag are pushed to GitHub.

## Cross-device source of truth
A fresh ChatGPT/Codex/Work session should read, in order:
1. `AGENTS.md`
2. `docs/JOURNAL_P0_FREEZE.md`
3. `docs/RESEARCH_STATE.md`
4. `docs/NEXT_STEPS.md`
5. `docs/DECISIONS.md`
6. `docs/EXPERIMENT_LOG.md`

Notion contains the human-facing journal plan and research conclusions. GitHub/server files define the executable frozen P0 protocol.
