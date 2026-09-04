# MetaSTC-J Decision Log

## 2026-09-02 — Target dynamic heterogeneity
Move the journal problem from relatively static task heterogeneity toward dynamic spatio-temporal heterogeneity; this provides a clearer journal-level new problem than incremental static-clustering changes.

## 2026-09-02 — Initial hypothesis: dynamic soft task discovery
The initial route explored context-dependent soft task discovery and task-conditioned adaptation. Subsequent experiments showed that explicit K-way routing was not reliably identified and could collapse toward uniform/static behavior.

## 2026-09-04/05 — Reposition task discovery
Do **not** use hard-to-soft K-way task discovery as the journal's main claimed contribution unless future evidence establishes meaningful regimes. Reframe the latent state as a continuous context-conditioned task/state representation.

## 2026-09-04/05 — Current main candidate: static-anchored gated residual meta-adaptation
Preserve strong Static MetaSTC and learn only a conservative context-conditioned residual correction controlled by a gate. This gives the journal extension a safer inheritance path from the conference model.

## 2026-09-05 — Mechanism evidence is the current bottleneck
Beijing five-seed attribution shows V14 is better than shared/no-task and usually better than constant-gate controls, but the simple `linear_cluster` control currently has lower average MAE and lower variance. Therefore ordinary average-MAE improvement is insufficient to claim dynamic adaptation.

## 2026-09-05 — Shift/hard-regime evidence decides the story
The decisive test is whether context-conditioned V14 is substantially more useful under measurable temporal/distribution shift or hard segments, with gate/context statistics explaining when adaptation helps. If simple controls remain equal/better in those settings, downgrade or redesign the story rather than iterate V15/V16 solely for metric chasing.

## 2026-09-05 — Freeze P0 before automation
Freeze implementation, seeds, controls, data/protocol, and Go/No-Go criteria before allowing automated continuation. Automation may execute predefined P0 evidence collection but must stop before hypothesis/architecture changes or P1.

## Persistent handoff
Use `docs/JOURNAL_P0_FREEZE.md`, `RESEARCH_STATE.md`, `NEXT_STEPS.md`, `DECISIONS.md`, and `EXPERIMENT_LOG.md` as the cross-device/session handoff layer. Notion is the human-facing research record; the frozen GitHub files define the executable protocol.


## 2026-09-05 post-freeze operational fix — dataset-native cluster counts
The failed P0-01 job stopped at Shanghai because the sweep assumed 5 clusters globally. Frozen checkpoint configs and labels show Beijing=5, Shanghai=3, LargeST=3. The sweep was minimally repaired to use those native counts. No research hypothesis, architecture, data split, objective, training budget, matched-control set, or evaluation protocol changed. P0-01 can resume from existing per-seed outputs.
