# MetaSTC-J Agent Guide

MetaSTC-J is the journal-extension workspace for MetaSTC, centered on dynamic spatio-temporal heterogeneity.

## Fresh-session read order
1. `AGENTS.md`
2. `docs/RESEARCH_STATE.md`
3. `docs/NEXT_STEPS.md`
4. `docs/DECISIONS.md`
5. `docs/EXPERIMENT_LOG.md`
6. Existing `AGENT.md`, `CONFIG_4090.md`, and project README/code notes as needed.

## Workspace rules
- Primary workspace is `/workspace/MetaSTC-J`.
- Never place MetaSTC-J temporary clones, test directories, or experiment outputs directly under `/workspace`; keep them inside this project.
- Preserve the reproducible conference-version baseline and distinguish it from journal-development code/results.
- Inspect active jobs and the exact code variant before editing or rerunning experiments.
- Do not delete baseline/result artifacts just to clean the tree.

## Journal route
1. Dynamic soft task discovery.
2. Task-conditioned meta-parameter generation/adaptation.
3. Selective test-time adaptation under shift when useful.
4. Evaluation of dynamic heterogeneity, distribution shift, generalization and cross-city behavior.

## State discipline
After meaningful work, update the appropriate file under `docs/`: confirmed state, decisions, experiment summary, or next steps. Exact metrics must come from checked result artifacts.

## Git note
As of 2026-09-02 this directory has a local `.git` but no commit history and no configured remote. Do not assume GitHub synchronization and do not create/bind a remote unless explicitly requested.
