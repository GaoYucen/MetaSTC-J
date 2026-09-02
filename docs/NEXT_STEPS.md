# MetaSTC-J Next Steps

Last updated: 2026-09-02

## P0 — Baseline freeze and record
- [ ] Locate the exact reproducible baseline code/config/results.
- [ ] Record verified Beijing, Shanghai, and LargeST MAE/RMSE.
- [ ] Ensure journal experiments cannot overwrite the frozen baseline.

## P0 — Dynamic soft task discovery
- [ ] Identify the current implementation entry point.
- [ ] Compare static/hard versus soft/dynamic task structure under consistent settings.
- [ ] Verify inferred task structure actually changes meaningfully with spatio-temporal context.

## P0 — Task-conditioned adaptation
- [ ] Condition meta-parameters/adaptation on inferred task/context information.
- [ ] Compare against shared/global parameterization.
- [ ] Check consistency across datasets/cities.

## P1 — Selective test-time adaptation and shift
- [ ] Define when adaptation should be triggered.
- [ ] Compare always-adapt, never-adapt and selective-adapt variants.
- [ ] Add distribution-shift and cross-city/generalization evaluation.

A fresh Codex session should read `AGENTS.md`, `RESEARCH_STATE.md`, and this file before taking the next unchecked P0 item.
