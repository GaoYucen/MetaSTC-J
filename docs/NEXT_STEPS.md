# MetaSTC-J Next Steps

Last updated: 2026-09-05

## P0 — Frozen evidence matrix
- [x] Freeze V14 mechanism implementation and official P0 protocol.
- [x] Freeze seeds `42–46` and matched controls: Static MetaSTC, `linear_cluster`, `constant_gate`, `shared_no_task`, V14.
- [x] Complete Beijing five-seed attribution study.
- [ ] Complete the same five-seed matched-control matrix for Shanghai epoch60.
- [ ] Complete the same five-seed matched-control matrix for LargeST epoch60.
- [ ] Report mean ± std, per-seed wins, MSE, and paired significance under identical model-selection/training budgets.

## P0 — Mechanism attribution: when does adaptation help?
- [ ] Record per-window static error, V14 gain, linear-control gain, gate value, and a predefined shift statistic.
- [ ] Stratify test windows by shift strength / hard temporal segments.
- [ ] Test whether V14 gains concentrate in regimes where the static anchor fails or distribution shift is stronger.
- [ ] Quantify correlation/calibration between gate, shift, static error, and realized adaptation benefit.

## P0 — Go / No-Go checkpoint
Proceed with the dynamic-heterogeneity journal story only if V14/context gating has clear additional value over simple matched controls, especially under shift/hard regimes, with interpretable mechanism evidence.

If `linear_cluster` or other simple controls remain equal/better across ordinary and high-shift settings, **stop before creating V15/V16 merely to chase average MAE**. Revisit or simplify the research hypothesis.

## P1 — Only after P0 passes
- [ ] Extend the stable mechanism to FiLM to show the method is not LSTM-specific.
- [ ] Define explicit temporal/distribution-shift protocols.
- [ ] Add cross-city/generalization experiments.
- [ ] Report worst-segment/tail MAE, adaptation frequency, parameter/runtime overhead, and failure cases.

## Automation boundary
Automated continuation may run the frozen P0 experiment matrix, aggregate results, fill missing seeds/datasets, and execute the predefined gate/shift analysis. It must stop for human research review before architecture redesign, hypothesis changes, protocol/data-split/objective changes, new model versions, or entry into P1.
