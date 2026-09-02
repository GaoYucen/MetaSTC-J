# Dynamic routing V1 smoke tests — 2026-09-02

Baseline: frozen `dev-260902-reproduce`, Beijing + LSTM, existing five cluster-adapted experts.

## V1 naive context-prototype soft routing
On 4096 test window-road samples:
- static hard MAE: 3.6484
- best dynamic soft MAE: 3.8990
- relative change: +6.87%
- routing became nearly uniform across five tasks (effective tasks ≈ 5)

Conclusion: raw current-window-to-task-prototype distance is not a useful routing signal for the existing static experts.

## V1.1 learned residual gate
Three-epoch smoke test, 10 training batches, 2 validation/test batches:
- static hard test MAE: 3.6484
- learned dynamic soft test MAE: 3.6547
- relative change: +0.17%
- mean alpha fell from initial 0.05 to about 0.036
- dynamic argmax never left the original static task

Conclusion: when the old static cluster experts are frozen, the learned gate prefers to fall back to static routing. Dynamic task discovery therefore needs to be trained jointly with task-specific adaptation rather than mixing already-specialized static experts after the fact.

Next: V2 joint dynamic soft task discovery + task-conditioned adaptation from the common/global initialization.
