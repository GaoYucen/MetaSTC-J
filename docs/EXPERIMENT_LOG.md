# MetaSTC-J Experiment Log

Keep this file concise. Raw logs, checkpoints, predictions and intermediate artifacts stay in project-local result directories.

## 2026-09-02 — Conference-version reproduction baseline
The original MetaSTC behavior was reproduced closely enough to serve as the journal-development baseline. Frozen baseline/reference artifacts should not be overwritten by journal experiments.

## 2026-09-02 to 2026-09-04 — Dynamic route evolution
- Direct dynamic soft routing did not reliably beat the static hard baseline.
- Continuous/task-conditioned variants produced some validation gains but exposed temporal-regime overfitting.
- Static-anchored residual correction was safer and produced consistent single-run improvements.
- Residual-control experiments showed that simple calibration can explain a substantial fraction of the gains; therefore mechanism attribution is mandatory.

## 2026-09-04 — V14 single-run official-protocol signal
V14 (Static MetaSTC Anchor + context-conditioned gated residual meta-adapter) improved MAE over strong static reproduction in the recorded single-run official LSTM L=12 experiments by approximately:
- Beijing: `-1.55%`
- Shanghai epoch60: `-2.66%`
- LargeST epoch60: `-4.74%`

These results justify continuing V14 as a candidate but are not sufficient mechanism evidence.

## 2026-09-04/05 — Beijing V14 attribution, five seeds
Seeds `42–46`, matched attribution variants:

| Variant | MAE mean ± std | Relative MAE vs static | Beats static |
|---|---:|---:|---:|
| V14 | 3.477888 ± 0.023204 | -0.604% | 4/5 |
| constant_gate | 3.494879 ± 0.028843 | -0.118% | 1/5 |
| shared_no_task | 3.560092 ± 0.011238 | +1.746% | 0/5 |
| linear_cluster | 3.469372 ± 0.000561 | -0.847% | 5/5 |

Interpretation: V14 is not yet superior to the strongest simple matched control (`linear_cluster`). The next experiment is not a new model version; it is the same frozen matrix on Shanghai/LargeST plus per-window shift/gate/gain attribution.

## 2026-09-05 — P0 freeze
Frozen implementation/protocol commit: `2a50bc0645a589652e238af5ee133448f5d4e1da`

Frozen tag: `metastc-j-p0-v14-freeze-20260905`

See `docs/JOURNAL_P0_FREEZE.md` for exact P0 scope, controls, decision rule, and implementation SHA256 values.


## 2026-09-05 post-freeze operational fix — dataset-native cluster counts
The failed P0-01 job stopped at Shanghai because the sweep assumed 5 clusters globally. Frozen checkpoint configs and labels show Beijing=5, Shanghai=3, LargeST=3. The sweep was minimally repaired to use those native counts. No research hypothesis, architecture, data split, objective, training budget, matched-control set, or evaluation protocol changed. P0-01 can resume from existing per-seed outputs.
