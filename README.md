# MetaSTC-J — MetaSTC Journal Extension

> Last updated: **2026-09-07**  
> Active branch: **`dev-260902-dynamic`**  
> Current phase: **journal manuscript consolidation / student handoff**  
> Frozen handoff commit: **`30ac60507ac5c52e7251309d8f68f0f3d73ae7dd`**

MetaSTC-J is the journal extension of the ICDM 2024 MetaSTC work. The current experimental exploration has been **closed and frozen**. The default next step is no longer open-ended architecture tuning; it is to consolidate the paper, verify all claims against the frozen evidence, and add only genuinely necessary reviewer-facing experiments after PI review.

## Start here

Students taking over this project should read the following files in order:

1. [`STUDENT_HANDOFF.md`](STUDENT_HANDOFF.md)
2. [`docs/STUDENT_HANDOFF_20260906.md`](docs/STUDENT_HANDOFF_20260906.md)
3. [`results/journal_handoff_20260906/unified_results.md`](results/journal_handoff_20260906/unified_results.md)
4. [`docs/JOURNAL_P0_FREEZE.md`](docs/JOURNAL_P0_FREEZE.md)
5. [`results/journal_handoff_20260906/lstm_matched_controls/summary.md`](results/journal_handoff_20260906/lstm_matched_controls/summary.md)
6. [`manuscript/TKDE_MetaSTC_v14/TKDE_MetaSTC_Journal.tex`](manuscript/TKDE_MetaSTC_v14/TKDE_MetaSTC_Journal.tex)

The frozen result package is located at:

```text
results/journal_handoff_20260906/
```

The current editable journal manuscript is located at:

```text
manuscript/TKDE_MetaSTC_v14/TKDE_MetaSTC_Journal.tex
```

## Current research route

The journal extension preserves the conference MetaSTC model as a strong **Static MetaSTC Anchor** and adds a conservative context-conditioned residual adaptation mechanism:

```text
y_hat = y_static + gate(context) * residual(context, task/state)
```

The intended interpretation is a **continuous context-conditioned latent task/state representation**, rather than a claim that discrete dynamic task discovery has been established.

Current frozen implementations:

- LSTM mechanism: `model_code/dynamic_residual_mechanism_v14.py`
- LSTM matched-control sweep: `model_code/dynamic_residual_mechanism_sweep_v14.py`
- FiLM realization: `model_code/dynamic_residual_meta_adapter_film_v14.py`

Do not silently modify the frozen V14 mechanism. Architecture changes require a new research decision and a new version.

## Latest paper-level results

The authoritative summary is [`results/journal_handoff_20260906/unified_results.md`](results/journal_handoff_20260906/unified_results.md). Results below use seeds **42–46**. Negative Δ means the journal V14 result is better than the corresponding ICDM 2024 MetaSTC result.

| Realization | Dataset | ICDM MAE | Journal MAE mean ± std | ΔMAE | MAE wins | ICDM MSE | Journal MSE mean ± std | ΔMSE | MSE wins |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LSTM | Beijing | 3.534 | **3.5109 ± 0.0191** | **-0.65%** | **5/5** | 27.433 | **27.2152 ± 0.2567** | **-0.79%** | 4/5 |
| LSTM | Shanghai | 4.524 | **4.2654 ± 0.0177** | **-5.72%** | **5/5** | 42.380 | **39.3072 ± 0.2180** | **-7.25%** | **5/5** |
| LSTM | LargeST | 4.644 | **4.4582 ± 0.0328** | **-4.00%** | **5/5** | 45.520 | **44.6292 ± 0.4482** | **-1.96%** | **5/5** |
| FiLM | Beijing | 3.367 | 3.3975 ± 0.0173 | +0.91% | 0/5 | 26.893 | **26.7645 ± 0.1578** | **-0.48%** | 4/5 |
| FiLM | Shanghai | 4.018 | 4.0368 ± 0.0435 | +0.47% | 1/5 | 37.076 | **36.2305 ± 0.2833** | **-2.28%** | **5/5** |
| FiLM | LargeST | 4.369 | **4.2718 ± 0.0154** | **-2.22%** | **5/5** | 43.333 | **43.0410 ± 0.2948** | **-0.67%** | 4/5 |

### Main performance conclusion

**LSTM V14 is the primary journal result.** Its five-seed mean MAE and MSE both improve over ICDM 2024 MetaSTC+LSTM on Beijing, Shanghai, and LargeST. More importantly, all **15 dataset-seed MAE runs** (3 datasets × 5 seeds) beat the corresponding conference-paper value.

This is sufficiently stable to serve as the main performance evidence. The project should not return to open-ended LSTM tuning merely to enlarge the numerical gains.

### FiLM conclusion

FiLM is supporting evidence that the framework transfers to another realization/backbone, but the gain is **dataset- and metric-dependent**:

- LargeST improves both MAE and MSE across five-seed means.
- Beijing and Shanghai improve mean MSE, but mean MAE is slightly worse than the conference FiLM result by about 0.91% and 0.47%, respectively.
- Therefore the paper may claim cross-realization applicability, but **must not claim universal performance improvement for FiLM**.

## Mechanism evidence and claim boundary

The matched-control experiments are essential for interpreting V14 correctly.

Current evidence does **not** support a universal claim that richer context-conditioned adaptation always dominates simpler residual controls:

- On **Beijing**, `linear_cluster` has lower matched-control mean MAE than V14.
- On **LargeST**, `shared_no_task` has lower matched-control mean MAE than V14.
- **Shanghai** is the clearest positive mechanism case: for V14 seed 42, V14 beats the linear control on all 40 test windows, and the high-static-error tercile shows about **0.0928 additional MAE benefit** over the linear control.

The journal story should therefore emphasize **selective/context-dependent value under difficult regimes**, while reporting the Beijing and LargeST counterevidence transparently.

### Parameter/runtime note

V14 adds roughly **10.9K trainable parameters**. The absolute parameter count is small, but the original LSTM experts are themselves very small, so the manuscript should **not** describe the parameter overhead as “negligible”. Batched inference measurements on the 4090 did not show a substantive latency increase.

## Official paper protocol and targets

The journal comparison must use the same paper protocol rather than internal smoke tests or unrelated temporal-holdout baselines:

- input length: `L = 12`
- prediction horizon: `P = 6`
- train/test split: `8:2`
- seeds for frozen journal evidence: `42, 43, 44, 45, 46`
- primary metrics: MAE and MSE

Correct ICDM 2024 L=12 targets:

| Dataset | MetaSTC+LSTM MAE/MSE | MetaSTC+FiLM MAE/MSE |
|---|---:|---:|
| Beijing | 3.534 / 27.433 | 3.367 / 26.893 |
| Shanghai | 4.524 / 42.380 | 4.018 / 37.076 |
| LargeST | 4.644 / 45.520 | 4.369 / 43.333 |

> **Important FiLM warning:** some raw FiLM `metrics.json` files contain a legacy `paper_target_l12` field inherited from the LSTM experiment path. Do **not** use that field for paper comparison. Use `unified_results.*` and the FiLM targets above.

## Frozen experiment assets

### Static MetaSTC anchors

The static anchors required by the frozen code are committed at their expected runtime paths:

```text
param/4090_tuned/lstm/beijing/
param/4090_tuned/epoch60/lstm/shanghai/
param/4090_tuned/epoch60/lstm/largest/

param/4090_tuned/film/beijing/
param/4090_tuned/film/shanghai/
param/4090_tuned/film/largest/
```

Each directory contains the corresponding configuration, cluster labels, global/cluster checkpoints, metrics, and training log needed for reproducibility.

### Journal V14 evidence

```text
results/journal_handoff_20260906/
├── unified_results.md / .csv / .json
├── lstm_matched_controls/
│   ├── beijing/
│   ├── shanghai/
│   ├── largest/
│   ├── p0_02_shift_gate_gain/
│   └── summary.md / summary.json
├── film_five_seed/
│   └── seed_42 ... seed_46/
└── diagnostics/
```

The package contains per-seed metrics, final adapters, matched controls, logs, and diagnostic evidence. This package is the preferred source for all paper-facing numbers.

## Manuscript assets

The current TKDE manuscript package is in:

```text
manuscript/TKDE_MetaSTC_v14/
```

It contains:

- `TKDE_MetaSTC_Journal.tex` — current editable journal version
- `TKDE_MetaSTC_Journal_review.pdf` — compact review PDF
- `TKDE_MetaSTC_original_20260906.tex` — preserved source snapshot
- `traffic-prediction.bib`
- IEEE template files
- figures actually referenced by the manuscript

LaTeX build products such as `.aux`, `.log`, `.fls`, `.fdb_latexmk`, and `synctex` files are intentionally not part of the handoff package.

## What students should do next

1. Build the final paper-facing LSTM + FiLM tables directly from `unified_results.*`; do not retrain merely to recreate existing numbers.
2. Use the three-dataset LSTM stability as the main performance result and FiLM as supporting cross-realization evidence.
3. Integrate the Shanghai positive mechanism case together with the Beijing/LargeST matched-control counterexamples.
4. Verify every number, table, figure, setting, and claim in the manuscript against the frozen result package.
5. Compile the current TKDE manuscript and perform a full **gap audit**.
6. Only after the gap audit, list genuinely missing reviewer-facing experiments (for example cross-city/generalization, explicit distribution shift, tail/worst-segment behavior) for PI approval before running them.

## What should not be done by default

- Do not continue Beijing FiLM hyperparameter chasing; the completed validation-only search did not produce a better justified choice.
- Do not create V15/V16 without a new research decision.
- Do not change the train/test split, paper targets, objective, or model-selection rules silently.
- Do not use the test set to choose hyperparameters.
- Do not hide the `linear_cluster` or `shared_no_task` counterevidence.
- Do not claim universal dynamic-task-discovery or universal mechanism dominance from the current results.

## 4090 research environment

Current server-side project environment used for the frozen experiments:

- workspace: `/workspace/MetaSTC-J`
- Python: `/opt/conda/bin/python`
- PyTorch: `2.6.0+cu124`
- GPUs: 2 × NVIDIA GeForce RTX 4090

FiLM uses FP32 because the current FFT configuration is not compatible with the intended FP16/BF16 path. The training/evaluation scripts use dataset-native cluster counts: Beijing=5, Shanghai=3, LargeST=3.

## Historical conference-reproduction baseline

Before the journal extension, the original MetaSTC behavior was reproduced closely enough to serve as the static anchor. Earlier single-run values and paper-scale diagnostics under `param/4090_tuned/` were useful during reproduction/debugging, but they are **not the authoritative journal conclusion anymore**.

For the journal version, always prefer:

```text
results/journal_handoff_20260906/unified_results.*
```

over old single-run tables or legacy diagnostic notes.

## Project status in one sentence

**MetaSTC-J has completed the current V14 experimental exploration; LSTM provides stable three-dataset journal gains, FiLM provides qualified cross-realization support, mechanism claims are bounded by matched-control evidence, and the project is now ready for student-led manuscript consolidation rather than continued open-ended tuning.**
