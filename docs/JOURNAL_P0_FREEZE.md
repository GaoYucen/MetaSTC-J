# MetaSTC-J Journal P0 Freeze

Freeze date: 2026-09-05
Branch: `dev-260902-dynamic`

## Frozen research question

The journal extension studies **dynamic spatio-temporal heterogeneity** while preserving the conference MetaSTC model as a strong static anchor. The current candidate is:

**Static MetaSTC Anchor + Context-Conditioned Gated Residual Meta-Adaptation**

The prediction form is conceptually:

`y_hat = y_static + gate(context) * residual(context, task/state)`

The journal story must not claim that discrete dynamic task discovery has been established unless later evidence explicitly supports that claim. The preferred interpretation is continuous context-conditioned latent task/state representation.

## Frozen P0 implementation

The official P0 mechanism implementation is:

- `model_code/dynamic_residual_mechanism_v14.py`
- `model_code/dynamic_residual_mechanism_sweep_v14.py`

These files are the implementation basis for P0 evidence collection. Architecture changes beyond bug fixes require a new version and a new research decision; they must not silently overwrite this freeze.

## Frozen P0 protocol

- Forecast protocol: official LSTM `L=12` setup matching the frozen reproduction baseline.
- Datasets: Beijing, Shanghai epoch60, LargeST epoch60.
- Seeds: `42, 43, 44, 45, 46`.
- Model selection and training budget must be identical across matched variants.
- Primary metric: test MAE; also report MSE, mean ± std, per-seed wins, and paired significance where applicable.
- Do not retune the frozen conference/static baseline during P0.

## Frozen matched controls

At minimum, P0 must compare:

1. Strong Static MetaSTC reproduction.
2. `linear_cluster` residual control.
3. `constant_gate` residual adapter.
4. `shared_no_task` adapter.
5. V14 context-conditioned gated residual meta-adapter.

Any added control must use matched data split, budget, and model-selection rules.

## Current evidence at freeze

Earlier single-run official-protocol results showed V14 improving MAE over strong static reproduction by approximately:

- Beijing: 1.55%
- Shanghai: 2.66%
- LargeST: 4.74%

However, the newer Beijing five-seed attribution study gives:

| Variant | MAE mean ± std | Relative MAE vs static | Beats static |
|---|---:|---:|---:|
| V14 | 3.477888 ± 0.023204 | -0.604% | 4/5 |
| constant_gate | 3.494879 ± 0.028843 | -0.118% | 1/5 |
| shared_no_task | 3.560092 ± 0.011238 | +1.746% | 0/5 |
| linear_cluster | 3.469372 ± 0.000561 | -0.847% | 5/5 |

Therefore V14 is **not yet proven superior to simple matched residual controls**. In particular, `linear_cluster` currently has lower mean MAE and substantially lower variance on Beijing.

## P0 decision rule

### Continue the dynamic-heterogeneity journal story if

- V14 is stable across datasets/seeds, and
- context/gating gives clear additional value over simple matched controls, especially under measurable temporal/distribution shift or hard segments, and
- gate/shift statistics explain when adaptation helps.

### Stop or downgrade the story if

- simple residual/linear controls match or beat V14 across ordinary and high-shift settings, or
- gains cannot be attributed to context-conditioned adaptation.

In that case, do not create V15/V16 merely to chase average MAE. Revisit the research hypothesis or simplify the extension.

## Next P0 evidence tasks

1. Complete the same five-seed matched-control matrix for Shanghai and LargeST.
2. Record per-window static error, V14 gain, linear-control gain, gate value, and shift statistic.
3. Stratify performance by shift strength / hard temporal segment and test whether V14 gains concentrate where static assumptions fail.
4. Only after P0 mechanism evidence passes, extend to FiLM, explicit temporal/distribution shift, cross-city generalization, tail/worst-segment metrics, and adaptation cost.

## Automation boundary

Automated continuation may execute the frozen P0 matrix, aggregate results, fill missing seeds/datasets, and run the predefined gate/shift analysis. It must stop for human/research review before architecture redesign, hypothesis changes, new model versions, altered data splits, altered objectives, or entering P1.

## Frozen implementation SHA256
```text
3b42073bdce7b5169e1ed10faed94d9b2781e3115de83e02fde6fa835ec0a66f  model_code/dynamic_residual_mechanism_v14.py
54e9c9d12b104c4f29b1224144c2a3b3dec4a18f3d71c759149e63faa4ad9137  model_code/dynamic_residual_mechanism_sweep_v14.py
```


## 2026-09-05 post-freeze operational fix — dataset-native cluster counts

The first governed P0-01 sweep exposed a loader bug: the sweep hard-coded five clusters for every dataset, while the frozen static checkpoints were trained with their native configured cluster counts: Beijing=5, Shanghai=3, LargeST=3. Shanghai and LargeST therefore correctly contain only `cluster_0.pt` through `cluster_2.pt`; attempting to load `cluster_3.pt` was erroneous.

The sweep now uses the dataset-specific frozen cluster count (`5/3/3`) while leaving the research question, static checkpoints, data split, seeds, matched variants, objectives, training budget, and evaluation metrics unchanged. This is an operational/reproducibility bugfix, not an architecture or protocol change.

- pre-fix sweep SHA256: `2e095c22fbb87e5ca8dda58f022cbf43106c106bf7cc2159c91543017a91ccbb`
- post-fix sweep SHA256: `54e9c9d12b104c4f29b1224144c2a3b3dec4a18f3d71c759149e63faa4ad9137`
