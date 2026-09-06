# MetaSTC-J Research State

Last updated: 2026-09-06

## Phase
The current V14 experimental exploration is closed. The project is in **journal manuscript consolidation / student handoff**.

Authoritative entry: `docs/STUDENT_HANDOFF_20260906.md`. Frozen evidence: `results/journal_handoff_20260906/`.

## Scientific state
- LSTM V14 is the primary result: five-seed mean MAE/MSE beats ICDM 2024 MetaSTC+LSTM on Beijing, Shanghai, LargeST; all 15 MAE seed-dataset runs beat the paper value.
- FiLM V14 is supporting cross-realization evidence: LargeST improves both metrics; Beijing/Shanghai improve MSE but have slightly worse mean MAE than paper FiLM.
- Richer context-conditioned adaptation is not universally dominant over matched controls; claims must expose the Beijing `linear_cluster` and LargeST `shared_no_task` counterevidence.
- No V15/V16 or further Beijing FiLM tuning is authorized by current evidence.

## Frozen paths
- Branch: `dev-260902-dynamic`
- Freeze tag: `metastc-j-p0-v14-freeze-20260905`
- LSTM: `model_code/dynamic_residual_mechanism_v14.py`, `model_code/dynamic_residual_mechanism_sweep_v14.py`
- FiLM: `model_code/dynamic_residual_meta_adapter_film_v14.py`
