# Frozen journal results — LSTM + FiLM

Negative Δ means the journal V14 result is better than the corresponding ICDM 2024 MetaSTC result. Std is sample std across seeds 42–46.

| Realization | Dataset | Paper MAE | Journal MAE mean ± std | ΔMAE | MAE wins | Paper MSE | Journal MSE mean ± std | ΔMSE | MSE wins |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LSTM | Beijing | 3.534 | 3.5109 ± 0.0191 | -0.65% | 5/5 | 27.433 | 27.2152 ± 0.2567 | -0.79% | 4/5 |
| LSTM | Shanghai | 4.524 | 4.2654 ± 0.0177 | -5.72% | 5/5 | 42.380 | 39.3072 ± 0.2180 | -7.25% | 5/5 |
| LSTM | LargeST | 4.644 | 4.4582 ± 0.0328 | -4.00% | 5/5 | 45.520 | 44.6292 ± 0.4482 | -1.96% | 5/5 |
| FiLM | Beijing | 3.367 | 3.3975 ± 0.0173 | +0.91% | 0/5 | 26.893 | 26.7645 ± 0.1578 | -0.48% | 4/5 |
| FiLM | Shanghai | 4.018 | 4.0368 ± 0.0435 | +0.47% | 1/5 | 37.076 | 36.2305 ± 0.2833 | -2.28% | 5/5 |
| FiLM | LargeST | 4.369 | 4.2718 ± 0.0154 | -2.22% | 5/5 | 43.333 | 43.0410 ± 0.2948 | -0.67% | 4/5 |

## Paper-level interpretation

- **Primary result:** LSTM V14 beats the conference LSTM result in five-seed mean MAE and MSE on Beijing, Shanghai, and LargeST; all 15 LSTM dataset-seed MAE runs beat the corresponding paper value.
- **Supporting result:** FiLM shows transfer of the framework, but the gain is dataset/metric dependent. LargeST improves both metrics; Beijing and Shanghai improve mean MSE while mean MAE is slightly above the corresponding paper FiLM result.
- Do not claim universal mechanism dominance: `linear_cluster` is stronger than V14 on Beijing matched-control MAE, and `shared_no_task` is stronger on LargeST matched-control MAE.

## FiLM target warning

Some raw FiLM `metrics.json` files contain a legacy `paper_target_l12` field inherited from the LSTM experiment path. Do **not** use that field for the FiLM paper comparison. Correct FiLM L=12 targets are Beijing 3.367/26.893, Shanghai 4.018/37.076, LargeST 4.369/43.333 (MAE/MSE).
