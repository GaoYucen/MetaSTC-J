# MetaSTC-J Reproduction Baseline — 2026-09-02

This directory freezes the lightweight code, configuration, logs and result summaries used for the current conference-paper reproduction before journal-version modifications.

## Runtime environment
```text
2026-09-02T08:01:15+00:00
81f635a75946
/opt/conda/envs/py11/bin/python
Python 3.11.15
torch= 2.5.1+cu121
torch_cuda= 12.1
numpy= 2.4.4
pandas= 3.0.3
sklearn= 1.9.0
cuda_available= True
gpu_count= 2
0, NVIDIA GeForce RTX 4090, 535.129.03, 24564 MiB
1, NVIDIA GeForce RTX 4090, 535.129.03, 24564 MiB
```

## Baseline result table (from root README.md)
| 数据集 | 模型 | MAE | RMSE | MSE | MAPE | R² | 训练时间（秒） |
|---|---|---:|---:|---:|---:|---:|---:|
| Beijing | MetaSTC + LSTM | 3.4990 | 5.2146 | 27.1918 | 0.1403 | 0.8461 | 419.724 |
| Beijing | MetaSTC + FiLM | 3.3819 | 5.2387 | 27.4442 | 0.1343 | 0.8447 | 808.419 |
| Shanghai | MetaSTC + LSTM | 4.3744 | 6.3861 | 40.7825 | 0.1845 | 0.7703 | 23.122 |
| Shanghai | MetaSTC + FiLM | 3.9524 | 6.0189 | 36.2273 | 0.1733 | 0.7959 | 18.873 |
| LargeST | MetaSTC + LSTM | 34.5379 | 50.0249 | 2502.4915 | 0.1916 | 0.9160 | 199.929 |
| LargeST | MetaSTC + FiLM | 32.4801 | 51.5553 | 2657.9458 | 0.1403 | 0.9107 | 413.325 |

## Notes
- Project root was not a Git worktree when this snapshot was created.
- Full checkpoints remain under `param/4090_tuned/`; their SHA256 hashes are recorded in `large_asset_sha256.txt`.
- Data files are not duplicated; their SHA256 hashes are recorded where practical.
- Future journal experiments should write to a new output namespace (recommended: `param/journal/`) and should not overwrite `param/4090_tuned/`.
