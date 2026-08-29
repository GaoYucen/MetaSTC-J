# MetaSTC-J 4090 模型配置说明

本文档说明 `/workspace/MetaSTC-J/model_code/` 下统一入口脚本的推荐配置。所有配置默认从项目根目录运行，并使用 `/opt/conda/bin/python`。

## 固定数据与流程

- 所有数据使用 288 个时间点。
- 输入长度 `look_back=12`，预测长度 `look_forward=6`。
- 时间划分为前 80% 训练、后 20% 测试；训练段最后 10% 作为验证集。
- 训练完成后从验证损失最低的全局 checkpoint 开始进行聚类适配；测试集只用于最终评估。
- 输出目录必须按实验分开，避免覆盖历史结果。

## 推荐基线配置

| 模型 | 数据集 | 聚类数 | batch size | 全局 epochs | adapt epochs | 精度 | 说明 |
|---|---|---:|---:|---:|---:|---|---|
| LSTM | Beijing | 5 | 8192 | 20 | 1 | FP16 AMP | 节点多，吞吐优先 |
| LSTM | Shanghai | 3 | 8192 | 60 | 1 | FP16 AMP | 60 epoch 已显著改善收敛；进一步优化可改 FP32 |
| LSTM | LargeST | 3 | 8192 | 60 | 1 | FP16 AMP | 固定使用前 288 点 |
| FiLM | Beijing | 5 | 4096 | 20 | 1 | FP32 | FFT 长度为 6，不能使用 FP16/BF16 |
| FiLM | Shanghai | 3 | 4096 | 20 | 1 | FP32 | 当前小数据集上较稳定 |
| FiLM | LargeST | 3 | 4096 | 60 | 1 | FP32 | 固定使用前 288 点；先以验证集选择最佳 epoch |

## 各模型适用建议

### Beijing

北京有约 7949 个路段，样本量最大。LSTM 使用 `batch_size=8192` 和 FP16 AMP；FiLM 使用 `batch_size=4096` 和 FP32。北京 FiLM 计算时间较长，主要受节点展开样本量和 FP32 频域计算影响。

### Shanghai

上海只有 144 个路段和单日 288 个时间点，有效时间窗口很少。LSTM 使用 20 epoch 时容易欠拟合，当前推荐先使用 60 epoch；如果继续优化，优先测试 FP32、`batch_size=2048/4096` 和 `adapt_epochs=3~5`。FiLM 保持 FP32，不建议为节省时间强行启用 AMP。

### LargeST

LargeST 原始文件形状为 `35040 × 3834`，但本项目的固定实验定义是只取前 288 个时间点，实际输入为 `3834 × 288`。不要修改为完整时间序列，也不要将当前结果与不同时间范围或不同量纲的结果直接比较。

在固定前 288 点的前提下，LSTM 优先使用 60 epoch；FiLM 使用验证集最佳 checkpoint，重点观察 MAE/MAPE 与 MSE/RMSE 的取舍。

## 运行命令

```bash
# Shanghai LSTM
CUDA_VISIBLE_DEVICES=0 /opt/conda/bin/python model_code/meta-LSTM_city.py \
  --epochs 60 --adapt-epochs 1 --no-plot \
  --output-dir param/4090_tuned/epoch60/lstm/shanghai

# LargeST LSTM
CUDA_VISIBLE_DEVICES=0 /opt/conda/bin/python model_code/meta-LSTM_LargeST.py \
  --epochs 60 --adapt-epochs 1 --no-plot \
  --output-dir param/4090_tuned/epoch60/lstm/largest

# LargeST FiLM
CUDA_VISIBLE_DEVICES=1 /opt/conda/bin/python model_code/meta-film_LargeST.py \
  --epochs 60 --adapt-epochs 1 --no-plot \
  --output-dir param/4090_tuned/epoch60/film/largest
```

快速检查时追加 `--max-batches 1 --epochs 1`。正式结果目录应包含 `config.json`、`global_best.pt`、聚类 checkpoint、`cluster_labels.txt`、`train.log` 和 `metrics.json`。
