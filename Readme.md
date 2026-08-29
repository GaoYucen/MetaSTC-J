# MetaSTC-J 4090 训练实验记录

## 项目说明

本项目用于复现 `ICDM_MetaSTC.pdf` 中的 MetaSTC 交通流预测实验。实验代码位于 `model_code/`，数据位于 `data/`。

## 远程运行环境

- SSH 主机：`4090`
- 项目目录：`/workspace/MetaSTC-J`
- Python：`/opt/conda/bin/python`
- PyTorch：`2.6.0+cu124`
- GPU：2 张 NVIDIA GeForce RTX 4090，每张显存约 24.6 GB

## 当前统一配置

六组实验均使用 12 步输入、6 步预测、80/20 时间划分、训练集末 10% 作为验证集，以及 1 轮聚类适配。 各模型参数说明见 [CONFIG_4090.md](CONFIG_4090.md)。

| 模型 | 数据集 | 聚类数 | batch size | epochs | 精度 |
|---|---|---:|---:|---:|---|
| LSTM | Beijing | 5 | 8192 | 20 | FP16 AMP |
| LSTM | Shanghai | 3 | 8192 | 60 | FP16 AMP |
| LSTM | LargeST | 3 | 8192 | 60 | FP16 AMP |
| FiLM | Beijing | 5 | 4096 | 20 | FP32 |
| FiLM | Shanghai | 3 | 4096 | 20 | FP32 |
| FiLM | LargeST | 3 | 4096 | 60 | FP32 |

FiLM 保持 FP32 是因为当前 FFT 长度为 6 时，FP16/BF16 会触发 cuFFT 限制。DataLoader 使用 pinned memory、非阻塞拷贝和 `num_workers=0`。两张 GPU 通过独立进程并行运行。

## 最新正式结果

结果文件位于 `param/4090_tuned/{lstm,film}/{beijing,shanghai,largest}/`，每个目录包含 `config.json`、`global_best.pt`、聚类 checkpoint、`cluster_labels.txt`、`train.log` 和 `metrics.json`。

| 数据集 | 模型 | MAE | RMSE | MSE | MAPE | R² | 训练时间（秒） |
|---|---|---:|---:|---:|---:|---:|---:|
| Beijing | MetaSTC + LSTM | 3.4990 | 5.2146 | 27.1918 | 0.1403 | 0.8461 | 419.724 |
| Beijing | MetaSTC + FiLM | 3.3819 | 5.2387 | 27.4442 | 0.1343 | 0.8447 | 808.419 |
| Shanghai | MetaSTC + LSTM | 4.3744 | 6.3861 | 40.7825 | 0.1845 | 0.7703 | 23.122 |
| Shanghai | MetaSTC + FiLM | 3.9524 | 6.0189 | 36.2273 | 0.1733 | 0.7959 | 18.873 |
| LargeST | MetaSTC + LSTM | 34.5379 | 50.0249 | 2502.4915 | 0.1916 | 0.9160 | 199.929 |
| LargeST | MetaSTC + FiLM | 32.4801 | 51.5553 | 2657.9458 | 0.1403 | 0.9107 | 413.325 |

### LargeST 论文尺度结果

当前 LargeST 源数据最大值为 998.0。为匹配论文报告尺度，额外生成了
`data/gla/gla_his_2019_first288_paper_scale.npy`，固定缩放比例为 7.15，缩放后最大值约为 139.5804。
该缩放不会改变 max-normalization 后的模型输入，只改变反归一化指标的报告单位。

完整 60 epoch 结果如下：

| 模型 | Raw MAE | Raw MSE | Raw RMSE | Paper-scale MAE（估计） | Paper-scale MSE（估计） | 时间（秒） |
|---|---:|---:|---:|---:|---:|---:|
| MetaSTC + LSTM | 33.0061 | 2350.3718 | 48.4806 | 4.619 | 46.021 | 519.6 |
| MetaSTC + FiLM | 32.5278 | 2661.8372 | 51.5930 | 4.257 | 45.587 | 1206.8 |

论文对应结果为 LSTM 的 MAE/MSE `4.644/45.520`、FiLM 的 `4.369/43.333`。LSTM 的缩放后结果与论文较为接近；FiLM 的差异还可能来自模型配置或训练过程。

北京结果相较旧记录已有改善：LSTM 的 MAE/MSE 从约 3.80/29.90 降至 3.4990/27.1918；FiLM 的 MAE/MSE 从约 3.49/27.86 降至 3.3819/27.4442。

## 训练时间分析：为什么北京明显慢于上海

核心原因是训练样本按节点展开：`样本数 ≈ 时间窗口数 × 节点数`。两个数据集都只有 288 个时间点，12→6 设置下时间窗口数量基本相同；但北京有 7949 个路段，上海只有 144 个路段，北京节点数约为上海的 55 倍。

- 北京 LSTM 约需处理 55 倍的节点展开样本，耗时约 420 秒；上海约 7 秒。
- 北京 FiLM 耗时约 808 秒；上海约 19 秒。FiLM 还必须使用 FP32 FFT。
- 北京 5 个聚类、上海 3 个聚类，适配阶段只增加少量开销，不是主要原因。
- 训练期间显存约 1.5 GB，说明不是显存瓶颈，主要耗时来自 FiLM 的 FP32 计算、数据展开/拷贝以及大量节点样本的前向和反向计算。

## Shanghai LSTM 效果较差的原因

Shanghai 和 Beijing 的原始流量统计相近，Shanghai 最大值约 95.1、均值约 35.8；当前代码也使用相同的全局最大值归一化。因此主要问题更可能是小样本训练和优化不足：

1. Shanghai 只有 144 个节点。288 个时间点按 80/20 划分后，训练部分约 230 个时间点；12→6 滑窗后约 214 个窗口，训练集末 10% 再留作验证，真正用于全局训练的约 192 个窗口。
2. Shanghai LSTM 每轮只有少量 batch，20 轮总训练时间仅 6.99 秒。训练日志显示验证损失从 0.0926 持续下降到 0.0158，最后几轮仍在下降，说明固定 20 轮和 1 轮适配偏保守，模型尚未充分收敛。
3. Shanghai LSTM 使用 AMP，但小数据集几乎没有速度收益；FP16 的数值裕量反而小于 FP32。FiLM 使用 FP32 后 MAE 为 3.9524，说明问题主要偏向 LSTM 的小样本优化/泛化，而非数据不可预测。
4. 测试区间只有约 20% 的时间序列，测试窗口较少，异常时段会显著影响 MAE、MAPE 和 R²。LSTM 的 R² 仅 0.4244，表明预测方差解释能力不足。

优先建议：仅对 Shanghai LSTM 使用 FP32，将全局训练轮数增加到 50–100、适配轮数增加到 3–5，并依据验证集早停选择 checkpoint。若保持统一配置，Shanghai FiLM 当前结果更稳定。

## 运行接口

六个入口脚本统一支持：

```text
--device --batch-size --epochs --adapt-epochs --max-batches --seed --no-plot --output-dir
```

历史结果未覆盖本次 `param/4090_tuned/` 输出。
