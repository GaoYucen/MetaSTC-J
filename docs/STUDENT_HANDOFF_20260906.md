# MetaSTC-J 学生交接说明（2026-09-06）

本文件是当前学生接手入口。实验探索已经收口，后续默认进入“论文整合 + 必要补实验判断”，而不是继续无边界调参。

## 当前核心结论

1. **LSTM 是期刊主结果。** V14 五种子均值在 Beijing、Shanghai、LargeST 的 MAE/MSE 都优于 ICDM 2024 MetaSTC+LSTM；三个数据集共 15 个 seed 的 MAE 全部优于论文值。完整数字见 `results/journal_handoff_20260906/unified_results.md`。
2. **FiLM 是跨 realization 的补强结果。** LargeST 的 MAE/MSE 都稳定优于会议版；Shanghai 与 Beijing 的 MSE 改善，但五种子 MAE 均值分别略差约 0.47% 与 0.91%。因此只能主张框架可迁移且收益具有数据集/指标依赖性，不能主张全面超越。
3. **机制结论必须克制。** Beijing 的 `linear_cluster` matched control 平均 MAE 优于 V14；LargeST 的 `shared_no_task` 平均 MAE优于 V14。Shanghai 是最清晰的正向案例：V14 seed42 在 40 个测试窗口全部胜过 linear control，高 static-error tercile 的额外 MAE 收益约 0.0928。
4. V14 增加约 10.9K 参数，绝对参数量小；但相对极小 LSTM expert 的比例不小，因此不要写“negligible parameter overhead”。4090 批量测试未观察到实质性推理延迟增加。

## 正确的论文 target

LSTM L=12：Beijing 3.534/27.433，Shanghai 4.524/42.380，LargeST 4.644/45.520（MAE/MSE）。

FiLM L=12：Beijing 3.367/26.893，Shanghai 4.018/37.076，LargeST 4.369/43.333（MAE/MSE）。

**注意：部分原始 FiLM `metrics.json` 的 `paper_target_l12` 字段继承了 LSTM 代码路径的旧值，不能用作 FiLM 会议版比较。** 以 `unified_results.*` 和上面的 FiLM target 为准。

## 首先阅读

1. `STUDENT_HANDOFF.md`
2. `docs/STUDENT_HANDOFF_20260906.md`
3. `results/journal_handoff_20260906/unified_results.md`
4. `docs/JOURNAL_P0_FREEZE.md`
5. `results/journal_handoff_20260906/lstm_matched_controls/summary.md`
6. `manuscript/TKDE_MetaSTC_v14/TKDE_MetaSTC_Journal.tex`

## 接下来应完成

- 用 `unified_results.*` 整理最终 LSTM + FiLM 论文表格，不再重训追数字。
- 把 LSTM 三数据集稳定提升作为主性能结论；FiLM 作为跨 backbone/realization 的补充证据。
- 把 Shanghai 正向 case study 与 Beijing/LargeST matched-control 反例同时写入，形成可信的机制边界。
- 核对论文里每个数字、表、图、设置与结论是否对应冻结结果。
- 编译当前 TKDE 稿并完成 gap audit；只有确实存在 reviewer-facing 缺口时，再向导师提出需要补什么实验。

## 当前不要做

- 不继续针对 Beijing FiLM 调参追 MAE；已有 validation-only 小网格没有给出更好的选择。
- 不未经导师同意创建 V15/V16。
- 不修改 train/test split、论文 target、模型选择规则，或用 test set 选超参。
- 不隐藏 `linear_cluster` / `shared_no_task` 的反例来强化机制故事。

## 可复现实验资产

冻结 static anchors 已提交到代码运行所期望的原路径：

- FiLM: `param/4090_tuned/film/{beijing,shanghai,largest}/`
- LSTM: `param/4090_tuned/lstm/beijing/` 与 `param/4090_tuned/epoch60/lstm/{shanghai,largest}/`

最终 V14 adapter checkpoints、原始指标和日志位于 `results/journal_handoff_20260906/`。FiLM 实现为 `model_code/dynamic_residual_meta_adapter_film_v14.py`；LSTM 冻结实现为 `model_code/dynamic_residual_mechanism_v14.py` / `dynamic_residual_mechanism_sweep_v14.py`。

## 论文资产

`manuscript/TKDE_MetaSTC_v14/` 中仅提交当前可编辑 TeX、bib/IEEE 模板、正文实际引用的图和 compact review PDF；`.aux/.log/.fls/.fdb_latexmk/synctex` 等构建垃圾没有提交。
