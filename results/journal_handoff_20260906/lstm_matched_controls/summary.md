# V14 Mechanism Validation — Efficient 5-seed Matched Controls

Official L=12/P=6 split; frozen tuned static MetaSTC anchors; cached static outputs; 5 seeds; 10 epochs; matched objectives.

| Dataset | Variant | Params | Test MAE mean±std | Relative MAE vs static | V14 wins paired |
|---|---|---:|---:|---:|---:|
| beijing | v14 | 10871 | 3.5109 ± 0.0191 | 0.34% ± 0.55 | — |
| beijing | constant_gate | 10831 | 3.5023 ± 0.0174 | 0.09% ± 0.50 | 1/5 |
| beijing | shared_no_task | 10839 | 3.5644 ± 0.0071 | 1.87% ± 0.20 | 5/5 |
| beijing | linear_cluster | 815 | 3.4693 ± 0.0005 | -0.85% ± 0.01 | 0/5 |
| shanghai | v14 | 10855 | 4.2654 ± 0.0177 | -2.49% ± 0.41 | — |
| shanghai | constant_gate | 10815 | 4.3130 ± 0.0065 | -1.40% ± 0.15 | 5/5 |
| shanghai | shared_no_task | 10839 | 4.2748 ± 0.0175 | -2.28% ± 0.40 | 5/5 |
| shanghai | linear_cluster | 489 | 4.3403 ± 0.0000 | -0.78% ± 0.00 | 5/5 |
| largest | v14 | 10855 | 4.4582 ± 0.0328 | -3.42% ± 0.71 | — |
| largest | constant_gate | 10815 | 4.4746 ± 0.0114 | -3.07% ± 0.25 | 3/5 |
| largest | shared_no_task | 10839 | 4.3988 ± 0.0125 | -4.71% ± 0.27 | 0/5 |
| largest | linear_cluster | 489 | 4.5810 ± 0.0011 | -0.76% ± 0.02 | 5/5 |
