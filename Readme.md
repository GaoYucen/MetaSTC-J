Environment:
Python 3.11
Pytorch 2.6.0

data:
- traffic_flow: samples for traffic flow data
- link_feature.txt: spatial features for roads

model_code:
- meta-LSTM: MetaSTC+LSTM
- meta-film: MetaSTC+Film
- ablation study: including clustering, distance function

To execute:

```
python model_code/meta-LSTM.py
```

```
python model_code/meta-film.py
```

### Experimental results:

Table 1: Performance of Models for Beijing (L=12)

| Model        | MSE | MAE | MAPE | R2 |
|--------------| --- |  | --- | --- |
| LSTM         | 46.483 | 4.837 | 0.000 | 0.000 |
| MetaSTC+LSTM | 27.771 | 3.542 | 0.114 | 0.804 |



