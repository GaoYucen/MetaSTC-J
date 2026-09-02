# MetaSTC-J

This repository contains the official implementation of **MetaSTC-J**, a meta-learning framework designed for capturing complex spatio-temporal correlations in traffic flow prediction. This project is an extended version of our ICDM 2024 paper.

## Environment

- **Python:** 3.11
- **PyTorch:** 2.6.0

## Project Structure

### Data
The `data/` directory contains the datasets used for training and evaluation:
- `traffic_flow/`: Directory contains the traffic flow data samples.
- `link_feature.txt`: Spatial features and attributes for the road network.

### Model Code for ICDM
The `model_code/` directory includes the core implementations:
- `meta-LSTM.py`: Implementation of the MetaSTC framework integrated with LSTM.
- `meta-film.py`: Implementation of the MetaSTC framework integrated with FiLM (Feature-wise Linear Modulation).
- `ablation study/`: Scripts for ablation experiments, including clustering analysis and distance function evaluations.

### Model Code for TKDE
- previous_version: The standard version of TKDE
- `metastc_lstm.py`: The debug version

## Usage

To train and evaluate the models, run the following commands from the project root:

**Run MetaSTC + LSTM:**
```bash
python model_code/meta-LSTM.py
```

**Run MetaSTC + FiLM:**
```bash
python model_code/meta-film.py
```

## Experimental Results

The following table shows the performance comparison on the Beijing dataset with a prediction horizon of $L=12$.

### Table 1: Performance Comparison (Beijing, L=12)

| Model        | MSE      | MAE     | MAPE    | $R^2$   |
|--------------|----------|---------|---------|---------|
| LSTM         | 46.483   | 4.837   | 0.000   | 0.000   |
| MetaSTC+LSTM | 27.771   | 3.542   | 0.114   | 0.804   |

---
*Note: The results above are based on the current experimental configuration. Ensure all data paths are correctly set before execution.*

## Official ICDM 2024 Paper Results and Journal Success Criterion

The canonical conference-paper numbers below are taken from **Table IV (Performance of MetaSTC and Baselines)** in the final ICDM paper `MetaSTC: A Backbone Agnostic Spatio-Temporal Framework for Traffic Forecasting`. These values, rather than the older Beijing summary table above, should be used as the formal comparison target for the journal extension.

### MetaSTC results reported in the paper

| Dataset | Model | MAE (L=12) | MAE (L=24) | MSE (L=12) | MSE (L=24) |
|---|---|---:|---:|---:|---:|
| Beijing | MetaSTC+LSTM | 3.534 | 3.710 | 27.433 | 29.040 |
| Beijing | MetaSTC+FiLM | **3.367** | **3.476** | **26.893** | **27.527** |
| Shanghai | MetaSTC+LSTM | 4.524 | 4.429 | 42.380 | 40.276 |
| Shanghai | MetaSTC+FiLM | **4.018** | **4.173** | **37.076** | **37.992** |
| LargeST | MetaSTC+LSTM | 4.644 | 5.032 | 45.520 | 49.102 |
| LargeST | MetaSTC+FiLM | **4.369** | **4.491** | **43.333** | **44.398** |

### Journal-version target

The journal extension should not be considered successful merely because a new module improves an internal smoke-test or temporal-holdout baseline. The primary success criterion is **paper-protocol performance that is better than the corresponding MetaSTC result reported above**, ideally with consistent gains across datasets/backbones rather than a single isolated case.

For the first-stage Beijing + LSTM development, the minimum formal target is therefore:

- **L=12:** MAE < **3.534** and MSE < **27.433**.
- **L=24:** MAE < **3.710** and MSE < **29.040**.

For a stronger journal claim, the improved method should also aim to beat the paper's strongest FiLM-based MetaSTC results under the same protocol (e.g., Beijing L=12 MAE **3.367**, Shanghai L=12 MAE **4.018**, LargeST L=12 MAE **4.369**) or demonstrate a consistent accuracy/robustness advantage across multiple settings.
