# MetaSTC-J

This repository contains the official implementation of **MetaSTC-J**, a meta-learning framework designed for capturing complex spatio-temporal correlations in traffic flow prediction. This project is an extended version of our ICDM 2024 paper.

## Environment

- **Python:** 3.11
- **PyTorch:** 2.6.0

## Project Structure

### Data
The `data/` directory contains the datasets used for training and evaluation:
- `traffic_flow/`: Directory containing traffic flow data samples.
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
