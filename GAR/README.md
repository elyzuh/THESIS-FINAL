# GAR (Generalized Autoregressive) Model

## Overview
This directory contains the implementation of the Generalized Autoregressive (GAR) baseline model for epidemiological forecasting. The GAR model extends traditional autoregressive models by incorporating nonlinear transformations through a neural network architecture.

## Model Architecture
The GAR model uses:
- **Input**: Window of historical observations (window × m features)
- **Hidden Layer**: Fully connected layer with ReLU activation for nonlinear feature extraction
- **Dropout**: Regularization layer to prevent overfitting
- **Output Layer**: Fully connected layer producing predictions for all regions

Unlike AR (which uses linear weights) and VAR (which uses linear matrices per lag), GAR learns nonlinear relationships through a feedforward neural network.

## Directory Structure
```
GAR/
├── models/
│   ├── __init__.py
│   └── GARmodel.py          # GAR model implementation
├── main.py                   # Main training script
├── utils.py                  # Data utilities
├── utils_ModelTrainEval.py  # Training and evaluation functions
├── Optim.py                  # Optimization utilities
├── PlotFunc.py              # Plotting functions
├── PlotData.py              # Data plotting utilities
├── GenerateAdjacentMatrix.py # Adjacency matrix generation
├── log_parser.py            # Log parsing utilities
├── cut_log.py               # Log cutting utilities
└── data/                    # Data directory
```

## Usage

### Training
```bash
python main.py --data ./data/your_data.csv \
               --model GARmodel \
               --window 168 \
               --horizon 12 \
               --hidRNN 50 \
               --dropout 0.2 \
               --epochs 100 \
               --batch_size 128 \
               --lr 0.001 \
               --save_name gar_model \
               --gpu 0
```

### Key Parameters
- `--data`: Path to the input data file
- `--model`: Model name (use 'GARmodel' or 'GAR')
- `--window`: Size of the lookback window (default: 168)
- `--horizon`: Prediction horizon (default: 12)
- `--hidRNN`: Number of hidden units in the neural network (default: 50)
- `--dropout`: Dropout rate for regularization (default: 0.2)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Training batch size (default: 128)
- `--lr`: Learning rate (default: 0.001)
- `--train`: Fraction of data for training (default: 0.6)
- `--valid`: Fraction of data for validation (default: 0.2)
- `--normalize`: Normalization method (0: none, 1: global, 2: per-variable)
- `--gpu`: GPU device number (None for CPU)

## Model Comparison

| Model | Type | Parameters |
|-------|------|-----------|
| **AR** | Linear | window × m weights |
| **VAR** | Linear | p × m × m matrices (p lags) |
| **GAR** | Nonlinear | (window×m → hidden → m) neural network |

## Outputs
The model outputs the following metrics:
- **RSE**: Root Squared Error
- **RAE**: Relative Absolute Error
- **Relative Error**: Mean relative error percentage
- **Correlation**: Average correlation across regions
- **R²**: Coefficient of determination

## Implementation Details
- Framework: PyTorch
- Loss Function: MSE (Mean Squared Error)
- Optimizer: Adam (default), also supports SGD, Adagrad, Adadelta
- Gradient Clipping: Applied to prevent exploding gradients
- Early Stopping: Based on validation loss

## Notes
- The GAR model is more flexible than AR/VAR due to its nonlinear architecture
- Requires tuning of the hidden layer size (`--hidRNN`)
- Dropout helps prevent overfitting on small datasets
- Can capture complex temporal patterns that linear models miss
