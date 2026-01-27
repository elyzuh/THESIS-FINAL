# Baseline Models Summary

This document provides an overview of the three baseline models implemented for epidemiological forecasting.

## Directory Structure

```
THESIS-FINAL/
├── AR/                    # Autoregressive Model
│   ├── models/
│   │   └── ARmodel.py
│   └── main.py
├── VAR/                   # Vector Autoregressive Model
│   ├── models/
│   │   └── VARmodel.py
│   └── main.py
├── GAR/                   # Generalized Autoregressive Model
│   ├── models/
│   │   └── GARmodel.py
│   ├── main.py
│   ├── README.md
│   └── MODEL_COMPARISON.md
└── CNNRNN-Res-SEIR/      # Main model (existing)
```

## Quick Start

### 1. AR Model (Simplest)
```bash
cd AR
python main.py --data ./data/us_hhs/data.csv \
               --model ARmodel \
               --window 168 \
               --horizon 12 \
               --epochs 100 \
               --save_name ar_baseline
```

### 2. VAR Model (Multivariate)
```bash
cd VAR
python main.py --data ./data/us_hhs/data.csv \
               --model VARmodel \
               --window 168 \
               --horizon 12 \
               --epochs 100 \
               --save_name var_baseline
```

### 3. GAR Model (Nonlinear)
```bash
cd GAR
python main.py --data ./data/us_hhs/data.csv \
               --model GARmodel \
               --window 168 \
               --horizon 12 \
               --hidRNN 50 \
               --dropout 0.2 \
               --epochs 100 \
               --save_name gar_baseline
```

## Model Comparison

| Model | Description | Parameters | Use Case |
|-------|-------------|------------|----------|
| **AR** | Linear univariate autoregression | window × m | Simple trends, independent variables |
| **VAR** | Linear multivariate autoregression | window × m × m | Cross-variable dependencies |
| **GAR** | Nonlinear neural network | (window×m)×hidden×m | Complex nonlinear patterns |

## Common Parameters

All models support the following parameters:

### Data Parameters
- `--data`: Path to data file (required)
- `--train`: Training split ratio (default: 0.6)
- `--valid`: Validation split ratio (default: 0.2)
- `--normalize`: Normalization method (0, 1, or 2)

### Model Parameters
- `--window`: Lookback window size (default: 168)
- `--horizon`: Prediction horizon (default: 12)
- `--output_fun`: Output activation ('sigmoid', 'tanh', or None)

### Training Parameters
- `--epochs`: Number of epochs (default: 100)
- `--batch_size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 0.001)
- `--optim`: Optimizer ('adam', 'sgd', 'adagrad', 'adadelta')
- `--dropout`: Dropout rate (default: 0.2)
- `--clip`: Gradient clipping value (default: 1.0)
- `--weight_decay`: L2 regularization (default: 0)

### Hardware Parameters
- `--gpu`: GPU device number (None for CPU)
- `--seed`: Random seed (default: 54321)

### Output Parameters
- `--save_dir`: Directory to save models (default: './save')
- `--save_name`: Model filename (default: 'tmp')

## GAR-Specific Parameters

The GAR model requires additional hyperparameters:
- `--hidRNN`: Hidden layer size (default: 50)
- `--dropout`: Dropout rate (default: 0.2)

## Evaluation Metrics

All models output the following metrics:

1. **RSE** (Root Squared Error): √(MSE)
2. **RAE** (Relative Absolute Error): Mean absolute error
3. **Relative Error**: Mean relative error as percentage
4. **Correlation**: Pearson correlation coefficient
5. **R²**: Coefficient of determination

## Output Format

During training:
```
| end of epoch 1 | time: 2.50s | train_loss 0.00123456 |
valid rse 0.1234 | valid rae 0.0567 | valid relative error: 12.34% |
valid corr 0.8765 | valid r2 0.7654
```

Final test results:
```
test rse 0.1234 | test rae 0.0567 | test relative error 12.34% |
test corr 0.8765 | test r2 0.7654
```

## Model Files

After training, models are saved as:
```
{save_dir}/{save_name}.pt
```

Load a trained model:
```python
model.load_state_dict(torch.load('save/model.pt'))
```

## Experimental Setup

### Recommended Settings for Comparison

#### Small Dataset (< 1000 samples)
```bash
# AR
--window 24 --hidRNN 32 --epochs 50 --batch_size 32

# VAR
--window 24 --epochs 50 --batch_size 32

# GAR
--window 24 --hidRNN 32 --dropout 0.3 --epochs 50 --batch_size 32
```

#### Medium Dataset (1000-10000 samples)
```bash
# AR
--window 168 --epochs 100 --batch_size 128

# VAR
--window 168 --epochs 100 --batch_size 128

# GAR
--window 168 --hidRNN 50 --dropout 0.2 --epochs 100 --batch_size 128
```

#### Large Dataset (> 10000 samples)
```bash
# AR
--window 336 --epochs 150 --batch_size 256

# VAR
--window 336 --epochs 150 --batch_size 256

# GAR
--window 336 --hidRNN 100 --dropout 0.1 --epochs 150 --batch_size 256
```

## Tips for Good Results

### AR Model
- Start with window size = 1 week (168 hours for hourly data)
- No need for dropout or regularization
- Fast training, use for quick baselines

### VAR Model
- Reduce window size if you have many variables (large m)
- May suffer from overfitting with high-dimensional data
- Consider adding weight_decay if overfitting occurs

### GAR Model
- Tune hidRNN based on data complexity (32-100)
- Use dropout (0.1-0.3) to prevent overfitting
- May need more epochs to converge
- Monitor train vs validation loss for overfitting

## Troubleshooting

### NaN Loss
- Reduce learning rate: `--lr 0.0001`
- Increase gradient clipping: `--clip 0.5`
- Check data for extreme values

### Overfitting
- Increase dropout: `--dropout 0.3`
- Add weight decay: `--weight_decay 0.0001`
- Reduce model complexity (for GAR: reduce `--hidRNN`)
- Add more training data

### Underfitting
- Increase model capacity (for GAR: increase `--hidRNN`)
- Increase window size: `--window 336`
- Train longer: `--epochs 200`
- Reduce regularization

### Slow Training
- Reduce batch size: `--batch_size 64`
- Use GPU: `--gpu 0`
- Reduce window size: `--window 84`

## Citation

If you use these baseline models in your research, please cite:

```bibtex
@misc{baselines2025,
  title={Baseline Models for Epidemiological Forecasting: AR, VAR, and GAR},
  author={Your Name},
  year={2025}
}
```

## Additional Documentation

- **GAR-specific details**: See `GAR/README.md`
- **Model comparison**: See `GAR/MODEL_COMPARISON.md`

## License

[Add your license information here]
