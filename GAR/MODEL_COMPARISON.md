# Baseline Model Comparison: AR, VAR, and GAR

## Model Architectures

### AR (Autoregressive)
**File**: `AR/models/ARmodel.py`

```python
# Architecture
- Input: [batch, window, m]
- Weight: [window, m]  # Linear weights per time step
- Bias: [m]
- Output: [batch, m]

# Forward Pass
output = sum(x * weight) + bias  # Element-wise per time step
```

**Key Characteristics**:
- Each variable's prediction depends only on its own past values
- Linear model with `window × m` parameters
- No cross-variable dependencies
- Simplest baseline model

---

### VAR (Vector Autoregressive)
**File**: `VAR/models/VARmodel.py`

```python
# Architecture
- Input: [batch, window (p), m]
- Weight: [p, m, m]  # One m×m matrix per lag
- Bias: [m]
- Output: [batch, m]

# Forward Pass
for each lag l in range(p):
    y += x[:, l, :] @ weight[l]  # Matrix multiplication
y = y + bias
```

**Key Characteristics**:
- Each variable's prediction depends on past values of ALL variables
- Linear model with `p × m × m` parameters
- Captures cross-variable dependencies
- More parameters than AR but still linear

---

### GAR (Generalized Autoregressive)
**File**: `GAR/models/GARmodel.py`

```python
# Architecture
- Input: [batch, window, m]
- fc1: Linear(window × m, hidRNN)  # Hidden layer
- dropout: Dropout(p)
- fc2: Linear(hidRNN, m)  # Output layer
- Output: [batch, m]

# Forward Pass
x = x.view(batch, -1)           # Flatten to [batch, window×m]
x = ReLU(fc1(x))                # Nonlinear transformation
x = dropout(x)                   # Regularization
output = fc2(x)                  # Final prediction
```

**Key Characteristics**:
- Nonlinear model using feedforward neural network
- Parameters: `(window×m + 1)×hidRNN + (hidRNN + 1)×m`
- Captures complex nonlinear temporal patterns
- Includes dropout for regularization
- More flexible than AR/VAR but requires more data

---

## Parameter Comparison

Assuming:
- `m = 10` (number of regions/variables)
- `window = 168` (lookback window)
- `hidRNN = 50` (for GAR)

| Model | Parameter Count | Formula |
|-------|----------------|---------|
| **AR** | 1,680 + 10 = **1,690** | `window × m + m` |
| **VAR** | 168 × 10 × 10 + 10 = **16,810** | `window × m × m + m` |
| **GAR** | (168×10)×50 + 50×10 + 50 + 10 = **84,560** | `(window×m + 1)×hidRNN + (hidRNN + 1)×m` |

---

## Complexity vs. Expressiveness

```
Expressiveness:  AR  <  VAR  <<  GAR
Parameters:      AR  <  VAR  <   GAR
Training Speed:  AR  >  VAR  >   GAR
Overfitting Risk: AR  <  VAR  <   GAR
```

---

## When to Use Each Model

### Use AR when:
- You have limited data
- Variables are largely independent
- You need a simple, interpretable baseline
- Fast training is important

### Use VAR when:
- You have moderate amounts of data
- Cross-variable dependencies are important
- You want to maintain linearity for interpretability
- You need to understand variable interactions

### Use GAR when:
- You have sufficient training data
- Relationships are likely nonlinear
- You want maximum predictive performance
- Interpretability is less critical
- You can afford longer training times

---

## Code Structure Consistency

All three models follow the same project structure:

```
{AR,VAR,GAR}/
├── models/
│   ├── __init__.py
│   └── {AR,VAR,GAR}model.py
├── main.py
├── utils.py
├── utils_ModelTrainEval.py
├── Optim.py
├── PlotFunc.py
├── PlotData.py
├── GenerateAdjacentMatrix.py
├── log_parser.py
├── cut_log.py
└── data/
```

---

## Training Command Comparison

### AR
```bash
python main.py --data ./data/file.csv --model ARmodel --window 168 --horizon 12
```

### VAR
```bash
python main.py --data ./data/file.csv --model VARmodel --window 168 --horizon 12
```

### GAR
```bash
python main.py --data ./data/file.csv --model GARmodel --window 168 --horizon 12 \
               --hidRNN 50 --dropout 0.2
```

**Note**: GAR requires additional hyperparameters:
- `--hidRNN`: Hidden layer size (default: 50)
- `--dropout`: Dropout rate (default: 0.2)

---

## Performance Metrics

All models output the same evaluation metrics:
- **RSE**: Root Squared Error
- **RAE**: Relative Absolute Error
- **Relative Error**: Mean relative error (%)
- **Correlation**: Average correlation coefficient
- **R²**: Coefficient of determination

---

## Implementation Details

### Shared Components
All models use:
- Same data preprocessing pipeline (`utils.py`)
- Same training loop (`utils_ModelTrainEval.py`)
- Same optimizer wrapper (`Optim.py`)
- Same plotting utilities (`PlotFunc.py`, `PlotData.py`)

### Model-Specific Differences
Only the `forward()` method differs:
- **AR**: Weighted sum
- **VAR**: Matrix multiplication per lag
- **GAR**: Multi-layer neural network

---

## Extension Points

### For AR:
- Could add lag selection
- Could incorporate seasonal components

### For VAR:
- Could add variable selection (sparse VAR)
- Could incorporate structural constraints

### For GAR:
- Could add more layers (deeper network)
- Could add recurrent connections (RNN/LSTM)
- Could add attention mechanisms
- Could add residual connections

---

## Summary

| Aspect | AR | VAR | GAR |
|--------|-----|-----|-----|
| **Type** | Linear, univariate dependencies | Linear, multivariate | Nonlinear, multivariate |
| **Complexity** | Low | Medium | High |
| **Data Requirements** | Low | Medium | High |
| **Interpretability** | High | Medium | Low |
| **Flexibility** | Low | Medium | High |
| **Training Time** | Fast | Medium | Slow |
| **Best For** | Simple trends | Linear interactions | Complex patterns |

Choose the model based on your specific needs regarding data availability, computational resources, and the complexity of patterns in your data.
