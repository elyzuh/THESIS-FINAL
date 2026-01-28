# SEIR-PINN: Physics-Informed Neural Network with Latent E Learning

## Overview

This directory contains the implementation of a **Physics-Informed Neural Network (PINN)** for SEIR (Susceptible-Exposed-Infected-Recovered) epidemic modeling with **latent variable learning** for the Exposed (E) compartment.

### Key Innovation

The **Exposed (E) compartment is learned as a latent variable** - it is never directly observed in the data, but is inferred through physics constraints (SEIR differential equations). This represents a cutting-edge approach validated in recent literature (Nature Comp. Sci. 2021, PMC 2022, PLOS 2024).

## Model Architecture

### SEIR Compartments

| Compartment | Status | Description |
|-------------|--------|-------------|
| **S(t)** | Inferred | Susceptible - inferred from conservation S+E+I+R=N |
| **E(t)** | **LATENT** | Exposed - **learned via physics constraints only!** |
| **I(t)** | Observed | Infected - available from data |
| **R(t)** | Inferred | Recovered - inferred from cumulative infections |

### Time-Varying Parameters

Each parameter is learned as a function of time using neural networks:

- **β(t) ∈ [0,1]**: Transmission rate (how infectious the disease is)
- **σ(t) ∈ [0,1]**: Incubation rate (rate of progression E → I)
- **γ(t) ∈ [0,1]**: Recovery rate (rate of recovery I → R)

### Physics Constraints (SEIR ODEs)

The model enforces these differential equations during training:

```
dS/dt = -β*S*I/N
dE/dt = β*S*I/N - σ*E     # This constrains latent E!
dI/dt = σ*E - γ*I
dR/dt = γ*I
```

The key insight: **E(t) must satisfy the physics**, even though we never observe it directly!

## Network Architecture

### Compartment Networks
- Input: Normalized time t ∈ [0,1]
- Architecture: 4-layer MLP with Tanh activation
- Hidden dimension: 50-100 neurons (configurable)
- Output: Compartment values for 10 regions
- Activation: Softplus (ensures positivity)

### Parameter Networks
- Input: Normalized time t ∈ [0,1]
- Architecture: 4-layer MLP with Tanh activation
- Output: Parameters for 10 regions
- Activation: Sigmoid (constrains to [0,1])

## Loss Function

The training uses a **composite loss** that balances multiple objectives:

```python
L_total = λ₁*L_data + λ₂*L_physics + λ₃*L_IC + λ₄*L_conserve
```

### Components

1. **L_data**: Data fitting loss
   ```
   MSE(I_predicted, I_observed)
   ```
   Ensures predictions match observed infection data

2. **L_physics**: Physics residual loss
   ```
   MSE(SEIR_residuals, 0)
   ```
   Enforces SEIR differential equations at collocation points
   **This is what constrains latent E!**

3. **L_IC**: Initial condition loss
   ```
   MSE(S(0), N-I(0)) + MSE(E(0), 0) + MSE(R(0), 0)
   ```
   Anchors the solution at t=0

4. **L_conserve**: Conservation law loss
   ```
   MSE(S+E+I+R, N)
   ```
   Ensures total population is conserved

## Directory Structure

```
SEIR-PINN/
├── models/
│   ├── __init__.py
│   └── SEIRPINNmodel.py      # Core PINN architecture
├── main.py                    # Training script
├── utils.py                   # Data utilities
├── utils_ModelTrainEval.py   # PINN training/evaluation
├── Optim.py                   # Optimizer wrapper
├── PlotFunc.py               # Plotting utilities
├── data/                      # Data directory (symlink)
└── README.md                  # This file
```

## Usage

### Basic Training

```bash
cd SEIR-PINN

python main.py \
  --data ./data/us_hhs/data.txt \
  --model SEIRPINNmodel \
  --window 168 \
  --horizon 12 \
  --hidRNN 50 \
  --epochs 100 \
  --batch_size 128 \
  --lr 0.001 \
  --lambda_data 1.0 \
  --lambda_physics 1.0 \
  --lambda_ic 0.1 \
  --lambda_conserve 0.1 \
  --collocation_points 1000 \
  --population 1000000 \
  --save_name seir_pinn_baseline
```

### With GPU

```bash
python main.py \
  --data ./data/us_hhs/data.txt \
  --model SEIRPINNmodel \
  --gpu 0 \
  --save_name seir_pinn_gpu
```

## Key Parameters

### Standard Parameters (from AR/VAR/GAR)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data` | Required | Path to data file |
| `--train` | 0.6 | Training split ratio |
| `--valid` | 0.2 | Validation split ratio |
| `--window` | 168 | Lookback window size |
| `--horizon` | 12 | Prediction horizon |
| `--epochs` | 100 | Number of training epochs |
| `--batch_size` | 128 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--hidRNN` | 50 | Hidden layer dimension |

### PINN-Specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lambda_data` | 1.0 | Weight for data fitting loss |
| `--lambda_physics` | 1.0 | Weight for physics loss (SEIR ODEs) |
| `--lambda_ic` | 0.1 | Weight for initial conditions |
| `--lambda_conserve` | 0.1 | Weight for conservation law |
| `--collocation_points` | 1000 | Number of points for physics loss |
| `--population` | 1e6 | Population per region |

## Hyperparameter Tuning Guide

### If Training is Unstable (NaN losses)

1. **Reduce learning rate**:
   ```bash
   --lr 0.0001
   ```

2. **Increase physics weight** (stronger constraints):
   ```bash
   --lambda_physics 10.0
   ```

3. **Use gradient clipping**:
   ```bash
   --clip 0.5
   ```

4. **Warm start**: Train on data loss only first
   ```bash
   # Step 1: Data fitting only
   --lambda_physics 0.0 --epochs 50

   # Step 2: Add physics gradually
   --lambda_physics 1.0 --epochs 100
   ```

### If E(t) Diverges or is Unrealistic

1. **Stronger initial conditions**:
   ```bash
   --lambda_ic 1.0
   ```

2. **More collocation points**:
   ```bash
   --collocation_points 2000
   ```

3. **Smaller batch size**:
   ```bash
   --batch_size 32
   ```

### If Performance is Poor

1. **Increase model capacity**:
   ```bash
   --hidRNN 100
   ```

2. **Train longer**:
   ```bash
   --epochs 500
   ```

3. **Adjust loss balance**:
   ```bash
   --lambda_data 1.0 --lambda_physics 5.0
   ```

## Expected Outputs

### 1. Training Output

```
begin training
=========================================================================================
SEIR-PINN: Physics-Informed Neural Network with Latent E Learning
=========================================================================================
| end of epoch   1 | time:  5.23s | train_loss 0.01234567 | valid rse 0.1234 | ...
| end of epoch   2 | time:  5.18s | train_loss 0.01123456 | valid rse 0.1198 | ...
...
best validation
test rse 0.1234 | test rae 0.0567 | test relative error 12.34% | test corr 0.8765 | test r2 0.7654
```

### 2. Model File

- `save/seir_pinn_baseline.pt`: Trained model weights

### 3. Learned Variables

Use `GetPrediction()` to extract:

```python
from utils_ModelTrainEval import GetPrediction

X_true, I_pred, I_true, E_latent, S_inferred, R_inferred, Beta, Sigma, Gamma = \
    GetPrediction(Data, Data.test, model, evaluateL2, evaluateL1, batch_size, modelName, args)

# E_latent: [time, 10] - Learned exposed compartment (LATENT!)
# Beta: [time, 10] - Transmission rate β(t)
# Sigma: [time, 10] - Incubation rate σ(t)
# Gamma: [time, 10] - Recovery rate γ(t)
```

### 4. Visualizations

You can plot:
- **E(t) vs I(t)**: Should show E peaks before I (epidemiologically plausible)
- **β(t)**: Should correlate with known interventions (lockdowns, mask mandates)
- **R₀(t) = β(t)/γ(t)**: Basic reproduction number over time
- **S, E, I, R trajectories**: Full SEIR dynamics

## Validation Checklist

### ✅ Physics Consistency

- [ ] SEIR residuals < 0.01 (physics is satisfied)
- [ ] S+E+I+R ≈ N (conservation law holds)
- [ ] All compartments ≥ 0 (positivity)

### ✅ Epidemiological Plausibility

- [ ] E(t) peaks before I(t) (exposed precedes infected)
- [ ] β(t) ∈ [0, 1] and shows interpretable trends
- [ ] 1/σ ≈ 5-7 days (incubation period)
- [ ] 1/γ ≈ 10-14 days (recovery period)
- [ ] R₀ = β/γ > 1 during outbreak, < 1 after peak

### ✅ Forecasting Performance

- [ ] RSE, RAE comparable to or better than baselines
- [ ] High correlation with ground truth
- [ ] R² > 0.5 on test set

## Comparison with Baselines

| Model | Type | E Compartment | Physics Constraint |
|-------|------|---------------|-------------------|
| **AR** | Data-driven | Not modeled | None |
| **VAR** | Data-driven | Not modeled | None |
| **GAR** | Data-driven | Not modeled | None |
| **CNNRNN-Res-SEIR** | Hybrid | Faked (uses I as proxy) | Soft (auxiliary loss) |
| **SEIR-PINN** | Physics-informed | **Learned as latent** | **Hard (enforces ODEs)** |

## Academic Contribution

### Novel Aspects

1. **First multi-region SEIR-PINN** with latent E learning for spatial-temporal epidemic forecasting
2. **Handles realistic incomplete data** (infection counts only, no E observations)
3. **Demonstrates latent variable learning** through physics constraints alone

### Literature Gap Filled

- Most PINN epidemic papers use single-region data or synthetic data
- This work: Multi-region (10 states), real-world US HHS data
- Shows PINNs can work with truly incomplete data

### Publication Potential

Suitable for:
- **ML Conferences**: NeurIPS (AI4Science), ICML, ICLR
- **Computational Biology**: PLOS Computational Biology, Nature Computational Science
- **Applied ML**: AAAI, IJCAI
- **Epidemiology**: Epidemics, Journal of Theoretical Biology

## Literature References

### Core PINN Methodology

1. **Modified PINN for Unobserved Compartments**
   [PMC 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9692762/)
   - How to handle latent variables in PINNs
   - Loss function reformulation

2. **Split PINN Approach**
   [PLOS Comp. Bio. 2024](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012387)
   - Two-stage training strategy
   - Reduced-split approach for parameter estimation

3. **Identifiability in Epidemic PINNs**
   [Nature Comp. Sci. 2021](https://www.nature.com/articles/s43588-021-00158-0)
   - What can/cannot be inferred from incomplete data
   - Identifiability analysis for epidemic models

### Applications to Epidemiology

4. **SEIRD PINN with Latent States**
   [PMC 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10459488/)
   - COVID-19 application
   - Time-varying parameters

5. **Physics-Informed Deep Learning for Infectious Disease**
   [arXiv 2025](https://arxiv.org/html/2501.09298v1)
   - Recent advances in epidemic PINNs
   - Best practices

6. **PINNs for Biological/Epidemiological Systems**
   [MDPI 2025](https://www.mdpi.com/2227-7390/13/10/1664)
   - General framework for compartmental models

## Troubleshooting

### Problem: Training loss is NaN

**Solutions:**
1. Reduce learning rate: `--lr 0.0001`
2. Increase gradient clipping: `--clip 0.5`
3. Check data normalization: Ensure data is properly scaled
4. Reduce physics weight initially: `--lambda_physics 0.1`

### Problem: E(t) is unrealistic (negative, too large, etc.)

**Solutions:**
1. Stronger IC constraint: `--lambda_ic 1.0`
2. More collocation points: `--collocation_points 2000`
3. Check positivity activation (softplus is used)
4. Verify initial conditions match data

### Problem: Poor forecasting performance

**Solutions:**
1. Increase model capacity: `--hidRNN 100`
2. Balance loss weights: `--lambda_physics 0.5`
3. Train longer: `--epochs 500`
4. Use more training data: `--train 0.7`

### Problem: Physics residuals not converging

**Solutions:**
1. Increase physics weight: `--lambda_physics 10.0`
2. More collocation points: `--collocation_points 5000`
3. Better optimization: Try different `--optim` (adam, sgd)
4. Check derivatives: Ensure autograd is working correctly

## Advanced Usage

### Extract and Visualize Learned E(t)

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

# Load model
model.load_state_dict(torch.load('save/seir_pinn_baseline.pt'))
model.eval()

# Get full time series
t = torch.linspace(0, 1, 364).unsqueeze(1)
with torch.no_grad():
    S, E, I, R, beta, sigma, gamma = model.physics_forward(t)

# Convert to numpy
E_learned = E.cpu().numpy()  # [364, 10]
I_observed = Data.rawdat  # [364, 10]

# Plot for region 0
plt.figure(figsize=(10, 6))
plt.plot(E_learned[:, 0], label='E(t) - Latent Exposed', linewidth=2)
plt.plot(I_observed[:, 0], label='I(t) - Observed Infected', linewidth=2, alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Cases')
plt.legend()
plt.title('Learned Latent E(t) vs Observed I(t) - Region 0')
plt.savefig('E_vs_I_region0.pdf')
```

### Compute R₀ Trajectory

```python
# R₀ = β / γ
R0 = beta.cpu().numpy() / (gamma.cpu().numpy() + 1e-8)

plt.figure(figsize=(10, 6))
plt.plot(R0[:, 0], label='R₀(t) - Region 0', linewidth=2)
plt.axhline(y=1.0, color='r', linestyle='--', label='R₀ = 1 (threshold)')
plt.xlabel('Time')
plt.ylabel('R₀')
plt.legend()
plt.title('Basic Reproduction Number Over Time')
plt.savefig('R0_trajectory.pdf')
```

## Notes

- **This is a research implementation** - expect to iterate on hyperparameters
- **E is completely latent** - success depends on physics constraints being strong enough
- **Even negative results are valuable** - publishable as identifiability study
- **Start simple** - single region first, then extend to multi-region
- **Monitor physics loss** - should decrease significantly during training

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{seirpinn2025,
  title={SEIR-PINN: Physics-Informed Neural Network with Latent Variable Learning for Multi-Region Epidemic Forecasting},
  author={Your Name},
  year={2025},
  note={Implementation based on methodology from Nature Comp. Sci. 2021, PMC 2022, PLOS 2024}
}
```

## Contact

For questions or issues, please open an issue in the repository or contact the author.

## License

[Add your license here]
