# SEIR-PINN Grid Search Implementation

Based on Liu et al. (2023) "Epidemiology-aware Deep Learning for Infectious Disease Dynamics Prediction" (CIKM '23)

## Overview

This directory contains a comprehensive grid search implementation for the SEIR-PINN model, following the hyperparameter search methodology from Liu et al. (2023). The grid search will evaluate 192 different hyperparameter configurations to find the optimal settings.

## Files Created

- `grid_search.py` - Main grid search script that runs all configurations
- `analyze_results.py` - Results analyzer that creates comparison tables
- `run_grid_search.sh` - Bash wrapper for convenient execution
- `GRID_SEARCH_README.md` - This file

## Prerequisites

### Required Python Packages

Ensure you have the following packages installed:

```bash
# Install via pip or conda
pip install torch numpy scikit-learn matplotlib tqdm pandas tabulate

# Or with conda
conda install pytorch numpy scikit-learn matplotlib tqdm pandas
pip install tabulate
```

### Data

The grid search uses the US-HHS-Flu dataset located at:
```
./data/us_hhs/data.txt
```

This should already exist (symlinked from the main data directory).

## Grid Search Parameters

### Hyperparameter Grid (Based on Liu et al. 2023)

The grid search explores the following hyperparameters:

| Parameter | Values | Source |
|-----------|--------|--------|
| `hidRNN` | [32, 64] | Liu: window size / GRU hidden |
| `lr` | [0.001, 0.01] | Liu: learning rate |
| `dropout` | [0.1, 0.2] | Liu: dropout rate |
| `epochs` | [100, 200] | Training duration |
| `lambda_physics` | [0.1, 0.5, 1.0] | Liu: epidemiology loss weight λ |
| `lambda_ic` | [0.1, 0.5] | PINN-specific: initial condition |
| `collocation_points` | [500, 1000] | PINN-specific: physics sampling |

**Total configurations**: 2 × 2 × 2 × 2 × 3 × 2 × 2 = **192 configs**

### Fixed Parameters (From Liu et al.)

- **Data split**: 60% train / 20% valid / 20% test
- **Window size**: 32
- **Batch size**: 128
- **Lambda data**: 1.0
- **Lambda conserve**: 0.1
- **Population**: 1,000,000

### Evaluation Horizons

Each configuration is evaluated at three prediction horizons (like Liu et al.):
- h = 1 (1-step ahead)
- h = 2 (2-step ahead)
- h = 4 (4-step ahead)

## Usage

### Quick Start

```bash
cd /Users/lawrencetulod/THESIS-FINAL/SEIR-PINN

# Test with 2 configurations first (recommended)
python3 grid_search.py --test_mode --max_configs 2 --gpu 0

# Run full grid search
bash run_grid_search.sh 0  # 0 is GPU ID
```

### Manual Execution

```bash
# Full grid search
python3 grid_search.py \
    --gpu 0 \
    --log_dir ./grid_search_results/logs \
    --model_dir ./grid_search_results/models \
    --results_csv ./grid_search_results/summary.csv \
    --checkpoint ./grid_search_results/checkpoint.pkl

# Resume from checkpoint if interrupted
python3 grid_search.py --resume --gpu 0
```

### Analyzing Results

After the grid search completes:

```bash
# Generate analysis and comparison tables
python3 analyze_results.py --results_csv ./grid_search_results/summary.csv

# View top 10 configurations
python3 analyze_results.py --results_csv ./grid_search_results/summary.csv --top_k 10
```

## Expected Runtime

- **Per configuration**: ~5 minutes (with 100 epochs)
- **Total configurations**: 192
- **Estimated total time**: 192 × 5 min = 960 minutes = **16 hours**

With 200 epochs, expect ~20 hours total.

The script saves progress incrementally, so it can be safely interrupted and resumed.

## Output Structure

```
grid_search_results/
├── logs/
│   ├── config_000_h1.log
│   ├── config_000_h2.log
│   ├── config_000_h4.log
│   ├── config_001_h1.log
│   └── ... (192 × 3 = 576 log files)
├── models/
│   ├── config_000_h1.pt (saved models, optional)
│   └── ...
├── summary.csv              # All results in tabular format
├── comparison_table.txt     # Formatted like Liu et al. Table 1
├── best_config.txt          # Best hyperparameters found
├── checkpoint.pkl           # For crash recovery
└── grid_search_YYYYMMDD_HHMMSS.log  # Full execution log
```

## Results Format

### summary.csv Columns

- `config_id`: Configuration number (0-191)
- `hidRNN`, `lr`, `dropout`, `epochs`, `lambda_physics`, `lambda_ic`, `collocation_points`: Hyperparameters
- `h1_rmse`, `h1_mae`, `h1_corr`, `h1_r2`: Horizon 1 metrics
- `h2_rmse`, `h2_mae`, `h2_corr`, `h2_r2`: Horizon 2 metrics
- `h4_rmse`, `h4_mae`, `h4_corr`, `h4_r2`: Horizon 4 metrics
- `success`: Boolean indicating if all horizons completed
- `timestamp`: When the config was run

### comparison_table.txt Format

Matches Liu et al. (2023) Table 1:

```
H | Metrics | Best    | Median  | Worst   | Std
--+---------+---------+---------+---------+-------
1 | RMSE    | 0.2402  | 0.2514  | 0.3031  | 0.0145
1 | MAE     | 0.1615  | 0.1751  | 0.2163  | 0.0102
1 | CORR    | 0.9511  | 0.9508  | 0.9301  | 0.0052
2 | RMSE    | 0.3091  | 0.3467  | 0.4142  | 0.0231
...
```

## Liu et al. (2023) Baseline Results

Target to match or beat (Epi-CNNRNN-Res on US-HHS-Flu):

| H | RMSE   | MAE    | CORR   |
|---|--------|--------|--------|
| 1 | 0.2402 | 0.1615 | 0.9499 |
| 2 | 0.3091 | 0.2093 | 0.9291 |
| 4 | 0.3952 | 0.2670 | 0.8812 |

**Goal**: SEIR-PINN should match or exceed these results after grid search optimization.

## Troubleshooting

### Missing Dependencies

```bash
# If you see "ModuleNotFoundError"
pip install torch numpy scikit-learn matplotlib tqdm pandas tabulate
```

### GPU Out of Memory

```bash
# Reduce batch size in fixed_params
# Edit grid_search.py line ~95:
'batch_size': 64,  # Instead of 128
```

### Crash Recovery

If the grid search crashes or is interrupted:

```bash
# Resume from checkpoint
python3 grid_search.py --resume --gpu 0

# The checkpoint saves after each configuration
# No work will be lost
```

### Verify Test Run First

Always test with a small subset before the full run:

```bash
# Test with just 2 configs (takes ~10 minutes)
python3 grid_search.py --test_mode --max_configs 2 --gpu 0

# Check the output
cat grid_search_results/summary.csv
```

## Advanced Usage

### Custom Parameter Grid

Edit `grid_search.py` function `get_param_grid()` to modify search space:

```python
param_grid = {
    'hidRNN': [32, 64, 128],  # Add 128
    'lr': [0.0001, 0.001, 0.01],  # Add 0.0001
    # ... etc
}
```

### Parallel Execution

To speed up, run multiple grid searches in parallel on different GPUs:

```bash
# Split configs into chunks manually and run on different GPUs
python3 grid_search.py --max_configs 96 --gpu 0 &  # Configs 0-95
python3 grid_search.py --max_configs 96 --gpu 1 &  # Would need to offset start
```

## Next Steps After Grid Search

1. **Analyze results**:
   ```bash
   python3 analyze_results.py --results_csv ./grid_search_results/summary.csv
   ```

2. **Identify best configuration**:
   - Check `best_config.txt`
   - Look at hyperparameter trends in the analysis

3. **Retrain with best config**:
   ```bash
   python3 main.py \
       --data ./data/us_hhs/data.txt \
       --model SEIRPINNmodel \
       --hidRNN <best_value> \
       --lr <best_value> \
       --lambda_physics <best_value> \
       # ... use values from best_config.txt
   ```

4. **Compare with baselines**:
   - AR, GAR, VAR models
   - Liu et al. Epi-CNNRNN-Res results

5. **Write up for thesis/paper**:
   - Use comparison_table.txt for results table
   - Discuss hyperparameter sensitivity
   - Compare SEIR-PINN vs Liu's approach

## References

Liu, Y., et al. (2023). "Epidemiology-aware Deep Learning for Infectious Disease Dynamics Prediction."
In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management (CIKM '23).

## Contact

For questions or issues with the grid search implementation, please refer to the main thesis documentation.
