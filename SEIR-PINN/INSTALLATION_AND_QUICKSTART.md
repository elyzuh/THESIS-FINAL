# SEIR-PINN Grid Search - Installation & Quick Start

## Step 1: Install Dependencies

```bash
cd /Users/lawrencetulod/THESIS-FINAL/SEIR-PINN

# Install required packages
pip3 install -r requirements_grid_search.txt

# Or install individually
pip3 install torch numpy scikit-learn matplotlib tqdm pandas tabulate
```

## Step 2: Verify Installation

```bash
# Test that all imports work
python3 -c "import torch; import numpy; import sklearn; import matplotlib; import tqdm; import pandas; import tabulate; print('All dependencies installed successfully!')"
```

## Step 3: Quick Test (Recommended First!)

Before running the full 16-hour grid search, test with just 2 configurations:

```bash
# Test run (takes ~5-10 minutes)
python3 grid_search.py --test_mode --max_configs 2 --gpu 0

# Check results
cat grid_search_results/summary.csv
```

**Expected output**: You should see 2 rows with metrics for h1, h2, h4.

## Step 4: Full Grid Search

Once the test works, launch the full grid search:

```bash
# Full grid search (16-20 hours)
bash run_grid_search.sh 0  # 0 = GPU ID

# Or run manually
python3 grid_search.py --gpu 0
```

**Note**: This will run 192 configurations × 3 horizons = 576 experiments.

## Step 5: Monitor Progress

The grid search saves progress incrementally. You can check status anytime:

```bash
# Check how many configs completed
wc -l grid_search_results/summary.csv

# View recent results
tail grid_search_results/summary.csv

# Check specific log
tail -f grid_search_results/logs/config_050_h1.log
```

## Step 6: Analyze Results

After completion (or even during the run):

```bash
# Generate analysis and comparison tables
python3 analyze_results.py --results_csv grid_search_results/summary.csv

# View outputs
cat grid_search_results/best_config.txt
cat grid_search_results/comparison_table.txt
```

## Troubleshooting

### Issue: ModuleNotFoundError

**Problem**: Missing Python packages

**Solution**:
```bash
pip3 install scikit-learn tqdm pandas tabulate
```

### Issue: CUDA out of memory

**Problem**: GPU memory insufficient

**Solution**: Edit `grid_search.py` line ~95 and reduce batch_size:
```python
'batch_size': 64,  # Instead of 128
```

### Issue: Grid search interrupted

**Problem**: Crash or manual interruption

**Solution**: Resume from checkpoint:
```bash
python3 grid_search.py --resume --gpu 0
```

All completed configurations are saved - no work is lost!

## What Gets Created

```
grid_search_results/
├── summary.csv                    # Main results file
├── best_config.txt                # Optimal hyperparameters
├── comparison_table.txt           # Liu et al. format table
├── checkpoint.pkl                 # For crash recovery
├── grid_search_20260128_*.log     # Full execution log
├── logs/
│   └── config_XXX_hY.log (576 files)
└── models/
    └── config_XXX_hY.pt (optional saved models)
```

## Understanding Results

### Best Configuration (best_config.txt)

Shows the hyperparameters that achieved lowest h=1 RMSE:

```
hidRNN: 64
lr: 0.001
dropout: 0.1
epochs: 200
lambda_physics: 1.0
lambda_ic: 0.5
collocation_points: 1000

Performance:
  h=1: RMSE: 0.2402, MAE: 0.1615, CORR: 0.9499
  ...
```

### Comparison Table (comparison_table.txt)

Shows best/median/worst across all configs:

```
H | Metrics | Best    | Median  | Worst
--+---------+---------+---------+-------
1 | RMSE    | 0.2402  | 0.2600  | 0.3100
1 | MAE     | 0.1615  | 0.1800  | 0.2200
1 | CORR    | 0.9511  | 0.9450  | 0.9200
```

Compare these numbers with Liu et al. baseline:
- **Liu's Epi-CNNRNN-Res**: h=1 RMSE=0.2402, MAE=0.1615, CORR=0.9499
- **Your SEIR-PINN**: Should match or beat these after optimization

## Timeline

- **Installation**: 2 minutes
- **Test run**: 5-10 minutes
- **Full grid search**: 16-20 hours (can run overnight)
- **Analysis**: 5 minutes

## Next Steps After Grid Search

1. ✅ **Identify best config**: Check `best_config.txt`

2. **Retrain best model**:
   ```bash
   python3 main.py \
       --data ./data/us_hhs/data.txt \
       --model SEIRPINNmodel \
       --hidRNN 64 \
       --lr 0.001 \
       --epochs 200 \
       --lambda_physics 1.0 \
       --lambda_ic 0.5 \
       --collocation_points 1000 \
       --save_name seir_pinn_best
   ```

3. **Compare with baselines**: AR, GAR, VAR models

4. **Write thesis section**: Use `comparison_table.txt` for results

5. **Visualize trends**: Plot hyperparameter effects on performance

## Reference

This implementation follows:

**Liu, Y., et al. (2023).** "Epidemiology-aware Deep Learning for Infectious Disease Dynamics Prediction."
*In Proceedings of CIKM '23*.

Their Epi-CNNRNN-Res model serves as our baseline to beat.

## Questions?

See `GRID_SEARCH_README.md` for detailed documentation.
