"""
Grid Search for SEIR-PINN Hyperparameters
Based on Liu et al. (2023) "Epidemiology-aware Deep Learning for Infectious Disease Dynamics Prediction" (CIKM '23)

This script performs a comprehensive grid search over hyperparameters for the SEIR-PINN model,
following the methodology from Liu et al. 2023.
"""

import itertools
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import argparse
import re
import time
from datetime import datetime
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='SEIR-PINN Grid Search')
    parser.add_argument('--log_dir', type=str, default='./grid_search_results/logs',
                        help='Directory to save experiment logs')
    parser.add_argument('--model_dir', type=str, default='./grid_search_results/models',
                        help='Directory to save trained models')
    parser.add_argument('--results_csv', type=str, default='./grid_search_results/summary.csv',
                        help='Path to save results CSV')
    parser.add_argument('--checkpoint', type=str, default='./grid_search_results/checkpoint.pkl',
                        help='Path to checkpoint file for resume')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--test_mode', action='store_true',
                        help='Run with reduced grid for testing')
    parser.add_argument('--max_configs', type=int, default=None,
                        help='Maximum number of configs to run (for testing)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    return parser.parse_args()


def get_param_grid(test_mode=False):
    """
    Define hyperparameter grid based on Liu et al. (2023).

    Adapted from Epi-CNNRNN-Res hyperparameters:
    - Window size: {32, 64}
    - Hidden dimension for GRU: {5, 10}
    - Dropout rate: 0.2 (fixed)
    - Physics loss weight λ: {0.5, 1}

    PINN-specific additions:
    - lambda_ic: Initial condition loss weight
    - collocation_points: Physics sampling density
    """
    if test_mode:
        # Small grid for testing
        param_grid = {
            'hidRNN': [32],
            'lr': [0.001, 0.01],
            'dropout': [0.1],
            'epochs': [10],
            'lambda_physics': [0.5],
            'lambda_ic': [0.1],
            'collocation_points': [500],
        }
    else:
        # Full grid search
        param_grid = {
            'hidRNN': [32, 64],                    # From Liu: window size / GRU hidden
            'lr': [0.001, 0.01],                   # From Liu: learning rate
            'dropout': [0.1, 0.2],                 # From Liu: dropout rate
            'epochs': [100, 200],                  # Training duration
            'lambda_physics': [0.1, 0.5, 1.0],     # From Liu: epidemiology loss weight
            'lambda_ic': [0.1, 0.5],               # PINN-specific: initial condition
            'collocation_points': [500, 1000],     # PINN-specific: physics sampling
        }

    return param_grid


def get_fixed_params():
    """
    Fixed parameters based on Liu et al. (2023) experimental settings.

    Data split: 60% train / 20% valid / 20% test
    Dataset: US-HHS-Flu (10 regions, 364 weeks)
    """
    fixed_params = {
        'data': './data/us_hhs/data.txt',
        'model': 'SEIRPINNmodel',
        'window': 32,                      # From Liu: {32, 64}, using 32 as default
        'train': 0.6,                      # From Liu: 60% train
        'valid': 0.2,                      # From Liu: 20% valid
        'batch_size': 128,                 # Standard batch size
        'lambda_data': 1.0,                # Data loss weight (baseline)
        'lambda_conserve': 0.1,            # Conservation constraint (soft)
        'population': 1000000,             # Population for normalization
        'normalize': 0,                    # No normalization
        'metric': 1,                       # Normalized metrics
    }

    return fixed_params


def build_command(params, fixed_params, horizon, save_name, gpu=0):
    """Build command string to run main.py with given parameters."""
    cmd = ['python3', 'main.py']

    # Add fixed parameters
    for key, value in fixed_params.items():
        cmd.extend([f'--{key}', str(value)])

    # Add variable parameters
    for key, value in params.items():
        cmd.extend([f'--{key}', str(value)])

    # Add horizon and save name
    cmd.extend(['--horizon', str(horizon)])
    cmd.extend(['--save_name', save_name])

    # Add GPU
    cmd.extend(['--gpu', str(gpu)])

    return cmd


def parse_output(output_str):
    """
    Parse metrics from main.py output.

    Expected format:
    test rse 0.2402 | test rae 0.1615 | test relative error 12.34% | test corr 0.9499 | test r2 0.7654
    """
    metrics = {}

    # Extract RMSE (rse)
    match = re.search(r'test rse ([\d.]+)', output_str)
    if match:
        metrics['rmse'] = float(match.group(1))

    # Extract MAE (rae)
    match = re.search(r'test rae ([\d.]+)', output_str)
    if match:
        metrics['mae'] = float(match.group(1))

    # Extract correlation
    match = re.search(r'test corr ([\d.]+)', output_str)
    if match:
        metrics['corr'] = float(match.group(1))

    # Extract R2
    match = re.search(r'test r2 ([\d.]+)', output_str)
    if match:
        metrics['r2'] = float(match.group(1))

    return metrics


def run_experiment(config_id, params, fixed_params, horizons, log_dir, gpu=0):
    """
    Run experiment for one configuration across multiple horizons.

    Returns:
        results_dict: Dictionary with metrics for each horizon
        success: Boolean indicating if all horizons completed successfully
    """
    results = {}
    all_success = True

    for horizon in horizons:
        save_name = f'config_{config_id:03d}_h{horizon}'
        log_file = Path(log_dir) / f'{save_name}.log'

        # Build command
        cmd = build_command(params, fixed_params, horizon, save_name, gpu=gpu)

        # Run experiment
        try:
            print(f'  Running horizon {horizon}...')

            with open(log_file, 'w') as f:
                process = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=3600,  # 1 hour timeout per horizon
                    text=True
                )

            # Read output
            with open(log_file, 'r') as f:
                output = f.read()

            # Parse metrics
            metrics = parse_output(output)

            if metrics:
                results[f'h{horizon}_rmse'] = metrics.get('rmse', np.nan)
                results[f'h{horizon}_mae'] = metrics.get('mae', np.nan)
                results[f'h{horizon}_corr'] = metrics.get('corr', np.nan)
                results[f'h{horizon}_r2'] = metrics.get('r2', np.nan)
            else:
                print(f'    Warning: Could not parse metrics for horizon {horizon}')
                results[f'h{horizon}_rmse'] = np.nan
                results[f'h{horizon}_mae'] = np.nan
                results[f'h{horizon}_corr'] = np.nan
                results[f'h{horizon}_r2'] = np.nan
                all_success = False

        except subprocess.TimeoutExpired:
            print(f'    Error: Timeout for horizon {horizon}')
            results[f'h{horizon}_rmse'] = np.nan
            results[f'h{horizon}_mae'] = np.nan
            results[f'h{horizon}_corr'] = np.nan
            results[f'h{horizon}_r2'] = np.nan
            all_success = False

        except Exception as e:
            print(f'    Error running horizon {horizon}: {e}')
            results[f'h{horizon}_rmse'] = np.nan
            results[f'h{horizon}_mae'] = np.nan
            results[f'h{horizon}_corr'] = np.nan
            results[f'h{horizon}_r2'] = np.nan
            all_success = False

    return results, all_success


def load_checkpoint(checkpoint_path):
    """Load checkpoint if it exists."""
    if Path(checkpoint_path).exists():
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        return checkpoint
    return None


def save_checkpoint(checkpoint_path, completed_configs, results_df):
    """Save checkpoint for crash recovery."""
    checkpoint = {
        'completed_configs': completed_configs,
        'results': results_df,
        'timestamp': datetime.now().isoformat()
    }
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)


def main():
    args = parse_args()

    # Create directories
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_csv).parent.mkdir(parents=True, exist_ok=True)

    # Get parameter grid
    param_grid = get_param_grid(test_mode=args.test_mode)
    fixed_params = get_fixed_params()

    # Evaluation horizons (from Liu et al.)
    horizons = [1, 2, 4]

    # Generate all parameter combinations
    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(itertools.product(*param_values))

    print(f'\n{"="*80}')
    print(f'SEIR-PINN Grid Search')
    print(f'Based on Liu et al. (2023) CIKM')
    print(f'{"="*80}\n')
    print(f'Total configurations: {len(all_combinations)}')
    print(f'Horizons: {horizons}')
    print(f'Total experiments: {len(all_combinations) * len(horizons)}')

    if args.max_configs:
        all_combinations = all_combinations[:args.max_configs]
        print(f'Limited to: {len(all_combinations)} configurations (test mode)')

    print(f'\nParameter grid:')
    for key, values in param_grid.items():
        print(f'  {key}: {values}')

    print(f'\nFixed parameters:')
    for key, value in fixed_params.items():
        print(f'  {key}: {value}')

    print(f'\n{"="*80}\n')

    # Load checkpoint if resuming
    completed_configs = set()
    results_list = []

    if args.resume:
        checkpoint = load_checkpoint(args.checkpoint)
        if checkpoint:
            completed_configs = set(checkpoint['completed_configs'])
            results_list = checkpoint['results'].to_dict('records')
            print(f'Resuming from checkpoint: {len(completed_configs)} configs already completed')

    # Run grid search
    start_time = time.time()

    for config_id, combination in enumerate(tqdm(all_combinations, desc='Grid Search Progress')):
        # Skip if already completed
        if config_id in completed_configs:
            continue

        # Create parameter dictionary
        params = dict(zip(param_keys, combination))

        print(f'\n[Config {config_id+1}/{len(all_combinations)}]')
        print(f'Parameters: {params}')

        # Run experiment
        results, success = run_experiment(
            config_id=config_id,
            params=params,
            fixed_params=fixed_params,
            horizons=horizons,
            log_dir=args.log_dir,
            gpu=args.gpu
        )

        # Add config parameters to results
        results['config_id'] = config_id
        results.update(params)
        results['success'] = success
        results['timestamp'] = datetime.now().isoformat()

        # Append to results list
        results_list.append(results)

        # Save results incrementally
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(args.results_csv, index=False)

        # Update checkpoint
        completed_configs.add(config_id)
        save_checkpoint(args.checkpoint, list(completed_configs), results_df)

        print(f'Results: h1_rmse={results.get("h1_rmse", "N/A"):.4f}, '
              f'h1_corr={results.get("h1_corr", "N/A"):.4f}')

    # Final summary
    elapsed_time = time.time() - start_time
    hours = elapsed_time / 3600

    print(f'\n{"="*80}')
    print(f'Grid Search Complete!')
    print(f'{"="*80}\n')
    print(f'Total time: {hours:.2f} hours')
    print(f'Results saved to: {args.results_csv}')
    print(f'Logs saved to: {args.log_dir}')
    print(f'\nNext steps:')
    print(f'  python analyze_results.py --results_csv {args.results_csv}')
    print(f'\n{"="*80}\n')


if __name__ == '__main__':
    main()
