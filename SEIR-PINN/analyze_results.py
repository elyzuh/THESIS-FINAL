"""
Analyze Grid Search Results for SEIR-PINN
Parse results and generate comparison tables in Liu et al. (2023) format
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from tabulate import tabulate


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze SEIR-PINN Grid Search Results')
    parser.add_argument('--results_csv', type=str, default='./grid_search_results/summary.csv',
                        help='Path to results CSV from grid search')
    parser.add_argument('--output_dir', type=str, default='./grid_search_results',
                        help='Directory to save analysis outputs')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top configurations to display')
    return parser.parse_args()


def load_results(csv_path):
    """Load results from CSV file."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(f'Results file not found: {csv_path}')

    df = pd.read_csv(csv_path)

    # Filter only successful runs
    if 'success' in df.columns:
        df_success = df[df['success'] == True].copy()
        print(f'Loaded {len(df)} total configs, {len(df_success)} successful')
    else:
        df_success = df.copy()
        print(f'Loaded {len(df)} configs')

    return df_success


def find_best_config(df, metric='h1_rmse', minimize=True):
    """Find best configuration based on specified metric."""
    if metric not in df.columns:
        print(f'Warning: Metric {metric} not found in results')
        return None

    # Remove NaN values
    df_valid = df[df[metric].notna()].copy()

    if len(df_valid) == 0:
        print(f'No valid values found for metric {metric}')
        return None

    if minimize:
        best_idx = df_valid[metric].idxmin()
    else:
        best_idx = df_valid[metric].idxmax()

    return df_valid.loc[best_idx]


def create_comparison_table(df, horizons=[1, 2, 4]):
    """
    Create comparison table in Liu et al. (2023) format.

    Format:
    H | Metrics | Best | Median | Worst | Std
    --+---------+------+--------+-------+-----
    1 | RMSE    | ...  | ...    | ...   | ...
    1 | MAE     | ...  | ...    | ...   | ...
    1 | CORR    | ...  | ...    | ...   | ...
    """
    table_data = []

    for h in horizons:
        for metric_name, metric_col in [('RMSE', f'h{h}_rmse'),
                                         ('MAE', f'h{h}_mae'),
                                         ('CORR', f'h{h}_corr')]:
            if metric_col not in df.columns:
                continue

            # Get valid values
            values = df[metric_col].dropna()

            if len(values) == 0:
                continue

            # Calculate statistics
            if metric_name == 'CORR':
                best = values.max()
                worst = values.min()
            else:
                best = values.min()
                worst = values.max()

            median = values.median()
            std = values.std()

            table_data.append([h, metric_name,
                              f'{best:.4f}',
                              f'{median:.4f}',
                              f'{worst:.4f}',
                              f'{std:.4f}'])

    return table_data


def print_comparison_table(table_data):
    """Print comparison table with nice formatting."""
    headers = ['H', 'Metrics', 'Best', 'Median', 'Worst', 'Std']
    print('\n' + '='*80)
    print('SEIR-PINN Grid Search Results (US-HHS-Flu Dataset)')
    print('='*80 + '\n')
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    print('')


def print_best_config(best_config, param_cols):
    """Print best configuration details."""
    print('\n' + '='*80)
    print('Best Configuration (by h=1 RMSE)')
    print('='*80 + '\n')

    print('Hyperparameters:')
    for col in param_cols:
        if col in best_config.index:
            print(f'  {col}: {best_config[col]}')

    print('\nPerformance:')
    for h in [1, 2, 4]:
        rmse_col = f'h{h}_rmse'
        mae_col = f'h{h}_mae'
        corr_col = f'h{h}_corr'

        if rmse_col in best_config.index:
            print(f'  h={h}:')
            print(f'    RMSE: {best_config[rmse_col]:.4f}')
            print(f'    MAE:  {best_config[mae_col]:.4f}')
            print(f'    CORR: {best_config[corr_col]:.4f}')

    print('\n' + '='*80 + '\n')


def save_best_config(best_config, output_path, param_cols):
    """Save best configuration to text file."""
    with open(output_path, 'w') as f:
        f.write('='*80 + '\n')
        f.write('Best SEIR-PINN Configuration\n')
        f.write('='*80 + '\n\n')

        f.write('Hyperparameters:\n')
        for col in param_cols:
            if col in best_config.index:
                f.write(f'  {col}: {best_config[col]}\n')

        f.write('\nPerformance:\n')
        for h in [1, 2, 4]:
            rmse_col = f'h{h}_rmse'
            mae_col = f'h{h}_mae'
            corr_col = f'h{h}_corr'

            if rmse_col in best_config.index:
                f.write(f'  h={h}:\n')
                f.write(f'    RMSE: {best_config[rmse_col]:.4f}\n')
                f.write(f'    MAE:  {best_config[mae_col]:.4f}\n')
                f.write(f'    CORR: {best_config[corr_col]:.4f}\n')

        f.write('\n' + '='*80 + '\n')

    print(f'Best config saved to: {output_path}')


def save_comparison_table(table_data, output_path):
    """Save comparison table to text file."""
    with open(output_path, 'w') as f:
        f.write('='*80 + '\n')
        f.write('SEIR-PINN Grid Search Results (US-HHS-Flu Dataset)\n')
        f.write('='*80 + '\n\n')

        headers = ['H', 'Metrics', 'Best', 'Median', 'Worst', 'Std']
        f.write(tabulate(table_data, headers=headers, tablefmt='grid'))
        f.write('\n')

    print(f'Comparison table saved to: {output_path}')


def analyze_hyperparameter_effects(df, param_cols, metric='h1_rmse'):
    """Analyze effect of each hyperparameter on performance."""
    print('\n' + '='*80)
    print(f'Hyperparameter Analysis (Effect on {metric})')
    print('='*80 + '\n')

    for param in param_cols:
        if param not in df.columns:
            continue

        print(f'{param}:')
        grouped = df.groupby(param)[metric].agg(['mean', 'std', 'min', 'max'])
        print(grouped.to_string())
        print('')


def display_top_configs(df, param_cols, top_k=5):
    """Display top k configurations."""
    print('\n' + '='*80)
    print(f'Top {top_k} Configurations (by h=1 RMSE)')
    print('='*80 + '\n')

    # Sort by h1_rmse
    df_sorted = df.sort_values('h1_rmse')

    display_cols = ['config_id'] + param_cols + ['h1_rmse', 'h1_mae', 'h1_corr']
    display_cols = [col for col in display_cols if col in df.columns]

    print(df_sorted[display_cols].head(top_k).to_string(index=False))
    print('')


def compare_with_liu_baseline():
    """Display Liu et al. (2023) baseline results for comparison."""
    print('\n' + '='*80)
    print('Liu et al. (2023) Baseline - Epi-CNNRNN-Res')
    print('US-HHS-Flu Dataset, 10 regions, 364 weeks')
    print('='*80 + '\n')

    baseline_data = [
        [1, 'RMSE', 0.2402],
        [1, 'MAE', 0.1615],
        [1, 'CORR', 0.9499],
        [2, 'RMSE', 0.3091],
        [2, 'MAE', 0.2093],
        [2, 'CORR', 0.9291],
        [4, 'RMSE', 0.3952],
        [4, 'MAE', 0.2670],
        [4, 'CORR', 0.8812],
    ]

    headers = ['H', 'Metric', 'Value']
    print(tabulate(baseline_data, headers=headers, tablefmt='grid'))
    print('\nGoal: Match or beat these numbers with SEIR-PINN\n')


def main():
    args = parse_args()

    # Load results
    print('\nLoading results...')
    df = load_results(args.results_csv)

    if len(df) == 0:
        print('No results to analyze')
        return

    # Identify parameter columns
    param_cols = ['hidRNN', 'lr', 'dropout', 'epochs',
                  'lambda_physics', 'lambda_ic', 'collocation_points']
    param_cols = [col for col in param_cols if col in df.columns]

    # Create comparison table
    table_data = create_comparison_table(df)
    print_comparison_table(table_data)

    # Save comparison table
    output_path = Path(args.output_dir) / 'comparison_table.txt'
    save_comparison_table(table_data, output_path)

    # Find and print best configuration
    best_config = find_best_config(df, metric='h1_rmse', minimize=True)
    if best_config is not None:
        print_best_config(best_config, param_cols)

        # Save best config
        output_path = Path(args.output_dir) / 'best_config.txt'
        save_best_config(best_config, output_path, param_cols)

    # Display top k configurations
    display_top_configs(df, param_cols, top_k=args.top_k)

    # Analyze hyperparameter effects
    analyze_hyperparameter_effects(df, param_cols)

    # Compare with Liu et al. baseline
    compare_with_liu_baseline()

    print('\n' + '='*80)
    print('Analysis Complete!')
    print('='*80 + '\n')


if __name__ == '__main__':
    main()
