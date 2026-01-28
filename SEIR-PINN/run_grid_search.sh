#!/bin/bash

# SEIR-PINN Grid Search Runner
# Based on Liu et al. (2023) CIKM methodology

echo "========================================"
echo "SEIR-PINN Grid Search"
echo "Based on Liu et al. (2023) CIKM"
echo "========================================"
echo ""

# Create output directories
echo "Creating output directories..."
mkdir -p grid_search_results/logs
mkdir -p grid_search_results/models
echo "Done."
echo ""

# Get GPU ID (default to 0)
GPU=${1:-0}
echo "Using GPU: $GPU"
echo ""

# Generate timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="grid_search_results/grid_search_${TIMESTAMP}.log"

echo "Starting grid search..."
echo "Log file: $LOG_FILE"
echo ""
echo "This will take approximately 16-20 hours for 192 configurations."
echo "Progress will be saved incrementally - safe to interrupt and resume."
echo ""

# Run grid search with logging
python3 grid_search.py \
    --gpu $GPU \
    --log_dir ./grid_search_results/logs \
    --model_dir ./grid_search_results/models \
    --results_csv ./grid_search_results/summary.csv \
    --checkpoint ./grid_search_results/checkpoint.pkl \
    2>&1 | tee "$LOG_FILE"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Grid search completed successfully!"
    echo "========================================"
    echo ""
    echo "Results saved to: grid_search_results/summary.csv"
    echo "Full log: $LOG_FILE"
    echo ""
    echo "Next step: Analyze results"
    echo "  python analyze_results.py --results_csv grid_search_results/summary.csv"
    echo ""
else
    echo ""
    echo "========================================"
    echo "Grid search failed or interrupted"
    echo "========================================"
    echo ""
    echo "To resume from checkpoint, run:"
    echo "  python3 grid_search.py --resume --gpu $GPU"
    echo ""
fi
