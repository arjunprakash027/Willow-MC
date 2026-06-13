#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Running model backtests in $PROJECT_ROOT..."
cd "$PROJECT_ROOT"

echo "----------------------------------------"
echo "1. Backtesting IPL Model..."
.venv/bin/python src/evaluation.py \
  --database data/willow.db \
  --run-model outputs/t20_ipl_run_model.txt \
  --wicket-model outputs/t20_ipl_wicket_model.txt \
  --dataset ipl_ball_by_ball \
  --n-matches 100 \
  --output-json outputs/ipl_backtest_results.json

echo "----------------------------------------"
echo "2. Backtesting T20 Model..."
.venv/bin/python src/evaluation.py \
  --database data/willow.db \
  --run-model outputs/t20_int_run_model.txt \
  --wicket-model outputs/t20_int_wicket_model.txt \
  --dataset t20_ball_by_ball \
  --n-matches 100 \
  --output-json outputs/t20_backtest_results.json
