#!/usr/bin/env bash
# Komplettlauf im Hintergrund. Start:   nohup bash run_all.sh valid > logs/run_valid.log 2>&1 &
# Fortschritt ansehen:                 tail -f logs/run_valid.log
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:-valid}"          # valid = Tuning auf 2025, test = finaler Lauf ab 2026
mkdir -p logs
echo "=== $(date) Start, Modus $MODE"
[ -f data/klines_1m.parquet ] || python s01_download.py --futures
[ -f data/dataset.parquet ]   || python s02_features.py
python s03_models.py --mode "$MODE"
python s04_foundation.py --mode "$MODE" --models chronos2 timestral --every 6
python s05_ensemble.py --mode "$MODE"
python s06_polymarket.py
python s07_options.py --mode long
python s08_pm_check.py
python s10_pm_vol.py
echo "=== $(date) Fertig. Ergebnisse: results/RESULTS_${MODE}.md und results/RESULTS_foundation_${MODE}.md"
