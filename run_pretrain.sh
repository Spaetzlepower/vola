#!/usr/bin/env bash
# VolFormer-P: Vortraining auf ~20 Coins (5-Min.-Token), dann BTC 2025 (valid) und 2026 (test). GPU, ca. 1,5 Std.
# Voraussetzung: data/klines_1m.parquet und data/dataset.parquet (run_full.sh einmal gelaufen).
#   nohup bash run_pretrain.sh > logs/pretrain.log 2>&1 &
set -e
cd "$(dirname "$0")"
mkdir -p logs
CPUS=$(python -c "import os;print(len(os.sched_getaffinity(0)))")
export OMP_NUM_THREADS=$CPUS MKL_NUM_THREADS=$CPUS PYTHONUNBUFFERED=1
MIN=${PRETRAIN_MIN:-30}
echo "=== $(date) Daten laden + Vortraining ($MIN Min.) + Validierung 2025"
timeout 3h python s16_pretrain.py --mode valid --pretrain-min $MIN
echo "=== $(date) Test 2026 (gleiches Vortraining, nur Daten bis 24.12.2024)"
timeout 2h python s16_pretrain.py --mode test --no-download
echo "=== $(date) Kombination mit Ensemble"
python s14_combo.py --a vfp
tar czf keep5.tar.gz results/*vfp* results/RESULTS_combo_vfp_*.md
echo "=== $(date) FERTIG. Sicherung: keep5.tar.gz"
