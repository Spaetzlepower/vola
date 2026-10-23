#!/usr/bin/env bash
# VolFormer (eigener kleiner Transformer) + VolNet v2 + Kombinationen. GPU. Voraussetzung: run_full.sh einmal gelaufen.
#   nohup bash run_volformer.sh > logs/volformer.log 2>&1 &
set -e
cd "$(dirname "$0")"
mkdir -p logs
CPUS=$(python -c "import os;print(len(os.sched_getaffinity(0)))")
export OMP_NUM_THREADS=$CPUS MKL_NUM_THREADS=$CPUS PYTHONUNBUFFERED=1
echo "=== $(date) VolFormer: Suche auf 2025"
python s15_volformer.py --mode valid --search
echo "=== $(date) VolFormer: Test 2026"
python s15_volformer.py --mode test
echo "=== $(date) VolFormer + Ensemble kombinieren"
python s15_volformer.py --mode valid                   # Vorhersagen 2025 mit bestem Setup (für Kombinationsgewichte)
python s14_combo.py --a vf
[ -f results/vol_preds_nn_test.parquet ] && python s14_combo.py --a nn || true
tar czf keep4.tar.gz results/*.md results/*.csv results/*.json
echo "=== $(date) FERTIG. Sicherung: keep4.tar.gz"
