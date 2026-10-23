#!/usr/bin/env bash
# Eigenes Modell VolNet (GPU empfohlen). Voraussetzung: run_full.sh ist einmal gelaufen (Daten, s03/s05-Ergebnisse).
#   nohup bash run_volnet.sh > logs/volnet.log 2>&1 &
set -e
cd "$(dirname "$0")"
mkdir -p logs
CPUS=$(python -c "import os;print(len(os.sched_getaffinity(0)))")
export OMP_NUM_THREADS=$CPUS MKL_NUM_THREADS=$CPUS PYTHONUNBUFFERED=1
echo "=== $(date) VolNet: Hyperparameter-Suche auf 2025"
python s12_volnet.py --mode valid --search
echo "=== $(date) VolNet: Test 2026 (einmalig, bestes Setup aus der Suche)"
python s12_volnet.py --mode test
# Polymarket-Auswertungen mit VolNet als zusätzlichem fairen Wert neu rechnen
python s10_pm_vol.py
[ -f inputs/pm_hourly_dump.json ] && python s11_pm_hourly.py --model nn || true
tar czf keep3.tar.gz results/*.md results/*.csv results/volnet_best.json data/fm_features.parquet data/dvol_1h.parquet
echo "=== $(date) FERTIG. Sicherung: keep3.tar.gz"
