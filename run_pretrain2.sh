#!/usr/bin/env bash
# VolFormer-P v2: HAR-Residual-Kopf + HAR-Kanäle + Markt-Kanäle (alle Coins bis heute). GPU, ca. 1,5 Std.
# Lädt die Daten sofort und wartet mit dem Training, bis run_pretrain.sh (v1) fertig ist.
#   nohup bash run_pretrain2.sh > logs/pretrain_v2.log 2>&1 &
set -e
cd "$(dirname "$0")"
mkdir -p logs
CPUS=$(python -c "import os;print(len(os.sched_getaffinity(0)))")
export OMP_NUM_THREADS=$CPUS MKL_NUM_THREADS=$CPUS PYTHONUNBUFFERED=1
MIN=${PRETRAIN_MIN:-40}
echo "=== $(date) v2: Daten bis heute laden"
timeout 2h python s16_pretrain.py --arch v2 --download-only
while pgrep -f "run_pretrain.sh" > /dev/null; do echo "  warte auf v1-Lauf ... $(date +%H:%M)"; sleep 60; done
echo "=== $(date) v2: Vortraining ($MIN Min.) + Validierung 2025"
timeout 3h python s16_pretrain.py --arch v2 --mode valid --no-download --pretrain-min $MIN
echo "=== $(date) v2: Test 2026"
timeout 2h python s16_pretrain.py --arch v2 --mode test --no-download
echo "=== $(date) v2: Kombination mit Ensemble"
python s14_combo.py --a vfp2
tar czf keep6.tar.gz results/*vfp2* results/RESULTS_combo_vfp2_*.md
echo "=== $(date) FERTIG. Sicherung: keep6.tar.gz"
