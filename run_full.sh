#!/usr/bin/env bash
# Kompletter Lauf auf einer frischen Instanz (GPU optional). Start:
#   nohup bash run_full.sh > logs/run.log 2>&1 &
set -e
cd "$(dirname "$0")"
mkdir -p logs
echo "=== $(date) Start"
[ -f data/fm_features.parquet ] || tar xzf inputs/keep.tar.gz          # Sicherung (FM-Features, DVOL)
[ -f data/klines_1m.parquet ] || python s01_download.py
python s02_features.py

# Tempo: REFIT=1 = monatlich neu trainieren (genauer, langsam), Standard 3 = quartalsweise (schnell)
REFIT="${REFIT:-3}"
# Wirklich erlaubte Kerne (Container-Limit), nicht die Kerne des ganzen Servers -> verhindert Thread-Stau
CPUS=$(python - <<'PY'
import os
n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
try:
    q, p = open("/sys/fs/cgroup/cpu.max").read().split()
    if q != "max":
        n = min(n, max(1, int(int(q) / int(p))))
except Exception:
    try:
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read()); p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if q > 0:
            n = min(n, max(1, q // p))
    except Exception:
        pass
print(n)
PY
)
export OMP_NUM_THREADS=$CPUS MKL_NUM_THREADS=$CPUS OPENBLAS_NUM_THREADS=$CPUS PYTHONUNBUFFERED=1
echo "Kerne laut nproc: $(nproc), erlaubt: $CPUS -> genutzt: $CPUS Threads; Neu-Training alle $REFIT Monate"

# GPU nebenher: Kronos auf dem Testzeitraum (eigenes Log; nutzt gespeicherte Ergebnisse, falls vorhanden)
nohup python s04_foundation.py --mode test --models kronos > logs/kronos.log 2>&1 &
KPID=$!

# CPU: Modelle, Ensemble, Polymarket-Auswertungen
python s03_models.py --mode valid --refit-months "$REFIT" --skip-touch
python s03_models.py --mode test --refit-months "$REFIT" --skip-touch
python s05_ensemble.py --mode valid --refit-months "$REFIT"
python s05_ensemble.py --mode test --refit-months "$REFIT"
python s06_polymarket.py --import inputs/pm_dump.json
python s08_pm_check.py
python s10_pm_vol.py
[ -f inputs/pm_dump_1m.json ] && python s09_gap_decay.py || echo "(keine Minutendaten -> s09 übersprungen)"
[ -f inputs/pm_hourly_dump.json ] && python s11_pm_hourly.py || echo "(keine Stundenmärkte -> s11 übersprungen)"

echo "=== $(date) Warte auf Kronos ..."
wait $KPID || echo "Kronos mit Fehler beendet, siehe logs/kronos.log"
python s04_foundation.py --mode test --models kronos || true            # Bericht inkl. Vergleich (aus Cache)

tar czf keep2.tar.gz results/*.md results/*.csv data/fm_features.parquet data/dvol_1h.parquet
echo "=== $(date) FERTIG. Sicherung: keep2.tar.gz"
