"""Zentrale Einstellungen für das BTC-Vola-Projekt."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
RESULTS = ROOT / "results"
EXT = ROOT / "ext"          # geklonte Repos (Kronos)
for p in (DATA, RAW, RESULTS, EXT):
    p.mkdir(parents=True, exist_ok=True)

SYMBOL = "BTCUSDT"
DOWNLOAD_START = "2021-01"   # erster Monat der 1m-Klines

BAR = "5min"                 # Basisraster
BARS_PER_HOUR = 12

# Prognosehorizonte in Stunden (Restlaufzeiten der Polymarket-Märkte)
HORIZONS_H = [1, 4, 24, 72, 168]

# Schwellen für Touch-Wahrscheinlichkeiten: |Kursbewegung| relativ zum Startkurs
THRESHOLDS = [0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10]

# Zeitraum, auf dem final getestet wird. Foundation Models haben ältere BTC-Daten
# womöglich im Pretraining gesehen -> nur ab hier sauber vergleichbar.
TEST_START = "2026-01-01"
# Zum Tuning von Hyperparametern: Validierungszeitraum (danach NICHT mehr am Test tunen!)
VALID_START, VALID_END = "2025-01-01", "2025-12-31"

SEED = 42
