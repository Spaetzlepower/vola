# BTC-Vola-Prognose für Touch-Märkte

Ziel: für Restlaufzeiten von 1 h bis 7 Tagen vorhersagen, wie stark BTC schwankt und mit welcher
Wahrscheinlichkeit Kursschwellen (±0,5 % … ±10 %) **berührt** werden. Diese Größe vergleichst du
später mit den Preisen bei Polymarket.

## Dateien

| Datei | Was sie macht | Laufzeit (ca.) |
|---|---|---|
| `s01_download.py` | Binance-Spot-1m-Klines (+ optional Futures: Open Interest, Funding) laden | 5–15 Min. |
| `s02_features.py` | 5m-Bars, Features, Zielgrößen, Saisonprofil | 1–2 Min. |
| `s03_models.py` | EWMA, HAR, GARCH, LightGBM, Klimatologie, LightGBM-Touch – Walk-forward | 10–40 Min. (CPU) |
| `s04_foundation.py` | Chronos-2, Timestral, Kronos zero-shot auf denselben Zeitpunkten | 1–3 Std. (GPU) |
| `s06_polymarket.py` | Polymarket-Wochenmärkte vs. Deribit-DVOL vs. eigenes Modell: Kalibrierung + Kurzfrist-Backtest | 10–30 Min. |
| `evaluate.py` | Metriken (QLIKE, Brier, LogLoss, Diebold-Mariano), Report | – |
| `analyse.ipynb` | Ablauf + Grafiken (Reliability-Diagramme, Vola-Verlauf) | – |
| `run_all.sh` | alles nacheinander im Hintergrund | – |

Ergebnisse landen in `results/` (`RESULTS_*.md`, CSVs, Prognosen als Parquet).

## Ausführen auf vast.ai (nur Browser)

1. Instanz: **RTX 3060**, Template **PyTorch (Jupyter)**, Standort **Europa**, ≥ 40 GB Disk, Verified.
2. In Jupyter: **Upload** → `btc-vola.zip` hochladen. Dann *New → Terminal*:
   ```bash
   cd ~ && unzip -o btc-vola.zip && cd btc-vola
   pip install -r requirements.txt
   ```
3. Kurzer Funktionstest (1–2 Min.):
   ```bash
   python -c "import torch; print(torch.cuda.get_device_name(0))"
   ```
4. Kompletten Validierungslauf im Hintergrund starten – läuft weiter, auch wenn du den Browser schließt:
   ```bash
   mkdir -p logs && nohup bash run_all.sh valid > logs/run_valid.log 2>&1 &
   tail -f logs/run_valid.log        # Strg+C beendet nur das Mitlesen, nicht den Lauf
   ```
   Oder Schritt für Schritt im Notebook `analyse.ipynb`.
5. Ergebnisse ansehen: `results/RESULTS_valid.md`, `results/RESULTS_foundation_valid.md` öffnen
   oder die Auswertungszellen im Notebook ausführen.

## Saubere Methodik (wichtig!)

- **valid** (2025) ist dein Spielplatz: Features, Parameter, Modellgrößen hier anpassen.
- **test** (ab 2026) erst ganz am Ende **einmal** laufen lassen: `bash run_all.sh test`.
  Wer am Testzeitraum herumschraubt, bis es gut aussieht, misst nur noch Glück.
- Foundation Models könnten BTC-Daten vor 2026 im Pretraining gesehen haben – deshalb zählt für sie
  vor allem der Test ab 2026.
- Alle Features nutzen nur Daten **vor** dem Prognosezeitpunkt; Modelle werden monatlich neu gefittet,
  mit Embargo in Länge des Horizonts.

## Nützliche Optionen

```bash
python s03_models.py --mode valid --refit-months 3      # schneller: nur quartalsweise neu fitten
python s03_models.py --mode valid --skip-garch --skip-touch
python s04_foundation.py --mode valid --models timestral --timestral-size 28M
python s04_foundation.py --mode valid --models kronos --paths 64 --kronos-size base
python s04_foundation.py --mode valid --limit 20 --paths 8                        # Schnelltest
```

## Makro-Termine (empfohlen)

`data/events.csv` mit US-Terminen (CPI, FOMC, NFP, PCE) anlegen, Zeit in **UTC**:
```
time,name
2026-01-13 13:30,CPI
2026-01-28 19:00,FOMC
```
Danach `python s02_features.py` und `s03_models.py` erneut ausführen. Zeitpunkte, an denen ein
Termin in den Horizont fällt, werden dann als Feature erkannt.

## Hinweise

- Polymarket nutzt **Binance Spot BTC/USDT** als Quelle → genau diese Daten werden geladen.
- Binance-Spot-Archive haben ab 2025 Zeitstempel in Mikrosekunden – wird automatisch erkannt.
- Kronos liefert Wahrscheinlichkeiten aus gesampelten Pfaden (Standard: 32) → grobe Auflösung;
  für feinere Werte `--paths 64` oder mehr (kostet linear mehr Zeit).
- Danach Instanz **zerstören**, damit keine Kosten weiterlaufen.

## Polymarket vs. Optionen (s06)

```bash
python s06_polymarket.py                 # nach s03 ausführen (nutzt dessen Vola-Prognosen)
python s06_polymarket.py --cost 0.03     # andere Kostenannahme (Round-trip je Anteil)
```
Lädt vergangene Wochenmärkte „What price will Bitcoin hit …“ (Gamma-/CLOB-API) und die Deribit-DVOL,
berechnet stündlich faire Touch-Wahrscheinlichkeiten und prüft (a) wer den Ausgang besser vorhersagt und
(b) ob sich der Polymarket-Preis innerhalb von 1/4/24 h Richtung fairer Wert bewegt (Backtest nach Kosten).
Rohdaten werden in `data/pm/` gecacht. Report: `results/RESULTS_polymarket.md`.
