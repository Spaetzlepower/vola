# Papierhandels-Bot (keine echten Orders)

Läuft live mit, vergleicht Polymarket-Bid/Ask der BTC-Märkte (stündlich „above“ + Woche „hit“) mit dem fairen Wert
und protokolliert, was er gekauft *hätte* – plus Abrechnung bei Auflösung.

```bash
pip install websockets pandas pyarrow
python bot/paper_bot.py --selftest        # Rechenlogik prüfen
python bot/paper_bot.py                   # live, Strg+C beendet; besser: nohup python bot/paper_bot.py > bot/bot.log 2>&1 &
```

Optionen: `--edge 0.05` (Mindestabstand), `--max-size 50` (Anteile je Trade), `--fee-bps 0` (Gebühren!),
`--min-tau 2` (keine Trades in den letzten Minuten), `--hours-ahead 6`.

Ausgaben in `bot/logs/`: `signals.csv`, `settled.csv`, `snapshots.csv` (Bid/Ask/fair alle 10 s – echte Orderbuch-Daten).
Nach einigen Tagen die drei Dateien zurückschicken → Auswertung mit echten Ausführungspreisen.

**Nur dort laufen lassen, wo Polymarket erlaubt und erreichbar ist** (z. B. bei Freunden). Sperren nicht umgehen.
Rechner/Server sollte durchgehend laufen; ein günstiger Cloud-Server reicht (keine GPU nötig).
