"""Schritt 17: Polymarket mit Ausstiegsstrategie statt "bis Auflösung halten" – auf ECHTEN Trades.

  python s17_pm_exit.py                      # Wochenmärkte, nutzt results/pm_trades_panel.parquet (aus s13)
  python s17_pm_exit.py --kind hour
  python s17_pm_exit.py --fee 0.01 --split 2026-04-01

Ablauf je Markt in Zeitreihenfolge (immer nur EINE offene Position je Markt):
  Signal  : ein Trade belegt Yes kaufbar zu p (ask) mit fair - p > Schwelle  -> Yes kaufen
            bzw. Yes verkaufbar zu p (bid) mit p - fair > Schwelle             -> Yes verkaufen (= No kaufen)
  Einstieg: nächster Trade derselben Art mindestens --delay Sekunden später, zu DESSEN Preis
  Ausstieg: an Trades der Gegenseite (bei Long: Yes verkaufbar), frühestens --delay s nach dem Einstieg:
            (a) Ziel   : Lücke zum fairen Wert dieses Zeitpunkts <= g (Konvergenz), Ausstieg zum nächsten
                         Gegenseiten-Trade mind. --delay s später
            (b) Zeit   : nach H ohne Ziel -> nächster Gegenseiten-Trade
            (c) sonst  : bis Auflösung halten (Wochenmarkt nach Berührung = 1)
  Fairer Wert an jedem Trade = zu DIESEM Zeitpunkt bekannt (s13: Binance-Minute davor, DVOL nach Kerzenschluss).
Parameter (Schwelle, g, H, Quelle) werden NUR auf dem Zeitraum vor --split gewählt, danach einmal getestet.
"""
import argparse
import math

import numpy as np
import pandas as pd

import config as C
from s06_polymarket import week_tstat

HS = {"5 Min": 300, "30 Min": 1800, "2 h": 7200, "12 h": 43200, "halten": np.inf}
GS = [0.0, 0.01, 0.02, -1.0]          # -1 = kein Ziel (nur Zeit)


def simulate(g, fv, th, gap_exit, H, delay, fee, cap, window=900):
    """g: Trades EINES Marktes (zeitlich sortiert). Gibt Liste abgeschlossener Positionen zurück."""
    t = g["_s"].values; p = g["yes_price"].values; ask = (g["avail"].values == "ask"); f = g[fv].values
    q = np.minimum(g["size"].values, cap); out = g["outcome"].values[0]
    n, res = len(g), []
    okp = np.isfinite(f) & (p > 0.01) & (p < 0.99)
    L_ = okp & ask & (f - p > th + fee)
    S_ = okp & ~ask & (p - f > th + fee)
    cand = np.flatnonzero(L_ | S_)
    if len(cand) == 0:
        return res
    t, p, f, q = t.tolist(), p.tolist(), f.tolist(), q.tolist()          # Python-Listen: schneller in den Schleifen
    ask_l = ask.tolist(); Ll = L_.tolist()
    free_from = 0
    for i in cand:
        if i < free_from:
            continue
        long_ = Ll[i]
        j = i + 1
        while j < n and (t[j] < t[i] + delay or ask_l[j] != long_):
            j += 1
        if j >= n:
            break
        if t[j] - t[i] > window:                                 # Signal veraltet, keine Liquidität mehr
            continue
        e, te, qe = p[j], t[j], q[j]
        k, exit_p, how = j + 1, None, "Auflösung"
        trig = None
        while k < n:
            if t[k] >= te + delay and ask_l[k] != long_:          # Gegenseite: dort kann man schließen
                if trig is not None and t[k] >= trig + delay:
                    exit_p, how = p[k], how_t; break
                if trig is None:
                    conv = gap_exit >= 0 and math.isfinite(f[k]) and ((f[k] - p[k] <= gap_exit) if long_ else (p[k] - f[k] <= gap_exit))
                    if conv:
                        trig, how_t = t[k], "Ziel"
                    elif t[k] - te >= H:
                        exit_p, how = p[k], "Zeit"; break
            k += 1
        if exit_p is None:
            exit_p = out
            k = n
        pnl = (exit_p - e) if long_ else (e - exit_p)
        pnl -= fee * (1 if how == "Auflösung" else 2)             # Gebühr je Seite (Auflösung kostet nichts)
        res.append((te, pnl, qe, e if long_ else 1 - e, how, (t[min(k, n - 1)] - te) / 60, long_))
        free_from = k + 1                                          # erst nach dem Ausstieg neu einsteigen
    return res


def run_grid(tr, fvs, a):
    rows, detail = [], {}
    groups = [g for _, g in tr.groupby("market", sort=False)]
    for fv in fvs:
        for th in a.th:
            for gx in GS:
                for hn, H in HS.items():
                    R = []
                    for g in groups:
                        R += simulate(g, fv, th, gx, H, a.delay, a.fee, a.cap, a.window)
                    if len(R) < 20:
                        continue
                    D = pd.DataFrame(R, columns=["te", "pnl", "q", "cost", "how", "dauer_min", "long"])
                    D["day"] = pd.to_datetime(D["te"], unit="s", utc=True).dt.floor("1D")
                    D["part"] = np.where(D["day"] < pd.Timestamp(a.split, tz="UTC"), "A", "B")
                    key = (fv, th, gx, hn)
                    detail[key] = D
                    for part, d in D.groupby("part"):
                        tt, nd = week_tstat(d["pnl"].values, d["day"].values)
                        dd = (d["pnl"] * d["q"]).groupby(d["day"]).sum()
                        rows.append({"Quelle": fv, "Schwelle": th, "Ziel-Lücke": "–" if gx < 0 else gx, "max. Dauer": hn,
                                     "Teil": part, "Trades": len(d), "Tage": nd, "PnL/Anteil": d["pnl"].mean(),
                                     "PnL $": (d["pnl"] * d["q"]).sum(), "Rendite/$": (d["pnl"] * d["q"]).sum() / (d["cost"] * d["q"]).sum(),
                                     "Treffer": (d["pnl"] > 0).mean(), "pos. Tage": (dd > 0).mean(), "t(Tage)": tt,
                                     "Dauer Median (Min)": d["dauer_min"].median(), "Anteil Ziel": (d["how"] == "Ziel").mean()})
    return pd.DataFrame(rows), detail


def main(a):
    tr = pd.read_parquet(C.RESULTS / "pm_trades_panel.parquet")
    tr = tr[tr["kind"] == a.kind].copy()
    tr["_s"] = ((tr["time"] - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta("1s")).astype("float64")
    tr = tr.sort_values(["market", "_s"]).reset_index(drop=True)
    fvs = [c for c in tr.columns if c.startswith("fv_") and tr[c].notna().mean() > 0.5]
    print(f"{len(tr):,} Trades, {tr['market'].nunique()} Märkte, {tr['time'].min():%Y-%m-%d} bis {tr['time'].max():%Y-%m-%d}, "
          f"Quellen {fvs}, Teil A < {a.split} (Auswahl), Teil B ab {a.split} (Test)")
    R, detail = run_grid(tr, fvs, a)
    if R.empty:
        print("Zu wenige Positionen."); return
    A = R[R["Teil"] == "A"].copy()
    A = A[A["Tage"] >= a.min_days]
    best = A.sort_values("t(Tage)", ascending=False).head(5)
    keys = [(r["Quelle"], r["Schwelle"], -1.0 if r["Ziel-Lücke"] == "–" else r["Ziel-Lücke"], r["max. Dauer"]) for _, r in best.iterrows()]
    B = pd.concat([pd.DataFrame()] + [R[(R["Teil"] == "B") & (R["Quelle"] == k[0]) & (R["Schwelle"] == k[1]) &
                     (R["Ziel-Lücke"].astype(str) == ("–" if k[2] < 0 else str(k[2]))) & (R["max. Dauer"] == k[3])] for k in keys])
    name = "Woche" if a.kind == "week" else "stündlich"
    L = [f"# Polymarket {name}: Einstieg bei Abweichung, Ausstieg nach Regel (echte Trades)", "",
         f"{len(tr):,} Taker-Trades, {tr['market'].nunique()} Märkte. Verzögerung {a.delay:.0f} s bei Ein- und Ausstieg, "
         f"Gebühr {a.fee} je Seite, max. {a.cap:.0f} Anteile je Position, eine Position je Markt.", "",
         f"Teil A (Auswahl der Regel): bis {a.split}. Teil B (Test, einmal): ab {a.split}.", "",
         "## Beste 5 Regeln auf Teil A (nach t-Wert über Tage)", "", best.round(4).to_markdown(index=False), "",
         "## Dieselben Regeln auf Teil B (Test)", "", B.round(4).to_markdown(index=False), "",
         "## Vergleich: bis Auflösung halten", "",
         R[(R["max. Dauer"] == "halten") & (R["Ziel-Lücke"].astype(str) == "–")].round(4).to_markdown(index=False), "",
         "## Alle Regeln", "", R.sort_values(["Teil", "t(Tage)"], ascending=[True, False]).round(4).to_markdown(index=False), "",
         "## Lesehilfe", "",
         "- Gilt nur, wenn eine Regel auf Teil B (nie zur Auswahl benutzt) positiv bleibt, mit t > 2 und vielen positiven Tagen.",
         "- 'Anteil Ziel' = wie oft die Lücke sich geschlossen hat; 'Dauer Median' = typische Haltedauer.",
         "- Ausstieg zum Preis eines echten Gegenseiten-Trades; Teilmengen/Orderbuchtiefe sind nicht modelliert (Menge max. --cap).", ""]
    out = C.RESULTS / f"RESULTS_pm_exit_{a.kind}.md"
    out.write_text("\n".join(L))
    print("\n".join(L[:14]))
    print(f"\nReport: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["week", "hour"], default="week")
    ap.add_argument("--th", type=float, nargs="+", default=[0.03, 0.05, 0.08, 0.12])
    ap.add_argument("--delay", type=float, default=30)
    ap.add_argument("--fee", type=float, default=0.0)
    ap.add_argument("--cap", type=float, default=100)
    ap.add_argument("--split", default="2026-04-01")
    ap.add_argument("--min-days", type=int, default=20)
    ap.add_argument("--window", type=float, default=900, help="max. Sekunden zwischen Signal und Einstiegs-Trade")
    main(ap.parse_args())
