"""Schritt 8: Ist der Polymarket-Edge echt? Robustheitsprüfung des Backtests aus s06.

  python s08_pm_check.py                       # nutzt results/pm_panel.parquet aus s06
  python s08_pm_check.py --fv fv_dvol_adj fv_ens_har_all --th 0.05 0.10

Prüft die typischen Fehlerquellen:
  1. Zeitversatz/Latenz: Einstieg erst zum Preis eine Stunde NACH dem Signal (realistisch: man reagiert nicht sofort).
     Bricht der Gewinn dann zusammen, war er (teilweise) ein Timing-Artefakt.
  2. Veraltete Preise: Wie lange hat sich der Preis vor dem Signal nicht bewegt? Gewinn nur bei "eingefrorenen"
     Preisen = vermutlich nicht handelbar.
  3. Kosten: 2, 4, 6 Cent je Anteil (Spread bei wenig gehandelten Stufen kann groß sein).
  4. Wo entsteht der Gewinn? Nach Seite (Yes/No kaufen), Preisbereich, Restlaufzeit, Volumen, Zeitraum, Richtung.
  5. Konzentration: Anteil positiver Wochen, schlechteste Woche, nur 1 Trade je Stufe und Tag.
  6. Rendite je eingesetztem Dollar (No zu 0,92 kaufen bindet 0,92 $, bringt aber max. 0,08 $).
"""
import argparse

import numpy as np
import pandas as pd

import config as C
from s06_polymarket import future_price, week_tstat


def staleness(P):
    """Stunden seit der letzten Preisänderung (je Stufe)."""
    P = P.sort_values(["market_id", "t"])
    chg = P.groupby("market_id")["pm"].diff().fillna(1).ne(0)
    grp = chg.groupby(P["market_id"]).cumsum()
    return P.groupby([P["market_id"], grp]).cumcount().reindex(P.index)


def trades(P, fv, th, H, cost, delay=0):
    """Signal zum Zeitpunkt t; Einstieg zum Preis t+delay; Ausstieg t+delay+H (H=None: bis Ende)."""
    entry = P["pm"] if delay == 0 else future_price(P, delay)
    gap = P[fv] - P["pm"]                                     # Signal nur aus Info bis t
    sel = (gap.abs() > th) & entry.notna() & P[fv].notna()
    # Einstieg muss noch sinnvoll sein (Stufe nicht inzwischen berührt/ausgelaufen)
    sel &= (entry > 0.001) & (entry < 0.999)
    if H is None:
        ex = P["outcome"]
    else:
        ex = future_price(P.assign(t=P["t"] + pd.Timedelta(hours=delay)), H) if delay else future_price(P, H)
    sel &= ex.notna()
    side = np.sign(gap[sel])                                  # +1 = Yes kaufen, -1 = No kaufen
    e, x = entry[sel], ex[sel]
    pnl = side * (x - e) - (cost / 2 if H is None else cost)   # bis Ende: kein Ausstiegs-Spread
    capital = np.where(side > 0, e, 1 - e)
    T = P.loc[sel].assign(side=side, entry=e, exit=x, pnl=pnl, capital=capital)
    return T


def summary(T, name):
    if len(T) < 20:
        return None
    t, nw = week_tstat(T["pnl"].values, T["event"].values)
    wk = T.groupby("event")["pnl"].mean()
    return {"Variante": name, "Trades": len(T), "Wochen": nw, "PnL/Anteil": T["pnl"].mean(),
            "Rendite/$": T["pnl"].sum() / T["capital"].sum(), "Treffer": (T["pnl"] > 0).mean(),
            "pos. Wochen": (wk > 0).mean(), "schlechteste Woche": wk.min(), "t(Wochen)": t}


def main(a):
    P = pd.read_parquet(C.RESULTS / "pm_panel.parquet").sort_values(["market_id", "t"]).reset_index(drop=True)
    P["stale_h"] = staleness(P)
    L = ["# Robustheitsprüfung Polymarket-Backtest", "",
         f"{P['event'].nunique()} Wochen, {P['market_id'].nunique()} Stufen, {len(P):,} Stunden-Beobachtungen.", ""]
    for fv in a.fv:
        if fv not in P:
            print(f"{fv} fehlt im Panel, übersprungen")
            continue
        for th in a.th:
            L += [f"## {fv}, Schwelle {th:.2f}", ""]
            rows = []
            for H, hn in ((4, "4h"), (24, "24h"), (None, "bis Ende")):
                for cost in (0.02, 0.04, 0.06):
                    rows.append(summary(trades(P, fv, th, H, cost), f"{hn}, Kosten {cost:.2f}"))
                rows.append(summary(trades(P, fv, th, H, 0.02, delay=1), f"{hn}, Einstieg 1h später, Kosten 0.02"))
            L += ["### Kosten & Latenz", "", pd.DataFrame([r for r in rows if r]).round(4).to_markdown(index=False), ""]

            T = trades(P, fv, th, 24, 0.02)
            if len(T) < 20:
                continue
            T["Seite"] = np.where(T["side"] > 0, "Yes kaufen", "No kaufen")
            T["Preisbereich"] = pd.cut(T["entry"], [0, .05, .1, .2, .4, .6, .8, .9, .95, 1])
            T["Restlaufzeit"] = pd.cut(T["T_h"], [0, 6, 24, 72, 1000], labels=["<6h", "6-24h", "1-3d", ">3d"])
            T["Volumen"] = pd.qcut(T["volume"].rank(method="first"), 3, labels=["niedrig", "mittel", "hoch"])
            T["Zeitraum"] = np.where(T["t"] < pd.Timestamp("2026-01-01", tz="UTC"), "2025 (valid)", "2026 (test)")
            T["eingefroren"] = pd.cut(T["stale_h"], [-1, 0, 2, 6, 1000], labels=["bewegt", "1-2h", "3-6h", ">6h"])
            L += ["### Wo entsteht der Gewinn? (24h halten, Kosten 0.02)", ""]
            for col in ("Seite", "dir", "Preisbereich", "Restlaufzeit", "Volumen", "Zeitraum", "eingefroren"):
                g = T.groupby(col, observed=True).agg(Trades=("pnl", "size"), PnL=("pnl", "mean"),
                                                      Rendite=("pnl", lambda s: s.sum() / T.loc[s.index, "capital"].sum()),
                                                      Treffer=("pnl", lambda s: (s > 0).mean()))
                L += [f"**{col}**", "", g.round(4).to_markdown(), ""]
            one = T.assign(day=T["t"].dt.floor("1D")).sort_values("t").groupby(["market_id", "day"]).head(1)
            r = summary(one, "nur 1 Trade je Stufe und Tag")
            if r:
                L += ["### Ohne Überlappung", "", pd.DataFrame([r]).round(4).to_markdown(index=False), ""]
    L += ["## Lesehilfe", "",
          "- **Einstieg 1h später** bricht stark ein → der Effekt war ein Timing-Artefakt oder nur mit sofortiger Reaktion nutzbar.",
          "- Gewinn fast nur bei **eingefroren > 6h** oder **Volumen niedrig** → vermutlich keine echten Gegenparteien zu diesem Preis.",
          "- Gewinn fast nur bei **No kaufen** in Preisbereich 0–10 % → Longshot-Bias (bekanntes Phänomen, kleine Rendite je $, "
          "aber Crash-Risiko: seltene große Verluste).",
          "- **2026 (test)** deutlich schwächer als 2025 → Effekt verschwindet oder war Zufall.", ""]
    out = C.RESULTS / "RESULTS_pm_check.md"
    out.write_text("\n".join(L))
    print("\n".join(L[:60]))
    print(f"\nReport: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fv", nargs="+", default=["fv_dvol_adj", "fv_ens_har_all"])
    ap.add_argument("--th", type=float, nargs="+", default=[0.10])
    main(ap.parse_args())
