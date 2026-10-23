"""Schritt 9: Wie schnell schließt sich eine Lücke zwischen Polymarket und fairem Wert? (Minutendaten)

  python s09_gap_decay.py --dump inputs/pm_dump_1m.json
  python s09_gap_decay.py --dump inputs/pm_dump_1m.json --th 0.05 0.10 --cost 0.02

Fairer Wert je Minute: Touch-Wahrscheinlichkeit aus aktuellem Binance-Kurs, Restlaufzeit und DVOL (ohne Risikoprämie).
Ereignis = Lücke |fair - Polymarket| überschreitet die Schwelle (neu; danach 60 Min. Pause je Stufe).
Für jede Reaktionszeit (0, 1, 2, 5, 10, 15, 30, 60 Min.) wird gemessen:
  - wie viel der Lücke dann noch übrig ist
  - Gewinn, wenn man erst dann einsteigt und 60 Min. später bzw. bei Auflösung aussteigt (nach Kosten)
Das beantwortet: Wie viel Zeit hat man? Sekunden (nur Bots) oder Minuten (machbar)?
"""
import argparse
import json

import numpy as np
import pandas as pd

import config as C
from s06_polymarket import parse_markets, touch_prob, vrp_factor, week_tstat

LAGS = [0, 1, 2, 5, 10, 15, 30, 60]


def market_minutes(r, hist, k1, dv, k):
    h = pd.DataFrame(hist.get("history", []))
    if h.empty:
        return None
    h["t"] = pd.to_datetime(h["t"], unit="s", utc=True)
    ps = h.drop_duplicates("t").set_index("t")["p"].astype(float).sort_index()
    # Vorsichtig: Ein Kurspunkt mit Zeitstempel t gilt erst ab t + 1 Min. als bekannt
    # (falls Polymarket den Minutenwert am Anfang des Intervalls stempelt).
    ps.index = ps.index + pd.Timedelta("1min")
    wstart = min(r.start, r.end - pd.Timedelta(days=7))
    lo_t, hi_t = max(ps.index[0], wstart).ceil("1min"), min(ps.index[-1], r.end)
    if hi_t <= lo_t:
        return None
    grid = pd.date_range(lo_t, hi_t, freq="1min", inclusive="left")
    pm = ps.reindex(grid, method="ffill", tolerance=pd.Timedelta("30min"))
    last_change = pd.DatetimeIndex(ps.index.to_series().reindex(grid, method="ffill"))   # letzte Kursmeldung
    w = k1[(k1.index >= wstart) & (k1.index < r.end)]
    hit_path = (w["high"] >= r.strike) if r.dir == "up" else (w["low"] <= r.strike)
    outcome = float(hit_path.any())
    first_hit = hit_path.idxmax() if hit_path.any() else r.end + pd.Timedelta(days=1)
    s = k1["close"].reindex(grid - pd.Timedelta("1min"), method="ffill").values
    T = ((r.end - grid) / pd.Timedelta("1h")).values
    b = np.log(r.strike / s) if r.dir == "up" else np.log(s / r.strike)
    dvt = dv.reindex(grid, method="ffill", tolerance=pd.Timedelta("3h")).values / 100
    fv = touch_prob(dvt * k * np.sqrt(np.maximum(T, 1e-9) / 8760), np.maximum(b, 0))
    fv[b <= 0] = 1.0
    df = pd.DataFrame({"t": grid, "pm": pm.values, "fv": fv, "S": s, "T_h": T,
                       "since_print_min": np.asarray((grid - last_change) / pd.Timedelta("1min"))})
    df["market_id"], df["event"], df["outcome"], df["first_hit"], df["end"] = r.market_id, r.event, outcome, first_hit, r.end
    return df[(df["t"] < first_hit) & (df["T_h"] > 0)].reset_index(drop=True)


def price_at(df, idx, lag):
    """Polymarket-Preis lag Minuten nach Zeile idx (1 bei zwischenzeitlicher Berührung, Ausgang nach Ablauf)."""
    t = df["t"].values[idx] + np.timedelta64(lag, "m")
    fh, end = df["first_hit"].iloc[0], df["end"].iloc[0]
    out = np.full(len(idx), np.nan)
    touched = t >= np.datetime64(fh.tz_convert(None))
    expired = t >= np.datetime64(end.tz_convert(None))
    pos = np.searchsorted(df["t"].values, t)
    ok = (pos < len(df)) & ~touched & ~expired
    ok[ok] &= df["t"].values[pos[ok]] == t[ok]
    out[ok] = df["pm"].values[pos[ok]]
    out[touched] = 1.0
    out[~touched & expired] = df["outcome"].iloc[0]
    return out, ok | touched | expired


def main(a):
    d = json.loads(open(a.dump).read())
    mk = parse_markets({s: (v[0] if isinstance(v, list) else v) for s, v in d["events"].items() if v})
    k1 = pd.read_parquet(C.DATA / "klines_1m.parquet", columns=["high", "low", "close"])
    mk = mk[mk["end"] <= k1.index.max()]
    dv = pd.read_parquet(C.DATA / "dvol_1h.parquet")["dvol"]
    dv.index = dv.index + pd.Timedelta(hours=1)
    bars = pd.read_parquet(C.DATA / "bars_5m.parquet", columns=["r"])
    k = vrp_factor(pd.read_parquet(C.DATA / "dvol_1h.parquet")["dvol"], bars, mk["start"].min())
    rows = []
    n_ok = 0
    for r in mk.itertuples():
        df = market_minutes(r, d["hist"].get(r.yes_token, {}), k1, dv, k)
        if df is None or len(df) < 120:
            continue
        n_ok += 1
        gap = (df["fv"] - df["pm"]).values
        for th in a.th:
            on = np.abs(gap) > th
            onset = on & ~np.r_[False, on[:-1]]
            idx, last = [], -10**9
            for i in np.flatnonzero(onset):                                   # max. 1 Ereignis je 60 Min. und Stufe
                if i - last >= 60:
                    idx.append(i); last = i
            if not idx:
                continue
            idx = np.array(idx)
            side = np.sign(gap[idx])
            for lag in LAGS:
                e, ok_e = price_at(df, idx, lag)
                x60, ok_x = price_at(df, idx, lag + 60)
                pos = np.minimum(idx + lag, len(df) - 1)
                rem = side * (df["fv"].values[pos] - e) / np.abs(gap[idx])          # übrig gebliebene Lücke (Anteil)
                valid = ok_e & (e > 0.001) & (e < 0.999)
                rows.append(pd.DataFrame({
                    "th": th, "lag": lag, "event": r.event, "market_id": r.market_id, "valid": valid,
                    "dir": r.dir, "side": np.where(side > 0, "Yes kaufen", "No kaufen"),
                    "pnl_60": np.where(valid & ok_x, side * (x60 - e) - a.cost, np.nan),
                    "pnl_end": np.where(valid, side * (df["outcome"].iloc[0] - e) - a.cost / 2, np.nan),
                    "rest_gap": np.where(valid, rem, np.nan), "gap0": np.abs(gap[idx]),
                    "since_print": df["since_print_min"].values[idx]}))
    R = pd.concat(rows, ignore_index=True)
    R.to_parquet(C.RESULTS / "gap_decay.parquet")
    L = ["# Wie schnell schließt sich die Lücke? (Minutendaten)", "",
         f"{n_ok} Stufen mit Minutendaten, {R['event'].nunique()} Wochen. Kosten {a.cost} je Anteil "
         f"(bei Auflösung nur halbe Kosten). Fairer Wert: DVOL x {k:.3f}.", ""]
    for th, g in R.groupby("th"):
        tab = []
        for lag, gg in g.groupby("lag"):
            t60, _ = week_tstat(gg["pnl_60"].dropna().values, gg.loc[gg["pnl_60"].notna(), "event"].values)
            te, _ = week_tstat(gg["pnl_end"].dropna().values, gg.loc[gg["pnl_end"].notna(), "event"].values)
            tab.append({"Reaktion (Min.)": lag, "Ereignisse": int(gg["valid"].sum()),
                        "übrige Lücke (Median)": gg["rest_gap"].median(),
                        "PnL Ausstieg +60 Min.": gg["pnl_60"].mean(), "t": t60,
                        "PnL bis Auflösung": gg["pnl_end"].mean(), "t ": te})
        L += [f"## Schwelle {th:.2f}", "", pd.DataFrame(tab).round(4).to_markdown(index=False), ""]
        g0 = g[g["lag"] == 0].copy()
        g0["Kursmeldung vor"] = pd.cut(g0["since_print"], [-1, 1, 5, 15, 60, 1e9], labels=["≤1 Min", "2-5", "6-15", "16-60", ">60"])
        L += ["**Lücken nach Alter der letzten Polymarket-Kursmeldung (sofortiger Einstieg)** — große Lücken bei alten "
              "Meldungen = vermutlich nur fehlende Daten, nicht handelbar", "",
              g0.groupby("Kursmeldung vor", observed=True).agg(n=("pnl_60", "size"), PnL60=("pnl_60", "mean"),
                                                               PnL_Ende=("pnl_end", "mean")).round(4).to_markdown(), ""]
    # Woher kommt der Gewinn bei Halten bis Auflösung? (Reaktion 60 Min. = vorsichtig)
    g60 = R[(R["lag"] == 60) & R["pnl_end"].notna()]
    if len(g60):
        L += ["## Halten bis Auflösung, Reaktion 60 Min.: nach Richtung und Seite", "",
              "Kommt der Gewinn fast nur aus einer Kombination (z. B. 'No kaufen' auf Stufen darüber in einer fallenden Phase), "
              "ist es eher der Kurstrend der wenigen Wochen als ein dauerhafter Effekt.", "",
              g60.groupby(["th", "dir", "side"]).agg(n=("pnl_end", "size"), PnL=("pnl_end", "mean"),
                                                    Treffer=("pnl_end", lambda x: (x > 0).mean())).round(4).to_markdown(), "",
              "**Je Woche (Schwelle 0.10)**", "",
              g60[g60["th"] == 0.10].groupby("event").agg(n=("pnl_end", "size"), PnL=("pnl_end", "mean")).round(4).to_markdown(), ""]
    L += ["## Lesehilfe", "",
          "- Fällt der PnL schon bei 1–2 Min. Reaktion auf ~0 → nur für schnelle Bots nutzbar.",
          "- Bleibt er bei 5–15 Min. deutlich positiv (t > 2) → mit einem einfachen automatisierten Ansatz machbar.",
          "- Gewinne nur bei alten Kursmeldungen (>15 Min.) → Datenartefakt (kein aktueller Preis vorhanden).", ""]
    out = C.RESULTS / "RESULTS_gap_decay.md"
    out.write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="inputs/pm_dump_1m.json")
    ap.add_argument("--th", type=float, nargs="+", default=[0.05, 0.10])
    ap.add_argument("--cost", type=float, default=0.02)
    main(ap.parse_args())
