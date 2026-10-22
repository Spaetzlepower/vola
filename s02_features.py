"""Schritt 2: 5m-Bars, Features und Zielgrößen bauen.

  python s02_features.py

Prognosezeitpunkte: jede volle Stunde t. Features nutzen nur Bars, die VOR t beginnen.
Zielgrößen für Horizont h (Stunden), Startkurs S0 = letzter Close vor t:
  y_rv_{h}  Summe quadrierter 5m-Log-Returns in [t, t+h)
  y_up_{h}  log(max High in [t, t+h) / S0)    -> "erreicht Preis darüber"
  y_dn_{h}  log(min Low  in [t, t+h) / S0)    -> "erreicht Preis darunter" (negativ)

Ergebnis: data/bars_5m.parquet, data/dataset.parquet, data/seasonality.parquet
Optional: data/events.csv (Spalten: time [UTC], name) mit Makro-Terminen (CPI, FOMC, NFP ...).
"""
import numpy as np
import pandas as pd

import config as C

H = C.BARS_PER_HOUR


def make_bars():
    k = pd.read_parquet(C.DATA / "klines_1m.parquet")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
           "quote_volume": "sum", "trades": "sum", "taker_buy_base": "sum"}
    b = k.resample(C.BAR, label="left", closed="left").agg(agg)
    b["n1m"] = k["close"].resample(C.BAR, label="left", closed="left").count()
    b["missing"] = b["n1m"] == 0
    b["close"] = b["close"].ffill()
    for c in ("open", "high", "low"):
        b[c] = b[c].fillna(b["close"])
    b[["volume", "quote_volume", "trades", "taker_buy_base"]] = b[["volume", "quote_volume", "trades", "taker_buy_base"]].fillna(0)
    b = b.dropna(subset=["close"])
    b["r"] = np.log(b["close"]).diff().fillna(0.0)
    print(f"5m-Bars: {len(b):,}, fehlend: {b['missing'].mean():.3%}")
    return b


def seasonality(b):
    """Intraday-/Wochen-Profil der Varianz (5m-of-week), nur aus Daten VOR Validierung/Test."""
    cutoff = min(pd.Timestamp(C.VALID_START, tz="UTC"), pd.Timestamp(C.TEST_START, tz="UTC"))
    pre = b[(b.index < cutoff) & ~b["missing"]]
    r2 = pre["r"] ** 2
    r2 = r2.clip(upper=r2.quantile(0.999))           # Ausreißer dämpfen
    bucket = pre.index.day_of_week * 288 + pre.index.hour * 12 + pre.index.minute // 5
    prof = r2.groupby(bucket).mean().reindex(range(2016)).interpolate().bfill().ffill()
    prof = pd.Series(np.convolve(np.r_[prof.values[-3:], prof.values, prof.values[:3]], np.ones(7) / 7, "valid"))
    prof = prof / prof.mean()
    prof.to_frame("seas").to_parquet(C.DATA / "seasonality.parquet")
    print(f"Saisonprofil aus {len(pre):,} Bars vor {cutoff.date()} (max/min = {prof.max() / prof.min():.1f}x)")
    return prof.values


def fwd_window(x, n, fn):
    """Wert von fn über x[i : i+n] für jedes i (NaN, wenn nicht genug Zukunft)."""
    rev = pd.Series(x[::-1])
    return getattr(rev.rolling(n, min_periods=n), fn)().values[::-1]


def build(b, prof):
    idx = b.index
    bucket = (idx.day_of_week * 288 + idx.hour * 12 + idx.minute // 5).values
    seas = prof[bucket]
    r2 = b["r"].values ** 2
    f = pd.DataFrame(index=idx)

    # --- rückwärtsgerichtete Features (bis einschließlich Bar i-1 -> später shift(1)) ---
    for name, w in {"1h": H, "4h": 4 * H, "24h": 24 * H, "72h": 72 * H, "7d": 168 * H, "30d": 720 * H}.items():
        f[f"rv_{name}"] = np.log(pd.Series(r2, idx).rolling(w, min_periods=w // 2).mean() + 1e-12)
        # saisonbereinigt: beobachtete Varianz relativ zum "normalen" Niveau dieser Tageszeiten
        f[f"rvd_{name}"] = np.log(pd.Series(r2 / seas, idx).rolling(w, min_periods=w // 2).mean() + 1e-12)
    for hl in (H, 6 * H, 24 * H, 168 * H):
        f[f"ewm_{hl // H}h"] = np.log(pd.Series(r2 / seas, idx).ewm(halflife=hl, min_periods=hl).mean() + 1e-12)
    park = np.log(b["high"] / b["low"]) ** 2 / (4 * np.log(2))
    f["park_24h"] = np.log(park.rolling(24 * H, min_periods=12 * H).mean() + 1e-12)
    f["park_1h"] = np.log(park.rolling(H, min_periods=6).mean() + 1e-12)
    lc = np.log(b["close"])
    for name, w in {"1h": H, "4h": 4 * H, "24h": 24 * H, "7d": 168 * H}.items():
        f[f"ret_{name}"] = lc - lc.shift(w)
    f["absret_1h"] = f["ret_1h"].abs()
    f["absret_24h"] = f["ret_24h"].abs()
    for name, w in {"1h": H, "24h": 24 * H, "7d": 168 * H}.items():
        f[f"range_{name}"] = np.log(b["high"].rolling(w).max() / b["low"].rolling(w).min())
    f["dist_hi_7d"] = lc - np.log(b["high"].rolling(168 * H).max())
    f["dist_lo_7d"] = lc - np.log(b["low"].rolling(168 * H).min())
    v30 = b["volume"].rolling(720 * H, min_periods=24 * H).mean()
    t30 = b["trades"].rolling(720 * H, min_periods=24 * H).mean()
    for name, w in {"1h": H, "24h": 24 * H}.items():
        f[f"volratio_{name}"] = np.log(b["volume"].rolling(w).mean() / v30 + 1e-9)
        f[f"trratio_{name}"] = np.log(b["trades"].rolling(w).mean() / t30 + 1e-9)
        tb = b["taker_buy_base"].rolling(w).sum()
        vv = b["volume"].rolling(w).sum()
        f[f"tbi_{name}"] = (2 * tb - vv) / (vv + 1e-9)          # Taker-Kauf/Verkauf-Ungleichgewicht
    f["avg_trade_1h"] = np.log((b["volume"].rolling(H).sum() + 1e-9) / (b["trades"].rolling(H).sum() + 1))
    f = f.shift(1)   # Bar i-1 -> Prognosezeitpunkt = Beginn von Bar i (keine Leckage)

    # --- zur Prognosezeit bekannte Größen ---
    f["S0"] = b["close"].shift(1)
    f["hour"] = idx.hour
    f["weekday"] = idx.day_of_week
    f["seas_now"] = np.log(seas)
    for h in C.HORIZONS_H:
        n = h * H
        f[f"seas_fwd_{h}"] = np.log(fwd_window(seas, n, "mean"))   # erwartetes Saisonniveau im Horizont

    # --- Zielgrößen ---
    lo, hi = b["low"].values, b["high"].values
    s0 = f["S0"].values
    for h in C.HORIZONS_H:
        n = h * H
        f[f"y_rv_{h}"] = fwd_window(r2, n, "sum")
        f[f"y_up_{h}"] = np.log(fwd_window(hi, n, "max") / s0)
        f[f"y_dn_{h}"] = np.log(fwd_window(lo, n, "min") / s0)
        f[f"y_miss_{h}"] = fwd_window(b["missing"].values.astype(float), n, "mean")

    ds = f[idx.minute == 0].copy()
    ds = ds[ds.index >= ds.index[0] + pd.Timedelta("30D")]   # Einschwingphase der Rolling-Features
    return ds


def add_futures(ds):
    p = C.DATA / "fut_metrics.parquet"
    if p.exists():
        m = pd.read_parquet(p)
        oi = [c for c in m.columns if c == "sum_open_interest"]
        if oi:
            s = np.log(m[oi[0]].where(m[oi[0]] > 0)).sort_index()
            s = s[~s.index.duplicated()]
            g = pd.DataFrame({"oi": s})
            g["oi_chg_1h"] = g["oi"] - g["oi"].shift(freq="1h").reindex(g.index, method="nearest", tolerance=pd.Timedelta("10min"))
            g["oi_chg_24h"] = g["oi"] - g["oi"].shift(freq="24h").reindex(g.index, method="nearest", tolerance=pd.Timedelta("10min"))
            for c in ("sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio"):
                if c in m.columns:
                    g[c] = np.log(m[c].where(m[c] > 0))
            g = g.drop(columns="oi")
            ds = pd.merge_asof(ds.sort_index(), g.sort_index(), left_index=True, right_index=True,
                               direction="backward", allow_exact_matches=False, tolerance=pd.Timedelta("2h"))
            print(f"Futures-Metrics hinzugefügt: {list(g.columns)}")
    p = C.DATA / "fut_funding.parquet"
    if p.exists():
        fr = pd.read_parquet(p).sort_index()
        ds = pd.merge_asof(ds.sort_index(), fr, left_index=True, right_index=True,
                           direction="backward", allow_exact_matches=False)
        print("Funding Rate hinzugefügt")
    return ds


def add_events(ds):
    p = C.DATA / "events.csv"                     # eigene Datei hat Vorrang
    if not p.exists():
        p = C.ROOT / "events_macro.csv"           # mitgelieferter Kalender (CPI, FOMC)
    if not p.exists():
        print("Kein Makro-Kalender gefunden -> ohne Makro-Termine")
        return ds
    ev = pd.to_datetime(pd.read_csv(p, comment="#")["time"], utc=True).sort_values().values
    t = ds.index.values
    j = np.searchsorted(ev, t, side="left")             # nächster Termin >= t
    nxt = np.where(j < len(ev), ev[np.minimum(j, len(ev) - 1)], np.datetime64("NaT"))
    prv = np.where(j > 0, ev[np.maximum(j - 1, 0)], np.datetime64("NaT"))
    to_next = (nxt - t) / np.timedelta64(1, "h")
    since = (t - prv) / np.timedelta64(1, "h")
    ds["h_to_event"] = np.clip(np.nan_to_num(to_next, nan=999), 0, 336)
    ds["h_since_event"] = np.clip(np.nan_to_num(since, nan=999), 0, 336)
    for h in C.HORIZONS_H:
        ds[f"event_in_{h}"] = (ds["h_to_event"] < h).astype(float)
    print(f"{len(ev)} Makro-Termine hinzugefügt")
    return ds


if __name__ == "__main__":
    b = make_bars()
    b.to_parquet(C.DATA / "bars_5m.parquet")
    prof = seasonality(b)
    ds = build(b, prof)
    ds = add_futures(ds)
    ds = add_events(ds)
    ds.to_parquet(C.DATA / "dataset.parquet")
    ycols = [c for c in ds.columns if c.startswith("y_")]
    print(f"OK: data/dataset.parquet  {len(ds):,} Prognosezeitpunkte, {ds.shape[1] - len(ycols)} Features, "
          f"{ds.index[0]} bis {ds.index[-1]}")
