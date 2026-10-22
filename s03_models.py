"""Schritt 3: Baselines + LightGBM, Walk-forward-Evaluation.

  python s03_models.py --mode valid     # Tuning-Lauf auf 2025 (hier darfst du Parameter ändern)
  python s03_models.py --mode test      # finaler Lauf ab 2026 (danach nichts mehr am Modell drehen)

Modelle (Varianz über den Horizont -> Touch-Wahrscheinlichkeit per Brownscher Bewegung):
  ewma          EWMA (Halbwertszeit 24h) der saisonbereinigten Varianz x Saisonprofil
  har           HAR-Regression (1h, 24h, 7d, 30d) in log-Varianz, saisonbereinigt
  garch         GJR-GARCH(1,1), Student-t, auf saisonbereinigten Stundenreturns
  lgbm          LightGBM-Regression der log-Varianz mit allen Features
Direkte Touch-Modelle:
  climatology   historische Trefferquote je Schwelle (Referenz ohne Können)
  lgbm_touch    LightGBM-Klassifikator: P(Schwelle wird erreicht), monoton in der Schwelle

Walk-forward: jeden Monat neu fitten, nur mit Zeitpunkten, deren Zielgröße VOR Monatsbeginn
vollständig bekannt war (Embargo = Horizont).
"""
import argparse
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

import config as C
from evaluate import barriers, reliability, touch_long, touch_table, vol_table, write_report

warnings.filterwarnings("ignore")
H = C.BARS_PER_HOUR
HAR_COLS = ["rvd_1h", "rvd_24h", "rvd_7d", "rvd_30d"]

LGB_REG = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=200,
               feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
               verbose=-1, seed=C.SEED, num_threads=0)
LGB_CLF = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500,
               feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
               verbose=-1, seed=C.SEED, num_threads=0)


def feature_cols(ds):
    return [c for c in ds.columns if not c.startswith(("y_", "seas_fwd_", "event_in_")) and c != "S0"]


def norm_target(ds, h):
    """log der Horizont-Varianz pro Bar, bereinigt um das erwartete Saisonniveau."""
    return np.log(np.maximum(ds[f"y_rv_{h}"], 1e-12) / (h * H)) - ds[f"seas_fwd_{h}"]


def denorm(pred_log, ds, h):
    return np.exp(pred_log + ds[f"seas_fwd_{h}"]) * h * H


def month_starts(start, end):
    return pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), freq="MS")


def train_mask(ds, cutoff, h):
    return (ds.index + pd.Timedelta(hours=h) <= cutoff) & ds[f"y_rv_{h}"].notna() & (ds[f"y_miss_{h}"] < 0.05)


# ---------------- Varianzmodelle ----------------

def fit_har(tr, h):
    X = np.c_[np.ones(len(tr)), tr[HAR_COLS].values]
    y = norm_target(tr, h).values
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    s2 = np.var(y[ok] - X[ok] @ beta)
    return beta, s2


def pred_har(model, te, h):
    beta, s2 = model
    X = np.c_[np.ones(len(te)), te[HAR_COLS].values]
    return denorm(X @ beta + 0.5 * s2, te, h)   # +0.5*s2: Lognormal-Korrektur E[exp]


def fit_lgbm_reg(tr, h, feats):
    y = norm_target(tr, h)
    ok = np.isfinite(y)
    tr, y = tr[ok], y[ok]
    k = int(len(tr) * 0.9)                       # letzte 10 % chronologisch als Early-Stopping-Set
    dtr = lgb.Dataset(tr[feats].iloc[:k], y.iloc[:k])
    dva = lgb.Dataset(tr[feats].iloc[k:], y.iloc[k:])
    m = lgb.train(LGB_REG, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    s2 = np.var(y.iloc[k:] - m.predict(tr[feats].iloc[k:]))
    return m, s2


def garch_forecasts(ds, start, end):
    """GJR-GARCH auf saisonbereinigten Stundenreturns; Parameter monatlich neu, Filterung laufend."""
    try:
        from arch import arch_model
    except ImportError:
        print("  'arch' nicht installiert -> GARCH übersprungen")
        return None
    b = pd.read_parquet(C.DATA / "bars_5m.parquet", columns=["r"])
    seas = pd.read_parquet(C.DATA / "seasonality.parquet")["seas"].values
    bucket = (b.index.day_of_week * 288 + b.index.hour * 12 + b.index.minute // 5).values
    b["seas"] = seas[bucket]
    hr = b.resample("1h").agg({"r": "sum", "seas": "mean"})
    z = (hr["r"] / np.sqrt(hr["seas"])) * 100        # Prozent, saisonbereinigt
    hmax = max(C.HORIZONS_H)
    out = {h: pd.Series(np.nan, index=ds.index) for h in C.HORIZONS_H}
    for ms in month_starts(start, end):
        me = ms + pd.offsets.MonthBegin(1)
        hist = z[z.index < ms]
        if len(hist) < 2000:
            continue
        am = arch_model(z[z.index < me], mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="t")
        res = am.fit(last_obs=ms, disp="off")
        # Stunde mit Beginn s ist bei Prognosezeitpunkt s+1h bekannt
        first = ms - pd.Timedelta(hours=1)
        fc = res.forecast(horizon=hmax, start=first, reindex=False).variance
        fc.index = fc.index + pd.Timedelta(hours=1)
        fc = fc[(fc.index >= ms) & (fc.index < me)]
        cum = fc.cumsum(axis=1)
        for h in C.HORIZONS_H:
            idx = fc.index.intersection(ds.index)
            # Summe der h Stundenvarianzen (in %^2, saisonbereinigt) -> mal Saisonniveau
            out[h].loc[idx] = cum.loc[idx].iloc[:, h - 1].values / 1e4 * np.exp(ds.loc[idx, f"seas_fwd_{h}"].values)
        print(f"  GARCH {ms:%Y-%m} gefittet")
    return out


# ---------------- Touch-Klassifikator ----------------

def stack_touch(df, h, d, var_ref):
    """Eine Zeile pro (Zeitpunkt, Schwelle). z = Schwelle / erwartete Vola (aus EWMA)."""
    bu, bd = barriers(C.THRESHOLDS)
    sig = np.sqrt(var_ref)
    parts = []
    for thr, b in zip(C.THRESHOLDS, bu if d == "up" else bd):
        x = df.copy()
        x["log_b"] = np.log(b)
        x["z"] = b / sig
        x["thr"] = thr
        if d == "up":
            x["hit"] = (df[f"y_up_{h}"] >= b).astype(np.int8)
        else:
            x["hit"] = (df[f"y_dn_{h}"] <= -b).astype(np.int8)
        parts.append(x)
    return pd.concat(parts)


def ewma_var(df, h):
    return np.exp(df["ewm_24h"]) * np.exp(df[f"seas_fwd_{h}"]) * h * H


def run(mode, refit_months, skip_garch, skip_touch):
    ds = pd.read_parquet(C.DATA / "dataset.parquet")
    if mode == "valid":
        start, end = C.VALID_START, C.VALID_END
    else:
        start, end = C.TEST_START, str(ds.index[-1].date())
    feats = feature_cols(ds)
    extra_h = {h: [c for c in ds.columns if c in (f"seas_fwd_{h}", f"event_in_{h}")] for h in C.HORIZONS_H}
    print(f"Modus {mode}: {start} bis {end}, {len(feats)} Features")

    garch = None if skip_garch else garch_forecasts(ds, start, end)
    vol_rows, touch_rows = [], []
    starts = month_starts(start, end)
    for h in C.HORIZONS_H:
        fh = feats + extra_h[h]
        for i in range(0, len(starts), refit_months):
            ms = starts[i]
            me = starts[i + refit_months] if i + refit_months < len(starts) else pd.Timestamp(end, tz="UTC") + pd.Timedelta("1D")
            te = ds[(ds.index >= ms) & (ds.index < me) & ds[f"y_rv_{h}"].notna() & (ds[f"y_miss_{h}"] < 0.05)]
            if te.empty:
                continue
            tr = ds[train_mask(ds, ms, h)]
            preds = {"ewma": ewma_var(te, h),
                     "har": pred_har(fit_har(tr, h), te, h)}
            m, s2 = fit_lgbm_reg(tr, h, fh)
            preds["lgbm"] = denorm(m.predict(te[fh]) + 0.5 * s2, te, h)
            if garch is not None:
                preds["garch"] = garch[h].reindex(te.index)
            for name, v in preds.items():
                vol_rows.append(pd.DataFrame({"origin": te.index, "h": h, "model": name,
                                              "var": np.asarray(v, float), "y": te[f"y_rv_{h}"].values}))
                if np.isfinite(v).all():
                    touch_rows.append(touch_long(te, np.asarray(v, float), h, f"{name}_bm"))

            # Klimatologie + LightGBM-Touch
            for d in ("up", "dn"):
                trs = stack_touch(tr, h, d, ewma_var(tr, h))
                tes = stack_touch(te, h, d, ewma_var(te, h))
                clim = trs.groupby("thr")["hit"].mean()
                touch_rows.append(pd.DataFrame({"origin": tes.index, "h": h, "dir": d, "thr": tes["thr"].values,
                                                "model": "climatology", "p": tes["thr"].map(clim).values,
                                                "hit": tes["hit"].values}))
                if skip_touch:
                    continue
                cf = fh + ["log_b", "z"]
                mono = [(-1 if c in ("log_b", "z") else 0) for c in cf]
                order = np.argsort(trs.index.values, kind="stable")
                trs = trs.iloc[order]
                k = int(len(trs) * 0.9)
                prm = dict(LGB_CLF, monotone_constraints=mono)
                clf = lgb.train(prm, lgb.Dataset(trs[cf].iloc[:k], trs["hit"].iloc[:k]), 1500,
                                valid_sets=[lgb.Dataset(trs[cf].iloc[k:], trs["hit"].iloc[k:])],
                                callbacks=[lgb.early_stopping(100, verbose=False)])
                touch_rows.append(pd.DataFrame({"origin": tes.index, "h": h, "dir": d, "thr": tes["thr"].values,
                                                "model": "lgbm_touch", "p": clf.predict(tes[cf]),
                                                "hit": tes["hit"].values}))
            print(f"  h={h:>3}h  {ms:%Y-%m}: train {len(tr):,}  test {len(te):,}")

    vp = pd.concat(vol_rows, ignore_index=True)
    tp = pd.concat(touch_rows, ignore_index=True)
    vp.to_parquet(C.RESULTS / f"vol_preds_{mode}.parquet")
    tp.to_parquet(C.RESULTS / f"touch_preds_{mode}.parquet")
    vt = vol_table(vp, ref="har")
    tt, tpe = touch_table(tp)
    vt.to_csv(C.RESULTS / f"vol_metrics_{mode}.csv", index=False)
    tt.to_csv(C.RESULTS / f"touch_metrics_{mode}.csv", index=False)
    reliability(tpe).to_csv(C.RESULTS / f"reliability_{mode}.csv", index=False)
    write_report(vt, tt, C.RESULTS / f"RESULTS_{mode}.md", f"Baselines + LightGBM ({mode}: {start} bis {end})",
                 notes="Referenz für DM-Test: HAR. Touch-Modelle mit Suffix _bm nutzen die Brownsche-Bewegung-Formel.")
    print(vt.sort_values(["h", "QLIKE"]).round(4).to_string(index=False))
    print(tt.groupby("model")[["Brier", "LogLoss"]].mean().sort_values("Brier").round(5).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["valid", "test"], default="valid")
    ap.add_argument("--refit-months", type=int, default=1, help="alle wie viele Monate neu fitten")
    ap.add_argument("--skip-garch", action="store_true")
    ap.add_argument("--skip-touch", action="store_true", help="LightGBM-Touch-Klassifikator weglassen (schneller)")
    a = ap.parse_args()
    run(a.mode, a.refit_months, a.skip_garch, a.skip_touch)
