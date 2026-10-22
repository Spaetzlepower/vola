"""Schritt 5: Stacking + Kalibrierung -> bester fairer Wert.

  python s05_ensemble.py --mode valid          # berechnet FM-Features beim ersten Mal (GPU, ~20-30 Min.)
  python s05_ensemble.py --mode test

1. FM-Features: Chronos-2 und Timestral prognostizieren für JEDE Stunde seit Datenbeginn die Varianz
   aller Horizonte (zero-shot, nur Vergangenheit). Gespeichert in data/fm_features.parquet.
   Achtung: Vor 2026 könnten die Modelle BTC-Daten im Pretraining gesehen haben -> ehrlich ist nur der Test ab 2026.
2. Modelle je Horizont (walk-forward, monatlich neu):
     ens_har       HAR
     ens_har_fm    HAR + FM-Prognosen
     ens_fm_avg    Mittel aus Chronos-2 und Timestral (ohne Fitting)
     ens_lgbm      LightGBM mit allen Features
     ens_lgbm_fm   LightGBM mit allen Features + FM-Prognosen
     ens_dvol      nur Options-Vola (Deribit DVOL), kalibriert  -> "was der Optionsmarkt erwartet"
     ens_har_all   HAR + FM + DVOL
     ens_lgbm_all  LightGBM mit allem (Features + FM + DVOL)
3. Kalibrierung auf den jüngsten 15 % der Trainingsdaten (out-of-sample, fängt Regimewechsel ab):
     Vola: log(y) = a + b * Prognose  (+ Lognormal-Korrektur)
     Touch: isotonische Regression je Horizont/Richtung auf die BM-Wahrscheinlichkeit  -> *_iso
"""
import argparse
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import config as C
from evaluate import barriers, bm_touch, reliability, touch_long, touch_table, vol_table, write_report
from s03_models import HAR_COLS, LGB_REG, denorm, feature_cols, fit_har, month_starts, norm_target, pred_har, train_mask

warnings.filterwarnings("ignore")
H = C.BARS_PER_HOUR
FM_PATH = C.DATA / "fm_features.parquet"


# ---------------- 1. FM-Features ----------------

def build_fm_features(ds, ctx, bs, dry=False):
    import s04_foundation as F
    _, hr = F.hourly_series()
    origins = ds.index[ds.index >= hr.index[0] + pd.Timedelta(hours=168)]
    print(f"FM-Features für {len(origins):,} Stunden (Gerät: {F.DEV}) ...")
    out = pd.DataFrame(index=origins)
    for name, fn in (("ts", lambda o: F.run_timestral(hr, o, "209M", ctx, bs, dry)),
                     ("ch", lambda o: F.run_chronos2(hr, o, ctx, bs, dry))):
        res = {h: [] for h in C.HORIZONS_H}
        step = 4096                                     # in Blöcken -> Speicher bleibt klein
        for i in range(0, len(origins), step):
            o = origins[i:i + step]
            v = fn(o)
            for h in C.HORIZONS_H:
                res[h].append(v[h])
            print(f"  {name}: {min(i + step, len(origins)):,}/{len(origins):,}")
        for h in C.HORIZONS_H:
            var = np.concatenate(res[h])
            out[f"fm_{name}_{h}"] = np.log(np.maximum(var, 1e-14) / (h * H)) - ds.loc[origins, f"seas_fwd_{h}"].values
    out.to_parquet(FM_PATH)
    print(f"OK: {FM_PATH}")
    return out


def dvol_features(ds):
    """Deribit DVOL (30T implizite Vola) als Features; lädt bei Bedarf (gecacht in data/dvol_1h.parquet)."""
    p = C.DATA / "dvol_1h.parquet"
    if p.exists():
        dv = pd.read_parquet(p)["dvol"]
    else:
        try:
            from s06_polymarket import dvol
            dv = dvol(pd.Timestamp("2021-03-24", tz="UTC"), ds.index[-1] + pd.Timedelta(hours=2))
        except Exception as e:
            print(f"DVOL nicht verfügbar ({e}) -> ohne Options-Vola")
            return None
    if dv is None or len(dv) == 0:
        return None
    # Kerze mit Zeitstempel ts schließt bei ts+1h -> erst dann bekannt
    dv = dv.copy()
    dv.index = dv.index + pd.Timedelta(hours=1)
    dv = dv[~dv.index.duplicated()].sort_index()
    x = dv.reindex(ds.index, method="ffill", tolerance=pd.Timedelta("3h"))
    f = pd.DataFrame(index=ds.index)
    f["dvol_lv"] = np.log((x / 100) ** 2 / (365 * 24 * H))          # implizite Varianz pro 5m-Bar (log)
    f["dvol_chg_24h"] = np.log(x / x.shift(24))
    f["dvol_vrp"] = f["dvol_lv"] - ds["rv_30d"]                       # implizit minus realisiert (Risikoprämie)
    print(f"DVOL-Features: {f['dvol_lv'].notna().mean():.0%} der Stunden abgedeckt")
    return f


# ---------------- 2./3. Modelle + Kalibrierung ----------------

def split_fit_cal(tr, h, frac=0.15):
    """Chronologisch: Fit-Teil, Embargo (h Stunden), Kalibrierteil (jüngste 15 %)."""
    k = int(len(tr) * (1 - frac))
    cal = tr.iloc[k:]
    fit = tr[tr.index + pd.Timedelta(hours=h) <= cal.index[0]]
    return fit, cal


def ols(X, y):
    X = np.c_[np.ones(len(X)), X]
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    return beta


def ols_pred(beta, X):
    return np.c_[np.ones(len(X)), X] @ beta


def lgbm_fit(fit, h, feats):
    y = norm_target(fit, h)
    ok = np.isfinite(y)
    f, y = fit[ok], y[ok]
    k = int(len(f) * 0.9)
    m = lgb.train(LGB_REG, lgb.Dataset(f[feats].iloc[:k], y.iloc[:k]), 2000,
                  valid_sets=[lgb.Dataset(f[feats].iloc[k:], y.iloc[k:])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def run(mode, refit_months, ctx, bs, dry=False):
    ds = pd.read_parquet(C.DATA / "dataset.parquet")
    fm = pd.read_parquet(FM_PATH) if FM_PATH.exists() else build_fm_features(ds, ctx, bs, dry)
    ds = ds.join(fm, how="left")
    fm_all = list(fm.columns)
    dvf = dvol_features(ds)
    dv_cols = []
    if dvf is not None:
        ds = ds.join(dvf)
        dv_cols = list(dvf.columns)
    start, end = (C.VALID_START, C.VALID_END) if mode == "valid" else (C.TEST_START, str(ds.index[-1].date()))
    base_feats = feature_cols(ds)
    base_feats = [c for c in base_feats if not c.startswith(("fm_", "dvol_"))]
    starts = month_starts(start, end)
    vol_rows, touch_rows = [], []
    bu, bd = barriers(C.THRESHOLDS)
    print(f"Modus {mode}: {start} bis {end}")

    for h in C.HORIZONS_H:
        fm_h = [f"fm_ts_{h}", f"fm_ch_{h}"]
        extra = [c for c in ds.columns if c in (f"seas_fwd_{h}", f"event_in_{h}")]
        for i in range(0, len(starts), refit_months):
            ms = starts[i]
            me = starts[i + refit_months] if i + refit_months < len(starts) else pd.Timestamp(end, tz="UTC") + pd.Timedelta("1D")
            te = ds[(ds.index >= ms) & (ds.index < me) & ds[f"y_rv_{h}"].notna() & (ds[f"y_miss_{h}"] < 0.05)]
            if te.empty:
                continue
            tr = ds[train_mask(ds, ms, h)]
            tr_fm = tr.dropna(subset=fm_h)
            fit, cal = split_fit_cal(tr, h)
            fit_fm, cal_fm = split_fit_cal(tr_fm, h)
            ext = fm_h + (["dvol_lv", "dvol_vrp"] if dv_cols else [])
            tr_x = tr.dropna(subset=ext)
            fit_x, cal_x = split_fit_cal(tr_x, h) if len(tr_x) > 2000 else (fit_fm, cal_fm)

            # rohe Prognosen (normierte log-Varianz) für Kalibrierteil und Test
            raw = {}
            b = ols(fit[HAR_COLS].values, norm_target(fit, h).values)
            raw["ens_har"] = (ols_pred(b, cal[HAR_COLS].values), ols_pred(b, te[HAR_COLS].values), cal)
            cols = HAR_COLS + fm_h
            b = ols(fit_fm[cols].values, norm_target(fit_fm, h).values)
            raw["ens_har_fm"] = (ols_pred(b, cal_fm[cols].values), ols_pred(b, te[cols].values), cal_fm)
            raw["ens_fm_avg"] = (cal_fm[fm_h].mean(1).values, te[fm_h].mean(1).values, cal_fm)
            if dv_cols and len(tr_x) > 2000:
                # reine Options-Vola (kalibriert) = "was der Optionsmarkt erwartet"
                raw["ens_dvol"] = (cal_x["dvol_lv"].values, te["dvol_lv"].values, cal_x)
                cols = HAR_COLS + ext
                b = ols(fit_x[cols].values, norm_target(fit_x, h).values)
                raw["ens_har_all"] = (ols_pred(b, cal_x[cols].values), ols_pred(b, te[cols].values), cal_x)
            f1 = base_feats + extra
            m = lgbm_fit(fit, h, f1)
            raw["ens_lgbm"] = (m.predict(cal[f1]), m.predict(te[f1]), cal)
            f2 = base_feats + extra + fm_all
            m = lgbm_fit(fit_fm, h, f2)
            raw["ens_lgbm_fm"] = (m.predict(cal_fm[f2]), m.predict(te[f2]), cal_fm)
            if dv_cols and len(tr_x) > 2000:
                f3 = f2 + dv_cols
                m = lgbm_fit(fit_x, h, f3)
                raw["ens_lgbm_all"] = (m.predict(cal_x[f3]), m.predict(te[f3]), cal_x)

            # Referenz: HAR wie in s03 (unkalibriert, auf ganzem Training) für DM-Tests
            preds = {"har": pred_har(fit_har(tr, h), te, h)}
            for name, (p_cal, p_te, c) in raw.items():
                y_cal = norm_target(c, h).values
                ok = np.isfinite(p_cal) & np.isfinite(y_cal)
                if ok.sum() < 200:
                    continue
                a = ols(p_cal[ok, None], y_cal[ok])                     # log(y) = a0 + a1 * Prognose
                s2 = np.var(y_cal[ok] - ols_pred(a, p_cal[ok, None]))
                preds[name] = denorm(ols_pred(a, p_te[:, None]) + 0.5 * s2, te, h)
                cal_var = denorm(ols_pred(a, p_cal[:, None]) + 0.5 * s2, c, h)
                # isotonische Touch-Kalibrierung je Richtung
                v_te = np.asarray(preds[name], float)
                for d in ("up", "dn"):
                    bs_ = bu if d == "up" else bd
                    pc = np.concatenate([bm_touch(cal_var, x) for x in bs_])
                    hc = np.concatenate([(c[f"y_up_{h}"] >= x).values if d == "up" else (c[f"y_dn_{h}"] <= -x).values for x in bs_])
                    okc = np.isfinite(pc)
                    iso = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip").fit(pc[okc], hc[okc])
                    for thr, x in zip(C.THRESHOLDS, bs_):
                        p_raw = bm_touch(v_te, x)
                        hit = (te[f"y_up_{h}"] >= x) if d == "up" else (te[f"y_dn_{h}"] <= -x)
                        good = np.isfinite(p_raw)
                        touch_rows.append(pd.DataFrame({"origin": te.index[good], "h": h, "dir": d, "thr": thr,
                                                        "model": f"{name}_iso", "p": iso.predict(p_raw[good]),
                                                        "hit": hit.values[good].astype(np.int8)}))
            for name, v in preds.items():
                v = np.asarray(v, float)
                ok = np.isfinite(v)
                vol_rows.append(pd.DataFrame({"origin": te.index[ok], "h": h, "model": name, "var": v[ok],
                                              "y": te[f"y_rv_{h}"].values[ok]}))
                touch_rows.append(touch_long(te[ok], v[ok], h, f"{name}_bm"))
            print(f"  h={h:>3}h  {ms:%Y-%m}: fit {len(fit):,} / kal {len(cal):,} / test {len(te):,}")

    vp = pd.concat(vol_rows, ignore_index=True)
    tp = pd.concat(touch_rows, ignore_index=True)
    tag = f"ens_{mode}"
    vp.to_parquet(C.RESULTS / f"vol_preds_{tag}.parquet")
    tp.to_parquet(C.RESULTS / f"touch_preds_{tag}.parquet")
    vt = vol_table(vp, ref="har")
    tt, tpe = touch_table(tp)
    vt.to_csv(C.RESULTS / f"vol_metrics_{tag}.csv", index=False)
    tt.to_csv(C.RESULTS / f"touch_metrics_{tag}.csv", index=False)
    reliability(tpe).to_csv(C.RESULTS / f"reliability_{tag}.csv", index=False)
    write_report(vt, tt, C.RESULTS / f"RESULTS_{tag}.md", f"Ensemble + Kalibrierung ({mode}: {start} bis {end})",
                 notes="Referenz 'har' = HAR aus s03 (unkalibriert). ens_* = kalibriert auf den jüngsten 15 % "
                       "des Trainings. *_bm = Brownsche-Bewegung-Formel, *_iso = zusätzlich isotonisch kalibriert.")
    print(vt.sort_values(["h", "QLIKE"]).round(4).to_string(index=False))
    s = tt.groupby("model").agg(Brier=("Brier", "mean"), LogLoss=("LogLoss", "mean"),
                                Bias=("mean_p", "mean"), Treffer=("hit_rate", "mean"))
    s["Bias"] = s["Bias"] - s["Treffer"]
    print(s.drop(columns="Treffer").sort_values("Brier").round(5).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["valid", "test"], default="valid")
    ap.add_argument("--refit-months", type=int, default=1)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--dry-run", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--rebuild-fm", action="store_true", help="FM-Features neu berechnen")
    a = ap.parse_args()
    if a.rebuild_fm and FM_PATH.exists():
        FM_PATH.unlink()
    run(a.mode, a.refit_months, a.ctx, a.bs, a.dry_run)
