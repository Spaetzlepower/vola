"""Schritt 7: Optionsmarkt – ist unsere 30-Tage-Vola-Prognose besser als die implizite Vola (DVOL)?

  python s07_options.py --mode long      # 2023 bis heute (mehr Trades, empfohlen für den Überblick)
  python s07_options.py --mode valid     # 2025
  python s07_options.py --mode test      # ab 2026 (ehrlichster Test)
  python s07_options.py --cost 1.5       # Kosten je Trade in Vola-Punkten (Spread + Delta-Hedging)

Idee: Wer eine bessere Vola-Prognose hat als der Optionsmarkt, verkauft Vola, wenn die Prognose unter der
impliziten liegt (z. B. delta-gehedgter Straddle), und kauft sie im umgekehrten Fall. Gewinn je Trade
≈ (implizite Vola - realisierte Vola) in Vola-Punkten (Vorzeichen je nach Richtung).

Prognosezeitpunkte: täglich 08:00 UTC (Deribit-Verfallszeit), Horizont 30 Tage.
Modelle (walk-forward, monatlich neu, Embargo 30 Tage):
  iv_raw     DVOL direkt
  iv_adj     DVOL x k (k = mittleres Verhältnis realisiert/implizit im Training -> Risikoprämie raus)
  har30      HAR auf realisierter Vola (1T, 7T, 30T)
  har30_iv   HAR + DVOL
  lgbm30     LightGBM mit allen Features + DVOL

Wichtige Einschränkung: DVOL ist ein Index (modellfreie 30T-Vola inkl. Skew, ähnlich VIX), nicht direkt handelbar.
ATM-Straddles notieren meist etwas UNTER DVOL -> der Ertrag von "Vola verkaufen" wird hier eher überschätzt.
"""
import argparse
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

import config as C
from evaluate import hac_tstat, qlike
from s03_models import LGB_REG, feature_cols

warnings.filterwarnings("ignore")
HZ = 30 * 24                      # Horizont in Stunden
ANN = 365 / 30


def load():
    ds = pd.read_parquet(C.DATA / "dataset.parquet")
    b = pd.read_parquet(C.DATA / "bars_5m.parquet", columns=["r"])
    p = C.DATA / "dvol_1h.parquet"
    if p.exists():
        dv = pd.read_parquet(p)["dvol"]
    else:
        from s06_polymarket import dvol
        dv = dvol(pd.Timestamp("2021-03-24", tz="UTC"), ds.index[-1] + pd.Timedelta(hours=2))
    dv = dv.copy()
    dv.index = dv.index + pd.Timedelta(hours=1)          # Stundenkerze erst nach Schluss bekannt
    dv = dv[~dv.index.duplicated()].sort_index()
    # Ziel: realisierte Varianz der nächsten 30 Tage ab Prognosezeitpunkt
    r2h = (b["r"] ** 2).resample("1h").sum()
    fwd = r2h[::-1].rolling(HZ, min_periods=HZ).sum()[::-1]          # Stunden [t, t+30T)
    d = ds[ds.index.hour == 8].copy()
    d["y30"] = fwd.reindex(d.index).values
    d["iv"] = dv.reindex(d.index, method="ffill", tolerance=pd.Timedelta("3h")).values
    d["iv_var"] = (d["iv"] / 100) ** 2 / ANN                         # implizite Varianz über 30 Tage
    d["log_iv"] = np.log(d["iv_var"])
    d["dvol_chg_7d"] = np.log(d["iv"] / d["iv"].shift(7))
    d["vrp_now"] = d["log_iv"] - (d["rv_30d"] + np.log(HZ * 12))     # implizit vs. realisiert letzte 30T
    for c in ("rv_24h", "rv_7d", "rv_30d"):
        d[c + "_tot"] = d[c] + np.log(HZ * 12)                       # auf 30-Tage-Varianz skaliert
    d = d.dropna(subset=["iv"])
    print(f"{len(d):,} Tage mit DVOL, {d.index[0].date()} bis {d.index[-1].date()} "
          f"(Ziel bekannt bis {d['y30'].last_valid_index().date()})")
    return d


def ols(X, y):
    X = np.c_[np.ones(len(X)), X]
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    res = y[ok] - X[ok] @ beta
    return beta, np.var(res)


def pred(beta, X):
    return np.c_[np.ones(len(X)), X] @ beta


def run(mode, cost, thresholds):
    d = load()
    last = d["y30"].last_valid_index()
    start = {"long": "2023-01-01", "valid": C.VALID_START, "test": C.TEST_START}[mode]
    end = C.VALID_END if mode == "valid" else str(last.date())
    feats = [c for c in feature_cols(d) if c not in ("y30", "iv", "iv_var")]   # enthält log_iv, dvol_chg_7d, vrp_now
    feats = list(dict.fromkeys(feats))
    har = ["rv_24h_tot", "rv_7d_tot", "rv_30d_tot"]
    rows = []
    for ms in pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), freq="MS"):
        me = ms + pd.offsets.MonthBegin(1)
        te = d[(d.index >= ms) & (d.index < me) & d["y30"].notna()]
        tr = d[(d.index + pd.Timedelta(hours=HZ) <= ms) & d["y30"].notna()]
        if te.empty or len(tr) < 200:
            continue
        ly = np.log(tr["y30"].values)
        out = {"iv_raw": te["iv_var"].values}
        k = np.mean(ly - tr["log_iv"].values)                      # log-Verhältnis realisiert/implizit
        s2k = np.var(ly - tr["log_iv"].values - k)
        out["iv_adj"] = np.exp(te["log_iv"].values + k + 0.5 * s2k)
        b, s2 = ols(tr[har].values, ly)
        out["har30"] = np.exp(pred(b, te[har].values) + 0.5 * s2)
        b, s2 = ols(tr[har + ["log_iv"]].values, ly)
        out["har30_iv"] = np.exp(pred(b, te[har + ["log_iv"]].values) + 0.5 * s2)
        kk = int(len(tr) * 0.85)
        m = lgb.train(dict(LGB_REG, min_data_in_leaf=40, num_leaves=15), lgb.Dataset(tr[feats].iloc[:kk], ly[:kk]), 1000,
                      valid_sets=[lgb.Dataset(tr[feats].iloc[kk:], ly[kk:])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        s2 = np.var(ly[kk:] - m.predict(tr[feats].iloc[kk:]))
        out["lgbm30"] = np.exp(m.predict(te[feats]) + 0.5 * s2)
        for name, f in out.items():
            rows.append(pd.DataFrame({"origin": te.index, "model": name, "f": f, "y": te["y30"].values,
                                      "iv_var": te["iv_var"].values}))
    P = pd.concat(rows, ignore_index=True)
    P.to_parquet(C.RESULTS / f"options_preds_{mode}.parquet")
    vol = lambda v: np.sqrt(v * ANN) * 100                              # Varianz (30T) -> annualisierte Vola in %

    # ---- Prognosegüte ----
    L = [f"# Optionsmarkt: 30-Tage-Prognose vs. DVOL ({mode}: {start} bis {end})", "",
         f"{P['origin'].nunique()} Tage (täglich 08:00 UTC, Horizont 30 Tage → aufeinanderfolgende Tage überlappen stark; "
         f"effektiv nur ~{P['origin'].nunique() // 30} unabhängige Perioden). Kosten je Trade: {cost} Vola-Punkte.", ""]
    ref = P[P.model == "iv_raw"].set_index("origin")
    met = []
    for m, g in P.groupby("model"):
        g = g.set_index("origin")
        ql = qlike(g["y"], g["f"])
        ly, lf = np.log(g["y"]), np.log(g["f"])
        t, pv = (hac_tstat(ql - qlike(ref["y"], ref["f"]).reindex(g.index), lag=30) if m != "iv_raw" else (np.nan, np.nan))
        met.append({"Modell": m, "QLIKE": ql.mean(), "MSE_log": ((ly - lf) ** 2).mean(),
                    "R2_log": 1 - ((ly - lf - (ly - lf).mean()) ** 2).sum() / ((ly - ly.mean()) ** 2).sum(),
                    "Ø Prognose (Vola %)": vol(g["f"]).mean(), "Ø realisiert (Vola %)": vol(g["y"]).mean(),
                    "DM_t vs iv_raw": t, "DM_p": pv})
    L += ["## Prognosegüte (niedriger QLIKE = besser; DM_t < 0 und p < 0.05 = signifikant besser als DVOL)", "",
          pd.DataFrame(met).sort_values("QLIKE").round(4).to_markdown(index=False), ""]

    # ---- Handel: Vola kaufen/verkaufen ----
    iv_vol = vol(ref["iv_var"])
    rv_vol = vol(ref["y"])
    edge_short = iv_vol - rv_vol                                       # Gewinn "Vola verkaufen" je Trade (Vola-Punkte)
    trades = []

    def summarize(name, sign, mask):
        pnl = (sign * edge_short - cost)[mask]
        if mask.sum() < 10:
            return
        t, _ = hac_tstat(pnl.values, lag=30)
        trades.append({"Strategie": name, "Tage im Markt": int(mask.sum()), "Ø PnL (Vola-Pkt)": pnl.mean(),
                       "Trefferquote": (pnl > 0).mean(), "schlechtester": pnl.min(),
                       "Ø PnL / Std": pnl.mean() / pnl.std(), "t (HAC)": t})

    allm = pd.Series(True, index=ref.index)
    summarize("Immer Vola verkaufen (Risikoprämie, Benchmark)", 1, allm)
    for m in [x for x in P["model"].unique() if x != "iv_raw"]:
        g = P[P.model == m].set_index("origin").reindex(ref.index)
        gap = vol(g["f"]) - iv_vol                                     # Prognose minus implizit (Vola-Punkte)
        for th in thresholds:
            summarize(f"{m}: verkaufen wenn Prognose < IV - {th}", 1, gap < -th)
            summarize(f"{m}: kaufen wenn Prognose > IV + {th}", -1, gap > th)
        summarize(f"{m}: Richtung nach Vorzeichen", np.sign(-gap), allm & gap.notna())
    T = pd.DataFrame(trades)
    L += ["## Handel (Gewinn je Trade in Vola-Punkten nach Kosten, Vega-normiert)", "",
          "1 Vola-Punkt ≈ Vega des Straddles. Verkaufen = Gewinn wenn realisierte < implizite Vola. "
          "t (HAC) berücksichtigt die Überlappung. Ein Modell lohnt sich nur, wenn es **besser als die Benchmark "
          "'Immer verkaufen'** ist, vor allem beim schlechtesten Trade und beim Verhältnis PnL/Std.", "",
          T.round(3).to_markdown(index=False), ""]
    # Monatsverlauf der Benchmark (zeigt Crash-Risiko)
    mon = pd.DataFrame({"implizit": iv_vol, "realisiert": rv_vol, "verkaufen-PnL": edge_short - cost}).resample("MS").mean()
    L += ["## Monatsverlauf (Mittel je Monat, Vola in %)", "", mon.round(1).to_markdown(), ""]
    out = C.RESULTS / f"RESULTS_options_{mode}.md"
    out.write_text("\n".join(L))
    print(pd.DataFrame(met).sort_values("QLIKE").round(4).to_string(index=False))
    print(T.sort_values("Ø PnL / Std", ascending=False).head(12).round(3).to_string(index=False))
    print(f"Report: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["long", "valid", "test"], default="long")
    ap.add_argument("--cost", type=float, default=1.5, help="Kosten je Trade in Vola-Punkten")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.0, 3.0, 6.0])
    a = ap.parse_args()
    run(a.mode, a.cost, a.thresholds)
