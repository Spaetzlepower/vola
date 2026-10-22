"""Gemeinsame Auswertung: Touch-Formel, Metriken, Signifikanztests, Report."""
import numpy as np
import pandas as pd
from scipy.stats import norm

import config as C


def barriers(thr):
    """Log-Abstände der Schwellen: oben log(1+x), unten |log(1-x)|."""
    thr = np.asarray(thr, float)
    return np.log1p(thr), -np.log1p(-thr)


def bm_touch(var, b):
    """P(Brownsche Bewegung ohne Drift erreicht Abstand b) = 2 * (1 - Phi(b / sigma))."""
    sig = np.sqrt(np.maximum(var, 1e-12))
    return np.clip(2 * norm.sf(b / sig), 1e-6, 1 - 1e-6)


def touch_long(ds_test, var_pred, h, model):
    """Aus einer Varianzprognose Touch-Wahrscheinlichkeiten für alle Schwellen/Richtungen."""
    rows = []
    bu, bd = barriers(C.THRESHOLDS)
    for thr, b_up, b_dn in zip(C.THRESHOLDS, bu, bd):
        for d, b, y in (("up", b_up, ds_test[f"y_up_{h}"] >= b_up), ("dn", b_dn, ds_test[f"y_dn_{h}"] <= -b_dn)):
            rows.append(pd.DataFrame({"origin": ds_test.index, "h": h, "dir": d, "thr": thr, "model": model,
                                      "p": bm_touch(var_pred, b), "hit": y.values.astype(np.int8)}))
    return pd.concat(rows, ignore_index=True)


# ---------------- Metriken ----------------

def qlike(y, f):
    y = np.maximum(y, 1e-12)
    return y / f - np.log(y / f) - 1


def hac_tstat(d, lag):
    """Diebold-Mariano-t-Statistik mit Newey-West-Varianz (Überlappung der Horizonte)."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 20:
        return np.nan, np.nan
    u = d - d.mean()
    v = u @ u / n
    for k in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / n
    t = d.mean() / np.sqrt(max(v, 1e-18) / n)
    return t, 2 * norm.sf(abs(t))


def vol_table(vp, ref="har"):
    """vp: DataFrame(origin, h, model, var, y). Ergebnis pro Horizont und Modell."""
    out = []
    for h, g in vp.groupby("h"):
        piv_f = g.pivot_table(index="origin", columns="model", values="var")
        y = g.groupby("origin")["y"].first().reindex(piv_f.index)
        piv_f = piv_f.dropna()            # nur gemeinsame Zeitpunkte vergleichen
        y = y.reindex(piv_f.index)
        ly = np.log(np.maximum(y, 1e-12))
        lref = qlike(y, piv_f[ref]) if ref in piv_f else None
        for m in piv_f.columns:
            f = piv_f[m]
            ql = qlike(y, f)
            lf = np.log(f)
            beta = np.polyfit(lf, ly, 1)
            r2 = 1 - np.var(ly - np.polyval(beta, lf)) / np.var(ly)
            t, p = hac_tstat(ql - lref, lag=h) if lref is not None and m != ref else (np.nan, np.nan)
            out.append({"h": h, "model": m, "n": len(f), "QLIKE": ql.mean(), "MSE_log": ((ly - lf) ** 2).mean(),
                        "MZ_R2": r2, "MZ_slope": beta[0], "bias_log": (ly - lf).mean(),
                        f"DM_t_vs_{ref}": t, "DM_p": p})
    return pd.DataFrame(out)


def touch_table(tp):
    """tp: DataFrame(origin, h, dir, thr, model, p, hit). Brier/LogLoss, gemittelt über Schwellen."""
    tp = tp.copy()
    # nur Zeitpunkte vergleichen, für die alle Modelle eine Prognose haben
    cnt = tp.groupby(["origin", "h"])["model"].nunique()
    full = cnt[cnt == tp["model"].nunique()].index
    tp = tp.set_index(["origin", "h"]).loc[lambda x: x.index.isin(full)].reset_index()
    p = tp["p"].clip(1e-4, 1 - 1e-4)
    tp["brier"] = (p - tp["hit"]) ** 2
    tp["logloss"] = -(tp["hit"] * np.log(p) + (1 - tp["hit"]) * np.log(1 - p))
    agg = tp.groupby(["h", "dir", "model"]).agg(n=("p", "size"), Brier=("brier", "mean"),
                                                 LogLoss=("logloss", "mean"), mean_p=("p", "mean"),
                                                 hit_rate=("hit", "mean")).reset_index()
    if "climatology" in agg["model"].values:
        clim = agg[agg.model == "climatology"].set_index(["h", "dir"])["Brier"]
        agg["BSS_vs_clim"] = 1 - agg["Brier"] / clim.reindex(pd.MultiIndex.from_frame(agg[["h", "dir"]])).values
    return agg, tp


def reliability(tp, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    tp = tp.copy()
    tp["bin"] = np.clip(np.digitize(tp["p"], edges) - 1, 0, bins - 1)
    return tp.groupby(["h", "model", "bin"]).agg(p_mean=("p", "mean"), hit_rate=("hit", "mean"),
                                                 n=("p", "size")).reset_index()


def write_report(vt, tt, path, title, notes=""):
    lines = [f"# {title}", "", notes, "",
             "## Volatilität (niedriger QLIKE = besser; DM_t < 0 und DM_p < 0.05 = signifikant besser als Referenz)", ""]
    for h, g in vt.groupby("h"):
        lines += [f"### Horizont {h}h", "", g.drop(columns="h").sort_values("QLIKE").round(4).to_markdown(index=False), ""]
    lines += ["## Touch-Wahrscheinlichkeiten (niedriger Brier/LogLoss = besser; BSS > 0 = besser als Klimatologie)", ""]
    for (h, d), g in tt.groupby(["h", "dir"]):
        lines += [f"### Horizont {h}h, Richtung {d}", "",
                  g.drop(columns=["h", "dir"]).sort_values("Brier").round(4).to_markdown(index=False), ""]
    path.write_text("\n".join(lines))
    print(f"Report: {path}")
