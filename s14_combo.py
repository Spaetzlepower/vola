"""Schritt 14: Kombination VolNet + Ensemble (Gewichte je Horizont NUR auf 2025 gewählt, dann 2026 getestet).

  python s14_combo.py

log(Var_combo) = w_h * log(Var_nn) + (1 - w_h) * log(Var_ens_lgbm_all), danach Skalierung c_h so, dass auf 2025
im Mittel y / Var = 1 (QLIKE-optimal). w_h und c_h kommen ausschließlich aus dem Validierungsjahr.
Touch-Wahrscheinlichkeiten: Brownsche-Bewegung-Formel aus Var_combo, isotonisch kalibriert auf 2025.
Ergebnis: results/vol_preds_combo_{valid,test}.parquet (Modell "combo") – wird von s06/s10/s11/s13 automatisch genutzt.
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import config as C
from evaluate import barriers, bm_touch, qlike, reliability, touch_table, vol_table, write_report

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--a", default="nn", help="Modell A (nn, nn2, vf)")
_ap.add_argument("--b", default="ens_lgbm_all", help="Modell B")
_args, _ = _ap.parse_known_args()
A, B = _args.a, _args.b


def load(mode):
    v = pd.concat([pd.read_parquet(C.RESULTS / f"vol_preds_{A}_{mode}.parquet"),
                   pd.read_parquet(C.RESULTS / f"vol_preds_ens_{mode}.parquet"),
                   pd.read_parquet(C.RESULTS / f"vol_preds_{mode}.parquet")])
    return v.drop_duplicates(["origin", "h", "model"])


def pivot(v, h):
    g = v[v.h == h].pivot_table(index="origin", columns="model", values="var")
    y = v[v.h == h].groupby("origin")["y"].first()
    g = g[[A, B]].dropna()
    return g, y.reindex(g.index)


def main():
    vv, vt = load("valid"), load("test")
    ds = pd.read_parquet(C.DATA / "dataset.parquet")
    params, rows = {}, {"valid": [], "test": []}
    for h in C.HORIZONS_H:
        g, y = pivot(vv, h)
        best = None
        for w in np.linspace(0, 1, 11):
            lv = w * np.log(g[A]) + (1 - w) * np.log(g[B])
            c = float(np.mean(y / np.exp(lv)))
            ql = qlike(y, np.exp(lv) * c).mean()
            if best is None or ql < best[0]:
                best = (ql, w, c)
        params[h] = best[1:]
        print(f"h={h:>3}h: Gewicht {A} {best[1]:.1f}, Skalierung {best[2]:.3f} (QLIKE 2025 {best[0]:.4f})")
        for mode, v in (("valid", vv), ("test", vt)):
            gg, yy = pivot(v, h)
            w, c = params[h]
            var = np.exp(w * np.log(gg[A]) + (1 - w) * np.log(gg[B])) * c
            rows[mode].append(pd.DataFrame({"origin": gg.index, "h": h, "model": "combo", "var": var.values, "y": yy.values}))
    bu, bd = barriers(C.THRESHOLDS)
    isos = {}
    for mode in ("valid", "test"):
        vc = pd.concat(rows[mode], ignore_index=True)
        vc["model"] = f"combo_{A}"
        vc.to_parquet(C.RESULTS / f"vol_preds_combo_{A}_{mode}.parquet")
        tps = []
        for h in C.HORIZONS_H:
            s = vc[vc.h == h].set_index("origin")
            d = ds.reindex(s.index)
            for dname, bs in (("up", bu), ("dn", bd)):
                p = np.stack([bm_touch(s["var"].values, x) for x in bs], 1)
                hit = np.stack([(d[f"y_up_{h}"].values >= x) if dname == "up" else (d[f"y_dn_{h}"].values <= -x) for x in bs], 1)
                if mode == "valid":
                    isos[(h, dname)] = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip").fit(p.ravel(), hit.ravel())
                pi = isos[(h, dname)].predict(p.ravel()).reshape(p.shape)
                for j, thr in enumerate(C.THRESHOLDS):
                    tps.append(pd.DataFrame({"origin": s.index, "h": h, "dir": dname, "thr": thr, "model": "combo_iso",
                                             "p": pi[:, j], "hit": hit[:, j].astype(np.int8)}))
        tp = pd.concat(tps, ignore_index=True)
        tp["model"] = f"combo_{A}_iso"
        tp.to_parquet(C.RESULTS / f"touch_preds_combo_{A}_{mode}.parquet")
        V = pd.concat([load(mode), vc])
        V = V[V["model"].isin(["har", "lgbm", A, "ens_lgbm_all", "ens_har_all", f"combo_{A}"])]
        T = pd.concat([tp] + [pd.read_parquet(C.RESULTS / f).query("model in ['nn_bm','nn_bm_iso','ens_har_all_iso','ens_lgbm_all_iso','har_bm']")
                              for f in (f"touch_preds_{A}_{mode}.parquet", f"touch_preds_ens_{mode}.parquet", f"touch_preds_{mode}.parquet")
                              if (C.RESULTS / f).exists()])
        T = T.drop_duplicates(["origin", "h", "dir", "thr", "model"])
        vtab = vol_table(V, ref="har")
        ttab, tpe = touch_table(T)
        write_report(vtab, ttab, C.RESULTS / f"RESULTS_combo_{A}_{mode}.md", f"Kombination VolNet + Ensemble ({mode})",
                     notes="Gewichte/Skalierung je Horizont nur aus 2025: " + ", ".join(f"{h}h: w={w:.1f}, c={c:.3f}" for h, (w, c) in params.items()))
        print(f"\n=== {mode}")
        print(vtab.sort_values(["h", "QLIKE"])[["h", "model", "QLIKE", "DM_t_vs_har", "DM_p"]].round(4).to_string(index=False))
        s = ttab.groupby("model").agg(Brier=("Brier", "mean"), p=("mean_p", "mean"), hit=("hit_rate", "mean"))
        s["Bias"] = s["p"] - s["hit"]
        print(s.drop(columns=["p", "hit"]).sort_values("Brier").round(5).to_string())


if __name__ == "__main__":
    main()
