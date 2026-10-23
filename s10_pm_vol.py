"""Schritt 10: Vola-Trade auf Polymarket – Paar aus nächster Stufe darüber und darunter.

  python s10_pm_vol.py                    # nutzt results/pm_panel.parquet (s06) + Vola-Prognosen (s03/s05)
  python s10_pm_vol.py --cost 0.02 --th 0.1 0.25 0.5

Idee: Aus den Preisen der nächsten Stufe darüber und darunter lässt sich die von Polymarket eingepreiste
Vola zurückrechnen (Touch-Formel umgekehrt). Erwartet unser Modell für die nächsten H Stunden MEHR Vola
-> "Yes" auf beide Stufen kaufen (Long Strangle), WENIGER -> "No" auf beide kaufen (Vola verkaufen).
Nach H Stunden schließen (1h, 4h, 24h) oder bis zum Ende halten.
Das Paar ist weitgehend richtungsneutral -> testet die Vola-Prognose, nicht die Latenz.

Kein Blick in die Zukunft: Prognose zum Zeitpunkt t nutzt nur Daten vor t; Einstieg zum Polymarket-Preis
bei t (letzter Kurspunkt <= t) bzw. vorsichtig erst bei t+1h; realisierte Vola nur zur Auswertung.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import norm

import config as C
from s06_polymarket import model_var_interp, price_at_time, week_tstat


def load_preds():
    vp = None
    for f in sorted(C.RESULTS.glob("vol_preds_*.parquet")):
        if "foundation" in f.name or "kronos" in f.name:
            continue
        x = pd.read_parquet(f)
        vp = x if vp is None else pd.concat([vp, x])
    if vp is None:
        raise SystemExit("Keine Vola-Prognosen gefunden – erst s03_models.py (valid + test) laufen lassen.")
    return vp.drop_duplicates(["origin", "h", "model"])


def pairs(P):
    """Je (Woche, Zeitpunkt) die nächste noch offene Stufe darüber und darunter."""
    Q = P[(P["pm"] > 0.02) & (P["pm"] < 0.98) & (P["b"] > 0) & (P["T_h"] > 1)]
    idx = Q.groupby(["event", "t", "dir"])["b"].idxmin()
    w = idx.unstack("dir").dropna()
    w.columns = [f"i_{c}" for c in w.columns]
    return w.astype(int).reset_index()


def implied_ann_vol(pm, b, T):
    """Touch-Formel p = 2*(1-Phi(b/sigma)) umgekehrt -> annualisierte Vola."""
    sig = b / norm.isf(np.clip(pm, 1e-4, 0.9999) / 2)
    return sig / np.sqrt(T / 8760)


def realized_remaining(P, W):
    """Realisierte annualisierte Vola von t bis Wochenende (nur zur Auswertung)."""
    b = pd.read_parquet(C.DATA / "bars_5m.parquet", columns=["r"])
    cum = (b["r"] ** 2).resample("1h").sum().cumsum()
    r = P.loc[W["i_up"]]
    c0 = cum.reindex(r["t"] - pd.Timedelta("1h"), method="ffill").values
    c1 = cum.reindex(r["end"] - pd.Timedelta("1h"), method="ffill").values
    return np.sqrt(np.maximum(c1 - c0, 1e-12) * 8760 / r["T_h"].values)


def main(a):
    P = pd.read_parquet(C.RESULTS / "pm_panel.parquet").reset_index(drop=True)
    vp = load_preds()
    W = pairs(P)
    U, D = P.loc[W["i_up"]].reset_index(drop=True), P.loc[W["i_dn"]].reset_index(drop=True)
    W["T_h"] = U["T_h"].values
    W["iv_up"] = implied_ann_vol(U["pm"].values, U["b"].values, U["T_h"].values)
    W["iv_dn"] = implied_ann_vol(D["pm"].values, D["b"].values, D["T_h"].values)
    W["iv_pm"] = (W["iv_up"] + W["iv_dn"]) / 2
    W["rv_rest"] = realized_remaining(P, W)
    models = [m for m in vp["model"].unique() if m in ("har", "lgbm", "nn") or m.startswith("ens_")]
    print(f"{len(W):,} Paare aus {W['event'].nunique()} Wochen; Modelle: {models}")

    L = ["# Vola-Trade auf Polymarket (Paar: nächste Stufe darüber + darunter)", "",
         f"{len(W):,} Paar-Zeitpunkte, {W['event'].nunique()} Wochen. Kosten {a.cost} je Anteil und Bein "
         f"(Hin+Rück; bei Auflösung halb).", ""]

    # --- Diagnose: preist Polymarket die Vola richtig ein? ---
    W["T_bin"] = pd.cut(W["T_h"], [0, 24, 72, 120, 1000], labels=["<1T", "1-3T", "3-5T", ">5T"])
    diag = W.groupby("T_bin", observed=True).agg(n=("iv_pm", "size"), implizit_PM=("iv_pm", "median"),
                                                 realisiert=("rv_rest", "median"))
    diag["Verhältnis impl/real"] = (W["iv_pm"] / W["rv_rest"]).groupby(W["T_bin"], observed=True).median()
    L += ["## Preist Polymarket die Vola richtig ein? (annualisiert, Median, bis Wochenende)", "",
          "Verhältnis > 1: Polymarket erwartet mehr Schwankung als eintritt (Vola verkaufen tendenziell lohnend).", "",
          diag.round(3).to_markdown(), ""]

    up_all, dn_all = P.loc[W["i_up"]], P.loc[W["i_dn"]]
    res = []

    def pnl_pair(side, delay, H, end=False):
        """side +1 = Yes auf beide (long Vola), -1 = No auf beide (short Vola)."""
        iu, idn = W["i_up"].values, W["i_dn"].values
        if delay:
            t0 = up_all["t"].values + np.timedelta64(delay, "h")
            eu = price_at_time(P, t0, iu).values
            ed = price_at_time(P, t0, idn).values
        else:
            eu, ed = up_all["pm"].values, dn_all["pm"].values
        if end:
            xu, xd, c = up_all["outcome"].values, dn_all["outcome"].values, a.cost / 2
        else:
            t1 = up_all["t"].values + np.timedelta64(delay + H, "h")
            xu = price_at_time(P, t1, iu).values
            xd = price_at_time(P, t1, idn).values
            c = a.cost
        pnl = side * ((xu - eu) + (xd - ed)) - 2 * c
        cap = np.where(side > 0, eu + ed, (1 - eu) + (1 - ed))
        ok = np.isfinite(pnl) & (eu > 0.001) & (eu < 0.999) & (ed > 0.001) & (ed < 0.999)
        return pnl, cap, ok

    def add(name, H, delay, side, mask, end=False):
        side = np.asarray(side, float)
        pnl, cap, ok = pnl_pair(side, delay, H, end)
        m = np.asarray(mask) & ok & (side != 0)
        if m.sum() < 30:
            return
        t, nw = week_tstat(pnl[m], W["event"].values[m])
        wk = pd.Series(pnl[m]).groupby(W["event"].values[m]).mean()
        res.append({"Strategie": name, "Halten": "bis Ende" if end else f"{H}h", "Einstieg": f"+{delay}h",
                    "Trades": int(m.sum()), "Wochen": nw, "PnL/Paar": pnl[m].mean(),
                    "Rendite/$": pnl[m].sum() / cap[m].sum(), "Treffer": (pnl[m] > 0).mean(),
                    "pos. Wochen": (wk > 0).mean(), "t(Wochen)": t})

    allm = np.ones(len(W), bool)
    for H in (1, 4, 24):
        for delay in (0, 1):
            add("Benchmark: immer Vola kaufen", H, delay, np.ones(len(W)), allm)
            add("Benchmark: immer Vola verkaufen", H, delay, -np.ones(len(W)), allm)
        for m in models:
            g = vp[(vp["model"] == m) & (vp["h"] == H)].set_index("origin")["var"]
            fc = np.sqrt(g.reindex(W["t"]).values * 8760 / H)                 # Prognose nächste H Stunden, annualisiert
            ratio = fc / W["iv_pm"].values
            for th in a.th:
                side = np.where(ratio > 1 + th, 1, np.where(ratio < 1 / (1 + th), -1, 0))
                for delay in (0, 1):
                    add(f"{m}, Schwelle {th:.2f}", H, delay, side, np.isfinite(ratio))
    # bis Ende halten: Prognose für die Restlaufzeit
    for m in models:
        fn = model_var_interp(vp, m)
        if fn is None:
            continue
        var = fn(W["t"], W["T_h"].values)
        ratio = np.sqrt(var * 8760 / W["T_h"].values) / W["iv_pm"].values
        for th in a.th:
            side = np.where(ratio > 1 + th, 1, np.where(ratio < 1 / (1 + th), -1, 0))
            add(f"{m}, Schwelle {th:.2f}", None, 0, side, np.isfinite(ratio), end=True)
    add("Benchmark: immer Vola verkaufen", None, 0, -np.ones(len(W)), allm, end=True)
    add("Benchmark: immer Vola kaufen", None, 0, np.ones(len(W)), allm, end=True)

    R = pd.DataFrame(res)
    for hold, g in R.groupby("Halten", sort=False):
        L += [f"## Halten: {hold}", "", g.drop(columns="Halten").sort_values(["Einstieg", "t(Wochen)"], ascending=[True, False])
              .round(4).to_markdown(index=False), ""]
    L += ["## Lesehilfe", "",
          "- Ein Modell ist nur interessant, wenn es die **Benchmarks** (immer kaufen/verkaufen) klar schlägt, "
          "nach Kosten positiv ist und auch mit **Einstieg +1h** hält.",
          "- Viele Kombinationen getestet → einzelne Treffer mit t ≈ 2 können Zufall sein. Entscheidend ist ein "
          "konsistentes Muster über Horizonte und Schwellen.", ""]
    out = C.RESULTS / "RESULTS_pm_vol.md"
    out.write_text("\n".join(L))
    print("\n".join(L[:14]))
    print(R.sort_values("t(Wochen)", ascending=False).head(15).round(4).to_string(index=False))
    print(f"Report: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=0.02)
    ap.add_argument("--th", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    main(ap.parse_args())
