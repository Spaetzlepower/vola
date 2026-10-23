"""Schritt 13: Realistischer Backtest mit ECHTEN Polymarket-Trades (statt Mittelwert-Preisen).

  python s13_pm_trades.py --trades inputs/pm_trades.json
  python s13_pm_trades.py --trades inputs/pm_trades.json --cap 100 --fee 0.0

Logik: Jeder Taker-Trade belegt, dass zu diesem Zeitpunkt ein Gegenangebot zu genau diesem Preis im Orderbuch lag:
  - Taker kauft Yes zu p (oder verkauft No zu 1-p)  -> Yes war zu p KAUFBAR
  - Taker verkauft Yes zu p (oder kauft No zu 1-p)  -> Yes war zu p VERKAUFBAR
Signal, wenn der kaufbare Preis klar unter dem fairen Wert liegt (kaufen) bzw. der verkaufbare klar darüber (verkaufen).
Einstieg:
  "sofort"   zum Preis dieses Trades (optimistisch: man hätte dieselbe Liquidität gebraucht)
  "+Ds"      zum Preis des NÄCHSTEN Trades derselben Art mindestens D Sekunden später (realistisch: Reaktionszeit),
             egal ob besser oder schlechter
Halten bis Auflösung; Menge = min(Trade-Menge, --cap). Gebühr je Anteil: --fee.
Fairer Wert: stündlich = P(Schlusskurs > K), Woche = P(Berührung); Vola aus DVOL·k und (falls vorhanden) Modell.
Kein Blick in die Zukunft: Binance-Kurs = letzte abgeschlossene Minute vor dem Trade, DVOL nach Kerzenschluss,
Modellprognose vom letzten vollen Stundenzeitpunkt; Wochenmärkte nur, solange die Stufe noch nicht berührt war.
"""
import argparse
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import norm

import config as C
from s06_polymarket import model_var_interp, vrp_factor, week_tstat
from s11_pm_hourly import resolution_time

DELAYS = [0, 5, 30, 60, 300]


def market_table(events):
    rows = []
    for slug, ev in events.items():
        ev = ev[0] if isinstance(ev, list) else ev
        if slug.startswith("bitcoin-above-on-"):
            kind, T = "hour", resolution_time(slug)
        elif slug.startswith("what-price-will-bitcoin-hit"):
            kind, T = "week", pd.Timestamp(ev["endDate"]).tz_convert("UTC") if pd.Timestamp(ev["endDate"]).tzinfo \
                else pd.Timestamp(ev["endDate"], tz="UTC")
        else:
            continue
        if T is None:
            continue
        for m in ev.get("markets", []):
            txt = f"{m.get('groupItemTitle') or ''} {m.get('question') or ''}"
            nums = [float(x.replace(",", "")) for x in re.findall(r"(\d{1,3}(?:,\d{3})+|\d{5,7})", txt)]
            if not nums:
                continue
            try:
                toks = json.loads(m["clobTokenIds"]) if isinstance(m.get("clobTokenIds"), str) else m["clobTokenIds"]
                outs = json.loads(m["outcomes"]) if isinstance(m.get("outcomes"), str) else (m.get("outcomes") or ["Yes", "No"])
            except Exception:
                continue
            yi = [o.lower() for o in outs].index("yes") if "yes" in [o.lower() for o in outs] else 0
            up = True
            if kind == "week":
                low = txt.lower()
                up = not ("↓" in txt or re.search(r"\b(dip|drop|fall|below|under|low)\b", low))
            for j, tok in enumerate(toks):
                rows.append({"token": str(tok), "slug": slug, "kind": kind, "T": T, "strike": nums[0], "up": up,
                             "is_yes": j == yi, "market": str(toks[yi])})
    return pd.DataFrame(rows)


def main(a):
    d = json.loads(open(a.trades).read())
    M = market_table(d["events"]).set_index("token")
    tr = pd.DataFrame(d["trades"]).rename(columns={"a": "token", "s": "side", "p": "price", "q": "size", "t": "ts"})
    tr["token"] = tr["token"].astype(str)
    tr = tr[tr["token"].isin(M.index)].copy()
    tr = tr.join(M, on="token")
    tr["time"] = pd.to_datetime(tr["ts"], unit="s", utc=True)
    # in Yes-Preis und Verfügbarkeit umrechnen
    tr["yes_price"] = np.where(tr["is_yes"], tr["price"], 1 - tr["price"])
    buyish = (tr["side"].str.upper() == "BUY")
    tr["avail"] = np.where(tr["is_yes"] == buyish, "ask", "bid")    # ask = Yes kaufbar, bid = Yes verkaufbar
    k1 = pd.read_parquet(C.DATA / "klines_1m.parquet", columns=["high", "low", "close"])
    tr = tr[(tr["T"] <= k1.index.max()) & (tr["time"] < tr["T"])].copy()
    print(f"{len(tr):,} Taker-Trades in {tr['slug'].nunique()} Events ({(tr.kind == 'hour').sum():,} stündlich, "
          f"{(tr.kind == 'week').sum():,} Woche)")

    # Ausgang je Markt
    out = {}
    for mk, g in tr.groupby("market"):
        r = g.iloc[0]
        T, strike = r["T"], r["strike"]                     # Achtung: r.T wäre die Transponierte der Series
        if r["kind"] == "hour":
            out[mk] = (float(k1.at[T, "close"] > strike) if T in k1.index else np.nan, pd.NaT)
        else:
            w = k1[(k1.index >= T - pd.Timedelta(days=7)) & (k1.index < T)]
            hit = (w["high"] >= strike) if r["up"] else (w["low"] <= strike)
            out[mk] = (float(hit.any()), hit.idxmax() if hit.any() else pd.NaT)
    tr["outcome"] = tr["market"].map(lambda m: out[m][0])
    tr["first_hit"] = pd.to_datetime(tr["market"].map(lambda m: out[m][1]), utc=True)
    # Wochenmärkte: nur solange noch nicht berührt (Minute der Berührung ausgeschlossen)
    tr = tr[~((tr.kind == "week") & tr["first_hit"].notna() & (tr["time"] >= tr["first_hit"]))].copy()

    # fairer Wert zum Trade-Zeitpunkt
    minute = tr["time"].dt.floor("1min")
    tr["S"] = k1["close"].reindex(minute - pd.Timedelta("1min"), method="ffill").values
    dvraw = pd.read_parquet(C.DATA / "dvol_1h.parquet")["dvol"]
    dv = dvraw.copy(); dv.index = dv.index + pd.Timedelta(hours=1)
    k = vrp_factor(dvraw, pd.read_parquet(C.DATA / "bars_5m.parquet", columns=["r"]), tr["time"].min() - pd.Timedelta(days=8))
    dvt = dv.reindex(tr["time"], method="ffill", tolerance=pd.Timedelta("3h")).values / 100 * k
    sec = lambda x: ((x - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta("1s")).values   # einheitenunabhängig
    T_end = np.where(tr["kind"] == "hour", sec(tr["T"] + pd.Timedelta("1min")), sec(tr["T"]))
    tau_h = (T_end - sec(tr["time"])) / 3600.0
    tr["tau_h"] = tau_h
    var_src = {"dvol": dvt ** 2 * tau_h / 8760}
    vp = None
    for f in sorted(C.RESULTS.glob("vol_preds_*.parquet")):
        if "foundation" not in f.name and "kronos" not in f.name:
            x = pd.read_parquet(f); vp = x if vp is None else pd.concat([vp, x])
    if vp is not None:
        vp = vp.drop_duplicates(["origin", "h", "model"])
        for m in [a.model, "nn", "ens_lgbm_all", "lgbm", "har"]:
            if m and m in vp["model"].unique():
                need = pd.DatetimeIndex(tr["time"].dt.floor("1h").unique())
                if need.isin(vp.loc[vp.model == m, "origin"]).mean() > 0.8:
                    fn = model_var_interp(vp, m)
                    orig = tr["time"].dt.floor("1h")
                    e = ((tr["time"] - orig) / pd.Timedelta("1h")).values
                    var_src[m] = fn(orig, tau_h + e) - fn(orig, e)
                    break
    S, K = tr["S"].values, tr["strike"].values
    for name, var in var_src.items():
        sig = np.sqrt(np.maximum(var, 1e-12))
        p_digital = norm.cdf((np.log(S / K) - 0.5 * sig ** 2) / sig)
        b = np.where(tr["up"], np.log(K / S), np.log(S / K))
        p_touch = np.where(b <= 0, 1.0, np.clip(2 * norm.sf(np.maximum(b, 0) / sig), 0, 1))
        tr[f"fv_{name}"] = np.where(tr["kind"] == "hour", p_digital, p_touch)
    tr = tr.dropna(subset=["S", "outcome"]).sort_values(["market", "time"]).reset_index(drop=True)
    tr["day"] = tr["time"].dt.floor("1D")
    fvs = [c for c in tr.columns if c.startswith("fv_")]

    L = ["# Backtest mit echten Polymarket-Trades", "",
         f"{len(tr):,} Taker-Trades, {tr['slug'].nunique()} Events. Menge je Papier-Trade = min(Trade-Menge, {a.cap}). "
         f"Gebühr {a.fee} je Anteil. DVOL-Faktor k = {k:.3f}. Halten bis Auflösung.", "",
         "Ein Trade zu Preis p belegt ein Gegenangebot zu p. 'sofort' = dieselbe Liquidität (optimistisch); "
         "'+Ds' = nächster Trade derselben Art ≥ D Sekunden später (realistisch). t-Statistik über Tage.", ""]
    # Überblick Liquidität
    ov = tr.groupby("kind").agg(Trades=("size", "size"), Menge_Median=("size", "median"),
                                Volumen_USD=("size", lambda s: (s * tr.loc[s.index, "yes_price"]).sum()))
    L += ["## Liquidität", "", ov.round(1).to_markdown(), ""]

    # nächste Trades derselben Art (für verzögerten Einstieg)
    tr["_tns"] = ((tr["time"] - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta("1s")).astype("float64")
    nxt = {D: np.full(len(tr), -1) for D in DELAYS if D > 0}
    for (mk, av), g in tr.groupby(["market", "avail"]):
        tt = g["_tns"].values
        for D in nxt:
            j = np.searchsorted(tt, tt + D, side="left")
            ok = j < len(g)
            jj = np.where(ok, j, 0)
            ok &= (tt[jj] - tt) <= (D + a.window)                       # Suchfenster für den nächsten Trade
            nxt[D][g.index.values[ok]] = g.index.values[jj[ok]]

    res, best_detail = [], None
    for kind in ("hour", "week"):
        for fv in fvs:
            for th in a.th:
                base = tr[(tr.kind == kind)]
                gap = np.where(base["avail"] == "ask", base[fv] - base["yes_price"], base["yes_price"] - base[fv])
                sig = base[(gap > th + a.fee) & (base["yes_price"] > 0.01) & (base["yes_price"] < 0.99)]
                # max. 1 Signal je Markt und Minute
                sig = sig.loc[~sig.assign(m=sig["time"].dt.floor("1min")).duplicated(["market", "m"])]
                for D in DELAYS:
                    if D == 0:
                        e = sig["yes_price"].values; q = np.minimum(sig["size"].values, a.cap); idx = sig.index.values
                    else:
                        j = nxt[D][sig.index.values]
                        ok = j >= 0
                        idx = sig.index.values[ok]; jj = j[ok]
                        e = tr.loc[jj, "yes_price"].values; q = np.minimum(tr.loc[jj, "size"].values, a.cap)
                    if len(idx) < 20:
                        continue
                    S_ = tr.loc[idx]
                    buy = (S_["avail"] == "ask").values
                    pnl = np.where(buy, S_["outcome"].values - e, e - S_["outcome"].values) - a.fee
                    cap = np.where(buy, e, 1 - e)
                    t, nd = week_tstat(pnl, S_["day"].values)
                    dd = pd.Series(pnl * q).groupby(S_["day"].values).sum()
                    row = {"Markt": "stündlich" if kind == "hour" else "Woche", "Quelle": fv, "Schwelle": th,
                           "Einstieg": "sofort" if D == 0 else f"+{D}s", "Trades": len(idx), "Tage": nd,
                           "PnL/Anteil": pnl.mean(), "PnL gesamt $": (pnl * q).sum(), "Rendite/$": (pnl * q).sum() / (cap * q).sum(),
                           "Treffer": (pnl > 0).mean(), "pos. Tage": (dd > 0).mean(), "t(Tage)": t}
                    res.append(row)
                    if D == 30 and th == a.th[-1] and fv == fvs[-1]:
                        best_detail = (kind, S_.assign(pnl=pnl, q=q, e=e, buy=buy))
    R = pd.DataFrame(res)
    for kind, g in R.groupby("Markt", sort=False):
        L += [f"## {kind}", "", g.drop(columns="Markt").round(4).to_markdown(index=False), ""]
    # Aufschlüsselung für einen realistischen Fall (+30 s, letzte Schwelle, letzte Quelle), je Markttyp
    for kind in ("hour", "week"):
        for fv in fvs[-1:]:
            base = tr[tr.kind == kind]
            gap = np.where(base["avail"] == "ask", base[fv] - base["yes_price"], base["yes_price"] - base[fv])
            sig = base[(gap > a.th[-1] + a.fee) & (base["yes_price"] > 0.01) & (base["yes_price"] < 0.99)]
            j = nxt[30][sig.index.values]; ok = j >= 0
            if ok.sum() < 20:
                continue
            S_ = sig[ok].copy(); e = tr.loc[j[ok], "yes_price"].values
            buy = (S_["avail"] == "ask").values
            S_["pnl"] = np.where(buy, S_["outcome"].values - e, e - S_["outcome"].values) - a.fee
            S_["Seite"] = np.where(buy, "Yes kaufen", "Yes verkaufen")
            S_["Restlaufzeit"] = pd.cut(S_["tau_h"] * 60, [0, 5, 15, 60, 360, 1e6], labels=["<5 Min", "5-15", "15-60", "1-6h", ">6h"])
            S_["Preis"] = pd.cut(e, [0, .1, .3, .5, .7, .9, 1])
            L += [f"### Aufschlüsselung {'stündlich' if kind == 'hour' else 'Woche'} ({fv}, Schwelle {a.th[-1]}, Einstieg +30s)", ""]
            for col in ("Seite", "Restlaufzeit", "Preis"):
                L += [g_.round(4).to_markdown() for g_ in [S_.groupby(col, observed=True).agg(
                    n=("pnl", "size"), PnL=("pnl", "mean"), Treffer=("pnl", lambda x: (x > 0).mean()))]] + [""]
    L += ["## Lesehilfe", "",
          "- Entscheidend sind die Zeilen **+30s / +60s**: So schnell kann man realistisch ohne Hochfrequenz-Technik reagieren.",
          "- Positiv mit t > 2 über viele Tage und **gleichmäßig über die Tage** (pos. Tage deutlich > 50 %) → echter Edge.",
          "- Nur 'sofort' positiv → Liquidität war nur für den schnellsten Bot da.",
          "- Gebühren prüfen: Polymarket erhebt bei manchen Krypto-Märkten Taker-Gebühren (--fee).", ""]
    outp = C.RESULTS / "RESULTS_pm_trades.md"
    outp.write_text("\n".join(L))
    tr.drop(columns=["_tns"]).to_parquet(C.RESULTS / "pm_trades_panel.parquet")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", default="inputs/pm_trades.json")
    ap.add_argument("--th", type=float, nargs="+", default=[0.03, 0.05, 0.10])
    ap.add_argument("--cap", type=float, default=100, help="max. Anteile je Papier-Trade")
    ap.add_argument("--fee", type=float, default=0.0, help="Gebühr je Anteil (absolut)")
    ap.add_argument("--model", default="")
    ap.add_argument("--window", type=float, default=900, help="Suchfenster (s) für den nächsten Trade beim verzögerten Einstieg")
    main(ap.parse_args())
