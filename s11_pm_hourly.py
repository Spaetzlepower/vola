"""Schritt 11: Stündliche Polymarket-Märkte "Bitcoin above K on <Datum>, <Uhrzeit> ET?" (Minutendaten).

  python s11_pm_hourly.py --dump inputs/pm_hourly_dump.json
  python s11_pm_hourly.py --dump inputs/pm_hourly_dump.json --cost 0.02 --model ens_lgbm_all

Auflösung: Schlusskurs der Binance-BTCUSDT-1m-Kerze zur genannten Uhrzeit (ET) > K -> Yes.
Fairer Wert (Endkurs, nicht Berührung):  P(S_T > K) = Phi( (ln(S/K) - sigma^2/2) / sigma ),
sigma^2 = erwartete Varianz von jetzt bis zur Auflösung, aus
  fv_dvol   Deribit-DVOL x k (Risikoprämie raus)
  fv_model  unser Vola-Modell (Prognosen aus s03/s05, auf die Restlaufzeit interpoliert)

Auswertung:
  a) Kalibrierung / Brier: Polymarket vs. faire Werte, nach Restlaufzeit
  b) Lücken-Handel bis Auflösung, Reaktionszeit 0..30 Min. (Kosten: halber Spread beim Einstieg)
  c) Vola-Trade: implizite Vola aus den Preisen nahe am Kurs vs. Modellprognose ->
     "Strangle" aus zwei Digitals (Yes über K_oben + No über K_unten) oder Gegenposition
Kein Blick in die Zukunft: Kurspunkte gelten erst 1 Min. nach Zeitstempel; Binance-Kurs = letzte abgeschlossene Minute;
DVOL nach Kerzenschluss; Modellprognose vom letzten vollen Stundenzeitpunkt vor t.
"""
import argparse
import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import norm

import config as C
from s06_polymarket import model_var_interp, vrp_factor, week_tstat

NY = ZoneInfo("America/New_York")
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]
LAGS = [0, 1, 2, 5, 10, 15, 30]


def resolution_time(slug):
    m = re.search(r"bitcoin-above-on-([a-z]+)-(\d+)-(\d{4})-(\d+)(am|pm)-et", slug)
    if not m:
        return None
    mon, day, year, h, ap = MONTHS.index(m.group(1)) + 1, int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(5)
    h = (h % 12) + (12 if ap == "pm" else 0)
    return pd.Timestamp(datetime(year, mon, day, h, tzinfo=NY)).tz_convert("UTC")


def parse(d):
    rows = []
    for slug, evs in d["events"].items():
        ev = evs[0] if isinstance(evs, list) else evs
        T = resolution_time(slug)
        if T is None:
            continue
        for m in ev.get("markets", []):
            txt = f"{m.get('groupItemTitle') or ''} {m.get('question', '')}"
            nums = [float(x.replace(",", "")) for x in re.findall(r"(\d{1,3}(?:,\d{3})+|\d{5,7})", txt)]
            if not nums:
                continue
            try:
                toks = json.loads(m["clobTokenIds"]) if isinstance(m.get("clobTokenIds"), str) else m["clobTokenIds"]
                outs = json.loads(m["outcomes"]) if isinstance(m.get("outcomes"), str) else m.get("outcomes", ["Yes", "No"])
                prices = json.loads(m["outcomePrices"]) if isinstance(m.get("outcomePrices"), str) else m.get("outcomePrices")
            except Exception:
                continue
            yi = [o.lower() for o in outs].index("yes") if "yes" in [o.lower() for o in outs] else 0
            res = np.nan
            if m.get("closed") and prices:
                pv = float(prices[yi])
                res = 1.0 if pv > 0.99 else (0.0 if pv < 0.01 else np.nan)
            rows.append({"event": slug, "T": T, "strike": nums[0], "token": toks[yi], "resolved_api": res})
    return pd.DataFrame(rows)


def digital(S, K, var):
    sig = np.sqrt(np.maximum(var, 1e-12))
    return norm.cdf((np.log(S / K) - 0.5 * sig ** 2) / sig)


def implied_sigma(p, S, K):
    """Digital-Formel numerisch umkehren (Gitter), liefert sigma über die Restlaufzeit."""
    grid = np.exp(np.linspace(np.log(1e-4), np.log(0.3), 400))
    P = norm.cdf((np.log(S / K)[:, None] - 0.5 * grid[None, :] ** 2) / grid[None, :])
    return grid[np.argmin(np.abs(P - p[:, None]), axis=1)]


def build(mk, d, k1, dv, k, fn):
    parts = []
    for r in mk.itertuples():
        h = pd.DataFrame(d["hist"].get(r.token, {}).get("history", []))
        if h.empty:
            continue
        h["t"] = pd.to_datetime(h["t"], unit="s", utc=True)
        ps = h.drop_duplicates("t").set_index("t")["p"].astype(float).sort_index()
        ps.index = ps.index + pd.Timedelta("1min")                       # erst 1 Min. nach Zeitstempel bekannt
        lo = max(ps.index[0].ceil("1min"), r.T - pd.Timedelta(hours=24))
        if lo >= r.T:
            continue
        grid = pd.date_range(lo, r.T, freq="1min", inclusive="left")
        pm = ps.reindex(grid, method="ffill", tolerance=pd.Timedelta("15min")).values
        # Zeitpunkt der letzten echten Preisänderung (Polymarket meldet den Mittelwert aus Bid/Ask jede Minute,
        # auch wenn sich nichts bewegt; ohne Handel ist das z. B. 0,50 bei Bid 0,01 / Ask 0,99 -> nicht handelbar)
        chg = ps[ps.diff().fillna(1).abs() > 1e-9]
        last_chg = pd.Series(chg.index, index=chg.index).reindex(grid, method="ffill")
        stale = ((grid - pd.DatetimeIndex(last_chg)) / pd.Timedelta("1min")).values
        S = k1["close"].reindex(grid - pd.Timedelta("1min"), method="ffill").values
        if r.T not in k1.index:
            continue
        outcome = float(k1.at[r.T, "close"] > r.strike)                  # Kerze zur Auflösungsminute
        tau = ((r.T + pd.Timedelta("1min") - grid) / pd.Timedelta("1h")).values
        dvt = dv.reindex(grid, method="ffill", tolerance=pd.Timedelta("3h")).values / 100
        f = pd.DataFrame({"t": grid, "pm": pm, "stale_min": stale, "S": S, "tau": tau, "event": r.event, "strike": r.strike,
                          "token": r.token, "outcome": outcome, "resolved_api": r.resolved_api})
        f["fv_dvol"] = digital(S, r.strike, (dvt * k) ** 2 * tau / 8760)
        if fn is not None:
            orig = grid.floor("1h")
            e = ((grid - orig) / pd.Timedelta("1h")).values
            v = fn(orig, tau + e) - fn(orig, e)                          # Varianz von t bis Auflösung
            f["fv_model"] = digital(S, r.strike, v)
            f["var_model"] = v
        f["var_dvol"] = (dvt * k) ** 2 * tau / 8760
        parts.append(f.dropna(subset=["pm", "S"]))
    return pd.concat(parts, ignore_index=True)


def main(a):
    d = json.loads(open(a.dump).read())
    mk = parse(d)
    k1 = pd.read_parquet(C.DATA / "klines_1m.parquet", columns=["close"])
    mk = mk[mk["T"] <= k1.index.max()]
    print(f"{mk['event'].nunique()} Stunden-Events, {len(mk)} Stufen (mit Binance-Daten)")
    dvraw = pd.read_parquet(C.DATA / "dvol_1h.parquet")["dvol"]
    dv = dvraw.copy()
    dv.index = dv.index + pd.Timedelta(hours=1)
    k = vrp_factor(dvraw, pd.read_parquet(C.DATA / "bars_5m.parquet", columns=["r"]), mk["T"].min() - pd.Timedelta(days=2))
    vp = None
    for f in sorted(C.RESULTS.glob("vol_preds_*.parquet")):
        if "foundation" not in f.name and "kronos" not in f.name:
            x = pd.read_parquet(f)
            vp = x if vp is None else pd.concat([vp, x])
    fn, model = None, None
    if vp is not None:
        vp = vp.drop_duplicates(["origin", "h", "model"])
        need = pd.DatetimeIndex(mk["T"].dt.floor("1h").unique())
        for m in [a.model, "ens_lgbm_all", "ens_har_all", "lgbm", "har"]:
            if not m or m not in vp["model"].unique():
                continue
            cov = need.isin(vp.loc[vp["model"] == m, "origin"]).mean()   # deckt das Modell den Zeitraum ab?
            if cov > 0.8:
                fn, model = model_var_interp(vp, m), m
                break
            print(f"  {m}: Prognosen decken nur {cov:.0%} des Zeitraums ab -> nächstes Modell")
    print(f"Modell für fairen Wert: {model or 'keins (nur DVOL)'}")
    P = build(mk, d, k1, dv, k, fn)
    n_all = len(P)
    flat50 = (P["pm"] - 0.5).abs() < 0.006
    active = (P["stale_min"] <= a.max_stale) & ~(flat50 & (P["stale_min"] > 1))
    P = P[active].copy()
    # Plausibilität über die Stufen eines Events: P(S_T > K) muss mit K fallen. Isotone (fallende) Anpassung je
    # (Event, Minute); Preise, die > 0.15 davon abweichen, sind Schein-Mittelwerte (nur Marktdaten, kein Modell).
    from sklearn.isotonic import IsotonicRegression
    dev = np.zeros(len(P))
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    for _, g in P.groupby(["event", "t"], sort=False):
        if len(g) >= 3:
            o = np.argsort(g["strike"].values)
            fit = iso.fit_transform(g["strike"].values[o], g["pm"].values[o])
            dev[P.index.get_indexer(g.index[o])] = np.abs(g["pm"].values[o] - fit)
    n_act = len(P)
    P = P[dev <= 0.15].copy()
    print(f"Plausibilitätsprüfung über Stufen: {n_act - len(P):,} weitere Scheinpreise entfernt")
    print(f"Aktive Preise: {len(P):,} von {n_all:,} ({len(P) / n_all:.0%}); entfernt: eingefrorene (> {a.max_stale} Min. "
          f"unverändert) und Schein-Mittelwerte um 0,50")
    P["day"] = P["t"].dt.floor("1D")
    P.to_parquet(C.RESULTS / "pm_hourly_panel.parquet")
    agree = P.groupby("token")[["outcome", "resolved_api"]].first().dropna()
    fvs = [c for c in ("fv_dvol", "fv_model") if c in P]
    L = ["# Stündliche Polymarket-Märkte „Bitcoin above K at hh:00 ET“", "",
         f"{P['event'].nunique()} Events, {P['token'].nunique()} Stufen, {len(P):,} Minuten-Beobachtungen. "
         f"Modell: {model}. DVOL-Faktor k = {k:.3f}. Kosten {a.cost} je Anteil (Hin+Rück; bis Auflösung halb).",
         f"Nur **aktive, plausible** Preise: in den letzten {a.max_stale} Min. verändert, keine festen 0,50, monoton über "
         f"die Stufen (Abweichung ≤ 0,15) "
         f"({len(P):,} von {n_all:,} Minuten). Polymarket liefert Mittelwerte aus Bid/Ask – echte Ausführungspreise "
         f"liegen um den halben Spread schlechter (Kosten-Annahme) und sind ohne Orderbuch-Daten nicht prüfbar.",
         f"Ausgang (Binance-Rekonstruktion) stimmt bei {(agree.outcome == agree.resolved_api).mean():.1%} von {len(agree)} "
         f"Stufen mit der Polymarket-Auflösung überein.", ""]

    # a) Kalibrierung nach Restlaufzeit
    P["tau_bin"] = pd.cut(P["tau"] * 60, [0, 5, 15, 60, 360, 1440], labels=["<5 Min", "5-15", "15-60", "1-6h", "6-24h"])
    rows = []
    for tb, g in P.groupby("tau_bin", observed=True):
        row = {"Restlaufzeit": tb, "n": len(g), "Brier PM": ((g.pm - g.outcome) ** 2).mean()}
        for c in fvs:
            row[f"Brier {c}"] = ((g[c] - g.outcome) ** 2).mean()
        rows.append(row)
    L += ["## a) Wer sagt den Ausgang besser vorher? (Brier, niedriger = besser)", "",
          pd.DataFrame(rows).round(5).to_markdown(index=False), ""]
    P["pm_bin"] = pd.cut(P["pm"], [0, .05, .15, .3, .5, .7, .85, .95, 1], include_lowest=True)
    cal = P.groupby("pm_bin", observed=True).agg(n=("pm", "size"), PM=("pm", "mean"), eingetreten=("outcome", "mean"),
                                                 **{c: (c, "mean") for c in fvs})
    L += ["### Kalibrierung nach Polymarket-Preis", "", cal.round(3).to_markdown(), ""]

    # b) Lücken-Handel bis Auflösung
    L += ["## b) Lücke fairer Wert − Preis: Einstieg nach Reaktionszeit, Halten bis Auflösung", "",
          "Ereignis = Lücke überschreitet die Schwelle (neu, 30 Min. Pause je Stufe), nur Restlaufzeit ≥ 10 Min. "
          "t-Statistik über Tage (Stunden eines Tages sind nicht unabhängig).", ""]
    P = P.sort_values(["token", "t"]).reset_index(drop=True)
    for c in fvs:
        tab = []
        for th in a.th:
            ev_rows = []
            for tok, g in P.groupby("token", sort=False):
                gap = (g[c] - g["pm"]).values
                on = (np.abs(gap) > th) & (g["tau"].values * 60 >= 10)
                onset = np.flatnonzero(on & ~np.r_[False, on[:-1]])
                last = -10**9
                for i in onset:
                    if i - last < 30:
                        continue
                    last = i
                    for lag in LAGS:
                        j = i + lag
                        if j >= len(g):
                            continue
                        e = g["pm"].values[j]
                        if not (0.01 < e < 0.99):
                            continue
                        side = np.sign(gap[i])
                        ev_rows.append((lag, side * (g["outcome"].values[0] - e) - a.cost / 2, g["day"].values[0],
                                        np.where(side > 0, e, 1 - e), side * (g[c].values[j] - e) / abs(gap[i])))
            E = pd.DataFrame(ev_rows, columns=["lag", "pnl", "day", "cap", "rest"])
            for lag, gg in E.groupby("lag"):
                t, nd = week_tstat(gg["pnl"].values, gg["day"].values)
                tab.append({"Schwelle": th, "Reaktion (Min.)": lag, "Trades": len(gg), "Tage": nd,
                            "übrige Lücke": gg["rest"].median(), "PnL/Anteil": gg["pnl"].mean(),
                            "Rendite/$": gg["pnl"].sum() / gg["cap"].sum(), "Treffer": (gg["pnl"] > 0).mean(), "t(Tage)": t})
        L += [f"### {c}", "", pd.DataFrame(tab).round(4).to_markdown(index=False), ""]

    # c) Vola-Trade mit zwei Digitals
    L += ["## c) Vola-Trade: Strangle aus Digitals (Yes über K_oben + No über K_unten), Halten bis Auflösung", "",
          "Implizite Vola aus der nächsten Stufe über und unter dem Kurs. Prognose > implizit·(1+Schwelle) → Strangle kaufen, "
          "< implizit/(1+Schwelle) → verkaufen (Gegenposition). Zeitpunkte: jede volle 5. Minute, Restlaufzeit 10–360 Min.", ""]
    Q = P[(P["t"].dt.minute % 5 == 0) & (P["tau"] * 60 >= 10) & (P["tau"] <= 6) & (P["pm"] > 0.03) & (P["pm"] < 0.97)].copy()
    up = Q[Q["strike"] > Q["S"]].sort_values("strike").groupby(["event", "t"]).head(1).set_index(["event", "t"])
    dn = Q[Q["strike"] <= Q["S"]].sort_values("strike").groupby(["event", "t"]).tail(1).set_index(["event", "t"])
    W = up.join(dn, lsuffix="_u", rsuffix="_d", how="inner")
    if len(W) > 50:
        iv_u = implied_sigma(W["pm_u"].values, W["S_u"].values, W["strike_u"].values)
        iv_d = implied_sigma(W["pm_d"].values, W["S_d"].values, W["strike_d"].values)
        W["iv"] = np.sqrt((iv_u ** 2 + iv_d ** 2) / 2)
        W["rv_dvol"] = np.sqrt(W["var_dvol_u"])
        res = []
        srcs = [("DVOL", "rv_dvol")] + ([("Modell", "rv_model")] if "var_model_u" in W else [])
        if "var_model_u" in W:
            W["rv_model"] = np.sqrt(W["var_model_u"])
        # Strangle: Yes(K_oben) + No(K_unten) -> zahlt 1 wenn S_T > K_oben oder S_T < K_unten (genau eine Seite)
        pnl_long = (W["outcome_u"] - W["pm_u"]) + ((1 - W["outcome_d"]) - (1 - W["pm_d"])) - a.cost
        cap_long = W["pm_u"] + (1 - W["pm_d"])
        for name, col in srcs:
            ratio = W[col] / W["iv"]
            for th in a.vth:
                for lbl, mask, sgn, cap in (("kaufen", ratio > 1 + th, 1, cap_long),
                                            ("verkaufen", ratio < 1 / (1 + th), -1, 2 - cap_long)):
                    if mask.sum() < 30:
                        continue
                    pnl = sgn * (pnl_long[mask] + a.cost) - a.cost
                    t, nd = week_tstat(pnl.values, W.loc[mask, "day_u"].values)
                    res.append({"Quelle": name, "Schwelle": th, "Richtung": lbl, "Trades": int(mask.sum()), "Tage": nd,
                                "PnL/Strangle": pnl.mean(), "Rendite/$": pnl.sum() / cap[mask].sum(), "t(Tage)": t})
        for lbl, sgn, cap in (("immer kaufen", 1, cap_long), ("immer verkaufen", -1, 2 - cap_long)):
            pnl = sgn * (pnl_long + a.cost) - a.cost
            t, nd = week_tstat(pnl.values, W["day_u"].values)
            res.append({"Quelle": "Benchmark", "Schwelle": np.nan, "Richtung": lbl, "Trades": len(W), "Tage": nd,
                        "PnL/Strangle": pnl.mean(), "Rendite/$": pnl.sum() / cap.sum(), "t(Tage)": t})
        L += [f"Implizit (Median, sigma bis Auflösung, annualisiert): "
              f"{(W['iv'] / np.sqrt(W['tau_u'] / 8760)).median():.3f}; DVOL·k: {(W['rv_dvol'] / np.sqrt(W['tau_u'] / 8760)).median():.3f}", "",
              pd.DataFrame(res).round(4).to_markdown(index=False), ""]
    L += ["## Lesehilfe", "",
          "- a) Ist Polymarket kurz vor Auflösung schlechter als der faire Wert → dort liegt die Chance.",
          "- b) Bleibt der PnL bei 5–15 Min. Reaktion positiv (t > 2) → ohne Bot nutzbar; nur bei 0–1 Min. → Bot nötig.",
          "- c) Schlägt eine Quelle die Benchmarks → echte Vola-Information, die Polymarket nicht einpreist.", ""]
    out = C.RESULTS / "RESULTS_pm_hourly.md"
    out.write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="inputs/pm_hourly_dump.json")
    ap.add_argument("--cost", type=float, default=0.02)
    ap.add_argument("--th", type=float, nargs="+", default=[0.05, 0.10])
    ap.add_argument("--vth", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    ap.add_argument("--model", default="ens_lgbm_all")
    ap.add_argument("--max-stale", type=float, default=5, help="Preis gilt nur als aktiv, wenn in den letzten N Min. verändert")
    main(ap.parse_args())
