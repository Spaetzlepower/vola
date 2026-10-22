"""Schritt 6: Polymarket-Wochenmärkte "What price will Bitcoin hit ..." vs. Optionen (DVOL) vs. eigenes Modell.

  python s06_polymarket.py                    # lädt Daten (gecacht) und wertet aus
  python s06_polymarket.py --weeks-back 120   # weiter zurück suchen
  python s06_polymarket.py --cost 0.02        # angenommene Round-trip-Kosten (Spread) je Anteil

Was passiert:
  1. Polymarket: vergangene Wochen-Events finden (Gamma-API), je Stufe den stündlichen Preisverlauf
     der "Yes"-Seite laden (CLOB /prices-history).
  2. Deribit DVOL (30-Tage implizite Vola) stündlich laden.
  3. Für jede Stunde und Stufe den fairen Wert berechnen (Touch-Wahrscheinlichkeit, Brownsche Bewegung):
       fv_dvol      DVOL direkt
       fv_dvol_adj  DVOL x k  (k = historisches Verhältnis realisierte/implizite Vola, Risikoprämie raus)
       fv_har/lgbm  eigene Vola-Prognosen aus s03 (falls vorhanden)
     Bereits berührte Stufen und abgelaufene Zeit werden korrekt berücksichtigt (aktueller Kurs, Restlaufzeit).
  4. Auswertung:
       a) Wer sagt den Ausgang besser vorher? (Brier, Kalibrierung, Longshot-Bias)
       b) Kurzfrist-Handel: Signal = fairer Wert - Polymarket-Preis. Bewegt sich der Preis in den
          nächsten 1/4/24 h zum fairen Wert? Backtest mit Kostenabschlag, t-Statistik über Wochen.

Ergebnis: results/RESULTS_polymarket.md, results/pm_panel.parquet, Rohdaten-Cache in data/pm/
"""
import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.stats import norm

import config as C

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
DERIBIT = "https://www.deribit.com/api/v2/public"
CACHE = C.DATA / "pm"
CACHE.mkdir(parents=True, exist_ok=True)
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]
UA = {"User-Agent": "Mozilla/5.0 (research script)", "Accept": "application/json"}


# ---------------- HTTP mit Cache ----------------

OFFLINE = False      # --offline: nur Cache benutzen (z. B. nach --import aus dem Browser)


def http_json(url, retries=4):
    if OFFLINE and "deribit" not in url:
        return None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                return None
            if e.code == 429:
                time.sleep(5 * (i + 1))
                continue
            err = e
        except Exception as e:
            err = e
        time.sleep(2 ** i)
    print(f"  WARN: {url} -> {err}")
    return None


def import_dump(path):
    """Browser-Export (pm_dump.json) in den Cache schreiben."""
    d = json.loads(open(path).read())
    fid = d.get("fidelity", 60)
    for slug, ev in d.get("events", {}).items():
        (CACHE / f"event_{slug}.json").write_text(json.dumps(ev))
    for tok, h in d.get("hist", {}).items():
        (CACHE / f"hist_{tok[-16:]}_{fid}.json").write_text(json.dumps(h))
    print(f"Import: {len(d.get('events', {}))} Events, {len(d.get('hist', {}))} Preisverläufe -> {CACHE}")


def cached(name, fn):
    p = CACHE / f"{name}.json"
    if p.exists():
        return json.loads(p.read_text())
    data = fn()
    if data is not None:
        p.write_text(json.dumps(data))
    return data


# ---------------- 1. Polymarket ----------------

def candidate_slugs(monday):
    """Slug-Varianten für die Woche Montag..Sonntag (Format variiert je nach Zeitraum)."""
    sun = monday + timedelta(days=6)
    m1, m2 = MONTHS[monday.month - 1], MONTHS[sun.month - 1]
    base = "what-price-will-bitcoin-hit"
    s = []
    for y in (f"-{sun.year}", ""):
        if m1 == m2:
            s.append(f"{base}-{m1}-{monday.day}-{sun.day}{y}")
        else:
            s.append(f"{base}-{m1}-{monday.day}-{m2}-{sun.day}{y}")
            s.append(f"{base}-{m1}-{monday.day}-{sun.day}{y}")
    return s


def find_events(weeks_back):
    events = {}
    today = pd.Timestamp.now(tz="UTC").normalize()
    monday0 = today - pd.Timedelta(days=today.day_of_week)
    for k in range(weeks_back):
        mon = (monday0 - pd.Timedelta(weeks=k)).date()
        for slug in candidate_slugs(mon):
            ev = cached(f"event_{slug}", lambda s=slug: http_json(f"{GAMMA}/events?slug={s}") or None)
            if ev:
                ev = ev[0] if isinstance(ev, list) else ev
                events[ev["slug"]] = ev
                break
    # Zusätzlich Volltextsuche (findet Events mit abweichendem Slug)
    for page in range(1, 6 if not OFFLINE else 1):
        q = urllib.parse.quote("What price will Bitcoin hit")
        res = cached(f"search_{page}", lambda p=page: http_json(
            f"{GAMMA}/public-search?q={q}&events_status=closed&limit_per_type=50&page={p}"))
        if not res:
            break
        found = 0
        for ev in res.get("events", []) or []:
            if "what-price-will-bitcoin-hit" in ev.get("slug", "") and ev["slug"] not in events:
                full = cached(f"event_{ev['slug']}", lambda s=ev["slug"]: http_json(f"{GAMMA}/events?slug={s}"))
                if full:
                    events[ev["slug"]] = full[0] if isinstance(full, list) else full
                    found += 1
        if not res.get("events"):
            break
    # nur Wochen-Events (Dauer 5-8 Tage), keine Tages-/Monatsmärkte
    out = {}
    for s, ev in events.items():
        st, en = pd.Timestamp(ev.get("startDate")), pd.Timestamp(ev.get("endDate"))
        if pd.notna(st) and pd.notna(en) and 4 <= (en - st).days <= 8:
            out[s] = ev
    print(f"Polymarket: {len(out)} Wochen-Events gefunden")
    return out


def parse_markets(events):
    rows = []
    for slug, ev in events.items():
        for m in ev.get("markets", []):
            q = m.get("question", "")
            title = m.get("groupItemTitle") or ""
            txt = f"{title} {q}"
            nums = re.findall(r"\$?\s?(\d{1,3}(?:,\d{3})+|\d{4,7})(?:\.\d+)?\s?(k|K)?", txt)
            if not nums:
                continue
            n, kk = nums[0]
            strike = float(n.replace(",", "")) * (1000 if kk else 1)
            low = txt.lower()
            if "↓" in txt or re.search(r"\b(dip|drop|fall|below|under|low)\b", low):
                d = "dn"
            elif "↑" in txt or re.search(r"\b(reach|hit|above|over|high)\b", low):
                d = "up"
            else:
                continue
            try:
                tok = json.loads(m["clobTokenIds"]) if isinstance(m.get("clobTokenIds"), str) else m.get("clobTokenIds")
                outcomes = json.loads(m["outcomes"]) if isinstance(m.get("outcomes"), str) else m.get("outcomes", ["Yes", "No"])
                prices = json.loads(m["outcomePrices"]) if isinstance(m.get("outcomePrices"), str) else m.get("outcomePrices")
            except Exception:
                continue
            if not tok:
                continue
            yi = [o.lower() for o in outcomes].index("yes") if "yes" in [o.lower() for o in outcomes] else 0
            res = np.nan
            if m.get("closed") and prices:
                pv = float(prices[yi])
                res = 1.0 if pv > 0.99 else (0.0 if pv < 0.01 else np.nan)
            rows.append({"event": slug, "market_id": m.get("id"), "question": q, "strike": strike, "dir": d,
                         "yes_token": tok[yi], "start": pd.Timestamp(ev["startDate"]).tz_convert("UTC")
                         if pd.Timestamp(ev["startDate"]).tzinfo else pd.Timestamp(ev["startDate"], tz="UTC"),
                         "end": pd.Timestamp(m.get("endDate") or ev["endDate"]).tz_convert("UTC")
                         if pd.Timestamp(m.get("endDate") or ev["endDate"]).tzinfo
                         else pd.Timestamp(m.get("endDate") or ev["endDate"], tz="UTC"),
                         "resolved_api": res, "volume": float(m.get("volume") or 0)})
    df = pd.DataFrame(rows)
    print(f"Polymarket: {len(df)} Stufen (Märkte) geparst")
    return df


def price_history(token, start, end, fidelity):
    s, e = int(start.timestamp()) - 86400, int(end.timestamp()) + 3600

    def get():
        r = http_json(f"{CLOB}/prices-history?market={token}&startTs={s}&endTs={e}&fidelity={fidelity}")
        if r and r.get("history"):
            return r
        r = http_json(f"{CLOB}/prices-history?market={token}&interval=max&fidelity={fidelity}")
        if r and r.get("history"):
            return r
        hist = []                                       # Fallback: tageweise abfragen
        for d0 in range(s, e, 86400):
            r = http_json(f"{CLOB}/prices-history?market={token}&startTs={d0}&endTs={min(d0 + 86400, e)}&fidelity={fidelity}")
            if r and r.get("history"):
                hist += r["history"]
        return {"history": hist}

    r = cached(f"hist_{token[-16:]}_{fidelity}", get)
    h = pd.DataFrame((r or {}).get("history", []))
    if h.empty:
        return pd.Series(dtype=float)
    h["t"] = pd.to_datetime(h["t"], unit="s", utc=True)
    return h.drop_duplicates("t").set_index("t")["p"].astype(float).sort_index()


# ---------------- 2. DVOL ----------------

def dvol(start, end):
    p = C.DATA / "dvol_1h.parquet"
    if p.exists() and OFFLINE:
        return pd.read_parquet(p)["dvol"]
    if p.exists():
        d = pd.read_parquet(p)
        if d.index.min() <= start + pd.Timedelta("400D") and d.index.max() >= end - pd.Timedelta("2D"):
            return d["dvol"]
    rows = []
    t = start
    while t < end:
        t2 = min(t + pd.Timedelta(days=30), end)
        r = http_json(f"{DERIBIT}/get_volatility_index_data?currency=BTC&resolution=3600"
                      f"&start_timestamp={int(t.timestamp() * 1000)}&end_timestamp={int(t2.timestamp() * 1000)}")
        if r and r.get("result", {}).get("data"):
            rows += r["result"]["data"]
        t = t2
    if not rows and p.exists():                       # keine Verbindung -> vorhandenen Cache nutzen
        print("  DVOL: keine neuen Daten abrufbar, nutze vorhandenen Cache")
        return pd.read_parquet(p)["dvol"]
    d = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"])
    d["time"] = pd.to_datetime(d["ts"], unit="ms", utc=True)
    d = d.drop_duplicates("time").set_index("time").sort_index()[["close"]].rename(columns={"close": "dvol"})
    d.to_parquet(p)
    print(f"DVOL: {len(d):,} Stundenwerte {d.index.min()} bis {d.index.max()}")
    return d["dvol"]


def vrp_factor(dv, bars, before):
    """k = Median(realisierte 30T-Vola danach / DVOL), nur Daten vor `before` (keine Leckage)."""
    r2 = (bars["r"] ** 2).resample("1h").sum()
    fwd = r2[::-1].rolling(720, min_periods=700).sum()[::-1].shift(-1)       # nächste 30 Tage
    rv = np.sqrt(fwd * 365 / 30) * 100
    j = pd.concat([dv, rv.rename("rv")], axis=1).dropna()
    j = j[j.index < before - pd.Timedelta(days=31)]
    if len(j) < 1000:
        print("  Zu wenig Historie für VRP-Faktor -> k = 0.85 (Annahme)")
        return 0.85
    k = float((j["rv"] / j["dvol"]).median())
    print(f"VRP-Faktor k = {k:.3f} (realisierte/implizite Vola, Median vor {before.date()}, n={len(j):,})")
    return k


# ---------------- 3. Faire Werte ----------------

def touch_prob(sigma, b):
    return np.clip(2 * norm.sf(b / np.maximum(sigma, 1e-9)), 0, 1)


def model_var_interp(vp, model):
    """vol_preds (origin, h, var) -> Funktion (origins, T_hours) -> kumulierte Varianz bis T."""
    g = vp[vp.model == model].pivot_table(index="origin", columns="h", values="var").sort_index()
    if g.empty:
        return None
    hs = np.array([0] + list(g.columns), float)

    def f(origins, T):
        G = g.reindex(origins)
        out = np.full(len(origins), np.nan)
        V = np.c_[np.zeros(len(G)), G.values]
        for i in range(len(origins)):
            if np.isnan(V[i]).any() or np.isnan(T[i]):
                continue
            if T[i] <= hs[-1]:
                out[i] = np.interp(T[i], hs, V[i])
            else:
                out[i] = V[i, -1] / hs[-1] * T[i]
        return out
    return f


def build_panel(mk, fidelity, k):
    k1 = pd.read_parquet(C.DATA / "klines_1m.parquet", columns=["high", "low", "close"])
    last_data = k1.index.max()
    mk = mk[mk["end"] <= last_data].copy()          # nur abgeschlossene Wochen mit Kursdaten
    if mk.empty:
        raise SystemExit("Keine abgeschlossenen Polymarket-Wochen im Zeitraum der Binance-Daten.")
    dv = dvol(mk["start"].min() - pd.Timedelta(days=400), mk["end"].max() + pd.Timedelta(hours=2))
    vp = None
    for f in sorted(C.RESULTS.glob("vol_preds_*.parquet")):
        if "foundation" not in f.name and "kronos" not in f.name:
            vp = pd.concat([vp, pd.read_parquet(f)]) if vp is not None else pd.read_parquet(f)
    interps = {}
    if vp is not None:
        vp = vp.drop_duplicates(["origin", "h", "model"])
        ens = sorted(m for m in vp["model"].unique() if m.startswith("ens_") and m != "ens_har")
        for m in ["har", "lgbm"] + ens:
            fn = model_var_interp(vp, m)
            if fn:
                interps[m] = fn
    print(f"Eigene Modelle verfügbar: {list(interps) or 'keine (erst s03_models.py laufen lassen)'}")

    parts = []
    for i, r in enumerate(mk.itertuples(), 1):
        ph = price_history(r.yes_token, r.start, r.end, fidelity)
        if ph.empty:
            continue
        # Stundenraster: Preis zum Zeitpunkt t = letzter Preis VOR/AN t (keine Zukunftsinformation)
        grid = pd.date_range(r.start.ceil("1h"), r.end, freq="1h", inclusive="left")
        ph = ph[ph.index < r.end].reindex(grid, method="ffill", tolerance=pd.Timedelta("2h")).dropna()
        if ph.empty:
            continue
        w = k1[(k1.index >= r.start) & (k1.index < r.end)]
        if w.empty:
            continue
        # Ausgang nach Binance-1m-Daten (Polymarket-Auflösungsquelle)
        hit_path = (w["high"] >= r.strike) if r.dir == "up" else (w["low"] <= r.strike)
        outcome = float(hit_path.any())
        first_hit = hit_path.idxmax() if hit_path.any() else pd.Timestamp.max.tz_localize("UTC")
        t = ph.index
        s_t = k1["close"].reindex(t - pd.Timedelta("1min"), method="ffill").values   # letzter Kurs vor t
        T = (r.end - t) / pd.Timedelta("1h")
        b = np.log(r.strike / s_t) if r.dir == "up" else np.log(s_t / r.strike)
        df = pd.DataFrame({"t": t, "first_hit": first_hit, "end": r.end, "event": r.event, "market_id": r.market_id, "dir": r.dir, "strike": r.strike,
                           "pm": ph.values, "S": s_t, "T_h": T, "b": b, "outcome": outcome,
                           "resolved_api": r.resolved_api, "volume": r.volume})
        df = df[(df["t"] < first_hit) & (df["b"] > 0) & (df["T_h"] > 0)]   # noch offen, noch nicht berührt
        if df.empty:
            continue
        dvt = dv.reindex(df["t"], method="ffill", tolerance=pd.Timedelta("3h")).values / 100
        sig_raw = dvt * np.sqrt(df["T_h"].values / (365 * 24))
        df["fv_dvol"] = touch_prob(sig_raw, df["b"].values)
        df["fv_dvol_adj"] = touch_prob(sig_raw * k, df["b"].values)
        for m, fn in interps.items():
            # Modellprognose vom letzten vollen Stundenzeitpunkt <= t
            orig = df["t"].dt.floor("1h")
            var = fn(orig, df["T_h"].values)
            df[f"fv_{m}"] = touch_prob(np.sqrt(var), df["b"].values)
        parts.append(df)
        if i % 50 == 0:
            print(f"  {i}/{len(mk)} Stufen verarbeitet")
    panel = pd.concat(parts, ignore_index=True)
    agree = panel.groupby("market_id")[["outcome", "resolved_api"]].first().dropna()
    if len(agree):
        print(f"Ausgang Binance-Rekonstruktion vs. Polymarket-API: {(agree.outcome == agree.resolved_api).mean():.1%} "
              f"Übereinstimmung ({len(agree)} Märkte)")
    return panel


# ---------------- 4. Auswertung ----------------

def week_tstat(x, groups):
    m = pd.Series(x).groupby(np.asarray(groups)).mean()
    if len(m) < 3 or m.std() == 0:
        return np.nan, len(m)
    return m.mean() / (m.std(ddof=1) / np.sqrt(len(m))), len(m)


def future_price(P, H):
    """Preis H Stunden später. Wurde die Stufe zwischenzeitlich berührt -> 1 (Yes sicher);
    ist der Markt vorher ausgelaufen -> Ausgang."""
    tH = P["t"] + pd.Timedelta(hours=H)
    look = P.set_index(["market_id", "t"])["pm"]
    fut = pd.Series(look.reindex(pd.MultiIndex.from_arrays([P["market_id"], tH])).values, index=P.index)
    touched = P["first_hit"] <= tH
    expired = P["end"] <= tH
    fut = fut.where(~touched, 1.0)
    fut = fut.where(touched | ~expired, P["outcome"])
    return fut


def analyse(panel, cost, out):
    fvs = [c for c in panel.columns if c.startswith("fv_")]
    P = panel.dropna(subset=fvs).copy()
    L = [f"# Polymarket vs. Optionen (DVOL) vs. eigenes Modell", "",
         f"{P['event'].nunique()} Wochen, {P['market_id'].nunique()} Stufen, {len(P):,} Stunden-Beobachtungen "
         f"(offene, noch nicht berührte Stufen). Angenommene Round-trip-Kosten: {cost:.3f} je Anteil.", "",
         "Hinweis: Polymarket-Preise sind der stündliche Kursverlauf (kein Bid/Ask). "
         "Für liquide Stufen eine brauchbare Näherung; die Kostenannahme deckt den Spread ab.", ""]

    # a) Vorhersagegüte für den Ausgang
    rows = []
    for c in ["pm"] + fvs:
        p = P[c].clip(1e-4, 1 - 1e-4)
        rows.append({"Quelle": c, "Brier": ((p - P.outcome) ** 2).mean(),
                     "LogLoss": -(P.outcome * np.log(p) + (1 - P.outcome) * np.log(1 - p)).mean(),
                     "mittl. Prognose": p.mean(), "Trefferquote": P.outcome.mean()})
    L += ["## a) Wer sagt den Ausgang besser vorher? (niedriger = besser)", "",
          pd.DataFrame(rows).round(4).to_markdown(index=False), ""]

    # Kalibrierung / Longshot-Bias nach Polymarket-Preisbereich
    bins = [0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 1.0]
    P["pm_bin"] = pd.cut(P["pm"], bins, include_lowest=True)
    cal = P.groupby("pm_bin", observed=True).agg(n=("pm", "size"), pm=("pm", "mean"), eingetreten=("outcome", "mean"),
                                                 **{c: (c, "mean") for c in fvs})
    L += ["### Kalibrierung nach Polymarket-Preis (pm zu hoch vs. 'eingetreten' = überteuert)", "",
          cal.round(3).to_markdown(), ""]
    P["T_bin"] = pd.cut(P["T_h"], [0, 6, 24, 72, 1000], labels=["<6h", "6-24h", "1-3d", ">3d"])
    gap = P.groupby(["T_bin", "dir"], observed=True).agg(
        n=("pm", "size"), pm=("pm", "mean"), eingetreten=("outcome", "mean"), **{c: (c, "mean") for c in fvs})
    L += ["### Nach Restlaufzeit und Richtung", "", gap.round(3).to_markdown(), ""]

    # b) Kurzfrist-Handel
    P = P.sort_values(["market_id", "t"])
    L += ["## b) Kurzfrist-Handel: Bewegt sich der Polymarket-Preis zum fairen Wert?", "",
          "Signal = fairer Wert − Preis. |Signal| > Schwelle → kaufen (Yes bei positiv, No bei negativ), "
          "nach H Stunden schließen. PnL je Anteil nach Kosten. t-Stat über Wochen-Mittelwerte "
          "(berücksichtigt, dass Stunden derselben Woche nicht unabhängig sind). "
          "'bis Ende' = halten bis Auflösung.", ""]
    res = []
    for H in (1, 4, 24):
        fut = future_price(P, H)
        for c in fvs:
            g = P[c] - P["pm"]
            for th in (0.03, 0.05, 0.10):
                sel = (g.abs() > th) & fut.notna()
                if sel.sum() < 30:
                    continue
                sgn = np.sign(g[sel])
                pnl = sgn * (fut[sel] - P.loc[sel, "pm"]) - cost
                tstat, nw = week_tstat(pnl.values, P.loc[sel, "event"].values)
                res.append({"H": f"{H}h", "Quelle": c, "Schwelle": th, "Trades": int(sel.sum()), "Wochen": nw,
                            "PnL/Anteil": pnl.mean(), "Trefferquote": (pnl > 0).mean(), "t(Wochen)": tstat})
    for c in fvs:
        g = P[c] - P["pm"]
        for th in (0.03, 0.05, 0.10):
            sel = g.abs() > th
            if sel.sum() < 30:
                continue
            sgn = np.sign(g[sel])
            pnl = sgn * (P.loc[sel, "outcome"] - P.loc[sel, "pm"]) - cost / 2
            tstat, nw = week_tstat(pnl.values, P.loc[sel, "event"].values)
            res.append({"H": "bis Ende", "Quelle": c, "Schwelle": th, "Trades": int(sel.sum()), "Wochen": nw,
                        "PnL/Anteil": pnl.mean(), "Trefferquote": (pnl > 0).mean(), "t(Wochen)": tstat})
    R = pd.DataFrame(res)
    if not R.empty:
        L += [R.round(4).to_markdown(index=False), ""]
    # Regression: Preisänderung auf Lücke (wie viel der Lücke schließt sich?)
    reg = []
    for H in (1, 4, 24):
        fut = future_price(P, H)
        ok = fut.notna()
        for c in fvs:
            x = (P[c] - P["pm"])[ok]
            y = (fut - P["pm"])[ok]
            if len(x) > 30 and x.std() > 0:
                beta = np.polyfit(x, y, 1)[0]
                reg.append({"H": f"{H}h", "Quelle": c, "Anteil der Lücke, der sich schließt (beta)": beta, "n": len(x)})
    if reg:
        L += ["### Wie viel der Lücke schließt sich? (beta > 0: Preis läuft Richtung fairer Wert)", "",
              pd.DataFrame(reg).round(4).to_markdown(index=False), ""]
    L += ["## Lesehilfe", "",
          "- **a)** Hat `pm` den niedrigsten Brier, ist der Markt schwer zu schlagen. Ist ein fv_* besser, "
          "steckt dort Information, die der Markt nicht voll einpreist.",
          "- **Kalibrierung:** Liegt `pm` in den Randbereichen systematisch über 'eingetreten', sind Longshots "
          "überteuert (Verkaufen bzw. No kaufen wäre die Gegenposition).",
          "- **b)** Nur Strategien mit positivem PnL **nach Kosten** und |t(Wochen)| > 2 über viele Wochen sind "
          "ernst zu nehmen. Viele Kombinationen getestet → einzelne Treffer können Zufall sein.", ""]
    out.write_text("\n".join(L))
    print(f"Report: {out}")
    if not R.empty:
        print(R.sort_values("PnL/Anteil", ascending=False).head(12).round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks-back", type=int, default=80)
    ap.add_argument("--fidelity", type=int, default=60, help="Auflösung der Polymarket-Preise in Minuten")
    ap.add_argument("--cost", type=float, default=0.02, help="Round-trip-Kosten je Anteil (Spread + Gebühren)")
    ap.add_argument("--import", dest="imp", help="pm_dump.json aus dem Browser-Skript importieren")
    ap.add_argument("--offline", action="store_true", help="keine Polymarket-Abfragen, nur Cache")
    a = ap.parse_args()
    if a.imp:
        import_dump(a.imp)
        OFFLINE = True
    OFFLINE = OFFLINE or a.offline
    events = find_events(a.weeks_back)
    if not events:
        raise SystemExit("Keine Events gefunden. Prüfe Internetzugang zu gamma-api.polymarket.com "
                         "und schick mir die Ausgabe.")
    mk = parse_markets(events)
    mk.to_csv(C.RESULTS / "pm_markets.csv", index=False)
    bars = pd.read_parquet(C.DATA / "bars_5m.parquet", columns=["r"])
    dv_hist = dvol(pd.Timestamp("2021-04-01", tz="UTC"), mk["end"].max())
    k = vrp_factor(dv_hist, bars, mk["start"].min())
    panel = build_panel(mk, a.fidelity, k)
    panel.to_parquet(C.RESULTS / "pm_panel.parquet")
    analyse(panel, a.cost, C.RESULTS / "RESULTS_polymarket.md")
