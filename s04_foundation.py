"""Schritt 4: Zero-shot Foundation Models (GPU empfohlen).

  python s04_foundation.py --mode valid --models chronos2 timestral kronos
  python s04_foundation.py --mode test  --models chronos2 timestral kronos --every 6

Ansätze:
  chronos2 / timestral   prognostizieren die stündliche, saisonbereinigte log-Realized-Variance
                         (168 Stunden voraus, Quantile) -> Varianz je Horizont -> Touch per BM-Formel
  kronos                 simuliert ganze Kerzenpfade (5m-Bars für 1h/4h/24h, 1h-Bars für 72h/168h)
                         -> Touch-Wahrscheinlichkeit = Anteil der Pfade, die die Schwelle erreichen

Damit es bezahlbar bleibt, wird nur jede `--every`-te Stunde prognostiziert. Verglichen wird auf
genau denselben Zeitpunkten mit den Ergebnissen aus s03_models.py (vorher laufen lassen).
"""
import argparse
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

import config as C
from evaluate import barriers, reliability, touch_long, touch_table, vol_table, write_report

H = C.BARS_PER_HOUR
HMAX = max(C.HORIZONS_H)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- Daten ----------------

def hourly_series():
    b = pd.read_parquet(C.DATA / "bars_5m.parquet")
    seas = pd.read_parquet(C.DATA / "seasonality.parquet")["seas"].values
    bucket = (b.index.dayofweek * 288 + b.index.hour * 12 + b.index.minute // 5).values
    b["seas"] = seas[bucket]
    b["r2"] = b["r"] ** 2
    hr = b.resample("1h").agg({"r2": "sum", "seas": "mean"})
    hr["lrv"] = np.log(hr["r2"] / (H * hr["seas"]) + 1e-10)        # saisonbereinigt, pro Bar
    # Saisonniveau für ALLE Stunden (auch in der Zukunft bekannt, rein kalenderbasiert)
    return b, hr


def seas_hour_future(origin, n):
    seas = pd.read_parquet(C.DATA / "seasonality.parquet")["seas"].values
    t = pd.date_range(origin, periods=n * H, freq="5min")
    v = seas[(t.dayofweek * 288 + t.hour * 12 + t.minute // 5).values]
    return v.reshape(n, H).mean(1)


def quantiles_to_var(q, levels, origins):
    """q: (B, HMAX, Q) Quantile der log-Stundenvarianz -> dict h -> Varianz über h Stunden."""
    levels = np.asarray(levels)
    med = q[..., np.argmin(abs(levels - 0.5))]
    s = (q[..., np.argmin(abs(levels - 0.9))] - q[..., np.argmin(abs(levels - 0.1))]) / 2.5631
    ev = np.exp(med + 0.5 * s ** 2)                                  # E[RV] je Stunde (Lognormal)
    seasf = np.stack([seas_hour_future(o, HMAX) for o in origins])
    per_hour = ev * H * seasf
    cum = per_hour.cumsum(1)
    return {h: cum[:, h - 1] for h in C.HORIZONS_H}


def contexts(hr, origins, ctx):
    lrv = hr["lrv"]
    pos = lrv.index.get_indexer(origins)          # Stunde, die bei origin beginnt -> Kontext endet davor
    X = np.full((len(origins), ctx), np.nan, np.float32)
    for i, p in enumerate(pos):
        lo = max(0, p - ctx)
        seg = lrv.values[lo:p]
        X[i, ctx - len(seg):] = seg
    return X


# ---------------- Modelle ----------------

def run_timestral(hr, origins, size, ctx, bs, dry):
    from timestral import TimestralForecaster
    if dry:
        sys.path.insert(0, str(C.EXT / "timestral" / "tests"))
        from test_forecaster import tiny_forecaster
        fc = tiny_forecaster()
    else:
        fc = TimestralForecaster.from_pretrained(f"mistralai/Timestral-{size}", device=DEV)
    X = contexts(hr, origins, ctx)
    out = []
    for i in range(0, len(X), bs):
        f = fc.forecast(X[i:i + bs], prediction_length=HMAX)
        out.append(f.quantiles[:, 0].float().cpu().numpy())
        print(f"  timestral {min(i + bs, len(X))}/{len(X)}", end="\r")
    print()
    return quantiles_to_var(np.concatenate(out), fc.quantile_levels, origins)


def run_chronos2(hr, origins, ctx, bs, dry):
    from chronos import Chronos2Pipeline
    levels = [0.1, 0.5, 0.9]
    if dry:
        pipe = Chronos2Pipeline.from_pretrained(str(C.EXT / "chronos2_tiny"), device_map="cpu")
    else:
        pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=DEV)
    X = contexts(hr, origins, ctx)
    out = []
    for i in range(0, len(X), bs):
        inp = [torch.tensor(x[~np.isnan(x)]) for x in X[i:i + bs]]
        q, _mean = pipe.predict_quantiles(inp, prediction_length=HMAX, quantile_levels=levels)
        q = torch.stack([t[0] if t.dim() == 3 else t for t in q]) if isinstance(q, list) else q
        out.append(q.float().cpu().numpy())
        print(f"  chronos2 {min(i + bs, len(X))}/{len(X)}", end="\r")
    print()
    q = np.concatenate(out)
    if q.ndim == 4:          # (B, 1, T, Q)
        q = q[:, 0]
    return quantiles_to_var(q, levels, origins)


def load_kronos(size, dry):
    repo = C.EXT / "Kronos"
    if not repo.exists():
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/shiyu-coder/Kronos", str(repo)], check=True)
    sys.path.insert(0, str(repo))
    from model import Kronos, KronosPredictor, KronosTokenizer
    if dry:
        tok = KronosTokenizer(d_in=6, d_model=32, n_heads=2, ff_dim=64, n_enc_layers=1, n_dec_layers=1,
                              ffn_dropout_p=0, attn_dropout_p=0, resid_dropout_p=0, s1_bits=4, s2_bits=4,
                              beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05, group_size=4)
        mod = Kronos(s1_bits=4, s2_bits=4, n_layers=1, d_model=32, n_heads=2, ff_dim=64, ffn_dropout_p=0,
                     attn_dropout_p=0, resid_dropout_p=0, token_dropout_p=0, learn_te=True)
        ctx = 512
    else:
        tname = "NeoQuasar/Kronos-Tokenizer-2k" if size == "mini" else "NeoQuasar/Kronos-Tokenizer-base"
        tok = KronosTokenizer.from_pretrained(tname)
        mod = Kronos.from_pretrained(f"NeoQuasar/Kronos-{size}")
        ctx = 2048 if size == "mini" else 512
    return KronosPredictor(mod.eval(), tok.eval(), device=DEV, max_context=ctx), ctx


def run_kronos(bars5, origins, size, n_paths, bs, dry):
    """Gibt (vol dict h->var, touch DataFrame) zurück."""
    pred, ctx = load_kronos(size, dry)
    ohlc = ["open", "high", "low", "close", "volume", "quote_volume"]
    bars1h = bars5[ohlc].resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                              "volume": "sum", "quote_volume": "sum"}).dropna()
    groups = [(bars5, "5min", [h for h in C.HORIZONS_H if h <= 24]),
              (bars1h, "1h", [h for h in C.HORIZONS_H if h > 24])]
    vol = {h: np.full(len(origins), np.nan) for h in C.HORIZONS_H}
    touch = []
    bu, bd = barriers(C.THRESHOLDS)
    for bars, freq, hs in groups:
        if not hs:
            continue
        step = pd.Timedelta(freq)
        per_h = int(pd.Timedelta("1h") / step)
        pred_len = max(hs) * per_h
        lookback = min(ctx, 512) if freq == "5min" else min(ctx, 512)
        pos = bars.index.get_indexer(origins)
        for i in range(0, len(origins), bs):
            idx = list(range(i, min(i + bs, len(origins))))
            dfs, xts, yts = [], [], []
            for j in idx:
                p = pos[j]
                x = bars.iloc[p - lookback:p][ohlc].rename(columns={"quote_volume": "amount"})
                xt = pd.Series(x.index)
                yt = pd.Series(pd.date_range(origins[j], periods=pred_len, freq=freq))
                for _ in range(n_paths):          # jede Kopie = ein unabhängig gesampelter Pfad
                    dfs.append(x.reset_index(drop=True)); xts.append(xt); yts.append(yt)
            t0 = time.time()
            outs = pred.predict_batch(dfs, xts, yts, pred_len=pred_len, T=1.0, top_p=0.9, sample_count=1, verbose=False)
            for k, j in enumerate(idx):
                paths = outs[k * n_paths:(k + 1) * n_paths]
                s0 = bars["close"].iloc[pos[j] - 1]
                hi = np.stack([np.maximum(pp["high"].values, pp[["open", "close"]].max(1).values) for pp in paths])
                lo = np.stack([np.minimum(pp["low"].values, pp[["open", "close"]].min(1).values) for pp in paths])
                cl = np.stack([pp["close"].values for pp in paths])
                lr = np.diff(np.log(np.c_[np.full(n_paths, s0), np.maximum(cl, 1e-9)]), axis=1)
                for h in hs:
                    n = h * per_h
                    vol[h][j] = (lr[:, :n] ** 2).sum(1).mean()
                    up = np.log(hi[:, :n].max(1) / s0)
                    dn = np.log(lo[:, :n].min(1) / s0)
                    for thr, b_up, b_dn in zip(C.THRESHOLDS, bu, bd):
                        for d, hits in (("up", up >= b_up), ("dn", dn <= -b_dn)):
                            touch.append((origins[j], h, d, thr, (hits.sum() + 0.5) / (n_paths + 1)))
            print(f"  kronos {freq}: {idx[-1] + 1}/{len(origins)}  ({time.time() - t0:.1f}s/Batch)", end="\r")
        print()
    return vol, pd.DataFrame(touch, columns=["origin", "h", "dir", "thr", "p"])


# ---------------- Ablauf ----------------

def main(a):
    ds = pd.read_parquet(C.DATA / "dataset.parquet")
    start, end = (C.VALID_START, C.VALID_END) if a.mode == "valid" else (C.TEST_START, str(ds.index[-1].date()))
    sel = ds[(ds.index >= pd.Timestamp(start, tz="UTC")) & (ds.index < pd.Timestamp(end, tz="UTC") + pd.Timedelta("1D"))]
    sel = sel[(sel.index.hour % a.every == 0)]
    sel = sel[sel[f"y_rv_{HMAX}"].notna() & (sel[f"y_miss_{HMAX}"] < 0.05)]
    if a.limit:
        sel = sel.iloc[:a.limit]
    origins = sel.index
    print(f"{len(origins)} Prognosezeitpunkte ({a.mode}), Gerät: {DEV}")
    bars5, hr = hourly_series()

    vol_rows, touch_rows = [], []

    def add_var(name, vdict):
        for h, v in vdict.items():
            ok = np.isfinite(v)
            if not ok.any():
                continue
            te = sel[ok]
            vol_rows.append(pd.DataFrame({"origin": te.index, "h": h, "model": name, "var": v[ok], "y": te[f"y_rv_{h}"].values}))
            touch_rows.append(touch_long(te, v[ok], h, f"{name}_bm"))

    if "timestral" in a.models:
        add_var(f"timestral_{a.timestral_size}", run_timestral(hr, origins, a.timestral_size, a.ctx, a.bs, a.dry_run))
    if "chronos2" in a.models:
        add_var("chronos2", run_chronos2(hr, origins, a.ctx, a.bs, a.dry_run))
    if "kronos" in a.models:
        v, tk = run_kronos(bars5, origins, a.kronos_size, a.paths, a.kronos_bs, a.dry_run)
        add_var(f"kronos_{a.kronos_size}", v)
        hit = []
        for r in tk.itertuples():
            y = sel.at[r.origin, f"y_up_{r.h}"] if r.dir == "up" else sel.at[r.origin, f"y_dn_{r.h}"]
            b = np.log1p(r.thr) if r.dir == "up" else -np.log1p(-r.thr)
            hit.append(int(y >= b) if r.dir == "up" else int(y <= -b))
        tk["hit"] = hit
        tk["model"] = f"kronos_{a.kronos_size}_paths"
        touch_rows.append(tk)

    vp = pd.concat(vol_rows, ignore_index=True)
    tp = pd.concat(touch_rows, ignore_index=True)
    # Vergleich mit Baselines auf denselben Zeitpunkten
    for f, lst in ((f"vol_preds_{a.mode}.parquet", "v"), (f"touch_preds_{a.mode}.parquet", "t")):
        p = C.RESULTS / f
        if p.exists():
            old = pd.read_parquet(p)
            old = old[old["origin"].isin(origins)]
            if lst == "v":
                vp = pd.concat([old, vp], ignore_index=True)
            else:
                tp = pd.concat([old, tp], ignore_index=True)
        else:
            print(f"Hinweis: {p.name} fehlt -> erst s03_models.py --mode {a.mode} laufen lassen für den Vergleich")
    tag = f"foundation_{a.mode}"
    vp.to_parquet(C.RESULTS / f"vol_preds_{tag}.parquet")
    tp.to_parquet(C.RESULTS / f"touch_preds_{tag}.parquet")
    ref = "har" if "har" in vp["model"].unique() else vp["model"].iloc[0]
    vt = vol_table(vp, ref=ref)
    tt, tpe = touch_table(tp)
    vt.to_csv(C.RESULTS / f"vol_metrics_{tag}.csv", index=False)
    tt.to_csv(C.RESULTS / f"touch_metrics_{tag}.csv", index=False)
    reliability(tpe).to_csv(C.RESULTS / f"reliability_{tag}.csv", index=False)
    write_report(vt, tt, C.RESULTS / f"RESULTS_{tag}.md", f"Foundation Models vs. Baselines ({a.mode}, jede {a.every}. Stunde)",
                 notes=f"{len(origins)} Zeitpunkte. Kronos-Pfade: {a.paths} je Zeitpunkt (Wahrscheinlichkeiten entsprechend grob).")
    print(vt.sort_values(["h", "QLIKE"]).round(4).to_string(index=False))
    print(tt.groupby("model")[["Brier", "LogLoss"]].mean().sort_values("Brier").round(5).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["valid", "test"], default="valid")
    ap.add_argument("--models", nargs="+", default=["chronos2", "timestral", "kronos"])
    ap.add_argument("--every", type=int, default=6, help="nur jede n-te Stunde prognostizieren")
    ap.add_argument("--limit", type=int, default=0, help="max. Anzahl Zeitpunkte (0 = alle), zum schnellen Testen")
    ap.add_argument("--ctx", type=int, default=2048, help="Kontextlänge in Stunden für chronos2/timestral")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--timestral-size", default="209M", choices=["5M", "28M", "123M", "209M"])
    ap.add_argument("--kronos-size", default="small", choices=["mini", "small", "base"])
    ap.add_argument("--paths", type=int, default=32, help="Kronos: gesampelte Pfade je Zeitpunkt")
    ap.add_argument("--kronos-bs", type=int, default=4, help="Kronos: Zeitpunkte pro Batch (x paths)")
    ap.add_argument("--dry-run", action="store_true", help="winzige Zufallsmodelle, nur zum Testen der Pipeline")
    main(ap.parse_args())
