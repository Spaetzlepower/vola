"""Schritt 16: VolFormer-P – eigenes Vortraining auf vielen Coins mit 5-Minuten-Token, dann BTC walk-forward.

  python s16_pretrain.py --mode valid                  # lädt Daten, trainiert vor (nur Daten < 2025), testet auf 2025
  python s16_pretrain.py --mode test                   # nutzt gespeichertes Vortraining, testet einmalig auf 2026
  python s16_pretrain.py --mode valid --pretrain-min 40 --symbols BTCUSDT ETHUSDT SOLUSDT

Stufe 0 (Daten): Binance-Spot-1m-Kerzen für ~20 große Coins (2019–2024) + BTC ab 2018 -> 5-Min.-Bars.
Stufe 1 (Vortraining): kleiner Decoder-only-Transformer, Token = 5-Min.-Bar, Kontext 576 (= 48 h),
  Verlust an JEDER Position (LLM-Prinzip): relative log-Vola für 15/30 Min., 1/4/24/72/168 h + Touch-Wahrscheinlichkeiten.
  Kanäle sind skalenfrei (relativ zur eigenen 7-Tage-Vola) + Vola-Niveau 1/7/30 Tage + Tageszeit -> über Coins übertragbar.
  Nur Daten, deren Ziele vor 2025-01-01 minus 7 Tage feststanden (gilt für valid UND test -> kein Blick in die Zukunft).
Stufe 2 (BTC): je Quartal das vortrainierte Modell auf BTC-Daten bis Blockbeginn weitertrainieren (Embargo 168 h),
  Kalibrierung auf den letzten 60 Tagen vor dem Embargo, Vorhersage zu jeder vollen Stunde (Vergleich mit allen Modellen) UND
  jede 5 Minuten für 15/30/60 Min. (Intraday-Bericht gegen einfache Vergleichsmodelle).
"""
import argparse
import json
import math
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression

import config as C
from evaluate import barriers, bm_touch, qlike, touch_long
from s03_models import denorm, month_starts, norm_target
from s12_volnet import DEV, report

warnings.filterwarnings("ignore")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT", "LTCUSDT", "LINKUSDT",
           "DOTUSDT", "TRXUSDT", "ETCUSDT", "BCHUSDT", "XLMUSDT", "ATOMUSDT", "AVAXUSDT", "FILUSDT", "UNIUSDT",
           "NEARUSDT", "AAVEUSDT"]
MULTI = C.DATA / "multi"
VH = [3, 6, 12, 48, 288, 864, 2016]                 # Vola-Horizonte in 5-Min.-Bars: 15m, 30m, 1h, 4h, 24h, 72h, 168h
TH_BARS = [12, 48, 288, 864, 2016]                  # Touch-Horizonte (= config.HORIZONS_H)
PRE_CUTOFF = pd.Timestamp(C.VALID_START, tz="UTC") - pd.Timedelta(days=7)
DEFAULT = dict(ctx=576, d=128, layers=4, heads=4, ff=512, dropout=0.1, lam_touch=0.5, lr=6e-4, wd=0.05, batch=24,
               pretrain_min=30, adapt_steps=1500, adapt_lr=1e-4)


# ---------------- Stufe 0: Daten ----------------

def download(symbols):
    import s01_download as D
    MULTI.mkdir(parents=True, exist_ok=True)
    for sym in symbols:
        out = MULTI / f"{sym}.parquet"
        if out.exists():
            continue
        first = "2018-01" if sym == "BTCUSDT" else "2019-01"
        last = "2020-12" if sym == "BTCUSDT" and (C.DATA / "klines_1m.parquet").exists() else "2024-12"
        jobs = [("m", m, f"{D.BASE}/spot/monthly/klines/{sym}/1m/{sym}-1m-{m}.zip") for m in D.months(first, last)]
        paths, _ = D.download(jobs, f"multi_{sym}")
        if not paths and sym != "BTCUSDT":
            print(f"  {sym}: keine Daten"); continue
        dfs = []
        for p in paths:
            try:
                df = D.read_zip_csv(p, D.KLINE_COLS)
                df["time"] = D.to_datetime_auto(df["open_time"])
                dfs.append(df[["time", "open", "high", "low", "close", "volume", "trades", "taker_buy_base"]])
            except Exception as e:
                print(f"  {p.name}: {e}")
        k = pd.concat(dfs).drop_duplicates("time").set_index("time").sort_index().astype("float64") if dfs else None
        if sym == "BTCUSDT" and (C.DATA / "klines_1m.parquet").exists():
            k2 = pd.read_parquet(C.DATA / "klines_1m.parquet")[["open", "high", "low", "close", "volume", "trades", "taker_buy_base"]]
            k = k2 if k is None else pd.concat([k[k.index < k2.index[0]], k2])
        b = k.resample("5min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
                                   "trades": "sum", "taker_buy_base": "sum"})
        b.astype("float32").to_parquet(out)
        import shutil
        shutil.rmtree(C.RAW / f"multi_{sym}", ignore_errors=True)          # Zip-Dateien löschen (Platz)
        print(f"  {sym}: {len(b):,} 5-Min.-Bars {b.index[0].date()} bis {b.index[-1].date()}")


def roll_mean(x, w):
    c = np.cumsum(np.r_[0.0, x])
    out = np.full(len(x), np.nan)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def fwd_sum(x, n):
    c = np.cumsum(np.r_[0.0, x])
    out = np.full(len(x), np.nan)
    out[:len(x) - n] = c[1 + n:] - c[1:len(x) - n + 1]          # Summe über i+1 .. i+n
    return out


def fwd_ext(x, n, fn):
    s = pd.Series(x[::-1])
    r = getattr(s.rolling(n, min_periods=n), fn)().values[::-1]   # über i .. i+n-1
    out = np.full(len(x), np.nan)
    out[:-1] = r[1:]                                               # -> über i+1 .. i+n
    return out


def arrays(b):
    """Kanäle (N, 13), relative Vola-Ziele (N, 7), Touch-Labels (N, 80), Basis-log-Varianz (N,) – alles kausal."""
    b = b.copy()
    b["close"] = b["close"].ffill()
    for c in ("open", "high", "low"):
        b[c] = b[c].fillna(b["close"])
    b[["volume", "trades", "taker_buy_base"]] = b[["volume", "trades", "taker_buy_base"]].fillna(0)
    b = b.dropna(subset=["close"])
    close = b["close"].values.astype(np.float64)
    r = np.r_[0.0, np.diff(np.log(close))]
    r2 = r ** 2
    v1d, v7d, v30d = roll_mean(r2, 288) + 1e-12, roll_mean(r2, 2016) + 1e-12, roll_mean(r2, 8640) + 1e-12
    rng = np.log(b["high"].values / b["low"].values)
    vol, trd = b["volume"].values.astype(np.float64), b["trades"].values.astype(np.float64)
    vm, tm = roll_mean(vol, 8640) + 1e-9, roll_mean(trd, 8640) + 1e-9
    idx = b.index
    hod = (idx.hour * 12 + idx.minute // 5).values / 288 * 2 * np.pi
    dow = idx.day_of_week.values / 7 * 2 * np.pi
    X = np.stack([np.clip(r / np.sqrt(v7d), -10, 10), np.log((r2 + 1e-14) / v7d), np.log((rng ** 2 + 1e-14) / v7d),
                  np.log(v1d / v7d), np.log(v7d / v30d), (np.log(v30d) + 13.0) / 2.0,
                  np.log(vol / vm + 1e-6), (2 * b["taker_buy_base"].values - vol) / (vol + 1e-9), np.log(trd / tm + 1e-6),
                  np.sin(hod), np.cos(hod), np.sin(dow), np.cos(dow)], 1)
    X = np.clip(np.nan_to_num(X), -10, 10).astype(np.float16)
    base = np.log(v7d)
    Yv = np.stack([np.log(fwd_sum(r2, n) / n + 1e-14) - base for n in VH], 1).astype(np.float16)
    hi, lo = b["high"].values.astype(np.float64), b["low"].values.astype(np.float64)
    bu, bd = barriers(C.THRESHOLDS)
    Tl = []
    for n in TH_BARS:
        up = np.log(fwd_ext(hi, n, "max") / close)
        dn = np.log(fwd_ext(lo, n, "min") / close)
        Tl += [np.stack([up >= x for x in bu], 1), np.stack([dn <= -x for x in bd], 1)]
    T = np.concatenate(Tl, 1).astype(np.uint8)
    valid = np.isfinite(Yv.astype(np.float32)).all(1) & (np.arange(len(b)) >= 8640)
    end_time = idx + pd.Timedelta(minutes=5) + pd.Timedelta(minutes=5 * max(VH))   # Ziele stehen hier fest
    return dict(idx=idx, X=X, Y=np.nan_to_num(Yv), T=T, valid=valid, end=end_time.values, base=base.astype(np.float32),
                r2=r2.astype(np.float32))


# ---------------- Modell ----------------

class Blk(nn.Module):
    def __init__(self, d, h, ff, p):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(p), nn.Linear(ff, d))
        self.drop = nn.Dropout(p)

    def forward(self, x, mask):
        y = self.n1(x)
        x = x + self.drop(self.att(y, y, y, attn_mask=mask, need_weights=False)[0])
        return x + self.drop(self.mlp(self.n2(x)))


class VolFormerP(nn.Module):
    def __init__(self, d_in, cfg):
        super().__init__()
        d = cfg["d"]
        self.emb = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.zeros(1, cfg["ctx"], d))
        self.blocks = nn.ModuleList([Blk(d, cfg["heads"], cfg["ff"], cfg["dropout"]) for _ in range(cfg["layers"])])
        self.norm = nn.LayerNorm(d)
        self.vol = nn.Linear(d, len(VH))
        self.touch = nn.Linear(d, len(TH_BARS) * 2 * len(C.THRESHOLDS))
        L = cfg["ctx"]
        self.register_buffer("mask", torch.triu(torch.full((L, L), float("-inf")), 1))

    def forward(self, x):
        L = x.shape[1]
        h = self.emb(x) + self.pos[:, -L:]
        for b in self.blocks:
            h = b(h, self.mask[:L, :L])
        h = self.norm(h)
        return self.vol(h), self.touch(h)


def loss_all(mu, y, lg, t, m, lam):
    m = m.float()
    den = m.sum().clamp(min=1)
    d = (y - mu).clamp(-20, 20)
    sl1 = (nn.functional.smooth_l1_loss(mu, y, reduction="none").mean(-1) * m).sum() / den
    ql = ((torch.exp(d) - d - 1).mean(-1) * m).sum() / den
    bce = (nn.functional.binary_cross_entropy_with_logits(lg, t, reduction="none").mean(-1) * m).sum() / den
    return 0.5 * sl1 + 0.5 * ql + lam * bce


class Sampler:
    """Fenster aus mehreren Assets; Verlust nur an Positionen mit gültigen Zielen, die vor 'cutoff' feststanden."""

    def __init__(self, assets, cfg, cutoff, weights=None):
        self.A, self.L, self.cfg = assets, cfg["ctx"], cfg
        cut = np.datetime64(cutoff.tz_convert(None)) if cutoff is not None else None
        self.ok = [a["valid"] & ((a["end"] <= cut) if cut is not None else True) for a in assets]
        self.ends = [np.flatnonzero(o & (np.arange(len(o)) >= self.L)) for o in self.ok]
        n = np.array([len(e) for e in self.ends], float)
        self.p = (n / n.sum()) if weights is None else np.asarray(weights) / np.sum(weights)
        self.ar = np.arange(-self.L + 1, 1)

    def batch(self, B):
        ks = np.random.choice(len(self.A), B, p=self.p)
        out = [[], [], [], []]
        for k in ks:
            e = np.random.choice(self.ends[k])
            w = e + self.ar
            a = self.A[k]
            out[0].append(a["X"][w]); out[1].append(a["Y"][w]); out[2].append(a["T"][w]); out[3].append(self.ok[k][w])
        f = lambda z, dt: torch.tensor(np.stack(z), dtype=dt, device=DEV)
        return f(out[0], torch.float32), f(out[1], torch.float32), f(out[2], torch.float32), f(out[3], torch.bool)


def train(model, sampler, cfg, lr, minutes=None, steps=None, es=None, log_every=200):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["wd"])
    t0, step, best, best_state = time.time(), 0, 1e9, None
    total = steps or 10**9
    while True:
        if steps is not None and step >= steps:
            break
        if minutes is not None and time.time() - t0 > minutes * 60:
            break
        frac = step / total if steps else min(1.0, (time.time() - t0) / (minutes * 60))
        for g in opt.param_groups:                                   # Warmup + Cosinus
            g["lr"] = lr * min(1.0, (step + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        model.train()
        x, y, t, m = sampler.batch(cfg["batch"])
        mu, lg = model(x)
        loss = loss_all(mu, y, lg, t, m, cfg["lam_touch"])
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if es is not None and step % log_every == 0:
            v = es()
            if v < best:
                best, best_state = v, {k: v_.detach().clone() for k, v_ in model.state_dict().items()}
        if step % log_every == 0:
            print(f"    Schritt {step}: Verlust {loss.item():.4f} ({(time.time() - t0) / 60:.1f} Min.)")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_at(model, a, idx_pos, L, bs=256):
    """Ausgabe an der letzten Position von Fenstern, die bei idx_pos enden."""
    model.eval()
    mus, ps = [], []
    ar = np.arange(-L + 1, 1)
    with torch.no_grad():
        for i in range(0, len(idx_pos), bs):
            w = idx_pos[i:i + bs][:, None] + ar[None, :]
            x = torch.tensor(a["X"][w], dtype=torch.float32, device=DEV)
            mu, lg = model(x)
            mus.append(mu[:, -1].float().cpu().numpy()); ps.append(torch.sigmoid(lg[:, -1]).float().cpu().numpy())
    return np.concatenate(mus), np.concatenate(ps)


# ---------------- Ablauf ----------------

def main(a):
    cfg = dict(DEFAULT)
    if a.quick:
        cfg.update(ctx=48, d=32, layers=1, heads=2, ff=64, batch=8, pretrain_min=0.3, adapt_steps=30)
    if a.pretrain_min is not None:
        cfg["pretrain_min"] = a.pretrain_min
    print(f"Gerät: {DEV}")
    if not a.no_download:
        download(a.symbols)
    t0 = time.time()
    assets, names = [], []
    for sym in a.symbols:
        p = MULTI / f"{sym}.parquet"
        if p.exists():
            assets.append(arrays(pd.read_parquet(p))); names.append(sym)
    btc = assets[names.index("BTCUSDT")]
    print(f"{len(assets)} Coins, {sum(int(x['valid'].sum()) for x in assets):,} Lernpositionen ({time.time() - t0:.0f}s)")
    d_in = btc["X"].shape[1]

    # Stufe 1: Vortraining (einmal, nur Daten mit Zielen vor PRE_CUTOFF)
    wpath = C.RESULTS / f"volformer_pre_{a.tag}.pt"
    model = VolFormerP(d_in, cfg).to(DEV)
    print(f"Modell: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} Mio. Parameter")
    if wpath.exists() and not a.retrain:
        model.load_state_dict(torch.load(wpath, map_location=DEV)); print(f"Vortraining geladen: {wpath.name}")
    else:
        smp = Sampler(assets, cfg, PRE_CUTOFF)
        n_pos = sum(int(o.sum()) for o in smp.ok)
        print(f"Stufe 1: Vortraining {cfg['pretrain_min']} Min. auf {n_pos:,} Positionen (Ziele vor {PRE_CUTOFF.date()})")
        train(model, smp, cfg, cfg["lr"], minutes=cfg["pretrain_min"])
        torch.save(model.state_dict(), wpath)
    pre_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Stufe 2: BTC walk-forward
    ds = pd.read_parquet(C.DATA / "dataset.parquet")
    pos_btc = np.searchsorted(btc["idx"].values, ds.index.values) - 1   # letzter 5-Min.-Bar, der VOR t endet
    start, end = (C.VALID_START, C.VALID_END) if a.mode == "valid" else (C.TEST_START, str(ds.index[-1].date()))
    starts = month_starts(start, end)
    Yh = np.stack([norm_target(ds, h).values for h in C.HORIZONS_H], 1)
    okh = np.isfinite(Yh).all(1) & np.stack([ds[f"y_miss_{h}"].values < 0.05 for h in C.HORIZONS_H], 1).all(1) & (pos_btc >= cfg["ctx"])
    bu, bd = barriers(C.THRESHOLDS)
    nthr = len(C.THRESHOLDS)
    hmap = {h: VH.index(h * 12) for h in C.HORIZONS_H}
    vol_rows, touch_rows, intra, ipred = [], [], [], []
    for bi in range(0, len(starts), a.refit_months):
        ms = starts[bi]
        me = starts[bi + a.refit_months] if bi + a.refit_months < len(starts) else pd.Timestamp(end, tz="UTC") + pd.Timedelta("1D")
        tr = np.flatnonzero(okh & (ds.index + pd.Timedelta(hours=168) <= ms))
        te = np.flatnonzero(okh & (ds.index >= ms) & (ds.index < me))
        if len(te) == 0:
            continue
        i_cal = tr[ds.index[tr] >= ms - pd.Timedelta(hours=168) - pd.Timedelta(days=a.cal_days)]   # jüngste Wochen
        cal_start = ds.index[i_cal[0]]
        # Weitertraining auf BTC: Ziele müssen vor Beginn des Kalibrierteils feststehen (Embargo), Early-Stop nicht nötig
        model.load_state_dict(pre_state)
        smp = Sampler([btc], cfg, cal_start - pd.Timedelta(hours=1))
        train(model, smp, cfg, cfg["adapt_lr"], steps=cfg["adapt_steps"], log_every=10**9)
        pc = pos_btc[i_cal]; pt = pos_btc[te]
        mu_c, p_c = predict_at(model, btc, pc, cfg["ctx"]); mu_t, p_t = predict_at(model, btc, pt, cfg["ctx"])
        dte, dcal = ds.iloc[te], ds.iloc[i_cal]
        for k, h in enumerate(C.HORIZONS_H):
            j = hmap[h]
            xc = mu_c[:, j] + btc["base"][pc]; xt = mu_t[:, j] + btc["base"][pt]   # absolute log-Varianz je Bar
            yc = Yh[i_cal, k]
            A = np.c_[np.ones(len(yc)), xc]
            beta, *_ = np.linalg.lstsq(A, yc, rcond=None)
            s2 = np.var(yc - A @ beta)
            var_t = np.asarray(denorm(beta[0] + beta[1] * xt + 0.5 * s2, dte, h))
            var_c = np.asarray(denorm(beta[0] + beta[1] * xc + 0.5 * s2, dcal, h))
            vol_rows.append(pd.DataFrame({"origin": dte.index, "h": h, "model": a.tag, "var": var_t, "y": dte[f"y_rv_{h}"].values}))
            touch_rows.append(touch_long(dte, var_t, h, f"{a.tag}_bm"))
            for jj, dname in enumerate(("up", "dn")):
                col = slice((2 * k + jj) * nthr, (2 * k + jj + 1) * nthr)
                bs_ = bu if dname == "up" else bd
                hit_c = np.stack([(dcal[f"y_up_{h}"].values >= x) if dname == "up" else (dcal[f"y_dn_{h}"].values <= -x) for x in bs_], 1)
                hit_t = np.stack([(dte[f"y_up_{h}"].values >= x) if dname == "up" else (dte[f"y_dn_{h}"].values <= -x) for x in bs_], 1)
                for name, pcal, ptest in ((f"{a.tag}_touch_iso", p_c[:, col], p_t[:, col]),
                                          (f"{a.tag}_bm_iso", np.stack([bm_touch(var_c, x) for x in bs_], 1),
                                           np.stack([bm_touch(var_t, x) for x in bs_], 1))):
                    iso = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip").fit(pcal.ravel(), hit_c.ravel())
                    for m_, thr in enumerate(C.THRESHOLDS):
                        touch_rows.append(pd.DataFrame({"origin": dte.index, "h": h, "dir": dname, "thr": thr, "model": name,
                                                        "p": iso.predict(ptest[:, m_]), "hit": hit_t[:, m_].astype(np.int8)}))
        # Intraday: jede 5. Minute im Block, Horizonte 15/30/60 Min. (relative Ziele, Kalibrierung auf den letzten 30 Tagen vor Block)
        bi_idx = btc["idx"]
        blk = np.flatnonzero((bi_idx >= ms) & (bi_idx < me) & btc["valid"])
        calp = np.flatnonzero((bi_idx >= ms - pd.Timedelta(days=37)) & (bi_idx < ms - pd.Timedelta(days=7)) & btc["valid"])
        if len(blk) and len(calp):
            blk, calp = blk[::a.intraday_stride], calp[::a.intraday_stride]
            mb, _ = predict_at(model, btc, blk, cfg["ctx"]); mc, _ = predict_at(model, btc, calp, cfg["ctx"])
            r2 = btc["r2"]
            for j, n in enumerate(VH[:3]):
                y_rel_c = btc["Y"][calp, j].astype(np.float32); y_rel_b = btc["Y"][blk, j].astype(np.float32)
                A = np.c_[np.ones(len(calp)), mc[:, j]]
                beta, *_ = np.linalg.lstsq(A, y_rel_c, rcond=None)
                s2 = np.var(y_rel_c - A @ beta)
                f_model = np.exp(beta[0] + beta[1] * mb[:, j] + 0.5 * s2)
                # Vergleich: letzte Stunde (12 Bars) bzw. EWMA (Halbwertszeit 1 h), jeweils relativ zur 7-Tage-Vola, kalibriert
                last1h = pd.Series(r2).rolling(12).mean().values / np.exp(btc["base"])
                ew = pd.Series(r2).ewm(halflife=12).mean().values / np.exp(btc["base"])
                y_lin = np.exp(y_rel_b)
                res = {"Horizont": f"{n * 5} Min.", "Block": f"{ms:%Y-%m}", "n": len(blk),
                       "QLIKE VolFormer-P": qlike(y_lin, f_model).mean()}
                for nm, base_ in (("letzte Stunde", last1h), ("EWMA 1h", ew), ("7-Tage-Vola", np.ones(len(r2)))):
                    sc_ = np.mean(np.exp(y_rel_c) / np.maximum(base_[calp], 1e-6))       # Skalierung auf Kalibrierzeitraum
                    res[f"QLIKE {nm}"] = qlike(y_lin, np.maximum(base_[blk], 1e-6) * sc_).mean()
                intra.append(res)
                ipred.append(pd.DataFrame({"time": bi_idx[blk] + pd.Timedelta(minutes=5), "minutes": n * 5,
                                           "var": f_model * np.exp(btc["base"][blk]) * n, "y": y_lin * np.exp(btc["base"][blk]) * n}))
        print(f"  Block ab {ms:%Y-%m}: BTC-Weitertraining bis {cal_start.date()}, test {len(te):,} Stunden ({time.time() - t0:.0f}s)")
    vp, tp = pd.concat(vol_rows, ignore_index=True), pd.concat(touch_rows, ignore_index=True)
    report(vp, tp, a.mode, cfg, a.tag)
    if intra:
        I = pd.DataFrame(intra)
        qc = [c for c in I.columns if c.startswith("QLIKE")]
        num = pd.DataFrame({hz: {**{c: np.average(g[c], weights=g["n"]) for c in qc}, "n": g["n"].sum()}
                            for hz, g in I.groupby("Horizont", sort=False)}).T
        pd.concat(ipred, ignore_index=True).to_parquet(C.RESULTS / f"vol_preds_{a.tag}_intraday_{a.mode}.parquet")
        txt = ["# VolFormer-P: Intraday-Vola (Prognose alle 5 Minuten)", "", f"Modus {a.mode}. QLIKE (niedriger = besser), relativ zur 7-Tage-Vola; "
               "Vergleichsmodelle auf den 30 Tagen vor jedem Block skaliert.", "", num.round(4).to_markdown(), "",
               "## je Block", "", I.round(4).to_markdown(index=False), ""]
        (C.RESULTS / f"RESULTS_{a.tag}_intraday_{a.mode}.md").write_text("\n".join(txt))
        print("\n".join(txt[:6]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["valid", "test"], default="valid")
    ap.add_argument("--symbols", nargs="+", default=SYMBOLS)
    ap.add_argument("--pretrain-min", type=float, default=None, help="Minuten Vortraining (Standard 30)")
    ap.add_argument("--refit-months", type=int, default=3)
    ap.add_argument("--cal-days", type=int, default=60, help="Kalibrierung auf den letzten n Tagen vor dem Embargo")
    ap.add_argument("--intraday-stride", type=int, default=1, help="jede n-te 5-Min.-Position im Intraday-Bericht")
    ap.add_argument("--retrain", action="store_true", help="Vortraining neu, auch wenn gespeichert")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--quick", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--tag", default="vfp")
    main(ap.parse_args())
