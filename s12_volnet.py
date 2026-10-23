"""Schritt 12: Eigenes Modell "VolNet" – Sequenz-CNN (DeepVol-Idee) + Tabellen-Features, Multi-Task.

  python s12_volnet.py --mode valid --search      # Hyperparameter-Suche NUR auf 2025, bestes Setup -> results/volnet_best.json
  python s12_volnet.py --mode valid               # mit bestem Setup (oder Standard)
  python s12_volnet.py --mode test                # einmalig auf 2026

Architektur:
  - Sequenzteil: letzte L 15-Min.-Bars (Standard 672 = 7 Tage), Kanäle: log-RV (saisonbereinigt), Rendite, log-Spanne,
    Volumen-/Trade-Verhältnis, Taker-Ungleichgewicht, Saisonniveau -> dilatiertes kausales CNN (Residual-Blöcke)
  - Tabellenteil: alle Features aus s02 + Foundation-Model-Prognosen + DVOL (wie s05) -> MLP
  - Köpfe: (1) log-Varianz für alle Horizonte (Verlust: 0.5·SmoothL1 + 0.5·QLIKE)
           (2) Touch-Wahrscheinlichkeiten je Horizont/Richtung/Schwelle (BCE)
Training: walk-forward (alle --refit-months neu), chronologisch Fit 75 % / Early-Stopping 12.5 % / Kalibrierung 12.5 %
mit Embargo; Ensemble über mehrere Seeds; danach Kalibrierung wie s05 (log-linear für Vola, isotonisch für Touch).
Kein Blick in die Zukunft: Sequenz endet mit dem letzten 15-Min.-Bar VOR dem Prognosezeitpunkt; Normierung der Sequenz
nur aus Daten vor 2025; Trainingszeitpunkte nur, wenn alle Ziele (bis 168 h) vor Blockbeginn feststanden.
"""
import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression

import config as C
from evaluate import barriers, bm_touch, reliability, touch_long, touch_table, vol_table, write_report
from s03_models import denorm, feature_cols, month_starts, norm_target

warnings.filterwarnings("ignore")
H = C.BARS_PER_HOUR
HS = C.HORIZONS_H
HMAX = max(HS)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT = dict(L=672, ch=32, levels=8, tab_hidden=128, dropout=0.2, lam_touch=1.0, lr=1e-3, wd=1e-4,
               batch=256, epochs=40, patience=5, seeds=3)


# ---------------- Daten ----------------

def seq_features():
    """15-Min.-Kanäle (kausal), normiert mit Statistik vor 2025."""
    b = pd.read_parquet(C.DATA / "bars_5m.parquet")
    seas = pd.read_parquet(C.DATA / "seasonality.parquet")["seas"].values
    b["seas"] = seas[(b.index.day_of_week * 288 + b.index.hour * 12 + b.index.minute // 5).values]
    q = b.resample("15min").agg({"r": "sum", "high": "max", "low": "min", "volume": "sum", "trades": "sum",
                                 "taker_buy_base": "sum", "seas": "mean"})
    q["rv"] = (b["r"] ** 2).resample("15min").sum()
    q = q.dropna(subset=["seas"])
    f = pd.DataFrame(index=q.index)
    f["lrv"] = np.log(q["rv"] / (3 * q["seas"]) + 1e-10)
    scale = np.sqrt(q["rv"].rolling(96 * 7, min_periods=96).mean()) + 1e-6
    f["ret"] = (q["r"] / scale).clip(-10, 10)
    f["lrange"] = np.log(np.log(q["high"] / q["low"]) + 1e-6)
    v30 = q["volume"].rolling(96 * 30, min_periods=96).mean()
    t30 = q["trades"].rolling(96 * 30, min_periods=96).mean()
    f["lvol"] = np.log(q["volume"] / v30 + 1e-6)
    f["ltr"] = np.log(q["trades"] / t30 + 1e-6)
    f["tbi"] = (2 * q["taker_buy_base"] - q["volume"]) / (q["volume"] + 1e-9)
    f["lseas"] = np.log(q["seas"])
    pre = f[f.index < pd.Timestamp(C.VALID_START, tz="UTC")]
    f = ((f - pre.mean()) / (pre.std() + 1e-9)).clip(-8, 8).fillna(0.0)
    return f


def load_table():
    ds = pd.read_parquet(C.DATA / "dataset.parquet")
    p = C.DATA / "fm_features.parquet"
    if p.exists():
        ds = ds.join(pd.read_parquet(p), how="left")
    try:
        from s05_ensemble import dvol_features
        dvf = dvol_features(ds)
        if dvf is not None:
            ds = ds.join(dvf)
    except Exception as e:
        print(f"DVOL-Features nicht verfügbar: {e}")
    feats = [c for c in feature_cols(ds)] + [c for c in ds.columns if c.startswith(("seas_fwd_", "event_in_"))]
    feats = list(dict.fromkeys(feats))
    return ds, feats


def targets(ds):
    Y = np.stack([norm_target(ds, h).values for h in HS], 1).astype(np.float32)          # (N, 5)
    bu, bd = barriers(C.THRESHOLDS)
    T = []
    for h in HS:
        T.append(np.stack([(ds[f"y_up_{h}"].values >= b) for b in bu], 1))
        T.append(np.stack([(ds[f"y_dn_{h}"].values <= -b) for b in bd], 1))
    T = np.concatenate(T, 1).astype(np.float32)                                         # (N, 5*2*8)
    ok = np.isfinite(Y).all(1) & np.stack([ds[f"y_miss_{h}"].values < 0.05 for h in HS], 1).all(1)
    return Y, T, ok


# ---------------- Modell ----------------

class Block(nn.Module):
    def __init__(self, ch, dil, drop):
        super().__init__()
        self.pad = 2 * dil
        self.c1 = nn.Conv1d(ch, ch, 3, dilation=dil)
        self.c2 = nn.Conv1d(ch, ch, 3, dilation=dil)
        self.act, self.drop = nn.GELU(), nn.Dropout(drop)

    def forward(self, x):
        y = self.act(self.c1(nn.functional.pad(x, (self.pad, 0))))
        y = self.drop(self.act(self.c2(nn.functional.pad(y, (self.pad, 0)))))
        return x + y


class VolNet(nn.Module):
    def __init__(self, c_in, tab_in, n_touch, cfg):
        super().__init__()
        ch = cfg["ch"]
        self.inp = nn.Conv1d(c_in, ch, 1)
        self.tcn = nn.Sequential(*[Block(ch, 2 ** i, cfg["dropout"]) for i in range(cfg["levels"])])
        self.att = nn.Linear(ch, 1)
        self.tab = nn.Sequential(nn.Linear(tab_in, cfg["tab_hidden"]), nn.GELU(), nn.Dropout(cfg["dropout"]),
                                 nn.Linear(cfg["tab_hidden"], 64), nn.GELU())
        self.trunk = nn.Sequential(nn.Linear(2 * ch + 64, 128), nn.GELU(), nn.Dropout(cfg["dropout"]))
        self.vol = nn.Linear(128, len(HS))
        self.touch = nn.Linear(128, n_touch)

    def forward(self, seq, tab):
        z = self.tcn(self.inp(seq))                       # (B, ch, L)
        last = z[:, :, -1]
        w = torch.softmax(self.att(z.transpose(1, 2)).squeeze(-1), 1)
        pooled = (z * w.unsqueeze(1)).sum(-1)
        h = self.trunk(torch.cat([last, pooled, self.tab(tab)], 1))
        return self.vol(h), self.touch(h)


def loss_fn(mu, y, logit, t, lam):
    sl1 = nn.functional.smooth_l1_loss(mu, y)
    d = (y - mu).clamp(-20, 20)
    ql = (torch.exp(d) - d - 1).mean()                    # QLIKE auf Varianz-Ebene (log-Parametrisierung)
    bce = nn.functional.binary_cross_entropy_with_logits(logit, t)
    return 0.5 * sl1 + 0.5 * ql + lam * bce


# ---------------- Training ----------------

class Data:
    def __init__(self, seq, pos, tab, Y, T, cfg):
        self.seq = torch.tensor(seq.values.T[None], dtype=torch.float32, device=DEV)   # (1, C, N15)
        self.pos, self.tab, self.Y, self.T, self.L = pos, tab, Y, T, cfg["L"]
        self.ar = torch.arange(-self.L, 0, device=DEV)

    def batch(self, idx):
        p = torch.tensor(self.pos[idx], device=DEV)
        win = (p[:, None] + self.ar[None, :]).clamp(min=0)                               # (B, L)
        s = self.seq[0][:, win].permute(1, 0, 2)                                          # (B, C, L)
        return (s, torch.tensor(self.tab[idx], device=DEV),
                torch.tensor(self.Y[idx], device=DEV), torch.tensor(self.T[idx], device=DEV))


def predict(model, data, idx, bs=1024):
    model.eval()
    mus, ps = [], []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            s, tb, _, _ = data.batch(idx[i:i + bs])
            mu, lg = model(s, tb)
            mus.append(mu.cpu().numpy()); ps.append(torch.sigmoid(lg).cpu().numpy())
    return np.concatenate(mus), np.concatenate(ps)


def train_one(data, i_fit, i_es, cfg, seed, n_tab, n_touch, c_in):
    torch.manual_seed(seed); np.random.seed(seed)
    model = VolNet(c_in, n_tab, n_touch, cfg).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    best, best_state, bad = 1e9, None, 0
    for ep in range(cfg["epochs"]):
        model.train()
        perm = np.random.permutation(i_fit)
        for i in range(0, len(perm), cfg["batch"]):
            s, tb, y, t = data.batch(perm[i:i + cfg["batch"]])
            mu, lg = model(s, tb)
            loss = loss_fn(mu, y, lg, t, cfg["lam_touch"])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = []
            for i in range(0, len(i_es), 2048):
                s, tb, y, t = data.batch(i_es[i:i + 2048])
                mu, lg = model(s, tb)
                vl.append(loss_fn(mu, y, lg, t, cfg["lam_touch"]).item() * len(s))
            vl = sum(vl) / len(i_es)
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return model, best


def run(mode, cfg, refit_months, tag="nn", quiet=False):
    t0 = time.time()
    ds, feats = load_table()
    seq = seq_features()
    Y, T, ok = targets(ds)
    pos = np.searchsorted(seq.index.values, ds.index.values)          # erster 15-Min.-Bar >= t -> Fenster endet davor
    ok &= pos >= cfg["L"]
    start, end = (C.VALID_START, C.VALID_END) if mode == "valid" else (C.TEST_START, str(ds.index[-1].date()))
    starts = month_starts(start, end)
    X_all = ds[feats].values.astype(np.float32)
    bu, bd = barriers(C.THRESHOLDS)
    nthr = len(C.THRESHOLDS)
    vol_rows, touch_rows = [], []
    for i in range(0, len(starts), refit_months):
        ms = starts[i]
        me = starts[i + refit_months] if i + refit_months < len(starts) else pd.Timestamp(end, tz="UTC") + pd.Timedelta("1D")
        tr = np.flatnonzero(ok & (ds.index + pd.Timedelta(hours=HMAX) <= ms))
        te = np.flatnonzero(ok & (ds.index >= ms) & (ds.index < me))
        if len(te) == 0 or len(tr) < 5000:
            continue
        n = len(tr)
        i_fit, i_es, i_cal = tr[:int(n * .75)], tr[int(n * .75):int(n * .875)], tr[int(n * .875):]
        emb = pd.Timedelta(hours=HMAX)                                 # Embargo zwischen den Teilen
        i_fit = i_fit[ds.index[i_fit] + emb <= ds.index[i_es[0]]]
        i_es = i_es[ds.index[i_es] + emb <= ds.index[i_cal[0]]]
        mu_x, sd_x = np.nanmean(X_all[i_fit], 0), np.nanstd(X_all[i_fit], 0) + 1e-6
        tab = np.nan_to_num(np.clip((X_all - mu_x) / sd_x, -10, 10)).astype(np.float32)
        y_mean = np.nanmean(Y[i_fit], 0)                               # Ziele zentrieren (QLIKE ist verschiebungsinvariant)
        data = Data(seq, pos, tab, np.nan_to_num(Y - y_mean), T, cfg)
        mus_cal, ps_cal, mus_te, ps_te = [], [], [], []
        for sd in range(cfg["seeds"]):
            model, best = train_one(data, i_fit, i_es, cfg, C.SEED + sd, tab.shape[1], T.shape[1], seq.shape[1])
            a, b = predict(model, data, i_cal); c, d = predict(model, data, te)
            mus_cal.append(a); ps_cal.append(b); mus_te.append(c); ps_te.append(d)
        mu_cal, p_cal = np.mean(mus_cal, 0) + y_mean, np.mean(ps_cal, 0)
        mu_te, p_te = np.mean(mus_te, 0) + y_mean, np.mean(ps_te, 0)
        dte, dcal = ds.iloc[te], ds.iloc[i_cal]
        for k, h in enumerate(HS):
            yc = Y[i_cal, k]
            A = np.c_[np.ones(len(yc)), mu_cal[:, k]]
            beta, *_ = np.linalg.lstsq(A, yc, rcond=None)
            s2 = np.var(yc - A @ beta)
            var_te = denorm(beta[0] + beta[1] * mu_te[:, k] + 0.5 * s2, dte, h)
            var_cal = denorm(beta[0] + beta[1] * mu_cal[:, k] + 0.5 * s2, dcal, h)
            vol_rows.append(pd.DataFrame({"origin": dte.index, "h": h, "model": tag, "var": np.asarray(var_te),
                                          "y": dte[f"y_rv_{h}"].values}))
            touch_rows.append(touch_long(dte, np.asarray(var_te), h, f"{tag}_bm"))
            for j, d in enumerate(("up", "dn")):
                col = slice((2 * k + j) * nthr, (2 * k + j + 1) * nthr)
                # (a) direkter Touch-Kopf, isotonisch kalibriert; (b) BM aus kalibrierter Varianz, isotonisch
                for name, pc, pt in ((f"{tag}_touch_iso", p_cal[:, col], p_te[:, col]),
                                     (f"{tag}_bm_iso", np.stack([bm_touch(np.asarray(var_cal), x) for x in (bu if d == "up" else bd)], 1),
                                      np.stack([bm_touch(np.asarray(var_te), x) for x in (bu if d == "up" else bd)], 1))):
                    hc = T[i_cal, col]
                    iso = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip").fit(pc.ravel(), hc.ravel())
                    ht = T[te, col]
                    for m_, thr in enumerate(C.THRESHOLDS):
                        touch_rows.append(pd.DataFrame({"origin": dte.index, "h": h, "dir": d, "thr": thr, "model": name,
                                                        "p": iso.predict(pt[:, m_]), "hit": ht[:, m_].astype(np.int8)}))
        if not quiet:
            print(f"  Block ab {ms:%Y-%m}: fit {len(i_fit):,} / es {len(i_es):,} / kal {len(i_cal):,} / test {len(te):,} "
                  f"({time.time() - t0:.0f}s)")
    vp = pd.concat(vol_rows, ignore_index=True)
    tp = pd.concat(touch_rows, ignore_index=True)
    return vp, tp


def report(vp, tp, mode, cfg):
    tag = f"nn_{mode}"
    vp.to_parquet(C.RESULTS / f"vol_preds_{tag}.parquet")
    tp.to_parquet(C.RESULTS / f"touch_preds_{tag}.parquet")
    # Vergleich mit Ensemble/Baselines auf denselben Zeitpunkten
    vv, tt_ = [vp], [tp]
    for f in (f"vol_preds_ens_{mode}.parquet", f"vol_preds_{mode}.parquet"):
        p = C.RESULTS / f
        if p.exists():
            x = pd.read_parquet(p)
            vv.append(x[x["model"].isin(["har", "lgbm", "ens_lgbm_all", "ens_har_all", "ens_lgbm_fm"])])
    for f in (f"touch_preds_ens_{mode}.parquet", f"touch_preds_{mode}.parquet"):
        p = C.RESULTS / f
        if p.exists():
            x = pd.read_parquet(p)
            tt_.append(x[x["model"].isin(["har_bm", "ens_har_all_iso", "ens_lgbm_all_iso", "climatology"])])
    V = pd.concat(vv, ignore_index=True).drop_duplicates(["origin", "h", "model"])
    Tt = pd.concat(tt_, ignore_index=True).drop_duplicates(["origin", "h", "dir", "thr", "model"])
    vt = vol_table(V, ref="har") if "har" in V["model"].unique() else vol_table(V, ref="nn")
    tt, tpe = touch_table(Tt)
    vt.to_csv(C.RESULTS / f"vol_metrics_{tag}.csv", index=False)
    tt.to_csv(C.RESULTS / f"touch_metrics_{tag}.csv", index=False)
    reliability(tpe).to_csv(C.RESULTS / f"reliability_{tag}.csv", index=False)
    write_report(vt, tt, C.RESULTS / f"RESULTS_{tag}.md", f"VolNet vs. Ensemble/Baselines ({mode})",
                 notes=f"Konfiguration: {json.dumps(cfg)}")
    print(vt.sort_values(["h", "QLIKE"]).round(4).to_string(index=False))
    s = tt.groupby("model").agg(Brier=("Brier", "mean"), p=("mean_p", "mean"), hit=("hit_rate", "mean"))
    s["Bias"] = s["p"] - s["hit"]
    print(s.drop(columns=["p", "hit"]).sort_values("Brier").round(5).to_string())


def score(vp, tp):
    """Ein Wert für die Suche: mittleres QLIKE (alle Horizonte) + 10 x mittlerer Brier."""
    ql = vol_table(vp, ref="nn").groupby("model")["QLIKE"].mean().iloc[0]
    tt, _ = touch_table(tp[tp["model"] == "nn_bm_iso"])
    return ql + 10 * tt["Brier"].mean(), ql, tt["Brier"].mean()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["valid", "test"], default="valid")
    ap.add_argument("--refit-months", type=int, default=3)
    ap.add_argument("--search", action="store_true", help="Hyperparameter-Suche (nur mit --mode valid)")
    ap.add_argument("--quick", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    print(f"Gerät: {DEV}")
    best_path = C.RESULTS / "volnet_best.json"
    cfg = dict(DEFAULT)
    if a.quick:
        cfg.update(L=96, ch=8, levels=3, epochs=2, seeds=1, tab_hidden=16)
    if a.search:
        assert a.mode == "valid", "Suche nur auf dem Validierungszeitraum!"
        grid = [dict(), dict(ch=64), dict(dropout=0.3), dict(L=336), dict(lam_touch=0.3), dict(levels=10, L=1344),
                dict(lr=3e-4, epochs=60)]
        res = []
        for g in grid:
            c = dict(cfg, **g)
            if a.quick:
                c.update(L=min(c["L"], 96), ch=min(c["ch"], 8), levels=3, epochs=2, seeds=1)
            vp, tp = run("valid", dict(c, seeds=1), max(a.refit_months, 6), quiet=True)
            sc, ql, br = score(vp, tp)
            res.append({**g, "score": sc, "QLIKE": ql, "Brier": br})
            print(f"  {g or 'Standard'}: Score {sc:.4f} (QLIKE {ql:.4f}, Brier {br:.5f})")
        R = pd.DataFrame(res).sort_values("score")
        R.to_csv(C.RESULTS / "volnet_search.csv", index=False)
        best = {k: v for k, v in R.iloc[0].items() if k in DEFAULT and pd.notna(v)}
        cfg.update({k: (int(v) if isinstance(DEFAULT[k], int) else float(v)) for k, v in best.items()})
        best_path.write_text(json.dumps(cfg))
        print(f"Bestes Setup: {cfg}")
    elif best_path.exists() and not a.quick:
        cfg = json.loads(best_path.read_text())
        print(f"Setup aus {best_path.name}: {cfg}")
    vp, tp = run(a.mode, cfg, a.refit_months)
    report(vp, tp, a.mode, cfg)
