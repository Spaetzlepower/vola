"""Schritt 15: "VolFormer" – kleiner Decoder-only-Transformer, von Grund auf auf BTC trainiert.

  python s15_volformer.py --mode valid --search     # kleine Suche NUR auf 2025 -> results/volnet_best_vf.json
  python s15_volformer.py --mode test               # einmaliger Test 2026

Architektur (Bausteine aus Kronos / TimesFM / Timestral / TTM):
  - Token = 1 Stunde: Patch aus 12 Fünf-Minuten-Bars x 6 Kanäle (log-RV saisonbereinigt, Rendite, log-Spanne,
    Volumen-/Trade-Verhältnis, Taker-Ungleichgewicht) + Tabellen-Features dieser Stunde (HAR, FM-Prognosen, DVOL, Termine)
  - Linear-Embedding -> N Transformer-Blöcke (kausale Self-Attention, Pre-LayerNorm, GELU-MLP), gelernte Positionen
  - Köpfe an JEDER Position: log-Varianz für 1/4/24/72/168 h + Touch-Wahrscheinlichkeiten (80)
  - LLM-Prinzip: Verlust an allen Positionen des Fensters (kausal maskiert) -> viele Lernsignale je Beispiel
Training/Evaluation exakt wie VolNet (s12): walk-forward, Embargo 168 h, Fit/Early-Stop/Kalibrierung, Seeds-Ensemble,
log-lineare Vola-Kalibrierung + isotonische Touch-Kalibrierung, Vergleich mit Ensemble/VolNet.
Kein Blick in die Zukunft: Token für Stunde t enthält nur Bars VOR dem Prognosezeitpunkt t; Ziele an Positionen
werden nur verwendet, wenn sie vollständig vor dem Blockbeginn feststanden.
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
from evaluate import barriers, bm_touch, touch_long
from s03_models import denorm, month_starts
from s12_volnet import DEV, HMAX, HS, load_table, loss_fn, report, score, seq_features, targets

warnings.filterwarnings("ignore")
DEFAULT = dict(ctx=168, d=128, layers=4, heads=4, ff=512, dropout=0.1, lam_touch=1.0, lr=5e-4, wd=0.05,
               batch=64, steps_per_epoch=400, epochs=40, patience=4, seeds=2)


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


class VolFormer(nn.Module):
    def __init__(self, d_in, n_touch, cfg):
        super().__init__()
        d = cfg["d"]
        self.emb = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.zeros(1, cfg["ctx"], d))
        self.blocks = nn.ModuleList([Blk(d, cfg["heads"], cfg["ff"], cfg["dropout"]) for _ in range(cfg["layers"])])
        self.norm = nn.LayerNorm(d)
        self.vol = nn.Linear(d, len(HS))
        self.touch = nn.Linear(d, n_touch)
        L = cfg["ctx"]
        self.register_buffer("mask", torch.triu(torch.full((L, L), float("-inf")), 1))

    def forward(self, x):                                # x: (B, L, d_in)
        L = x.shape[1]
        h = self.emb(x) + self.pos[:, -L:]
        for b in self.blocks:
            h = b(h, self.mask[:L, :L])
        h = self.norm(h)
        return self.vol(h), self.touch(h)                # (B, L, 5), (B, L, 80)


def tokens(ds, feats):
    """Je Stunde (= Zeile in dataset) ein Token: Patch der 12 Fünf-Minuten-Bars VOR t + Tabellen-Features bei t."""
    s5 = seq_features("5min").drop(columns=["lseas"], errors="ignore")
    pos5 = np.searchsorted(s5.index.values, ds.index.values)             # erster 5-Min.-Bar >= t
    X5 = s5.values.astype(np.float32)
    idx = (pos5[:, None] + np.arange(-12, 0)[None, :]).clip(0)
    patch = X5[idx].reshape(len(ds), -1)                                  # (N, 12*C)
    patch[pos5 < 12] = 0
    return patch


def run(mode, cfg, refit_months, tag="vf", quiet=False):
    t0 = time.time()
    ds, feats = load_table()
    Y, T, ok = targets(ds)
    patch = tokens(ds, feats)
    X_tab = ds[feats].values.astype(np.float32)
    L = cfg["ctx"]
    N = len(ds)
    start, end = (C.VALID_START, C.VALID_END) if mode == "valid" else (C.TEST_START, str(ds.index[-1].date()))
    starts = month_starts(start, end)
    bu, bd = barriers(C.THRESHOLDS)
    nthr = len(C.THRESHOLDS)
    vol_rows, touch_rows = [], []
    for bi in range(0, len(starts), refit_months):
        ms = starts[bi]
        me = starts[bi + refit_months] if bi + refit_months < len(starts) else pd.Timestamp(end, tz="UTC") + pd.Timedelta("1D")
        avail = ok & (ds.index + pd.Timedelta(hours=HMAX) <= ms)              # Ziel vor Blockbeginn bekannt
        tr = np.flatnonzero(avail & (np.arange(N) >= L))
        te = np.flatnonzero(ok & (ds.index >= ms) & (ds.index < me) & (np.arange(N) >= L - 1))
        if len(te) == 0 or len(tr) < 5000:
            continue
        n = len(tr)
        i_fit, i_es, i_cal = tr[:int(n * .75)], tr[int(n * .75):int(n * .875)], tr[int(n * .875):]
        emb = pd.Timedelta(hours=HMAX)
        i_fit = i_fit[ds.index[i_fit] + emb <= ds.index[i_es[0]]]
        i_es = i_es[ds.index[i_es] + emb <= ds.index[i_cal[0]]]
        fit_mask = np.zeros(N, bool); fit_mask[i_fit] = True
        mu_x, sd_x = np.nanmean(X_tab[i_fit], 0), np.nanstd(X_tab[i_fit], 0) + 1e-6
        tab = np.nan_to_num(np.clip((X_tab - mu_x) / sd_x, -10, 10)).astype(np.float32)
        X = torch.tensor(np.concatenate([patch, tab], 1), device=DEV)
        y_mean = np.nanmean(Y[i_fit], 0)
        Yt = torch.tensor(np.nan_to_num(Y - y_mean), device=DEV)
        Tt = torch.tensor(T, device=DEV)
        Mfit = torch.tensor(fit_mask, device=DEV)
        ar = torch.arange(-L + 1, 1, device=DEV)

        def window(ends):
            w = torch.as_tensor(ends, device=DEV)[:, None] + ar[None, :]   # (B, L), endet bei 'ends'
            return X[w], Yt[w], Tt[w], Mfit[w]

        def predict(model, idx, bs=512):
            model.eval(); mus, ps = [], []
            with torch.no_grad():
                for i in range(0, len(idx), bs):
                    x, _, _, _ = window(idx[i:i + bs])
                    mu, lg = model(x)
                    mus.append(mu[:, -1].cpu().numpy()); ps.append(torch.sigmoid(lg[:, -1]).cpu().numpy())
            return np.concatenate(mus), np.concatenate(ps)

        res_cal, res_te = [], []
        for sd in range(cfg["seeds"]):
            torch.manual_seed(C.SEED + sd); np.random.seed(C.SEED + sd)
            model = VolFormer(X.shape[1], T.shape[1], cfg).to(DEV)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
            best, best_state, bad = 1e9, None, 0
            for ep in range(cfg["epochs"]):
                model.train()
                for _ in range(cfg["steps_per_epoch"]):
                    ends = np.random.choice(i_fit, cfg["batch"])
                    x, y, t, m = window(ends)
                    mu, lg = model(x)
                    m = m.float()
                    if m.sum() < 1:
                        continue
                    d = (y - mu).clamp(-20, 20)
                    sl1 = (nn.functional.smooth_l1_loss(mu, y, reduction="none").mean(-1) * m).sum() / m.sum()
                    ql = ((torch.exp(d) - d - 1).mean(-1) * m).sum() / m.sum()
                    bce = (nn.functional.binary_cross_entropy_with_logits(lg, t, reduction="none").mean(-1) * m).sum() / m.sum()
                    loss = 0.5 * sl1 + 0.5 * ql + cfg["lam_touch"] * bce
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                # Early Stopping: Verlust an der letzten Position (wie bei der Vorhersage)
                model.eval()
                with torch.no_grad():
                    vl, nv = 0.0, 0
                    for i in range(0, len(i_es), 512):
                        x, y, t, _ = window(i_es[i:i + 512])
                        mu, lg = model(x)
                        vl += loss_fn(mu[:, -1], y[:, -1], lg[:, -1], t[:, -1], cfg["lam_touch"]).item() * len(x); nv += len(x)
                    vl /= nv
                if vl < best - 1e-4:
                    best, bad = vl, 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= cfg["patience"]:
                        break
            model.load_state_dict(best_state)
            res_cal.append(predict(model, i_cal)); res_te.append(predict(model, te))
        mu_cal = np.mean([r[0] for r in res_cal], 0) + y_mean
        p_cal = np.mean([r[1] for r in res_cal], 0)
        mu_te = np.mean([r[0] for r in res_te], 0) + y_mean
        p_te = np.mean([r[1] for r in res_te], 0)
        dte, dcal = ds.iloc[te], ds.iloc[i_cal]
        for k, h in enumerate(HS):
            yc = Y[i_cal, k]
            A = np.c_[np.ones(len(yc)), mu_cal[:, k]]
            beta, *_ = np.linalg.lstsq(A, yc, rcond=None)
            s2 = np.var(yc - A @ beta)
            var_te = np.asarray(denorm(beta[0] + beta[1] * mu_te[:, k] + 0.5 * s2, dte, h))
            var_cal = np.asarray(denorm(beta[0] + beta[1] * mu_cal[:, k] + 0.5 * s2, dcal, h))
            vol_rows.append(pd.DataFrame({"origin": dte.index, "h": h, "model": tag, "var": var_te, "y": dte[f"y_rv_{h}"].values}))
            touch_rows.append(touch_long(dte, var_te, h, f"{tag}_bm"))
            for j, dname in enumerate(("up", "dn")):
                col = slice((2 * k + j) * nthr, (2 * k + j + 1) * nthr)
                bs_ = bu if dname == "up" else bd
                for name, pc, pt in ((f"{tag}_touch_iso", p_cal[:, col], p_te[:, col]),
                                     (f"{tag}_bm_iso", np.stack([bm_touch(var_cal, x) for x in bs_], 1),
                                      np.stack([bm_touch(var_te, x) for x in bs_], 1))):
                    iso = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip").fit(pc.ravel(), T[i_cal, col].ravel())
                    for m_, thr in enumerate(C.THRESHOLDS):
                        touch_rows.append(pd.DataFrame({"origin": dte.index, "h": h, "dir": dname, "thr": thr, "model": name,
                                                        "p": iso.predict(pt[:, m_]), "hit": T[te, col][:, m_].astype(np.int8)}))
        if not quiet:
            print(f"  Block ab {ms:%Y-%m}: fit {len(i_fit):,} / es {len(i_es):,} / kal {len(i_cal):,} / test {len(te):,} "
                  f"({time.time() - t0:.0f}s, {sum(p.numel() for p in model.parameters()) / 1e6:.2f} Mio. Parameter)")
    return pd.concat(vol_rows, ignore_index=True), pd.concat(touch_rows, ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["valid", "test"], default="valid")
    ap.add_argument("--refit-months", type=int, default=3)
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--quick", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--tag", default="vf")
    a = ap.parse_args()
    print(f"Gerät: {DEV}")
    cfg = dict(DEFAULT)
    best_path = C.RESULTS / f"volnet_best_{a.tag}.json"
    if a.quick:
        cfg.update(ctx=24, d=32, layers=1, heads=2, ff=64, steps_per_epoch=20, epochs=2, seeds=1)
    if a.search:
        assert a.mode == "valid", "Suche nur auf dem Validierungszeitraum!"
        grid = [dict(), dict(layers=6, d=160, heads=5, ff=640), dict(ctx=336), dict(lam_touch=0.3)]
        res = []
        for g in grid:
            c = dict(cfg, **g)
            if a.quick:
                c.update(ctx=min(c["ctx"], 24), d=32, layers=1, heads=2, ff=64, steps_per_epoch=20, epochs=2)
            vp, tp = run("valid", dict(c, seeds=1), max(a.refit_months, 6), tag=a.tag, quiet=True)
            sc, ql, br = score(vp, tp, a.tag)
            res.append({**g, "score": sc, "QLIKE": ql, "Brier": br})
            print(f"  {g or 'Standard'}: Score {sc:.4f} (QLIKE {ql:.4f}, Brier {br:.5f})")
        R = pd.DataFrame(res).sort_values("score")
        R.to_csv(C.RESULTS / f"volnet_search_{a.tag}.csv", index=False)
        best = {k: v for k, v in R.iloc[0].items() if k in DEFAULT and pd.notna(v)}
        cfg.update({k: (int(v) if isinstance(DEFAULT[k], int) else float(v)) for k, v in best.items()})
        best_path.write_text(json.dumps(cfg))
        print(f"Bestes Setup: {cfg}")
    elif best_path.exists() and not a.quick:
        cfg = json.loads(best_path.read_text())
        print(f"Setup aus {best_path.name}: {cfg}")
    vp, tp = run(a.mode, cfg, a.refit_months, tag=a.tag)
    report(vp, tp, a.mode, cfg, a.tag)
