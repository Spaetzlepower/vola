"""Papierhandels-Bot: Polymarket-BTC-Märkte live gegen fairen Wert – OHNE echte Orders.

  pip install websockets
  python bot/paper_bot.py                      # Standard: stündliche "above"-Märkte + Wochen-"hit"-Markt
  python bot/paper_bot.py --edge 0.05 --max-size 50 --fee-bps 0
  python bot/paper_bot.py --selftest           # Offline-Test der Rechenlogik

Was er tut (alles nur lesen + protokollieren):
  - Binance BTCUSDT live (Websocket bookTicker) -> aktueller Kurs; Vola laufend aus 1-Min.-Renditen (EWMA,
    mit Tagesprofil aus data/seasonality.parquet, falls vorhanden)
  - Polymarket-Orderbücher live (Websocket market channel) für die aktiven Märkte
  - fairer Wert je Stufe: stündlich = P(Endkurs > K), Woche = P(Berührung) (bereits berührt -> 1)
  - Signal, wenn Ask < fair − edge (Yes kaufen) oder Bid > fair + edge (Yes verkaufen = No kaufen);
    "Ausführung" zum angezeigten Preis, max. angezeigte Menge bzw. --max-size; 60 s Pause je Stufe
  - Abrechnung bei Auflösung mit dem Binance-1m-Schlusskurs (stündlich) bzw. Berührung (Woche)
Ausgaben in bot/logs/: signals.csv (jeder Papier-Trade), settled.csv (abgerechnet), snapshots.csv (Bid/Ask/fair
alle 10 s, die fehlenden echten Orderbuch-Daten für spätere Auswertungen), Zusammenfassung alle 5 Min. im Terminal.

Wichtig: Nur von einem Ort laufen lassen, an dem Polymarket erlaubt ist. Echte Orders sind bewusst NICHT eingebaut.
"""
import argparse
import asyncio
import csv
import json
import math
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

GAMMA = "https://gamma-api.polymarket.com"
PM_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
BN_WS = ["wss://stream.binance.com:9443/ws/btcusdt@bookTicker", "wss://data-stream.binance.vision/ws/btcusdt@bookTicker"]
BN_REST = ["https://api.binance.com", "https://data-api.binance.vision"]
NY = ZoneInfo("America/New_York")
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]
LOG = Path(__file__).resolve().parent / "logs"


# ---------------- Mathematik ----------------

def ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def fair_above(S, K, var):
    """P(S_T > K), driftlos, Varianz der log-Rendite bis Auflösung."""
    if var <= 0:
        return 1.0 if S > K else 0.0
    s = math.sqrt(var)
    return ncdf((math.log(S / K) - 0.5 * var) / s)


def fair_touch(S, K, var, up, touched):
    if touched:
        return 1.0
    b = math.log(K / S) if up else math.log(S / K)
    if b <= 0:
        return 1.0
    if var <= 0:
        return 0.0
    return min(1.0, 2 * (1 - ncdf(b / math.sqrt(var))))


# ---------------- Hilfen ----------------

def http_json(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (paper-bot)", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def binance(path):
    last = None
    for base in BN_REST:
        try:
            return http_json(base + path)
        except Exception as e:
            last = e
    raise RuntimeError(f"Binance REST nicht erreichbar: {last}")


def seasonality():
    """Varianz-Profil je 5-Min.-Slot der Woche (Mittel 1), falls vorhanden."""
    p = Path(__file__).resolve().parents[1] / "data" / "seasonality.parquet"
    try:
        import pandas as pd
        return pd.read_parquet(p)["seas"].values.tolist()
    except Exception:
        return [1.0] * 2016


def slot(ts):
    d = datetime.fromtimestamp(ts, timezone.utc)
    return d.weekday() * 288 + d.hour * 12 + d.minute // 5


class Vol:
    """EWMA der saisonbereinigten 1-Min.-Varianz; erwartete Varianz über [t, T] mit Tagesprofil."""

    def __init__(self, seas, halflife_min=360):
        self.seas, self.lam = seas, 0.5 ** (1 / halflife_min)
        self.v, self.last_px, self.last_min = None, None, None

    def seed(self, closes_with_ts):
        for ts, px in closes_with_ts:
            self.update_minute(ts, px)

    def update_minute(self, ts, px):
        m = int(ts // 60)
        if self.last_min is not None and m > self.last_min and self.last_px:
            r2 = math.log(px / self.last_px) ** 2 / max(self.seas[slot(ts)], 0.05)
            self.v = r2 if self.v is None else self.lam * self.v + (1 - self.lam) * r2
        if self.last_min is None or m > self.last_min:
            self.last_min, self.last_px = m, px

    def var_until(self, now, T):
        if self.v is None or T <= now:
            return 0.0
        tot, t = 0.0, now
        while t < T:                                        # in 5-Min.-Schritten aufsummieren
            step = min(300, T - t)
            tot += self.v * self.seas[slot(t)] * step / 60
            t += step
        return tot


# ---------------- Märkte finden ----------------

def hourly_slug(dt_utc):
    et = dt_utc.astimezone(NY)
    h = et.hour % 12 or 12
    return f"bitcoin-above-on-{MONTHS[et.month - 1]}-{et.day}-{et.year}-{h}{'am' if et.hour < 12 else 'pm'}-et", et


def weekly_slugs(now_utc):
    et = now_utc.astimezone(NY)
    mon = (et - timedelta(days=et.weekday())).date()
    sun = mon + timedelta(days=6)
    m1, m2 = MONTHS[mon.month - 1], MONTHS[sun.month - 1]
    base = "what-price-will-bitcoin-hit"
    out = []
    for y in (f"-{sun.year}", ""):
        out += [f"{base}-{m1}-{mon.day}-{sun.day}{y}"] if m1 == m2 else [f"{base}-{m1}-{mon.day}-{m2}-{sun.day}{y}",
                                                                       f"{base}-{m1}-{mon.day}-{sun.day}{y}"]
    return out


def parse_markets(ev, kind, T):
    import re
    res = []
    for m in ev.get("markets", []):
        if m.get("closed"):
            continue
        txt = f"{m.get('groupItemTitle') or ''} {m.get('question', '')}"
        nums = [float(x.replace(",", "")) for x in re.findall(r"(\d{1,3}(?:,\d{3})+|\d{5,7})", txt)]
        if not nums:
            continue
        toks = json.loads(m["clobTokenIds"]) if isinstance(m.get("clobTokenIds"), str) else m.get("clobTokenIds")
        outs = json.loads(m["outcomes"]) if isinstance(m.get("outcomes"), str) else m.get("outcomes", ["Yes", "No"])
        yi = [o.lower() for o in outs].index("yes") if "yes" in [o.lower() for o in outs] else 0
        up = None
        if kind == "week":
            low = txt.lower()
            up = not ("↓" in txt or any(w in low for w in ("dip", "below", "drop", "fall")))
        res.append({"token": toks[yi], "strike": nums[0], "kind": kind, "up": up, "T": T, "slug": ev.get("slug"),
                    "start": datetime.fromisoformat(ev["startDate"].replace("Z", "+00:00")).timestamp()
                    if ev.get("startDate") else None})
    return res


def discover(hours_ahead=6):
    now = datetime.now(timezone.utc)
    found = []
    base = now.replace(minute=0, second=0, microsecond=0)
    for k in range(0, hours_ahead + 1):
        slug, et = hourly_slug(base + timedelta(hours=k))
        try:
            ev = http_json(f"{GAMMA}/events?slug={slug}")
        except Exception:
            ev = None
        if ev:
            T = (base + timedelta(hours=k)).timestamp()
            found += parse_markets(ev[0], "hour", T)
    for s in weekly_slugs(now):
        try:
            ev = http_json(f"{GAMMA}/events?slug={s}")
        except Exception:
            ev = None
        if ev:
            T = datetime.fromisoformat(ev[0]["endDate"].replace("Z", "+00:00")).timestamp()
            found += parse_markets(ev[0], "week", T)
            break
    return found


# ---------------- Bot ----------------

class Bot:
    def __init__(self, a):
        self.a = a
        self.S, self.S_ts = None, 0
        self.vol = Vol(seasonality())
        self.markets, self.books = {}, {}                  # token -> Markt; token -> {"bids":{p:s}, "asks":{p:s}}
        self.hi, self.lo = {}, {}                          # Wochenfenster: laufendes Hoch/Tief
        self.cool, self.positions = {}, []
        LOG.mkdir(parents=True, exist_ok=True)
        self.f_sig = self._csv("signals.csv", ["time", "slug", "kind", "strike", "up", "side", "price", "size", "fair",
                                               "edge", "S", "sigma_rest", "tau_min", "bid", "ask"])
        self.f_set = self._csv("settled.csv", ["time", "slug", "strike", "side", "price", "size", "outcome", "pnl"])
        self.f_snap = self._csv("snapshots.csv", ["time", "token", "slug", "kind", "strike", "bid", "ask", "bid_sz",
                                                  "ask_sz", "fair", "S", "tau_min"])
        self.pnl, self.n_trades = 0.0, 0

    def _csv(self, name, header):
        p = LOG / name
        new = not p.exists()
        f = open(p, "a", newline="")
        w = csv.writer(f)
        if new:
            w.writerow(header)
        return (f, w)

    def write(self, fw, row):
        fw[1].writerow(row); fw[0].flush()

    # --- Kurs ---
    def on_binance(self, msg):
        b, a_ = float(msg["b"]), float(msg["a"])
        self.S, self.S_ts = (a_ + b) / 2, time.time()
        self.vol.update_minute(self.S_ts, self.S)
        for tok, m in self.markets.items():
            if m["kind"] == "week":
                self.hi[tok] = max(self.hi.get(tok, self.S), self.S)
                self.lo[tok] = min(self.lo.get(tok, self.S), self.S)

    # --- Orderbuch ---
    def on_pm(self, msg):
        for ev in (msg if isinstance(msg, list) else [msg]):
            et = ev.get("event_type")
            if et == "book":
                tok = ev["asset_id"]
                self.books[tok] = {"bids": {float(x["price"]): float(x["size"]) for x in ev.get("bids", [])},
                                   "asks": {float(x["price"]): float(x["size"]) for x in ev.get("asks", [])}}
                self.check(tok)
            elif et == "price_change":
                for ch in ev.get("price_changes", []):
                    tok = ch["asset_id"]
                    bk = self.books.setdefault(tok, {"bids": {}, "asks": {}})
                    side = "bids" if ch.get("side", "").upper() == "BUY" else "asks"
                    p, s = float(ch["price"]), float(ch["size"])
                    if s == 0:
                        bk[side].pop(p, None)
                    else:
                        bk[side][p] = s
                    self.check(tok)

    def best(self, tok):
        bk = self.books.get(tok)
        if not bk or not bk["bids"] or not bk["asks"]:
            return None
        bid = max(bk["bids"]); ask = min(bk["asks"])
        return bid, bk["bids"][bid], ask, bk["asks"][ask]

    def fair(self, tok, now):
        m = self.markets[tok]
        if self.S is None or now - self.S_ts > 10:
            return None, None
        if m["kind"] == "hour":
            T_close = m["T"] + 60                            # Schlusskurs der Minute T
            var = self.vol.var_until(now, T_close)
            return fair_above(self.S, m["strike"], var), var
        var = self.vol.var_until(now, m["T"])
        touched = (self.hi.get(tok, 0) >= m["strike"]) if m["up"] else (self.lo.get(tok, 1e12) <= m["strike"])
        return fair_touch(self.S, m["strike"], var, m["up"], touched), var

    def check(self, tok):
        if tok not in self.markets:
            return
        now = time.time()
        m = self.markets[tok]
        if now >= m["T"] - self.a.min_tau * 60:
            return
        bb = self.best(tok)
        fv, var = self.fair(tok, now)
        if bb is None or fv is None or now - self.cool.get(tok, 0) < 60:
            return
        bid, bsz, ask, asz = bb
        fee = self.a.fee_bps / 1e4
        side = None
        if ask < fv - self.a.edge - fee and ask < 0.99:
            side, price, size = "BUY_YES", ask, min(asz, self.a.max_size)
        elif bid > fv + self.a.edge + fee and bid > 0.01:
            side, price, size = "SELL_YES", bid, min(bsz, self.a.max_size)
        if side:
            self.cool[tok] = now
            tau = (m["T"] - now) / 60
            self.positions.append({**m, "side": side, "price": price, "size": size, "t": now})
            self.n_trades += 1
            self.write(self.f_sig, [datetime.now(timezone.utc).isoformat(), m["slug"], m["kind"], m["strike"], m["up"],
                                    side, price, size, round(fv, 4), round(abs(fv - price), 4), round(self.S, 2),
                                    round(math.sqrt(max(var, 0)), 5), round(tau, 1), bid, ask])

    # --- Abrechnung ---
    def settle(self):
        now = time.time()
        keep = []
        for p in self.positions:
            out = None
            if p["kind"] == "hour" and now > p["T"] + 120:
                try:
                    k = binance(f"/api/v3/klines?symbol=BTCUSDT&interval=1m&startTime={int(p['T'] * 1000)}&limit=1")
                    out = 1.0 if float(k[0][4]) > p["strike"] else 0.0
                except Exception:
                    pass
            elif p["kind"] == "week":
                tok = p["token"]
                touched = (self.hi.get(tok, 0) >= p["strike"]) if p["up"] else (self.lo.get(tok, 1e12) <= p["strike"])
                if touched:
                    out = 1.0
                elif now > p["T"] + 120:
                    out = 0.0
            if out is None:
                keep.append(p); continue
            fee = self.a.fee_bps / 1e4
            pnl = ((out - p["price"]) if p["side"] == "BUY_YES" else (p["price"] - out)) * p["size"] - fee * p["size"]
            self.pnl += pnl
            self.write(self.f_set, [datetime.now(timezone.utc).isoformat(), p["slug"], p["strike"], p["side"],
                                    p["price"], p["size"], out, round(pnl, 4)])
        self.positions = keep

    # --- Schleifen ---
    async def binance_loop(self):
        import websockets
        i = 0
        while True:
            url = BN_WS[i % len(BN_WS)]
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    print(f"Binance verbunden ({url})")
                    async for raw in ws:
                        self.on_binance(json.loads(raw))
            except Exception as e:
                print(f"Binance-Verbindung weg ({e}), neuer Versuch ...")
                i += 1
                await asyncio.sleep(3)

    async def pm_loop(self):
        import websockets
        while True:
            toks = list(self.markets)
            if not toks:
                await asyncio.sleep(10); continue
            try:
                async with websockets.connect(PM_WS, ping_interval=None) as ws:
                    await ws.send(json.dumps({"assets_ids": toks, "type": "market"}))
                    print(f"Polymarket verbunden, {len(toks)} Stufen abonniert")
                    subscribed = set(toks)

                    async def heartbeat():
                        while True:
                            await asyncio.sleep(10)
                            await ws.send("PING")
                    hb = asyncio.create_task(heartbeat())
                    try:
                        while True:
                            raw = await asyncio.wait_for(ws.recv(), timeout=60)
                            if raw == "PONG":
                                continue
                            self.on_pm(json.loads(raw))
                            if set(self.markets) != subscribed:            # neue Märkte -> neu verbinden
                                break
                    finally:
                        hb.cancel()
            except Exception as e:
                print(f"Polymarket-Verbindung weg ({e}), neuer Versuch ...")
                await asyncio.sleep(3)

    async def housekeeping(self):
        last_disc, last_sum, last_snap = 0, time.time(), 0
        while True:
            now = time.time()
            if now - last_disc > 600:
                try:
                    found = await asyncio.to_thread(discover, self.a.hours_ahead)
                    self.markets = {m["token"]: m for m in found if m["T"] > now}
                    for m in found:
                        if m["kind"] == "week" and m["token"] not in self.hi:
                            await asyncio.to_thread(self.seed_week, m)
                    print(f"{datetime.now():%H:%M} Märkte: {sum(m['kind'] == 'hour' for m in self.markets.values())} stündlich, "
                          f"{sum(m['kind'] == 'week' for m in self.markets.values())} Woche")
                except Exception as e:
                    print(f"Markt-Suche fehlgeschlagen: {e}")
                last_disc = now
            if now - last_snap > 10:
                for tok, m in list(self.markets.items()):
                    bb, (fv, _) = self.best(tok), self.fair(tok, now)
                    if bb and fv is not None:
                        self.write(self.f_snap, [datetime.now(timezone.utc).isoformat(), tok, m["slug"], m["kind"], m["strike"],
                                                 bb[0], bb[2], bb[1], bb[3], round(fv, 4), round(self.S, 2), round((m["T"] - now) / 60, 1)])
                last_snap = now
            self.settle()
            if now - last_sum > 300:
                print(f"{datetime.now():%H:%M} Papier-Trades: {self.n_trades}, offen: {len(self.positions)}, "
                      f"abgerechneter PnL: {self.pnl:+.2f} $  (S={self.S}, sigma/h={math.sqrt(max(self.vol.var_until(now, now + 3600), 0)):.4f})")
                last_sum = now
            await asyncio.sleep(2)

    def seed_week(self, m):
        """Hoch/Tief seit Wochenbeginn aus Binance-1m-Kerzen (Wochenmärkte: Berührung zählt ab Start)."""
        start = m["start"] or (m["T"] - 7 * 86400)
        hi, lo, t = 0.0, 1e12, int(start * 1000)
        while t < time.time() * 1000:
            k = binance(f"/api/v3/klines?symbol=BTCUSDT&interval=1m&startTime={t}&limit=1000")
            if not k:
                break
            hi = max(hi, max(float(x[2]) for x in k)); lo = min(lo, min(float(x[3]) for x in k))
            t = int(k[-1][0]) + 60000
        self.hi[m["token"]], self.lo[m["token"]] = hi, lo

    def seed_vol(self):
        closes, t = [], int((time.time() - 3 * 86400) * 1000)
        while t < time.time() * 1000:
            k = binance(f"/api/v3/klines?symbol=BTCUSDT&interval=1m&startTime={t}&limit=1000")
            if not k:
                break
            closes += [(int(x[0]) / 1000 + 60, float(x[4])) for x in k]
            t = int(k[-1][0]) + 60000
        self.vol.seed(closes)
        print(f"Vola initialisiert aus {len(closes)} Minuten: sigma/h ≈ "
              f"{math.sqrt(self.vol.var_until(time.time(), time.time() + 3600)):.4f}")

    async def run(self):
        await asyncio.to_thread(self.seed_vol)
        await asyncio.gather(self.binance_loop(), self.pm_loop(), self.housekeeping())


def selftest():
    assert abs(fair_above(100, 100, 1e-4) - 0.498) < 0.01
    assert fair_above(101, 100, 1e-6) > 0.99 and fair_above(99, 100, 1e-6) < 0.01
    assert fair_touch(100, 101, 0.0, True, True) == 1.0
    p = fair_touch(100, 102, 0.0004, True, False)
    assert 0.3 < p < 0.4, p                                 # 2*(1-Phi(0.0198/0.02)) ≈ 0.32
    v = Vol([1.0] * 2016)
    px, ts = 100.0, 1_700_000_000
    import random
    random.seed(1)
    for i in range(3000):
        px *= math.exp(random.gauss(0, 0.001)); v.update_minute(ts + 60 * i, px)
    est = v.var_until(ts, ts + 3600)
    assert 0.6 * 60e-6 < est < 1.6 * 60e-6, est             # 60 Min. x (0.001)^2
    class A: edge, max_size, fee_bps, min_tau, hours_ahead = 0.05, 10, 0, 2, 6
    b = Bot(A())
    b.S, b.S_ts = 100.0, time.time()
    b.vol = v
    tok = "t1"
    b.markets[tok] = {"token": tok, "strike": 100.0, "kind": "hour", "up": None, "T": time.time() + 3600, "slug": "x", "start": None}
    b.on_pm({"event_type": "book", "asset_id": tok, "bids": [{"price": "0.30", "size": "100"}],
             "asks": [{"price": "0.35", "size": "40"}]})
    assert b.n_trades == 1 and b.positions[0]["side"] == "BUY_YES", b.positions   # fair ≈ 0.5 > 0.35 + 0.05
    b.on_pm([{"event_type": "price_change", "price_changes": [{"asset_id": tok, "price": "0.35", "size": "0", "side": "SELL"}]}])
    print("Selbsttest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", type=float, default=0.05, help="Mindestabstand zum fairen Wert (nach Gebühren)")
    ap.add_argument("--max-size", type=float, default=50, help="max. Anteile je Papier-Trade")
    ap.add_argument("--fee-bps", type=float, default=0, help="Gebühr in Basispunkten je Anteil (Polymarket prüfen!)")
    ap.add_argument("--min-tau", type=float, default=2, help="keine Trades in den letzten N Minuten vor Auflösung")
    ap.add_argument("--hours-ahead", type=int, default=6)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        asyncio.run(Bot(a).run())
