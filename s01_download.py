"""Schritt 1: Binance-Daten von data.binance.vision laden.

  python s01_download.py                 # Spot 1m-Klines ab config.DOWNLOAD_START
  python s01_download.py --futures       # zusätzlich Futures-Metrics (Open Interest) + Funding

Ergebnis:
  data/klines_1m.parquet          Spot BTCUSDT 1m (Polymarket-Auflösungsquelle)
  data/fut_metrics.parquet        (optional) Open Interest / Long-Short, 5m
  data/fut_funding.parquet        (optional) Funding Rate

Bereits geladene Dateien werden übersprungen -> Skript kann jederzeit neu gestartet werden.
"""
import argparse
import hashlib
import io
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import numpy as np
import pandas as pd

import config as C

BASE = "https://data.binance.vision/data"
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def http_get(url, retries=4):
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            err = e
        except Exception as e:  # Netzwerkfehler -> erneut versuchen
            err = e
        time.sleep(2 ** i)
    raise RuntimeError(f"Download fehlgeschlagen: {url} ({err})")


def fetch(url, dest):
    """Zip laden, per SHA256 prüfen, speichern. Gibt Pfad oder None (nicht vorhanden) zurück."""
    if dest.exists():
        return dest
    blob = http_get(url)
    if blob is None:
        return None
    chk = http_get(url + ".CHECKSUM")
    if chk:
        expected = chk.decode().split()[0]
        if hashlib.sha256(blob).hexdigest() != expected:
            raise RuntimeError(f"Checksumme falsch: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(blob)
    tmp.rename(dest)
    return dest


def months(start, end):
    p = pd.Period(start, "M")
    while p <= pd.Period(end, "M"):
        yield p
        p += 1


def plan_files(kind_monthly, kind_daily, fname, start):
    """Volle Monate als Monatsdatei, laufenden Monat (+ fehlende Vormonate) als Tagesdateien."""
    today = date.today()
    last_full = (pd.Period(today, "M") - 1)
    jobs = []
    for m in months(start, last_full):
        jobs.append(("m", m, f"{BASE}/{kind_monthly}/{fname(str(m))}.zip"))
    d = date(today.year, today.month, 1)
    while d < today:
        jobs.append(("d", d, f"{BASE}/{kind_daily}/{fname(d.isoformat())}.zip"))
        d += timedelta(days=1)
    return jobs


def download(jobs, subdir, workers=8):
    paths, missing_months = [], []
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch, url, C.RAW / subdir / url.rsplit("/", 1)[1]): (k, t, url) for k, t, url in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            k, t, url = futs[f]
            p = f.result()
            if p is None:
                if k == "m":
                    missing_months.append(t)
            else:
                paths.append(p)
            if i % 25 == 0 or i == len(futs):
                print(f"  {subdir}: {i}/{len(futs)} Dateien")
    return sorted(paths), missing_months


def read_zip_csv(path, names=None):
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0])
    first = raw[:200].decode(errors="ignore").split("\n", 1)[0]
    has_header = any(c.isalpha() for c in first.split(",")[0])
    df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None)
    if names is not None:
        df = df.iloc[:, :len(names)]
        df.columns = names
    return df


def to_datetime_auto(x):
    """Binance-Spot nutzt ab 2025 Mikrosekunden statt Millisekunden."""
    x = pd.to_numeric(x).astype("int64")
    us = x > 10**14
    ms = np.where(us, x // 1000, x)
    return pd.to_datetime(ms, unit="ms", utc=True)


def klines(start):
    sym = C.SYMBOL
    jobs = plan_files(f"spot/monthly/klines/{sym}/1m", f"spot/daily/klines/{sym}/1m",
                      lambda t: f"{sym}-1m-{t}", start)
    paths, missing = download(jobs, "spot_klines_1m")
    # Fehlt eine Monatsdatei (z. B. Monat gerade erst vorbei), Tagesdateien nachladen
    extra = []
    for m in missing:
        d = m.start_time.date()
        while d <= m.end_time.date():
            extra.append(("d", d, f"{BASE}/spot/daily/klines/{sym}/1m/{sym}-1m-{d.isoformat()}.zip"))
            d += timedelta(days=1)
    if extra:
        p2, _ = download(extra, "spot_klines_1m")
        paths += p2
    print(f"Lese {len(paths)} Dateien ...")
    df = pd.concat([read_zip_csv(p, KLINE_COLS) for p in paths], ignore_index=True)
    df["time"] = to_datetime_auto(df["open_time"])
    df = df.drop(columns=["open_time", "close_time", "ignore"])
    num = ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]
    df[num] = df[num].astype("float64")
    df = df.drop_duplicates("time").sort_values("time").set_index("time")
    out = C.DATA / "klines_1m.parquet"
    df.to_parquet(out)
    full = pd.date_range(df.index[0], df.index[-1], freq="1min")
    gap = 1 - len(df) / len(full)
    print(f"OK: {out}  {len(df):,} Zeilen  {df.index[0]} bis {df.index[-1]}  fehlende Minuten: {gap:.3%}")


def futures(start):
    sym = C.SYMBOL
    # Funding: Monatsdateien
    jobs = [("m", m, f"{BASE}/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{m}.zip")
            for m in months(start, pd.Period(date.today(), "M") - 1)]
    paths, _ = download(jobs, "fut_funding")
    if paths:
        df = pd.concat([read_zip_csv(p) for p in paths], ignore_index=True)
        df.columns = [c.lower() for c in df.columns]
        tcol = df.columns[0]
        df["time"] = to_datetime_auto(df[tcol])
        rate = [c for c in df.columns if "rate" in c][0]
        df = df[["time", rate]].rename(columns={rate: "funding_rate"}).drop_duplicates("time").sort_values("time")
        df.set_index("time").to_parquet(C.DATA / "fut_funding.parquet")
        print(f"OK: Funding {len(df):,} Zeilen")
    # Metrics (Open Interest etc.): nur Tagesdateien
    s = max(pd.Timestamp(start + "-01"), pd.Timestamp("2021-12-01")).date()
    jobs, d = [], s
    while d < date.today():
        jobs.append(("d", d, f"{BASE}/futures/um/daily/metrics/{sym}/{sym}-metrics-{d.isoformat()}.zip"))
        d += timedelta(days=1)
    paths, _ = download(jobs, "fut_metrics", workers=16)
    if paths:
        df = pd.concat([read_zip_csv(p) for p in paths], ignore_index=True)
        df.columns = [c.lower() for c in df.columns]
        df["time"] = pd.to_datetime(df[df.columns[0]], utc=True)
        keep = [c for c in df.columns if c.startswith(("sum_open_interest", "count_", "sum_toptrader", "sum_taker"))]
        df = df[["time"] + keep].drop_duplicates("time").sort_values("time").set_index("time")
        df = df.apply(pd.to_numeric, errors="coerce")
        df.to_parquet(C.DATA / "fut_metrics.parquet")
        print(f"OK: Metrics {len(df):,} Zeilen, Spalten: {keep}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=C.DOWNLOAD_START)
    ap.add_argument("--futures", action="store_true")
    a = ap.parse_args()
    klines(a.start)
    if a.futures:
        futures(a.start)
