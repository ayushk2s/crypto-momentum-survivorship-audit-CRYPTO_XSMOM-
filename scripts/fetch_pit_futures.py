"""Point-in-time Binance USDT-M perpetual dataset, INCLUDING delisted contracts.

Fixes the survivorship bias of the original 30-coin universe (which was today's
survivors only). Source: data.binance.vision monthly archives (daily klines +
funding), which keep files for delisted symbols. The current month is topped up
from the live REST API for symbols still trading.

Output (data_cache/um/):
  klines_1d.parquet   long table: symbol, date, open, high, low, close, quote_volume
  funding.parquet     long table: symbol, ts, fundingRate
"""
from __future__ import annotations

import io
import re
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_cache" / "um"
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
VISION = "https://data.binance.vision/"
FAPI = "https://fapi.binance.com/fapi/v1"
START_MONTH = "2020-09"  # gives lookback history before the 2021-01 test start
STABLE = re.compile(r"^(USDC|BUSD|TUSD|FDUSD|USDP|DAI|UST|EUR|GBP|AEUR|USDE|BTCDOM|DEFI|FOOTBALL|BLUEBIRD)")

session = requests.Session()
session.mount("https://", requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=3))


def s3_list(prefix: str, folders: bool) -> list[str]:
    out, marker = [], ""
    while True:
        r = session.get(S3, params={"delimiter": "/", "prefix": prefix, "marker": marker}, timeout=30)
        r.raise_for_status()
        tag = "Prefix" if folders else "Key"
        block = r.text.split("<CommonPrefixes>") if folders else r.text.split("<Contents>")
        items = [re.search(f"<{tag}>(.*?)</{tag}>", b).group(1) for b in block[1:]]
        out += items
        if "<IsTruncated>true</IsTruncated>" not in r.text:
            return out
        m = re.search(r"<NextMarker>(.*?)</NextMarker>", r.text)
        marker = m.group(1) if m else items[-1]


def read_zip_csv(url: str, names: list[str]) -> pd.DataFrame | None:
    r = session.get(url, timeout=60)
    if r.status_code != 200:
        return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        raw = z.read(z.namelist()[0]).decode()
    df = pd.read_csv(io.StringIO(raw), header=None)
    if not str(df.iloc[0, 0]).lstrip("-").isdigit():  # some files have a header row
        df = df.iloc[1:]
    df = df.iloc[:, : len(names)]
    df.columns = names
    return df


def symbol_klines(sym: str) -> pd.DataFrame | None:
    keys = [k for k in s3_list(f"data/futures/um/monthly/klines/{sym}/1d/", folders=False)
            if k.endswith(".zip") and k[-11:-4] >= START_MONTH]
    frames = []
    for k in keys:
        d = read_zip_csv(VISION + k, ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume"])
        if d is not None:
            frames.append(d)
    if not frames:
        return None
    df = pd.concat(frames)
    df["date"] = pd.to_datetime(df.open_time.astype("int64"), unit="ms").dt.normalize()
    df["symbol"] = sym
    for c in ("open", "high", "low", "close", "quote_volume"):
        df[c] = df[c].astype(float)
    return df[["symbol", "date", "open", "high", "low", "close", "quote_volume"]]


def symbol_funding(sym: str) -> pd.DataFrame | None:
    keys = [k for k in s3_list(f"data/futures/um/monthly/fundingRate/{sym}/", folders=False)
            if k.endswith(".zip") and k[-11:-4] >= START_MONTH]
    frames = [d for k in keys if (d := read_zip_csv(VISION + k, ["ts", "interval", "fundingRate"])) is not None]
    if not frames:
        return None
    df = pd.concat(frames)
    df["ts"] = pd.to_datetime(df.ts.astype("int64"), unit="ms")
    df["fundingRate"] = df.fundingRate.astype(float)
    df["symbol"] = sym
    return df[["symbol", "ts", "fundingRate"]]


def live_topup(sym: str, since: pd.Timestamp) -> pd.DataFrame | None:
    """Current (incomplete) month is not archived yet -- pull it from REST."""
    r = session.get(f"{FAPI}/klines", params={"symbol": sym, "interval": "1d",
                                               "startTime": int(since.timestamp() * 1000), "limit": 100}, timeout=30)
    if r.status_code != 200 or not r.json():
        return None
    df = pd.DataFrame(r.json()).iloc[:, :8]
    df.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume"]
    df["date"] = pd.to_datetime(df.open_time, unit="ms").dt.normalize()
    df["symbol"] = sym
    for c in ("open", "high", "low", "close", "quote_volume"):
        df[c] = df[c].astype(float)
    df = df[df.close_time < pd.Timestamp.now(tz="UTC").timestamp() * 1000]  # completed bars only
    return df[["symbol", "date", "open", "high", "low", "close", "quote_volume"]]


def live_funding(sym: str, since: pd.Timestamp) -> pd.DataFrame | None:
    r = session.get(f"{FAPI}/fundingRate", params={"symbol": sym, "startTime": int(since.timestamp() * 1000), "limit": 1000}, timeout=30)
    if r.status_code != 200 or not r.json():
        return None
    df = pd.DataFrame(r.json())
    df["ts"] = pd.to_datetime(df.fundingTime, unit="ms")
    df["fundingRate"] = df.fundingRate.astype(float)
    return df[["symbol", "ts", "fundingRate"]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    syms = [p.rstrip("/").split("/")[-1] for p in s3_list("data/futures/um/monthly/klines/", folders=True)]
    syms = [s for s in syms if s.endswith("USDT") and not STABLE.match(s) and "_" not in s]
    print(f"{len(syms)} USDT-M perpetual symbols in archive (live + delisted)", flush=True)

    with ThreadPoolExecutor(24) as ex:
        kl = [d for d in ex.map(symbol_klines, syms) if d is not None]
    k = pd.concat(kl, ignore_index=True)
    last = k.date.max()
    print(f"archived klines: {k.symbol.nunique()} symbols, {len(k)} rows, to {last.date()}", flush=True)
    since = last + pd.Timedelta(days=1)
    with ThreadPoolExecutor(8) as ex:
        top = [d for d in ex.map(lambda s: live_topup(s, since), sorted(k.symbol.unique())) if d is not None]
    if top:
        k = pd.concat([k] + top, ignore_index=True)
    k = k.drop_duplicates(["symbol", "date"]).sort_values(["symbol", "date"])
    k.to_parquet(OUT / "klines_1d.parquet", index=False)
    print(f"klines saved: {k.symbol.nunique()} symbols to {k.date.max().date()}", flush=True)

    # funding only for symbols that are ever liquid enough to enter a top-60 universe
    dv = k.pivot_table(index="date", columns="symbol", values="quote_volume").rolling(30, min_periods=20).median()
    ever = sorted(dv.rank(axis=1, ascending=False).le(60).any().loc[lambda s: s].index)
    print(f"fetching funding for {len(ever)} symbols that ever rank top-60 by volume", flush=True)
    with ThreadPoolExecutor(24) as ex:
        fl = [d for d in ex.map(symbol_funding, ever) if d is not None]
    f = pd.concat(fl, ignore_index=True)
    fsince = f.ts.max().normalize() + pd.Timedelta(days=1)
    with ThreadPoolExecutor(4) as ex:
        ftop = [d for d in ex.map(lambda s: live_funding(s, fsince), ever) if d is not None]
    f = pd.concat([f] + ftop, ignore_index=True).drop_duplicates(["symbol", "ts"]).sort_values(["symbol", "ts"])
    f.to_parquet(OUT / "funding.parquet", index=False)
    print(f"funding saved: {f.symbol.nunique()} symbols, {len(f)} rows to {f.ts.max()}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
