"""Pull real historical funding-rate history for the full ~30-asset
universe from Binance USDT-M futures, 2021-01-01 to present."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import requests

from data_io import list_universe

START = "2021-01-01"
END = "2026-10-20"
OUT = Path(__file__).resolve().parents[1] / "reports" / "funding_rates_universe.csv"
URL = "https://fapi.binance.com/fapi/v1/fundingRate"


def fetch_symbol(symbol: str) -> pd.DataFrame:
    rows = []
    cursor = int(pd.Timestamp(START).timestamp() * 1000)
    end_ms = int(pd.Timestamp(END).timestamp() * 1000)
    while cursor < end_ms:
        params = {"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000}
        try:
            r = requests.get(URL, params=params, timeout=15)
        except requests.RequestException as e:
            print(f"  {symbol}: request error {e}, stopping")
            break
        if r.status_code != 200:
            print(f"  {symbol}: HTTP {r.status_code} -- {r.text[:150]}")
            break
        batch = r.json()
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(batch)
        cursor = batch[-1]["fundingTime"] + 1
        if len(batch) < 1000:
            break
        time.sleep(0.1)
    if not rows:
        return pd.DataFrame(columns=["symbol", "ts", "fundingRate"])
    df = pd.DataFrame(rows)[["symbol", "fundingTime", "fundingRate"]]
    df["fundingRate"] = df["fundingRate"].astype(float)
    df["ts"] = pd.to_datetime(df["fundingTime"], unit="ms")
    return df[["symbol", "ts", "fundingRate"]]


def main():
    universe = list_universe()
    frames = []
    for sym in universe:
        df = fetch_symbol(sym)
        print(f"{sym:10s} n={len(df):5d}")
        frames.append(df)
        time.sleep(0.1)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(OUT, index=False)
    print(f"\nWrote {OUT} ({len(combined)} rows, {combined.symbol.nunique()} symbols)")


if __name__ == "__main__":
    main()
