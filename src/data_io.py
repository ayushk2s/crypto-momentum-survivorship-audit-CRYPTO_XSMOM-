"""Load and resample crypto OHLCV to daily bars for the full ~30-asset
universe, for cross-sectional momentum research."""
from __future__ import annotations

from pathlib import Path

import polars as pl

RAW_ROOT = Path("C:/Users/GIGA/Documents/candlestick data/1m")
CACHE_ROOT = Path(__file__).resolve().parents[1] / "data_cache"


def list_universe() -> list[str]:
    return sorted(p.name.replace("USDT1m.csv", "") + "USDT" for p in RAW_ROOT.glob("*USDT1m.csv"))


def load_daily(symbol: str) -> pl.DataFrame:
    cache = CACHE_ROOT / f"{symbol}_1d.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    src = RAW_ROOT / f"{symbol}1m.csv"
    lf = pl.scan_csv(src).rename({"Date": "ts", "Open": "open", "High": "high",
                                   "Low": "low", "Close": "close", "Volume": "volume"})
    lf = lf.with_columns(pl.col("ts").str.to_datetime(strict=False)).sort("ts")
    daily = (
        lf.group_by_dynamic("ts", every="1d", closed="left", label="left")
        .agg(pl.col("open").first().alias("open"), pl.col("high").max().alias("high"),
             pl.col("low").min().alias("low"), pl.col("close").last().alias("close"),
             pl.col("volume").sum().alias("volume"))
        .drop_nulls()
        .collect()
    )
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    daily.write_parquet(cache)
    return daily


def load_universe_panel(universe: list[str]) -> dict[str, pl.DataFrame]:
    return {s: load_daily(s) for s in universe}
