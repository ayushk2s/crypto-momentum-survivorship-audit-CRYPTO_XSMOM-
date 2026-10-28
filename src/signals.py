"""Cross-sectional momentum signal with a funding-crowding tilt.

Causal by construction: at each rebalance date t, the momentum score uses
only returns up to and including bar t-1's close; funding tilt uses only
funding events strictly before t; execution is at bar t's CLOSE (one full
day after the signal bar) -- see run_backtest's w_daily.shift(1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_price_panel(daily: dict[str, "pl.DataFrame"]) -> pd.DataFrame:
    """Wide panel of close prices, outer-joined on date, forward-filled
    only across genuine trading gaps (not before a symbol's listing)."""
    frames = {}
    for sym, df in daily.items():
        s = df.to_pandas().set_index("ts")["close"]
        frames[sym] = s
    panel = pd.DataFrame(frames).sort_index()
    return panel


def momentum_score(panel: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """Trailing lookback_days return, computed causally: score known at
    close of day t uses price[t] / price[t-lookback] - 1, i.e. no future
    data. Shifted by 1 at the call site before being used for weights."""
    return panel.pct_change(lookback_days, fill_method=None)


def funding_crowding_multiplier(funding_panel: pd.DataFrame, lookback_days: int,
                                 cap_pct_per_8h: float = 0.0005) -> pd.DataFrame:
    """Rolling mean funding rate per symbol, used to penalize crowded
    positions: a long in a very-positive-funding name, or a short in a
    very-negative-funding name, gets its weight scaled down."""
    roll = funding_panel.rolling(lookback_days, min_periods=max(3, lookback_days // 4)).mean()
    return roll.clip(-cap_pct_per_8h, cap_pct_per_8h) / cap_pct_per_8h  # in [-1, 1]


def build_weights(panel: pd.DataFrame, funding_panel: pd.DataFrame | None,
                   lookback_days: int, rebalance_every: int, n_quantile: float,
                   vol_lookback: int, funding_tilt_strength: float,
                   min_history_days: int) -> pd.DataFrame:
    """Returns a DataFrame of target portfolio weights, indexed by
    rebalance date, columns = symbols. Weight signed (+long, -short),
    gross leverage normalized to 1.0 each rebalance.
    """
    ret = panel.pct_change(fill_method=None)
    mom = momentum_score(panel, lookback_days).shift(1)  # causal: known at t using data through t-1
    vol = ret.rolling(vol_lookback).std().shift(1)
    listed_long_enough = panel.notna().rolling(min_history_days, min_periods=min_history_days).count().shift(1) >= min_history_days

    dates = panel.index[::rebalance_every]
    weights = pd.DataFrame(0.0, index=dates, columns=panel.columns)

    fund_mult = None
    if funding_panel is not None:
        fund_mult = funding_crowding_multiplier(funding_panel, lookback_days).shift(1)

    for d in dates:
        if d not in mom.index:
            continue
        row = mom.loc[d]
        eligible = row[listed_long_enough.loc[d].reindex(row.index).fillna(False)].dropna()
        if len(eligible) < 10:
            continue
        n = max(1, int(round(len(eligible) * n_quantile)))
        longs = eligible.nlargest(n).index
        shorts = eligible.nsmallest(n).index

        v = vol.loc[d].reindex(eligible.index)
        inv_vol = (1.0 / v.replace(0, np.nan)).fillna(0)

        w = pd.Series(0.0, index=panel.columns)
        if len(longs):
            lw = inv_vol.loc[longs]
            if fund_mult is not None:
                fm = fund_mult.loc[d].reindex(longs).fillna(0)
                lw = lw * (1 - funding_tilt_strength * fm.clip(lower=0))  # penalize crowded-long names
            lw = lw / lw.sum() if lw.sum() > 0 else pd.Series(1.0 / len(longs), index=longs)
            w.loc[longs] = lw.values * 0.5
        if len(shorts):
            sw = inv_vol.loc[shorts]
            if fund_mult is not None:
                fm = fund_mult.loc[d].reindex(shorts).fillna(0)
                sw = sw * (1 - funding_tilt_strength * (-fm.clip(upper=0)))  # penalize crowded-short names
            sw = sw / sw.sum() if sw.sum() > 0 else pd.Series(1.0 / len(shorts), index=shorts)
            w.loc[shorts] = -sw.values * 0.5
        weights.loc[d] = w

    return weights
