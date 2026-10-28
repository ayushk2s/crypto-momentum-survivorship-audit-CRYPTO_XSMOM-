"""Event-driven-ish backtest: weights held constant (buy-and-hold between
rebalances) in weight-space, daily P&L = sum(weight * asset daily return),
minus turnover cost at rebalance and real funding cost/benefit every day a
position is held.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(panel: pd.DataFrame, weights: pd.DataFrame, funding_daily: pd.DataFrame | None,
                  cost_bps_per_turnover: float = 10.0) -> pd.Series:
    ret = panel.pct_change(fill_method=None)
    dates = panel.index
    w_daily = weights.reindex(dates).ffill().fillna(0.0)

    port_ret = (w_daily.shift(1) * ret).sum(axis=1)

    # turnover cost at each rebalance date
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    cost = turnover * (cost_bps_per_turnover / 10000.0)
    cost_daily = cost.reindex(dates).fillna(0.0)
    port_ret = port_ret - cost_daily

    if funding_daily is not None:
        fd = funding_daily.reindex(dates).reindex(columns=panel.columns).fillna(0.0)
        # long positions pay +funding (cost when positive), shorts receive it (so -weight * funding = correct sign flip)
        funding_pnl = (w_daily.shift(1) * (-fd)).sum(axis=1)
        port_ret = port_ret + funding_pnl

    port_ret.iloc[0] = 0.0
    return port_ret.rename("ret")


def stats(ret: pd.Series, rf: float = 0.0) -> dict:
    ret = ret.dropna()
    eq = (1 + ret).cumprod()
    span_years = (ret.index[-1] - ret.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / span_years) - 1 if span_years > 0 and eq.iloc[-1] > 0 else -1.0
    sharpe = (ret.mean() - rf / 365) / ret.std() * np.sqrt(365) if ret.std() > 0 else 0.0
    downside_dev = np.sqrt((np.minimum(ret, 0) ** 2).mean())
    sortino = (ret.mean() - rf / 365) / downside_dev * np.sqrt(365) if downside_dev > 0 else 0.0
    dd = (eq / eq.cummax() - 1)
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd < -0.001 else float("inf")
    return dict(cagr=cagr, sharpe=sharpe, sortino=sortino, max_dd=max_dd, calmar=calmar,
                final_equity=eq.iloc[-1], span_years=span_years, n_days=len(ret))
