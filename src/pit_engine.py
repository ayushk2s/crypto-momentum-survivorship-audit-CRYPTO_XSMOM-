"""Corrected cross-sectional momentum engine on a point-in-time USDT-M universe.

Fixes relative to src/backtest.py + src/signals.py:
  * universe is chosen point-in-time by trailing liquidity, delisted contracts included
  * no forward-filled prices (pct_change(fill_method=None)); a held contract that
    stops trading is force-closed at its last price, with cost
  * holdings drift between rebalances; costs are charged on the actual traded
    notional against the drifted book (not on target-weight differences)
  * per-symbol cost = taker fee + liquidity-scaled half-spread + square-root impact
  * staggered tranches remove the dependence on which weekday you rebalance
  * explicit execution lag: signal from close[t-lag], trade at close[t]
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

UM = Path(__file__).resolve().parents[1] / "data_cache" / "um"
A = 365


@dataclass(frozen=True)
class Config:
    lookback: int = 28
    quantile: float = 0.20
    vol_lookback: int = 20
    universe_n: int = 30
    min_history: int = 60
    rebalance_every: int = 7
    tranches: int = 7          # staggered sleeves; 1 = single weekday (original behaviour)
    lag: int = 0               # 0 = trade at the close of the signal bar (live: minutes after 00:00 UTC)
    fee_bps: float = 5.0
    cost_mult: float = 1.0     # stress multiplier on fee + spread + impact
    capital: float = 100_000.0
    impact_coef: float = 0.7
    use_funding: bool = True
    legs: str = "both"        # both | long | short (diagnostics)
    max_drift: float = 0.0     # >0: daily, cut any position whose |weight| grew beyond max_drift x its target (squeeze stop)

    def key(self) -> str:
        return f"lb{self.lookback}_q{self.quantile}_u{self.universe_n}"


@dataclass
class Data:
    close: pd.DataFrame
    qvol: pd.DataFrame
    funding: pd.DataFrame
    ret: pd.DataFrame = field(init=False)

    def __post_init__(self):
        self.ret = self.close.pct_change(fill_method=None)


def load(end: str | None = None) -> Data:
    k = pd.read_parquet(UM / "klines_1d.parquet")
    close = k.pivot(index="date", columns="symbol", values="close").sort_index()
    qv = k.pivot(index="date", columns="symbol", values="quote_volume").reindex(close.index)
    f = pd.read_parquet(UM / "funding.parquet")
    fd = (f.assign(date=f.ts.dt.normalize()).pivot_table(index="date", columns="symbol", values="fundingRate", aggfunc="sum")
          .reindex(index=close.index, columns=close.columns).fillna(0.0))
    if end:
        close, qv, fd = close.loc[:end], qv.loc[:end], fd.loc[:end]
    close = close.loc[:, close.notna().any()]
    return Data(close, qv.reindex(columns=close.columns), fd.reindex(columns=close.columns))


class Precomp:
    """Signal inputs that depend only on (data, lookback, vol_lookback) -- cached across sleeves."""
    _cache: dict = {}

    @classmethod
    def get(cls, d: Data, lookback: int, vol_lookback: int):
        key = (id(d), lookback, vol_lookback)
        if key not in cls._cache:
            mom = (d.close / d.close.shift(lookback) - 1).to_numpy()
            vol = d.ret.rolling(vol_lookback, min_periods=vol_lookback).std().to_numpy()
            adv = d.qvol.rolling(30, min_periods=20).median().to_numpy()
            hist = d.close.notna().cumsum().to_numpy()
            cls._cache[key] = (mom, vol, adv, hist)
        return cls._cache[key]


def sleeve(d: Data, cfg: Config, offset: int, start: str = "2021-01-01", picker=None, attrib: dict | None = None) -> pd.Series:
    """One weekly-rebalanced sleeve. `picker(i, eligible_idx, n)` overrides the
    momentum ranking (used for random-name null benchmarks)."""
    mom, vol, adv, hist = Precomp.get(d, cfg.lookback, cfg.vol_lookback)
    close = d.close.to_numpy()
    R = np.nan_to_num(d.ret.to_numpy(), nan=0.0)
    has_px = ~np.isnan(close)
    F = d.funding.to_numpy() if cfg.use_funding else np.zeros_like(R)
    dates = d.close.index
    t0 = int(dates.searchsorted(pd.Timestamp(start)))
    n_sym = close.shape[1]
    h = np.zeros(n_sym)
    eq = 1.0
    out = np.zeros(len(dates))
    tgt_last = np.zeros(n_sym)

    def cost_bps(i, trade):
        s = i - cfg.lag
        a = np.nan_to_num(adv[s], nan=1e6)
        a = np.maximum(a, 1e5)
        half_spread = np.clip(2.0 * np.sqrt(100e6 / a), 0.5, 25.0)
        sig = np.nan_to_num(vol[s], nan=0.05)
        impact = cfg.impact_coef * sig * np.sqrt(np.abs(trade) * eq * cfg.capital / cfg.tranches / a) * 1e4
        return (cfg.fee_bps + half_spread + impact) * cfg.cost_mult

    for i in range(t0, len(dates)):
        pnl = float(h @ R[i])
        if attrib is not None:
            for j in np.flatnonzero(h):
                attrib[j] = attrib.get(j, 0.0) + h[j] * (R[i, j] - F[i, j])
        fund = -float(h @ F[i])
        g = 1.0 + pnl + fund
        h = h * (1.0 + R[i]) / g if g > 0 else np.zeros(n_sym)
        cost = 0.0
        # force-close anything that has stopped trading
        dead = (h != 0) & ~has_px[i]
        if dead.any():
            tr = np.where(dead, -h, 0.0)
            cost += float(np.abs(tr) @ (cost_bps(i, tr) / 1e4))
            h = np.where(dead, 0.0, h)
        if cfg.max_drift > 0:
            blown = (np.abs(h) > cfg.max_drift * np.abs(tgt_last)) & (tgt_last != 0)
            if blown.any():
                tr = np.where(blown, -h, 0.0)
                cost += float(np.abs(tr) @ (cost_bps(i, tr) / 1e4))
                h = np.where(blown, 0.0, h)
                tgt_last = np.where(blown, 0.0, tgt_last)
        if (i - t0 - offset) % cfg.rebalance_every == 0 and i - t0 >= offset:
            s = i - cfg.lag
            ok = has_px[i] & has_px[s] & (hist[s] >= cfg.min_history) & ~np.isnan(mom[s]) & ~np.isnan(vol[s]) & (vol[s] > 0) & ~np.isnan(adv[s])
            idx = np.flatnonzero(ok)
            tgt = np.zeros(n_sym)
            if len(idx) >= 10:
                idx = idx[np.argsort(-adv[s, idx])][: cfg.universe_n]
                n = max(1, int(round(len(idx) * cfg.quantile)))
                if picker is None:
                    order = idx[np.argsort(-mom[s, idx])]
                    longs, shorts = order[:n], order[-n:]
                else:
                    longs, shorts = picker(i, idx, n)
                iv = 1.0 / vol[s]
                if cfg.legs in ("both", "long"):
                    tgt[longs] = 0.5 * iv[longs] / iv[longs].sum()
                if cfg.legs in ("both", "short"):
                    tgt[shorts] = -0.5 * iv[shorts] / iv[shorts].sum()
            tr = tgt - h
            cost += float(np.abs(tr) @ (cost_bps(i, tr) / 1e4))
            h = tgt
            tgt_last = tgt.copy()
        r = g - 1.0 - cost
        out[i] = r
        eq *= 1.0 + r
    return pd.Series(out, index=dates).iloc[t0:]


def run(d: Data, cfg: Config, start: str = "2021-01-01", picker_factory=None) -> pd.Series:
    k = cfg.tranches
    sleeves = [sleeve(d, cfg, off * cfg.rebalance_every // k, start, picker_factory(off) if picker_factory else None)
               for off in range(k)]
    return pd.concat(sleeves, axis=1).mean(axis=1).rename(cfg.key())


def perf(r: pd.Series) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / A
    dd = eq / eq.cummax() - 1
    sd = r.std()
    down = np.sqrt((np.minimum(r, 0) ** 2).mean())
    return dict(cagr=eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1, vol=sd * np.sqrt(A),
                sharpe=r.mean() / sd * np.sqrt(A) if sd > 0 else 0.0,
                sortino=r.mean() / down * np.sqrt(A) if down > 0 else 0.0,
                max_dd=dd.min(), calmar=(eq.iloc[-1] ** (1 / yrs) - 1) / abs(dd.min()) if dd.min() < 0 else np.inf,
                tstat=r.mean() / sd * np.sqrt(len(r)) if sd > 0 else 0.0, n=len(r))


def as_dict(cfg: Config) -> dict:
    return asdict(cfg)
