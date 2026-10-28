"""Monte Carlo + permutation test for lookback_days=14.

AUDIT NOTE (2026-10-22): this config was NOT validated out-of-sample.
walkforward_montecarlo.py only ever ran lookback=28. lookback=14 is the best row
of param_sweep.py, which spans 2021-2026 including the 'holdout'; on DEV
(2021-2023) alone it ranks 17/19 (Sharpe 0.60). Treat it as holdout-snooped.
Superseded by scripts/validate_pit.py + scripts/holdout_test.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from backtest import run_backtest, stats
from data_io import list_universe, load_universe_panel
from signals import build_price_panel, build_weights
from scripts.run_backtest import load_funding_panel

ROOT = Path(__file__).resolve().parents[1]
CFG = dict(lookback_days=14, rebalance_every=7, n_quantile=0.20, vol_lookback=20,
           funding_tilt=0.5, min_history_days=60, cost_bps=10.0)


def main():
    universe = list_universe()
    daily = load_universe_panel(universe)
    panel = build_price_panel(daily)
    _, funding_daily = load_funding_panel()

    weights = build_weights(panel, funding_daily, CFG["lookback_days"], CFG["rebalance_every"],
                             CFG["n_quantile"], CFG["vol_lookback"], CFG["funding_tilt"], CFG["min_history_days"])
    ret = run_backtest(panel, weights, funding_daily, CFG["cost_bps"])
    obs_st = stats(ret)
    print(f"Validated config full-period: CAGR={obs_st['cagr']*100:.2f}% Sharpe={obs_st['sharpe']:.3f} "
          f"Sortino={obs_st['sortino']:.3f} MaxDD={obs_st['max_dd']*100:.2f}% Calmar={obs_st['calmar']:.3f}")

    print("\n=== MONTE CARLO: block bootstrap, 20000 x 1yr paths ===")
    r = ret.dropna().to_numpy()
    rng = np.random.default_rng(0)
    block, n_paths, horizon = 20, 20000, 365
    n_blocks = -(-horizon // block)
    ann_rets = np.empty(n_paths); max_dds = np.empty(n_paths); ruin = np.zeros(n_paths, dtype=bool)
    for p in range(n_paths):
        starts = rng.integers(0, max(1, len(r) - block), size=n_blocks)
        path = np.concatenate([r[s:s + block] for s in starts])[:horizon]
        eq = np.cumprod(1 + path)
        dd = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)
        ann_rets[p] = eq[-1] - 1
        max_dds[p] = dd.min()
        ruin[p] = eq.min() <= 0.5
    print(f"1yr return: p5={np.percentile(ann_rets,5)*100:.1f}% median={np.median(ann_rets)*100:.1f}% "
          f"p95={np.percentile(ann_rets,95)*100:.1f}%")
    print(f"max DD: median={np.median(max_dds)*100:.1f}% P(DD>20%)={(max_dds<=-0.20).mean()*100:.1f}% "
          f"P(DD>30%)={(max_dds<=-0.30).mean()*100:.1f}%")
    print(f"P(lose 50%+ in 1yr): {ruin.mean()*100:.2f}%")

    print("\n=== PERMUTATION TEST: 2000 draws, random long/short sign per rebalance ===")
    obs_sharpe = obs_st["sharpe"]
    rng2 = np.random.default_rng(1)
    null_sharpes = np.empty(2000)
    for i in range(2000):
        rand_w = weights.copy()
        for d in rand_w.index:
            row = rand_w.loc[d]
            nz = row[row != 0]
            if len(nz) == 0:
                continue
            signs = rng2.choice([1, -1], size=len(nz))
            rand_w.loc[d, nz.index] = nz.abs().to_numpy() * signs
        null_ret = run_backtest(panel, rand_w, funding_daily, CFG["cost_bps"])
        null_sharpes[i] = stats(null_ret)["sharpe"]
    p_value = (1 + np.sum(null_sharpes >= obs_sharpe)) / 2001
    print(f"Observed Sharpe: {obs_sharpe:.3f}")
    print(f"Null Sharpe: mean={null_sharpes.mean():.3f} std={null_sharpes.std():.3f} "
          f"[2.5,97.5]=[{np.percentile(null_sharpes,2.5):.3f},{np.percentile(null_sharpes,97.5):.3f}]")
    print(f"p-value: {p_value:.5f}")

    ret.to_csv(ROOT / "reports" / "validated_config_daily_returns.csv")


if __name__ == "__main__":
    main()
