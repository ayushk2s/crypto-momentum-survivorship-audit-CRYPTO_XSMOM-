"""NOTE: this is a fixed-config period split, not a walk-forward (no refit).
Superseded by scripts/validate_pit.py.

Frozen dev/validation/holdout split + block-bootstrap Monte Carlo,
same discipline as the rest of this engagement's audits."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from backtest import run_backtest, stats
from data_io import list_universe, load_universe_panel
from signals import build_price_panel, build_weights
from run_backtest import load_funding_panel

BASE = dict(lookback_days=28, rebalance_every=7, n_quantile=0.20, vol_lookback=20,
            funding_tilt=0.5, min_history_days=60, cost_bps=10.0)

ROOT = Path(__file__).resolve().parents[1]
PERIODS = {
    "DEV 2021-2023": ("2021-01-01", "2024-01-01"),
    "VALIDATION 2024": ("2024-01-01", "2025-01-01"),
    "HOLDOUT 2025-2026": ("2025-01-01", "2026-04-01"),
}


def main():
    universe = list_universe()
    daily = load_universe_panel(universe)
    panel = build_price_panel(daily)
    _, funding_daily = load_funding_panel()

    weights = build_weights(panel, funding_daily, BASE["lookback_days"], BASE["rebalance_every"],
                             BASE["n_quantile"], BASE["vol_lookback"], BASE["funding_tilt"], BASE["min_history_days"])
    ret = run_backtest(panel, weights, funding_daily, BASE["cost_bps"])

    print("=" * 70)
    print("WALK-FORWARD: frozen config, three non-overlapping periods")
    print("=" * 70)
    results = {}
    for label, (s, e) in PERIODS.items():
        sub = ret[(ret.index >= s) & (ret.index < e)]
        st = stats(sub)
        results[label] = st
        print(f"{label}: CAGR={st['cagr']*100:7.2f}%  Sharpe={st['sharpe']:5.2f}  "
              f"Sortino={st['sortino']:5.2f}  MaxDD={st['max_dd']*100:7.2f}%  Calmar={st['calmar']:5.2f}  n_days={st['n_days']}")
    pd.DataFrame(results).T.to_csv(ROOT / "reports" / "walkforward_results.csv")

    print("\n" + "=" * 70)
    print("MONTE CARLO: block bootstrap, 20000 x 1yr paths")
    print("=" * 70)
    r = ret.dropna().to_numpy()
    rng = np.random.default_rng(0)
    block = 20
    n_paths, horizon = 20000, 365
    n_blocks = -(-horizon // block)
    ann_rets = np.empty(n_paths)
    max_dds = np.empty(n_paths)
    ruin = np.zeros(n_paths, dtype=bool)
    for p in range(n_paths):
        starts = rng.integers(0, max(1, len(r) - block), size=n_blocks)
        path = np.concatenate([r[s:s + block] for s in starts])[:horizon]
        eq = np.cumprod(1 + path)
        dd = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)
        ann_rets[p] = eq[-1] - 1
        max_dds[p] = dd.min()
        ruin[p] = eq.min() <= 0.5

    print(f"1yr return: p5={np.percentile(ann_rets,5)*100:.1f}%  median={np.median(ann_rets)*100:.1f}%  "
          f"p95={np.percentile(ann_rets,95)*100:.1f}%")
    print(f"max DD: median={np.median(max_dds)*100:.1f}%  P(DD>30%)={(max_dds<=-0.30).mean()*100:.1f}%  "
          f"P(DD>50%)={(max_dds<=-0.50).mean()*100:.1f}%")
    print(f"P(lose 50%+ in 1yr): {ruin.mean()*100:.2f}%")

    # permutation test: random long/short sign flip per rebalance, same timing
    print("\n" + "=" * 70)
    print("PERMUTATION TEST: random sign flip of the momentum ranking, 2000 draws")
    print("=" * 70)
    obs_sharpe = stats(ret)["sharpe"]
    null_sharpes = np.empty(2000)
    rng2 = np.random.default_rng(1)
    ret_panel = panel.pct_change()
    for i in range(2000):
        rand_w = weights.copy()
        for d in rand_w.index:
            row = rand_w.loc[d]
            nz = row[row != 0]
            if len(nz) == 0:
                continue
            signs = rng2.choice([1, -1], size=len(nz))
            rand_w.loc[d, nz.index] = nz.abs().to_numpy() * signs
        null_ret = run_backtest(panel, rand_w, funding_daily, BASE["cost_bps"])
        null_sharpes[i] = stats(null_ret)["sharpe"]
    p_value = (1 + np.sum(null_sharpes >= obs_sharpe)) / 2001
    print(f"Observed Sharpe: {obs_sharpe:.3f}")
    print(f"Null Sharpe: mean={null_sharpes.mean():.3f} std={null_sharpes.std():.3f} "
          f"[2.5%,97.5%]=[{np.percentile(null_sharpes,2.5):.3f},{np.percentile(null_sharpes,97.5):.3f}]")
    print(f"p-value: {p_value:.5f}")


if __name__ == "__main__":
    main()
