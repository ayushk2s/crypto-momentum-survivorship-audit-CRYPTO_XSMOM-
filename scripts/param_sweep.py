"""Bounded, predeclared parameter sensitivity sweep -- one-at-a-time around
the base config, same discipline applied to wave3 earlier in this engagement."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from backtest import run_backtest, stats
from data_io import list_universe, load_universe_panel
from signals import build_price_panel, build_weights
from run_backtest import load_funding_panel

ROOT = Path(__file__).resolve().parents[1]

BASE = dict(lookback_days=28, rebalance_every=7, n_quantile=0.20, vol_lookback=20,
            funding_tilt=0.5, min_history_days=60, cost_bps=10.0)

GRIDS = {
    "lookback_days": [14, 21, 28, 42, 56],
    "rebalance_every": [3, 7, 14, 21],
    "n_quantile": [0.10, 0.15, 0.20, 0.30, 0.40],
    "vol_lookback": [10, 20, 30, 45],
    "funding_tilt": [0.0, 0.25, 0.5, 0.75, 1.0],
}


def run_one(panel, funding_daily, cfg):
    w = build_weights(panel, funding_daily, cfg["lookback_days"], cfg["rebalance_every"],
                       cfg["n_quantile"], cfg["vol_lookback"], cfg["funding_tilt"], cfg["min_history_days"])
    r = run_backtest(panel, w, funding_daily, cfg["cost_bps"])
    return stats(r)


def main():
    universe = list_universe()
    daily = load_universe_panel(universe)
    panel = build_price_panel(daily)
    _, funding_daily = load_funding_panel()

    rows = []
    base_st = run_one(panel, funding_daily, BASE)
    rows.append({**BASE, "varying": "BASE", **base_st})
    print(f"BASE -> CAGR={base_st['cagr']*100:.2f}% Sharpe={base_st['sharpe']:.2f} "
          f"MaxDD={base_st['max_dd']*100:.2f}% Calmar={base_st['calmar']:.2f}")

    for param, grid in GRIDS.items():
        for val in grid:
            if val == BASE[param]:
                continue
            cfg = dict(BASE)
            cfg[param] = val
            st = run_one(panel, funding_daily, cfg)
            rows.append({**cfg, "varying": param, **st})
            print(f"vary {param}={val} -> CAGR={st['cagr']*100:7.2f}% Sharpe={st['sharpe']:5.2f} "
                  f"MaxDD={st['max_dd']*100:7.2f}% Calmar={st['calmar']:5.2f}")

    pd.DataFrame(rows).to_csv(ROOT / "reports" / "param_sweep.csv", index=False)
    print(f"\nWrote {ROOT / 'reports' / 'param_sweep.csv'}")


if __name__ == "__main__":
    main()
