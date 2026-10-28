"""Main runner: cross-sectional momentum + funding-crowding tilt, full
30-asset crypto universe, weekly rebalance. Prints base-case stats."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from backtest import run_backtest, stats
from data_io import list_universe, load_universe_panel
from signals import build_price_panel, build_weights

ROOT = Path(__file__).resolve().parents[1]

LOOKBACK_DAYS = 28
REBALANCE_EVERY = 7
N_QUANTILE = 0.20
VOL_LOOKBACK = 20
FUNDING_TILT = 0.5
MIN_HISTORY_DAYS = 60
COST_BPS = 10.0


def load_funding_panel():
    fp = ROOT / "reports" / "funding_rates_universe.csv"
    if not fp.exists():
        return None, None
    f = pd.read_csv(fp)
    f["ts"] = pd.to_datetime(f["ts"])
    f["symbol"] = f["symbol"].str.replace("USDT", "", regex=False) + "USDT"
    wide = f.pivot_table(index="ts", columns="symbol", values="fundingRate", aggfunc="sum")
    daily_funding = wide.resample("1D").sum()  # sum of the day's funding events (fraction)
    return wide, daily_funding


def main():
    universe = list_universe()
    print(f"Universe: {len(universe)} symbols")
    daily = load_universe_panel(universe)
    panel = build_price_panel(daily)
    print(f"Price panel: {panel.shape[0]} days x {panel.shape[1]} symbols, "
          f"{panel.index[0].date()} -> {panel.index[-1].date()}")

    _, funding_daily = load_funding_panel()
    if funding_daily is not None:
        print(f"Funding panel loaded: {funding_daily.shape}")
    else:
        print("No funding data found yet -- run fetch_funding_universe.py first. Proceeding without funding tilt/cost.")

    weights = build_weights(panel, funding_daily, LOOKBACK_DAYS, REBALANCE_EVERY, N_QUANTILE,
                             VOL_LOOKBACK, FUNDING_TILT, MIN_HISTORY_DAYS)
    ret = run_backtest(panel, weights, funding_daily, COST_BPS)
    st = stats(ret)

    print("\n=== BASE CASE ===")
    for k, v in st.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    ret.to_csv(ROOT / "reports" / "base_case_daily_returns.csv")
    print(f"\nSaved {ROOT / 'reports' / 'base_case_daily_returns.csv'}")


if __name__ == "__main__":
    main()
