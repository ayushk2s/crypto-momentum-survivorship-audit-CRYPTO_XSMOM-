# Crypto Cross-Sectional Momentum — A Pre-Registered Backtest Audit

> **Verdict: NO-GO.** A crypto momentum strategy that backtested at Sharpe 0.68–0.90 earns **Sharpe 0.22**
> once the coin universe is built point-in-time (delisted coins included), and **lost money on seven months
> of sealed, never-seen data** while BTC rose 26.7%.

This repository contains the strategy, the audit that broke it, and a reusable pipeline for testing crypto
cross-sectional strategies without survivorship bias or holdout snooping.

📄 **Full write-up: [docs/PAPER.md](docs/PAPER.md)**

![Same strategy, same engine — only the coin list differs](docs/figures/fig1_survivor_vs_pit.png)

---

## Key findings

| | Original backtest | Corrected (point-in-time) |
|---|---|---|
| Universe | 30 coins still trading in 2026 | Top-30 by trailing volume from 892 Binance perps, incl. delisted |
| Net Sharpe (2021 → Mar 2026) | 0.68 – 0.90 | **0.22** (t = 0.51) |
| CAGR | 17 – 24% | **0.9%** |
| Max drawdown | −25% to −36% | **−59.5%** |
| Sealed holdout (Apr → Oct 2026) | −7.6% to −16.3% | **−10.9%** |

- **The edge was the coin list.** On the same engine, the 30 hand-picked survivors give Sharpe 0.81, while the
  point-in-time universe gives 0.22. Survivor names contributed +1.55 of P&L; every other name contributed −2.38.
- **The "validated" config was selected on its own holdout.** It ranked 17th of 19 on the development period.
- **The rebalance weekday alone moved Sharpe from 0.39 to 1.24.**
- **The short leg loses about 30% a year.** One short squeeze (MYX, Sept 2025: $1.16 → $14.48 in four days)
  cost one sleeve 162% of its capital.
- **8 of 14 pre-registered deployment gates fail**, including Deflated Sharpe (0.04), probability of
  backtest overfitting (0.45) and rolling walk-forward (Sharpe −0.15).

## What this repo is useful for

Even though the strategy failed, the pipeline is reusable for any daily crypto cross-sectional idea:

- **Point-in-time Binance USDT-M data:** 892 perpetuals including delisted contracts, plus real 8-hour funding,
  pulled from the public `data.binance.vision` archive.
- **A corrected backtest engine:**
  - liquidity-ranked universe built only from past data;
  - holdings that drift between rebalances;
  - fee + spread + square-root impact cost model;
  - forced exits when a contract stops trading;
  - staggered rebalancing tranches.
- **A validation suite:**
  - Deflated Sharpe and probability of backtest overfitting;
  - rolling walk-forward with refit;
  - random-name benchmark;
  - cost, latency and capacity stress tests;
  - leg decomposition and regime analysis;
  - bootstrap risk of ruin.
- **Pre-registration plumbing:** gates written before results, a frozen config with a hash, and a holdout
  script that refuses to run without the freeze.

## Quick start

Requires Python 3.13.

```bash
git clone <this-repo> && cd CRYPTO_XSMOM
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt        # Linux/macOS: .venv/bin/pip

python scripts/fetch_pit_futures.py   # download point-in-time perp data + funding (~10 min, ~27 MB)
python scripts/validate_pit.py        # run all research-period gates, freeze the config (~2 min)
python scripts/holdout_test.py        # open the sealed holdout exactly once
python scripts/make_figures.py        # regenerate docs/figures/
```

On Windows, if the console fails to print Unicode characters, set `PYTHONUTF8=1` first.

## Repository layout

```
CRYPTO_XSMOM/
├── docs/
│   ├── PAPER.md                  # full research write-up
│   └── figures/                  # figures used in the paper
├── src/
│   ├── pit_engine.py             # corrected point-in-time engine (use this)
│   ├── signals.py, backtest.py   # original engine (kept for the audit trail, bugs fixed)
│   └── data_io.py                # original loader for local 1-minute spot data
├── scripts/
│   ├── fetch_pit_futures.py      # point-in-time perp klines + funding, incl. delisted
│   ├── validate_pit.py           # gates G1–G10, G12–G14; freezes reports/frozen_config.json
│   ├── holdout_test.py           # gate G11, sealed holdout
│   ├── make_figures.py
│   └── run_backtest.py, param_sweep.py, walkforward_montecarlo.py,
│       validate_config.py, fetch_funding_universe.py   # original pipeline (legacy)
├── reports/
│   ├── DEPLOYMENT_GATES.md       # gates, written before any point-in-time result
│   ├── DEPLOYMENT_VERDICT.md     # gate-by-gate verdict
│   ├── frozen_config.json        # frozen candidate + research-returns hash
│   ├── pit/                      # logs and return series from the corrected runs
│   └── *.csv, *_log.txt          # outputs of the original pipeline
└── requirements.txt
```

The legacy pipeline (`run_backtest.py` etc.) reads 1-minute spot CSVs from a local path in `src/data_io.py`.
That data is not included, so those scripts are kept as an audit trail only. Everything in the corrected
pipeline runs from public data.

## Using the engine for your own idea

```python
import sys; sys.path.insert(0, "src")
from pit_engine import Config, load, run, perf

data = load(end="2026-03-28")                        # research period only
cfg = Config(lookback=28, quantile=0.2, universe_n=30, cost_mult=2.0)
returns = run(data, cfg)                             # daily net returns, tranche-averaged
print(perf(returns))                                 # cagr, sharpe, sortino, max_dd, tstat, ...
```

`Config` exposes lookback, quantile, universe size, minimum history, rebalance interval, number of
tranches, execution lag, fee, cost multiplier, capital (for impact), funding on/off, leg selection
(`both`/`long`/`short`) and an optional squeeze stop (`max_drift`).

Before trusting a new result, write its gates down first, then keep at least six months of data sealed.

## Limitations

- **Costs:** spread and impact are modelled, not measured from real fills.
- **Execution:** daily-close execution only.
- **Venue coverage:** Binance perpetuals only.
- **Holdout length:** the sealed holdout is 207 days, which has low statistical power on its own.

See [docs/PAPER.md §8](docs/PAPER.md#8-limitations).

## Disclaimer

Research code, not investment advice. Nothing here recommends trading any instrument. Crypto derivatives can
lose more than the capital allocated to them.
