# Pre-registered deployment gates — CRYPTO_XSMOM

Written 2026-10-22 **before** running the point-in-time (PIT) validation.
The thresholds may not be changed after results are seen. Every gate must pass
for a GO; any FAIL = NO-GO. A WARN means it passes but needs a written mitigation.

## Pre-registered candidate (chosen before seeing PIT results)
The original BASE spec, the first config written in this repo, with no tuning:
lookback 28d, top/bottom 20% quantile, inverse-vol(20d) weights, 50/50 dollar-neutral,
weekly rebalance run as 7 staggered daily tranches (this removes the weekday lottery),
universe = top-30 USDT-M perps by trailing 30d median quote volume (point-in-time,
delisted contracts included), min 60d listing history, funding tilt dropped (it had no effect).

## Research period / sealed holdout
- Research: 2021-01-01 → 2026-03-28 (already seen; treat as in-sample)
- Sealed holdout: 2026-03-29 → latest completed bar. Opened exactly once, after the config is frozen.

## Cost model (base)
Per side: 5 bp taker fee + half-spread 2bp·sqrt(100 / ADV$M) clipped to [0.5, 25] bp
+ square-root impact 0.7·σ_daily·sqrt(order$/ADV$). Real 8h funding. Capital $100k.

## Gates
| # | Gate | Threshold |
|---|------|-----------|
| G1 | Net Sharpe, research period, PIT universe, base costs | ≥ 0.70 |
| G2 | t-stat of mean daily return | ≥ 2.0 |
| G3 | Deflated Sharpe Ratio, N = 100 trials | ≥ 0.95 |
| G4 | Probability of Backtest Overfitting (CSCV, 16 blocks, full grid) | ≤ 0.30 |
| G5 | Rolling walk-forward (refit quarterly on trailing 2y) OOS Sharpe | ≥ 0.50 |
| G6 | Cost stress: Sharpe at 2× costs ≥ 0.50 and at 3× costs > 0 | both |
| G7 | Latency: +1 day execution delay keeps ≥ 60% of Sharpe | ≥ 60% |
| G8 | Stability: no calendar year > 50% of total log P&L; ≥ 4 of 6 years positive | both |
| G9 | Beats random-name benchmark (same dates, counts, costs) | p < 0.05 |
| G10 | Parameter neighbourhood: ≥ 70% of one-step neighbours have Sharpe ≥ 0.5×candidate | ≥ 70% |
| G11 | Sealed holdout: return not below the 5th percentile of bootstrapped same-length paths | pass |
| G12 | Capacity: Sharpe drop ≤ 0.10 at $1M vs $100k | ≤ 0.10 |
| G13 | Risk: at chosen leverage, bootstrapped 1-yr P(drawdown > 30%) ≤ 10% | ≤ 10% |
| G14 | Reproducibility: pinned env, deterministic re-run matches to 1e-9 | pass |
