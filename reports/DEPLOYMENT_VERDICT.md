# CRYPTO_XSMOM — Deployment verdict: **NO-GO**

Date: 2026-10-22 · Gates pre-registered in `DEPLOYMENT_GATES.md` before any point-in-time (PIT) result was seen.
Reproduce with `.venv` (pinned in `requirements.txt`):
`scripts/fetch_pit_futures.py` → `scripts/validate_pit.py` → `scripts/holdout_test.py`.

## What was fixed
| Problem in original build | Fix |
|---|---|
| Universe = 30 coins that still exist today (survivorship + hindsight) | PIT universe: top-30 USDT-M perps by trailing 30d median volume, 892 contracts incl. delisted (LUNA, SRM, EOS, MATIC, HNT…) — `fetch_pit_futures.py`, `src/pit_engine.py` |
| Spot prices for a perp strategy | Binance USDT-M perp daily klines + real 8h funding for 508 contracts |
| `pct_change()` forward-filled dead/delisted prices | `fill_method=None`; held contracts that stop trading are force-closed with cost |
| Constant weights between rebalances (free daily rebalancing) | Holdings drift; cost charged on actual traded notional |
| Flat 10 bp cost | Fee 5 bp + liquidity-scaled half-spread + square-root impact, capital-aware |
| Results depended on rebalance weekday (Sharpe 0.39–1.24) | 7 staggered tranches |
| "Validated" LB14 was selected using the holdout | Pre-registered config; sealed holdout opened once after freezing (`frozen_config.json`) |
| "Walk-forward" had no refit | Quarterly refit on trailing 2y over a 54-config grid |
| No PBO/DSR, empty MC logs | CSCV PBO, DSR(N=100), bootstrap ruin, random-name null — all logged in `reports/pit/` |
| Non-standard Sortino; wrong docstrings | Fixed in `src/backtest.py`, `src/signals.py`, `scripts/validate_config.py` |

## Gate results (pre-registered candidate: lb28, q0.20, top-30, weekly×7 tranches)
| Gate | Result | Evidence |
|---|---|---|
| G1 Sharpe ≥ 0.70 | FAIL | 0.22 (CAGR 0.9%, MaxDD −59.5%) |
| G2 t ≥ 2 | FAIL | 0.51; Sharpe 95% CI [−0.65, 1.09] |
| G3 DSR ≥ 0.95 | FAIL | 0.036 |
| G4 PBO ≤ 0.30 | FAIL | 0.45 |
| G5 Walk-forward OOS ≥ 0.50 | FAIL | −0.15 (2023-01 → 2026-03) |
| G6 Cost stress | FAIL | 0.22 / 0.11 / −0.01 at 1× / 2× / 3× |
| G7 Latency | PASS | lag+1 day 0.37 (noise: both ≈ 0) |
| G8 Stability | FAIL | 2025 −38.8%; whole P&L = 2021; ex-best-10-days log P&L −0.72 |
| G9 Random-name null | PASS (weak) | p = 0.01, but null mean −0.78 is cost-driven |
| G10 Neighbourhood | PASS (trivial) | threshold is half of a 0.22 Sharpe |
| G11 Sealed holdout | PASS (low power) | −10.9%, Sharpe −0.33, MaxDD −36.6% (25th pct of a very wide distribution) while BTC +26.7% |
| G12 Capacity | PASS | $1M 0.17 |
| G13 Risk | FAIL | No leverage ≥ 0.5× keeps P(1y DD > 30%) ≤ 10% (1×: 35.6%) |
| G14 Reproducibility | PASS | bit-identical re-run |

**8 FAIL / 6 PASS → NO-GO.**

## Why the original backtest looked good
On the hand-picked 30 survivors the same engine gives Sharpe ~1.0; on the PIT universe 0.22.
P&L attribution (sleeve 0): old-30 names +1.55, all other names −2.38. The edge was the coin list, chosen in 2026
with knowledge of which coins survived and grew (SOL, AVAX, XRP top contributors).
The short leg alone loses −29.6%/yr; MYX (+14,000% squeeze, Aug–Sep 2025) alone cost one sleeve −162%.

## Sealed holdout (2026-03-29 → 2026-10-21), every variant
| Variant | Research Sharpe | Holdout |
|---|---|---|
| Original "validated" LB14, old 30 coins | 0.98 | −16.3%, Sharpe −1.28 |
| Original BASE LB28, old 30 coins | 0.81 | −7.6%, Sharpe −0.53 |
| Pre-registered PIT candidate | 0.22 | −10.9%, Sharpe −0.33 |
| Post-hoc PIT + squeeze stop | 0.59 | −13.2%, Sharpe −0.46 |

## What would be needed before any re-submission
A *new* hypothesis (not retuning this one), tested on the PIT universe, with a fresh sealed period, plus
≥ 3 months of paper trading with live fills, a position-level squeeze stop, a kill switch at −15% drawdown, per-name
weight caps, a delisting-announcement filter, and exchange-constraint handling (min notional, lot size, margin mode).
