"""Regenerate the figures in docs/figures/ used by docs/PAPER.md."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pit_engine import A, Config, Data, load, run  # noqa: E402

FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
HOLDOUT = pd.Timestamp("2026-03-29")
OLD30 = [s + "USDT" for s in "AAVE ADA APT ARB ASTER ATOM AVAX BCH BNB BTC CFX DOGE DOT ETH FET FIL INJ LDO LINK "
         "LTC MATIC NEAR OP RUNE SEI SOL SUI TRX UNI XRP".split()]

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "lines.linewidth": 2, "savefig.dpi": 160, "savefig.bbox": "tight"})


def sharpe(r):
    return r.mean() / r.std() * np.sqrt(A)


def shade_holdout(ax):
    ax.axvspan(HOLDOUT, ax.get_xlim()[1] if False else pd.Timestamp("2026-10-21"), color=GRID, alpha=0.6, lw=0)
    ax.text(HOLDOUT + pd.Timedelta(days=8), ax.get_ylim()[1], "sealed\nholdout", va="top", ha="left", fontsize=8, color=INK2)


def main():
    d = load()
    d30 = Data(d.close[OLD30], d.qvol[OLD30], d.funding[OLD30])
    survivor = run(d30, Config(universe_n=999, tranches=1, lag=1))
    pit = run(d, Config())
    import matplotlib.ticker  # noqa: F401

    # Fig 1 -- the whole story in one chart
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for r, c, lab in ((survivor, ORANGE, f"Hand-picked 30 survivors (Sharpe {sharpe(survivor[:HOLDOUT]):.2f})"),
                      (pit, BLUE, f"Point-in-time universe (Sharpe {sharpe(pit[:HOLDOUT]):.2f})")):
        eq = (1 + r).cumprod()
        ax.plot(eq.index, eq, color=c, label=lab)
        ax.annotate(f"{eq.iloc[-1]:.2f}×", (eq.index[-1], eq.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.set_yscale("log"); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([0.5, 1, 2, 4]); ax.set_yticklabels(["0.5×", "1×", "2×", "4×"])
    sq = pd.Timestamp("2025-09-09"); y_sq = (1 + pit).cumprod().loc[sq]
    ax.annotate("MYX short squeeze\n(Sep 2025)", (sq, y_sq), xytext=(-120, -70), textcoords="offset points", fontsize=8,
                color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.axhline(1, color=INK2, lw=0.8)
    shade_holdout(ax)
    ax.set_title("Same strategy, same engine — only the coin list differs")
    ax.set_ylabel("Growth of 1 (log scale)")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(FIG / "fig1_survivor_vs_pit.png"); plt.close(fig)

    # Fig 2 -- grid Sharpe distribution
    M = pd.read_parquet(ROOT / "reports" / "pit" / "grid_returns.parquet")
    gs = M.apply(sharpe).sort_values()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    colors = [ORANGE if k == Config().key() else BLUE for k in gs.index]
    ax.bar(range(len(gs)), gs.values, color=colors, width=0.75)
    ax.axhline(0, color=INK2, lw=0.8)
    i = list(gs.index).index(Config().key())
    ax.annotate("pre-registered\ncandidate", (i, gs.iloc[i]), xytext=(0, 28), textcoords="offset points", ha="center",
                fontsize=8, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.annotate(f"best: {gs.index[-1]}", (len(gs) - 1, gs.iloc[-1]), xytext=(-4, 6), textcoords="offset points",
                ha="right", fontsize=8)
    ax.set_xticks([]); ax.set_xlabel("54 configurations (lookback × quantile × universe size), sorted")
    ax.set_ylabel("Net Sharpe, 2021–Mar 2026")
    ax.set_title("Parameter grid on the point-in-time universe: wide, unstable spread")
    fig.savefig(FIG / "fig2_grid_sharpe.png"); plt.close(fig)

    # Fig 3 -- yearly returns of the candidate (incl. holdout part of 2026)
    yr = pit.groupby(pit.index.year).apply(lambda x: (1 + x).prod() - 1)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.bar(yr.index.astype(str), yr.values * 100, color=[BLUE if v >= 0 else ORANGE for v in yr.values], width=0.6)
    for x, v in zip(yr.index.astype(str), yr.values):
        ax.annotate(f"{v*100:+.1f}%", (x, v * 100), xytext=(0, 4 if v >= 0 else -14), textcoords="offset points",
                    ha="center", fontsize=9)
    ax.axhline(0, color=INK2, lw=0.8); ax.set_ylim(-48, 68)
    ax.set_ylabel("Calendar-year return (%)")
    ax.set_title("Point-in-time candidate by year: one good year, then nothing")
    fig.savefig(FIG / "fig3_yearly.png"); plt.close(fig)

    # Fig 4 -- holdout results of every variant
    rows = [("Original LB14 \"validated\" (30 survivors)", run(d30, Config(lookback=14, universe_n=999, tranches=1, lag=1))),
            ("Original BASE LB28 (30 survivors)", survivor),
            ("Pre-registered PIT candidate", pit),
            ("PIT + squeeze stop (post-hoc)", run(d, Config(max_drift=2.0)))]
    vals = [((1 + r[HOLDOUT:]).prod() - 1) * 100 for _, r in rows]
    btc = d.close["BTCUSDT"]; btc_ret = (btc.iloc[-1] / btc.loc[:HOLDOUT].iloc[-2] - 1) * 100
    fig, ax = plt.subplots(figsize=(9, 3.2))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, vals, color=ORANGE, height=0.55)
    for yy, v in zip(y, vals):
        ax.annotate(f"{v:+.1f}%", (v, yy), xytext=(-6, 0), textcoords="offset points", ha="right", va="center", fontsize=9)
    ax.set_xlim(min(vals) * 1.25, 1)
    ax.set_yticks(y); ax.set_yticklabels([n for n, _ in rows]); ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel(f"Return, 2026-03-29 → 2026-10-21 (BTC over the same window: {btc_ret:+.1f}%)")
    ax.set_title("Sealed holdout: every variant lost money")
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG / "fig4_holdout.png"); plt.close(fig)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
