"""Full pre-deployment validation on the research period ONLY (sealed holdout untouched).
Evaluates every gate in reports/DEPLOYMENT_GATES.md except G11 (see holdout_test.py),
then freezes the pre-registered config to reports/frozen_config.json.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
import time
from dataclasses import replace
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as ss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pit_engine import A, Config, as_dict, load, perf, run, sleeve  # noqa: E402

RESEARCH_END = "2026-03-28"
OUT = ROOT / "reports" / "pit"
OUT.mkdir(parents=True, exist_ok=True)
GRID = dict(lookback=[7, 14, 21, 28, 42, 56], quantile=[0.1, 0.2, 0.3], universe_n=[20, 30, 50])
N_TRIALS = 100  # 54-config grid here + 19 earlier sweep + unknown earlier "waves"
CAND = Config()
gates: dict[str, tuple[str, str]] = {}
log_lines: list[str] = []


def log(s=""):
    print(s, flush=True)
    log_lines.append(str(s))


def gate(k, ok, evidence, warn=False):
    gates[k] = ("PASS" if ok and not warn else "WARN" if ok else "FAIL", evidence)
    log(f"  >> {k}: {gates[k][0]} — {evidence}")


def sh(r):
    r = r.dropna()
    return r.mean() / r.std() * np.sqrt(A) if r.std() > 0 else 0.0


def main():
    t_start = time.time()
    d = load(end=RESEARCH_END)
    log(f"Research data: {d.close.shape[1]} contracts (live+delisted), {d.close.index[0].date()} -> {d.close.index[-1].date()}")

    # ---------- universe audit ----------
    adv = d.qvol.rolling(30, min_periods=20).median()
    hist = d.close.notna().cumsum()
    in_u = adv.where(hist >= 60).rank(axis=1, ascending=False).le(30)
    ever = in_u.loc["2021-01-01":].any()
    last_px = d.close.apply(lambda c: c.last_valid_index())
    dead = [s for s in ever[ever].index if last_px[s] < d.close.index[-1] - pd.Timedelta(days=5)]
    log(f"Contracts ever in PIT top-30: {int(ever.sum())}; of which delisted/stopped before research end: {len(dead)}")
    log(f"  delisted examples: {', '.join(sorted(dead)[:25])}")

    # ---------- grid ----------
    keys = list(itertools.product(*GRID.values()))
    rets = {}
    for lb, q, u in keys:
        c = replace(CAND, lookback=lb, quantile=q, universe_n=u)
        rets[c.key()] = run(d, c)
    M = pd.DataFrame(rets)
    M.to_parquet(OUT / "grid_returns.parquet")
    cand = M[CAND.key()]
    gs = M.apply(sh)
    log(f"\nGrid: {len(keys)} configs. Sharpe distribution: min {gs.min():.2f} median {gs.median():.2f} max {gs.max():.2f}; best = {gs.idxmax()}")
    log(gs.sort_values(ascending=False).round(2).to_string())

    # ---------- G1/G2 core ----------
    p = perf(cand)
    log(f"\n=== CANDIDATE {CAND.key()} (pre-registered) ===")
    log(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in p.items()))
    gate("G1 net Sharpe >= 0.70", p["sharpe"] >= 0.70, f"Sharpe {p['sharpe']:.2f}, CAGR {p['cagr']*100:.1f}%, MaxDD {p['max_dd']*100:.1f}%")
    gate("G2 t-stat >= 2.0", p["tstat"] >= 2.0, f"t = {p['tstat']:.2f}")

    # ---------- G3 DSR ----------
    r = cand.to_numpy(); T = len(r); srd = r.mean() / r.std()
    sk, ku = ss.skew(r), ss.kurtosis(r, fisher=False)
    se = np.sqrt((1 - sk * srd + (ku - 1) / 4 * srd ** 2) / (T - 1))
    var_sr = (gs / np.sqrt(A)).var()
    em = 0.5772156649
    sr0 = np.sqrt(var_sr) * ((1 - em) * ss.norm.ppf(1 - 1 / N_TRIALS) + em * ss.norm.ppf(1 - 1 / (N_TRIALS * np.e)))
    dsr = ss.norm.cdf((srd - sr0) / se)
    ci = ((srd - 1.96 * se) * np.sqrt(A), (srd + 1.96 * se) * np.sqrt(A))
    log(f"Sharpe 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]  skew {sk:.2f} kurt {ku:.1f}  PSR(>0) {ss.norm.cdf(srd/se):.3f}")
    gate("G3 DSR(N=100) >= 0.95", dsr >= 0.95, f"DSR {dsr:.3f} (expected max-of-{N_TRIALS} Sharpe under null = {sr0*np.sqrt(A):.2f})")

    # ---------- G4 PBO (CSCV) ----------
    S = 16
    blocks = np.array_split(np.arange(len(M)), S)
    X = M.to_numpy()
    bm = np.array([X[b].mean(0) for b in blocks]); bs2 = np.array([(X[b] ** 2).mean(0) for b in blocks]); bn = np.array([len(b) for b in blocks])
    logits = []
    for combo in itertools.combinations(range(S), S // 2):
        ins = np.zeros(S, bool); ins[list(combo)] = True
        def srs(mask):
            w = bn[mask][:, None]; m = (bm[mask] * w).sum(0) / w.sum(); v = (bs2[mask] * w).sum(0) / w.sum() - m ** 2
            return m / np.sqrt(v)
        si, so = srs(ins), srs(~ins)
        best = si.argmax()
        rank = (so < so[best]).sum() + 0.5
        w_ = rank / (len(so) + 1)
        logits.append(np.log(w_ / (1 - w_)))
    logits = np.array(logits)
    pbo = (logits <= 0).mean()
    gate("G4 PBO <= 0.30", pbo <= 0.30, f"PBO {pbo:.2f} over {comb(S, S//2)} CSCV splits")

    # ---------- G5 rolling walk-forward with refit ----------
    qs = pd.date_range("2023-01-01", RESEARCH_END, freq="QS")
    wf, picks = [], []
    for a, b in zip(qs, list(qs[1:]) + [pd.Timestamp(RESEARCH_END) + pd.Timedelta(days=1)]):
        train = M.loc[a - pd.Timedelta(days=730): a - pd.Timedelta(days=1)]
        best = train.apply(sh).idxmax(); picks.append(best)
        wf.append(M.loc[a: b - pd.Timedelta(days=1), best])
    wf = pd.concat(wf)
    wf.to_frame("ret").to_parquet(OUT / "walkforward_oos.parquet")
    log(f"Walk-forward picks per quarter: {picks}")
    log(f"Same window, pre-registered candidate Sharpe {sh(cand.loc[wf.index[0]:]):.2f}; grid median {M.loc[wf.index[0]:].apply(sh).median():.2f}")
    gate("G5 walk-forward OOS Sharpe >= 0.50", sh(wf) >= 0.5, f"WF OOS 2023-01→2026-03 Sharpe {sh(wf):.2f}")

    # ---------- G6 cost stress, G7 latency, G12 capacity ----------
    def var(**kw):
        return run(d, replace(CAND, **kw))
    c2, c3 = sh(var(cost_mult=2.0)), sh(var(cost_mult=3.0))
    gate("G6 cost stress", c2 >= 0.5 and c3 > 0, f"Sharpe 1x {p['sharpe']:.2f} / 2x {c2:.2f} / 3x {c3:.2f}")
    l1 = sh(var(lag=1))
    gate("G7 +1 day latency keeps >= 60%", l1 >= 0.6 * p["sharpe"], f"lag0 {p['sharpe']:.2f} -> lag1 {l1:.2f} ({l1/p['sharpe']*100 if p['sharpe'] else 0:.0f}%)")
    caps = {c: sh(var(capital=c)) for c in (1e6, 1e7, 5e7)}
    log(f"Capacity: $100k {p['sharpe']:.2f} | " + " | ".join(f"${c/1e6:.0f}M {v:.2f}" for c, v in caps.items()))
    gate("G12 capacity ($1M drop <= 0.10)", p["sharpe"] - caps[1e6] <= 0.10, f"$1M Sharpe {caps[1e6]:.2f}")
    nofund = sh(var(use_funding=False))
    lo, so_ = var(legs="long"), var(legs="short")
    log(f"Without funding P&L: {nofund:.2f}.  Long leg only: Sharpe {sh(lo):.2f} CAGR {perf(lo)['cagr']*100:.1f}% | short leg only: Sharpe {sh(so_):.2f} CAGR {perf(so_)['cagr']*100:.1f}%")
    wk = [sh(sleeve(d, replace(CAND, tranches=1), o)) for o in range(7)]
    log(f"Single-weekday sleeves (the old weekday lottery): {[round(x, 2) for x in wk]} -> tranche ensemble {p['sharpe']:.2f}")

    # ---------- G8 stability ----------
    lr = np.log1p(cand)
    yr = lr.groupby(lr.index.year).sum()
    share = yr.max() / lr.sum() if lr.sum() > 0 else np.inf
    log("Yearly: " + "  ".join(f"{y}: {np.expm1(v)*100:+.1f}% (Sh {sh(cand[cand.index.year == y]):.2f})" for y, v in yr.items()))
    gate("G8 stability", share <= 0.5 and (yr > 0).sum() >= 4, f"best year = {share*100:.0f}% of log P&L; positive years {(yr>0).sum()}/{len(yr)}")
    for k in (10, 20):
        log(f"  log P&L without best {k} days: {lr.sum() - lr.nlargest(k).sum():.3f} (total {lr.sum():.3f})")

    # regimes
    btc = d.close["BTCUSDT"]; bt = (btc / btc.shift(90) - 1).shift(1); bv = d.ret["BTCUSDT"].rolling(30).std().shift(1)
    df = pd.DataFrame({"r": cand, "tr": pd.cut(bt, [-9, -0.15, 0.15, 9], labels=["bear", "chop", "bull"]),
                       "vr": pd.qcut(bv.loc[cand.index], 3, labels=["low", "mid", "high"]), "btc": d.ret["BTCUSDT"]}).dropna()
    log("Regime Sharpe (BTC 90d trend): " + str(df.groupby("tr", observed=True).r.apply(sh).round(2).to_dict()))
    log("Regime Sharpe (BTC 30d vol tercile): " + str(df.groupby("vr", observed=True).r.apply(sh).round(2).to_dict()))
    log(f"Correlation to BTC {df.r.corr(df.btc):.2f}; beta {np.polyfit(df.btc, df.r, 1)[0]:.3f}")

    # ---------- G9 random-name null ----------
    rng = np.random.default_rng(42)
    nulls = []
    for _ in range(100):
        def factory(off):
            def pick(i, idx, n):
                p_ = rng.permutation(idx); return p_[:n], p_[n:2 * n]
            return pick
        nulls.append(sh(run(d, CAND, picker_factory=factory)))
    nulls = np.array(nulls)
    pv = (1 + (nulls >= p["sharpe"]).sum()) / (len(nulls) + 1)
    gate("G9 beats random-name null", pv < 0.05, f"null Sharpe mean {nulls.mean():.2f} sd {nulls.std():.2f}; p = {pv:.3f}")

    # ---------- G10 neighbourhood ----------
    def nb(par, vals):
        i = vals.index(getattr(CAND, par)); return [vals[j] for j in (i - 1, i + 1) if 0 <= j < len(vals)]
    neigh = [replace(CAND, **{par: v}).key() for par, vals in GRID.items() for v in nb(par, vals)]
    ns = gs[neigh]
    frac = (ns >= 0.5 * p["sharpe"]).mean() if p["sharpe"] > 0 else 0.0
    gate("G10 parameter neighbourhood", frac >= 0.7, f"neighbours {ns.round(2).to_dict()} -> {frac*100:.0f}% >= half of candidate")

    # ---------- G13 leverage / ruin ----------
    x = cand.to_numpy(); rngb = np.random.default_rng(0)
    paths = np.stack([np.concatenate([x[s:s + 20] for s in rngb.integers(0, len(x) - 20, 19)])[:365] for _ in range(10000)])
    lev_ok = 0.0
    for L in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        eq = np.cumprod(1 + np.maximum(paths * L, -0.99), axis=1)
        mdd = (eq / np.maximum.accumulate(eq, axis=1) - 1).min(1)
        pdd = (mdd < -0.30).mean()
        log(f"  {L}x gross: 1y median {np.median(eq[:, -1]-1)*100:+.0f}%  p5 {np.percentile(eq[:, -1]-1, 5)*100:+.0f}%  P(loss) {(eq[:, -1] < 1).mean()*100:.0f}%  P(DD>30%) {pdd*100:.1f}%  P(DD>50%) {(mdd < -0.5).mean()*100:.1f}%")
        if pdd <= 0.10:
            lev_ok = L
    gate("G13 risk at chosen leverage", lev_ok >= 1.0, f"max leverage with P(1y DD>30%) <= 10%: {lev_ok}x", warn=lev_ok < 1.0 and lev_ok > 0)

    # ---------- G14 reproducibility ----------
    again = run(d, CAND)
    diff = float((again - cand).abs().max())
    gate("G14 reproducibility", diff < 1e-9, f"max abs diff on re-run {diff:.1e}; env pinned in requirements.txt")

    # ---------- freeze ----------
    frozen = dict(config=as_dict(CAND), research_end=RESEARCH_END, frozen_at=pd.Timestamp.now().isoformat(),
                  research_returns_sha256=hashlib.sha256(cand.round(12).to_numpy().tobytes()).hexdigest(),
                  research_sharpe=p["sharpe"], walkforward_last_pick=picks[-1])
    (ROOT / "reports" / "frozen_config.json").write_text(json.dumps(frozen, indent=2))
    cand.to_frame("ret").to_parquet(OUT / "candidate_research_returns.parquet")
    log("\n=== GATE SUMMARY (research period) ===")
    for k, (v, e) in gates.items():
        log(f"{v:4s}  {k}  |  {e}")
    (OUT / "gates_research.json").write_text(json.dumps(gates, indent=2))
    (OUT / "validate_pit_log.txt").write_text("\n".join(log_lines))
    log(f"\nruntime {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
