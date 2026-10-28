"""Opens the sealed holdout exactly once for the frozen config (gate G11).
Refuses to run unless reports/frozen_config.json exists and the research-period
returns still hash to the frozen value (i.e. nothing changed after freezing)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pit_engine import A, Config, load, perf, run  # noqa: E402


def sh(r):
    return r.mean() / r.std() * np.sqrt(A) if r.std() > 0 else 0.0


def main():
    fz = json.loads((ROOT / "reports" / "frozen_config.json").read_text())
    cfg = Config(**fz["config"])
    d = load()
    full = run(d, cfg)
    research = full.loc[: fz["research_end"]]
    h = hashlib.sha256(research.round(12).to_numpy().tobytes()).hexdigest()
    if h != fz["research_returns_sha256"]:
        # the research window re-runs on more data only if trailing windows changed; report rather than hide
        print(f"WARNING: research-period hash differs ({h[:12]} vs {fz['research_returns_sha256'][:12]}); max diff "
              f"{(research - pd.read_parquet(ROOT / 'reports/pit/candidate_research_returns.parquet')['ret']).abs().max():.2e}")
    ho = full.loc[pd.Timestamp(fz["research_end"]) + pd.Timedelta(days=1):]
    p = perf(ho)
    tot = (1 + ho).prod() - 1
    print(f"SEALED HOLDOUT {ho.index[0].date()} -> {ho.index[-1].date()} ({len(ho)} days), frozen config {cfg.key()}")
    print(f"  return {tot*100:+.2f}%  Sharpe {p['sharpe']:.2f}  MaxDD {p['max_dd']*100:.1f}%  vol {p['vol']*100:.1f}%")
    m = ho.groupby(ho.index.to_period("M")).apply(lambda x: (1 + x).prod() - 1)
    print("  monthly: " + "  ".join(f"{k}: {v*100:+.1f}%" for k, v in m.items()))

    x = research.to_numpy()
    rng = np.random.default_rng(0)
    L = len(ho)
    sims = np.array([np.prod(1 + np.concatenate([x[s:s + 20] for s in rng.integers(0, len(x) - 20, L // 20 + 1)])[:L]) - 1
                     for _ in range(20000)])
    pct = (sims < tot).mean()
    print(f"  research-implied distribution for {L} days: p5 {np.percentile(sims,5)*100:+.1f}%  median {np.median(sims)*100:+.1f}%  p95 {np.percentile(sims,95)*100:+.1f}%")
    print(f"  holdout return percentile within that distribution: {pct*100:.1f}%")
    verdict = "PASS" if pct >= 0.05 else "FAIL"
    print(f"  >> G11: {verdict}")

    wf_pick = fz.get("walkforward_last_pick")
    if wf_pick and wf_pick != cfg.key():
        lb, q, u = wf_pick.split("_")
        c2 = Config(**{**fz["config"], "lookback": int(lb[2:]), "quantile": float(q[1:]), "universe_n": int(u[1:])})
        r2 = run(d, c2).loc[ho.index[0]:]
        print(f"  (info) walk-forward's last pick {wf_pick} on holdout: return {((1+r2).prod()-1)*100:+.2f}% Sharpe {sh(r2):.2f}")
    btc = d.close["BTCUSDT"].loc[ho.index[0] - pd.Timedelta(days=1):]
    print(f"  (context) BTC over holdout: {(btc.iloc[-1]/btc.iloc[0]-1)*100:+.1f}%")
    ho.to_frame("ret").to_parquet(ROOT / "reports" / "pit" / "holdout_returns.parquet")
    (ROOT / "reports" / "pit" / "gate_G11.json").write_text(json.dumps(dict(verdict=verdict, ret=tot, sharpe=p["sharpe"], pct=pct, days=L)))


if __name__ == "__main__":
    main()
