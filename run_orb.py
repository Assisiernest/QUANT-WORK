"""Run the ORB + ICT/SMC strategy (orb_ict_smc_bot.py) on a CSV of bars.

    # one run with the defaults in orb_ict_smc_bot.CONFIG (change any with --set key=value)
    python run_orb.py data/USDJPY.csv --symbol USDJPY --session NY
    python run_orb.py data/USDJPY.csv --set entry_depth=0.5 --set ict_filter=fvg+sweep

    # the pack's engine sanity check: first 5m candle's direction, 10R or 16:00 exit, no costs
    python run_orb.py data/USDJPY.csv --sanity

    # the 96-version grid for one session: selected on the first 2/3, confirmed on the last 1/3,
    # every version also re-run at 2x costs
    python run_orb.py data/USDJPY.csv --grid --session NY --jobs 4

    # NQ futures: 1 tick (0.25) slippage on stops, $4.50 round trip = 0.225 points per $20/pt contract
    python run_orb.py nq_1m.csv --symbol NQ --kind cfd --tz America/New_York --point 0.25 \
        --slippage-points 1 --commission 0.225 --grid --session NY

Results are in R (multiples of the planned risk = entry to stop), after the costs in the data
(spread column), slippage and commission. Pass criteria (pack, "Validation"): on held-out data
average R > 0 with the 95% CI above 0, still > 0 at 2x costs, at least 150 trades, and the
Deflated Sharpe Ratio check (all 576 versions count as trials).
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import NormalDist

import pandas as pd

from backtest_engine.analytics import probabilistic_sharpe
from backtest_engine.broker import FixedSlippage
from backtest_engine.data_loader import MT5_SERVER_TZ, load_ohlcv, resample_ohlcv
from backtest_engine.multi import Asset
from backtest_engine.qcalgo import run_algorithm

import orb_ict_smc_bot as bot

GRID = {
    "range_minutes": [5, 15],
    "entry_depth": [0.0, 1 / 3, 1 / 2],
    "target_r": [1.2, 1.3],
    "level_variant": ["plain", "session", "day", "all"],
    "ict_filter": ["off", "on"],   # "on" = --ict-on (default fvg)
}
TOTAL_TRIALS = 576  # the pack's full grid: 96 per asset/session x 2 assets x 3 sessions
MIN_TRADES = 150

_DF: pd.DataFrame | None = None
_ARGS = None


# --- running ------------------------------------------------------------------------------------
def load(args) -> pd.DataFrame:
    df = load_ohlcv(args.csv, tz=args.tz)
    if args.resample:
        df = resample_ohlcv(df, args.resample)
    return df


def _init(args) -> None:
    global _DF, _ARGS
    _ARGS, _DF = args, load(args)


def run_one(overrides: dict, cost_mult: float = 1.0) -> tuple[pd.DataFrame, dict, object]:
    """One backtest. Returns (trade journal with R, skip counts, engine result)."""
    args, df = _ARGS, _DF
    if cost_mult != 1.0 and "spread" in df.columns:
        df = df.assign(spread=df.spread * cost_mult)
    cfg = {"symbol": args.symbol, "asset_kind": args.kind, "verbose": False,
           "commission_rt": args.commission * cost_mult, **overrides}
    cls = type("OrbRun", (bot.OrbIctSmc,), {"overrides": cfg})
    slip = FixedSlippage(args.slippage_points * cost_mult) if args.slippage_points else None
    run = run_algorithm(cls, [Asset(args.symbol, df, point=args.point)], slippage=slip)
    algo = cls.last
    return pd.DataFrame(algo.journal), dict(algo.skips), run


def _grid_task(item):
    key, overrides, cost_mult = item
    journal, skips, _ = run_one(overrides, cost_mult)
    return key, cost_mult, journal, skips


# --- statistics ---------------------------------------------------------------------------------
def r_stats(r: pd.Series) -> dict:
    n = len(r)
    mean = r.mean() if n else float("nan")
    sd = r.std() if n > 1 else float("nan")
    half = 1.96 * sd / math.sqrt(n) if n > 1 else float("nan")
    return {"trades": n, "avg_r": mean, "ci_low": mean - half, "ci_high": mean + half,
            "win_rate": (r > 0).mean() if n else float("nan"), "total_r": r.sum()}


def deflated_sharpe(r: pd.Series, trial_sharpes: list[float], n_trials: int) -> float:
    """Bailey & Lopez de Prado: probability the per-trade Sharpe beats the best Sharpe you'd
    expect from n_trials tries of a strategy with no edge."""
    sr = pd.Series(trial_sharpes).dropna()
    if len(r) < 3 or len(sr) < 2:
        return float("nan")
    var = sr.var()
    gamma = 0.5772156649
    z = NormalDist().inv_cdf
    sr0 = math.sqrt(var) * ((1 - gamma) * z(1 - 1 / n_trials) + gamma * z(1 - 1 / (n_trials * math.e)))
    return probabilistic_sharpe(r, benchmark_sharpe=sr0)


def split_time(df: pd.DataFrame) -> pd.Timestamp:
    """First 2/3 of the bars select, the last 1/3 confirm."""
    return df.index[int(len(df) * 2 / 3)]


def summary(journal: pd.DataFrame, cut: pd.Timestamp) -> dict:
    if journal.empty:
        return {f"{p}_{k}": float("nan") if k != "trades" else 0 for p in ("is", "oos")
                for k in r_stats(pd.Series(dtype=float))}
    t = pd.to_datetime(journal.entry_time, utc=True)
    out = {}
    for part, mask in (("is", t < cut), ("oos", t >= cut)):
        out.update({f"{part}_{k}": v for k, v in r_stats(journal.r[mask]).items()})
    return out


# --- commands -----------------------------------------------------------------------------------
def cmd_single(args, overrides: dict) -> None:
    journal, skips, run = run_one(overrides)
    cfg = {**bot.CONFIG, **overrides}
    print(f"Config: {', '.join(f'{k}={v}' for k, v in cfg.items() if k not in ('smc', 'news_times'))}")
    print(f"ICT/SMC settings: {cfg['smc'].describe()}")
    if journal.empty:
        print("No trades.", "Skipped:", skips)
        return
    st = r_stats(journal.r)
    print(f"\n{st['trades']} trades | avg {st['avg_r']:+.3f}R (95% CI {st['ci_low']:+.3f} to {st['ci_high']:+.3f}) "
          f"| win rate {st['win_rate']:.1%} | total {st['total_r']:+.1f}R")
    print("\nBy session / setup / exit:")
    print(journal.groupby(["session", "setup", "exit_reason"]).r.agg(["count", "mean"]).round(3).to_string())
    print("\nWith vs without each ICT fact at entry (filters only change trades when switched on):")
    for col in ("ict_fvg", "ict_block"):
        print(f"  {col}: " + ", ".join(f"{k}: {len(g)} trades {g.r.mean():+.3f}R" for k, g in journal.groupby(col)))
    same = int(journal.same_bar_stop.sum())
    if same:
        print(f"\nNote: {same} trades filled and touched their stop in the same bar; the engine starts "
              "checking exits on the next bar, so those may be optimistic. Use 1-minute data.")
    print("\nSkipped / cancelled setups:", skips)
    cut = split_time(_DF)
    s = summary(journal, cut)
    print(f"\nIn-sample (to {cut:%Y-%m-%d}): {s['is_trades']} trades {s['is_avg_r']:+.3f}R | "
          f"out-of-sample: {s['oos_trades']} trades {s['oos_avg_r']:+.3f}R")
    stats = run.result.combined.stats()
    print(f"Engine: net profit {stats.get('net_pnl', float('nan')):,.2f}, max drawdown "
          f"{stats.get('max_drawdown_pct', float('nan')):.2f}%")
    if args.out:
        journal.to_csv(args.out, index=False)
        print(f"Journal saved to {args.out}")


def cmd_grid(args, base: dict) -> None:
    ict_on = args.ict_on
    versions = []
    for values in itertools.product(*GRID.values()):
        ov = dict(zip(GRID, values))
        ov["ict_filter"] = ict_on if ov["ict_filter"] == "on" else "off"
        versions.append({**base, **ov})
    tasks = [(k, ov, c) for k, ov in enumerate(versions) for c in (1.0, 2.0)]
    print(f"Running {len(versions)} versions x 2 cost levels = {len(tasks)} backtests on {args.jobs} process(es)...")
    results: dict[tuple[int, float], pd.DataFrame] = {}
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs, initializer=_init, initargs=(args,)) as pool:
            for n, (k, c, journal, _) in enumerate(pool.map(_grid_task, tasks, chunksize=2), 1):
                results[(k, c)] = journal
                print(f"\r  {n}/{len(tasks)}", end="", flush=True)
    else:
        for n, t in enumerate(tasks, 1):
            k, c, journal, _ = _grid_task(t)
            results[(k, c)] = journal
            print(f"\r  {n}/{len(tasks)}", end="", flush=True)
    print()

    cut = split_time(_DF)
    rows = []
    for k, ov in enumerate(versions):
        row = {g: ov[g] for g in GRID}
        row["entry_depth"] = {0.0: "edge", 1 / 3: "1/3", 1 / 2: "1/2"}[ov["entry_depth"]]
        row.update(summary(results[(k, 1.0)], cut))
        row.update({f"{key}_2x": v for key, v in summary(results[(k, 2.0)], cut).items()
                    if key.endswith(("avg_r", "ci_low"))})
        j = results[(k, 1.0)]
        is_r = j.r[pd.to_datetime(j.entry_time, utc=True) < cut] if not j.empty else pd.Series(dtype=float)
        row["is_sharpe"] = is_r.mean() / is_r.std() if len(is_r) > 2 and is_r.std() > 0 else float("nan")
        rows.append(row)
    table = pd.DataFrame(rows)

    # Select on in-sample only, then judge the pick on the held-out third.
    eligible = table[table.is_trades >= 30]
    if eligible.empty:
        print("No version has 30+ in-sample trades; not enough data to select.")
    else:
        best_k = eligible.is_avg_r.idxmax()
        best = table.loc[best_k]
        j = results[(best_k, 1.0)]
        t = pd.to_datetime(j.entry_time, utc=True)
        dsr = deflated_sharpe(j.r[t < cut], table.is_sharpe.tolist(), max(TOTAL_TRIALS, len(table)))
        checks = {
            "OOS avg R > 0": best.oos_avg_r > 0,
            "OOS 95% CI above 0": best.oos_ci_low > 0,
            "OOS avg R > 0 at 2x costs": best.oos_avg_r_2x > 0,
            f"OOS trades >= {MIN_TRADES}": best.oos_trades >= MIN_TRADES,
            "Deflated Sharpe >= 0.95": dsr >= 0.95,
        }
        print(f"\nSelected on in-sample (to {cut:%Y-%m-%d}): " +
              ", ".join(f"{g}={best[g]}" for g in GRID))
        print(f"  in-sample:  {int(best.is_trades)} trades, {best.is_avg_r:+.3f}R")
        print(f"  held-out:   {int(best.oos_trades)} trades, {best.oos_avg_r:+.3f}R "
              f"(95% CI {best.oos_ci_low:+.3f} to {best.oos_ci_high:+.3f}), at 2x costs {best.oos_avg_r_2x:+.3f}R")
        print(f"  Deflated Sharpe (in-sample, {max(TOTAL_TRIALS, len(table))} trials): {dsr:.3f}")
        for name, ok in checks.items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print("  Verdict:", "PASSES every check" if all(checks.values()) else "does not pass: no proven edge")
    out = args.out or f"backtest_results/orb_grid_{args.symbol}_{'-'.join(base['sessions'])}.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\nAll {len(table)} versions saved to {out}")


def parse_value(text: str):
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    if "/" in text and all(p.strip().replace(".", "", 1).isdigit() for p in text.split("/", 1)):
        a, b = text.split("/", 1)
        return float(a) / float(b)
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv")
    p.add_argument("--symbol", default="USDJPY")
    p.add_argument("--kind", default="forex", choices=["forex", "cfd", "equity", "crypto"])
    p.add_argument("--tz", default=MT5_SERVER_TZ, help=f"time zone of the file's timestamps (default {MT5_SERVER_TZ})")
    p.add_argument("--resample", help="e.g. 1min, if the file is finer than you want")
    p.add_argument("--point", type=float, help="price of one spread/slippage point (default: guessed)")
    p.add_argument("--slippage-points", type=float, default=0.0, help="slippage on stop and market fills, points")
    p.add_argument("--commission", type=float, default=0.0, help="round-trip commission per unit, price units")
    p.add_argument("--session", action="append", choices=["ASIA", "LONDON", "NY"],
                   help="session(s) to trade (repeat for several; default NY)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a CONFIG value")
    p.add_argument("--sanity", action="store_true", help="the published rule, no costs (expect about +0.13R on NQ)")
    p.add_argument("--grid", action="store_true", help="run the 96-version grid with in/out-of-sample checks")
    p.add_argument("--ict-on", default="fvg", help="what ict_filter=on means in the grid (default fvg)")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--out", help="CSV to save the journal (single run) or grid table to")
    args = p.parse_args(argv)

    overrides = {"sessions": tuple(args.session or ["NY"])}
    for item in args.set:
        key, _, value = item.partition("=")
        if key not in bot.CONFIG:
            sys.exit(f"Unknown setting {key!r}; see CONFIG in orb_ict_smc_bot.py")
        overrides[key] = parse_value(value)
    if args.sanity:
        overrides.update(mode="published", sessions=("NY",), range_minutes=5)
        args.slippage_points, args.commission = 0.0, 0.0
    _init(args)
    if args.sanity:
        global _DF
        if "spread" in _DF.columns:
            _DF = _DF.assign(spread=0.0)
    if args.grid:
        cmd_grid(args, overrides)
    else:
        cmd_single(args, overrides)


if __name__ == "__main__":
    main()
