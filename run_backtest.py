"""Backtest an MA crossover on the MT5 USDJPY data.

Prints metrics, long vs short, portfolio by day, the trade log and P&L by month as tables, and saves
the equity curve (PNG) plus metrics, the full trade log, daily portfolio and per-bar history (CSV).

    python run_backtest.py                       # 5min bars, 20/50 SMA, 10,000 units, market orders
    python run_backtest.py --timeframe 1h --fast 10 --slow 30 --long-only --show
    python run_backtest.py --sl 20 --tp 40 --sizing risk --risk-pct 1 --slippage 2
    python run_backtest.py --entry limit --entry-offset 3 --expiry 6 --trail 15
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from functools import partial
from pathlib import Path

import pandas as pd
from rich.console import Console

from backtest_engine import MovingAverageCrossover, point_size
from backtest_engine.broker import FixedSlippage, RandomSlippage
from backtest_engine.data_loader import DEFAULT_FILE, MT5_SERVER_TZ, load_ohlcv, resample_ohlcv
from backtest_engine.optimize import RunConfig
from backtest_engine.orders import OrderPolicy
from backtest_engine.report import money, print_report, save_csv
from backtest_engine.sizing import FixedUnits, PercentEquity, RiskPercent


def add_common_args(p: argparse.ArgumentParser) -> None:
    """Data, costs, orders and sizing: shared by run_backtest.py and run_walkforward.py."""
    g = p.add_argument_group("data")
    g.add_argument("--file", default=str(DEFAULT_FILE), help="MT5 CSV export")
    g.add_argument("--symbol", default="USDJPY")
    g.add_argument("--timeframe", default="5min", help='bar size, e.g. "1min", "5min", "1h", "4h" (New York-aligned)')
    g.add_argument("--cash", type=float, default=10_000, help="starting balance, USD")
    g.add_argument("--long-only", action="store_true", help="go flat instead of short")
    g.add_argument("--sides", choices=["both", "long"], default="both",
                   help='"long" is the same as --long-only (used by the VS Code picker)')

    g = p.add_argument_group("slippage (points, always against you)")
    g.add_argument("--slippage", type=float, default=0, help="fixed slippage per fill, e.g. 2")
    g.add_argument("--slippage-max", type=float, default=0, help="random slippage between 0 and this per fill")
    g.add_argument("--seed", type=int, default=0, help="seed for random slippage")

    g = p.add_argument_group("orders (distances in pips)")
    g.add_argument("--entry", choices=["market", "limit", "stop"], default="market",
                   help="entry order type: market (next open), limit (better price), stop (breakout)")
    g.add_argument("--entry-offset", type=float, default=0, help="limit/stop entry distance from the signal close")
    g.add_argument("--expiry", type=int, default=None, help="cancel an unfilled limit/stop entry after N bars")
    g.add_argument("--sl", type=float, default=None, help="stop-loss distance (0 = off)")
    g.add_argument("--tp", type=float, default=None, help="take-profit distance (0 = off)")
    g.add_argument("--trail", type=float, default=None, help="trailing-stop distance (0 = off)")

    g = p.add_argument_group("position sizing")
    g.add_argument("--sizing", choices=["fixed", "equity", "risk"], default="fixed",
                   help="fixed units, %% of equity (notional), or %% of equity risked to the stop-loss")
    g.add_argument("--units", type=float, default=10_000, help="fixed size in units (10,000 = 0.1 lot)")
    g.add_argument("--equity-pct", type=float, default=100, help="--sizing equity: notional as %% of equity")
    g.add_argument("--risk-pct", type=float, default=1.0, help="--sizing risk: %% of equity lost if the stop is hit")
    g.add_argument("--leverage", type=float, default=30, help="max position size as a multiple of equity")
    g.add_argument("--lot-step", type=float, default=1_000, help="round sizes down to this many units (1,000 = micro lot)")


def check_common_args(p: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    args.long_only = args.long_only or args.sides == "long"
    if args.sizing == "risk" and not (args.sl or args.trail):
        p.error("--sizing risk needs --sl (or --trail) to know how much each trade risks")
    if args.entry != "market" and args.entry_offset < 0:
        p.error("--entry-offset must be >= 0")


def load_bars(args) -> pd.DataFrame:
    df = load_ohlcv(args.file, tz=MT5_SERVER_TZ)
    return df if args.timeframe in ("1min", "1m", "") else resample_ohlcv(df, args.timeframe)


def build_config(args, point: float) -> RunConfig:
    if args.sizing == "fixed":
        sizer = None  # exactly --units every trade
    else:
        common = dict(lot_step=args.lot_step, min_units=args.lot_step, max_leverage=args.leverage)
        sizer = (PercentEquity(pct=args.equity_pct, **common) if args.sizing == "equity"
                 else RiskPercent(pct=args.risk_pct, **common))
    slippage = (RandomSlippage(args.slippage_max, args.seed) if args.slippage_max
                else FixedSlippage(args.slippage) if args.slippage else None)
    orders = OrderPolicy(entry=args.entry, entry_offset_pips=args.entry_offset, expiry_bars=args.expiry,
                         stop_loss_pips=args.sl or None, take_profit_pips=args.tp or None,
                         trailing_stop_pips=args.trail or None)
    return RunConfig(symbol=args.symbol, point=point, cash=args.cash, units=args.units, sizer=sizer,
                     orders=orders, slippage=slippage,
                     make_strategy=partial(MovingAverageCrossover, long_only=args.long_only))


def run_tag(args) -> str:
    """Short file-name suffix for the non-default settings, so runs don't overwrite each other."""
    tags = ["long"] if args.long_only else []
    if args.entry != "market":
        tags.append(f"{args.entry}{args.entry_offset:g}")
    tags += [f"{k}{v:g}" for k, v in [("sl", args.sl), ("tp", args.tp), ("trail", args.trail)] if v]
    if args.sizing == "risk":
        tags.append(f"risk{args.risk_pct:g}")
    elif args.sizing == "equity":
        tags.append(f"eq{args.equity_pct:g}")
    if args.slippage_max:
        tags.append(f"slipmax{args.slippage_max:g}")
    elif args.slippage:
        tags.append(f"slip{args.slippage:g}")
    return "".join(f"_{t}" for t in tags)


def settings_panel(args, config: RunConfig, df: pd.DataFrame, result) -> dict[str, str]:
    return {
        "Bars": f"{len(df):,} x {args.timeframe}  ({df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d} UTC)",
        "Start": money(args.cash),
        "Sides": "long only" if args.long_only else "long and short",
        "Sizing": result.meta["sizing"],
        "Orders": result.meta["orders"],
        "Slippage": result.meta["slippage"],
        "Spread": "MT5 spread per bar, paid on every buy",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fast", type=int, default=20, help="fast SMA length (bars)")
    p.add_argument("--slow", type=int, default=50, help="slow SMA length (bars)")
    add_common_args(p)
    g = p.add_argument_group("output")
    g.add_argument("--days", type=int, default=15, help="days shown in the portfolio table")
    g.add_argument("--trades", default="20", help='trades shown in the trade log: a number, or "all"')
    g.add_argument("--out", default="backtest_results", help="folder for the chart and CSV files")
    g.add_argument("--show", action="store_true", help="also open the chart in a zoomable window")
    g.add_argument("--open", action="store_true", help="open the saved chart in a VS Code tab")
    args = p.parse_args()
    check_common_args(p, args)
    if args.trades != "all" and not args.trades.isdigit():
        p.error('--trades must be a number or "all"')
    args.trades = None if args.trades == "all" else int(args.trades)
    return args


def main() -> None:
    args = parse_args()
    console = Console()

    df = load_bars(args)
    point = point_size(df["close"].median())
    config = build_config(args, point)
    title = (f"{args.symbol} {args.timeframe} - SMA {args.fast}/{args.slow} crossover"
             f"{' (long only)' if args.long_only else ''}")

    started = time.perf_counter()
    result = config.run(df, {"fast": args.fast, "slow": args.slow}, title=title)
    elapsed = time.perf_counter() - started

    settings = settings_panel(args, config, df, result)
    settings["Run time"] = f"{elapsed:.2f}s ({len(df) / elapsed:,.0f} bars/s)"
    print_report(result, console, decimals=3 if point == 0.001 else 5, days=args.days, trades=args.trades,
                 settings=settings)

    stem = f"{args.symbol}_{args.timeframe}_sma{args.fast}-{args.slow}{run_tag(args)}"
    chart = Path(args.out) / f"{stem}_equity.png"
    result.plot_equity(chart, show=args.show)
    files = [chart, *save_csv(result, args.out, stem)]
    console.print("[bold cyan]Saved[/] (Ctrl+click a path to open it)\n" + "\n".join(f"  {f}" for f in files))
    if args.open:
        open_in_vscode(chart, console)


def open_in_vscode(path: Path, console: Console) -> None:
    """Open a file as a VS Code tab (reuses the current window); no-op if the `code` CLI is missing."""
    code = shutil.which("code")
    if code is None:
        console.print("[dim]VS Code's `code` command isn't on PATH; open the chart from the path above.[/]")
        return
    subprocess.run([code, "--reuse-window", str(path.resolve())], check=False)


if __name__ == "__main__":
    main()
