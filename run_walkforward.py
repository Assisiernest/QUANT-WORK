"""Out-of-sample and walk-forward testing of the MA crossover on the MT5 USDJPY data.

Out-of-sample: search the parameter grid on the first part of the data, then trade the
best settings, unchanged, on the rest. Walk-forward: repeat that in windows, re-tuning
before each test window, and chain the test windows into one out-of-sample result.

    python run_walkforward.py                                  # walk-forward, 4 rolling windows
    python run_walkforward.py --mode oos --split 0.7           # one in-sample / out-of-sample split
    python run_walkforward.py --fast 5,10,20 --slow 50,100,200 --objective net_pnl --anchored
    python run_walkforward.py --sl 20 --tp 40 --sizing risk --slippage 2   # same cost/order/sizing options

Takes the same data, slippage, order and sizing options as run_backtest.py.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from backtest_engine import point_size
from backtest_engine.optimize import OBJECTIVES, out_of_sample, walk_forward
from backtest_engine.report import print_out_of_sample, print_report, print_walk_forward, save_csv
from run_backtest import add_common_args, build_config, check_common_args, load_bars, open_in_vscode, run_tag


def int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("test")
    g.add_argument("--mode", choices=["walkforward", "oos"], default="walkforward")
    g.add_argument("--fast", type=int_list, default=[5, 10, 20, 30, 50], help="fast SMA lengths to try, comma-separated")
    g.add_argument("--slow", type=int_list, default=[50, 100, 150, 200], help="slow SMA lengths to try, comma-separated")
    g.add_argument("--objective", choices=OBJECTIVES, default="sharpe", help="what the parameter search maximises")
    g.add_argument("--min-trades", type=int, default=10, help="ignore settings with fewer in-sample trades")
    g.add_argument("--split", type=float, default=0.7, help="--mode oos: fraction of bars that are in-sample")
    g.add_argument("--windows", type=int, default=4, help="--mode walkforward: number of test windows")
    g.add_argument("--train", type=float, default=0.5, help="--mode walkforward: fraction of bars in the first training window")
    g.add_argument("--anchored", action="store_true", help="--mode walkforward: training always starts at the first bar")
    g.add_argument("--top", type=int, default=10, help="parameter combinations shown")
    add_common_args(p)
    g = p.add_argument_group("output")
    g.add_argument("--trades", default="20", help='trades shown in the trade log: a number, or "all"')
    g.add_argument("--out", default="backtest_results", help="folder for the chart and CSV files")
    g.add_argument("--open", action="store_true", help="open the saved chart in a VS Code tab")
    args = p.parse_args()
    check_common_args(p, args)
    if not 0 < args.split < 1 or not 0 < args.train < 1:
        p.error("--split and --train must be between 0 and 1")
    args.trades = None if args.trades == "all" else int(args.trades)
    return args


def main() -> None:
    args = parse_args()
    console = Console()
    df = load_bars(args)
    point = point_size(df["close"].median())
    config = build_config(args, point)
    grid = {"fast": args.fast, "slow": args.slow}
    combos = sum(f < s for f in args.fast for s in args.slow)
    decimals = 3 if point == 0.001 else 5
    stem = f"{args.symbol}_{args.timeframe}_{args.mode}{run_tag(args)}"

    console.print(Panel.fit(
        f"[bold]{args.symbol} {args.timeframe} SMA crossover: "
        f"{'walk-forward' if args.mode == 'walkforward' else 'out-of-sample'}[/]\n"
        f"Bars:      {len(df):,}  ({df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d} UTC)\n"
        f"Grid:      fast {args.fast} x slow {args.slow} = {combos} valid combinations\n"
        f"Objective: {args.objective} (min {args.min_trades} trades)\n"
        + (f"Windows:   {args.windows} test windows, first {args.train:.0%} of bars for training, "
           f"{'anchored' if args.anchored else 'rolling'}\n" if args.mode == "walkforward"
           else f"Split:     first {args.split:.0%} of bars in-sample\n")
        + f"Orders:    {config.orders.describe()}\n"
          f"Sizing:    {(config.sizer.describe() if config.sizer else f'fixed {args.units:,.0f} units')}\n"
          f"Slippage:  {config.slippage.describe() if config.slippage else 'none'}",
        title="Out-of-sample test", border_style="cyan"))

    started = time.perf_counter()
    if args.mode == "oos":
        oos = out_of_sample(df, grid, config, split=args.split, objective=args.objective, min_trades=args.min_trades)
        console.print(f"[dim]{time.perf_counter() - started:.1f}s[/]")
        print_out_of_sample(oos, console, top=args.top)
        console.rule("Out-of-sample period in detail")
        print_report(oos.out_of_sample, console, decimals=decimals, trades=args.trades)
        chart_result, detail = oos.full, oos.out_of_sample
        tables = {"search": oos.table, "comparison": oos.comparison()}
    else:
        wf = walk_forward(df, grid, config, windows=args.windows, train_frac=args.train, anchored=args.anchored,
                          objective=args.objective, min_trades=args.min_trades)
        console.print(f"[dim]{time.perf_counter() - started:.1f}s[/]")
        print_walk_forward(wf, console)
        console.rule("Chained out-of-sample windows in detail")
        print_report(wf.combined, console, decimals=decimals, trades=args.trades)
        chart_result, detail = wf.combined, wf.combined
        tables = {"windows": wf.windows}

    out = Path(args.out)
    chart = out / f"{stem}_equity.png"
    chart_result.plot_equity(chart)
    files = [chart, *save_csv(detail, out, stem)]
    for name, table in tables.items():
        path = out / f"{stem}_{name}.csv"
        table.round(6).to_csv(path)
        files.append(path)
    console.print("[bold cyan]Saved[/] (Ctrl+click a path to open it)\n" + "\n".join(f"  {f}" for f in files))
    if args.open:
        open_in_vscode(chart, console)


if __name__ == "__main__":
    main()
