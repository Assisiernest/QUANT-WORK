"""Reporting for a BacktestResult: terminal tables, equity-curve chart and CSV export.

    from backtest_engine.report import print_report, plot_equity, save_csv
    print_report(result)                         # metrics, long/short, portfolio by day, trade log, months
    print_trade_log(result, last=None)           # every trade
    plot_equity(result, "results/run.png")       # equity + balance, drawdown, position
    save_csv(result, "results", "run")           # run_metrics / _trade_log / _daily / _history .csv
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .engine import BacktestResult

# Chart colours (light surface). Categorical slots in fixed order, plus neutral ink.
SURFACE = "#fcfcfb"
INK, INK_2, INK_3, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
EQUITY = "#2a78d6"  # slot 1, blue
LONG = "#1baf7a"  # slot 3, aqua
SHORT = "#eb6834"  # slot 2, orange
DRAWDOWN = "#e34948"  # slot 8, red


# --- terminal tables --------------------------------------------------------
def money(x: float) -> str:
    return f"{'-' if x < 0 else ''}${abs(x):,.2f}"


def _signed(x: float, text: str) -> str:
    return f"[green]{text}[/]" if x > 0 else f"[red]{text}[/]" if x < 0 else text


def _position(units: float) -> str:
    if not units:
        return "[dim]flat[/]"
    return f"[green]long {units:,.0f}[/]" if units > 0 else f"[red]short {-units:,.0f}[/]"


# Metrics whose sign is good/bad news get coloured; the rest stay in plain ink.
_COLOURED = {"net_pnl", "return_pct", "cagr_pct", "total_pips", "expectancy", "expectancy_pips",
             "sharpe", "sortino", "calmar", "recovery_factor"}
_ALWAYS_RED = {"max_drawdown_pct", "max_drawdown_usd", "gross_loss", "largest_loss", "avg_loss", "avg_mae_pips",
               "spread_paid", "slippage_paid"}


def print_metrics(result: "BacktestResult", console=None) -> None:
    """Metrics in four panels (Returns, Risk, Trades, Exposure), side by side when they fit."""
    from rich.columns import Columns
    from rich.console import Console
    from rich.table import Table

    from .metrics import compute_metrics, format_value

    console = console or Console()
    panels = []
    for section, items in compute_metrics(result).items():
        table = Table(title=section, title_style="bold cyan", header_style="bold", show_header=False)
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")
        for m in items:
            text = format_value(m.value, m.kind)
            if m.key in _COLOURED and isinstance(m.value, (int, float)):
                text = _signed(m.value, text)
            elif m.key in _ALWAYS_RED and text != "-":
                text = f"[red]{text}[/]"
            table.add_row(m.name, text)
        panels.append(table)
    console.print("[bold]Metrics[/]")
    console.print(Columns(panels, padding=(0, 2)))


_EXIT_CODES = {"signal": "sig", "stop loss": "[red]SL[/]", "take profit": "[green]TP[/]",
               "trailing stop": "[yellow]TS[/]", "end of data": "[dim]end[/]",
               "margin call": "[bold red]MC[/]"}


def print_trade_log(result: "BacktestResult", console=None, last: int | None = 20, decimals: int = 3) -> None:
    """The trade log: every trade, or the last `last` of them. The CSV export has every column."""
    from rich.console import Console
    from rich.table import Table

    from .metrics import format_duration

    console = console or Console()
    log = result.trade_log()
    shown = log if not last else log.tail(last)
    title = f"Trade log ({'all ' if len(shown) == len(log) else f'last {len(shown)} of '}{len(log)} trades)"
    t = Table(title=title, title_style="bold cyan", header_style="bold",
              caption="MAE/MFE: worst/best open P&L while held (pips). Exit: sig = signal, SL = stop-loss, "
                      "TP = take-profit, TS = trailing stop, end = end of data.", caption_style="dim")
    for col, justify in [("#", "right"), ("Side", "left"), ("Units", "right"), ("Entry (UTC)", "left"),
                         ("Exit (UTC)", "left"), ("Held", "right"), ("Entry", "right"), ("Exit", "right"),
                         ("Pips", "right"), ("P&L", "right"), ("MAE", "right"), ("MFE", "right"),
                         ("Exit by", "left"), ("Balance", "right")]:
        t.add_column(col, justify=justify)
    for n, r in shown.iterrows():
        side = "[green]long[/]" if r.side == "long" else "[red]short[/]"
        t.add_row(str(n), side, f"{r.units:,.0f}", f"{r.entry_time:%m-%d %H:%M}", f"{r.exit_time:%m-%d %H:%M}",
                  format_duration(r.duration), f"{r.entry_price:.{decimals}f}", f"{r.exit_price:.{decimals}f}",
                  _signed(r.pips, f"{r.pips:+.1f}"), _signed(r.pnl, money(r.pnl)),
                  f"{r.mae_pips:.1f}", f"{r.mfe_pips:+.1f}", _EXIT_CODES.get(r.exit_reason, r.exit_reason),
                  money(r.balance))
    console.print(t)


def print_by_side(result: "BacktestResult", console=None) -> None:
    from rich.console import Console
    from rich.table import Table

    console = console or Console()
    t = Table(title="Long vs short", title_style="bold cyan", header_style="bold")
    for col in ["", "Trades", "Win rate", "Net P&L", "Pips", "Profit factor", "Avg trade", "Spread paid",
                "Slippage paid"]:
        t.add_column(col, justify="left" if not col else "right")
    for side, r in result.by_side().iterrows():
        t.add_row(f"[bold]{side}[/]" if side == "All" else side, f"{int(r.trades):,}", f"{r.win_rate_pct:.1f}%",
                  _signed(r.net_pnl, money(r.net_pnl)), _signed(r.pips, f"{r.pips:+,.1f}"),
                  f"{r.profit_factor:.2f}", _signed(r.avg_trade, money(r.avg_trade)), money(r.spread_paid),
                  money(r.slippage_paid))
    console.print(t)


def print_report(result: "BacktestResult", console=None, settings: dict[str, str] | None = None,
                 decimals: int = 3, days: int = 15, trades: int | None = 20) -> None:
    """Print the backtest as tables: settings, metrics, long vs short, portfolio by day,
    trade log (last `trades`, or all when trades is None/0) and P&L by month."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = console or Console()
    if settings:
        width = max(map(len, settings))
        body = "\n".join(f"{k + ':':<{width + 2}}{v}" for k, v in settings.items())
        console.print(Panel.fit(f"[bold]{result.title}[/]\n{body}", title="Backtest", border_style="cyan"))

    print_metrics(result, console)
    print_by_side(result, console)

    daily = result.daily().tail(days)
    book = Table(title=f"Portfolio by day (last {len(daily)}, at 17:00 New York close)",
                 title_style="bold cyan", header_style="bold")
    for col in ["Trading day", "Close", "Position", "Balance", "Open P&L", "Equity", "Day P&L",
                "Closed", "Drawdown"]:
        book.add_column(col, justify="left" if col in ("Trading day", "Position") else "right")
    for day, r in daily.iterrows():
        book.add_row(f"{day:%a %Y-%m-%d}", f"{r.close:.{decimals}f}", _position(r.position),
                     money(r.balance), _signed(r.open_pnl, money(r.open_pnl)), money(r.equity),
                     _signed(r.day_pnl, money(r.day_pnl)), str(int(r.trades_closed)),
                     f"{r.drawdown_pct:.2f}%")
    console.print(book)

    print_trade_log(result, console, last=trades, decimals=decimals)

    m = Table(title="P&L by month (trade exit)", title_style="bold cyan", header_style="bold")
    for col in ["Month", "Trades", "Wins", "Pips", "P&L", "Equity at month end"]:
        m.add_column(col, justify="left" if col == "Month" else "right")
    month_of = lambda idx: pd.DatetimeIndex(idx).tz_localize(None).to_period("M")  # noqa: E731
    month_end = result.equity.groupby(month_of(result.equity.index)).last()
    by_month = result.trades.groupby(month_of(result.trades.exit_time))
    for month, g in by_month:
        m.add_row(str(month), str(len(g)), str((g.pnl > 0).sum()),
                  _signed(g.pips.sum(), f"{g.pips.sum():+,.1f}"), _signed(g.pnl.sum(), money(g.pnl.sum())),
                  money(month_end.get(month, float("nan"))))
    console.print(m)


# --- chart ------------------------------------------------------------------
def _break_gaps(t: pd.DatetimeIndex, *columns: pd.Series):
    """Put a NaN row in each weekend/holiday gap so lines and fills stop there instead of bridging it."""
    step = pd.Series(t).diff().median()
    gaps = pd.Series(t).diff().gt(step * 6).to_numpy()
    if not gaps.any():
        return (t, *columns)
    breaks = t[gaps.nonzero()[0] - 1] + step
    frame = pd.DataFrame({i: c.to_numpy() for i, c in enumerate(columns)}, index=t)
    frame = pd.concat([frame, pd.DataFrame(index=breaks, columns=frame.columns, dtype=float)]).sort_index()
    return (frame.index, *(frame[i] for i in range(len(columns))))


def plot_equity(result: "BacktestResult", path: str | Path | None = None, show: bool = False):
    """Three stacked panels on one time axis:

    1. equity (marked to market every bar) and balance (closed trades only), USD;
    2. drawdown from the running equity peak, %;
    3. position held: long above zero, short below.
    Saves a PNG when path is given; show=True opens an interactive window.
    Returns the matplotlib Figure (a notebook displays it inline).
    """
    import matplotlib

    style = {
        "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2,
        "ytick.color": INK_2, "axes.titlecolor": INK, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    }
    # Style only this chart, and leave the global matplotlib backend alone: without show,
    # the Figure isn't attached to pyplot, so it works headless and shows once in a notebook.
    with matplotlib.rc_context(style):
        if show:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(12, 7.5))
        else:
            from matplotlib.figure import Figure
            fig = Figure(figsize=(12, 7.5))
        _draw(result, fig)
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _draw(result: "BacktestResult", fig) -> None:
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    h = result.history
    t0 = h.index.tz_convert("UTC").tz_localize(None)  # matplotlib wants naive times; labelled UTC
    t, balance, equity, drawdown, pos = _break_gaps(t0, h.balance, h.equity, h.drawdown_pct, h.position)
    s = result.stats()

    ax_eq, ax_dd, ax_pos = fig.subplots(
        3, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1.2, 1], "hspace": 0.12},
    )

    # 1. Equity and balance.
    ax_eq.axhline(result.initial_cash, color=INK_3, lw=1, ls=(0, (2, 3)))
    ax_eq.annotate(f"start {money(result.initial_cash)}", (1, result.initial_cash),
                   xycoords=("axes fraction", "data"), xytext=(0, 4), textcoords="offset points",
                   ha="right", color=INK_3, fontsize=8)
    ax_eq.plot(t, balance, color=INK_2, lw=1.2, drawstyle="steps-post", label="Balance (closed trades)")
    ax_eq.plot(t, equity, color=EQUITY, lw=2, label="Equity (incl. open trade)")
    end = h.equity.iloc[-1]
    ax_eq.annotate(money(end), (t0[-1], end), xytext=(6, 0), textcoords="offset points",
                   va="center", color=INK, fontsize=9, fontweight="bold")
    ax_eq.set_ylabel("USD")
    ax_eq.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    # Legend in its own strip above the plot, so it never sits on the curve.
    ax_eq.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2, frameon=False, labelcolor=INK_2,
                 borderaxespad=0.2)
    ax_eq.set_title(
        f"{result.title}   |   return {s['return_pct']:+.2f}%   max drawdown {s['max_drawdown_pct']:.2f}%"
        f"   trades {s['trades']}   win rate {s['win_rate_pct']:.0f}%",
        loc="left", fontsize=10, fontweight="bold", pad=24,
    )

    # 2. Drawdown.
    ax_dd.fill_between(t, drawdown, 0, color=DRAWDOWN, alpha=0.25, lw=0)
    ax_dd.plot(t, drawdown, color=DRAWDOWN, lw=1.2)
    worst = h.drawdown_pct.idxmin()
    ax_dd.annotate(f"{h.drawdown_pct.min():.2f}%", (worst.tz_convert("UTC").tz_localize(None), h.drawdown_pct.min()),
                   xytext=(6, -2), textcoords="offset points", color=INK, fontsize=8)
    ax_dd.set_ylabel("Drawdown")
    ax_dd.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
    ax_dd.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))

    # 3. Position.
    ax_pos.fill_between(t, pos.clip(lower=0), 0, step="post", color=LONG, alpha=0.6, lw=0, label="Long")
    ax_pos.fill_between(t, pos.clip(upper=0), 0, step="post", color=SHORT, alpha=0.6, lw=0, label="Short")
    ax_pos.axhline(0, color=INK_3, lw=0.8)
    ax_pos.set_ylabel("Units")
    ax_pos.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:+.0f}k" if v else "0"))
    ax_pos.legend(loc="upper left", frameon=False, labelcolor=INK_2, ncols=2)

    for ax in (ax_eq, ax_dd, ax_pos):
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
    ax_pos.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_pos.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_pos.xaxis.get_major_locator()))
    ax_pos.set_xlabel("Time (UTC)")
    ax_pos.set_xlim(t0[0], t0[-1])
    _mark_periods(result, (ax_eq, ax_dd, ax_pos))


def _mark_periods(result: "BacktestResult", axes) -> None:
    """Walk-forward window boundaries, or the in-sample / out-of-sample split, as vertical lines."""
    naive = lambda ts: pd.Timestamp(ts).tz_convert("UTC").tz_localize(None)  # noqa: E731
    ax_top = axes[0]
    label_y = 0.99  # just inside the top of the equity panel, below the legend strip
    end = naive(result.history.index[-1])
    if "windows" in result.meta:
        for i, (start, end, label) in enumerate(result.meta["windows"]):
            for ax in axes:
                if i % 2:
                    ax.axvspan(naive(start), naive(end), color=INK_3, alpha=0.06, lw=0)
                ax.axvline(naive(start), color=INK_3, lw=0.8, ls=(0, (2, 2)))
            ax_top.text(naive(start), label_y, f" {label}", transform=ax_top.get_xaxis_transform(),
                        va="top", ha="left", fontsize=8, color=INK_2)
    if "split" in result.meta:
        cut = naive(result.meta["split"])
        for ax in axes:
            ax.axvspan(cut, end, color=EQUITY, alpha=0.05, lw=0)
            ax.axvline(cut, color=INK_2, lw=1, ls=(0, (4, 2)))
        ax_top.text(cut, label_y, "  out-of-sample →", transform=ax_top.get_xaxis_transform(),
                    va="top", ha="left", fontsize=8, color=INK_2)
        ax_top.text(cut, label_y, "← in-sample  ", transform=ax_top.get_xaxis_transform(),
                    va="top", ha="right", fontsize=8, color=INK_2)


# --- export -----------------------------------------------------------------
def save_csv(result: "BacktestResult", folder: str | Path, stem: str) -> list[Path]:
    """Write the metrics, full trade log, daily portfolio and per-bar history as CSV; returns the paths."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    names = ("metrics", "by_side", "trade_log", "daily", "history")
    paths = {name: folder / f"{stem}_{name}.csv" for name in names}
    metrics = result.metrics()
    metrics["value"] = [_csv_value(v) for v in metrics.value]
    metrics.to_csv(paths["metrics"], index=False)
    _round(result.by_side()).to_csv(paths["by_side"], index_label="side")
    log = result.trade_log()
    log.insert(log.columns.get_loc("duration"), "hold_minutes", log.duration.dt.total_seconds() / 60)
    _round(log.drop(columns="duration")).to_csv(paths["trade_log"])
    _round(result.daily()).to_csv(paths["daily"])
    _round(result.history).to_csv(paths["history"])
    return list(paths.values())


def _round(df: pd.DataFrame) -> pd.DataFrame:
    """Round the number columns only (timestamps can't be rounded this way)."""
    return df.round({c: 6 for c in df.select_dtypes("number").columns})


def _csv_value(v):
    """Durations as minutes and floats without binary noise, so spreadsheets read them as numbers."""
    if isinstance(v, pd.Timedelta):
        return v.total_seconds() / 60
    return round(v, 6) if isinstance(v, float) else v


# --- optimisation, out-of-sample, walk-forward -----------------------------
def print_param_table(table: pd.DataFrame, console=None, top: int = 10, title: str | None = None) -> None:
    """The best `top` parameter combinations from optimize()."""
    from rich.console import Console
    from rich.table import Table

    console = console or Console()
    params = list(table.attrs.get("params", []))
    objective = table.attrs.get("objective", "score")
    t = Table(title=title or f"Parameter search: top {min(top, len(table))} of {len(table)} by {objective}",
              title_style="bold cyan", header_style="bold")
    for col in ["#", *params, objective, "Trades", "Return", "Net P&L", "Sharpe", "Profit factor", "Max DD",
                "Win rate"]:
        t.add_column(col, justify="right")
    for i, r in table.head(top).iterrows():
        score = "-" if r.score == float("-inf") else f"{r.score:.2f}"
        t.add_row(str(i + 1), *[f"{r[p]:g}" for p in params], f"[bold]{score}[/]", f"{int(r.trades):,}",
                  _signed(r.return_pct, f"{r.return_pct:+.2f}%"), _signed(r.net_pnl, money(r.net_pnl)),
                  f"{r.sharpe:.2f}", f"{r.profit_factor:.2f}", f"{r.max_drawdown_pct:.2f}%",
                  f"{r.win_rate_pct:.1f}%")
    console.print(t)


_COMPARE = [("return_pct", "Return", "change"), ("cagr_pct", "Annualised return", "change"),
            ("net_pnl", "Net P&L", "money"), ("max_drawdown_pct", "Max drawdown", "pct"),
            ("sharpe", "Sharpe ratio", "ratio"), ("profit_factor", "Profit factor", "ratio"),
            ("win_rate_pct", "Win rate", "pct"), ("trades", "Trades", "int"),
            ("trades_per_day", "Trades per day", "ratio"), ("expectancy", "Expectancy per trade", "money")]


def print_out_of_sample(oos, console=None, top: int = 10) -> None:
    """In-sample parameter search, then in-sample vs out-of-sample for the chosen parameters."""
    from rich.console import Console
    from rich.table import Table

    from .metrics import format_value

    console = console or Console()
    print_param_table(oos.table, console, top,
                      title=f"In-sample parameter search (before {oos.split:%Y-%m-%d %H:%M} UTC), by {oos.objective}")
    a, b = oos.in_sample.stats(), oos.out_of_sample.stats()
    chosen = ", ".join(f"{k}={v}" for k, v in oos.params.items())
    t = Table(title=f"Chosen on in-sample: {chosen}. Same settings, untouched data:", title_style="bold cyan",
              header_style="bold")
    t.add_column("Metric")
    t.add_column(f"In-sample\n{oos.in_sample.history.index[0]:%m-%d} to {oos.split:%m-%d}", justify="right")
    t.add_column(f"Out-of-sample\n{oos.split:%m-%d} to {oos.out_of_sample.history.index[-1]:%m-%d}",
                 justify="right")
    for key, name, kind in _COMPARE:
        t.add_row(name, format_value(a[key], kind), _signed(b[key], format_value(b[key], kind))
                  if kind in ("change", "money") else format_value(b[key], kind))
    console.print(t)


def print_walk_forward(wf, console=None) -> None:
    """One row per walk-forward window, then the stitched out-of-sample result."""
    from rich.console import Console
    from rich.table import Table

    from .metrics import format_value

    console = console or Console()
    w = wf.windows
    params = [c for c in w.columns if c.startswith("param_")]
    t = Table(title=f"Walk-forward: {len(w)} windows, {'anchored' if wf.anchored else 'rolling'} training, "
                    f"re-optimised on {wf.objective}", title_style="bold cyan", header_style="bold")
    for col in ["Window", "Train (UTC)", "Test (UTC)", *[p.removeprefix("param_") for p in params],
                f"In-sample {wf.objective}", "IS return", "OOS return", "OOS P&L", "OOS trades", "OOS max DD",
                "Equity after"]:
        t.add_column(col, justify="left" if col in ("Train (UTC)", "Test (UTC)") else "right")
    for i, r in w.iterrows():
        t.add_row(f"W{i}", f"{r.train_start:%m-%d} to {r.train_end:%m-%d}",
                  f"{r.test_start:%m-%d} to {r.test_end:%m-%d}", *[f"{r[p]:g}" for p in params],
                  f"{r.is_score:.2f}", f"{r.is_return_pct:+.2f}%",
                  _signed(r.oos_return_pct, f"{r.oos_return_pct:+.2f}%"), _signed(r.oos_net_pnl, money(r.oos_net_pnl)),
                  f"{int(r.oos_trades):,}", f"{r.oos_max_drawdown_pct:.2f}%", money(r.oos_end_equity))
    console.print(t)
    s = wf.combined.stats()
    eff = wf.efficiency
    summary = Table(title="Walk-forward out-of-sample, all windows chained", title_style="bold cyan",
                    show_header=False)
    summary.add_column("Metric", style="dim")
    summary.add_column("Value", justify="right")
    for key, name, kind in _COMPARE:
        text = format_value(s[key], kind)
        summary.add_row(name, _signed(s[key], text) if kind in ("change", "money") else text)
    summary.add_row("Walk-forward efficiency", "-" if eff != eff else _signed(eff, f"{eff:.2f}"))
    console.print(summary)
    console.print("[dim]Efficiency = out-of-sample return per day / in-sample return per day; "
                  "around 0.5 or more suggests the in-sample edge carried over.[/]")
