"""Backtest Studio: a browser front end for backtest_engine.

    streamlit run app.py

Upload one or more OHLCV CSV files (each file is an asset), pick or write a strategy,
set orders, sizing and costs, and run. Results: QuantConnect-style statistics, equity vs
buy-and-hold, drawdowns, monthly returns, rolling Sharpe, the trade log, per-asset
breakdown, out-of-sample / walk-forward testing and CSV / HTML exports.
"""

from __future__ import annotations

import inspect
import io
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from backtest_engine import point_size
from backtest_engine.analytics import (daily_returns, drawdown_periods, monthly_returns, qc_statistics,
                                       rolling_sharpe, statistics_frame)
from backtest_engine.broker import FixedSlippage, RandomSlippage
from backtest_engine.data_loader import DEFAULT_FILE, MT5_SERVER_TZ, DataValidationError, load_ohlcv, resample_ohlcv
from backtest_engine.metrics import format_duration, format_value
from backtest_engine.multi import Asset, MultiAssetResult, run_multi
from backtest_engine.optimize import OBJECTIVES, RunConfig, out_of_sample, walk_forward
from backtest_engine.orders import OrderPolicy
from backtest_engine.portfolio import Portfolio, is_fx_pair
from backtest_engine.quantconnect import QCResult, parse_qc
from backtest_engine.qcalgo import QC_TEMPLATE, AlgorithmError, run_algorithm

import workspace
from auth import PERMISSIONS, AuthError, User, UserStore
from backtest_engine.signals import (BUILTINS, CUSTOM_TEMPLATE, SignalStrategy, StrategySpec, check_lookahead,
                                     load_custom)
from backtest_engine.sizing import PercentEquity, RiskPercent

st.set_page_config(page_title="Backtest Studio", page_icon=":material/monitoring:", layout="wide")

# Chart colours: categorical slots in fixed order, plus status-free neutrals.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
GREY = "#8a8985"
TIMEZONES = {
    "UTC": "UTC",
    "MT5 server time (UTC+2, fixed)": MT5_SERVER_TZ,
    "MT5 server time (UTC+2/+3, European DST)": "Europe/Athens",
    "New York": "America/New_York",
    "London": "Europe/London",
}
TIMEFRAMES = ["As uploaded", "1min", "5min", "15min", "30min", "1h", "4h", "1D"]


# --- small helpers -----------------------------------------------------------------
def show_chart(fig: go.Figure) -> None:
    try:
        st.plotly_chart(fig, width="stretch", theme="streamlit")
    except TypeError:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def show_table(df, **kw) -> None:
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def money(x: float) -> str:
    return format_value(x, "money")


def cal() -> bool:
    """Whether ratios use QuantConnect's calendar-day returns (sidebar switch)."""
    return bool(st.session_state.get("qc_calendar", True))


STAT_CSS = """
<style>
.stat-grid {display:grid; grid-template-columns:repeat(var(--cols), minmax(0,1fr)); column-gap:32px;}
.stat-grid .row {display:flex; justify-content:space-between; gap:12px; padding:7px 2px;
                 border-bottom:1px solid rgba(128,128,128,0.18); font-size:0.92rem;}
.stat-grid .row span:first-child {opacity:0.75;}
.stat-grid .row span:last-child {font-variant-numeric:tabular-nums; font-weight:600; white-space:nowrap;}
.stat-title {font-weight:600; margin:4px 0 6px 0;}
</style>
"""


def stat_grid(rows: list[tuple[str, str]], cols: int = 2, title: str | None = None) -> None:
    """Label / value pairs in `cols` columns, filled top to bottom like QuantConnect's statistics."""
    import html
    per = -(-len(rows) // cols)
    ordered = [rows[c * per + r] for r in range(per) for c in range(cols) if c * per + r < len(rows)]
    cells = "".join(f'<div class="row"><span>{html.escape(k)}</span><span>{html.escape(v)}</span></div>'
                    for k, v in ordered)
    head = f'<div class="stat-title">{html.escape(title)}</div>' if title else ""
    st.markdown(f'{STAT_CSS}{head}<div class="stat-grid" style="--cols:{cols}">{cells}</div>', unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 380, title: str | None = None) -> go.Figure:
    # Title in the top strip, legend in its own row just above the plot, so they never overlap.
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=78 if title else 34, b=10),
                      title=dict(text=title, x=0, xanchor="left", y=0.99, yanchor="top") if title else None,
                      hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)")
    return fig


def gapped(series: pd.Series) -> pd.Series:
    """Insert a NaN in each weekend/holiday gap so lines stop there instead of bridging it."""
    if len(series) < 3:
        return series
    step = series.index.to_series().diff().median()
    gaps = series.index.to_series().diff() > step * 6
    if not gaps.any():
        return series
    breaks = pd.Series(np.nan, index=series.index[gaps.to_numpy().nonzero()[0] - 1] + step)
    return pd.concat([series, breaks]).sort_index()


def guess_symbol(name: str) -> str:
    stem = Path(name).stem.upper()
    lead = re.match(r"[A-Z]{6}(?![A-Z])", stem)  # EURUSD.csv, XAUUSD_5min.csv
    if lead:
        return lead.group(0)
    m = re.search(r"[A-Z]{6}", re.sub(r"[^A-Z]", "", stem))  # MT5_USDJPY_export.csv
    return m.group(0) if m and is_fx_pair(m.group(0)) else re.sub(r"[^A-Z0-9]+", "", stem)[:12] or "ASSET"


def looks_like_mt5(head: bytes) -> bool:
    return b"<DATE>" in head[:200].upper()


@st.cache_data(show_spinner=False, max_entries=20)
def parse_csv(data: bytes, tz: str) -> pd.DataFrame:
    return load_ohlcv(io.BytesIO(data), tz=tz)


@st.cache_data(show_spinner=False, max_entries=20)
def read_path(path: str) -> bytes:
    return Path(path).read_bytes()


# --- sidebar: data -------------------------------------------------------------------
@dataclass
class AssetInput:
    key: str
    name: str
    data: bytes


def sidebar_data() -> tuple[list[Asset], str, list[str]]:
    st.sidebar.header("1 · Data")
    st.sidebar.caption("One CSV per asset: a date/time column plus open, high, low, close "
                       "(volume and spread optional). MetaTrader, TradingView and Yahoo exports work.")
    uploads = st.sidebar.file_uploader("Upload CSV files", type=["csv", "txt"], accept_multiple_files=True)
    inputs = [AssetInput(f"up:{u.name}", u.name, u.getvalue()) for u in uploads or []]
    allowed = current_user().can("local_files")
    paths = "" if not allowed else st.sidebar.text_input("…or file paths on this computer (comma-separated)",
                                  placeholder=r"C:\data\EURUSD.csv, C:\data\XAUUSD.csv")
    if paths.strip():
        for p in [p.strip().strip('"') for p in paths.split(",") if p.strip()]:
            if Path(p).exists():
                inputs.append(AssetInput(f"path:{p}", Path(p).name, read_path(p)))
            else:
                st.sidebar.error(f"Not found: {p}")
    if not inputs and allowed and DEFAULT_FILE.exists():
        if st.sidebar.toggle("Use the MT5 USDJPY sample", value=True):
            inputs.append(AssetInput(f"path:{DEFAULT_FILE}", DEFAULT_FILE.name, read_path(str(DEFAULT_FILE))))

    timeframe = st.sidebar.selectbox("Timeframe", TIMEFRAMES, index=TIMEFRAMES.index("5min") if inputs else 0,
                                     help="Bars are resampled on the 17:00 New York forex day. "
                                          "'As uploaded' keeps the file's own bars.")
    assets, notes = [], []
    for i, inp in enumerate(inputs):
        with st.sidebar.expander(f"{inp.name}", expanded=len(inputs) <= 2):
            symbol = st.text_input("Symbol", guess_symbol(inp.name), key=f"sym:{inp.key}",
                                   help="FX pairs like USDJPY / EURUSD convert P&L to USD. Anything else "
                                        "(stocks, indices, crypto) is treated as priced in USD.").upper().strip()
            tz_default = list(TIMEZONES).index("MT5 server time (UTC+2, fixed)") if looks_like_mt5(inp.data) else 0
            tz_label = st.selectbox("Time zone of the file", list(TIMEZONES), index=tz_default, key=f"tz:{inp.key}")
            c1, c2 = st.columns(2)
            weight = c1.number_input("Weight", 0.0, 100.0, 1.0, 0.5, key=f"w:{inp.key}",
                                     help="Share of the capital, relative to the other assets")
            point_txt = c2.text_input("Point size", "auto", key=f"pt:{inp.key}",
                                      help="Price of one spread point (0.001 for JPY pairs, 0.00001 for "
                                           "EURUSD). 'auto' guesses from the price. 1 pip = 10 points.")
            c1, c2 = st.columns(2)
            lot_txt = c1.text_input("Lot step", "auto", key=f"lot:{inp.key}",
                                    help="Sizes round down to this many units. auto: 1,000 for FX pairs "
                                         "(micro lot), 0.01 for anything else.")
            units_txt = c2.text_input("Fixed units", "auto", key=f"u:{inp.key}",
                                      help="Fixed sizing only. auto: the global units for FX pairs; about 1x "
                                           "this asset's share of the capital for anything else.")
            try:
                df = parse_csv(inp.data, TIMEZONES[tz_label])
            except (DataValidationError, ValueError) as exc:
                st.error(f"Couldn't read this file: {exc}")
                continue
            if timeframe != "As uploaded":
                df = resample_ohlcv(df, timeframe)
            try:
                point = None if point_txt.strip().lower() in ("", "auto") else float(point_txt)
                lot = (1_000 if is_fx_pair(symbol) else 0.01) if lot_txt.strip().lower() in ("", "auto") \
                    else float(lot_txt)
                units = None if units_txt.strip().lower() in ("", "auto") else float(units_txt)
            except ValueError:
                st.error("Point size, lot step and fixed units must be numbers or 'auto'")
                continue
            st.caption(f"{len(df):,} bars · {df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d} UTC")
            if (df.open == df.close).all() and (df.high == df.low).all():
                notes.append(f"{symbol}: close prices only, so stops and targets inside a bar are approximate.")
            assets.append(Asset(symbol or f"ASSET{i + 1}", df, weight, point, lot_step=lot, units=units))

    if assets:
        lo = min(a.df.index[0] for a in assets).date()
        hi = max(a.df.index[-1] for a in assets).date()
        rng = st.sidebar.date_input("Date range", (lo, hi), min_value=lo, max_value=hi)
        if isinstance(rng, (tuple, list)) and len(rng) == 2:
            start = pd.Timestamp(rng[0], tz="UTC")
            end = pd.Timestamp(rng[1], tz="UTC") + pd.Timedelta(days=1)
            assets = [replace(a, df=a.df[(a.df.index >= start) & (a.df.index < end)]) for a in assets]
            assets = [a for a in assets if len(a.df) > 1]
    return assets, timeframe, notes


# --- sidebar: strategy ---------------------------------------------------------------
@dataclass
class StrategyChoice:
    name: str
    factory: callable  # **params -> Strategy
    params: dict
    spec: StrategySpec | None  # built-ins; None for custom code
    custom_fn: callable = None
    long_only: bool = False
    kind: str = "builtin"  # builtin, custom (a signals function) or qc (a QuantConnect algorithm)
    code: str = ""


def sidebar_strategy() -> StrategyChoice | None:
    st.sidebar.header("2 · Strategy")
    code_ok = current_user().can("run_code")
    options = [s.name for s in BUILTINS.values()] + (["Custom (Python)", QC_CHOICE] if code_ok else [])
    choice = st.sidebar.selectbox("Strategy", options)
    if not code_ok:
        st.sidebar.caption("Custom code is turned off for your account (an admin can allow it).")
    if choice == QC_CHOICE:
        return sidebar_qc_algorithm()
    long_only = st.sidebar.toggle("Long only", value=False, help="Go flat instead of short")
    if choice != "Custom (Python)":
        spec = next(s for s in BUILTINS.values() if s.name == choice)
        st.sidebar.caption(spec.description)
        params = {}
        cols = st.sidebar.columns(min(3, len(spec.params)) or 1)
        for i, p in enumerate(spec.params):
            kw = dict(min_value=p.low, max_value=p.high, value=p.default, step=p.step)
            if p.is_int:
                kw = {k: int(v) for k, v in kw.items()}
            params[p.name] = cols[i % len(cols)].number_input(p.name.replace("_", " "), key=f"p:{spec.key}:{p.name}", **kw)
        return StrategyChoice(spec.name, spec.factory(long_only), params, spec, long_only=long_only)

    saved_strategies_ui("custom_code", "custom")
    up = st.sidebar.file_uploader("Load a strategy .py file", type=["py"])
    if up is not None:
        st.session_state["custom_code"] = up.getvalue().decode("utf-8")
    code = st.sidebar.text_area("Strategy code", st.session_state.get("custom_code", CUSTOM_TEMPLATE), height=320,
                                help="Define signals(df, ...) returning +1 / -1 / 0 per bar. "
                                     "Runs on this computer like any Python script.")
    st.session_state["custom_code"] = code
    try:
        fn, defaults = load_custom(code)
    except Exception as exc:  # syntax errors etc. in the user's code
        st.sidebar.error(f"Strategy code error: {exc}")
        return None
    params = {}
    for name, default in defaults.items():
        if isinstance(default, bool) or not isinstance(default, (int, float)):
            params[name] = default
            continue
        params[name] = st.sidebar.number_input(name.replace("_", " "), value=default,
                                               step=1 if isinstance(default, int) else 0.1, key=f"cp:{name}")
    factory = lambda **kw: SignalStrategy(fn, long_only=long_only, **kw)  # noqa: E731
    return StrategyChoice("Custom strategy", factory, params, None, custom_fn=fn, long_only=long_only, kind="custom",
                          code=code)


# --- sidebar: orders, sizing, costs -------------------------------------------------------
def sidebar_execution(algo_mode: bool = False) -> dict:
    if algo_mode:
        return sidebar_execution_algo()
    st.sidebar.header("3 · Orders & exits")
    entry = st.sidebar.selectbox("Entry order", ["market", "limit", "stop"],
                                 help="market: next bar's open · limit: wait for a better price · "
                                      "stop: enter on a breakout")
    c1, c2 = st.sidebar.columns(2)
    offset = c1.number_input("Entry offset (pips)", 0.0, 1000.0, 0.0, 0.5, disabled=entry == "market")
    expiry = c2.number_input("Expiry (bars, 0 = none)", 0, 10_000, 0, disabled=entry == "market")
    c1, c2 = st.sidebar.columns(2)
    sl = c1.number_input("Stop-loss (pips)", 0.0, 10_000.0, 0.0, 1.0, help="Distance from the fill; 0 = off")
    tp = c2.number_input("Take-profit (pips)", 0.0, 10_000.0, 0.0, 1.0, help="0 = off")
    trail = st.sidebar.number_input("Trailing stop (pips)", 0.0, 10_000.0, 0.0, 1.0, help="0 = off")

    st.sidebar.header("4 · Capital, sizing & costs")
    cash = st.sidebar.number_input("Starting capital (USD)", 100.0, 1e9, 10_000.0, 1_000.0)
    sizing = st.sidebar.radio("Position sizing", ["Fixed units", "% risk per trade", "% of equity"], index=2,
                              horizontal=True, help="% of equity works for any asset. Fixed units means the "
                                                    "same number of units on every asset (10,000 = 0.1 FX lot, "
                                                    "but 10,000 shares or ounces elsewhere).")
    c1, c2 = st.sidebar.columns(2)
    units = risk = eq_pct = None
    if sizing == "Fixed units":
        units = c1.number_input("Units per trade", 1.0, 1e9, 10_000.0, 1_000.0, help="10,000 = 0.1 lot; shares for stocks")
    elif sizing == "% risk per trade":
        risk = c1.number_input("Risk per trade (%)", 0.01, 100.0, 1.0, 0.25)
        if not (sl or trail):
            st.sidebar.warning("Risk sizing needs a stop-loss or trailing stop.")
    else:
        eq_pct = c1.number_input("Position (% of equity)", 1.0, 10_000.0, 100.0, 10.0)
    leverage = c2.number_input("Max leverage", 0.1, 1000.0, 30.0, 1.0, disabled=sizing == "Fixed units")
    lot = 1_000.0  # default rounding; each asset sets its own lot step (Data section)
    slip_kind = st.sidebar.radio("Slippage", ["None", "Fixed", "Random"], horizontal=True)
    slip = 0.0
    if slip_kind != "None":
        slip = st.sidebar.number_input("Slippage (points)" if slip_kind == "Fixed" else "Max slippage (points)",
                                       0.0, 10_000.0, 2.0, 0.5)
    rf = st.sidebar.number_input("Risk-free rate (% a year)", 0.0, 20.0, 0.0, 0.25,
                                 help="Used for Sharpe, Sortino, alpha and Treynor") / 100
    st.sidebar.toggle("QuantConnect-style daily returns", value=True, key="qc_calendar",
                      help="On: weekends count as zero-return days, as QuantConnect samples them, so ratios "
                           "are comparable with its reports (for forex this lowers volatility by about a "
                           "fifth). Off: only trading days.")
    return dict(entry=entry, offset=offset, expiry=int(expiry) or None, sl=sl or None, tp=tp or None,
                trail=trail or None, cash=cash, sizing=sizing, units=units, risk=risk, eq_pct=eq_pct,
                leverage=leverage, lot=lot, slip_kind=slip_kind, slip=slip, rf=rf)


def resolve_fixed_units(assets: list[Asset], cash: float, global_units: float) -> list[Asset]:
    """Fill in 'auto' fixed sizes: the global units for FX pairs, ~1x the capital share otherwise."""
    total_w = sum(a.weight for a in assets) or 1
    out = []
    for a in assets:
        units = a.units
        if units is None:
            if is_fx_pair(a.symbol):
                units = global_units
            else:
                capital = cash * a.weight / total_w
                step = a.lot_step or 0.01
                units = max(step, np.floor(capital / float(a.df.close.iloc[0]) / step) * step)
        out.append(replace(a, units=units))
    return out


def build_config(x: dict, strategy: StrategyChoice) -> RunConfig:
    common = dict(lot_step=x["lot"], min_units=x["lot"], max_leverage=x["leverage"])
    sizer = None
    if x["sizing"] == "% risk per trade":
        sizer = RiskPercent(pct=x["risk"], **common)
    elif x["sizing"] == "% of equity":
        sizer = PercentEquity(pct=x["eq_pct"], **common)
    slippage = (FixedSlippage(x["slip"]) if x["slip_kind"] == "Fixed" and x["slip"]
                else RandomSlippage(x["slip"], seed=0) if x["slip_kind"] == "Random" and x["slip"] else None)
    orders = OrderPolicy(entry=x["entry"], entry_offset_pips=x["offset"], expiry_bars=x["expiry"],
                         stop_loss_pips=x["sl"], take_profit_pips=x["tp"], trailing_stop_pips=x["trail"])
    return RunConfig(cash=x["cash"], units=x["units"] or 10_000, sizer=sizer, orders=orders, slippage=slippage,
                     make_strategy=strategy.factory)


# --- charts --------------------------------------------------------------------------
def equity_figure(multi: MultiAssetResult, show_assets: bool) -> go.Figure:
    r = multi.combined
    h = r.history
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)
    scatter = go.Scattergl if len(h) > 5_000 else go.Scatter
    line = lambda s, **kw: scatter(x=(g := gapped(s)).index, y=g.to_numpy(), **kw)  # noqa: E731
    fig.add_trace(line(multi.benchmark.reindex(h.index), name="Buy & hold (same assets)",
                       line=dict(color=GREY, width=1.2, dash="dot")), 1, 1)
    if show_assets and len(multi.per_asset) > 1:
        for i, (sym, a) in enumerate(multi.per_asset.items()):
            fig.add_trace(line(a.equity, name=f"{sym} sub-account",
                               line=dict(color=SERIES[(i + 1) % len(SERIES)], width=1)), 1, 1)
    fig.add_trace(line(h.equity, name="Strategy equity", line=dict(color=BLUE, width=2)), 1, 1)
    fig.add_hline(y=r.initial_cash, line=dict(color=GREY, width=1, dash="dash"), row=1, col=1)
    fig.add_trace(line(h.drawdown_pct, name="Drawdown %", fill="tozeroy",
                       line=dict(color=RED, width=1), fillcolor="rgba(227,73,72,0.18)"), 2, 1)
    fig.update_yaxes(title_text="USD", tickprefix="$", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", ticksuffix="%", row=2, col=1)
    return style_fig(fig, 560)


def price_trades_figure(asset: Asset, result) -> go.Figure:
    df, t = asset.df, result.trades
    fig = go.Figure()
    many = len(df) > 3_000
    if many:
        close = gapped(df.close)
        fig.add_trace(go.Scattergl(x=close.index, y=close.to_numpy(), name="Close", line=dict(color=GREY, width=1)))
    else:
        fig.add_trace(go.Candlestick(x=df.index, open=df.open, high=df.high, low=df.low, close=df.close, name="Price",
                                     increasing_line_color=AQUA, decreasing_line_color=RED, showlegend=False))
    crowded = len(t) > 150
    for side, symbol, color in [("long", "triangle-up", AQUA), ("short", "triangle-down", ORANGE)]:
        e = t[t.side == side]
        if len(e):
            fig.add_trace(go.Scattergl(x=e.entry_time, y=e.entry_price, mode="markers", name=f"{side.capitalize()} entry",
                                       marker=dict(symbol=symbol, size=8 if crowded else 10, color=color,
                                                   line=dict(width=1, color="white")),
                                       customdata=np.c_[e.pnl, e.pips, e.exit_reason],
                                       hovertemplate="%{x}<br>entry %{y}<br>P&L $%{customdata[0]:.2f} "
                                                     "(%{customdata[1]:+.1f} pips)<br>exit: %{customdata[2]}<extra></extra>"))
    if len(t):
        # Many trades: exits start hidden (click "Exit" in the legend) so they don't bury the entries.
        fig.add_trace(go.Scattergl(x=t.exit_time, y=t.exit_price, mode="markers", name="Exit",
                                   marker=dict(symbol="x-thin", size=7, color=VIOLET, opacity=0.6,
                                               line=dict(width=1.5, color=VIOLET)),
                                   visible="legendonly" if crowded else True))
    fig.update_layout(xaxis_rangeslider_visible=False)
    return style_fig(fig, 460)


def monthly_heatmap(table: pd.DataFrame) -> go.Figure:
    z = table.to_numpy(dtype=float)
    lim = np.nanmax(np.abs(z)) if np.isfinite(z).any() else 1
    text = [[("" if np.isnan(v) else f"{v:+.2f}%") for v in row] for row in z]
    fig = go.Figure(go.Heatmap(z=z, x=list(table.columns), y=[str(y) for y in table.index], zmin=-lim, zmax=lim,
                               colorscale=[[0, RED], [0.5, "#e6e5e1"], [1, AQUA]], text=text,
                               texttemplate="%{text}", hovertemplate="%{y} %{x}: %{text}<extra></extra>",
                               colorbar=dict(ticksuffix="%")))
    fig.update_yaxes(autorange="reversed", type="category")
    return style_fig(fig, 140 + 48 * len(table))


# --- result views -----------------------------------------------------------------------
def kpis(multi: MultiAssetResult, rf: float) -> None:
    qc = {m.key: m for m in qc_statistics(multi.combined, multi.benchmark, rf, cal())}
    s = multi.combined.stats()
    tiles = [
        ("Net profit", format_value(qc["net_profit_pct"].value, "change"), money(s["net_pnl"])),
        ("CAGR", format_value(qc["cagr_pct"].value, "change"), None),
        ("Sharpe", format_value(qc["sharpe"].value, "ratio"), None),
        ("Sortino", format_value(qc["sortino"].value, "ratio"), None),
        ("Prob. Sharpe", format_value(qc["psr_pct"].value, "pct"), None),
        ("Max drawdown", f"-{qc['drawdown_pct'].value:.2f}%", money(s["max_drawdown_usd"])),
        ("Win rate", format_value(qc["win_rate_pct"].value, "pct"), f"{s['trades']:,} trades"),
        ("Profit factor", format_value(s["profit_factor"], "ratio"), None),
    ]
    arrow = {"delta_arrow": "off"} if "delta_arrow" in inspect.signature(st.metric).parameters else {}
    for row in (tiles[:4], tiles[4:]):
        for col, (label, value, sub) in zip(st.columns(4), row):
            col.metric(label, value, sub, delta_color="off", border=True, **arrow)


def tab_overview(multi: MultiAssetResult, assets: list[Asset], rf: float) -> None:
    kpis(multi, rf)
    show_assets = len(multi.per_asset) > 1 and st.toggle("Show each asset's sub-account", value=False)
    show_chart(equity_figure(multi, show_assets))
    names = list(multi.per_asset)
    sym = st.selectbox("Price and trades for", names) if len(names) > 1 else names[0]
    asset = next(a for a in assets if a.symbol == sym)
    many = len(multi.per_asset[sym].trades) > 150
    st.caption(f"{sym}: entries (▲ long, ▼ short) and exits (×{', hidden: click Exit in the legend' if many else ''}). "
               "Hover an entry for its P&L; drag to zoom, double-click to reset.")
    show_chart(price_trades_figure(asset, multi.per_asset[sym]))


def tab_statistics(multi: MultiAssetResult, rf: float) -> None:
    stats = statistics_frame(multi.combined, multi.benchmark, rf, cal())
    qc = stats[stats.section == "Overview (QuantConnect)"]
    st.subheader("Strategy statistics")
    st.caption("QuantConnect's backtest statistics. Alpha, beta, information ratio, tracking error and "
               "Treynor are measured against buy-and-hold of the same assets and weights. Probabilistic Sharpe "
               "is the chance the true Sharpe ratio is above 1 (QuantConnect's reference).")
    stat_grid(list(zip(qc.metric, qc.display)), cols=2)
    st.subheader("Detailed metrics")
    sections = [s for s in stats.section.unique() if s != "Overview (QuantConnect)"]
    for i in range(0, len(sections), 2):
        for col, section in zip(st.columns(2), sections[i:i + 2]):
            part = stats[stats.section == section]
            with col:
                stat_grid(list(zip(part.metric, part.display)), cols=1, title=section)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Long vs short")
        show_table(multi.combined.by_side().style.format({
            "trades": "{:,.0f}", "win_rate_pct": "{:.1f}%", "net_pnl": "${:,.2f}", "pips": "{:+,.1f}",
            "profit_factor": "{:.2f}", "avg_trade": "${:,.2f}", "spread_paid": "${:,.2f}", "slippage_paid": "${:,.2f}"}))
    with c2:
        st.subheader("Deepest drawdowns")
        dd = drawdown_periods(multi.combined.equity)
        dd["length"] = dd.length.map(format_duration)
        show_table(dd.style.format({"depth_pct": "{:.2f}%", "peak": "{:%Y-%m-%d %H:%M}", "trough": "{:%Y-%m-%d %H:%M}",
                                    "recovery": lambda v: "not yet" if pd.isna(v) else f"{v:%Y-%m-%d %H:%M}"}))


def tab_returns(multi: MultiAssetResult, rf: float) -> None:
    r = multi.combined
    st.subheader("Monthly returns")
    show_chart(monthly_heatmap(monthly_returns(r.equity, r.initial_cash)))
    c1, c2 = st.columns(2)
    rets = daily_returns(r.equity, r.initial_cash, cal()) * 100
    bench = daily_returns(multi.benchmark, multi.benchmark.iloc[0], cal()) * 100
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=rets, name="Strategy", marker_color=BLUE, opacity=0.8, nbinsx=40))
        fig.add_trace(go.Histogram(x=bench, name="Buy & hold", marker_color=GREY, opacity=0.5, nbinsx=40))
        fig.update_layout(barmode="overlay", hovermode="closest")
        fig.update_xaxes(ticksuffix="%", title_text="Daily return")
        show_chart(style_fig(fig, 340, "Daily returns"))
    with c2:
        window = st.slider("Rolling window (trading days)", 5, 120, 20)
        rs = rolling_sharpe(r.equity, window, rf, cal())
        fig = go.Figure(go.Scatter(x=rs.index, y=rs, name="Rolling Sharpe", line=dict(color=BLUE, width=2)))
        fig.add_hline(y=0, line=dict(color=GREY, width=1))
        show_chart(style_fig(fig, 340, f"Rolling Sharpe ({window} days)"))
    cum = (1 + rets / 100).cumprod() - 1
    cum_b = (1 + bench / 100).cumprod() - 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum.index, y=cum * 100, name="Strategy", line=dict(color=BLUE, width=2)))
    fig.add_trace(go.Scatter(x=cum_b.index, y=cum_b * 100, name="Buy & hold", line=dict(color=GREY, width=1.5, dash="dot")))
    fig.update_yaxes(ticksuffix="%")
    show_chart(style_fig(fig, 320, "Cumulative return by trading day"))


def tab_trades(multi: MultiAssetResult) -> None:
    log = multi.combined.trade_log()
    if log.empty:
        st.info("No trades. Try different parameters, a longer date range, or check the sizing settings.")
        return
    c1, c2, c3 = st.columns(3)
    syms = sorted(log.symbol.unique()) if "symbol" in log else []
    pick_sym = c1.multiselect("Asset", syms, syms) if syms else []
    pick_side = c2.multiselect("Side", ["long", "short"], ["long", "short"])
    reasons = sorted(log.exit_reason.unique())
    pick_reason = c3.multiselect("Exit reason", reasons, reasons)
    view = log[log.side.isin(pick_side) & log.exit_reason.isin(pick_reason)]
    if syms:
        view = view[view.symbol.isin(pick_sym)]
    st.caption(f"{len(view):,} of {len(log):,} trades · net {money(view.pnl.sum())} · "
               f"win rate {(view.pnl > 0).mean() * 100:.1f}%")
    shown = view.assign(duration=view.duration.map(format_duration))
    show_table(shown, height=420, column_config={
        "entry_time": st.column_config.DatetimeColumn("Entry (UTC)", format="YYYY-MM-DD HH:mm"),
        "exit_time": st.column_config.DatetimeColumn("Exit (UTC)", format="YYYY-MM-DD HH:mm"),
        "units": st.column_config.NumberColumn("Units", format="%,.0f"),
        "entry_price": st.column_config.NumberColumn("Entry", format="%.5f"),
        "exit_price": st.column_config.NumberColumn("Exit", format="%.5f"),
        "pips": st.column_config.NumberColumn("Pips", format="%+.1f"),
        "pnl": st.column_config.NumberColumn("P&L", format="$%.2f"),
        "return_pct": st.column_config.NumberColumn("Return", format="%+.3f%%"),
        "spread_cost": st.column_config.NumberColumn("Spread", format="$%.2f"),
        "slippage_cost": st.column_config.NumberColumn("Slippage", format="$%.2f"),
        "mae_pips": st.column_config.NumberColumn("MAE", format="%.1f"),
        "mfe_pips": st.column_config.NumberColumn("MFE", format="%+.1f"),
        "balance": st.column_config.NumberColumn("Balance", format="$%,.2f"),
        "cum_pnl": st.column_config.NumberColumn("Cumulative P&L", format="$%,.2f"),
    })
    c1, c2 = st.columns(2)
    with c1:
        colors = np.where(view.pnl > 0, AQUA, RED)
        fig = go.Figure(go.Bar(x=view.index, y=view.pnl, marker_color=colors, name="P&L",
                               hovertemplate="trade %{x}: $%{y:.2f}<extra></extra>"))
        fig.update_xaxes(title_text="Trade #")
        fig.update_yaxes(tickprefix="$")
        show_chart(style_fig(fig, 340, "P&L per trade"))
    with c2:
        fig = go.Figure(go.Scatter(x=view.mae_pips, y=view.pips, mode="markers", name="Trades",
                                   marker=dict(color=np.where(view.pnl > 0, AQUA, RED), size=8, opacity=0.75,
                                               line=dict(width=1, color="white")),
                                   hovertemplate="MAE %{x:.1f} pips<br>result %{y:+.1f} pips<extra></extra>"))
        fig.update_layout(hovermode="closest")
        fig.update_xaxes(title_text="Worst open loss while held (MAE, pips)")
        fig.update_yaxes(title_text="Result (pips)")
        show_chart(style_fig(fig, 340, "How far trades went against you vs how they ended"))
    c1, c2 = st.columns(2)
    with c1:
        counts = view.exit_reason.value_counts()
        fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=BLUE, text=counts.values, textposition="outside"))
        show_chart(style_fig(fig, 300, "Exits by reason"))
    with c2:
        fig = go.Figure(go.Histogram(x=view.duration.dt.total_seconds() / 3600, marker_color=BLUE, nbinsx=30))
        fig.update_xaxes(title_text="Hours held")
        show_chart(style_fig(fig, 300, "Holding time"))


def tab_assets(multi: MultiAssetResult) -> None:
    table = multi.asset_table()
    show_table(table.style.format({
        "weight_pct": "{:.1f}%", "start": "${:,.0f}", "final_equity": "${:,.2f}", "net_pnl": "${:,.2f}",
        "return_pct": "{:+.2f}%", "buy_hold_pct": "{:+.2f}%", "sharpe": "{:.2f}", "sortino": "{:.2f}",
        "max_drawdown_pct": "{:.2f}%", "trades": "{:,.0f}", "win_rate_pct": "{:.1f}%", "profit_factor": "{:.2f}",
        "fees": "${:,.2f}"}))
    if len(multi.per_asset) < 2:
        st.caption("Upload more CSV files (one per asset) to compare assets and see how they combine.")
        return
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for i, (sym, r) in enumerate(multi.per_asset.items()):
            fig.add_trace(go.Scatter(x=r.history.index, y=(r.equity / r.initial_cash - 1) * 100, name=sym,
                                     line=dict(color=SERIES[i % len(SERIES)], width=1.8)))
        fig.update_yaxes(ticksuffix="%")
        show_chart(style_fig(fig, 380, "Return of each asset's sub-account"))
    with c2:
        corr = multi.return_correlation()
        fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                                   colorscale=[[0, ORANGE], [0.5, "#e6e5e1"], [1, BLUE]],
                                   text=corr.round(2).values, texttemplate="%{text}"))
        show_chart(style_fig(fig, 380, "Correlation of daily strategy returns"))


def tab_portfolio(multi: MultiAssetResult) -> None:
    r = multi.combined
    daily = r.daily()
    st.caption("The account at each 17:00 New York close. Balance moves when trades close; "
               "equity includes open trades.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily.index, y=daily.equity, name="Equity", line=dict(color=BLUE, width=2)))
    fig.add_trace(go.Scatter(x=daily.index, y=daily.balance, name="Balance", line=dict(color=GREY, width=1.5, shape="hv")))
    fig.update_yaxes(tickprefix="$")
    show_chart(style_fig(fig, 320))
    h = r.history
    if "long_positions" in h:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=h.index, y=h.long_positions, name="Assets long", line=dict(color=AQUA, shape="hv"), fill="tozeroy"))
        fig.add_trace(go.Scatter(x=h.index, y=-h.short_positions, name="Assets short", line=dict(color=ORANGE, shape="hv"), fill="tozeroy"))
        title = "Open positions (number of assets)"
    else:
        fig = go.Figure(go.Scattergl(x=h.index, y=h.position, name="Units held", line=dict(color=BLUE, shape="hv")))
        title = "Position held (units; negative = short)"
    show_chart(style_fig(fig, 260, title))
    show_table(daily.style.format({"close": "{:.5f}", "position": "{:+,.0f}", "balance": "${:,.2f}",
                                   "open_pnl": "${:,.2f}", "equity": "${:,.2f}", "day_pnl": "${:,.2f}",
                                   "drawdown_pct": "{:.2f}%"}, na_rep="-"), height=420)


def tab_optimize(assets: list[Asset], config: RunConfig, strategy: StrategyChoice) -> None:
    st.caption("Keep data back to check the strategy isn't just fitted to this history. **Out-of-sample**: "
               "search parameters on the first part, then trade the best ones unchanged on the rest. "
               "**Walk-forward**: re-tune before each test window and chain the test windows together.")
    names = [k for k, v in strategy.params.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not names:
        st.info("This strategy has no numeric parameters to tune.")
        return
    c1, c2, c3 = st.columns(3)
    sym = c1.selectbox("Asset", [a.symbol for a in assets], key="opt_asset")
    mode = c2.radio("Test", ["Walk-forward", "Out-of-sample"], horizontal=True)
    objective = c3.selectbox("Maximise", OBJECTIVES, index=0)
    grid = {}
    cols = st.columns(len(names))
    for col, name in zip(cols, names):
        v = strategy.params[name]
        default = ", ".join(str(x) for x in sorted({max(1, round(v * f)) if isinstance(v, int) else round(v * f, 3)
                                                     for f in (0.5, 1, 1.5, 2)}))
        txt = col.text_input(f"{name} values", default, key=f"grid:{name}")
        try:
            vals = [type(v)(float(x)) for x in txt.split(",") if x.strip()]
        except ValueError:
            st.error(f"{name}: use comma-separated numbers")
            return
        grid[name] = vals or [v]
    c1, c2, c3, c4 = st.columns(4)
    min_trades = c1.number_input("Min trades", 0, 10_000, 10)
    if mode == "Out-of-sample":
        split = c2.slider("In-sample share", 0.3, 0.9, 0.7, 0.05)
    else:
        windows = c2.number_input("Test windows", 1, 20, 4)
        train = c3.slider("First training share", 0.2, 0.8, 0.5, 0.05)
        anchored = c4.toggle("Anchored training", value=False)
    n_runs = int(np.prod([len(v) for v in grid.values()]))
    st.caption(f"{n_runs} combinations{'' if mode == 'Out-of-sample' else f' x {windows} windows'} "
               "(invalid ones, like fast >= slow, are skipped).")
    if not st.button("Run test", type="primary"):
        return
    asset = next(a for a in assets if a.symbol == sym)
    df = asset.df
    cfg = replace(config, symbol=sym, point=asset.point or point_size(df.close.median()),
                  units=asset.units or config.units)
    if cfg.sizer is not None and asset.lot_step:
        cfg = replace(cfg, sizer=replace(cfg.sizer, lot_step=asset.lot_step, min_units=asset.lot_step))
    with st.spinner("Testing parameter combinations…"):
        try:
            if mode == "Out-of-sample":
                res = out_of_sample(df, grid, cfg, split=split, objective=objective, min_trades=min_trades)
            else:
                res = walk_forward(df, grid, cfg, windows=int(windows), train_frac=train, anchored=anchored,
                                   objective=objective, min_trades=min_trades)
        except ValueError as exc:
            st.error(str(exc))
            return
    if mode == "Out-of-sample":
        st.success(f"Best on in-sample ({objective}): {res.params}")
        c1, c2 = st.columns([3, 2])
        with c1:
            show_table(res.table.head(15).style.format(precision=2), height=420)
        with c2:
            cmp = res.comparison()
            show_table(cmp.style.format("{:,.2f}"))
        full = res.full
        fig = go.Figure(go.Scatter(x=full.history.index, y=full.equity, name="Equity", line=dict(color=BLUE, width=2)))
        fig.add_vrect(x0=res.split, x1=full.history.index[-1], fillcolor=BLUE, opacity=0.07, line_width=0,
                      annotation_text="out-of-sample", annotation_position="top left")
        fig.add_vline(x=res.split, line=dict(color=GREY, dash="dash"))
        fig.update_yaxes(tickprefix="$")
        show_chart(style_fig(fig, 380, "Chosen parameters over the whole period"))
    else:
        eff = res.efficiency
        st.success(f"Walk-forward efficiency: {eff:.2f} (out-of-sample return per day ÷ in-sample; "
                   "around 0.5 or more suggests the edge carried over)" if eff == eff else
                   "Walk-forward efficiency: n/a (in-sample returns weren't positive)")
        w = res.windows
        show_table(w.drop(columns=["train_days", "test_days"]).style.format(precision=2))
        c = res.combined
        fig = go.Figure(go.Scatter(x=c.history.index, y=c.equity, name="Out-of-sample equity", line=dict(color=BLUE, width=2)))
        for i, row in w.iterrows():
            fig.add_vline(x=row.test_start, line=dict(color=GREY, dash="dot"))
            fig.add_annotation(x=row.test_start, y=1, yref="paper", text=f"W{i}", showarrow=False, xanchor="left")
        fig.update_yaxes(tickprefix="$")
        show_chart(style_fig(fig, 380, "Chained out-of-sample windows"))
        m = c.metrics()
        show_table(m[m.section.isin(["Returns", "Risk"])].set_index("metric")[["display"]])


def html_report(multi: MultiAssetResult, rf: float, settings: dict) -> bytes:
    stats = statistics_frame(multi.combined, multi.benchmark, rf, cal())
    figs = [equity_figure(multi, False), monthly_heatmap(monthly_returns(multi.combined.equity, multi.combined.initial_cash))]
    parts = [f"<h1>{multi.combined.title}</h1>",
             "<p>" + " · ".join(f"<b>{k}</b>: {v}" for k, v in settings.items()) + "</p>",
             figs[0].to_html(include_plotlyjs="cdn", full_html=False),
             figs[1].to_html(include_plotlyjs=False, full_html=False)]
    for section in stats.section.unique():
        part = stats[stats.section == section][["metric", "display"]]
        parts.append(f"<h2>{section}</h2>" + part.to_html(index=False, header=False, border=0))
    log = multi.combined.trade_log()
    parts.append("<h2>Trade log</h2>" + log.to_html(float_format=lambda v: f"{v:,.4f}", border=0))
    css = ("body{font-family:system-ui,sans-serif;max-width:1200px;margin:24px auto;padding:0 16px;color:#222}"
           "table{border-collapse:collapse;font-size:13px}td,th{padding:4px 10px;border-bottom:1px solid #eee;"
           "text-align:right}td:first-child{text-align:left}")
    return f"<!doctype html><meta charset=utf-8><title>{multi.combined.title}</title><style>{css}</style>{''.join(parts)}".encode()


def tab_export(multi: MultiAssetResult, rf: float, settings: dict) -> None:
    r = multi.combined
    stem = re.sub(r"[^A-Za-z0-9]+", "_", r.title).strip("_")[:60] or "backtest"
    files = {
        "Statistics (CSV)": (statistics_frame(r, multi.benchmark, rf, cal()), f"{stem}_statistics.csv", False),
        "Trade log (CSV)": (r.trade_log(), f"{stem}_trade_log.csv", True),
        "Daily portfolio (CSV)": (r.daily(), f"{stem}_daily.csv", True),
        "Every bar (CSV)": (r.history, f"{stem}_history.csv", True),
        "Per-asset summary (CSV)": (multi.asset_table(), f"{stem}_assets.csv", True),
    }
    cols = st.columns(3)
    for i, (label, (frame, name, index)) in enumerate(files.items()):
        cols[i % 3].download_button(label, frame.to_csv(index=index).encode(), name, "text/csv")
    cols[len(files) % 3].download_button("Full report (HTML)", html_report(multi, rf, settings), f"{stem}_report.html",
                                         "text/html", help="Interactive charts, every statistic and the trade log")



# --- QuantConnect results ---------------------------------------------------------------
QC_SAMPLE = Path(__file__).parent / "Smooth Blue Lion.json"


@st.cache_data(show_spinner=False, max_entries=5)
def load_qc(data: bytes) -> QCResult:
    return parse_qc(data)


def render_quantconnect() -> None:
    st.sidebar.header("QuantConnect result")
    st.sidebar.caption("Upload the results file from a QuantConnect backtest (the .json you can download "
                       "from its results page). It's read here, on this computer, and checked for problems.")
    up = st.sidebar.file_uploader("Backtest results (.json)", type=["json"])
    allowed = current_user().can("local_files")
    path = "" if not allowed else st.sidebar.text_input("…or a file path", placeholder=r"C:\results\My Backtest.json")
    data = up.getvalue() if up else None
    if data is None and path.strip():
        p = Path(path.strip().strip('"'))
        if p.exists():
            data = p.read_bytes()
        else:
            st.sidebar.error(f"Not found: {p}")
    if data is None and allowed and QC_SAMPLE.exists() and st.sidebar.toggle(f"Use the {QC_SAMPLE.stem} sample", value=True):
        data = QC_SAMPLE.read_bytes()
    if data is None:
        st.info("Upload a QuantConnect backtest results file in the sidebar.")
        return
    try:
        q = load_qc(data)
    except Exception as exc:  # malformed or unexpected file
        st.error(f"Couldn't read this file: {exc}")
        return

    st.caption(f"{q.name} · {q.start:%Y-%m-%d} to {q.end:%Y-%m-%d} · {q.status or 'status unknown'} · "
               f"{q.currency} account · {len(q.orders):,} orders · {len(q.closed_trades):,} closed trades")
    qc_findings(q)
    qc_tiles(q)
    tabs = st.tabs(["Overview", "Statistics", "Closed trades", "Orders", "QuantConnect notes"])
    with tabs[0]:
        qc_overview(q)
    with tabs[1]:
        qc_statistics_tab(q)
    with tabs[2]:
        qc_trades_tab(q)
    with tabs[3]:
        qc_orders_tab(q)
    with tabs[4]:
        qc_notes_tab(q)


def qc_findings(q: QCResult) -> None:
    high = [f for f in q.findings if f.severity == "high"]
    medium = [f for f in q.findings if f.severity == "medium"]
    low = [f for f in q.findings if f.severity == "info"]
    if high:
        st.subheader("What stands out")
    for f in high:
        st.error(f"**{f.title}.** {f.detail}")
    for f in medium:
        st.warning(f"**{f.title}.** {f.detail}")
    if low:
        with st.expander(f"{len(low)} smaller notes"):
            for f in low:
                st.markdown(f"**{f.title}.** {f.detail}")


def qc_tiles(q: QCResult) -> None:
    nan = float("nan")
    p, ts = q.portfolio, q.trade_stats
    realised = float(q.closed_trades.pnl.sum()) if len(q.closed_trades) else nan
    tiles = [
        ("Net profit", f"{p.get('totalNetProfit', nan) * 100:+.2f}%", money(q.end_equity - q.start_equity)),
        ("Closed-trade P&L", money(realised), f"{len(q.closed_trades)} trades"),
        ("Sharpe", f"{p.get('sharpeRatio', nan):.2f}", f"Sortino {p.get('sortinoRatio', nan):.2f}"),
        ("Prob. Sharpe", f"{p.get('probabilisticSharpeRatio', nan) * 100:.1f}%", None),
        ("Drawdown", f"-{p.get('drawdown', nan) * 100:.1f}%", None),
        ("Win rate", f"{p.get('winRate', nan) * 100:.0f}%", f"payoff {p.get('profitLossRatio', nan):.2f}"),
        ("Beta", f"{p.get('beta', nan):.2f}", f"alpha {p.get('alpha', nan):.2f}"),
        ("Trade profit factor", str(ts.get("profitFactor", "-")), None),
    ]
    arrow = {"delta_arrow": "off"} if "delta_arrow" in inspect.signature(st.metric).parameters else {}
    for row in (tiles[:4], tiles[4:]):
        for col, (label, value, sub) in zip(st.columns(4), row):
            col.metric(label, value, sub, delta_color="off", border=True, **arrow)


def qc_overview(q: QCResult) -> None:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
    if len(q.benchmark):
        fig.add_trace(go.Scatter(x=q.benchmark.index, y=q.benchmark_equity(), name="Benchmark (rebased)",
                                 line=dict(color=GREY, width=1.2, dash="dot")), 1, 1)
    ct = q.closed_trades
    if len(ct):
        bal = pd.concat([pd.Series([q.start_equity], index=[q.equity.index[0]]), ct.set_index("exit_time").balance])
        fig.add_trace(go.Scatter(x=bal.index, y=bal, name="Balance from closed trades only",
                                 line=dict(color=ORANGE, width=1.6, shape="hv")), 1, 1)
    fig.add_trace(go.Scatter(x=q.equity.index, y=q.equity, name="Account equity", line=dict(color=BLUE, width=2)), 1, 1)
    fig.add_hline(y=q.start_equity, line=dict(color=GREY, width=1, dash="dash"), row=1, col=1)
    if len(q.drawdown):
        fig.add_trace(go.Scatter(x=q.drawdown.index, y=q.drawdown, name="Drawdown %", fill="tozeroy",
                                 line=dict(color=RED, width=1), fillcolor="rgba(227,73,72,0.18)"), 2, 1)
    fig.update_yaxes(title_text=q.currency, tickprefix="$", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", ticksuffix="%", row=2, col=1)
    show_chart(style_fig(fig, 560))
    st.caption("The orange line is what the closed trades alone would have done. When it sits below the blue "
               "equity line, the gap is unrealised profit on a position that's still open.")
    c1, c2 = st.columns(2)
    if q.exposure["long"].notna().any() or q.exposure["short"].notna().any():
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=q.exposure.index, y=q.exposure["long"], name="Long",
                                     line=dict(color=AQUA, shape="hv"), fill="tozeroy"))
            fig.add_trace(go.Scatter(x=q.exposure.index, y=-q.exposure["short"], name="Short",
                                     line=dict(color=ORANGE, shape="hv"), fill="tozeroy"))
            fig.update_yaxes(title_text="x equity")
            show_chart(style_fig(fig, 300, "Exposure (position value ÷ equity)"))
    if len(q.turnover):
        with c2:
            fig = go.Figure(go.Bar(x=q.turnover.index, y=q.turnover * 100, marker_color=BLUE))
            fig.update_yaxes(ticksuffix="%")
            show_chart(style_fig(fig, 300, "Portfolio turnover per day"))


QC_TRADE_STAT_NAMES = {
    "totalNumberOfTrades": "Total trades", "numberOfWinningTrades": "Winning trades",
    "numberOfLosingTrades": "Losing trades", "totalProfitLoss": "Total profit/loss", "totalProfit": "Total profit",
    "totalLoss": "Total loss", "largestProfit": "Largest profit", "largestLoss": "Largest loss",
    "averageProfitLoss": "Average profit/loss", "averageProfit": "Average profit", "averageLoss": "Average loss",
    "averageTradeDuration": "Average duration", "medianTradeDuration": "Median duration",
    "maxConsecutiveWinningTrades": "Max consecutive wins", "maxConsecutiveLosingTrades": "Max consecutive losses",
    "profitLossRatio": "Profit-loss ratio", "winLossRatio": "Win-loss ratio", "winRate": "Win rate",
    "averageMAE": "Average MAE", "averageMFE": "Average MFE", "largestMAE": "Largest MAE", "largestMFE": "Largest MFE",
    "maximumClosedTradeDrawdown": "Max closed-trade drawdown", "maximumIntraTradeDrawdown": "Max intra-trade drawdown",
    "profitLossStandardDeviation": "P&L standard deviation", "profitLossDownsideDeviation": "P&L downside deviation",
    "profitFactor": "Profit factor", "sharpeRatio": "Sharpe ratio (per trade)", "sortinoRatio": "Sortino ratio (per trade)",
    "profitToMaxDrawdownRatio": "Profit to max drawdown", "totalFees": "Total fees",
}


def qc_statistics_tab(q: QCResult) -> None:
    st.subheader("Statistics as QuantConnect reported them")
    stat_grid(list(q.statistics.items()), cols=2)
    if q.trade_stats:
        st.subheader("Trade statistics")
        st.caption("Calculated by QuantConnect from the closed trades only.")
        stat_grid([(label, str(q.trade_stats[k])) for k, label in QC_TRADE_STAT_NAMES.items() if k in q.trade_stats],
                  cols=2)
    if q.runtime:
        st.subheader("At the end of the backtest")
        stat_grid(list(q.runtime.items()), cols=2)


def qc_trades_tab(q: QCResult) -> None:
    ct = q.closed_trades
    if ct.empty:
        st.info("No closed trades in this file.")
        return
    st.caption(f"{len(ct)} closed trades, {ct.entry_time.min():%Y-%m-%d} to {ct.exit_time.max():%Y-%m-%d}. "
               "QuantConnect only lists trades that were closed.")
    view = ct.assign(duration=ct.duration.map(format_duration))
    show_table(view, height=380, column_config={
        "entry_time": st.column_config.DatetimeColumn("Entry (UTC)", format="YYYY-MM-DD HH:mm"),
        "exit_time": st.column_config.DatetimeColumn("Exit (UTC)", format="YYYY-MM-DD HH:mm"),
        "quantity": st.column_config.NumberColumn("Units", format="%,.0f"),
        "pnl": st.column_config.NumberColumn("P&L", format="$%.2f"),
        "mae_usd": st.column_config.NumberColumn("MAE", format="$%.2f"),
        "mfe_usd": st.column_config.NumberColumn("MFE", format="$%.2f"),
        "balance": st.column_config.NumberColumn("Balance", format="$%,.2f"),
    })
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(x=list(range(1, len(ct) + 1)), y=ct.pnl, marker_color=np.where(ct.pnl > 0, AQUA, RED),
                               hovertemplate="trade %{x}: $%{y:.2f}<extra></extra>"))
        fig.update_xaxes(title_text="Trade #")
        fig.update_yaxes(tickprefix="$")
        show_chart(style_fig(fig, 330, "P&L per closed trade"))
    with c2:
        fig = go.Figure(go.Scatter(x=ct.mae_usd, y=ct.pnl, mode="markers", name="Trades",
                                   marker=dict(color=np.where(ct.pnl > 0, AQUA, RED), size=8, opacity=0.8,
                                               line=dict(width=1, color="white"))))
        fig.update_layout(hovermode="closest")
        fig.update_xaxes(title_text="Worst open loss while held (MAE, $)", tickprefix="$")
        fig.update_yaxes(title_text="Result", tickprefix="$")
        show_chart(style_fig(fig, 330, "How far trades went against you vs how they ended"))


def qc_orders_tab(q: QCResult) -> None:
    o = q.orders
    if o.empty:
        st.info("No orders in this file.")
        return
    statuses = sorted(o.status.unique())
    pick = st.multiselect("Status", statuses, statuses)
    st.caption("Invalid means the brokerage model refused the order. Cancelled orders were withdrawn before filling.")
    shown = o[o.status.isin(pick)]
    red = "background-color: rgba(227,73,72,0.22)"
    show_table(shown.style.apply(lambda row: [red if row.status == "Invalid" else ""] * len(row), axis=1), height=420)
    show_table(o.groupby(["type", "status"]).size().unstack(fill_value=0))


def qc_notes_tab(q: QCResult) -> None:
    if not q.analysis:
        st.info("QuantConnect didn't attach any notes to this backtest.")
        return
    st.caption("Notes QuantConnect itself generated about this run.")
    for a in q.analysis:
        with st.expander(a.get("name", "Note")):
            st.markdown(f"**Issue.** {a.get('issue', '')}")
            if a.get("sample"):
                st.markdown(f"**Sample.** {a['sample']}")
            for sol in a.get("solutions", []):
                st.markdown(f"- {sol}")



# --- accounts ---------------------------------------------------------------------------------
PERM_LABELS = {
    "run_code": "Run custom Python code (custom strategies and QuantConnect algorithms run on this computer)",
    "local_files": "Read files on this computer by path (and the bundled samples)",
}


def current_user() -> User | None:
    name = st.session_state.get("user")
    return UserStore().get(name) if name else None


def admin_from_secrets(store: UserStore) -> None:
    """Create the admin named in the app's secrets ([admin] username / password / name) if it's missing.

    On Streamlit Cloud the disk is wiped on every restart, so this puts the admin back each time."""
    try:
        cfg = st.secrets.get("admin")
    except Exception:  # no secrets file at all (normal when running locally)
        return
    if not cfg or not cfg.get("username") or not cfg.get("password"):
        return
    if store.get(cfg["username"]) is None:
        try:
            store.create_user(cfg["username"], cfg["password"], cfg.get("name", ""), role="admin")
        except AuthError as exc:
            st.error(f"The admin in the app's secrets couldn't be created: {exc}")


def running_locally() -> bool:
    """True when the browser is on this computer (localhost). Only then can a visitor create the
    first admin account; anywhere else the admin must come from the app's secrets."""
    try:
        host = (st.context.headers.get("host") or "").split(":")[0].lower()
    except Exception:
        host = ""
    return host in ("", "localhost", "127.0.0.1", "[::1]", "::1")


def auth_gate() -> User | None:
    """The signed-in user, or None after drawing the sign-in (or first-run) screen."""
    store = UserStore()
    admin_from_secrets(store)
    name = st.session_state.get("user")
    if name:
        user = store.get(name)
        if user is not None:
            return user
        st.session_state.clear()  # the account was deleted meanwhile
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.title("Backtest Studio")
        if not store.has_users() and not running_locally():
            st.subheader("This app isn't set up yet")
            st.write("No admin account is configured. The owner needs to add one in the app's **Secrets** "
                     "(on Streamlit Cloud: the app's ⋮ menu → Settings → Secrets):")
            st.code('[admin]\nusername = "your-name"\npassword = "a long password"\nname = "Your Name"', language="toml")
            return None
        if not store.has_users():
            st.subheader("Create the admin account")
            st.caption("No accounts exist yet. This first account is the admin: it can add other users "
                       "and decide what they're allowed to do.")
            with st.form("first_admin"):
                username = st.text_input("Username")
                display = st.text_input("Your name")
                pw = st.text_input("Password", type="password", help="At least 8 characters")
                again = st.text_input("Password again", type="password")
                if st.form_submit_button("Create account", type="primary"):
                    try:
                        if pw != again:
                            raise AuthError("The two passwords don't match.")
                        user = store.create_user(username, pw, display, role="admin")
                        st.session_state["user"] = user.username
                        st.rerun()
                    except AuthError as exc:
                        st.error(str(exc))
            return None
        st.subheader("Sign in")
        with st.form("login"):
            username = st.text_input("Username")
            pw = st.text_input("Password", type="password")
            if st.form_submit_button("Sign in", type="primary"):
                try:
                    st.session_state["user"] = store.authenticate(username, pw).username
                    st.rerun()
                except AuthError as exc:
                    st.error(str(exc))
        st.caption("Forgot your password? Ask an admin to reset it, or on the computer running the app: "
                   "`python manage_users.py passwd <username>`.")
    return None


ACCOUNT_PAGES = {"Saved runs": "saved", "Account": "account", "Users": "users"}


def sidebar_account(user: User) -> None:
    c1, c2 = st.sidebar.columns([3, 2])
    c1.markdown(f"**{user.name}**  \n{'Admin' if user.is_admin else 'User'} · {user.username}")
    if c2.button("Sign out", key="sign_out"):
        st.session_state.clear()
        st.rerun()
    # Account pages open in the main area; the sidebar (and its backtest settings) stays as it is.
    labels = [k for k in ACCOUNT_PAGES if k != "Users" or user.is_admin]
    for col, label in zip(st.sidebar.columns(len(labels)), labels):
        if col.button(label, key=f"page:{label}"):
            st.session_state["page"] = ACCOUNT_PAGES[label]
            st.rerun()
    st.sidebar.divider()


def render_account_page(page: str, user: User) -> None:
    if st.button("← Back", key="page_back"):
        st.session_state["page"] = None
        st.rerun()
    if page == "saved":
        render_saved_runs(user)
    elif page == "account":
        render_account(user)
    elif page == "users" and user.is_admin:
        render_admin(user)


def saved_strategies_ui(state_key: str, kind: str) -> None:
    """Load / save / delete the user's own strategy code (shown above the code box)."""
    user = current_user()
    names = workspace.list_strategies(user.username)
    with st.sidebar.expander(f"My saved strategies ({len(names)})"):
        if names:
            pick = st.selectbox("Saved", names, key=f"pick:{kind}")
            c1, c2 = st.columns(2)
            if c1.button("Load", key=f"load:{kind}"):
                st.session_state[state_key] = workspace.load_strategy(user.username, pick)
                st.rerun()
            if c2.button("Delete", key=f"del:{kind}"):
                workspace.delete_strategy(user.username, pick)
                st.rerun()
        name = st.text_input("Save the code below as", key=f"savename:{kind}", placeholder="e.g. EMA trend v2")
        if st.button("Save", key=f"save:{kind}"):
            try:
                saved = workspace.save_strategy(user.username, name, st.session_state.get(state_key, ""))
                st.success(f"Saved as {saved}. Only you can see it.")
            except ValueError as exc:
                st.error(str(exc))


QC_CHOICE = "QuantConnect algorithm (Python)"


def sidebar_qc_algorithm() -> StrategyChoice:
    st.sidebar.caption("Paste an algorithm written for QuantConnect (a class with initialize and on_data). "
                       "The symbols it adds (add_forex, add_equity...) are matched to the symbols of your "
                       "uploaded CSVs. Orders and position sizes come from the algorithm.")
    saved_strategies_ui("qc_code", "qc")
    up = st.sidebar.file_uploader("Load an algorithm .py file", type=["py"], key="qc_upload")
    if up is not None:
        st.session_state["qc_code"] = up.getvalue().decode("utf-8")
    code = st.sidebar.text_area("Algorithm code", st.session_state.get("qc_code", QC_TEMPLATE), height=380,
                                help="from AlgorithmImports import * works. Runs on this computer like any Python script.")
    st.session_state["qc_code"] = code
    m = re.search(r"class\s+(\w+)\s*\(\s*QCAlgorithm", code)
    return StrategyChoice(m.group(1) if m else "QuantConnect algorithm", None, {}, None, kind="qc", code=code)


def sidebar_execution_algo() -> dict:
    st.sidebar.header("3 · Orders & sizing")
    st.sidebar.caption("Set by the algorithm: set_holdings, market_order, limit_order, stop_market_order, liquidate...")
    st.sidebar.header("4 · Capital & costs")
    cash = st.sidebar.number_input("Starting capital (USD)", 100.0, 1e9, 10_000.0, 1_000.0,
                                   help="Used when the algorithm doesn't call set_cash")
    slip_kind = st.sidebar.radio("Slippage", ["None", "Fixed"], horizontal=True, key="qc_slip_kind",
                                 help="Points against you on market, stop and market-on-open fills")
    slip = st.sidebar.number_input("Slippage (points)", 0.0, 10_000.0, 2.0, 0.5, key="qc_slip") if slip_kind == "Fixed" else 0.0
    rf = st.sidebar.number_input("Risk-free rate (% a year)", 0.0, 20.0, 0.0, 0.25,
                                 help="Used for Sharpe, Sortino, alpha and Treynor") / 100
    st.sidebar.toggle("QuantConnect-style daily returns", value=True, key="qc_calendar",
                      help="On: weekends count as zero-return days, as QuantConnect samples them.")
    return dict(entry="market", offset=0.0, expiry=None, sl=None, tp=None, trail=None, cash=cash,
                sizing="Set by the algorithm", units=None, risk=None, eq_pct=None, leverage=30.0, lot=1_000.0,
                slip_kind=slip_kind, slip=slip, rf=rf)


def tab_algorithm(run) -> None:
    st.caption(f"{run.name}: what the algorithm printed and every order it placed.")
    for chart, series in run.plots.items():
        fig = go.Figure()
        for i, (name, points) in enumerate(series.items()):
            xs, ys = zip(*points)
            fig.add_trace(go.Scattergl(x=list(xs), y=list(ys), name=name, line=dict(color=SERIES[i % len(SERIES)], width=1.5)))
        show_chart(style_fig(fig, 320, f"{chart} (self.plot)"))
    st.subheader(f"Log ({len(run.logs):,} lines)")
    if run.logs:
        st.code("\n".join(run.logs[-1_000:]), language=None)
        st.download_button("Download the full log", "\n".join(run.logs).encode(), "algorithm_log.txt", "text/plain")
    else:
        st.caption("Nothing printed. self.debug(...) and self.log(...) lines show up here.")
    st.subheader(f"Orders ({len(run.orders):,})")
    if len(run.orders):
        counts = run.orders.status.value_counts()
        st.caption(" · ".join(f"{k}: {v}" for k, v in counts.items()) + " (order times in the algorithm's time zone)")
        show_table(run.orders, height=380, hide_index=True)


def save_run_ui(multi: MultiAssetResult, rf: float, settings: dict) -> None:
    user = current_user()
    st.subheader("Save to my account")
    st.caption("Keeps the statistics, trade log, daily portfolio and the full report under **My saved runs**. "
               "Only you can see them.")
    c1, c2 = st.columns([3, 1])
    name = c1.text_input("Name", multi.combined.title, key="save_run_name", label_visibility="collapsed")
    if c2.button("Save this run", key="save_run", type="primary"):
        r = multi.combined
        q = {m.key: m for m in qc_statistics(r, multi.benchmark, rf, cal())}
        headline = {label: format_value(q[k].value, q[k].kind) for label, k in [
            ("Net profit", "net_profit_pct"), ("CAGR", "cagr_pct"), ("Sharpe", "sharpe"), ("Sortino", "sortino"),
            ("Drawdown", "drawdown_pct"), ("Win rate", "win_rate_pct"), ("Total orders", "total_orders")]}
        files = {
            "statistics.csv": statistics_frame(r, multi.benchmark, rf, cal()).to_csv(index=False).encode(),
            "trade_log.csv": r.trade_log().to_csv().encode(),
            "daily.csv": r.daily().to_csv().encode(),
            "report.html": html_report(multi, rf, settings),
        }
        try:
            workspace.save_run(user.username, name or r.title, settings, headline, files)
            st.success("Saved. Open it any time with **Saved runs** under your name in the sidebar.")
        except ValueError as exc:
            st.error(str(exc))


def render_saved_runs(user: User) -> None:
    st.subheader("My saved runs")
    runs = workspace.list_runs(user.username)
    if not runs:
        st.info("Nothing saved yet. Run a backtest, then use **Save this run** on its Export tab.")
        return
    when = lambda r: r.get("created", "")[:16].replace("T", " ") + " UTC"  # noqa: E731
    show_table(pd.DataFrame([{"Saved": when(r), "Name": r["title"], **r.get("headline", {})} for r in runs]),
               hide_index=True)
    labels = {f"{when(r)} · {r['title']}": r for r in runs}
    run = labels[st.selectbox("Open a run", list(labels))]
    c1, c2 = st.columns(2)
    with c1:
        stat_grid(list(run.get("headline", {}).items()), cols=1, title="Headline")
    with c2:
        stat_grid([(k, str(v)) for k, v in run.get("settings", {}).items()], cols=1, title="Settings")
    cols = st.columns(max(1, len(run.get("files", []))))
    for col, f in zip(cols, run.get("files", [])):
        col.download_button(f, workspace.run_file(user.username, run["id"], f), f"{run['id']}_{f}",
                            key=f"dl:{run['id']}:{f}")
    if "report.html" in run.get("files", []):
        with st.expander("View the report here"):
            import streamlit.components.v1 as components
            components.html(workspace.run_file(user.username, run["id"], "report.html").decode("utf-8"),
                            height=900, scrolling=True)
    confirm = st.checkbox("Yes, delete this run", key=f"confirm:{run['id']}")
    if st.button("Delete run", disabled=not confirm, key=f"delrun:{run['id']}"):
        workspace.delete_run(user.username, run["id"])
        st.rerun()


def render_account(user: User) -> None:
    st.subheader("My account")
    perms = "all (admin)" if user.is_admin else (", ".join(user.permissions) or "none")
    stat_grid([("Username", user.username), ("Name", user.name), ("Role", user.role), ("Permissions", perms),
               ("Last sign-in", (user.last_login[:16].replace("T", " ") + " UTC") if user.last_login else "-")], cols=1)
    st.subheader("Change password")
    with st.form("change_password", clear_on_submit=True):
        current = st.text_input("Current password", type="password")
        new = st.text_input("New password", type="password", help="At least 8 characters")
        again = st.text_input("New password again", type="password")
        if st.form_submit_button("Change password"):
            try:
                if new != again:
                    raise AuthError("The two new passwords don't match.")
                UserStore().set_password(user.username, new, current_password=current)
                st.success("Password changed.")
            except AuthError as exc:
                st.error(str(exc))


def render_admin(user: User) -> None:
    st.subheader("Manage users")
    store = UserStore()
    users = store.list_users()
    show_table(pd.DataFrame([{
        "Username": u.username, "Name": u.name, "Role": u.role,
        "Permissions": "all" if u.is_admin else (", ".join(u.permissions) or "-"),
        "Created": u.created[:10], "Last sign-in": u.last_login[:16].replace("T", " ") or "never",
    } for u in users]), hide_index=True)
    st.caption("Admins can do everything. Give **run code** only to people you trust: their strategies run "
               "as real Python on this computer, with access to its files.")

    with st.expander("Add a user", expanded=len(users) == 1):
        with st.form("add_user", clear_on_submit=True):
            c1, c2 = st.columns(2)
            username = c1.text_input("Username")
            name = c2.text_input("Name")
            pw = c1.text_input("Temporary password", type="password", help="Tell them to change it under My account")
            role = c2.selectbox("Role", ["user", "admin"])
            perms = [p for p in PERMISSIONS if st.checkbox(PERM_LABELS[p], key=f"newperm:{p}")]
            if st.form_submit_button("Add user", type="primary"):
                try:
                    store.create_user(username, pw, name, role, perms)
                    st.success(f"Added {username.strip().lower()}.")
                except AuthError as exc:
                    st.error(str(exc))

    st.markdown("**Change a user**")
    target = st.selectbox("User", [u.username for u in users], key="admin_target")
    t = store.get(target)
    c1, c2 = st.columns(2)
    with c1, st.form(f"edit:{target}"):
        role = st.selectbox("Role", ["user", "admin"], index=1 if t.is_admin else 0)
        perms = [p for p in PERMISSIONS if st.checkbox(PERM_LABELS[p], value=p in t.permissions, key=f"perm:{target}:{p}")]
        if st.form_submit_button("Save changes"):
            try:
                store.update(target, role=role, permissions=perms)
                st.success("Saved.")
                st.rerun()
            except AuthError as exc:
                st.error(str(exc))
    with c2:
        with st.form(f"reset:{target}", clear_on_submit=True):
            pw = st.text_input(f"New password for {target}", type="password")
            if st.form_submit_button("Reset password"):
                try:
                    store.set_password(target, pw)
                    st.success("Password reset.")
                except AuthError as exc:
                    st.error(str(exc))
        if st.button("Unlock (after too many wrong passwords)", key=f"unlock:{target}"):
            store.unlock(target)
            st.success("Unlocked.")
        own = target == user.username
        confirm = st.checkbox(f"Yes, delete {target}", key=f"confirmdel:{target}", disabled=own)
        if st.button("Delete user", key=f"deluser:{target}", disabled=own or not confirm,
                     help="You can't delete your own account" if own else None):
            try:
                store.delete_user(target)
                st.rerun()
            except AuthError as exc:
                st.error(str(exc))


# --- main ---------------------------------------------------------------------------------
def main() -> None:
    user = auth_gate()
    if user is None:
        return
    st.title("Backtest Studio")
    sidebar_account(user)
    pages = ["Run a backtest", "Analyse a QuantConnect result"]
    if st.session_state.get("mode") not in pages:
        st.session_state["mode"] = pages[0]
    mode = st.sidebar.radio("Mode", pages, key="mode",
                            help="Run your own backtest here, or load a results file from QuantConnect "
                                 "to see its statistics and check it for problems.")
    st.sidebar.divider()
    page = st.session_state.get("page")
    if mode == "Analyse a QuantConnect result":
        if page:
            render_account_page(page, user)
        else:
            render_quantconnect()
        return
    assets, timeframe, notes = sidebar_data()
    strategy = sidebar_strategy()
    execution = sidebar_execution(algo_mode=strategy is not None and strategy.kind == "qc")
    run = st.sidebar.button("Run backtest", type="primary", width="stretch") if _supports_width() else \
        st.sidebar.button("Run backtest", type="primary", use_container_width=True)
    if run:
        st.session_state["page"] = None
    elif st.session_state.get("page"):
        render_account_page(st.session_state["page"], user)
        return

    if not assets:
        st.info("Upload one or more CSV files in the sidebar to start: one file per asset, with a date/time column "
                "and open, high, low, close prices.")
        return
    for note in notes:
        st.warning(note)
    if strategy is None:
        return

    config = build_config(execution, strategy)
    settings = {
        "Assets": ", ".join(f"{a.symbol} ({a.weight:g})" for a in assets), "Timeframe": timeframe,
        "Strategy": f"{strategy.name} {strategy.params}" + (" long only" if strategy.long_only else ""),
        "Orders": config.orders.describe(), "Capital": money(execution["cash"]), "Sizing": execution["sizing"],
        "Slippage": execution["slip_kind"] + (f" {execution['slip']:g} pts" if execution["slip_kind"] != "None" else ""),
    }
    if strategy.kind == "qc":
        settings.update(Strategy=f"{strategy.name} (QuantConnect algorithm)", Orders="set by the algorithm")
    signature = repr((settings, [len(a.df) for a in assets], execution, strategy.code))

    if execution["sizing"] == "Fixed units" and strategy.kind != "qc":
        assets = resolve_fixed_units(assets, execution["cash"], execution["units"])
        total_w = sum(a.weight for a in assets) or 1
        for a in assets:
            capital = execution["cash"] * a.weight / total_w
            notional = Portfolio(a.symbol).notional_usd(a.units, float(a.df.close.iloc[0]))
            if capital and notional / capital > 30:
                st.warning(f"{a.symbol}: {a.units:,.4g} units is about {money(notional)}, "
                           f"{notional / capital:,.0f}x its {money(capital)} of capital. "
                           "Use % of equity or % risk sizing, or fewer units.")

    if run:
        if execution["sizing"] == "% risk per trade" and not (execution["sl"] or execution["trail"]):
            st.error("Risk sizing needs a stop-loss or trailing stop (sidebar, section 3).")
            return
        if strategy.custom_fn is not None:
            problems = check_lookahead(strategy.custom_fn, assets[0].df, strategy.params)
            for p in problems[:2]:
                st.warning(f"Look-ahead check: {p}. Results will be too good to be true.")
        title = f"{strategy.name} {', '.join(f'{k}={v}' for k, v in strategy.params.items())}".strip()
        started = time.perf_counter()
        algo_run = None
        try:
            with st.spinner(f"Backtesting {sum(len(a.df) for a in assets):,} bars…"):
                if strategy.kind == "qc":
                    algo_run = run_algorithm(strategy.code, assets, cash=execution["cash"], slippage=config.slippage,
                                             title=strategy.name)
                    multi = algo_run.result
                else:
                    multi = run_multi(assets, config, strategy.params, title=title)
        except AlgorithmError as exc:
            st.error(f"The algorithm stopped with an error: {exc}")
            return
        except Exception as exc:  # show engine / user-code errors in the page
            st.error(f"Backtest failed: {exc}")
            return
        st.session_state.update(result=multi, result_algo=algo_run, result_assets=assets, result_sig=signature,
                                result_settings=settings, elapsed=time.perf_counter() - started)

    multi = st.session_state.get("result")
    if multi is None:
        st.info("Set up the strategy in the sidebar and press **Run backtest**.")
        preview = go.Figure()
        for i, a in enumerate(assets):
            preview.add_trace(go.Scattergl(x=a.df.index, y=a.df.close / a.df.close.iloc[0] * 100, name=a.symbol,
                                           line=dict(color=SERIES[i % len(SERIES)], width=1.5)))
        preview.update_yaxes(title_text="Price (start = 100)")
        show_chart(style_fig(preview, 380, "Loaded data"))
        return
    if st.session_state.get("result_sig") != signature:
        st.warning("Settings changed since this run. Press **Run backtest** to update the results.")
    run_assets = st.session_state["result_assets"]
    st.caption(f"{multi.combined.title} · {', '.join(multi.per_asset)} · "
               f"{multi.combined.history.index[0]:%Y-%m-%d} to {multi.combined.history.index[-1]:%Y-%m-%d} UTC · "
               f"ran in {st.session_state['elapsed']:.1f}s")
    rf = execution["rf"]
    algo_run = st.session_state.get("result_algo")
    names = ["Overview", "Statistics", "Returns", "Trades", "Assets", "Portfolio", "Optimise", "Export"]
    tabs = st.tabs(names + (["Algorithm"] if algo_run else []))
    with tabs[0]:
        tab_overview(multi, run_assets, rf)
    with tabs[1]:
        tab_statistics(multi, rf)
    with tabs[2]:
        tab_returns(multi, rf)
    with tabs[3]:
        tab_trades(multi)
    with tabs[4]:
        tab_assets(multi)
    with tabs[5]:
        tab_portfolio(multi)
    with tabs[6]:
        if algo_run:
            st.info("Parameter search and walk-forward work with the built-in and signals strategies. For a "
                    "pasted QuantConnect algorithm, change its parameters in the code and run again.")
        else:
            tab_optimize(run_assets, config, strategy)
    with tabs[7]:
        tab_export(multi, rf, st.session_state["result_settings"])
        st.divider()
        save_run_ui(multi, rf, st.session_state["result_settings"])
    if algo_run:
        with tabs[8]:
            tab_algorithm(algo_run)


def _supports_width() -> bool:
    return "width" in inspect.signature(st.button).parameters


main()
