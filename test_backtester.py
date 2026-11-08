import numpy as np
import pandas as pd
import pytest

from backtest_engine import (BarFeed, MovingAverageCrossover, Portfolio, SimulatedBroker, point_size,
                        run_backtest)
from backtest_engine.data_loader import DEFAULT_FILE, MT5_SERVER_TZ


def random_walk(n=3000, start=150.0, step=0.05, spread=3.0, seed=0):
    rng = np.random.default_rng(seed)
    close = start + np.cumsum(rng.normal(0, step, n))
    open_ = np.r_[start, close[:-1]]
    wiggle = np.abs(rng.normal(0, step / 2, n))
    idx = pd.date_range("2026-01-05", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": open_, "high": np.maximum(open_, close) + wiggle, "low": np.minimum(open_, close) - wiggle,
        "close": close, "volume": 1.0, "spread": spread,
    }, index=idx)


def backtest(df, symbol="USDJPY", fast=10, slow=30, units=10_000, long_only=False, point=0.001):
    portfolio = Portfolio(symbol, cash=10_000, units=units, point=point)
    result = run_backtest(BarFeed(df), MovingAverageCrossover(fast, slow, long_only), portfolio,
                          SimulatedBroker(point))
    return result, portfolio


def expected_crosses(df, fast, slow):
    """Vectorised reference: bars where the fast SMA crosses the slow SMA."""
    f, s = df.close.rolling(fast).mean(), df.close.rolling(slow).mean()
    state = np.sign(f - s).replace(0, np.nan).ffill()
    return state[(state != state.shift()) & state.shift().notna()]


def test_fills_happen_at_next_bar_open_with_no_lookahead():
    df = random_walk()
    result, _ = backtest(df)
    crosses = expected_crosses(df, 10, 30)
    signal_pos = [df.index.get_loc(t) for t in crosses.index]
    # Every cross on bar i fills at bar i+1's open (a cross on the very last bar never fills).
    fill_bars = [df.index[i + 1] for i in signal_pos if i + 1 < len(df)]
    signal_fills = result.fills[result.fills.reason == "signal"]
    # A flip is two fills at the same open: close the old position, then enter the new one.
    assert list(signal_fills.time.drop_duplicates()) == fill_bars
    for row in signal_fills.itertuples():
        bid = df.open[row.time]
        expected = bid + df.spread[row.time] * 0.001 if row.units > 0 else bid
        assert row.price == pytest.approx(expected)


def test_positions_flip_between_full_long_and_short():
    result, portfolio = backtest(random_walk())
    held = result.history.position
    after_first_entry = held[held.ne(0).idxmax():].iloc[:-1]
    assert set(after_first_entry.round()) == {10_000, -10_000}  # always in, never flat or doubled
    assert portfolio.position == 0  # closed at the end


def test_long_only_never_goes_short():
    result, _ = backtest(random_walk(), long_only=True)
    assert (result.fills.units.cumsum().round() >= 0).all()
    assert set(result.trades.side) == {"long"}


def test_accounting_adds_up():
    result, portfolio = backtest(random_walk())
    assert portfolio.cash == pytest.approx(10_000 + result.trades.pnl.sum())
    assert result.equity.iloc[-1] == pytest.approx(portfolio.cash)
    assert result.stats()["trades"] == len(result.trades) > 0


def test_usdjpy_pnl_converted_to_usd_at_exit():
    result, _ = backtest(random_walk())
    t = result.trades.iloc[0]
    direction = 1 if t.side == "long" else -1
    assert t.pnl == pytest.approx(t.units * direction * (t.exit_price - t.entry_price) / t.exit_price)
    assert t.pips == pytest.approx(direction * (t.exit_price - t.entry_price) / 0.01)


def test_spread_cost_is_exact_for_usd_quote_pair():
    # EURUSD-style: P&L is already in USD, so spread cost = units bought x spread x point.
    base = random_walk(start=1.10, step=0.0005, spread=0.0)
    no_cost, _ = backtest(base, symbol="EURUSD", point=0.00001)
    with_cost, _ = backtest(base.assign(spread=10.0), symbol="EURUSD", point=0.00001)
    bought = with_cost.fills.units.clip(lower=0).sum()
    assert no_cost.trades.pnl.sum() - with_cost.trades.pnl.sum() == pytest.approx(bought * 10 * 0.00001)


def test_portfolio_history_tracks_every_bar():
    df = random_walk()
    result, _ = backtest(df)
    h = result.portfolio_history()
    assert list(h.index) == list(df.index)
    assert h.equity.to_numpy() == pytest.approx((h.balance + h.open_pnl).to_numpy())
    assert (h.drawdown_pct <= 0).all()
    assert h.position.iloc[-1] == 0 and h.open_pnl.iloc[-1] == 0  # closed out at the end
    flat = h.position == 0
    assert (h.open_pnl[flat] == 0).all() and h.avg_price[flat].isna().all()


def test_daily_portfolio_adds_up():
    result, _ = backtest(random_walk())
    d = result.daily()
    assert 10_000 + d.day_pnl.sum() == pytest.approx(result.equity.iloc[-1])
    assert d.trades_closed.sum() == len(result.trades)


def test_equity_chart_and_csv_export(tmp_path):
    from backtest_engine.report import save_csv

    result, _ = backtest(random_walk())
    fig = result.plot_equity(tmp_path / "equity.png")
    assert (tmp_path / "equity.png").stat().st_size > 10_000
    assert len(fig.axes) == 3
    for path in save_csv(result, tmp_path, "run"):
        assert path.exists() and path.stat().st_size > 0


def test_trade_log_running_balance_and_numbering():
    result, portfolio = backtest(random_walk())
    log = result.trade_log()
    assert list(log.index) == list(range(1, len(log) + 1))
    assert log.balance.iloc[-1] == pytest.approx(portfolio.cash)
    assert log.cum_pnl.iloc[-1] == pytest.approx(log.pnl.sum())
    before = log.balance - log.pnl
    assert log.return_pct.to_numpy() == pytest.approx((log.pnl / before * 100).to_numpy())
    assert set(log.exit_reason) <= {"signal", "end of data"}
    assert log.exit_reason.iloc[-1] == "end of data"


def test_mae_and_mfe_bound_each_trade():
    log = backtest(random_walk())[0].trade_log()
    assert (log.mae_pips <= 0).all() and (log.mfe_pips >= 0).all()
    assert (log.mae_pips <= log.pips + 1e-9).all() and (log.pips <= log.mfe_pips + 1e-9).all()


def test_trade_spread_cost_matches_what_was_charged():
    # USD-quote pair, so no conversion: each trade's spread = units x spread x point (one buy per round trip).
    df = random_walk(start=1.10, step=0.0005, spread=10.0)
    result, _ = backtest(df, symbol="EURUSD", point=0.00001)
    assert result.trades.spread_cost.to_numpy() == pytest.approx(10_000 * 10 * 0.00001)
    assert result.stats()["spread_paid"] == pytest.approx(result.fills.spread_cost.sum())


def test_bars_held_add_up_to_time_in_market():
    result, _ = backtest(random_walk())
    in_market = (result.history.position != 0).sum()
    # The final close-out happens at the last bar's close, after that bar was counted.
    assert result.trades.bars_held.sum() == in_market + 1


def test_metrics_are_consistent():
    result, _ = backtest(random_walk())
    s, t = result.stats(), result.trades
    assert s["winners"] + s["losers"] <= s["trades"] == len(t)
    assert s["gross_profit"] + s["gross_loss"] == pytest.approx(t.pnl.sum())
    assert s["net_pnl"] == pytest.approx(t.pnl.sum())
    assert s["profit_factor"] == pytest.approx(s["gross_profit"] / -s["gross_loss"])
    assert s["expectancy"] == pytest.approx(t.pnl.mean())
    assert s["max_drawdown_pct"] <= 0 and s["max_drawdown_usd"] <= 0
    assert 0 <= s["exposure_pct"] <= 100
    m = result.metrics()
    assert list(m.section.unique()) == ["Returns", "Risk", "Trades", "Exposure"]
    assert m.display.notna().all()
    sides = result.by_side()
    assert sides.loc["Long", "trades"] + sides.loc["Short", "trades"] == sides.loc["All", "trades"]
    assert sides.loc["All", "net_pnl"] == pytest.approx(s["net_pnl"])


def test_max_consecutive_streaks():
    from backtest_engine.metrics import _streak
    assert _streak(pd.Series([True, True, False, True, True, True, False])) == 3
    assert _streak(pd.Series([], dtype=bool)) == 0


def test_longest_drawdown_measures_peak_to_recovery():
    from backtest_engine.metrics import longest_drawdown
    idx = pd.date_range("2026-01-01", periods=6, freq="1D", tz="UTC")
    eq = pd.Series([100, 90, 95, 101, 99, 100], index=idx, dtype=float)
    # Peak 100 on day 0, regained on day 3 (3 days); peak 101 on day 3, still below at the end (2 days).
    assert longest_drawdown(eq) == pd.Timedelta(days=3)


def test_report_prints_metrics_and_full_trade_log():
    from rich.console import Console

    from backtest_engine.report import print_report, print_trade_log

    result, _ = backtest(random_walk())
    console = Console(record=True, width=200)
    print_report(result, console, trades=5)
    text = console.export_text()
    for heading in ["Returns", "Risk", "Trades", "Exposure", "Long vs short", "Trade log (last 5 of"]:
        assert heading in text
    console = Console(record=True, width=200)
    print_trade_log(result, console, last=None)
    assert f"Trade log (all {len(result.trades)} trades)" in console.export_text()


def test_bad_inputs():
    with pytest.raises(ValueError):
        MovingAverageCrossover(50, 20)
    with pytest.raises(ValueError):
        Portfolio("EURGBP")
    with pytest.raises(ValueError):
        BarFeed(random_walk().iloc[::-1])
    assert point_size(155.0) == 0.001 and point_size(1.1) == 0.00001


@pytest.mark.skipif(not DEFAULT_FILE.exists(), reason="MT5 USDJPY export not on this machine")
def test_runs_on_real_usdjpy_data():
    feed = BarFeed.from_csv(DEFAULT_FILE, tz=MT5_SERVER_TZ, timeframe="1h")
    portfolio = Portfolio("USDJPY", units=10_000, point=0.001)
    result = run_backtest(feed, MovingAverageCrossover(20, 50), portfolio, SimulatedBroker(0.001))
    assert len(result.equity) == len(feed)
    assert len(result.trades) > 0
    assert result.equity.iloc[-1] == pytest.approx(10_000 + result.trades.pnl.sum())
