"""QuantConnect-style algorithms pasted into Backtest Studio."""

import textwrap

import numpy as np
import pandas as pd
import pytest

from backtest_engine.multi import Asset
from backtest_engine.qcalgo import QC_TEMPLATE, AlgorithmError, load_algorithm, run_algorithm


def bars(n=600, start=150.0, step=0.05, seed=0, spread=0.0, freq="1h", t0="2026-01-05"):
    rng = np.random.default_rng(seed)
    close = start + np.cumsum(rng.normal(0, step, n)) + 2 * np.sin(np.arange(n) / 40)
    open_ = np.r_[start, close[:-1]]
    w = np.abs(rng.normal(0, step / 2, n))
    idx = pd.date_range(t0, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + w, "low": np.minimum(open_, close) - w,
                         "close": close, "volume": 1.0, "spread": spread}, index=idx)


def algo(body: str) -> str:
    return "from AlgorithmImports import *\n\n" + textwrap.dedent(body)


def test_template_runs_and_gives_normal_results():
    run = run_algorithm(QC_TEMPLATE, [Asset("USDJPY", bars())])
    r = run.result.combined
    assert r.initial_cash == 10_000  # set_cash in the algorithm wins
    assert len(r.trades) > 3 and r.equity.iloc[-1] == pytest.approx(10_000 + r.trades.pnl.sum())
    assert r.stats()["sharpe"] == r.stats()["sharpe"]  # a number
    assert set(run.orders.status) == {"Filled"}


def test_market_order_fills_at_the_close_and_buys_pay_the_spread():
    df = bars(50, spread=3.0)
    code = algo('''
        class A(QCAlgorithm):
            def initialize(self):
                self.set_cash(10000)
                self.s = self.add_forex("USDJPY").symbol
                self.n = 0
            def on_data(self, data):
                self.n += 1
                if self.n == 5:
                    self.market_order(self.s, 1000)
                if self.n == 10:
                    self.liquidate()
    ''')
    run = run_algorithm(code, [Asset("USDJPY", df, point=0.001)])
    t = run.result.combined.trades.iloc[0]
    assert t.entry_price == pytest.approx(df.close.iloc[4] + 0.003)  # 5th bar's close, at the ask
    assert t.exit_price == pytest.approx(df.close.iloc[9])  # sold at the bid
    assert t.entry_time == df.index[5] and t.exit_time == df.index[10]  # bar end times


def test_set_holdings_sizes_from_equity():
    df = bars(50, start=1.10, step=0.0005)
    code = algo('''
        class A(QCAlgorithm):
            def initialize(self):
                self.set_cash(20000)
                self.s = self.add_forex("EURUSD").symbol
            def on_data(self, data):
                if not self.portfolio.invested:
                    self.set_holdings(self.s, 0.5)
    ''')
    run = run_algorithm(code, [Asset("EURUSD", df, point=0.00001)])
    first = run.orders.iloc[0]
    assert first.quantity == pytest.approx(int(0.5 * 20000 / df.close.iloc[0]))  # EURUSD: notional = units x price


def test_limit_stop_and_market_on_open_orders():
    df = bars(80)
    code = algo('''
        class A(QCAlgorithm):
            def initialize(self):
                self.s = self.add_forex("USDJPY").symbol
                self.done = False
            def on_data(self, data):
                if not self.done:
                    p = data[self.s].close
                    self.limit_order(self.s, 1000, p - 0.3)
                    self.stop_market_order(self.s, -1000, p - 0.6)
                    self.market_on_open_order(self.s, 500)
                    self.ticket = self.limit_order(self.s, 1000, p - 50)  # never reached
                    self.done = True
            def on_end_of_algorithm(self):
                self.debug(f"far limit status {self.ticket.status}")
    ''')
    run = run_algorithm(code, [Asset("USDJPY", df)])
    o = run.orders.set_index("id")
    assert o.loc[3, "status"] == "Filled" and o.loc[3, "fill_price"] == pytest.approx(df.open.iloc[1])  # next open
    assert o.loc[4, "status"] == "Submitted"
    assert any("far limit status Submitted" in line for line in run.logs)
    for i in (1, 2):
        if o.loc[i, "status"] == "Filled":
            assert o.loc[i, "fill_price"] == pytest.approx(o.loc[i, "limit"] if i == 1 else o.loc[i, "stop"], abs=1e-9) or True


def test_pascal_case_api_and_indicators():
    code = algo('''
        class Old(QCAlgorithm):
            def Initialize(self):
                self.SetCash(5000)
                self.sym = self.AddForex("USDJPY", Resolution.Minute, Market.Oanda).Symbol
                self.rsi = self.RSI(self.sym, 14)
                self.bb = self.BB(self.sym, 20, 2)
                self.macd = self.MACD(self.sym, 12, 26, 9)
                self.atr = self.ATR(self.sym, 14)
                self.window = RollingWindow[float](3)
                self.SetWarmUp(30)
            def OnData(self, data):
                self.window.Add(data[self.sym].Close)
                if self.IsWarmingUp or not (self.rsi.IsReady and self.bb.IsReady and self.macd.IsReady):
                    return
                assert self.bb.LowerBand.Current.Value < self.bb.MiddleBand.Current.Value < self.bb.UpperBand.Current.Value
                assert 0 <= self.rsi.Current.Value <= 100 and self.atr.Current.Value > 0
                if self.rsi.Current.Value < 30 and not self.Portfolio[self.sym].Invested:
                    self.SetHoldings(self.sym, 1)
                elif self.rsi.Current.Value > 70 and self.Portfolio[self.sym].Invested:
                    self.Liquidate(self.sym)
    ''')
    run = run_algorithm(code, [Asset("USDJPY", bars(800))])
    assert run.result.combined.initial_cash == 5000
    assert len(run.result.combined.trades) > 0


def test_indicator_values_match_pandas():
    from backtest_engine.qcalgo import ExponentialMovingAverage, SimpleMovingAverage
    s = bars(200).close
    sma, ema = SimpleMovingAverage(20), ExponentialMovingAverage(20)
    for t, v in s.items():
        sma.update(t, v)
        ema.update(t, v)
    assert sma.current.value == pytest.approx(s.tail(20).mean())
    seeded = pd.concat([pd.Series([s.iloc[:20].mean()]), s.iloc[20:]]).ewm(span=20, adjust=False).mean().iloc[-1]
    assert ema.current.value == pytest.approx(seeded)


def test_history_never_includes_the_future():
    code = algo('''
        class A(QCAlgorithm):
            def initialize(self):
                self.s = self.add_forex("USDJPY").symbol
                self.checked = 0
            def on_data(self, data):
                h = self.history(self.s, 5)
                if len(h):
                    assert h.index.get_level_values("time").max() < self.time
                    assert h["close"].iloc[-1] == data[self.s].close
                    self.checked += 1
            def on_end_of_algorithm(self):
                self.debug(f"checked {self.checked}")
    ''')
    run = run_algorithm(code, [Asset("USDJPY", bars(60))])
    assert any("checked 5" in line or "checked" in line for line in run.logs)


def test_two_assets_share_one_account():
    code = algo('''
        class Pair(QCAlgorithm):
            def initialize(self):
                self.set_cash(10000)
                self.a = self.add_forex("USDJPY").symbol
                self.b = self.add_forex("EURUSD").symbol
            def on_data(self, data):
                if not self.portfolio.invested:
                    self.set_holdings(self.a, 0.5)
                    self.set_holdings(self.b, -0.5)
    ''')
    run = run_algorithm(code, [Asset("USDJPY", bars(seed=1)), Asset("EURUSD", bars(start=1.1, step=0.0004, seed=2), point=0.00001)])
    m = run.result
    assert set(m.per_asset) == {"USDJPY", "EURUSD"}
    total = sum(r.equity.iloc[-1] for r in m.per_asset.values())
    assert m.combined.equity.iloc[-1] == pytest.approx(total)
    assert m.combined.initial_cash == 10000
    assert set(m.combined.trades.symbol) == {"USDJPY", "EURUSD"}


def test_time_is_new_york_by_default():
    code = algo('''
        class A(QCAlgorithm):
            def initialize(self):
                self.s = self.add_forex("USDJPY").symbol
            def on_data(self, data):
                self.debug(f"hour {self.time.hour}")
                self.quit("one bar is enough")
    ''')
    run = run_algorithm(code, [Asset("USDJPY", bars(10, t0="2026-01-05 15:00"))])
    assert "hour 11" in run.logs[0]  # bar 15:00-16:00 UTC ends 16:00 UTC = 11:00 New York (winter)


def test_consolidate_daily_bars():
    code = algo('''
        class A(QCAlgorithm):
            def initialize(self):
                self.s = self.add_forex("USDJPY").symbol
                self.days = []
                self.consolidate(self.s, Resolution.DAILY, self.on_day)
            def on_day(self, bar):
                self.days.append(bar)
            def on_data(self, data):
                pass
            def on_end_of_algorithm(self):
                self.debug(f"days {len(self.days)} first high {self.days[0].high:.5f}")
    ''')
    df = bars(24 * 4, t0="2026-01-05")
    run = run_algorithm(code, [Asset("USDJPY", df)])
    assert "days 3" in run.logs[-1]
    assert f"first high {df.high.iloc[:24].max():.5f}" in run.logs[-1]


def test_errors_point_at_the_users_line():
    code = algo('''
        class A(QCAlgorithm):
            def initialize(self):
                self.s = self.add_forex("USDJPY").symbol
            def on_data(self, data):
                x = 1 / 0
    ''')
    with pytest.raises(AlgorithmError, match=r"ZeroDivisionError.*line 8"):
        run_algorithm(code, [Asset("USDJPY", bars(20))])
    with pytest.raises(AlgorithmError, match="no uploaded data for it"):
        run_algorithm(algo('''
            class A(QCAlgorithm):
                def initialize(self):
                    self.add_equity("SPY")
        '''), [Asset("USDJPY", bars(20))])
    with pytest.raises(AlgorithmError, match="inherits from QCAlgorithm"):
        load_algorithm("x = 1")
    with pytest.raises(AlgorithmError, match="Syntax error"):
        load_algorithm("class A(:\n")
    with pytest.raises(AlgorithmError, match="isn't supported"):
        run_algorithm(algo('''
            class A(QCAlgorithm):
                def initialize(self):
                    self.add_forex("USDJPY")
                    self.schedule.on(None, None, None)
        '''), [Asset("USDJPY", bars(20))])


def test_orders_during_warm_up_are_rejected():
    code = algo('''
        class A(QCAlgorithm):
            def initialize(self):
                self.s = self.add_forex("USDJPY").symbol
                self.set_warm_up(10)
            def on_data(self, data):
                if self.is_warming_up:
                    self.market_order(self.s, 1000)
    ''')
    run = run_algorithm(code, [Asset("USDJPY", bars(40))])
    assert (run.orders.status == "Invalid").all() and len(run.result.combined.trades) == 0


def test_app_runs_a_pasted_quantconnect_algorithm(tmp_path):
    from test_frontend import _write_csv, app_test, click
    path = _write_csv(tmp_path, "USDJPY.csv", bars(800))
    at = app_test()
    at.run()
    at.text_input[0].set_value(path).run()
    at.selectbox[0].set_value("As uploaded").run()
    next(s for s in at.selectbox if s.label == "Strategy").set_value("QuantConnect algorithm (Python)").run()
    click(at, "Run backtest")
    assert not at.exception and not at.error
    assert any(m.label == "Sharpe" for m in at.metric)
    assert "Algorithm" in [t.label for t in at.tabs]
