# region imports
from AlgorithmImports import *
# endregion
from collections import deque
from datetime import datetime, timedelta

import pytz

NEW_YORK = pytz.timezone("America/New_York")


class RangeBreakoutFvgBot(QCAlgorithm):
    """Range Breakout + FVG Bot (QuantConnect port of range-breakout-fvg-bot.pine).

    You set the range with parameters (start/end time and, optionally, the high and
    low prices). After the range ends, the bot waits for a candle to break fully out
    of it (the whole candle, wicks included, beyond the line). On an upside breakout
    it looks for a bullish fair value gap (FVG) formed after the breakout and places a
    buy stop order at the midpoint of the candle that completes the gap; a downside
    breakout does the opposite. Stops sit beyond the FVG, targets are 3R, the stop
    moves to break-even at 1R, size risks a fixed % of equity per trade, and no new
    orders are placed around high-impact news.

    All times are UTC. Every parameter below can be changed in the project's
    Parameters panel without editing the code.
    """

    def initialize(self):
        # ---------- Market & backtest ----------
        self.pair = self._str("pair", "EURUSD")
        bar_minutes = self._int("bar_minutes", 15)
        self.set_time_zone(TimeZones.UTC)
        self.set_start_date(self._dt("backtest_start", "2025-01-01 00:00"))
        self.set_end_date(self._dt("backtest_end", "2026-10-01 00:00"))
        self.set_cash(self._float("cash", 10000))
        self.set_brokerage_model(BrokerageName.OANDA_BROKERAGE, AccountType.MARGIN)

        # ---------- Range ----------
        # range_mode "daily": every day the range is the high/low between
        # daily_range_start and daily_range_end (default: Asian session), and
        # breakouts are traded until daily_stop. Use this for long backtests.
        # range_mode "fixed": one range from range_start to range_end, like drawing
        # it on the chart. range_high / range_low = 0 means "auto": use the highest
        # high and lowest low of the candles between the two times.
        self.range_mode = self._str("range_mode", "daily").lower()
        self.daily_start = self._time("daily_range_start", "00:00")
        self.daily_end = self._time("daily_range_end", "07:00")
        self.daily_stop = self._time("daily_stop", "20:00")
        self.range_start = self._dt("range_start", "2026-10-27 00:00")
        self.range_end = self._dt("range_end", "2026-10-28 00:00")
        self.manual_high = self._float("range_high", 0)
        self.manual_low = self._float("range_low", 0)
        self.range_day = None

        # ---------- Breakout ----------
        self.reset_inside = self._bool("reset_inside", True)
        self.max_trades = self._int("max_trades", 2)

        # ---------- Fair value gap ----------
        self.brk_in_fvg = self._bool("breakout_candle_in_fvg", True)
        self.fvg_outside = self._bool("fvg_outside_range", True)
        self.min_fvg_atr = self._float("min_fvg_atr", 0.1)
        self.expiry_bars = self._int("expiry_bars", 20)

        # ---------- Exits & risk ----------
        self.stop_mode = self._str("stop_mode", "FVG first candle")  # or "Far edge of FVG", "Range line"
        self.sl_buf_atr = self._float("stop_buffer_atr", 0.1)
        self.rr = self._float("reward_risk", 3.0)
        self.use_be = self._bool("use_break_even", True)
        self.be_at_r = self._float("break_even_at_r", 1.0)
        self.risk_pct = self._float("risk_pct", 1.0)
        self.lot_step = self._int("lot_step", 1000)

        # ---------- Session ----------
        self.use_session = self._bool("use_session", False)
        self.sess_start = self._time("session_start", "07:00")
        self.sess_end = self._time("session_end", "17:00")

        # ---------- News filter ----------
        self.use_news = self._bool("use_news", True)
        self.news_before = timedelta(minutes=self._int("news_minutes_before", 30))
        self.news_after = timedelta(minutes=self._int("news_minutes_after", 30))
        self.auto_nfp = self._bool("auto_nfp", True)
        self.cancel_news = self._bool("cancel_on_news", True)
        news_tz = pytz.timezone(self._str("news_tz", "UTC"))
        self.news_times = []
        # Comma-separated "YYYY-MM-DD HH:MM" list, e.g. "2026-11-12 13:30, 2026-11-18 19:00"
        for item in self._str("news_times", "").split(","):
            item = item.strip()
            if item:
                local = news_tz.localize(datetime.strptime(item, "%Y-%m-%d %H:%M"))
                self.news_times.append(local.astimezone(pytz.utc).replace(tzinfo=None))

        # ---------- Alerts ----------
        # Alerts always go to the log. In live trading they can also be emailed.
        self.notify_email = self._str("notify_email", "")

        # ---------- Data & indicators ----------
        self.symbol = self.add_forex(self.pair, Resolution.MINUTE, Market.OANDA).symbol
        self.digits = 3 if "JPY" in self.pair else 5
        self.atr = AverageTrueRange(self._int("atr_length", 14), MovingAverageType.WILDERS)
        consolidator = QuoteBarConsolidator(timedelta(minutes=bar_minutes))
        self.subscription_manager.add_consolidator(self.symbol, consolidator)
        # Register the ATR first so it is updated before on_bar runs
        self.register_indicator(self.symbol, self.atr, consolidator)
        consolidator.data_consolidated += self.on_bar
        self.set_warm_up(timedelta(days=5))

        # ---------- State ----------
        self.bars = deque(maxlen=3)
        self.bar_index = 0
        self.auto_hi = None
        self.auto_lo = None
        self.state = 0          # 1 = broke up, hunting bullish FVG; -1 = broke down, hunting bearish FVG
        self.brk_bar = None
        self.trades_taken = 0
        self.prev_news_near = False

        self.pending = False    # a setup exists (order may or may not be placed yet)
        self.placed = False     # False = waiting for price to be on the right side of the midpoint
        self.pend_dir = 0
        self.pend_bar = 0
        self.pend_units = 0
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.fvg_top = None
        self.fvg_bot = None
        self.risk_dist = None

        self.entry_ticket = None
        self.sl_ticket = None
        self.tp_ticket = None
        self.position_dir = 0
        self.fill_price = None
        self.filled_units = 0
        self.be_done = False
        self.close_ticket = None  # market order closing an unwanted/unprotected position
        self.entry_ids = set()    # every entry order we have placed

    # ---------- Main logic, once per consolidated bar ----------
    def on_bar(self, sender, bar):
        self.bars.append(bar)
        self.bar_index += 1
        daily = self.range_mode == "daily"

        # In daily mode a new range starts every day at daily_range_start
        if daily and bar.time.date() != self.range_day and bar.time.time() >= self.daily_start:
            self.start_new_day(bar.time.date())

        if bar.time >= self.range_start and bar.end_time <= self.range_end:
            self.auto_hi = bar.high if self.auto_hi is None else max(self.auto_hi, bar.high)
            self.auto_lo = bar.low if self.auto_lo is None else min(self.auto_lo, bar.low)

        if self.is_warming_up or not self.atr.is_ready or len(self.bars) < 3:
            return

        manual = not daily and self.manual_high > 0 and self.manual_low > 0
        hi = max(self.manual_high, self.manual_low) if manual else self.auto_hi
        lo = min(self.manual_high, self.manual_low) if manual else self.auto_lo
        before_stop = not daily or (bar.time.date() == self.range_day and bar.time.time() < self.daily_stop)
        watching = bar.time >= self.range_end and before_stop and hi is not None and lo is not None
        atr = self.atr.current.value
        c1, c2, c3 = self.bars

        # The news window is checked against this bar and the next one, since an
        # order placed at this bar's close is live during the next bar.
        news_near = self.is_news_near(bar.time, bar.end_time + (bar.end_time - bar.time))
        if news_near and not self.prev_news_near:
            total = int((self.news_before + self.news_after).total_seconds() // 60)
            self.alert(f"HIGH-IMPACT NEWS soon on {self.pair} - no new trades for the next {total} minutes"
                       + (", unfilled orders cancelled" if self.cancel_news else ""))
        self.prev_news_near = news_near

        # 1) Pending setup: should it be cancelled? (fills are handled in on_order_events)
        do_cancel, why = False, ""
        if self.pending:
            long = self.pend_dir == 1
            if self.bar_index - self.pend_bar >= self.expiry_bars:
                do_cancel, why = True, "not filled in time"
            elif (bar.close < self.fvg_bot) if long else (bar.close > self.fvg_top):
                do_cancel, why = True, "price closed through the FVG"
            elif not self.placed and ((bar.high >= self.tp_price) if long else (bar.low <= self.tp_price)):
                do_cancel, why = True, "price reached the target before the stop order could be placed"
            elif news_near and self.cancel_news:
                do_cancel, why = True, "high-impact news nearby"
            elif not before_stop:
                do_cancel, why = True, "trading window for the day is over"

        # 2) Breakouts and resets (the whole candle, wicks included, must be beyond the line)
        if watching:
            if self.reset_inside and ((self.state == 1 and bar.close < hi) or (self.state == -1 and bar.close > lo)):
                if self.pending and self.pend_dir == self.state:
                    do_cancel, why = True, "price closed back inside the range"
                self.state = 0

            if bar.low > hi and self.state != 1 and self.trades_taken < self.max_trades:
                if self.pending and self.pend_dir == -1:
                    do_cancel, why = True, "opposite breakout"
                self.state = 1
                self.brk_bar = self.bar_index
                self.alert(f"BULLISH BREAKOUT {self.pair} above {self.fmt(hi)} - watching for a bullish FVG")

            if bar.high < lo and self.state != -1 and self.trades_taken < self.max_trades:
                if self.pending and self.pend_dir == 1:
                    do_cancel, why = True, "opposite breakout"
                self.state = -1
                self.brk_bar = self.bar_index
                self.alert(f"BEARISH BREAKOUT {self.pair} below {self.fmt(lo)} - watching for a bearish FVG")

        if do_cancel and self.pending:
            self.cancel_setup(why)

        # Safety net: never hold a position without a working stop loss. Only while
        # the market is open, since OANDA rejects orders queued for the next open.
        holding = self.portfolio[self.symbol].quantity
        if holding != 0 and not self.is_protected() and not self.is_closing() and self.is_market_open(self.symbol):
            self.transactions.cancel_open_orders(self.symbol)
            self.close_ticket = self.market_order(self.symbol, -holding, asynchronous=True, tag="Unprotected position")
            self.alert(f"WARNING {self.pair} - position had no working stop loss, closed at market")
            self.sl_ticket = None
            self.tp_ticket = None
            self.entry_ticket = None
            self.pending = False
            self.placed = False
            self.position_dir = 0

        # 3) Look for an FVG formed after the breakout
        flat = not self.portfolio[self.symbol].invested and self.sl_ticket is None
        fvg_after_brk = self.brk_bar is not None and (
            self.bar_index - 1 >= self.brk_bar if self.brk_in_fvg else self.bar_index - 2 > self.brk_bar)
        can_trade = (watching and not self.pending and flat and self.trades_taken < self.max_trades
                     and self.in_session(bar.time) and not news_near and fvg_after_brk)

        if can_trade and self.state == 1 and c3.low > c1.high:
            top, bot = c3.low, c1.high
            if top - bot >= atr * self.min_fvg_atr and (not self.fvg_outside or bot >= hi):
                stop_ref = {"FVG first candle": c1.low, "Far edge of FVG": bot, "Range line": hi}[self.stop_mode]
                self.new_setup(1, (c3.high + c3.low) / 2, stop_ref - atr * self.sl_buf_atr, top, bot, bar.close)

        elif can_trade and self.state == -1 and c3.high < c1.low:
            top, bot = c1.low, c3.high
            if top - bot >= atr * self.min_fvg_atr and (not self.fvg_outside or top <= lo):
                stop_ref = {"FVG first candle": c1.high, "Far edge of FVG": top, "Range line": lo}[self.stop_mode]
                self.new_setup(-1, (c3.high + c3.low) / 2, stop_ref + atr * self.sl_buf_atr, top, bot, bar.close)

        # 3b) Place the stop order at the midpoint. A buy stop must sit above the
        # current price (a sell stop below it), so if price is on the wrong side we
        # wait for a candle to close on the right side first.
        if self.pending and not self.placed and not news_near and (
                bar.close < self.entry_price if self.pend_dir == 1 else bar.close > self.entry_price):
            self.entry_ticket = self.stop_market_order(self.symbol, self.pend_units * self.pend_dir,
                                                       round(self.entry_price, self.digits), tag="Entry")
            self.entry_ids.add(self.entry_ticket.order_id)
            self.placed = True
            side = "BUY STOP" if self.pend_dir == 1 else "SELL STOP"
            self.alert(f"{side} {self.pair} @ {self.fmt(self.entry_price)} | SL {self.fmt(self.sl_price)} | "
                       f"TP {self.fmt(self.tp_price)} | size {self.pend_units} units")

        # 4) Break-even: once price has moved be_at_r x risk in our favour, move the stop to the fill price
        if self.use_be and not self.be_done and self.sl_ticket is not None and self.risk_dist:
            long = self.position_dir == 1
            trigger = self.entry_price + self.risk_dist * self.be_at_r * self.position_dir
            if (bar.high >= trigger) if long else (bar.low <= trigger):
                self.be_done = True
                fields = UpdateOrderFields()
                fields.stop_price = round(self.fill_price, self.digits)
                self.sl_ticket.update(fields)
                self.alert(f"MOVE STOP TO BREAK-EVEN on {self.pair} @ {self.fmt(self.fill_price)}")

    def start_new_day(self, day):
        # Open positions keep their SL/TP; only unfilled setups are dropped
        if self.pending:
            self.cancel_setup("new day's range started")
        self.range_day = day
        self.range_start = datetime.combine(day, self.daily_start)
        self.range_end = datetime.combine(day, self.daily_end)
        self.auto_hi = None
        self.auto_lo = None
        self.state = 0
        self.brk_bar = None
        self.trades_taken = 0

    # ---------- Orders ----------
    def new_setup(self, direction, entry, sl, top, bot, close):
        dist = (entry - sl) * direction
        if dist <= 0:
            return
        self.pending = True
        self.placed = False
        self.pend_dir = direction
        self.pend_bar = self.bar_index
        self.pend_units = self.calc_units(dist)
        self.entry_price = entry
        self.sl_price = sl
        self.tp_price = entry + dist * self.rr * direction
        self.fvg_top = top
        self.fvg_bot = bot
        self.risk_dist = dist
        self.be_done = False
        if direction == 1 and close >= entry:
            self.alert(f"BUY SETUP {self.pair} - waiting for a close below {self.fmt(entry)} to place a BUY STOP there")
        elif direction == -1 and close <= entry:
            self.alert(f"SELL SETUP {self.pair} - waiting for a close above {self.fmt(entry)} to place a SELL STOP there")

    def cancel_setup(self, why):
        what = "BUY" if self.pend_dir == 1 else "SELL"
        if self.placed and self.entry_ticket is not None:
            self.entry_ticket.cancel(why)
            self.alert(f"CANCEL {what} STOP order on {self.pair} - {why}")
        else:
            self.alert(f"CANCEL {what} setup on {self.pair} (no order was placed) - {why}")
        self.pending = False
        self.placed = False
        self.entry_ticket = None

    def on_order_event(self, order_event):
        if order_event.status != OrderStatus.FILLED:
            return
        oid = order_event.order_id

        # An entry order we had already cancelled filled anyway (the cancel and the
        # fill happened at the same moment): we don't want this trade, close it.
        if oid in self.entry_ids and (self.entry_ticket is None or oid != self.entry_ticket.order_id):
            self.close_ticket = self.market_order(self.symbol, -order_event.fill_quantity, asynchronous=True,
                                                  tag="Close cancelled entry")
            self.alert(f"Cancelled entry on {self.pair} filled anyway - closed at market")
            return

        if self.entry_ticket is not None and oid == self.entry_ticket.order_id:
            self.pending = False
            self.placed = False
            self.trades_taken += 1
            self.position_dir = self.pend_dir
            self.fill_price = order_event.fill_price
            self.filled_units = abs(order_event.fill_quantity)
            exit_qty = -order_event.fill_quantity
            self.sl_ticket = self.stop_market_order(self.symbol, exit_qty, round(self.sl_price, self.digits), tag="Stop loss")
            self.tp_ticket = self.limit_order(self.symbol, exit_qty, round(self.tp_price, self.digits), tag="Take profit")
            side = "BUY" if self.position_dir == 1 else "SELL"
            self.alert(f"FILLED {side} {self.pair} @ {self.fmt(self.fill_price)} | SL {self.fmt(self.sl_price)} | "
                       f"TP {self.fmt(self.tp_price)}")

        elif self.sl_ticket is not None and oid == self.sl_ticket.order_id:
            self.tp_ticket.cancel("Stop loss filled")
            self.trade_closed("BREAK-EVEN stop hit" if self.be_done else "STOP LOSS hit", order_event.fill_price)

        elif self.tp_ticket is not None and oid == self.tp_ticket.order_id:
            self.sl_ticket.cancel("Take profit filled")
            self.trade_closed("TAKE PROFIT hit", order_event.fill_price)

    def trade_closed(self, reason, exit_price):
        conv = self.securities[self.symbol].quote_currency.conversion_rate
        pnl = (exit_price - self.fill_price) * self.filled_units * self.position_dir * conv
        self.alert(f"TRADE CLOSED {self.pair} - {reason} @ {self.fmt(exit_price)} | P&L {pnl:.2f} "
                   f"{self.account_currency} | equity {self.portfolio.total_portfolio_value:.2f}")
        self.sl_ticket = None
        self.tp_ticket = None
        self.entry_ticket = None
        self.position_dir = 0

    # ---------- Helpers ----------
    def is_protected(self):
        t = self.sl_ticket
        return t is not None and t.status not in (OrderStatus.INVALID, OrderStatus.CANCELED, OrderStatus.FILLED)

    def is_closing(self):
        t = self.close_ticket
        return t is not None and t.status not in (OrderStatus.INVALID, OrderStatus.CANCELED, OrderStatus.FILLED)

    def calc_units(self, dist):
        # Units such that hitting the stop loses risk_pct of equity, converted from the
        # pair's quote currency (e.g. JPY for USDJPY) into the account currency.
        conv = self.securities[self.symbol].quote_currency.conversion_rate
        raw = self.portfolio.total_portfolio_value * self.risk_pct / 100 / (dist * conv)
        return max(self.lot_step, int(raw // self.lot_step) * self.lot_step)

    def is_news_near(self, start, end):
        if not self.use_news:
            return False
        events = list(self.news_times)
        if self.auto_nfp:
            # US jobs report: first Friday of the month, 8:30 New York time
            for t in (start, end):
                ny = pytz.utc.localize(t).astimezone(NEW_YORK)
                if ny.weekday() == 4 and ny.day <= 7:
                    nfp = NEW_YORK.localize(datetime(ny.year, ny.month, ny.day, 8, 30))
                    events.append(nfp.astimezone(pytz.utc).replace(tzinfo=None))
        return any(end >= ev - self.news_before and start <= ev + self.news_after for ev in events)

    def in_session(self, t):
        return not self.use_session or self.sess_start <= t.time() < self.sess_end

    def fmt(self, price):
        return f"{price:.{self.digits}f}"

    def alert(self, msg):
        self.log(msg)
        if self.live_mode and self.notify_email:
            self.notify.email(self.notify_email, "Range Breakout + FVG Bot", msg)

    def on_end_of_algorithm(self):
        self.log(f"Finished: {self.trades_taken} trades, final equity {self.portfolio.total_portfolio_value:.2f}")

    # ---------- Parameters ----------
    def _str(self, name, default):
        value = self.get_parameter(name)
        return value if value not in (None, "") else default

    def _int(self, name, default):
        return int(self._str(name, str(default)))

    def _float(self, name, default):
        return float(self._str(name, str(default)))

    def _bool(self, name, default):
        return self._str(name, str(default)).strip().lower() in ("true", "1", "yes")

    def _dt(self, name, default):
        return datetime.strptime(self._str(name, default), "%Y-%m-%d %H:%M")

    def _time(self, name, default):
        return datetime.strptime(self._str(name, default), "%H:%M").time()
