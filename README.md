# Quant Work

Personal workspace for algorithmic trading: two TradingView (Pine Script v6) forex strategies, a small Python risk calculator, and research notes on QuantConnect.

| File | What it is |
|---|---|
| [range-breakout-fvg-bot.pine](range-breakout-fvg-bot.pine) | Range breakout + fair value gap (FVG) strategy with stop-order entries, break-even and a news filter |
| [ema-trend-bot.pine](ema-trend-bot.pine) | EMA crossover trend strategy with RSI, session and ATR-based exits |
| [TRADE PROFIT-LOSS CALCULATOR.py](TRADE%20PROFIT-LOSS%20CALCULATOR.py) | Command-line calculator: account balance + risk % + result in R → P&L and new balance |
| `Algorithmic Trading Platform - QuantConnect.com.html` (+ `_files/`) | Saved copy of the QuantConnect web page, for reference only |

---

## Loading a Pine strategy in TradingView

1. Open a chart in TradingView and open the **Pine Editor** (bottom panel).
2. Paste the contents of the `.pine` file and click **Add to chart**.
3. Open the strategy's **Settings** (gear icon) to change inputs.
4. Check results in the **Strategy Tester** tab.

Both strategies are written for forex. They start with $10,000 in USD, use 2 ticks of slippage, and size each trade so that hitting the stop loses a fixed % of equity (1% by default). Size is converted from the pair's quote currency (e.g. JPY for USDJPY) into USD and rounded to micro lots (1,000 units).

### Alerts

Both scripts send their messages through `alert()`. To get them, create **one** alert on the strategy and pick the **"alert() function calls only"** condition. Messages include the entry, SL, TP and position size.

---

## Range Breakout + FVG Bot

[range-breakout-fvg-bot.pine](range-breakout-fvg-bot.pine)

### How it trades

1. **Range.** When you add the script, TradingView asks you to click two points on the chart: the first (top-left) sets the range high and start time, the second (bottom-right) sets the range low and end time. In *Auto* mode the bot uses the highest high and lowest low between the two times instead of the prices you clicked.
2. **Breakout.** After the range ends, a breakout needs a candle that is **entirely** outside the range, wicks included (`low > range high` or `high < range low`).
3. **FVG.** After an upside breakout the bot looks for a bullish fair value gap (`low > high[2]`); after a downside breakout, a bearish one. By default the gap has to sit fully outside the range and be at least 0.1 × ATR in size.
4. **Entry.** A **stop order** at the midpoint of the candle that completes the gap. A buy stop has to sit above the current price, so if price is already above the midpoint the bot waits for a candle to close below it before placing the order. A sell stop works the same way in reverse.
5. **Exits.** The stop goes beyond the first FVG candle, plus a 0.1 × ATR buffer. You can change this to the far edge of the FVG or the range line. The target is 3R, and the stop moves to break-even once price reaches 1R.

### When a setup is cancelled

An unfilled setup is cancelled if:
- it hasn't filled after 20 bars,
- price closes through the FVG,
- price reaches the target before the order could be placed,
- price closes back inside the range (if that option is on),
- price breaks out the opposite way,
- high-impact news is near.

The bot takes at most 2 trades per range.

### News filter

- **Manual list:** add one event per line as `YYYY-MM-DD HH:MM` in the *News times* box, and pick the time zone they're written in. Copy high-impact (red-folder) events for your pairs from a calendar such as Forex Factory, e.g. CPI, FOMC or rate decisions.
- **NFP:** turned on by default. It blocks the first Friday of every month at 8:30 New York time automatically.
- By default no new orders are placed from 30 min before to 30 min after each event, and unfilled orders are cancelled.

> **Before using it:** the default range (27–28 Oct 2026, 1.1000 / 1.0900) and the two example news times are placeholders. Click your own range and replace the news list.

### Chart

- Blue box with green and red lines: the range.
- Green or red background: the bot is looking for an FVG after a breakout.
- Grey background: news block.
- Green or red boxes: FVGs that produced a setup.
- Blue, red and green lines: the entry, SL and TP of the pending or open trade.

---

## EMA Trend Bot

[ema-trend-bot.pine](ema-trend-bot.pine)

- **Long:** the 20 EMA crosses above the 50 EMA, price is above the 200 EMA, and RSI(14) is below 70.
- **Short:** the 20 EMA crosses below the 50 EMA, price is below the 200 EMA, and RSI(14) is above 30.
- **Session:** new trades only open 07:00–17:00 UTC (London and New York).
- **Exits:** the stop is 1.5 × ATR(14) and the target is 2R.

You can turn off the trend filter, the session filter, longs or shorts separately in the settings.

---

## Trade Profit-Loss Calculator

```bash
python "TRADE PROFIT-LOSS CALCULATOR.py"
```

It asks for your account balance, the risk % and the trade result in R (e.g. `2` for a 2R win, `-1` for a full stop-out). It prints the amount risked, the P&L, the new balance and whether the trade was a profit, a loss or breakeven.

Example: $10,000 at 1% risk with a 3R win means $100 risked, +$300 profit and a new balance of $10,300.

Requires Python 3 (standard library only).

---

## Backtest Studio (web app)

A browser front end for the backtest engine below: upload data, pick a strategy, run, and read the results. It runs on this computer, and your files never leave it.

```bash
pip install -r requirements.txt
streamlit run app.py
```

In VS Code, press F5 and pick **Backtest Studio (web app)**. It opens at http://localhost:8501.

1. **Data** (sidebar): upload one CSV per asset (or type file paths). MetaTrader, TradingView and Yahoo exports work, and a file with only a close/price column is accepted. Set each asset's symbol, time zone, capital **weight**, point size and lot step. Choose a timeframe (bars are resampled on the 17:00 New York day) and a date range.
2. **Strategy:** SMA, EMA, MACD, RSI, Bollinger, Donchian, buy and hold, or **Custom (Python)**: write or upload a `signals(df, ...)` function returning +1 / -1 / 0 per bar. A look-ahead check warns you if it uses future bars.
3. **Orders & exits**, **capital, sizing and costs:** market/limit/stop entries, stop-loss, take-profit, trailing stop, fixed / % of equity / % risk sizing, slippage and a risk-free rate.
4. **Run backtest.** Tabs: **Overview** (headline numbers, equity vs buy-and-hold, drawdown, price with trades), **Statistics** (QuantConnect-style: Sharpe, Sortino, probabilistic Sharpe, alpha, beta, information ratio, Treynor, tracking error, turnover and more, plus the detailed metrics), **Returns** (monthly heatmap, rolling Sharpe, distributions), **Trades** (filterable log, MAE/MFE), **Assets** (per-asset results and correlation), **Portfolio**, **Optimise** (out-of-sample and walk-forward) and **Export** (CSVs and a full HTML report).

**Accounts:** the app asks you to sign in. The first time it opens, it asks you to create the **admin** account. Admins add other people with the **Users** button under their name. Each person has a username and password, their own **Saved runs** (save any backtest from its Export tab) and their own saved strategies.
- **Passwords** are stored only as salted scrypt hashes in `app_data/users.json`. Five wrong passwords lock an account for 5 minutes.
- **Permissions:** two abilities are off for ordinary users until an admin allows them:
  - **run code** (custom strategies and QuantConnect algorithms run as real Python on the computer hosting the app)
  - **local files** (reading files on that computer by path)
- **Locked out?** On the computer running the app:

  ```bash
  python manage_users.py list
  ```

  ```bash
  python manage_users.py passwd <username>
  ```

  There's also `unlock`, `add`, `role`, `allow` and `delete`.

The `app_data` folder holds the accounts and everyone's saved work, so keep it private. Signing in lasts until the browser tab is reloaded.

**Paste a QuantConnect algorithm:** in **2 · Strategy**, choose **QuantConnect algorithm (Python)** and paste an algorithm written for QuantConnect (or load a `.py`). `from AlgorithmImports import *` works, and so do QuantConnect's newer snake_case names and its older PascalCase ones.
- **How symbols are found:** the symbols the algorithm adds (`add_forex("EURUSD")`, `add_equity("SPY")`...) are matched to the symbols of your uploaded CSVs.
- **What's covered:** `set_holdings`, `liquidate`, market / limit / stop / stop-limit / market-on-open orders, `portfolio`, `securities`, `transactions`, `history`, `consolidate`, warm-up, `plot`, `debug`, `on_order_event`, `RollingWindow` and the indicators SMA, EMA, RSI, Bollinger bands, ATR, MACD, STD, MAX, MIN, ROC and MOM.
- **How it fills, as a QuantConnect backtest does:** market orders fill at the bar's close, at the ask for buys. Limit and stop orders fill when a later bar reaches them. `self.time` is New York time.
- **What's not available yet:** scheduled events, universe selection, options and the algorithm framework. They give a clear error.
- **Results:** the same tabs as any backtest, plus an **Algorithm** tab with the log, every order and anything you `plot`.

**Analyse a QuantConnect result:** switch the sidebar *Mode* to "Analyse a QuantConnect result" and upload the results `.json` from a QuantConnect backtest (the sample `Smooth Blue Lion.json` is included). The app shows QuantConnect's statistics, trade statistics, equity against the benchmark, orders and closed trades, and checks for common problems: orders the broker refused (so a position was never closed), profit that is unrealised on a position still open at the end, closed trades that stop long before the backtest does, and a beta so high the result is mostly leverage.

**Matching QuantConnect's numbers:** the statistics follow QuantConnect's conventions, checked against that sample file. CAGR uses calendar days ÷ 365. Daily returns include weekends as zero days, which for forex lowers volatility by about a fifth (the sidebar switch *QuantConnect-style daily returns* turns it off). Sortino uses the standard deviation of the negative days. Probabilistic Sharpe is the chance the true Sharpe is above 1. Trade statistics count a break-even trade as a loser. Alpha and beta can differ a little because QuantConnect samples equity and the benchmark at different times of day.

**Multiple assets:** the capital is split by weight into one sub-account per asset. The same strategy trades each, and the equity curves are added up on a shared timeline, so the combined drawdown, Sharpe and the rest are portfolio-level. Sub-accounts don't share margin. Currency pairs need USD as one side (no crosses like EURGBP yet); metals, stocks, indices and crypto are treated as USD-priced.

**Sizing across assets:** "% of equity" (the default) works for any asset. "Fixed units" means the same units everywhere, which is 10,000 units of USDJPY but 10,000 ounces of gold, so the app warns when that is far more than the capital. If equity reaches zero the position is closed ("margin call") and trading stops.

### Deploying to Streamlit Community Cloud

1. Push this repository to GitHub (it can stay private).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub and choose **Create app**. Pick this repository, branch `main` and main file `app.py`.
3. Under **Advanced settings**, choose Python **3.11** or newer. In **Secrets**, paste the lines from [.streamlit/secrets.toml.example](.streamlit/secrets.toml.example) with your own admin username and password.
4. Deploy. Sign in with that admin account, then add other people with **Users**.

Things to know about the cloud version:
- **The disk is wiped whenever the app restarts or you push changes.** Accounts other than the admin and everyone's **Saved runs** disappear then (download anything you want to keep from the Export tab). The admin from Secrets is recreated automatically.
- **The open "create the admin account" screen only appears on your own computer (localhost).** On the cloud, the admin must come from Secrets.
- **Sample data:** the app uses `data/USDJPY.csv` when your MT5 folder isn't there, which on the cloud it never is.
- **The "run code" permission matters more on a server:** people with it run Python on Streamlit's machine. Keep it for people you trust.

## Backtest engine (Python)

[backtest_engine/](backtest_engine/) is an event-driven backtester that runs on MetaTrader 5 CSV exports. [data_loader.py](backtest_engine/data_loader.py) reads and cleans the CSV, converts broker server time (UTC+2) to UTC, and resamples bars on the 17:00 New York forex day. The engine feeds the bars one at a time to a strategy (an SMA crossover for now), the MT5 spread is charged on every buy, and the portfolio records the account on every bar.

| Feature | Options | Where |
|---|---|---|
| **Entry orders** | market (next bar's open), limit (better price by N pips), stop (breakout by N pips), optional expiry in bars | `--entry`, `--entry-offset`, `--expiry` · [orders.py](backtest_engine/orders.py) |
| **Exits** | stop-loss, take-profit, trailing stop (pips from the fill). If the stop and the target are both inside one bar, the stop is assumed to hit first. Gaps fill at the open. | `--sl`, `--tp`, `--trail` · [broker.py](backtest_engine/broker.py) |
| **Slippage** | fixed or seeded-random points per fill, always against you; on market, stop, stop-loss and trailing fills (limits and take-profits fill at their price) | `--slippage`, `--slippage-max`, `--seed` |
| **Position sizing** | fixed units; % of equity (notional); **% risk per trade** sized to the stop-loss. Rounded down to micro lots, capped at 30x leverage | `--sizing fixed/equity/risk`, `--units`, `--equity-pct`, `--risk-pct`, `--leverage`, `--lot-step` · [sizing.py](backtest_engine/sizing.py) |
| **Out-of-sample** | search a parameter grid on the first part of the data, then trade the best settings on the rest | `run_walkforward.py --mode oos --split 0.7` · [optimize.py](backtest_engine/optimize.py) |
| **Walk-forward** | re-tune before each test window (rolling or anchored training), then chain the test windows into one out-of-sample result | `run_walkforward.py --windows 4 --train 0.5 [--anchored]` |

With the defaults (market orders, no exits, no slippage, 10,000 units) results are the same as before these features existed.

### Running it in VS Code

Open the **QUANT WORK** folder in VS Code (File → Open Folder). The Python and Jupyter extensions are needed.

| To see | Do this |
|---|---|
| Results, portfolio by day, trades and P&L by month as tables, plus the equity curve | **Run and Debug** (Ctrl+Shift+D) → pick **Backtest: 5min SMA 20/50** → F5. The tables print in the terminal and the chart opens in a tab. |
| Stop-loss / take-profit, 1% risk sizing and 2-point slippage | **Backtest: 5min, SL 20 / TP 40, 1% risk, 2pt slippage** |
| Your own timeframe, SMA lengths, long/short, order type, exits, sizing and slippage | **Backtest: choose settings** → F5, then answer the prompts at the top of the window (0 = off for SL/TP/trailing) |
| Walk-forward or out-of-sample results | **Walk-forward: 5min, 4 rolling windows**, **Out-of-sample: 5min, 70/30 split**, or **Walk-forward / out-of-sample: choose settings**. Or open [walkforward_explorer.ipynb](walkforward_explorer.ipynb) → **Run All**. |
| A chart you can zoom and pan | **Backtest: 5min with zoomable chart window** |
| Everything in one scrollable page | Open [backtest_explorer.ipynb](backtest_explorer.ipynb) → **Run All**. Change the settings in its second cell. |
| The loaded price data | **Load USDJPY data**, or open [usdjpy_explorer.ipynb](usdjpy_explorer.ipynb) |
| Tests | The Testing panel (flask icon), or **Run all tests** |

The terminal report shows:
- **Metrics** in four panels: Returns, Risk (drawdown, Sharpe, Sortino, Calmar), Trades (win rate, profit factor, expectancy, streaks) and Exposure (time in market, holding times, MAE/MFE)
- **Long vs short**
- **Portfolio by day**
- **Trade log:** the last 20 trades by default; use `--trades all` for every trade
- **P&L by month**

Each backtest also saves to `backtest_results/`:
- `*_equity.png`: the equity curve
- `*_metrics.csv`: every metric
- `*_by_side.csv`: the long vs short breakdown
- `*_trade_log.csv`: every trade, with entry/exit, pips, P&L, return, spread paid, MAE/MFE and running balance
- `*_daily.csv`: the account at each trading day's close
- `*_history.csv`: the account on every bar

The CSVs open in Excel. Ctrl+click a path in the terminal to open it.

From a terminal in this folder:

```bash
python run_backtest.py --timeframe 1h --fast 10 --slow 30 --long-only --open
```

```bash
python run_backtest.py --sl 20 --tp 40 --sizing risk --risk-pct 1 --slippage 2
```

```bash
python run_walkforward.py --fast 5,10,20 --slow 50,100,200 --objective sharpe --windows 4 --anchored
```

The walk-forward report shows each window's dates, the parameters it chose, and its in-sample and out-of-sample results. Then it shows the chained out-of-sample metrics, the walk-forward efficiency (out-of-sample return per day ÷ in-sample), the full report and the trade log, and saves a `*_windows.csv`. Out-of-sample mode prints the in-sample parameter search and an in-sample vs out-of-sample comparison, and its chart marks the split. The search tries every fast/slow combination, so a big grid on 1-minute bars takes a while.

The price data comes from `C:\Users\hp\OneDrive\Desktop\MT5\USDJPY.csv` by default; use `--file` to point it at another export. Tables print best with the terminal panel wide or maximised.

---

## Disclaimer

These are personal research and learning tools, not financial advice. Backtest results in TradingView depend on the chart's timeframe, the broker's data feed and the slippage settings, and they do not guarantee live results. Forward-test on a demo account before risking real money.
