"""A small event-driven backtester and the data loader it runs on."""

from .broker import SimulatedBroker
from .data import BarFeed
from .engine import BacktestResult, point_size, run_backtest
from .events import Bar, FillEvent, MarketEvent, OrderEvent, SignalEvent
from .portfolio import Portfolio, Snapshot, Trade
from .metrics import Metric, compute_metrics
from .report import plot_equity, print_metrics, print_report, print_trade_log, save_csv
from .strategy import MovingAverageCrossover, Strategy

__all__ = [
    "BacktestResult", "Bar", "BarFeed", "FillEvent", "MarketEvent", "Metric", "MovingAverageCrossover",
    "OrderEvent", "Portfolio", "SignalEvent", "SimulatedBroker", "Snapshot", "Strategy", "Trade",
    "compute_metrics", "plot_equity", "point_size", "print_metrics", "print_report", "print_trade_log",
    "run_backtest", "save_csv",
]
