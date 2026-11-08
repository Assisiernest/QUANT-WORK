"""Load OHLCV price data from CSV into a clean, validated pandas DataFrame.

Output: DatetimeIndex (UTC, sorted, unique) named "timestamp" and float columns
open, high, low, close, volume (plus spread, if the file has it).

Handles comma/semicolon/tab-separated files, including MetaTrader 4/5 exports
(<DATE> and <TIME> in separate columns, tick volume in <TICKVOL>).
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]
EXTRA = ["spread"]

# The forex trading day ends at 17:00 New York time.
NEW_YORK = "America/New_York"
NY_CLOSE_HOUR = 17

# Common header spellings from brokers / TradingView / MT4 / MT5 / Yahoo exports.
ALIASES = {
    "timestamp": "timestamp", "time": "timestamp", "date": "timestamp",
    "datetime": "timestamp", "date_time": "timestamp", "gmt_time": "timestamp",
    "open": "open", "o": "open",
    "high": "high", "h": "high",
    "low": "low", "l": "low",
    "close": "close", "c": "close", "price": "close", "last": "close", "adj_close": "adj_close",
    "volume": "volume", "vol": "volume", "v": "volume",
    "tick_volume": "tick_volume", "tickvol": "tick_volume",
    "spread": "spread",
}


class DataValidationError(ValueError):
    """Raised when price data is malformed."""


def _sniff_sep(path: Path) -> str:
    with path.open(newline="") as f:
        return _sniff_header(f.readline())


def _sniff_header(header: str) -> str:
    try:
        return csv.Sniffer().sniff(header, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    def key(col: str) -> str:
        return str(col).strip().lower().replace(" ", "_").replace("<", "").replace(">", "")

    keys = {c: key(c) for c in df.columns}
    by_key = {k: c for c, k in keys.items()}
    if "date" in by_key and "time" in by_key:
        # MetaTrader-style split date/time columns.
        combined = df[by_key["date"]].astype(str) + " " + df[by_key["time"]].astype(str)
        df = df.drop(columns=[by_key["date"], by_key["time"]])
        df.insert(0, "timestamp", combined)
        keys = {c: key(c) for c in df.columns}

    df = df.rename(columns={c: ALIASES.get(k, k) for c, k in keys.items()})
    if df.columns.duplicated().any():
        dupes = sorted(set(df.columns[df.columns.duplicated()]))
        raise DataValidationError(f"Ambiguous columns after normalising: {dupes}")
    return df


def _parse_timestamps(values: pd.Series, tz: str) -> pd.DatetimeIndex:
    if pd.api.types.is_numeric_dtype(values):
        # Epoch seconds vs milliseconds.
        unit = "ms" if values.abs().max() > 1e11 else "s"
        idx = pd.to_datetime(values, unit=unit, utc=True)
    else:
        try:
            # Fast path: one format inferred from the first row.
            idx = pd.to_datetime(values, utc=False)
        except (ValueError, TypeError):
            idx = pd.to_datetime(values, errors="coerce", utc=False, format="mixed")
        if idx.isna().any():
            bad = values[idx.isna()].head(3).tolist()
            raise DataValidationError(f"Unparseable timestamps, e.g. {bad}")
        if idx.dt.tz is None:
            idx = _localize(idx, tz)
        idx = idx.dt.tz_convert("UTC")
    return pd.DatetimeIndex(idx, name="timestamp")


def _localize(idx: pd.Series, tz: str) -> pd.Series:
    """Attach a timezone, coping with daylight-saving clock changes.

    When clocks go back, one hour of local times happens twice; pandas works out
    which is which from the order of the rows (the file must be in time order).
    When clocks go forward, one hour never exists; any such times move forward.
    """
    try:
        return idx.dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    except Exception as exc:  # pytz/pandas raise several types here
        raise DataValidationError(
            f"Can't place repeated clock-change times in {tz!r}; is the file in time order? ({exc})"
        ) from exc


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check OHLC consistency; returns df unchanged or raises DataValidationError."""
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing columns: {missing}")
    if df[OHLCV].isna().any().any():
        raise DataValidationError("NaN values in OHLCV data")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise DataValidationError("Non-positive prices found")
    if (df["volume"] < 0).any():
        raise DataValidationError("Negative volume found")
    bad = (df["high"] < df[["open", "close", "low"]].max(axis=1)) | (
        df["low"] > df[["open", "close", "high"]].min(axis=1)
    )
    if bad.any():
        raise DataValidationError(f"{int(bad.sum())} bars where high/low don't bound open/close, "
                                  f"first at {df.index[bad.argmax()]}")
    return df


def load_ohlcv(
    path: str | Path | IO,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    tz: str = "UTC",
    resample: str | None = None,
    dropna: bool = True,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Load an OHLCV CSV file.

    Args:
        path: CSV file path, or an open file / uploaded file (bytes or text).
        start, end: Optional inclusive date bounds (interpreted in UTC if naive).
        tz: Timezone of naive timestamps in the file; output is always UTC.
        resample: Optional pandas offset alias (e.g. "1h", "1D") to aggregate bars.
        dropna: Drop rows with missing OHLC values instead of raising.
        **read_csv_kwargs: Passed through to pandas.read_csv (e.g. sep=";").

    A file with only a close (or price) column is accepted: open, high and low are set
    to the close, so bar-range features (stop-loss inside a bar, MAE/MFE) are approximate.
    """
    if hasattr(path, "read"):
        raw = path.read()
        text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw
        read_csv_kwargs.setdefault("sep", _sniff_header(text.split("\n", 1)[0]))
        source = io.StringIO(text)
    else:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(source)
        read_csv_kwargs.setdefault("sep", _sniff_sep(source))
    df = _normalise_columns(pd.read_csv(source, **read_csv_kwargs))

    if "timestamp" not in df.columns:
        raise DataValidationError("No timestamp/date/time column found")
    if "close" in df.columns and not {"open", "high", "low"} & set(df.columns):
        df["open"] = df["high"] = df["low"] = df["close"]
        df.attrs["close_only"] = True
    if "tick_volume" in df.columns and (
        "volume" not in df.columns or (pd.to_numeric(df["volume"], errors="coerce").fillna(0) == 0).all()
    ):
        # Forex has no real volume; MetaTrader puts zeros in <VOL> and the useful number in <TICKVOL>.
        df["volume"] = df["tick_volume"]
    if "volume" not in df.columns:
        df["volume"] = 0.0
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing columns: {missing}")

    df.index = _parse_timestamps(df.pop("timestamp"), tz)
    columns = OHLCV + [c for c in EXTRA if c in df.columns]
    df = df[columns].apply(pd.to_numeric, errors="coerce").astype(float)

    if dropna:
        df = df.dropna(subset=["open", "high", "low", "close"])
        df["volume"] = df["volume"].fillna(0.0)

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if start is not None:
        df = df[df.index >= _to_utc(start)]
    if end is not None:
        df = df[df.index <= _to_utc(end)]

    if resample:
        df = resample_ohlcv(df, resample)

    if df.empty:
        raise DataValidationError("No rows left after loading/filtering")
    return validate(df)


def resample_ohlcv(df: pd.DataFrame, rule: str, ny_close: bool = True) -> pd.DataFrame:
    """Aggregate bars to a coarser timeframe, dropping empty periods.

    With ny_close (the default), bars are aligned to the forex day, which runs
    17:00 to 17:00 New York time and follows US daylight saving. Daily bars then
    open at 22:00 UTC in winter and 21:00 UTC in summer, with no stray Sunday bar,
    and 4h bars start at 17:00, 21:00, 01:00... New York. The index is each bar's
    open time in UTC; use trading_day() to get the date a daily bar belongs to.
    """
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    if "spread" in df.columns:
        agg["spread"] = "mean"
    offset = pd.tseries.frequencies.to_offset(rule)
    if not (ny_close and isinstance(offset, (pd.offsets.Tick, pd.offsets.Day))):
        # Weekly/monthly rules have their own anchors.
        return df.resample(rule).agg(agg).dropna(subset=["open"])

    # Bin on New York clock time moved forward 7h, so 17:00 New York becomes midnight and
    # every bin follows US daylight saving; then move the bin starts back. (pandas' own
    # `offset` works in fixed UTC hours, so it drifts by an hour when New York changes clocks.)
    shift = pd.Timedelta(hours=24 - NY_CLOSE_HOUR)
    ny = df.tz_convert(NEW_YORK)
    out = ny.set_axis(ny.index.tz_localize(None) + shift).resample(rule).agg(agg)
    out = out.dropna(subset=["open"])
    # Clocks change at 02:00 on a Sunday, when forex is shut; for 24/7 data the repeated
    # autumn hour is merged into one bar and treated as daylight time.
    out.index = (out.index - shift).tz_localize(
        NEW_YORK, ambiguous=np.ones(len(out), dtype=bool), nonexistent="shift_forward"
    ).tz_convert("UTC")
    return out


def trading_day(index: pd.DatetimeIndex) -> pd.Index:
    """Trading date of bars that open at 17:00 New York (Sunday 17:00 -> Monday)."""
    ny = index.tz_convert(NEW_YORK)
    return pd.Index((ny + pd.Timedelta(hours=24 - NY_CLOSE_HOUR)).date, name="trading_day")


def _to_utc(ts: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


# The MT5 export on this computer if it's there (latest data), otherwise the copy in the repo's
# data/ folder (what a deployed app, e.g. on Streamlit Cloud, uses).
_MT5_FILE = Path(r"C:\Users\hp\OneDrive\Desktop\MT5\USDJPY.csv")
_REPO_SAMPLE = Path(__file__).resolve().parent.parent / "data" / "USDJPY.csv"
DEFAULT_FILE = _MT5_FILE if _MT5_FILE.exists() else _REPO_SAMPLE

# The MT5 broker's server clock. Checked from Friday closes in USDJPY.csv (Dec 2025 - Mar 2026):
# UTC+2 both before and after the US clock change, so it doesn't follow New York time.
# If data after 29 Mar shows the Friday close back at 23:59, the broker follows European
# summer time instead: switch this to "Europe/Athens". ("Etc/GMT-2" means UTC+2; the sign is reversed.)
MT5_SERVER_TZ = "Etc/GMT-2"


def _print_bars(console, df: pd.DataFrame, title: str, decimals: int, daily: bool = False) -> None:
    from rich.table import Table

    table = Table(title=title, title_style="bold cyan", header_style="bold", show_lines=False)
    if daily:
        table.add_column("Trading day")
        table.add_column("Opens (UTC)", style="dim")
        labels = [[f"{d:%a %Y-%m-%d}", f"{ts:%m-%d %H:%M}"] for d, ts in zip(trading_day(df.index), df.index)]
    else:
        table.add_column("Time (UTC)", style="dim")
        labels = [[f"{ts:%Y-%m-%d %H:%M}"] for ts in df.index]
    for col in df.columns:
        table.add_column(col.capitalize(), justify="right")
    for label, (_, row) in zip(labels, df.iterrows()):
        cells = [f"{row[c]:,.{decimals}f}" for c in ["open", "high", "low", "close"]]
        cells.append(f"{row['volume']:,.0f}")
        if "spread" in df.columns:
            cells.append(f"{row['spread']:.1f}")
        table.add_row(*label, *cells)
    console.print(table)


def _print_summary(console, df: pd.DataFrame, decimals: int) -> None:
    from rich.table import Table

    stats = df.describe().drop(index="count")
    table = Table(title="Summary statistics", title_style="bold cyan", header_style="bold")
    table.add_column("Stat", style="dim")
    for col in stats.columns:
        table.add_column(col.capitalize(), justify="right")
    for stat, row in stats.iterrows():
        cells = [f"{row[c]:,.{decimals}f}" for c in ["open", "high", "low", "close"]]
        cells += [f"{row[c]:,.1f}" for c in stats.columns[4:]]
        table.add_row(stat, *cells)
    console.print(table)


def main(argv: list[str]) -> None:
    from rich.console import Console
    from rich.panel import Panel

    # Run with no arguments to load the MT5 USDJPY export, or pass a CSV path.
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT_FILE
    console = Console()

    df = load_ohlcv(path, tz=MT5_SERVER_TZ)
    decimals = 3 if df["close"].median() > 20 else 5  # JPY pairs quote to 3 dp, most others to 5
    step = df.index.to_series().diff().mode()
    console.print(Panel.fit(
        f"[bold]{path.name}[/]  (server time {MT5_SERVER_TZ} converted to UTC)\n"
        f"Bars:      {len(df):,}\n"
        f"From:      {df.index[0]:%Y-%m-%d %H:%M}\n"
        f"To:        {df.index[-1]:%Y-%m-%d %H:%M}\n"
        f"Bar size:  {step.iloc[0] if not step.empty else 'n/a'}\n"
        f"Columns:   {', '.join(df.columns)}",
        title="Loaded OK", border_style="green",
    ))
    _print_bars(console, df.head(10), "First 10 bars (1 min)", decimals)
    _print_bars(console, df.tail(10), "Last 10 bars (1 min)", decimals)
    _print_summary(console, df, decimals)
    _print_bars(console, resample_ohlcv(df, "1h").tail(10), "Last 10 bars resampled to 1 hour", decimals)
    _print_bars(console, resample_ohlcv(df, "1D").tail(10),
                "Last 10 daily bars (17:00-17:00 New York)", decimals, daily=True)


if __name__ == "__main__":
    main(sys.argv)
