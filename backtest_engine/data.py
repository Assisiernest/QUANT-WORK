"""Bar feed: streams bars from the data loader one at a time, oldest first."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from .data_loader import load_ohlcv, resample_ohlcv

from .events import Bar


class BarFeed:
    """Yields Bars in time order, so nothing downstream can look ahead."""

    def __init__(self, df: pd.DataFrame):
        missing = {"open", "high", "low", "close"} - set(df.columns)
        if missing:
            raise ValueError(f"Bar data missing columns: {sorted(missing)}")
        if not df.index.is_monotonic_increasing:
            raise ValueError("Bar data must be sorted oldest first")
        self.df = df

    @classmethod
    def from_csv(cls, path: str | Path, tz: str = "UTC", timeframe: str | None = None) -> "BarFeed":
        """Load a CSV with load_ohlcv; timeframe (e.g. "1h") resamples on the New York day."""
        df = load_ohlcv(path, tz=tz)
        if timeframe:
            df = resample_ohlcv(df, timeframe)
        return cls(df)

    def __len__(self) -> int:
        return len(self.df)

    def __iter__(self) -> Iterator[Bar]:
        has_volume = "volume" in self.df.columns
        has_spread = "spread" in self.df.columns
        for row in self.df.itertuples():
            yield Bar(
                time=row.Index,
                open=row.open,
                high=row.high,
                low=row.low,
                close=row.close,
                volume=row.volume if has_volume else 0.0,
                spread=row.spread if has_spread else 0.0,
            )
