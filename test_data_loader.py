import pandas as pd
import pytest

from backtest_engine.data_loader import DataValidationError, load_ohlcv, resample_ohlcv, trading_day

GOOD = """Date,Open,High,Low,Close,Volume
2024-01-01 00:00,1.10,1.12,1.09,1.11,100
2024-01-01 01:00,1.11,1.13,1.10,1.12,150
2024-01-01 02:00,1.12,1.14,1.11,1.13,200
2024-01-01 03:00,1.13,1.15,1.12,1.14,250
"""


@pytest.fixture
def write(tmp_path):
    def _write(text, name="data.csv"):
        p = tmp_path / name
        p.write_text(text)
        return p
    return _write


def test_loads_and_normalises(write):
    df = load_ohlcv(write(GOOD))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "timestamp"
    assert str(df.index.tz) == "UTC"
    assert len(df) == 4
    assert df["close"].iloc[-1] == pytest.approx(1.14)


def test_sorts_and_dedupes(write):
    lines = GOOD.strip().splitlines()
    shuffled = "\n".join([lines[0], lines[3], lines[1], lines[2], lines[1].replace(",100", ",999")])
    df = load_ohlcv(write(shuffled))
    assert df.index.is_monotonic_increasing
    assert df.index.is_unique
    assert df["volume"].iloc[0] == 999  # last duplicate wins


def test_column_aliases_and_missing_volume(write):
    df = load_ohlcv(write("time,o,h,l,c\n2024-01-01,1,2,0.5,1.5\n"))
    assert df["volume"].iloc[0] == 0.0


def test_epoch_seconds_and_millis(write):
    s = load_ohlcv(write("timestamp,open,high,low,close,volume\n1704067200,1,2,0.5,1.5,1\n", "s.csv"))
    ms = load_ohlcv(write("timestamp,open,high,low,close,volume\n1704067200000,1,2,0.5,1.5,1\n", "ms.csv"))
    assert s.index[0] == ms.index[0] == pd.Timestamp("2024-01-01", tz="UTC")


def test_timezone_conversion(write):
    df = load_ohlcv(write(GOOD), tz="America/New_York")
    assert df.index[0] == pd.Timestamp("2024-01-01 05:00", tz="UTC")


def test_date_filter(write):
    df = load_ohlcv(write(GOOD), start="2024-01-01 01:00", end="2024-01-01 02:00")
    assert len(df) == 2


def test_resample(write):
    df = load_ohlcv(write(GOOD), resample="2h")
    assert len(df) == 2
    first = df.iloc[0]
    assert (first["open"], first["high"], first["low"], first["close"], first["volume"]) == (
        pytest.approx(1.10), pytest.approx(1.13), pytest.approx(1.09), pytest.approx(1.12), 250
    )


def test_resample_drops_gaps():
    idx = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 05:00"], tz="UTC")
    df = pd.DataFrame({"open": [1, 2], "high": [1, 2], "low": [1, 2], "close": [1, 2], "volume": [1, 1]},
                      index=idx, dtype=float)
    assert len(resample_ohlcv(df, "1h")) == 2


def test_drops_rows_with_missing_prices(write):
    df = load_ohlcv(write(GOOD + "2024-01-01 04:00,,,,,\n"))
    assert len(df) == 4


def test_invalid_high_low_raises(write):
    with pytest.raises(DataValidationError, match="high/low"):
        load_ohlcv(write("date,open,high,low,close,volume\n2024-01-01,1.5,1.2,1.0,1.1,1\n"))


def test_non_positive_price_raises(write):
    with pytest.raises(DataValidationError, match="Non-positive"):
        load_ohlcv(write("date,open,high,low,close,volume\n2024-01-01,0,1,0,1,1\n"))


def test_missing_columns_raises(write):
    with pytest.raises(DataValidationError, match="Missing columns"):
        load_ohlcv(write("date,open,high,close\n2024-01-01,1,2,1.5\n"))


def test_missing_timestamp_raises(write):
    with pytest.raises(DataValidationError, match="timestamp"):
        load_ohlcv(write("open,high,low,close\n1,2,0.5,1.5\n"))


def test_bad_timestamp_raises(write):
    with pytest.raises(DataValidationError, match="Unparseable"):
        load_ohlcv(write("date,open,high,low,close\nnot-a-date,1,2,0.5,1.5\n"))


def test_empty_after_filter_raises(write):
    with pytest.raises(DataValidationError, match="No rows"):
        load_ohlcv(write(GOOD), start="2030-01-01")


def test_mt5_export(write):
    text = (
        "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\r\n"
        "2025.12.17\t16:59:00\t155.465\t155.493\t155.465\t155.487\t40\t0\t4\r\n"
        "2025.12.17\t17:00:00\t155.486\t155.502\t155.481\t155.481\t38\t0\t3\r\n"
    )
    df = load_ohlcv(write(text))
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "spread"]
    assert df.index[0] == pd.Timestamp("2025-12-17 16:59", tz="UTC")
    assert df["volume"].tolist() == [40, 38]  # tick volume, since <VOL> is all zero
    assert df["spread"].tolist() == [4, 3]


def test_mt5_server_time_to_utc(write):
    df = load_ohlcv(write(GOOD), tz="Etc/GMT-2")  # UTC+2
    assert df.index[0] == pd.Timestamp("2023-12-31 22:00", tz="UTC")


def test_clocks_go_back_repeated_hour(write):
    # Europe/Athens, 26 Oct 2025: 04:00 local becomes 03:00 again, so 03:30 happens twice.
    text = "date,open,high,low,close\n" + "".join(
        f"2025-10-26 {t},1,2,0.5,1.5\n" for t in ["02:30", "03:00", "03:30", "03:00", "03:30", "04:00"]
    )
    df = load_ohlcv(write(text), tz="Europe/Athens")
    assert len(df) == 6  # both 03:00s and both 03:30s kept as different moments
    assert df.index.is_unique
    assert df.index[1] == pd.Timestamp("2025-10-26 00:00", tz="UTC")  # first 03:00 (UTC+3)
    assert df.index[3] == pd.Timestamp("2025-10-26 01:00", tz="UTC")  # second 03:00 (UTC+2)


def test_clocks_go_forward_missing_hour(write):
    # Europe/Athens, 29 Mar 2026: 03:00 local jumps to 04:00, so 03:30 doesn't exist.
    text = "date,open,high,low,close\n2026-03-29 02:30,1,2,0.5,1.5\n2026-03-29 03:30,1,2,0.5,1.5\n"
    df = load_ohlcv(write(text), tz="Europe/Athens")
    assert df.index[1] == pd.Timestamp("2026-03-29 01:00", tz="UTC")  # moved forward to 04:00 local


def _hourly(start, end):
    idx = pd.date_range(start, end, freq="1h", tz="UTC")
    idx = idx[idx.day_of_week < 5]  # drop Sat/Sun UTC for simplicity
    return pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 1.0}, index=idx)


def test_daily_bars_run_17_to_17_new_york():
    days = resample_ohlcv(_hourly("2026-03-02 00:00", "2026-03-11 23:00"), "1D")
    opens = [f"{ts:%a %m-%d %H:%M}" for ts in days.index]
    # 17:00 New York = 22:00 UTC in winter, 21:00 UTC after US clocks change on 8 Mar.
    assert "Thu 03-05 22:00" in opens
    assert "Mon 03-09 21:00" in opens
    assert all(ts.tz_convert("America/New_York").hour == 17 for ts in days.index[1:])
    assert days.loc["2026-03-04 22:00", "volume"] == 24  # a full Thu bar


def test_4h_bars_follow_new_york_clock_change():
    bars = resample_ohlcv(_hourly("2026-03-02 00:00", "2026-03-11 23:00"), "4h")
    hours = set(bars.index.tz_convert("America/New_York").hour)
    assert hours == {17, 21, 1, 5, 9, 13}


def test_daily_bars_without_ny_close_use_utc_midnight():
    days = resample_ohlcv(_hourly("2026-03-02 00:00", "2026-03-04 23:00"), "1D", ny_close=False)
    assert all(ts.hour == 0 for ts in days.index)


def test_trading_day_labels():
    opens = pd.DatetimeIndex(["2026-03-08 21:00", "2026-03-05 22:00"], tz="UTC")
    # Sunday 17:00 New York opens Monday's bar; Thursday 17:00 opens Friday's.
    assert [str(d) for d in trading_day(opens)] == ["2026-03-09", "2026-03-06"]


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_ohlcv(tmp_path / "nope.csv")
