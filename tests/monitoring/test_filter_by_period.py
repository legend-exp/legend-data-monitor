import pandas as pd

from legend_data_monitor.monitoring import filter_by_period

IGNORE_KEYS = {
    "p03": {
        "start_keys": ["20230327T145702Z", "20230406T135529Z"],
        "stop_keys": ["20230327T145751Z", "20230406T235540Z"],
    },
    "p04": {
        "start_keys": ["20230424T123443Z", "20230424T185631Z"],
        "stop_keys": ["20230424T185631Z", "20230425T001708Z"],
    },
}


def make_aware(series):
    # ensure a series index is UTC-aware
    if series.index.tz is None:
        return series.tz_localize("UTC")
    return series


def to_aware(ts: str):
    t = pd.to_datetime(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t


# check if a series has no timestamps inside a start-stop range
def check_removed(series, start_str, stop_str):
    start = to_aware(start_str)
    stop = to_aware(stop_str)
    assert not any(
        (series.index >= start) & (series.index <= stop)
    ), f"Timestamps between {start_str} and {stop_str} were not removed"


def test_filter_by_period_removes_ignore_ranges():
    idx = pd.date_range("2023-03-27 14:57:00", "2023-04-06 23:55:40", freq="s")
    s = pd.Series(range(len(idx)), index=idx)
    s = make_aware(s)

    filtered = filter_by_period(s, ["p03", "p04"])

    # p03
    for start, stop in zip(
        IGNORE_KEYS["p03"]["start_keys"], IGNORE_KEYS["p03"]["stop_keys"]
    ):
        check_removed(filtered, start, stop)
    # p04
    for start, stop in zip(
        IGNORE_KEYS["p04"]["start_keys"], IGNORE_KEYS["p04"]["stop_keys"]
    ):
        check_removed(filtered, start, stop)


def test_filter_by_period_string():
    # test with period being a string
    idx = pd.date_range("2023-03-27 14:57:00", "2023-03-27 15:00:00", freq="s")
    s = pd.Series(range(len(idx)), index=idx)
    s = make_aware(s)

    filtered = filter_by_period(s, "p03")

    for start, stop in zip(
        IGNORE_KEYS["p03"]["start_keys"], IGNORE_KEYS["p03"]["stop_keys"]
    ):
        check_removed(filtered, start, stop)


def test_filter_by_period_empty_result():
    # all data are filtered out
    idx = pd.date_range("2023-03-27T14:57:30Z", "2023-03-27T14:57:40Z", freq="s")
    s = pd.Series(range(len(idx)), index=idx)
    s = make_aware(s)

    filtered = filter_by_period(s, "p03")
    assert len(filtered) == 0


def test_filter_by_period_no_filtering():
    # no filtering
    idx = pd.date_range("2023-01-01 00:00:00", "2023-01-01 00:01:00", freq="s")
    s = pd.Series(range(len(idx)), index=idx)
    s = make_aware(s)

    original_length = len(s)
    filtered = filter_by_period(s, "p03")

    # unchanged since no data falls in ignore ranges
    assert len(filtered) == original_length
    pd.testing.assert_series_equal(filtered, s)
