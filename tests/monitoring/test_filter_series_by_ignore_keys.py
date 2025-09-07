import pandas as pd
import pytest

from legend_data_monitor.monitoring import filter_series_by_ignore_keys


@pytest.fixture
def sample_series():
    idx = pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC")
    values = [10, 20, 30, 40, 50]
    return pd.Series(values, index=idx)


@pytest.fixture
def ignore_keys():
    return {
        "p1": {
            "start_keys": ["20250102T000000Z", "20250104T000000Z"],
            "stop_keys": ["20250102T235959Z", "20250104T235959Z"],
        }
    }


def test_key_not_in_ignore(sample_series, ignore_keys):
    # series should be unchanged if key not present
    result = filter_series_by_ignore_keys(sample_series, ignore_keys, "pX")
    pd.testing.assert_series_equal(result, sample_series)


def test_single_ignore_range(sample_series, ignore_keys):
    result = filter_series_by_ignore_keys(sample_series, ignore_keys, "p1")
    expected_idx = pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-05"], utc=True)
    expected = pd.Series([10, 30, 50], index=expected_idx)
    pd.testing.assert_series_equal(result, expected)
