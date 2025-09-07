import numpy as np
import pandas as pd

from legend_data_monitor.monitoring import resample_series


def test_resample_series_with_gaps():
    idx = pd.to_datetime(
        [
            "2023-08-28 12:00:00",
            "2023-08-28 12:00:01",
            "2023-08-28 12:00:04",  # gap :02-03
            "2023-08-28 12:00:05",
            "2023-08-28 12:00:06",
            "2023-08-28 12:00:07",
            "2023-08-28 12:00:08",
            "2023-08-28 12:00:09",
            "2023-08-28 12:00:10",
        ]
    )
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9], index=idx)
    resampling_time = "2s"

    mask = s.resample(resampling_time).count() > 0
    mean_resampled, std_resampled = resample_series(s, resampling_time, mask)

    expected_index = pd.date_range(
        "2023-08-28 12:00:00", periods=6, freq="2s"
    ).tz_localize("UTC")
    expected_mean = pd.Series([1.5, np.nan, 3.5, 5.5, 7.5, 9.0], index=expected_index)
    expected_std = pd.Series(
        [
            np.std([1, 2], ddof=1),
            np.nan,
            np.std([3, 4], ddof=1),
            np.std([5, 6], ddof=1),
            np.std([7, 8], ddof=1),
            np.nan,
        ],
        index=expected_index,
    )

    pd.testing.assert_series_equal(mean_resampled, expected_mean)
    pd.testing.assert_series_equal(std_resampled, expected_std)
