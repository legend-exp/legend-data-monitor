import pandas as pd

from legend_data_monitor.monitoring import compute_diff_and_rescaling


def test_compute_diffs_variations():
    s = pd.Series([10, 20, 30])
    reference = 10
    escale = 2.0

    # variations=True
    diff, diff_scaled = compute_diff_and_rescaling(
        s, reference, escale, variations=True
    )

    expected_diff = pd.Series([(10 - 10) / 10, (20 - 10) / 10, (30 - 10) / 10])
    expected_scaled = expected_diff * escale

    pd.testing.assert_series_equal(diff, expected_diff)
    pd.testing.assert_series_equal(diff_scaled, expected_scaled)


def test_compute_diffs_no_variations():
    s = pd.Series([10, 20, 30])
    reference = 10
    escale = 2.0

    # variations=False
    diff, diff_scaled = compute_diff_and_rescaling(
        s, reference, escale, variations=False
    )

    expected_diff = s.copy()
    expected_scaled = s * escale

    pd.testing.assert_series_equal(diff, expected_diff)
    pd.testing.assert_series_equal(diff_scaled, expected_scaled)
