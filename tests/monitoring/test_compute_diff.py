import numpy as np

from legend_data_monitor.monitoring import compute_diff


def test_compute_diff_normal_case():
    values = np.array([1.0, 2.0, 3.0])
    init_value = 1.0
    scale = 10.0

    result = compute_diff(values, init_value, scale)
    expected = np.array([0.0, 10.0, 20.0])

    np.testing.assert_allclose(result, expected)


def test_compute_diff_initial_zero():
    values = np.array([1.0, 2.0, 3.0])
    init_value = 0.0
    scale = 10.0

    result = compute_diff(values, init_value, scale)
    assert np.all(np.isnan(result))
