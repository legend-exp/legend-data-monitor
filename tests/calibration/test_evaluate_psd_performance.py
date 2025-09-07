import numpy as np

from legend_data_monitor.calibration import evaluate_psd_performance


def test_insufficient_data_all_nan():
    mean_vals = [np.nan, np.nan]
    sigma_vals = [np.nan, np.nan]
    run_labels = ["r001", "r002"]

    result = evaluate_psd_performance(mean_vals, sigma_vals, run_labels, "r002", "DET1")
    assert result["status"] is None
    assert result["slow_shift_fail_runs"] == []
    assert result["sudden_shift_fail_runs"] == []


def test_no_failures():
    mean_vals = [1.0, 1.02, 1.03]
    sigma_vals = [0.1, 0.1, 0.1]
    run_labels = ["r001", "r002", "r003"]

    result = evaluate_psd_performance(mean_vals, sigma_vals, run_labels, "r003", "DET1")
    assert result["status"] is True
    assert result["slow_shift_fail_runs"] == []
    assert result["sudden_shift_fail_runs"] == []


def test_slow_shift_failure():
    mean_vals = [1.0, 2.0, 1.0]
    sigma_vals = [0.5, 0.5, 0.5]
    run_labels = ["r001", "r002", "r003"]

    result = evaluate_psd_performance(mean_vals, sigma_vals, run_labels, "r002", "DET1")
    assert result["status"] is False
    assert "r002" in result["slow_shift_fail_runs"]


def test_sudden_shift_failure():
    mean_vals = [1.0, 1.5, 1.5]
    sigma_vals = [0.1, 0.1, 0.1]
    run_labels = ["r001", "r002", "r003"]

    result = evaluate_psd_performance(mean_vals, sigma_vals, run_labels, "r002", "DET1")
    assert result["status"] is False
    assert "r001TOr002" in result["sudden_shift_fail_runs"]


def test_sudden_shift_nan_branch():
    mean_vals = [1.0, np.nan]
    sigma_vals = [0.1, 0.1]
    run_labels = ["r001", "r002"]

    result = evaluate_psd_performance(mean_vals, sigma_vals, run_labels, "r002", "DET1")

    assert np.isnan(result["sudden_shifts"][0])
    assert result["sudden_shift_fail_runs"] == []
