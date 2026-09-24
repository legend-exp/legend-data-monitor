"""Run means are recomputed from the complete run, independent of chunking."""

import numpy as np
import pandas as pd
import pytest

from legend_data_monitor.save_data import recompute_run_means


def _v1(tmp_path):
    path = str(tmp_path / "l200-p22-r000-phy-geds.hdf")
    idx = pd.date_range("2026-07-01", periods=100, freq="1min", tz="UTC")
    # a drifting baseline: 100 at the start of the run, rising by 1 per minute
    absolute = pd.DataFrame({1104000: np.arange(100, 200, dtype="float32")}, index=idx)
    absolute.index.name = "datetime"
    absolute.to_hdf(path, key="IsPulser_Baseline", mode="a")
    # what a chunked append leaves behind: a mean from some later chunk
    stale = pd.DataFrame({1104000: [150.0]}, index=idx[50:51], dtype="float32")
    stale.to_hdf(path, key="IsPulser_Baseline_mean", mode="a")
    ((absolute / 150.0 - 1) * 100).to_hdf(path, key="IsPulser_Baseline_var", mode="a")
    return path, absolute


def test_mean_is_the_first_tenth_of_the_run(tmp_path):
    path, absolute = _v1(tmp_path)
    assert recompute_run_means(path) == 1
    mean = pd.read_hdf(path, key="IsPulser_Baseline_mean")
    # span is 99 min, first 10% = t < 9.9 min -> the first 10 samples, mean 104.5
    assert float(mean[1104000].iloc[0]) == pytest.approx(104.5)
    assert mean.index[0] == absolute.index[0]
    var = pd.read_hdf(path, key="IsPulser_Baseline_var")
    np.testing.assert_allclose(
        var[1104000].to_numpy(),
        (absolute[1104000].to_numpy() / 104.5 - 1) * 100,
        rtol=1e-5,
    )


def test_mean_without_absolute_or_variation_keys(tmp_path):
    path, _ = _v1(tmp_path)
    pd.DataFrame({1: [1.0]}).to_hdf(path, key="IsPulser_Orphan_mean", mode="a")
    assert recompute_run_means(path) == 1  # orphan mean left alone
    assert recompute_run_means(path) == 1  # idempotent
