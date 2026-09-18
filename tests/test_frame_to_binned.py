"""frame_to_binned fills per detector column.

The long-format form of this (repeat the timestamps, tile an object-dtype
detector array over every column, ravel the values) materialises three
n_events x n_detectors arrays at once, which drove the build's memory peak on a
full run. These tests pin that the per-column fill gives identical output.
"""

import numpy as np
import pandas as pd

from legend_data_monitor.contract import schema
from legend_data_monitor.processing import binning

DETS = ["V02160A", "V02160B", "P00574A"]


def _frame(n=500, seed=3, with_gaps=True):
    rng = np.random.default_rng(seed)
    t0 = 1_700_000_000.0
    idx = pd.to_datetime(np.sort(rng.uniform(t0, t0 + 3600, n)), unit="s", utc=True)
    data = rng.normal(100, 5, (n, len(DETS)))
    if with_gaps:
        # detectors are not all live for the whole window
        data[: n // 3, 1] = np.nan
        data[:, 2] = np.nan
        data[n // 2 :, 2] = rng.normal(50, 1, n - n // 2)
    return pd.DataFrame(data, index=idx, columns=DETS)


def _long_format_reference(df, cadence=schema.BASE_CADENCE):
    """Replicate the previous implementation, kept here as the equivalence oracle."""
    ts_all = df.index.tz_convert("UTC") if df.index.tz else df.index.tz_localize("UTC")
    unix = ts_all.asi8 / 1e9
    names = list(df.columns)
    stacked = df.to_numpy(dtype=float)
    n_evt, n_det = stacked.shape
    return binning.fill_time_series(
        np.repeat(unix, n_det),
        np.tile(np.asarray(names, dtype=object), n_evt),
        stacked.ravel(),
        names,
        unix.min(),
        unix.max(),
        cadence,
    )


def test_matches_the_long_format_fill():
    df = _frame()
    got = binning.frame_to_binned(df)
    want = _long_format_reference(df)

    gv, wv = got.hist.view(), want.hist.view()
    assert np.array_equal(gv["count"], wv["count"])
    np.testing.assert_allclose(gv["value"], wv["value"], rtol=0, atol=0)
    np.testing.assert_allclose(
        gv["_sum_of_deltas_squared"], wv["_sum_of_deltas_squared"]
    )
    np.testing.assert_array_equal(np.isnan(got.mins), np.isnan(want.mins))
    np.testing.assert_allclose(
        got.mins[~np.isnan(got.mins)], want.mins[~np.isnan(want.mins)]
    )
    np.testing.assert_allclose(
        got.maxs[~np.isnan(got.maxs)], want.maxs[~np.isnan(want.maxs)]
    )


def test_matches_at_every_contract_cadence():
    df = _frame(seed=7)
    for cadence in schema.CADENCES:
        got = binning.frame_to_binned(df, cadence=cadence)
        want = _long_format_reference(df, cadence=cadence)
        assert np.array_equal(
            got.hist.view()["count"], want.hist.view()["count"]
        ), cadence


def test_detector_with_no_finite_values_is_kept_as_an_empty_column():
    df = _frame(with_gaps=False)
    df["V02160B"] = np.nan
    binned = binning.frame_to_binned(df)
    col = list(df.columns).index("V02160B")
    assert binned.hist.view()["count"][:, col].sum() == 0
    assert np.isnan(binned.mins[:, col]).all()
    # the other detectors are unaffected
    assert binned.hist.view()["count"][:, 0].sum() > 0


def test_explicit_window_bounds_are_respected():
    df = _frame()
    unix = df.index.asi8 / 1e9
    got = binning.frame_to_binned(df, t_start=unix.min(), t_stop=unix.max())
    want = _long_format_reference(df)
    assert got.hist.view()["count"].shape == want.hist.view()["count"].shape


def test_min_max_bracket_the_bin_mean():
    df = _frame()
    binned = binning.frame_to_binned(df)
    view = binned.hist.view()
    filled = view["count"] > 0
    assert (binned.mins[filled] <= view["value"][filled] + 1e-9).all()
    assert (binned.maxs[filled] >= view["value"][filled] - 1e-9).all()
