"""How the v1 pandas HDF outputs are laid out on disk.

These are size/precision guarantees, not behaviour: the pipeline used to
write uncompressed float64 pivots, which cost ~7x the disk they needed --
2.6x from the dtype and the missing compression, and another 2.7x from the
slack that rewriting an *uncompressed* fixed-format key leaves behind.
"""

import numpy as np
import pandas as pd
import pytest

from legend_data_monitor import save_data, utils


def _events(n=400, channels=(1084803, 1084804)):
    return pd.DataFrame(
        {
            "datetime": np.repeat(
                pd.date_range("2026-01-01", periods=n // len(channels), freq="min"),
                len(channels),
            ),
            "channel": np.tile(channels, n // len(channels)),
            "trapEmax": np.linspace(1000.0, 2000.0, n).astype("float64"),
        }
    )


def test_narrow_to_native_undoes_nan_widening_only():
    """A NaN-filled float32 field comes back float64; a real float64 stays."""
    df = pd.DataFrame(
        {
            "timestamp": np.array([1.7e9, 1.7e9 + 1], dtype="float64"),
            "is_valid_bl_slope_classifier": np.array([1.0, np.nan], dtype="float64"),
            "trapEmax_ctc_cal": np.array([1000.0, 2000.0], dtype="float64"),
        }
    )
    native = {
        "timestamp": np.dtype("float64"),
        "is_valid_bl_slope_classifier": np.dtype("float32"),
        "trapEmax_ctc_cal": np.dtype("float64"),
    }
    out = utils.narrow_to_native_dtypes(df, native)
    assert out["is_valid_bl_slope_classifier"].dtype == "float32"
    assert np.isnan(out["is_valid_bl_slope_classifier"].iloc[1])
    # a float32 unix timestamp cannot even resolve a second at 1.7e9
    assert out["timestamp"].dtype == "float64"
    assert out["timestamp"].iloc[1] - out["timestamp"].iloc[0] == 1.0
    # parameters the tier really stores as float64 keep their precision: they
    # feed "value/mean - 1", where float32 inputs cost most of the digits
    assert out["trapEmax_ctc_cal"].dtype == "float64"


def test_pivots_are_stored_as_float32(tmp_path):
    path = str(tmp_path / "l200-p22-r000-phy-geds.hdf")
    save_data.get_pivot(_events(), "trapEmax", "IsPulser_Trapemax", path, "overwrite")
    stored = pd.read_hdf(path, key="IsPulser_Trapemax")
    assert (stored.dtypes == "float32").all()
    # ... and still the same numbers, to float32
    assert np.allclose(
        stored.iloc[0].to_numpy(), [1000.0, 1000 + 1000 / 399], rtol=1e-6
    )


def test_pivots_are_compressed(tmp_path):
    """Compression is what keeps rewritten keys from orphaning their old blocks."""
    import tables

    path = str(tmp_path / "l200-p22-r000-phy-geds.hdf")
    save_data.get_pivot(_events(), "trapEmax", "IsPulser_Trapemax", path, "overwrite")
    with tables.open_file(path) as f:
        node = f.get_node("/IsPulser_Trapemax/block0_values")
        assert node.filters.complib == utils.HDF_COMPRESSION["complib"]
        assert node.filters.complevel == utils.HDF_COMPRESSION["complevel"]


@pytest.mark.parametrize("chunks", [4])
def test_appending_chunks_does_not_bloat_the_file(tmp_path, chunks):
    """The 2.7x slack: an uncompressed key rewrite orphans the block it replaces."""
    path = str(tmp_path / "l200-p22-r000-phy-geds.hdf")
    import os

    rows = 20000
    for chunk in range(chunks):
        events = _events(n=rows)
        events["datetime"] += pd.Timedelta(days=chunk)
        save_data.get_pivot(events, "trapEmax", "IsPulser_Trapemax", path, "append")
    live = pd.read_hdf(path, key="IsPulser_Trapemax").memory_usage(deep=True).sum()
    assert os.path.getsize(path) < 1.5 * live
