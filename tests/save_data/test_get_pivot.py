"""get_pivot merges appended chunks by what the column IS, not by its name."""

import numpy as np
import pandas as pd
import pytest

from legend_data_monitor import save_data


def _chunk(i, parameter, channels=(1084803, 1084804), rows=60):
    start = pd.Timestamp("2026-01-01") + pd.Timedelta(hours=i)
    times = pd.date_range(start, periods=rows, freq="min")
    frame = pd.DataFrame(
        {
            "datetime": np.repeat(times, len(channels)),
            "channel": np.tile(channels, rows),
            parameter: 1000.0 + i + np.arange(rows * len(channels)) * 0.001,
        }
    )
    # the pipeline writes <param>_mean and <param>_var alongside the values
    frame[parameter + "_mean"] = 1000.0 + i
    frame[parameter + "_var"] = (
        frame[parameter] / frame[parameter + "_mean"] - 1
    ) * 100
    return frame


def _append_chunks(path, parameter, key, n_chunks=3):
    for i in range(n_chunks):
        frame = _chunk(i, parameter)
        save_data.get_pivot(frame, parameter, key, path, "append", kind="abs")
        save_data.get_pivot(
            frame, parameter + "_mean", key + "_mean", path, "append", kind="mean"
        )
        save_data.get_pivot(
            frame, parameter + "_var", key + "_var", path, "append", kind="var"
        )


@pytest.mark.parametrize("parameter", ["baseline", "bl_mean", "pz_mean"])
def test_appended_chunks_accumulate_whatever_the_name(tmp_path, parameter):
    """bl_mean used to be mistaken for a run mean and truncated to one row."""
    path = str(tmp_path / "l200-p22-r000-phy-geds.hdf")
    _append_chunks(path, parameter, "IsPulser_X")

    absolute = pd.read_hdf(path, key="IsPulser_X")
    assert len(absolute) == 3 * 60  # every chunk's rows survive
    mean = pd.read_hdf(path, key="IsPulser_X_mean")
    assert len(mean) == 1 and float(mean.iloc[0, 0]) == 1002.0  # newest chunk
    var = pd.read_hdf(path, key="IsPulser_X_var")
    # the % variation spans the whole history, recomputed with the newest mean
    assert var.index.equals(absolute.index)
    expected = (absolute / 1002.0 - 1) * 100
    assert np.allclose(var.to_numpy(), expected.to_numpy(), atol=1e-3)


def test_unknown_kind_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        save_data.get_pivot(
            _chunk(0, "baseline"),
            "baseline",
            "K",
            str(tmp_path / "x.hdf"),
            "append",
            kind="mystery",
        )


def test_overwrite_saving_ignores_history(tmp_path):
    path = str(tmp_path / "l200-p22-r000-phy-geds.hdf")
    for i in range(2):
        save_data.get_pivot(
            _chunk(i, "baseline"),
            "baseline",
            "IsPulser_X",
            path,
            "overwrite",
            kind="abs",
        )
    assert len(pd.read_hdf(path, key="IsPulser_X")) == 60
