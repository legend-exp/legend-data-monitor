"""Per-event reductions of the ragged SiPM fields (processing/spms.py)."""

import numpy as np
import pandas as pd
import pytest
from lgdo import lh5
from lgdo.types import Array, Table, VectorOfVectors

from legend_data_monitor.loading import phy_files
from legend_data_monitor.processing import spms


def _write_spms_file(path, n_events=4):
    table = Table(
        {
            "timestamp": Array(np.arange(n_events, dtype=float) + 100.0),
            "energy_in_pe": VectorOfVectors(
                [[1.0, 2.5], [], [0.5, 4.0, 3.0], [9.0]], dtype=np.float32
            ),
            "trigger_pos": VectorOfVectors(
                [[500.0, 700.0], [], [300.0, 900.0, 950.0], [50.0]], dtype=np.float32
            ),
            "is_valid_hit": VectorOfVectors(
                [[True, True], [], [False, True, True], [False]], dtype=bool
            ),
            "has_any_noise": Array(np.array([False, False, True, False])),
        }
    )
    lh5.write(table, "hit", str(path), group="ch1057600", wo_mode="overwrite")
    return str(path)


def test_expand_fields_adds_sources_and_keeps_plain():
    fields, derived = spms.expand_fields(["pe_max", "has_any_noise", "timestamp"])
    assert fields == ["energy_in_pe", "is_valid_hit", "has_any_noise", "timestamp"]
    assert derived == ["pe_max"]
    assert spms.is_reduction("n_pulses") and not spms.is_reduction("wf_mode")


def test_loader_reduces_ragged_fields(tmp_path):
    path = _write_spms_file(tmp_path / "spms.lh5")
    frame = phy_files.load_channel_frame(
        [path],
        "hit",
        ["ch1057600"],
        ["n_pulses", "pe_sum", "pe_max", "first_trigger_ns"],
    )
    # only scalars leave the loader: the ragged sources are dropped
    assert set(frame.columns) == {
        "timestamp",
        "channel",
        "n_pulses",
        "pe_sum",
        "pe_max",
        "first_trigger_ns",
    }
    assert list(frame["n_pulses"]) == [2, 0, 2, 0]
    np.testing.assert_allclose(frame["pe_sum"], [3.5, 0.0, 7.0, 0.0])
    np.testing.assert_array_equal(frame["pe_max"], [2.5, np.nan, 4.0, np.nan])
    np.testing.assert_array_equal(
        frame["first_trigger_ns"], [500.0, np.nan, 900.0, np.nan]
    )
    assert all(frame[c].dtype == np.float32 for c in ["pe_sum", "pe_max", "n_pulses"])
    assert frame["channel"].dtype == np.int32


def test_loader_keeps_requested_source_column(tmp_path):
    path = _write_spms_file(tmp_path / "spms.lh5")
    frame = phy_files.load_channel_frame(
        [path], "hit", ["ch1057600"], ["n_pulses", "has_any_noise"]
    )
    assert list(frame["has_any_noise"]) == [False, False, True, False]
    assert "is_valid_hit" not in frame.columns


def test_unknown_op_is_config_error(monkeypatch):
    from legend_data_monitor import errors, utils

    monkeypatch.setitem(utils.SPMS_REDUCTIONS, "bad", {"fields": ["x"], "op": "nope"})
    frame = pd.DataFrame({"x": pd.Series([[1.0]], dtype=object)})
    with pytest.raises(errors.ConfigError):
        spms.reduce_frame(frame, ["bad"], ["bad"])
