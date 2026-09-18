"""qc_distributions must survive ignore-key ranges that intersect the run.

The energy mask used to be built from an ignore-key-filtered frame and applied
to unfiltered ones, so pandas raised ``putmask: mask and data must be the same
size`` for any period whose ranges actually drop rows -- taking ft_summary,
event_rate_qc and dead_time down with it (reported from the p18 backfill).
"""

import numpy as np
import pandas as pd
import pytest

from legend_data_monitor import monitoring, utils
from legend_data_monitor.contract import reader

PERIOD, RUN = "p99", "r000"
RAWIDS = [1104000, 1104001]
CLASSIFIER = "IsValidBlSlopeClassifier"


def _det_info():
    return {
        "detectors": {
            "V01234A": {"daq_rawid": RAWIDS[0], "processable": True},
            "V05678B": {"daq_rawid": RAWIDS[1], "processable": True},
        },
        "str_chns": {1: ["V01234A", "V05678B"]},
    }


def _write_v1(root, n=60):
    run_dir = root / PERIOD / RUN
    run_dir.mkdir(parents=True)
    path = run_dir / f"l200-{PERIOD}-{RUN}-phy-geds.hdf"
    idx = pd.date_range("2026-07-01", periods=n, freq="1min", tz="UTC")
    idx.name = "datetime"
    energy = pd.DataFrame(
        np.linspace(20.0, 3000.0, n * 2).reshape(n, 2), index=idx, columns=RAWIDS
    )
    energy.to_hdf(path, key="IsPhysics_TrapemaxCtcCal", mode="a")
    for flag in ("All", "IsPulser", "IsBsln", "IsPhysics"):
        frame = pd.DataFrame(
            np.random.default_rng(0).normal(0, 2, (n, 2)), index=idx, columns=RAWIDS
        )
        frame.to_hdf(path, key=f"{flag}_{CLASSIFIER}", mode="a")
    return run_dir


def _ignore_middle_third():
    """Build an ignore-keys entry that really drops rows of the frames above."""
    return {
        PERIOD: {
            "start_keys": ["20260701T000900Z"],
            "stop_keys": ["20260701T002900Z"],
        }
    }


def test_qc_distributions_with_intersecting_ignore_keys(tmp_path, monkeypatch):
    _write_v1(tmp_path)
    monkeypatch.setattr(utils, "IGNORE_KEYS", _ignore_middle_third())
    # would raise ValueError("putmask: mask and data must be the same size")
    monitoring.qc_distributions(
        "", str(tmp_path), str(tmp_path), "20260701T000000Z", PERIOD, RUN, _det_info()
    )
    frame = reader.read_frame(
        monitoring.period_contract_path(str(tmp_path), PERIOD),
        f"qc_classifier_frac/{RUN}",
    )
    assert set(frame["event_type"]) == {"All", "IsPulser", "IsBsln", "IsPhysics"}
    assert set(frame["detector"]) == {"V01234A", "V05678B"}
    # the dropped rows are gone from every flag, physics included
    counts = frame.set_index(["event_type", "detector"])["n_events"]
    assert counts.loc[("All", "V01234A")] == 39
    # IsPhysics is additionally masked to energies > 25 keV
    assert counts.loc[("IsPhysics", "V01234A")] < counts.loc[("All", "V01234A")]


def test_qc_distributions_without_ignore_keys(tmp_path, monkeypatch):
    _write_v1(tmp_path)
    monkeypatch.setattr(utils, "IGNORE_KEYS", {})
    monitoring.qc_distributions(
        "", str(tmp_path), str(tmp_path), "20260701T000000Z", PERIOD, RUN, _det_info()
    )
    frame = reader.read_frame(
        monitoring.period_contract_path(str(tmp_path), PERIOD),
        f"qc_classifier_frac/{RUN}",
    )
    counts = frame.set_index(["event_type", "detector"])["n_events"]
    assert counts.loc[("All", "V01234A")] == 60


def test_qc_distributions_missing_file_is_not_fatal(tmp_path):
    assert (
        monitoring.qc_distributions(
            "", str(tmp_path), str(tmp_path), "k", PERIOD, "r999", _det_info()
        )
        is None
    )


@pytest.mark.parametrize("n_dropped", [0, 21])
def test_mask_and_frames_keep_the_same_length(tmp_path, monkeypatch, n_dropped):
    """The mask is built on the raw frame, so shapes always line up."""
    _write_v1(tmp_path)
    monkeypatch.setattr(
        utils, "IGNORE_KEYS", _ignore_middle_third() if n_dropped else {}
    )
    monitoring.qc_distributions(
        "", str(tmp_path), str(tmp_path), "20260701T000000Z", PERIOD, RUN, _det_info()
    )


def _store(tmp_path, frames):
    path = tmp_path / "v1.hdf"
    for key, frame in frames.items():
        frame.to_hdf(path, key=key, mode="a")
    return pd.HDFStore(path, "r")


def _frame(index, cols=RAWIDS, value=1.0):
    return pd.DataFrame(value, index=index, columns=cols)


def test_load_and_filter_survives_a_misaligned_mask(tmp_path):
    """A mask whose labels the target lacks used to raise from putmask."""
    idx = pd.date_range("2026-07-01", periods=10, freq="1min", tz="UTC")
    target = _frame(idx)
    # a duplicated timestamp in the mask makes label alignment ambiguous
    dupes = idx.append(pd.DatetimeIndex([idx[3]]))
    with _store(tmp_path, {"target": target}) as store:
        mask = _frame(dupes) > 0
        with pytest.raises(ValueError):
            store["target"].where(mask)  # the original failure mode
        out = utils.load_and_filter(store, "/target", mask=mask)
        assert out.empty  # dropped, not raised, and not silently uncut


def test_load_and_filter_aligns_by_position_when_shapes_match(tmp_path):
    idx = pd.date_range("2026-07-01", periods=6, freq="1min", tz="UTC")
    other = pd.date_range("2027-01-01", periods=6, freq="1min", tz="UTC")
    with _store(tmp_path, {"target": _frame(idx, value=5.0)}) as store:
        mask = pd.DataFrame([[True, False]] * 6, index=other, columns=RAWIDS)
        out = utils.load_and_filter(store, "/target", mask=mask)
        assert out[RAWIDS[0]].notna().all() and out[RAWIDS[1]].isna().all()


def test_load_and_filter_plain_paths(tmp_path):
    idx = pd.date_range("2026-07-01", periods=4, freq="1min", tz="UTC")
    with _store(tmp_path, {"target": _frame(idx)}) as store:
        assert utils.load_and_filter(store, "/missing").empty
        assert len(utils.load_and_filter(store, "/target")) == 4
        masked = utils.load_and_filter(store, "/target", mask=_frame(idx) > 2)
        assert masked.isna().all().all()
