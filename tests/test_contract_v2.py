"""Contract v2 tests: binning correctness, round-trip, merge, h5py-only compatibility."""

import json

import h5py
import numpy as np
import pandas as pd
import pytest

from legend_data_monitor.contract import reader, schema, writer
from legend_data_monitor.processing import binning

DETS = ["V02160A", "V02160B", "P00574A"]


def _events(n=5000, seed=1, t0=1_700_000_000.0, hours=2.0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(t0, t0 + hours * 3600, n)
    d = rng.choice(DETS, n)
    v = rng.normal(100, 5, n)
    return t, d, v, t0, t0 + hours * 3600


def _binned(seed=1):
    t, d, v, t0, t1 = _events(seed=seed)
    return binning.fill_time_series(t, d, v, DETS, t0, t1)


# -------------------------------------------------------------------------
# binning correctness
# -------------------------------------------------------------------------


def test_bin_means_match_pandas_resample():
    t, d, v, t0, t1 = _events()
    binned = binning.fill_time_series(t, d, v, DETS, t0, t1)
    got = binned.to_frame("mean")

    df = pd.DataFrame(
        {"datetime": pd.to_datetime(t, unit="s", utc=True), "det": d, "val": v}
    )
    origin = pd.to_datetime(t0, unit="s", utc=True).floor("1min")
    expected = (
        df.pivot_table(index="datetime", columns="det", values="val", aggfunc="mean")
        .resample("1min", origin=origin)
        .mean()
    )
    joint = got.reindex(expected.index)[DETS]
    # compare where both have data
    for det in DETS:
        a, b = joint[det], expected[det]
        mask = a.notna() & b.notna()
        assert mask.sum() > 50
        # pivot_table pre-averages same-timestamp collisions; tolerance is loose
        assert np.allclose(a[mask], b[mask], rtol=1e-9)


def test_rebin_matches_direct_fill():
    t, d, v, t0, t1 = _events()
    base = binning.fill_time_series(t, d, v, DETS, t0, t1, cadence="1min")
    rebinned = base.rebin(60)
    direct = binning.fill_time_series(t, d, v, DETS, t0, t1, cadence="60min")
    bv, dv = rebinned.hist.view(), direct.hist.view()
    assert np.allclose(bv["count"], dv["count"])
    mask = bv["count"] > 0
    assert np.allclose(bv["value"][mask], dv["value"][mask])
    assert np.allclose(rebinned.mins[mask], direct.mins[mask])
    assert np.allclose(rebinned.maxs[mask], direct.maxs[mask])


def test_min_max_catch_spikes_hidden_by_mean():
    t0 = 1_700_000_000.0
    t = np.array([t0 + 10, t0 + 20, t0 + 30])
    d = np.array(["V02160A"] * 3)
    v = np.array([100.0, 100.0, 500.0])  # single-event spike
    binned = binning.fill_time_series(t, d, v, DETS, t0, t0 + 60)
    bin_idx = binned.hist.axes[0].index(t0 + 10)
    assert binned.maxs[bin_idx, 0] == 500.0
    assert binned.mins[bin_idx, 0] == 100.0
    assert binned.to_frame("mean").iloc[bin_idx, 0] == pytest.approx(700.0 / 3)


def test_merge_is_sum_and_minmax():
    b1, b2 = _binned(seed=1), _binned(seed=2)
    merged = b1 + b2
    assert np.allclose(
        merged.hist.view()["count"],
        b1.hist.view()["count"] + b2.hist.view()["count"],
    )
    assert np.array_equal(merged.mins, np.fmin(b1.mins, b2.mins), equal_nan=True)
    assert np.array_equal(merged.maxs, np.fmax(b1.maxs, b2.maxs), equal_nan=True)


# -------------------------------------------------------------------------
# writer/reader round-trip
# -------------------------------------------------------------------------


def test_roundtrip_binned_series(tmp_path):
    binned = _binned()
    path = str(tmp_path / "l200-p19-r001-phy-geds.hdf")
    keys = writer.write_binned_series(
        path,
        "IsPulser",
        "TrapemaxCtcCal",
        binned,
        attrs={"unit": "keV", "label": "Cal. gain", "limits": [-0.05, 0.05]},
    )
    assert keys == [
        "hist/IsPulser_TrapemaxCtcCal/1min",
        "hist/IsPulser_TrapemaxCtcCal/10min",
        "hist/IsPulser_TrapemaxCtcCal/60min",
    ]
    assert reader.read_schema_version(path) == schema.SCHEMA_VERSION

    back = reader.read_binned_series(path, "IsPulser", "TrapemaxCtcCal", "1min")
    v0, v1 = binned.hist.view(), back.hist.view()
    assert np.allclose(v0["count"], v1["count"])
    assert np.allclose(v0["value"], v1["value"], equal_nan=True)
    # storage is narrowed to float32 on write (halves the file and the time a
    # reader spends inflating it); everything must still match to float32
    assert np.allclose(binned.mins, back.mins, rtol=1e-6, equal_nan=True)
    assert list(back.detectors) == DETS

    with h5py.File(path, "r") as f:
        storage = f[keys[0]]["storage"]
        assert storage["counts"].dtype == np.int32
        assert storage["values"].dtype == np.float32
        assert storage["variances"].dtype == np.float32
        assert f[keys[0]]["min"].dtype == np.float32

    _, _, _, attrs = reader.read_hist(path, keys[0])
    assert attrs["unit"] == "keV"
    assert json.loads(attrs["limits"]) == [-0.05, 0.05]
    assert attrs["schema"] == schema.SCHEMA_VERSION


def test_roundtrip_distribution_and_frames(tmp_path):
    path = str(tmp_path / "l200-p19-r001-phy-geds.hdf")
    dist = binning.fill_distribution(np.random.default_rng(0).normal(0, 1, 1000))
    writer.write_distribution(path, "IsPulser", "Baseline", dist)
    back, _, _, _ = reader.read_hist(path, schema.dist_key("IsPulser", "Baseline"))
    assert np.allclose(dist.view(), back.view())

    means = pd.DataFrame([[1.0, 2.0, 3.0]], columns=DETS)
    writer.write_frame(path, schema.mean_key("IsPulser", "Baseline"), means)
    assert reader.read_frame(path, "IsPulser_Baseline_mean").iloc[0, 1] == 2.0

    detectors = {
        "V02160A": {
            "daq_rawid": 1104000,
            "string": 2,
            "position": 3,
            "processable": True,
            "usability": "on",
            "mass_in_kg": 1.0,
        }
    }
    writer.write_detector_map(path, detectors)
    dmap = reader.read_frame(path, "detector_map")
    assert dmap.loc[0, "rawid"] == 1104000

    keys = reader.list_hist_keys(path)
    assert "hist/IsPulser_Baseline_dist" in keys


def test_manifest_written_and_flagged_ranges(tmp_path):
    files = {"l200-p19-r001-phy-geds.hdf": {"keys": ["hist/IsPulser_Baseline/1min"]}}
    path = writer.write_manifest(str(tmp_path), "p19", "r001", files, "2.0.0")
    manifest = reader.read_manifest(str(tmp_path), "p19", "r001")
    assert manifest["schema_version"] == schema.SCHEMA_VERSION
    assert manifest["cadences"] == ["1min", "10min", "60min"]
    assert "flags" in manifest["key_vocabulary"]
    assert isinstance(manifest["flagged_ranges"], list)
    assert path.endswith("l200-p19-r001-manifest.json")


# -------------------------------------------------------------------------
# no-library-import compatibility: plain h5py + json only
# -------------------------------------------------------------------------


def test_v2_readable_with_plain_h5py(tmp_path):
    import subprocess
    import sys

    binned = _binned()
    path = str(tmp_path / "l200-p19-r001-phy-geds.hdf")
    writer.write_binned_series(path, "IsPulser", "Trapemax", binned)

    script = f"""
import sys
for mod in list(sys.modules):
    assert not mod.startswith("legend_data_monitor"), mod
import h5py
import numpy as np
with h5py.File({path!r}, "r") as f:
    assert int(f.attrs["lmon_schema_version"]) == 2
    g = f["hist/IsPulser_Trapemax/1min"]
    counts = g["storage/counts"][...]
    values = g["storage/values"][...]
    variances = g["storage/variances"][...]
    mins = g["min"][...]
    assert counts.sum() > 0
    assert values.shape == variances.shape == counts.shape
    assert np.isfinite(mins).any()
    assert "legend_data_monitor" not in sys.modules
print("plain-h5py OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "plain-h5py OK" in result.stdout


def test_apply_remove_keys_noop_without_entries():
    idx = pd.date_range("2026-07-01", periods=10, freq="1min", tz="UTC")
    df = pd.DataFrame({"V02160A": range(10)}, index=idx, dtype=float)
    out = writer.apply_remove_keys(df, "p99", "r999")
    pd.testing.assert_frame_equal(out, df)


def test_contract_file_carries_no_slack(tmp_path):
    """Narrowing float64 *in the file* would orphan the blocks uhi wrote.

    Needs a histogram big enough that the storage arrays dominate the fixed
    ~0.5 MB of HDF5 group metadata, or an orphaned float64 copy hides in the
    noise.
    """
    import os

    t, d, v, t0, t1 = _events(n=400_000, hours=300.0)
    binned = binning.fill_time_series(t, d, v, DETS, t0, t1)
    path = str(tmp_path / "l200-p19-r001-phy-geds.hdf")
    writer.write_binned_series(path, "IsPulser", "Trapemax", binned)

    with h5py.File(path, "r") as f:
        stored = []
        f.visititems(
            lambda n, o: (
                stored.append(o.id.get_storage_size())
                if isinstance(o, h5py.Dataset)
                else None
            )
        )
        assert f["hist/IsPulser_Trapemax/1min/storage/values"].dtype == np.float32
    # an orphaned float64 copy of the storage measures 1.37x here
    assert os.path.getsize(path) < 1.15 * sum(stored)


def test_apply_remove_keys_honours_period_and_run_scope(monkeypatch):
    from legend_data_monitor.config import settings

    idx = pd.date_range("2026-07-01", periods=4, freq="1h", tz="UTC")
    df = pd.DataFrame({"V01234A": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    entry = {"from": "2026-07-01 01:00Z", "to": "2026-07-01 02:00Z"}
    monkeypatch.setattr(
        settings, "REMOVE_KEYS", {"V01234A": [entry | {"period": "p22", "run": "r001"}]}
    )
    # the scoped entry applies to its own run only
    hit = writer.apply_remove_keys(df, "p22", "r001")
    assert hit["V01234A"].isna().sum() == 2
    for period, run in [("p22", "r002"), ("p21", "r001")]:
        miss = writer.apply_remove_keys(df, period, run)
        assert miss["V01234A"].notna().all()
    # an unscoped entry still applies everywhere
    monkeypatch.setattr(settings, "REMOVE_KEYS", {"V01234A": [entry]})
    assert writer.apply_remove_keys(df, "p21", "r009")["V01234A"].isna().sum() == 2
