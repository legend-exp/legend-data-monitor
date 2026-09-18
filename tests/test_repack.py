"""Repacking existing output files into the current layout."""

import os

import numpy as np
import pandas as pd
import pytest

from legend_data_monitor import repack, utils


def _legacy_run(tmp_path, period="p22", run="r000"):
    run_dir = tmp_path / "generated/plt/hit/phy" / period / run
    run_dir.mkdir(parents=True)
    path = str(run_dir / f"l200-{period}-{run}-phy-geds.hdf")
    frame = pd.DataFrame(
        np.random.default_rng(0).normal(1000, 1, (20000, 20)).astype("float64"),
        index=pd.date_range("2026-01-01", periods=20000, freq="s"),
    )
    # the legacy layout: uncompressed float64, rewritten once per chunk
    for _ in range(3):
        frame.to_hdf(path, key="IsPulser_Trapemax", mode="a")
    pd.DataFrame.from_dict({"unit": "ADC"}, orient="index", columns=["Value"]).to_hdf(
        path, key="IsPulser_Trapemax_info", mode="a"
    )
    return path, frame


def test_repack_shrinks_and_preserves_values(tmp_path):
    path, frame = _legacy_run(tmp_path)
    before, after = repack.repack_pandas_hdf(path)

    assert after < before / 2
    assert os.path.getsize(path) == after
    assert not os.path.exists(path + ".repack")

    back = pd.read_hdf(path, key="IsPulser_Trapemax")
    assert (back.dtypes == "float32").all()
    assert np.allclose(back.to_numpy(), frame.to_numpy(), rtol=1e-6)
    assert back.index.equals(frame.index)
    # plotting metadata survives untouched
    assert pd.read_hdf(path, key="IsPulser_Trapemax_info").loc["unit", "Value"] == "ADC"


def test_repack_is_idempotent(tmp_path):
    path, _ = _legacy_run(tmp_path)
    _, once = repack.repack_pandas_hdf(path)
    before, after = repack.repack_pandas_hdf(path)
    # already in the current layout: left exactly as it was
    assert before == after == once


def test_repack_leaves_the_original_when_it_fails(tmp_path, monkeypatch):
    path, _ = _legacy_run(tmp_path)
    before = os.path.getsize(path)

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_hdf", boom)
    with pytest.raises(RuntimeError):
        repack.repack_pandas_hdf(path)
    assert os.path.getsize(path) == before
    assert not _residue(path)


def _residue(path):
    import glob as _glob

    return _glob.glob(path + ".repack*")


def test_repack_run_covers_both_file_kinds(tmp_path):
    path, _ = _legacy_run(tmp_path)
    contract, _ = _legacy_contract(tmp_path / "generated/plt/hit/phy/p22/r000")
    results = repack.repack_run(str(tmp_path), "p22", "r000")
    assert sorted(results) == sorted([path, contract])
    for before, after in results.values():
        assert after < before


# -------------------------------------------------------------------------
# contract (schema2) files
# -------------------------------------------------------------------------


def _legacy_contract(tmp_path):
    """Build a contract file as the writer produced it before the narrowing."""
    import h5py
    from uhi.io import hdf5 as uhi_hdf5

    from legend_data_monitor.contract import schema, writer
    from legend_data_monitor.processing import binning

    dets = ["V02160A", "V02160B", "P00574A"]
    rng = np.random.default_rng(3)
    n, t0 = 200_000, 1_700_000_000.0
    t1 = t0 + 300 * 3600
    binned = binning.fill_time_series(
        rng.uniform(t0, t1, n), rng.choice(dets, n), rng.normal(100, 5, n), dets, t0, t1
    )
    path = str(tmp_path / "l200-p22-r000-phy-geds-schema2.hdf")
    with h5py.File(path, "w") as f:
        f.attrs[schema.ROOT_ATTR] = schema.SCHEMA_VERSION
        group = f.create_group("hist/IsPulser_Trapemax/1min")
        uhi_hdf5.write(group, binned.hist)
        group.attrs["schema"] = schema.SCHEMA_VERSION
        group.create_dataset("min", data=binned.mins, compression="gzip")
        group.create_dataset("max", data=binned.maxs, compression="gzip")
    writer.write_frame(
        path, "detector_map", pd.DataFrame({"name": dets, "mass": [1.0, 2.0, 3.0]})
    )
    return path, binned


def test_repack_contract_narrows_storage_and_keeps_everything_readable(tmp_path):
    import h5py

    from legend_data_monitor.contract import reader

    path, binned = _legacy_contract(tmp_path)
    before, after = repack.repack_contract_hdf(path)
    assert after < before

    with h5py.File(path, "r") as f:
        storage = f["hist/IsPulser_Trapemax/1min/storage"]
        assert storage["values"].dtype == np.float32
        assert storage["variances"].dtype == np.float32
        assert storage["counts"].dtype == np.int32
        assert f["hist/IsPulser_Trapemax/1min/min"].dtype == np.float32
        assert f.attrs["schema"] if "schema" in f.attrs else True
        # `axes` object references must still resolve after the compacting copy
        refs = f["hist/IsPulser_Trapemax/1min/axes"][...]
        assert [f[r].name.rsplit("/", 1)[-1] for r in refs] == ["axis_0", "axis_1"]

    back = reader.read_binned_series(path, "IsPulser", "Trapemax", "1min")
    assert np.allclose(
        back.hist.view()["value"],
        binned.hist.view()["value"],
        rtol=1e-6,
        equal_nan=True,
    )
    # the pandas frames alongside must survive the copy untouched
    frame = pd.read_hdf(path, key="detector_map")
    assert list(frame["name"]) == ["V02160A", "V02160B", "P00574A"]


def test_repack_contract_is_idempotent(tmp_path):
    path, _ = _legacy_contract(tmp_path)
    _, once = repack.repack_contract_hdf(path)
    before, after = repack.repack_contract_hdf(path)
    assert before == after == once


# -------------------------------------------------------------------------
# stripping classifier pivots (they live on, binned, in the contract)
# -------------------------------------------------------------------------

CLASSIFIER_KEYS = [
    "All_IsValidTailRmsClassifier",
    "IsPulser_IsValidBlPolyRmsClassifier",
]
SURVIVOR_KEYS = ["IsPulser_AoeCustom", "IsPulser_IsSaturated", "IsPulser_Trapemax"]


def _run_with_classifiers(tmp_path, with_contract=True, complete_contract=True):
    """Build a current-layout v1 file with classifier pivots, plus its contract."""
    from legend_data_monitor.contract import writer
    from legend_data_monitor.processing import binning

    run_dir = tmp_path / "generated/plt/hit/phy/p22/r000"
    run_dir.mkdir(parents=True, exist_ok=True)
    v1 = str(run_dir / "l200-p22-r000-phy-geds.hdf")

    rng = np.random.default_rng(5)
    frames = {}
    for key in SURVIVOR_KEYS + CLASSIFIER_KEYS:
        frames[key] = pd.DataFrame(
            rng.normal(0, 1, (2000, 5)).astype("float32"),
            index=pd.date_range("2026-01-01", periods=2000, freq="s"),
        )
        frames[key].to_hdf(v1, key=key, mode="a", **utils.HDF_COMPRESSION)
    pd.DataFrame.from_dict({"unit": "ADC"}, orient="index", columns=["Value"]).to_hdf(
        v1, key="IsPulser_Trapemax_info", mode="a"
    )

    if with_contract:
        contract = str(run_dir / "l200-p22-r000-phy-geds-schema2.hdf")
        dets = ["V02160A", "V02160B"]
        t0 = 1_700_000_000.0
        binned = binning.fill_time_series(
            rng.uniform(t0, t0 + 3600, 500),
            rng.choice(dets, 500),
            rng.normal(0, 1, 500),
            dets,
            t0,
            t0 + 3600,
        )
        binnable = CLASSIFIER_KEYS if complete_contract else CLASSIFIER_KEYS[:1]
        for key in binnable:
            flag, param = key.split("_", 1)
            writer.write_binned_series(contract, flag, param, binned)
    return v1, frames


def _livetime_key(v1):
    """Return the key utils.get_livetime would pick (same selection expression)."""
    import h5py

    with h5py.File(v1, "r") as f:
        keys = sorted(f.keys())
    return [
        k for k in keys if "IsPulser" in k and "info" not in k and "_pulser" not in k
    ][0]


def test_strip_removes_classifiers_and_nothing_else(tmp_path):
    v1, frames = _run_with_classifiers(tmp_path)
    livetime_key_before = _livetime_key(v1)
    before, after = repack.strip_classifier_pivots(str(tmp_path), "p22", "r000")

    assert after < before
    assert os.path.getsize(v1) == after
    assert not _residue(v1)

    with pd.HDFStore(v1, "r") as store:
        keys = sorted(key.lstrip("/") for key in store.keys())
    assert keys == sorted(SURVIVOR_KEYS + ["IsPulser_Trapemax_info"])
    for key in SURVIVOR_KEYS:
        assert pd.read_hdf(v1, key=key).equals(frames[key])
    assert pd.read_hdf(v1, key="IsPulser_Trapemax_info").loc["unit", "Value"] == "ADC"
    # the strip must not change which key the livetime lookup lands on
    assert _livetime_key(v1) == livetime_key_before == "IsPulser_AoeCustom"


def test_strip_refuses_without_contract(tmp_path):
    v1, _ = _run_with_classifiers(tmp_path, with_contract=False)
    before = os.path.getsize(v1)
    got = repack.strip_classifier_pivots(str(tmp_path), "p22", "r000")
    assert got == (before, before)
    assert os.path.getsize(v1) == before
    assert not _residue(v1)
    with pd.HDFStore(v1, "r") as store:
        assert len(store.keys()) == len(SURVIVOR_KEYS + CLASSIFIER_KEYS) + 1


def test_strip_refuses_on_incomplete_contract(tmp_path):
    """One classifier missing from the contract poisons the whole strip."""
    v1, _ = _run_with_classifiers(tmp_path, complete_contract=False)
    before = os.path.getsize(v1)
    got = repack.strip_classifier_pivots(str(tmp_path), "p22", "r000")
    assert got == (before, before)
    with pd.HDFStore(v1, "r") as store:
        keys = [key.lstrip("/") for key in store.keys()]
    for key in CLASSIFIER_KEYS:
        assert key in keys


def test_strip_is_idempotent(tmp_path):
    _run_with_classifiers(tmp_path)
    _, once = repack.strip_classifier_pivots(str(tmp_path), "p22", "r000")
    before, after = repack.strip_classifier_pivots(str(tmp_path), "p22", "r000")
    assert before == after == once


def test_repack_contract_fixes_a_narrowed_but_slacked_file(tmp_path):
    """An interrupted earlier repack leaves float32 storage with slack behind."""
    import h5py

    path, _ = _legacy_contract(tmp_path)
    repack.repack_contract_hdf(path)
    # fake the bad state: orphan a large block in the *middle* of the file
    # (a trailing hole would just be truncated away at close)
    with h5py.File(path, "r+") as f:
        f.create_dataset("padding", data=np.zeros(500_000))
    with h5py.File(path, "r+") as f:
        f.create_dataset("keeper", data=np.zeros(8))
    with h5py.File(path, "r+") as f:
        del f["padding"]
    slacked = os.path.getsize(path)
    before, after = repack.repack_contract_hdf(path)
    assert before == slacked
    assert after < slacked
    assert not _residue(path)


def test_repack_contract_failure_leaves_original_untouched(tmp_path, monkeypatch):
    path, _ = _legacy_contract(tmp_path)
    before = os.path.getsize(path)

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(repack, "_compact", boom)
    with pytest.raises(RuntimeError):
        repack.repack_contract_hdf(path)
    assert os.path.getsize(path) == before
    assert not _residue(path)
    # the original must still be float64: nothing was narrowed in place
    import h5py

    with h5py.File(path, "r") as f:
        assert f["hist/IsPulser_Trapemax/1min/storage/values"].dtype == np.float64
