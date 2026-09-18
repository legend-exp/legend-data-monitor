"""End-to-end test of contract.build on a synthetic v1 per-run file."""

import numpy as np
import pandas as pd

from legend_data_monitor.contract import build, reader


def _make_v1_file(tmp_path, period="p19", run="r001"):
    run_dir = tmp_path / "generated/plt/hit/phy" / period / run
    run_dir.mkdir(parents=True)
    path = run_dir / f"l200-{period}-{run}-phy-geds.hdf"

    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-07-01", periods=240, freq="20s", tz="UTC")
    rawids = [1104000, 1104001]
    abs_df = pd.DataFrame(rng.normal(6000, 20, (240, 2)), index=idx, columns=rawids)
    abs_df.index.name = "datetime"
    abs_df.to_hdf(path, key="IsPulser_Trapemax", mode="a")
    var_df = (abs_df / abs_df.mean() - 1) * 100
    var_df.to_hdf(path, key="IsPulser_Trapemax_var", mode="a")
    abs_df.mean().to_frame().T.to_hdf(path, key="IsPulser_Trapemax_mean", mode="a")
    info = pd.DataFrame({"Value": ["ADC", "trapEmax"]}, index=["unit", "label"])
    info.to_hdf(path, key="IsPulser_Trapemax_info", mode="a")
    return str(tmp_path), run_dir


def test_build_contract_files_from_v1(tmp_path):
    root, run_dir = _make_v1_file(tmp_path)
    manifest_path = build.build_contract_files(root, "p19", "r001")
    assert manifest_path is not None

    manifest = reader.read_manifest(str(run_dir), "p19", "r001")
    assert manifest["schema_version"] == 2
    fname = next(iter(manifest["files"]))
    assert fname.endswith("-schema2.hdf")
    keys = manifest["files"][fname]["keys"]
    assert "hist/IsPulser_Trapemax/1min" in keys
    assert "hist/IsPulser_Trapemax/60min" in keys
    assert "hist/IsPulser_Trapemax_dist" in keys
    assert "IsPulser_Trapemax_mean" in keys
    assert not any(k.endswith("_info") for k in keys)

    v2 = str(run_dir / fname)
    binned = reader.read_binned_series(v2, "IsPulser", "Trapemax", "1min")
    # without metadata the columns are stringified rawids
    assert binned.detectors == ["1104000", "1104001"]
    total = binned.hist.view()["count"].sum()
    assert total == 480  # 240 timestamps x 2 detectors

    # binned means aggregate the raw series faithfully
    frame = binned.to_frame("mean")
    assert abs(np.nanmean(frame.to_numpy()) - 6000) < 10

    # 60min rebin preserves totals
    b60 = reader.read_binned_series(v2, "IsPulser", "Trapemax", "60min")
    assert b60.hist.view()["count"].sum() == total


def test_build_contract_files_missing_input(tmp_path):
    assert build.build_contract_files(str(tmp_path), "p19", "r001") is None


def test_param_attrs_cover_aux_and_variation_variants():
    """Every variant of a parameter gets label/unit; the aux ones say so."""
    from legend_data_monitor.contract.build import _param_attrs

    base = _param_attrs("/IsPulser_Baseline")
    assert base["unit"] == "ADC" and base["label"] == "FPGA baseline"
    var = _param_attrs("/IsPulser_Baseline_var")
    assert var["unit"] == "%" and var["limits"] == [-5, 5]
    ratio_var = _param_attrs("/IsPulser_Baseline_pulser01anaRatio_var")
    assert ratio_var["unit"] == "%" and ratio_var["label"].endswith("/ pulser01ana")
    ratio = _param_attrs("/IsPulser_Baseline_pulser01anaRatio")
    assert ratio["unit"] == "a. u."
    diff = _param_attrs("/IsPulser_Baseline_pulser01anaDiff")
    assert diff["unit"] == "ADC" and diff["label"].endswith("- pulser01ana")
    mean = _param_attrs("/IsPulser_BlMean_mean")
    assert mean["label"] == "Mean Baseline" and mean["unit"] == "ADC"


def test_keys_filter_refreshes_in_place(tmp_path):
    """keys= rewrites only the named v1 keys and keeps the rest of the file."""
    import json
    import os

    import h5py
    import numpy as np
    import pandas as pd

    from legend_data_monitor.contract import build, reader

    run_dir = tmp_path / "generated/plt/hit/phy/p22/r000"
    run_dir.mkdir(parents=True)
    v1 = str(run_dir / "l200-p22-r000-phy-geds.hdf")
    idx = pd.date_range("2026-01-01", periods=300, freq="min")
    cols = [1084803, 1084804]
    pd.DataFrame(1.0, index=idx, columns=cols).to_hdf(
        v1, key="IsPulser_Baseline", mode="a"
    )
    pd.DataFrame(2.0, index=idx, columns=cols).to_hdf(
        v1, key="IsPulser_BlMean", mode="a"
    )
    pd.DataFrame([[2.0, 2.0]], columns=cols).to_hdf(
        v1, key="IsPulser_BlMean_mean", mode="a"
    )
    build.build_contract_files(str(tmp_path), "p22", "r000")
    v2 = str(run_dir / "l200-p22-r000-phy-geds-schema2.hdf")

    # the producer fixes BlMean: refresh just that family
    pd.DataFrame(5.0, index=idx, columns=cols).to_hdf(
        v1, key="IsPulser_BlMean", mode="a"
    )
    pd.DataFrame([[5.0, 5.0]], columns=cols).to_hdf(
        v1, key="IsPulser_BlMean_mean", mode="a"
    )
    manifest = build.build_contract_files(
        str(tmp_path), "p22", "r000", keys=["IsPulser_BlMean", "IsPulser_BlMean_mean"]
    )

    refreshed = reader.read_binned_series(v2, "IsPulser", "BlMean", "10min")
    values = refreshed.hist.view()["value"]
    assert np.allclose(values[values != 0], 5.0)
    assert float(pd.read_hdf(v2, "IsPulser_BlMean_mean").iloc[0, 0]) == 5.0
    untouched = reader.read_binned_series(v2, "IsPulser", "Baseline", "10min")
    assert np.allclose(
        untouched.hist.view()["value"][untouched.hist.view()["count"] > 0], 1.0
    )
    # the manifest still lists everything, not just this pass
    with open(manifest) as f:
        keys = json.load(f)["files"][os.path.basename(v2)]["keys"]
    assert (
        "hist/IsPulser_Baseline/10min" in keys and "hist/IsPulser_BlMean/10min" in keys
    )
    assert "IsPulser_BlMean_mean" in keys
    with h5py.File(v2, "r") as f:
        assert f["hist/IsPulser_BlMean/1min/storage/values"].dtype == np.float32


def test_keyed_refresh_restores_a_missing_detector_map(tmp_path, monkeypatch):
    """A keyed refresh must not leave a contract without /detector_map."""
    from legend_data_monitor import utils

    root, run_dir = _make_v1_file(tmp_path)
    info = {
        "detectors": {
            "V01234A": {
                "daq_rawid": 1104000,
                "string": 1,
                "position": 1,
                "processable": True,
                "usability": "on",
                "mass_in_kg": 1.0,
            },
            "V05678B": {
                "daq_rawid": 1104001,
                "string": 1,
                "position": 2,
                "processable": True,
                "usability": "on",
                "mass_in_kg": 1.0,
            },
        }
    }
    monkeypatch.setattr(utils, "build_detector_info", lambda path: info)
    build.build_contract_files(root, "p19", "r001", metadata_path="meta")
    v2 = run_dir / "l200-p19-r001-phy-geds-schema2.hdf"
    import h5py

    with h5py.File(v2, "a") as f:
        del f["detector_map"]  # simulate a file that lost the key
    build.build_contract_files(
        root, "p19", "r001", metadata_path="meta", keys=["IsPulser_Trapemax"]
    )
    assert isinstance(pd.read_hdf(v2, "detector_map"), pd.DataFrame)
    manifest = reader.read_manifest(str(run_dir), "p19", "r001")
    keys = manifest["files"]["l200-p19-r001-phy-geds-schema2.hdf"]["keys"]
    assert "detector_map" in keys
