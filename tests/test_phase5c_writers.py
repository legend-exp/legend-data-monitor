"""Phase 5c: the writers that free the last figure-only data from the shelves."""

import numpy as np
import pandas as pd

from legend_data_monitor import calibration, monitoring
from legend_data_monitor.contract import reader, schema
from legend_data_monitor.processing import binning


def test_event_rate_qc_roundtrip(tmp_path):
    times = pd.to_datetime(np.linspace(1.7e9, 1.7e9 + 6 * 3600, 720), unit="s")
    key = monitoring.write_event_rate_qc(
        str(tmp_path),
        "p22",
        "r000",
        {"All events": times, "Failing QC": times[:100], "Empty": times[:0]},
        on_mass=140.0,
    )
    assert key == "event_rate_qc/r000"
    path = monitoring.period_contract_path(str(tmp_path), "p22")
    frame = reader.read_frame(path, key)
    assert set(frame.columns) == {"all_events", "failing_qc", "on_mass_kg"}
    assert (frame["on_mass_kg"] == 140.0).all()
    # 720 events over 6 h = 120/h -> 120/3600*1000/140 mHz/kg
    assert np.isclose(frame["all_events"].iloc[1], 120 / 3600 * 1000 / 140)


def test_event_rate_qc_with_no_events_writes_nothing(tmp_path):
    empty = pd.to_datetime(pd.Series([], dtype="float64"), unit="s")
    assert (
        monitoring.write_event_rate_qc(
            str(tmp_path), "p22", "r000", {"All events": empty}, 140.0
        )
        is None
    )


def test_escale_summary_flattens_all_leaf_kinds(tmp_path):
    partitions = {
        "V01234A": {
            "fwhms_peaks": {"2614.5": {"p22-r000": 2.5, "p22-r001": 2.6}},
            "bl_std": {"p22-r000": 10.0},
            "cal_params": {"p22-r000": [0.1, 0.2]},
        }
    }
    key = calibration.write_escale_summary(str(tmp_path), "p22", "r001", partitions)
    assert key == "escale/r001"
    path = monitoring.period_contract_path(str(tmp_path), "p22", "cal")
    frame = reader.read_frame(path, key)
    assert len(frame) == 5
    peaked = frame[frame["parameter"] == "fwhms_peaks"]
    assert set(peaked["peak"]) == {"2614.5"} and len(peaked) == 2
    assert set(frame[frame["parameter"].str.startswith("cal_params")]["parameter"]) == {
        "cal_params_c0",
        "cal_params_c1",
    }
    flat = frame[frame["parameter"] == "bl_std"].iloc[0]
    assert flat["peak"] == "" and flat["period_run"] == "p22-r000"


def test_psd_stability_roundtrip(tmp_path):
    runs = ["r000", "r001", "r002"]
    key = calibration.write_psd_stability(
        str(tmp_path),
        "p22",
        "r002",
        "V01234A",
        runs,
        [0.01, 0.011, np.nan],
        [1e-4, 1e-4, np.nan],
        [0.002, 0.002, np.nan],
        [1e-5, 1e-5, np.nan],
        {
            "status": True,
            "slow_shift_fail_runs": ["r001"],
            "sudden_shift_fail_runs": [],
        },
    )
    assert key == "psd_stability/r002/V01234A"
    path = monitoring.period_contract_path(str(tmp_path), "p22", "cal")
    frame = reader.read_frame(path, key)
    assert list(frame["run"]) == runs
    assert frame["slow_shift"].tolist() == [False, True, False]
    assert not frame["sudden_shift"].any()
    assert (frame["status"] == "True").all()


def test_writers_route_by_data_type(tmp_path):
    monitoring.write_dead_time(str(tmp_path), "p22", "r000", 1.0, 0.1, data_type="lac")
    lac = monitoring.period_contract_path(str(tmp_path), "p22", "lac")
    assert lac.endswith("l200-p22-lac-monitoring.hdf")
    assert reader.read_frame(lac, "dead_time/r000")["dead_time_s"].iloc[0] == 1.0
    series = {"V01234A": pd.Series([1.0], index=pd.to_datetime(["2026-01-01"]))}
    monitoring.write_stability_series(
        str(tmp_path), "p22", "r000", "gain_shift", "corr", series, data_type="ssc"
    )
    ssc = monitoring.period_contract_path(str(tmp_path), "p22", "ssc")
    assert reader.read_frame(ssc, "gain_shift/corr/r000").shape == (1, 1)


def test_fill_distribution_2d_per_detector_counts():
    frame = pd.DataFrame({"A": [0.0, 1.0, 20.0, np.nan], "B": [-20.0, 0.5, 0.5, 0.5]})
    hist = binning.fill_distribution_2d(frame, n_bins=10, value_range=(-15.0, 15.0))
    assert [str(c) for c in hist.axes[1]] == ["A", "B"]
    counts = hist.view(flow=True)
    # A: two in range, one overflow, one NaN dropped; B: three in range, one underflow
    a, b = counts[:, 0], counts[:, 1]
    assert a[1:-1].sum() == 2 and a[-1] == 1
    assert b[1:-1].sum() == 3 and b[0] == 1
    assert hist.sum(flow=True) == 7


def test_build_writes_classifier_dist2d(tmp_path):
    import h5py

    from legend_data_monitor.contract import build

    run_dir = tmp_path / "generated/plt/hit/phy/p22/r000"
    run_dir.mkdir(parents=True)
    v1 = str(run_dir / "l200-p22-r000-phy-geds.hdf")
    idx = pd.date_range("2026-01-01", periods=500, freq="min")
    rng = np.random.default_rng(0)
    pd.DataFrame(
        rng.normal(0, 3, (500, 2)), index=idx, columns=[1084803, 1084804]
    ).to_hdf(v1, key="All_IsValidTailRmsClassifier", mode="a")
    pd.DataFrame(
        rng.normal(0, 1, (500, 2)), index=idx, columns=[1084803, 1084804]
    ).to_hdf(v1, key="IsPulser_Baseline", mode="a")
    build.build_contract_files(str(tmp_path), "p22", "r000")
    v2 = str(run_dir / "l200-p22-r000-phy-geds-schema2.hdf")
    with h5py.File(v2, "r") as f:
        assert schema.dist2d_key("All", "IsValidTailRmsClassifier") in f
        # non-classifier keys keep only the pooled 1-D dist
        assert schema.dist2d_key("IsPulser", "Baseline") not in f
        assert schema.dist_key("IsPulser", "Baseline") in f
        values = f[schema.dist2d_key("All", "IsValidTailRmsClassifier")][
            "storage/values"
        ][...]
        assert values.shape[0] == 78  # 76 bins + flow


def test_fill_distribution_range_ignores_outliers():
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(0, 1, 10_000), [1e4]])
    hist = binning.fill_distribution(values, n_bins=100)
    lo, hi = hist.axes[0].edges[0], hist.axes[0].edges[-1]
    assert -3.5 < lo < -2 and 2 < hi < 3.5  # the bulk, not the outlier
    counts = hist.view(flow=True)
    assert counts[-1] >= 1  # the outlier survives in the overflow bin
    assert (counts[1:-1] > 0).sum() > 50  # and the bulk spans many bins
