"""SiPM threshold verdicts into qcp_summary.yaml and issue records."""

import json

import numpy as np
import yaml

from legend_data_monitor import monitoring, utils
from legend_data_monitor.contract import issues, writer
from legend_data_monitor.processing import binning

PERIOD, RUN = "p22", "r012"
DETS = ["S060", "S061"]


def _spms_contract(root):
    run_dir = root / PERIOD / RUN
    run_dir.mkdir(parents=True)
    path = str(run_dir / f"l200-{PERIOD}-{RUN}-phy-spms-schema2.hdf")
    rng = np.random.default_rng(3)
    n, t0 = 20000, 1_700_000_000.0
    t = rng.uniform(t0, t0 + 24 * 3600, n)
    d = rng.choice(DETS, n)
    # S061 is noisy 10 % of the time, S060 0.2 %
    noisy = rng.uniform(size=n) < np.where(d == "S061", 0.10, 0.002)
    binned = binning.fill_time_series(
        t, d, noisy.astype(float), DETS, t0, t0 + 24 * 3600
    )
    writer.write_binned_series(path, "All", "HasAnyNoise", binned)
    writer.write_detector_map(
        path,
        {
            "S060": {
                "daq_rawid": 1064000,
                "barrel": "IB",
                "fiber": "IB015016",
                "position": "top",
                "processable": True,
                "usability": "on",
            },
            "S061": {
                "daq_rawid": 1064001,
                "barrel": "IB",
                "fiber": "IB015016",
                "position": "bottom",
                "processable": True,
                "usability": "on",
            },
        },
        subsystem="spms",
    )
    figs = run_dir / "figs"
    figs.mkdir()
    (figs / "All_HasAnyNoise_IB_bottom.png").write_bytes(b"")
    (figs / "All_HasAnyNoise_IB_top.png").write_bytes(b"")
    return run_dir


def test_spms_thresholds_grade_and_emit_issue(tmp_path):
    root = tmp_path / "generated/plt/hit/phy"
    run_dir = _spms_contract(root)
    graded = monitoring.check_spms_thresholds(str(root), PERIOD, RUN)
    assert graded["S060"]["spms_noisy_frac"] is True
    assert graded["S061"]["spms_noisy_frac"] is False
    with open(run_dir / f"l200-{PERIOD}-{RUN}-qcp_summary.yaml") as f:
        summary = yaml.safe_load(f)
    assert summary["S061"]["phy"]["spms_noisy_frac"] is False

    spms_info = {
        "S061": {"daq_rawid": 1064001, "barrel": "IB", "position": "bottom"},
        "S060": {"daq_rawid": 1064000, "barrel": "IB", "position": "top"},
    }
    utils.check_cal_phy_thresholds(
        str(root), PERIOD, RUN, "phy", {}, detector_info=spms_info
    )
    path = issues.issues_file_path(str(tmp_path), PERIOD, RUN, "phy")
    with open(path) as f:
        records = [json.loads(line) for line in f.read().splitlines()]
    assert [r["detector"] for r in records] == ["S061"]
    rec = records[0]
    assert rec["metric"] == "spms_noisy_frac" and rec["rawid"] == 1064001
    assert rec.get("string") is None and rec.get("position") is None
    assert rec["data_ref"]["key"] == "hist/All_HasAnyNoise"
    assert rec["data_ref"]["file"].endswith("-spms-schema2.hdf")
    assert [p.rsplit("/", 1)[-1] for p in rec["plots"]] == [
        "All_HasAnyNoise_IB_bottom.png"
    ]
    assert rec["excursion"]["recovered"] is False


def test_spms_thresholds_without_contract(tmp_path):
    assert monitoring.check_spms_thresholds(str(tmp_path), PERIOD, RUN) == {}


def test_array_wide_noise_collapses_into_one_issue(tmp_path):
    """A burst on every SiPM is one event, not 58 detector problems."""
    root = tmp_path / "generated/plt/hit/phy"
    run_dir = root / PERIOD / RUN
    run_dir.mkdir(parents=True)
    dets = [f"S{i:03d}" for i in range(10)]
    path = str(run_dir / f"l200-{PERIOD}-{RUN}-phy-spms-schema2.hdf")
    rng = np.random.default_rng(7)
    n, t0 = 40000, 1_700_000_000.0
    t = rng.uniform(t0, t0 + 24 * 3600, n)
    d = rng.choice(dets, n)
    noisy = rng.uniform(size=n) < 0.12  # every channel noisy, all run long
    binned = binning.fill_time_series(
        t, d, noisy.astype(float), dets, t0, t0 + 24 * 3600
    )
    writer.write_binned_series(path, "All", "HasAnyNoise", binned)
    writer.write_detector_map(
        path,
        {
            det: {
                "daq_rawid": 1064000 + i,
                "barrel": "IB",
                "fiber": "IB015016",
                "position": "top",
                "processable": True,
                "usability": "on",
            }
            for i, det in enumerate(dets)
        },
        subsystem="spms",
    )
    graded = monitoring.check_spms_thresholds(str(root), PERIOD, RUN)
    assert all(g["spms_noisy_frac"] is False for g in graded.values())

    utils.check_cal_phy_thresholds(
        str(root),
        PERIOD,
        RUN,
        "phy",
        {},
        detector_info={det: {"daq_rawid": 1064000 + i} for i, det in enumerate(dets)},
    )
    with open(issues.issues_file_path(str(tmp_path), PERIOD, RUN, "phy")) as f:
        records = [json.loads(line) for line in f.read().splitlines()]
    assert len(records) == 1
    rec = records[0]
    assert rec["detector"] == "spms-array" and rec["metric"] == "spms_noisy_frac"
    assert rec["affected_detectors"] == dets and rec["affected_frac"] == 1.0
    assert rec["data_ref"]["key"] == "hist/All_HasAnyNoise"
