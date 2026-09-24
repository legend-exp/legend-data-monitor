"""plot_run: figures regenerated from the contract file alone.

Rendering is separated from data generation so an unattended run can skip it
(--plots off) and the figures can be produced afterwards, cheaply, without any
access to the production tree. These tests pin that the renderer needs only
the contract file and reports what it wrote.
"""

import numpy as np
import pandas as pd

from legend_data_monitor import automatic_run
from legend_data_monitor.contract import writer
from legend_data_monitor.processing import binning

DETS = ["V02160A", "V02160B", "P00574A"]


def _contract_run(tmp_path, period="p22", run="r012", data_type="phy"):
    """Build a minimal but real contract-v2 run directory."""
    run_dir = tmp_path / "generated" / "plt" / "hit" / data_type / period / run
    run_dir.mkdir(parents=True)
    path = str(run_dir / f"l200-{period}-{run}-{data_type}-geds-schema2.hdf")

    rng = np.random.default_rng(1)
    n, t0 = 4000, 1_700_000_000.0
    t = rng.uniform(t0, t0 + 2 * 3600, n)
    d = rng.choice(DETS, n)
    v = rng.normal(1000, 5, n)
    binned = binning.fill_time_series(t, d, v, DETS, t0, t0 + 2 * 3600)

    for flag, param, _unit in automatic_run.HEADLINE_PNG_KEYS[:1]:
        writer.write_binned_series(path, flag, param, binned)
    writer.write_frame(
        path,
        "detector_map",
        pd.DataFrame(
            [
                {"name": DETS[0], "rawid": 1084803, "string": 1, "position": 1},
                {"name": DETS[1], "rawid": 1084804, "string": 1, "position": 2},
                {"name": DETS[2], "rawid": 1084805, "string": 2, "position": 1},
            ]
        ),
    )
    return run_dir


def test_render_run_plots_writes_one_figure_per_string(tmp_path):
    run_dir = _contract_run(tmp_path)
    saved = automatic_run.render_run_plots(str(tmp_path), "p22", "r012")
    # one headline key x two strings
    assert len(saved) == 2
    assert all(p.endswith(".png") for p in saved)
    assert sorted(p.split("_st")[-1] for p in saved) == ["01.png", "02.png"]
    assert (run_dir / "figs").is_dir()


def test_render_run_plots_returns_absolute_paths(tmp_path):
    """auto-giorgio attaches these paths directly."""
    import os

    _contract_run(tmp_path)
    saved = automatic_run.render_run_plots(str(tmp_path), "p22", "r012")
    assert all(os.path.isabs(p) for p in saved)
    assert all(os.path.isfile(p) for p in saved)


def test_render_run_plots_emits_saved_plot_lines(tmp_path, caplog):
    """SAVED_PLOT is the attachment contract; it must fire when run standalone."""
    _contract_run(tmp_path)
    with caplog.at_level("INFO"):
        saved = automatic_run.render_run_plots(str(tmp_path), "p22", "r012")
    lines = [r.getMessage() for r in caplog.records if "SAVED_PLOT" in r.getMessage()]
    assert len(lines) == len(saved)


def test_render_run_plots_without_a_contract_file_is_not_fatal(tmp_path):
    # a run processed before contract v2, or a wrong period/run
    assert automatic_run.render_run_plots(str(tmp_path), "p22", "r999") == []


def test_render_run_plots_spms_only_groups_by_barrel_and_position(tmp_path):
    run_dir = tmp_path / "generated/plt/hit/phy/p22/r012"
    run_dir.mkdir(parents=True)
    path = str(run_dir / "l200-p22-r012-phy-spms-schema2.hdf")
    dets = ["S060", "S061"]
    rng = np.random.default_rng(2)
    n, t0 = 2000, 1_700_000_000.0
    binned = binning.fill_time_series(
        rng.uniform(t0, t0 + 3600, n),
        rng.choice(dets, n),
        rng.integers(0, 2, n).astype(float),
        dets,
        t0,
        t0 + 3600,
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
    saved = automatic_run.render_run_plots(str(tmp_path), "p22", "r012")
    names = sorted(p.rsplit("/", 1)[-1] for p in saved)
    assert names == ["All_HasAnyNoise_IB_bottom.png", "All_HasAnyNoise_IB_top.png"]


def test_render_run_plots_draws_the_fep_summary_from_the_cal_file(tmp_path):
    """check_calibration writes the FEP box summary next to the phy file."""
    from legend_data_monitor import monitoring

    run_dir = _contract_run(tmp_path)
    frame = pd.DataFrame(
        [
            {
                "ged": DETS[0],
                "string": 1,
                "pos": 1,
                "mean": 0.1,
                "std": 0.05,
                "min": -0.2,
                "max": 0.3,
                "fwhm": 2.5,
                "usability": "on",
            },
            {
                "ged": DETS[2],
                "string": 2,
                "pos": 1,
                "mean": -0.1,
                "std": 0.02,
                "min": -0.3,
                "max": 0.2,
                "fwhm": np.nan,
                "usability": "off",
            },
        ]
    )
    monitoring.write_detector_summary(
        str(tmp_path / "generated/plt/hit/phy"),
        "p22",
        "r012",
        "FEP_gain_stab",
        frame,
        data_type="cal",
    )
    saved = automatic_run.render_run_plots(str(tmp_path), "p22", "r012")
    fep = [p for p in saved if p.endswith("p22_r012_FEP_gain_stab.pdf")]
    assert len(fep) == 1
    assert fep[0].startswith(str(run_dir / "mtg" / "pdf"))


def test_last_cycle_comes_from_the_manifest(tmp_path):
    run_dir = _contract_run(tmp_path)
    assert automatic_run._manifest_last_cycle(str(run_dir), "p22", "r012") is None
    writer.write_manifest(
        str(run_dir),
        "p22",
        "r012",
        {},
        package_version="test",
        last_cycle="20260101T000000Z",
    )
    assert (
        automatic_run._manifest_last_cycle(str(run_dir), "p22", "r012")
        == "20260101T000000Z"
    )


def test_fep_summary_is_only_looked_up_for_phy_runs(tmp_path, caplog):
    _contract_run(tmp_path, data_type="ssc")
    saved = automatic_run.render_run_plots(
        str(tmp_path), "p22", "r012", data_type="ssc"
    )
    assert not [p for p in saved if "FEP_gain_stab" in p]
    assert "detector_summary/FEP_gain_stab" not in caplog.text
