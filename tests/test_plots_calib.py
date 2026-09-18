"""Contract-driven calibration renderers (escale panels, PSD stability)."""

import os

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from legend_data_monitor import calibration
from legend_data_monitor.plots import calib

PERIOD, RUN = "p22", "r000"
DET = "V01234A"


def _detector_map():
    return pd.DataFrame(
        {"name": [DET], "rawid": [1084803], "string": [1], "position": [2]}
    )


def _write_escale(root):
    partitions = {
        DET: {
            "fwhms_peaks": {"2614.511": {"p22-r000": 2.5, "p22-r001": 2.6}},
            "fwhms_err_peaks": {"2614.511": {"p22-r000": 0.1, "p22-r001": 0.1}},
            "residuals": {"2103.511": {"p22-r000": 0.1, "p22-r001": -0.2}},
            "bl_std": {"p22-r000": 10.0, "p22-r001": 10.5},
            "aoe_mu": {"p22-r000": 0.01, "p22-r001": 0.011},
            "aoe_mu_err": {"p22-r000": 1e-4, "p22-r001": 1e-4},
            "cal_params": {"p22-r000": [0.1, 0.2]},
        }
    }
    calibration.write_escale_summary(str(root), PERIOD, RUN, partitions)


def test_escale_pdf_name_and_grid(tmp_path):
    import matplotlib.pyplot as plt

    _write_escale(tmp_path)
    paths = calib.plot_escale_panels(
        str(tmp_path), PERIOD, RUN, detector_map=_detector_map()
    )
    assert [os.path.basename(p) for p in paths] == [
        f"{PERIOD}_string1_pos2_{DET}_ESCALEusability.pdf"
    ]
    assert f"{os.sep}{PERIOD}{os.sep}mtg{os.sep}pdf{os.sep}st1{os.sep}" in paths[0]

    frame = pd.read_hdf(
        str(tmp_path / PERIOD / f"l200-{PERIOD}-cal-monitoring.hdf"), f"escale/{RUN}"
    )
    fig = calib._build_escale_figure(DET, 1, frame, ["p22-r000", "p22-r001"])
    assert len(fig.axes) == 12  # legacy 4x3 grid
    titles = {ax.get_title() for ax in fig.axes}
    for expected in ["Usability", "FWHM at FEP", "SEP residuals", "AoE mu"]:
        assert expected in titles
    plt.close(fig)


def test_escale_usability_shading(tmp_path):
    _write_escale(tmp_path)
    status = {DET: {"usability": {"p22-r000": "on", "p22-r001": "ac"}}}
    paths = calib.plot_escale_panels(
        str(tmp_path),
        PERIOD,
        RUN,
        detector_map=_detector_map(),
        detector_status=status,
    )
    assert len(paths) == 1


def test_psd_pdf_name_and_panels(tmp_path):
    import matplotlib.pyplot as plt

    calibration.write_psd_stability(
        str(tmp_path),
        PERIOD,
        RUN,
        DET,
        ["r000", "r001", "r002"],
        [0.010, 0.0101, 0.0102],
        [1e-4, 1e-4, 1e-4],
        [0.002, 0.002, 0.002],
        [1e-5, 1e-5, 1e-5],
        {"status": True, "slow_shift_fail_runs": [], "sudden_shift_fail_runs": []},
    )
    paths = calib.plot_psd_stability(
        str(tmp_path), PERIOD, RUN, detector_map=_detector_map()
    )
    assert [os.path.basename(p) for p in paths] == [
        f"{PERIOD}_string1_pos2_{DET}_AoE_stab.pdf"
    ]

    eval_result = calibration.evaluate_psd_performance(
        [0.010, 0.0101, 0.0102],
        [0.002, 0.002, 0.002],
        ["r000", "r001", "r002"],
        "r002",
        DET,
    )
    fig = calib._build_psd_figure(
        DET,
        ["r000", "r001", "r002"],
        [0.010, 0.0101, 0.0102],
        [1e-4] * 3,
        [0.002] * 3,
        [1e-5] * 3,
        eval_result,
    )
    assert len(fig.axes) == 4
    ylabels = {ax.get_ylabel() for ax in fig.axes}
    assert "Mean stability" in ylabels and "Sigma stability" in ylabels
    plt.close(fig)


def test_missing_inputs_are_not_fatal(tmp_path):
    assert calib.plot_escale_panels(str(tmp_path), PERIOD, RUN) == []
    assert calib.plot_psd_stability(str(tmp_path), PERIOD, RUN) == []
