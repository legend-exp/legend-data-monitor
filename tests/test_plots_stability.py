"""Contract-driven stability renderers (ports of plot_time_series / fep figures)."""

import os

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from legend_data_monitor import calibration, monitoring
from legend_data_monitor.plots import stability

PERIOD, RUN = "p22", "r000"
DET = "V01234A"


def _detector_map():
    return pd.DataFrame(
        {"name": [DET], "rawid": [1084803], "string": [1], "position": [2]}
    )


def _series(scale=1.0):
    idx = pd.date_range("2026-01-01", periods=48, freq="h")
    return pd.Series(np.linspace(-0.5, 0.5, 48) * scale, index=idx)


def _write_stability_inputs(root):
    monitoring.write_stability_series(
        str(root), PERIOD, RUN, "gain_shift", "corr", {DET: _series()}
    )
    monitoring.write_stability_series(
        str(root), PERIOD, RUN, "gain_shift", "corr_std", {DET: _series(0.1).abs()}
    )
    monitoring.write_stability_series(
        str(root), PERIOD, RUN, "param_stability", "TrapemaxCtcCal", {DET: _series()}
    )
    monitoring.write_stability_series(
        str(root),
        PERIOD,
        RUN,
        "param_stability",
        "TrapemaxCtcCal_std",
        {DET: _series(0.1).abs()},
    )
    monitoring.write_stability_series(
        str(root), PERIOD, RUN, "pul_cusp", "kevdiff", {DET: _series(0.5)}
    )
    monitoring.write_cal_points(
        str(root),
        PERIOD,
        RUN,
        [
            {
                "detector": DET,
                "string": 1,
                "position": 2,
                "run_start": pd.Timestamp("2026-01-01"),
                "fep_diff": 0.2,
                "cal_const_diff": -0.1,
                "res": 2.8,
                "res_quad": 2.6,
            }
        ],
    )


def test_stability_pdf_names_and_locations(tmp_path):
    _write_stability_inputs(tmp_path)
    paths = stability.plot_stability_series(
        str(tmp_path), PERIOD, RUN, detector_map=_detector_map()
    )
    names = {os.path.basename(p): p for p in paths}
    # gain shift is a PERIOD-level figure; the parameter one is run-level
    gain = f"{PERIOD}_string1_pos2_{DET}_corr_gain_shift.pdf"
    param = f"{PERIOD}_{RUN}_string1_pos2_{DET}_pulser_stab.pdf"
    assert gain in names and param in names
    assert f"{os.sep}{PERIOD}{os.sep}mtg{os.sep}pdf{os.sep}st1{os.sep}" in names[gain]
    assert (
        f"{os.sep}{PERIOD}{os.sep}{RUN}{os.sep}mtg{os.sep}pdf{os.sep}st1{os.sep}"
        in names[param]
    )


def test_gain_shift_figure_content(tmp_path):
    import matplotlib.pyplot as plt

    cal = pd.DataFrame(
        [
            {
                "detector": DET,
                "string": 1,
                "position": 2,
                "run_start": pd.Timestamp("2026-01-01"),
                "fep_diff": 0.2,
                "cal_const_diff": -0.1,
                "res": 2.8,
                "res_quad": 2.6,
            }
        ]
    )
    fig = stability._build_gain_shift_figure(
        PERIOD, DET, 1, 2, _series(), _series(0.1).abs(), _series(0.5), cal, True, False
    )
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    for expected in ["PULS01ANA", "GED corrected", "FEP gain", "cal. const. diff"]:
        assert expected in labels
    assert any("±1$\\sigma$" in lbl for lbl in labels)
    plt.close(fig)


def test_fep_gain_pdf_name(tmp_path):
    stats = pd.DataFrame(
        [
            {"time": 0.0, "mean": 2614.5, "std": 0.4, "count": 30},
            {"time": 600.0, "mean": 2614.8, "std": 0.4, "count": 31},
        ]
    )
    calibration.write_fep_gain_contract(
        str(tmp_path),
        PERIOD,
        RUN,
        {DET: {"stats": stats, "drift": pd.Series([0.0, 0.23])}},
    )
    paths = stability.plot_fep_gain(
        str(tmp_path), PERIOD, RUN, detector_map=_detector_map()
    )
    assert [os.path.basename(p) for p in paths] == [
        f"{PERIOD}_{RUN}_string1_pos2_{DET}_FEP_gain_stab.pdf"
    ]
    assert f"{os.sep}{RUN}{os.sep}mtg{os.sep}pdf{os.sep}st1{os.sep}" in paths[0]


def test_missing_inputs_are_not_fatal(tmp_path):
    assert stability.plot_stability_series(str(tmp_path), PERIOD, RUN) == []
    assert stability.plot_fep_gain(str(tmp_path), PERIOD, RUN) == []
