"""Contract-driven QC renderers (ports of the qc_* shelve figures)."""

import os

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from legend_data_monitor import monitoring  # noqa: E402
from legend_data_monitor.contract import writer  # noqa: E402
from legend_data_monitor.plots import qc  # noqa: E402
from legend_data_monitor.processing import binning  # noqa: E402

PERIOD, RUN = "p22", "r000"


def _detector_map():
    return pd.DataFrame(
        {
            "name": ["V01234A", "V05678B"],
            "rawid": [1084803, 1084804],
            "string": [1, 1],
            "position": [1, 2],
        }
    )


def _rates_frame():
    idx = pd.date_range("2026-01-01", periods=24, freq="h")
    return pd.DataFrame({"V01234A": 3.0, "V05678B": 6.0}, index=idx)


def _write_rate_inputs(root):
    for flag in ["IsDischarge", "IsValidBlSlope"]:
        monitoring.write_qc_rate_series(str(root), PERIOD, RUN, flag, _rates_frame())
    detectors = {
        "V01234A": {"daq_rawid": 1084803},
        "V05678B": {"daq_rawid": 1084804},
    }
    monitoring.write_qc_rates(
        str(root),
        PERIOD,
        RUN,
        {"IsDischarge": {1084803: 3.0, 1084804: 6.0}},
        detectors,
    )
    monitoring.write_dead_time(str(root), PERIOD, RUN, 12.0, 0.034)


def test_rate_series_pdf_names_and_content(tmp_path):
    _write_rate_inputs(tmp_path)
    paths = qc.plot_qc_rate_series(
        str(tmp_path), PERIOD, RUN, detector_map=_detector_map()
    )
    names = {os.path.basename(p) for p in paths}
    # thresholded flag uses the MTG title ("_rate" already inside), others get _rate
    assert f"{PERIOD}_{RUN}_string1_discharge_rate.pdf" in names
    assert f"{PERIOD}_{RUN}_string1_IsValidBlSlope_rate.pdf" in names
    for p in paths:
        assert os.path.exists(p)
        assert f"{os.sep}mtg{os.sep}pdf{os.sep}st1{os.sep}" in p


def test_rate_series_figure_content(tmp_path):
    import itertools

    import matplotlib.pyplot as plt

    cycle = itertools.cycle(plt.cm.tab20.colors)
    fig = qc._rate_series_figure(
        PERIOD,
        RUN,
        "IsDischarge",
        1,
        [("V01234A", 1), ("V05678B", 2)],
        _rates_frame(),
        {"V01234A": 3.0, "V05678B": 6.0},
        cycle,
    )
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "V01234A - pos 1 - 3.0 mHz" in labels
    # the 5 mHz upper threshold line is drawn for IsDischarge
    assert any("upper threshold" in lbl for lbl in labels)
    plt.close(fig)


def test_qc_average_pdf_name_and_deadtime_title(tmp_path):
    import matplotlib.pyplot as plt

    _write_rate_inputs(tmp_path)
    paths = qc.plot_qc_average(str(tmp_path), PERIOD, RUN, detector_map=_detector_map())
    assert [os.path.basename(p) for p in paths] == [
        f"{PERIOD}_{RUN}_discharge_rate_avg.pdf"
    ]
    fig = qc._average_figure(
        PERIOD,
        RUN,
        "IsDischarge",
        {1084803: 3.0},
        {1: [("V01234A", 1084803), ("V05678B", 1084804)]},
        {"dead_time_s": 12.0, "dead_time_pct": 0.034},
    )
    assert "tot dead time 0.034%" in fig.axes[0].get_title()
    assert fig.axes[0].get_yscale() == "log"
    plt.close(fig)


def test_classifier_distributions_from_dist2d(tmp_path):
    run_dir = tmp_path / PERIOD / RUN
    run_dir.mkdir(parents=True)
    contract = str(run_dir / f"l200-{PERIOD}-{RUN}-phy-geds-schema2.hdf")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(0, 3, (500, 2)), columns=["V01234A", "V05678B"])
    par = "IsValidTailRmsClassifier"
    for flag in ["All", "IsPulser"]:
        writer.write_distribution_2d(
            contract,
            flag,
            par,
            binning.fill_distribution_2d(frame, n_bins=76, value_range=(-15.0, 15.4)),
        )
    rows = [
        {
            "run": RUN,
            "classifier": par,
            "detector": det,
            "string": 1,
            "event_type": flag,
            "percent_in_range": 91.5,
            "n_events": 500,
        }
        for det in frame.columns
        for flag in ["All", "IsPulser"]
    ]
    monitoring.write_qc_classifier_fractions(str(tmp_path), PERIOD, RUN, rows)

    paths = qc.plot_classifier_distributions(
        str(tmp_path), PERIOD, RUN, detector_map=_detector_map()
    )
    assert [os.path.basename(p) for p in paths] == [f"{PERIOD}_{RUN}_string1_{par}.pdf"]


def test_missing_inputs_are_not_fatal(tmp_path):
    assert qc.plot_qc_rate_series(str(tmp_path), PERIOD, RUN) == []
    assert qc.plot_qc_average(str(tmp_path), PERIOD, RUN) == []
    assert qc.plot_classifier_distributions(str(tmp_path), PERIOD, RUN) == []
