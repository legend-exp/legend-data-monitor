"""Renderers of the run-summary figures (plots/summary.py)."""

import itertools
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402

from legend_data_monitor import monitoring, utils  # noqa: E402
from legend_data_monitor.contract import writer as contract_writer  # noqa: E402
from legend_data_monitor.plots import summary  # noqa: E402

PERIOD, RUN = "p22", "r000"


def _hourly_index(n=6):
    return pd.date_range("2026-01-01", periods=n, freq="h")


def _ft_rates():
    idx = _hourly_index()
    return pd.DataFrame(
        {"V01234A": np.linspace(0.0, 1.0, 6), "V05678B": np.linspace(1.0, 2.0, 6)},
        index=idx,
    )


def _build_ft_contract(root):
    idx = _hourly_index()
    monitoring.write_ft_series(str(root), PERIOD, RUN, "per_detector", _ft_rates())
    per_string = pd.DataFrame(
        {"1": np.linspace(0.0, 1.0, 6), "2": np.linspace(1.0, 2.0, 6)}, index=idx
    )
    monitoring.write_ft_series(str(root), PERIOD, RUN, "per_string", per_string)
    monitoring.write_ft_series(
        str(root), PERIOD, RUN, "total_forced", pd.Series(3600.0, index=idx)
    )
    monitoring.write_ft_series(
        str(root), PERIOD, RUN, "survival_fraction", pd.Series(99.0, index=idx)
    )


def _write_detector_map(root):
    run_dir = root / PERIOD / RUN
    run_dir.mkdir(parents=True, exist_ok=True)
    detectors = {
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
            "string": 2,
            "position": 1,
            "processable": True,
            "usability": "on",
            "mass_in_kg": 1.0,
        },
    }
    contract_writer.write_detector_map(
        str(run_dir / f"l200-{PERIOD}-{RUN}-phy-geds-schema2.hdf"), detectors
    )


def _summary_frame():
    return pd.DataFrame(
        [
            {
                "ged": "V01234A",
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
                "ged": "V05678B",
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


def test_ft_summary_saves_legacy_names(tmp_path):
    _build_ft_contract(tmp_path)
    _write_detector_map(tmp_path)
    paths = summary.plot_ft_summary(
        str(tmp_path), PERIOD, RUN, png_dir=str(tmp_path / "png")
    )
    assert all(os.path.exists(p) for p in paths)
    pdfs = {os.path.relpath(p, tmp_path) for p in paths if p.endswith(".pdf")}
    assert pdfs == {
        f"{PERIOD}/{RUN}/mtg/pdf/st1/{PERIOD}_{RUN}_string1_FT_failure.pdf",
        f"{PERIOD}/{RUN}/mtg/pdf/st2/{PERIOD}_{RUN}_string2_FT_failure.pdf",
        f"{PERIOD}/{RUN}/mtg/pdf/{PERIOD}_{RUN}_all_strings_FT_failure.pdf",
        f"{PERIOD}/{RUN}/mtg/pdf/{PERIOD}_{RUN}_all_strings_FT_SF.pdf",
    }
    pngs = {os.path.basename(p) for p in paths if p.endswith(".png")}
    assert len(pngs) == 4 and f"{PERIOD}_{RUN}_string1_FT_failure.png" in pngs


def test_ft_string_figure_content():
    cycle = itertools.cycle(plt.cm.tab20.colors)
    fig = summary._ft_string_figure(
        PERIOD, RUN, 1, _ft_rates(), 100.0, "20260101T000000Z", cycle
    )
    ax = fig.axes[0]
    # the percent axis is a child SecondaryAxis, not a top-level figure axes
    secondary = [c for c in ax.child_axes if c.get_ylabel()]
    assert [c.get_ylabel() for c in secondary] == ["FT failure fraction (%)"]
    legend = ax.get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["V01234A", "V05678B"]
    assert legend.get_title().get_text() == "Last cycle: 20260101T000000Z"
    assert ax.get_ylabel() == "Normalized FT failure rate (mHz/kg)"
    assert fig._suptitle.get_text() == f"{PERIOD} - {RUN} - string 1"
    plt.close(fig)


def test_event_rate_qc_pdf_name_and_figure(tmp_path):
    times = pd.to_datetime(np.linspace(1.7e9, 1.7e9 + 6 * 3600, 720), unit="s")
    monitoring.write_event_rate_qc(
        str(tmp_path),
        PERIOD,
        RUN,
        {
            "All events": times,
            "Failing QC": times[:300],
            "Surviving QC": times[300:],
        },
        on_mass=140.0,
    )
    paths = summary.plot_event_rate_qc(str(tmp_path), PERIOD, RUN)
    expected = os.path.join(
        str(tmp_path), PERIOD, RUN, "mtg/pdf", f"{PERIOD}_{RUN}_event_rate_qc.pdf"
    )
    assert paths == [expected] and os.path.exists(expected)

    frame = pd.read_hdf(
        monitoring.period_contract_path(str(tmp_path), PERIOD),
        key=f"event_rate_qc/{RUN}",
    )
    fig = summary._event_rate_figure(frame, None)
    ax = fig.axes[0]
    legend = ax.get_legend()
    # legacy label order, absent series (delayed discharges) skipped
    assert [t.get_text() for t in legend.get_texts()] == [
        "All events",
        "Failing QC",
        "Surviving QC",
    ]
    assert legend.get_title().get_text() == "ON mass = 140.0 kg"
    assert ax.patches[0].get_edgecolor() == to_rgba("dimgrey")
    assert ax.get_ylabel() == "Hourly rate normalized by ON mass (mHz/kg)"
    plt.close(fig)

    fig = summary._event_rate_figure(frame, "20260101T000000Z")
    title = fig.axes[0].get_legend().get_title().get_text()
    assert title == "Last cycle: 20260101T000000Z\nON mass = 140.0 kg"
    plt.close(fig)


def test_detector_summary_cal_metric_by_key_or_title(tmp_path):
    monitoring.write_detector_summary(
        str(tmp_path), PERIOD, RUN, "FEP_gain_stab", _summary_frame(), data_type="cal"
    )
    for metric in ["FEP_variation", "FEP_gain_stab"]:
        paths = summary.plot_detector_summary(
            str(tmp_path), PERIOD, RUN, metric, data_type="cal"
        )
        assert [os.path.basename(p) for p in paths] == [
            f"{PERIOD}_{RUN}_FEP_gain_stab.pdf"
        ]
        assert paths[0].endswith(
            f"{PERIOD}/{RUN}/mtg/pdf/{PERIOD}_{RUN}_FEP_gain_stab.pdf"
        )


def test_detector_summary_figure_content():
    info = utils.MTG_PLOT_INFO["FEP_variation"]
    fig = summary._detector_summary_figure(PERIOD, RUN, _summary_frame(), info, None)
    ax = fig.axes[0]
    ticklabels = ax.get_xticklabels()
    assert [t.get_text() for t in ticklabels] == ["V01234A", "V05678B"]
    assert ticklabels[1].get_color() == "red"  # usability "off"
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    for expected in ["Mean", "Min/Max", "Usability: off", "Usability: ac"]:
        assert expected in labels
    assert ax.get_ylim() == (-6.0, 6.0)
    assert ax.get_ylabel() == "FEP gain variation [keV]"
    assert ax.get_title() == f"{PERIOD} {RUN}"
    plt.close(fig)


def test_missing_keys_return_empty(tmp_path):
    assert summary.plot_event_rate_qc(str(tmp_path), PERIOD, RUN) == []
    assert summary.plot_ft_summary(str(tmp_path), PERIOD, RUN) == []
    assert (
        summary.plot_detector_summary(str(tmp_path), PERIOD, RUN, "pulser_stab") == []
    )
