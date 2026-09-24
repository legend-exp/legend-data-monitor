"""SiPM handling in AnalysisData and the per-barrel plot structure."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from legend_data_monitor import analysis_data, plotting


def _spms_frame():
    times = pd.date_range("2026-07-01", periods=40, freq="1min", tz="UTC")
    rows = []
    for rawid, name, fiber, position in [
        (1064000, "S060", "IB015016", "top"),
        (1064001, "S061", "IB015016", "bottom"),
        (1056002, "S055", "OB005006", "top"),
    ]:
        for t in times:
            rows.append(
                {
                    "channel": rawid,
                    "datetime": t,
                    "name": name,
                    "location": fiber,
                    "position": position,
                    "barrel": fiber[:2],
                    "cc4_id": None,
                    "cc4_channel": None,
                    "status": "on",
                    "flag_pulser": False,
                    "flag_fc_bsln": True,
                    "flag_muon": False,
                    "wf_mode": 3747.0 + (rawid % 10),
                }
            )
    df = pd.DataFrame(rows)
    for col in ["name", "location", "position", "barrel"]:
        df[col] = df[col].astype("category")
    return df


def _analysis(df):
    return analysis_data.AnalysisData(
        df,
        selection={
            "parameters": "wf_mode",
            "event_type": "FCbsln",
            "variation": True,
            "saving": None,
            "plt_path": "",
            "path": "",
            "version": "",
            "cuts": [],
            "time_window": "10min",
            "resampled": "only",
            "plot_style": "vs time",
            "plot_structure": "per barrel",
        },
    )


def test_spms_detected_by_barrel_and_gets_channel_mean():
    data = _analysis(_spms_frame())
    assert data.is_spms() and not data.is_geds()
    assert data.data["wf_mode_mean"].notna().all()
    np.testing.assert_allclose(data.data["wf_mode_var"], 0.0, atol=1e-6)

    geds = _spms_frame()
    geds["barrel"] = np.nan
    assert not _analysis(geds).is_spms()


def test_plot_per_barrel_and_position(tmp_path):
    data = _analysis(_spms_frame())
    plot_info = {
        "subsystem": "spms",
        "title": "Baseline mode",
        "plot_style": "vs time",
        "parameters": ["wf_mode"],
        "parameter": "wf_mode",
        "label": "Baseline (mode)",
        "unit": "%",
        "unit_label": "%",
        "param_mean": None,
        "locname": "fiber",
        "time_window": "10min",
        "resampled": "only",
        "range": [None, None],
        "std": False,
        "plot_structure": "per barrel",
        "limits": [None, None],
        "event_type": "FCbsln",
    }
    pdf_path = tmp_path / "spms.pdf"
    with PdfPages(pdf_path) as pdf:
        plotting.plot_per_barrel_and_position(data.data, plot_info, pdf)
    # IB top, IB bottom, OB top: one page per (barrel, position)
    assert pdf_path.stat().st_size > 0


def test_plot_settings_reject_cross_subsystem_structures():
    from legend_data_monitor import utils

    def conf(subsys, structure):
        return {
            "subsystems": {
                subsys: {
                    "p": {
                        "parameters": "wf_mode",
                        "event_type": "FCbsln",
                        "plot_structure": structure,
                        "plot_style": "vs time",
                        "time_window": "10min",
                    }
                }
            }
        }

    assert utils.check_plot_settings(conf("spms", "per barrel"))
    assert utils.check_plot_settings(conf("geds", "per string"))
    assert utils.check_plot_settings(conf("spms", "per channel"))
    assert not utils.check_plot_settings(conf("geds", "per barrel"))
    assert not utils.check_plot_settings(conf("spms", "array"))
