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
