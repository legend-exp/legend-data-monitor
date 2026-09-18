"""Tests for the auto-giorgio-facing contracts: log tree, error blocks, issues, task isolation."""

import json
import re

import numpy as np
import pandas as pd
import pytest

from legend_data_monitor import errors, issues, logs, tasks

# -------------------------------------------------------------------------
# error blocks / log tree
# -------------------------------------------------------------------------


def test_format_error_block_is_parseable():
    try:
        raise errors.DataError("something broke")
    except errors.DataError as exc:
        block = logs.format_error_block("build_monitoring_hdf", "p19", "r001", exc)

    lines = block.splitlines()
    assert lines[0] == "ERROR in task build_monitoring_hdf (period=p19, run=r001):"
    assert lines[-1] == "END ERROR"
    assert "Traceback (most recent call last):" in block
    assert "DataError: something broke" in block


def test_log_tree_layout_and_task_isolation(tmp_path):
    log_root = logs.log_tree_root(str(tmp_path), invocation_key="20260101T000000Z")
    assert log_root.endswith("generated/tmp/log/20260101T000000Z")

    calls = []

    def ok_task(logger=None):
        calls.append("ok")
        logger.info("hello")

    def failing_task(logger=None):
        calls.append("fail")
        raise errors.DataError("boom")

    task_list = [
        tasks.Task("first", failing_task, "p19", "r001"),
        tasks.Task("second", ok_task, "p19", "r001"),
    ]
    results, exit_code = tasks.run_tasks(task_list, log_root)

    # isolation: the failing first task did not stop the second
    assert calls == ["fail", "ok"]
    assert exit_code == tasks.EXIT_TASK_FAILED
    assert [r.ok for r in results] == [False, True]

    task_log = tmp_path / (
        "generated/tmp/log/20260101T000000Z/first/first-p19-r001.log"
    )
    content = task_log.read_text()
    assert "ERROR in task first (period=p19, run=r001):" in content
    assert "END ERROR" in content
    assert "DataError: boom" in content

    orch = (
        tmp_path / "generated/tmp/log/20260101T000000Z/orchestrator.log"
    ).read_text()
    assert re.search(r"FAILED task=first .*DataError: boom", orch)
    assert re.search(r"END task=second .*status=ok", orch)


def test_run_tasks_all_ok_exit_code(tmp_path):
    log_root = logs.log_tree_root(str(tmp_path))
    results, exit_code = tasks.run_tasks(
        [tasks.Task("t", lambda logger=None: None, "p", "r")], log_root
    )
    assert exit_code == tasks.EXIT_OK


# -------------------------------------------------------------------------
# issues
# -------------------------------------------------------------------------


def _issue(**kwargs):
    base = dict(
        detector="V02160A",
        metric="gain_var",
        severity="alert",
        period="p15",
        run="r002",
        datatype="phy",
    )
    base.update(kwargs)
    return issues.Issue(**base)


def test_issue_id_is_stable_dedup_key():
    assert _issue().issue_id == "p15-r002-phy-V02160A-gain_var"
    # identical inputs -> identical id (cluster/dedup key)
    assert _issue().issue_id == _issue().issue_id


def test_issue_jsonl_roundtrip(tmp_path):
    issue = _issue(
        observed=0.53,
        threshold=[-0.05, 0.05],
        unit="%",
        excursion=issues.Excursion(
            frac_out=0.42, max_deviation=0.48, longest_s=86400.0, recovered=False
        ),
        first_seen_run="r001",
        raw_ref={"tier_dir": "/prod/tier/dsp/phy/p15/r002", "channel": "ch1104000"},
    )
    path = issues.issues_file_path(str(tmp_path), "p15", "r002", "phy")
    issues.write_issues(path, [issue])

    assert path.endswith("generated/mon/issues/p15/r002/l200-p15-r002-phy-issues.jsonl")
    with open(path) as f:
        lines = f.read().splitlines()
    assert len(lines) == 1
    loaded = json.loads(lines[0])
    assert loaded["issue_id"] == "p15-r002-phy-V02160A-gain_var"
    assert loaded["schema"] == 1
    assert loaded["excursion"]["recovered"] is False
    assert loaded["raw_ref"]["channel"] == "ch1104000"


def test_issue_plots_absolute_paths(tmp_path):
    issue = _issue(plots=["relative/fig.png"])
    path = issues.issues_file_path(str(tmp_path), "p15", "r002", "phy")
    issues.write_issues(path, [issue])
    with open(path) as f:
        loaded = json.loads(f.read().splitlines()[0])
    assert all(p.startswith("/") for p in loaded["plots"])


def test_issue_log_block_format():
    issue = _issue(
        observed=0.53,
        threshold=[-0.05, 0.05],
        unit="%",
        excursion=issues.Excursion(
            frac_out=0.42, max_deviation=0.48, longest_s=86400.0, recovered=False
        ),
        plots=["/out/figs/st2/V02160A_gain.png"],
    )
    block = issues.format_issue_block(issue, "/out/issues.jsonl")
    lines = block.splitlines()
    assert lines[0].startswith(
        "ISSUE detector=V02160A metric=gain_var severity=alert "
        "(period=p15, run=r002, datatype=phy):"
    )
    assert lines[-1] == "END ISSUE"
    assert any("payload=/out/issues.jsonl" in line for line in lines)
    assert any("plot=/out/figs/st2/V02160A_gain.png" in line for line in lines)


# -------------------------------------------------------------------------
# excursion evaluation
# -------------------------------------------------------------------------


def _series(values, freq="1min"):
    idx = pd.date_range("2026-07-01", periods=len(values), freq=freq, tz="UTC")
    return pd.Series(values, index=idx)


def test_excursion_none_when_in_range():
    assert issues.evaluate_excursion(_series([0.0, 0.01, -0.02]), -0.05, 0.05) is None


def test_excursion_transient_recovers():
    exc = issues.evaluate_excursion(_series([0.0, 0.1, 0.0, 0.0]), -0.05, 0.05)
    assert exc is not None
    assert exc.recovered is True
    assert exc.frac_out == pytest.approx(0.25)
    assert exc.max_deviation == pytest.approx(0.05)


def test_excursion_persistent_not_recovered():
    exc = issues.evaluate_excursion(_series([0.0, 0.1, 0.2, 0.3]), -0.05, 0.05)
    assert exc.recovered is False
    assert exc.frac_out == pytest.approx(0.75)
    assert exc.max_deviation == pytest.approx(0.25)
    # 3 samples at 1min spacing -> 120 s contiguous stretch
    assert exc.longest_s == pytest.approx(120.0)


def test_excursion_one_sided_threshold():
    exc = issues.evaluate_excursion(_series([1.0, 5.0]), None, 4.0)
    assert exc is not None
    assert exc.max_deviation == pytest.approx(1.0)
    assert issues.evaluate_excursion(_series([1.0, 5.0]), 0.0, None) is None


# -------------------------------------------------------------------------
# severity grading + metric details
#
# A failed threshold alone is not newsworthy: the cal bands are a few times
# the mean fit error, so on real data ~30% of the array trips one per run.
# -------------------------------------------------------------------------


def test_severity_alert_only_for_sustained_unrecovered_excursion():
    sustained = issues.evaluate_excursion(_series([0.0, 0.1, 0.2, 0.3]), -0.05, 0.05)
    assert issues.classify_severity(0.3, [-0.05, 0.05], sustained) == "alert"


def test_severity_warning_for_transient_and_recovered():
    transient = issues.evaluate_excursion(_series([0.0, 0.1, 0.0, 0.0]), -0.05, 0.05)
    assert issues.classify_severity(0.1, [-0.05, 0.05], transient) == "warning"


def test_severity_warning_for_brief_excursion():
    values = [0.0] * 50 + [0.2]
    brief = issues.evaluate_excursion(_series(values), -0.05, 0.05)
    # ~2% of samples out of range -> below the 5% floor
    assert issues.classify_severity(0.2, [-0.05, 0.05], brief) == "warning"


def test_severity_warning_when_no_magnitudes_available():
    assert issues.classify_severity(None, None, None) == "warning"


def test_metric_details_roundtrip_and_are_popped():
    issues.clear_details()
    issues.record_detail(
        "p15", "r002", "phy", "V02160A", "gain_var", observed=0.53, unit="%"
    )
    detail = issues.pop_detail("p15", "r002", "phy", "V02160A", "gain_var")
    assert detail == {"observed": 0.53, "unit": "%"}
    # popped once: a second read cannot re-attach stale numbers to a later run
    assert issues.pop_detail("p15", "r002", "phy", "V02160A", "gain_var") == {}


def test_metric_details_are_keyed_per_run():
    issues.clear_details()
    issues.record_detail("p15", "r002", "phy", "V02160A", "gain_var", observed=1.0)
    assert issues.pop_detail("p15", "r003", "phy", "V02160A", "gain_var") == {}
    issues.clear_details()


def test_severity_single_value_grades_on_distance_past_band():
    # cal metrics have no time series: a value just past a 3-sigma band is a
    # warning, one far outside it is an alert
    band = [2.9, 3.1]
    assert issues.classify_severity(3.12, band, None) == "warning"
    assert issues.classify_severity(3.5, band, None) == "alert"
    # improvements (below the band) grade the same way by distance
    assert issues.classify_severity(2.88, band, None) == "warning"
    assert issues.classify_severity(2.4, band, None) == "alert"


def test_severity_single_value_needs_a_finite_band():
    assert issues.classify_severity(5.0, [None, 3.0], None) == "warning"
    assert issues.classify_severity(5.0, [3.0, 3.0], None) == "warning"


def test_check_threshold_details_reach_the_issue_record(tmp_path):
    """The evaluator's magnitudes must survive into the JSONL, not just the verdict."""
    import yaml

    from legend_data_monitor import utils

    issues.clear_details()
    idx = pd.date_range("2026-08-01", periods=6, freq="10min", tz="UTC")
    # sustained, unrecovered departure above the +2 threshold
    series = pd.Series([0.0, 0.5, 3.0, 3.5, 4.0, 4.5], index=idx)
    output = {"V02160A": {"cal": {"fwhm_ok": True}, "phy": {}}}
    utils.check_threshold(
        series,
        "V02160A",
        None,
        [idx[0]],
        [-2.0, 2.0],
        "pulser_stab",
        output,
        period="p22",
        run="r012",
    )
    assert output["V02160A"]["phy"]["pulser_stab"] is False

    run_dir = tmp_path / "generated" / "plt" / "hit" / "phy" / "p22" / "r012"
    run_dir.mkdir(parents=True)
    summary = run_dir / "l200-p22-r012-qcp_summary.yaml"
    summary.write_text(yaml.dump(output))

    found = utils.check_cal_phy_thresholds(
        str(run_dir.parents[1]),
        "p22",
        "r012",
        "phy",
        ["V02160A"],
        detector_info={"V02160A": {"daq_rawid": 1104000, "string": 4, "position": 2}},
    )
    (issue,) = (i for i in found if i.metric == "pulser_stab")
    assert issue.observed == pytest.approx(4.5)
    assert issue.threshold == [-2.0, 2.0]
    assert issue.excursion.recovered is False
    assert issue.excursion.frac_out == pytest.approx(4 / 6)
    assert issue.severity == "alert"
    assert issue.rawid == 1104000 and issue.string == 4 and issue.position == 2
    assert issue.window[0].startswith("2026-08-01")
    issues.clear_details()


def _emit(tmp_path, output, key, detector="V02160A"):
    """Write the qcp yaml and run the emitter for one datatype key."""
    import yaml

    from legend_data_monitor import utils

    run_dir = tmp_path / "generated" / "plt" / "hit" / "phy" / "p22" / "r012"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "l200-p22-r012-qcp_summary.yaml").write_text(yaml.dump(output))
    return utils.check_cal_phy_thresholds(
        str(run_dir.parents[1]),
        "p22",
        "r012",
        key,
        [detector],
        detector_info={detector: {"daq_rawid": 1104000, "string": 4, "position": 2}},
    )


def test_check_threshold_reports_unit_and_the_worst_side(tmp_path):
    """A two-sided band failing LOW must report the low sample, with a unit."""
    from legend_data_monitor import utils

    issues.clear_details()
    idx = pd.date_range("2026-08-01", periods=4, freq="1h", tz="UTC")
    series = pd.Series([0.0, -12.0, 1.0, 2.0], index=idx)  # baseln_stab band is +-10 %
    output = {"V02160A": {"cal": {"fwhm_ok": True}, "phy": {}}}
    utils.check_threshold(
        series,
        "V02160A",
        None,
        [idx[0]],
        [-10, 10],
        "baseln_stab",
        output,
        period="p22",
        run="r012",
    )
    (issue,) = (i for i in _emit(tmp_path, output, "phy") if i.metric == "baseln_stab")
    assert issue.observed == pytest.approx(-12.0)
    assert issue.unit == "%"
    assert issue.excursion.frac_out == pytest.approx(1 / 4)
    assert issue.severity == "warning"  # brief excursion
    issues.clear_details()


def test_fep_detail_carries_an_excursion_in_seconds(tmp_path):
    from legend_data_monitor import calibration

    issues.clear_details()
    # six 600 s bins, drift leaves the +-2 keV band for the last three
    computed = {
        "stats": pd.DataFrame({"time": np.arange(6) * 600.0}),
        "drift": pd.Series([0.1, 0.2, 0.3, 2.5, 3.0, 3.5]),
    }
    calibration.record_fep_detail("p22", "r012", "cal", "V02160A", computed, 1.7e9)
    output = {"V02160A": {"cal": {"FEP_gain_stab": False}, "phy": {}}}
    (issue,) = (
        i for i in _emit(tmp_path, output, "cal") if i.metric == "FEP_gain_stab"
    )
    assert issue.observed == pytest.approx(3.5)
    assert issue.threshold == [-2.0, 2.0] and issue.unit == "keV"
    assert issue.excursion.longest_s == pytest.approx(
        1200.0
    )  # 3 bins = 2 gaps of 600 s
    assert issue.excursion.recovered is False
    assert issue.window[0].startswith("2023-11-14")  # anchored on the first event
    assert issue.data_ref["key"] == "fep_gain_stab/r012"
    issues.clear_details()


def test_psd_detail_has_no_excursion_but_a_magnitude(tmp_path):
    from legend_data_monitor import calibration

    issues.clear_details()
    runs = ["r010", "r011", "r012"]
    eval_result = {
        "status": False,
        "slow_shifts": [0.0, 0.1, 0.9],
        "sudden_shifts": [0.0, 0.05, 0.1],
        "slow_shift_fail_runs": ["r012"],
        "sudden_shift_fail_runs": [],
    }
    calibration.record_psd_detail("p22", "r012", "V02160A", runs, eval_result)
    output = {"V02160A": {"cal": {"AoE_stab": False}, "phy": {}}}
    (issue,) = (i for i in _emit(tmp_path, output, "cal") if i.metric == "AoE_stab")
    assert issue.observed == pytest.approx(0.9)
    assert issue.threshold == [-0.5, 0.5] and issue.unit == "sigma"
    assert issue.excursion is None
    assert issue.severity == "warning"  # 0.9 is inside one band-width past +0.5
    assert issue.data_ref["key"] == "psd_stability/r012/V02160A"
    issues.clear_details()


def test_qc_rate_detail_reaches_the_record_with_excursion(tmp_path):
    """discharge_rate used to never reach the emitter at all."""
    issues.clear_details()
    idx = pd.date_range("2026-08-01", periods=5, freq="1h", tz="UTC")
    hourly = pd.Series([1.0, 8.0, 9.0, 9.0, 8.5], index=idx)  # limit 5 mHz
    issues.record_detail(
        "p22",
        "r012",
        "phy",
        "V02160A",
        "discharge_rate",
        observed=7.1,
        threshold=[None, 5],
        unit="mHz",
        window=[str(idx[0]), str(idx[-1])],
        excursion=issues.evaluate_excursion(hourly, None, 5),
    )
    output = {"V02160A": {"cal": {}, "phy": {"discharge_rate": False}}}
    (issue,) = (
        i for i in _emit(tmp_path, output, "phy") if i.metric == "discharge_rate"
    )
    assert issue.unit == "mHz" and issue.observed == pytest.approx(7.1)
    assert issue.excursion.frac_out == pytest.approx(4 / 5)
    assert issue.severity == "alert"  # sustained and not recovered
    assert issue.data_ref["key"] == "qc_rate_series/IsDischarge/r012"
    issues.clear_details()
