import pytest

import legend_data_monitor.monitoring as monitoring
from legend_data_monitor.monitoring import add_calibration_runs

CALIB_RUNS = {"p1": [101, 102], "p2": [201], "p3": [301, 302, 303]}


@pytest.fixture(autouse=True)
def patch_calib_runs(monkeypatch):
    monkeypatch.setattr(monitoring, "CALIB_RUNS", CALIB_RUNS)


def test_single_period_list_input():
    runs = [1, 2, 3]
    updated = add_calibration_runs("p1", runs)
    assert updated == [1, 2, 3, 101, 102]


def test_period_not_in_calib():
    runs = [1, 2]
    updated = add_calibration_runs("px", runs)
    assert updated == [1, 2]  # unchanged


def test_single_period_dict_input():
    runs_dict = {"p1": [10], "p2": [20]}
    updated = add_calibration_runs("p1", runs_dict)
    assert updated["p1"] == [10, 101, 102]
    assert updated["p2"] == [20]


def test_list_of_periods_dict_input():
    runs_dict = {"p1": [1], "p2": [2], "p3": [3]}
    updated = add_calibration_runs(["p1", "p3"], runs_dict)
    assert updated["p1"] == [1, 101, 102]
    assert updated["p2"] == [2]
    assert updated["p3"] == [3, 301, 302, 303]


def test_period_not_in_dict():
    runs_dict = {"p1": [1]}
    updated = add_calibration_runs("p2", runs_dict)
    assert updated == {"p1": [1]}
