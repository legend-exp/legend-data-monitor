import pytest

from legend_data_monitor.monitoring import get_energy_key


def test_runcal_present():
    ecal_results = {
        "cuspEmax_ctc_runcal": {"det1": 123, "det2": 456},
        "cuspEmax_ctc_cal": {"det1": 999},
    }
    result = get_energy_key(ecal_results)
    assert result == {"det1": 123, "det2": 456}


def test_cal_only():
    ecal_results = {"cuspEmax_ctc_cal": {"det1": 999}}
    result = get_energy_key(ecal_results)
    assert result == {"det1": 999}


def test_no_keys():
    ecal_results = {"other_key": 42}
    result = get_energy_key(ecal_results)
    assert result == {}


@pytest.mark.parametrize(
    "ecal_results, expected",
    [
        ({"cuspEmax_ctc_runcal": {"a": 1}}, {"a": 1}),
        ({"cuspEmax_ctc_cal": {"b": 2}}, {"b": 2}),
        ({}, {}),
    ],
)
def test_parametrized(ecal_results, expected):
    assert get_energy_key(ecal_results) == expected


def test_estimator_order_comes_from_settings(monkeypatch):
    """The candidates are configured, and the first present one wins."""
    from legend_data_monitor import utils
    from legend_data_monitor.monitoring import find_energy_key

    both = {"cuspEmax_ctc_runcal": {"a": 1}, "cuspEmax_ctc_cal": {"a": 2}}
    assert find_energy_key(both) == ("cuspEmax_ctc_runcal", {"a": 1})

    flipped = dict(utils.EXPERIMENT["energy"], estimators=["cuspEmax_ctc_cal"])
    monkeypatch.setitem(utils.EXPERIMENT, "energy", flipped)
    assert find_energy_key(both) == ("cuspEmax_ctc_cal", {"a": 2})


def test_missing_estimator_is_not_an_error():
    from legend_data_monitor.monitoring import find_energy_key, get_energy_key

    assert find_energy_key({"zacEmax_ctc_cal": {}}) == (None, {})
    assert get_energy_key({}) == {}


def test_uncalibrated_variable_strips_the_suffix():
    from legend_data_monitor.monitoring import uncalibrated_variable

    assert uncalibrated_variable("cuspEmax_ctc_cal") == "cuspEmax_ctc"
    assert uncalibrated_variable("cuspEmax_ctc_runcal") == "cuspEmax_ctc"
    assert uncalibrated_variable("zacEmax") == "zacEmax"
