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
