from unittest.mock import patch

import numpy as np
import pytest

from legend_data_monitor.monitoring import get_calib_pars


@pytest.fixture
def mock_helpers():
    with (
        patch("legend_data_monitor.monitoring.add_calibration_runs") as mock_add_calib,
        patch(
            "legend_data_monitor.monitoring.utils.get_tiers_pars_folders"
        ) as mock_get_tiers,
        patch(
            "legend_data_monitor.monitoring.get_calib_data_dict"
        ) as mock_get_calib_data,
    ):

        mock_add_calib.side_effect = lambda period, run_list: run_list
        mock_get_tiers.return_value = (["tier1", "tier2"], ["pars1", "pars2"])

        def dummy_get_calib_data(
            calib_data, channel_info, tiers, pars, period, run, tier, key_result, fit
        ):
            calib_data["fep"].append(2614 + run)
            calib_data["fep_err"].append(0.1)
            calib_data["cal_const"].append(1.0 + run * 0.01)
            calib_data["cal_const_err"].append(0.01)
            calib_data["run_start"].append(run)
            calib_data["run_end"].append(run + 1)
            calib_data["res"].append(0.05)
            calib_data["res_quad"].append(0.001)
            return calib_data

        mock_get_calib_data.side_effect = dummy_get_calib_data

        yield mock_add_calib, mock_get_tiers, mock_get_calib_data


def test_get_calib_pars(mock_helpers):
    period = "p1"
    run_list = [1, 2, 3]
    channel_info = [100000, "DET1"]
    partition = False
    escale = 2039

    result = get_calib_pars(
        path="/fake/path",
        period=period,
        run_list=run_list,
        channel_info=channel_info,
        partition=partition,
        escale=escale,
        fit="linear",
    )

    # check returned keys
    expected_keys = [
        "fep",
        "fep_err",
        "cal_const",
        "cal_const_err",
        "run_start",
        "run_end",
        "res",
        "res_quad",
        "cal_const_diff",
        "fep_diff",  # new keys
    ]
    for key in expected_keys:
        assert key in result

    # check that arrays have correct length = number of inspected calib runs
    for key in [
        "fep",
        "fep_err",
        "cal_const",
        "cal_const_err",
        "run_start",
        "run_end",
        "res",
        "res_quad",
        "cal_const_diff",
        "fep_diff",
    ]:
        assert len(result[key]) == len(run_list)

    # check values
    assert np.isclose(result["fep"][0], 2615)
    assert np.isclose(result["fep_err"][0], 0.1)
    assert np.isclose(result["cal_const"][0], 1.01)
    assert np.isclose(result["cal_const_err"][0], 0.01)
    assert np.isclose(result["run_start"][0], 1)
    assert np.isclose(result["run_end"][0], 2)
    assert np.isclose(result["res"][0], 0.05)
    assert np.isclose(result["res_quad"][0], 0.001)
    assert np.isclose(result["cal_const_diff"][0], 0.0)
    assert np.isclose(result["fep_diff"][0], 0.0)
    assert np.isclose(result["cal_const_diff"][1], (1.02 - 1.01) / 1.01 * escale)
    assert np.isclose(result["fep_diff"][1], (2616 - 2615) / 2615 * escale)
