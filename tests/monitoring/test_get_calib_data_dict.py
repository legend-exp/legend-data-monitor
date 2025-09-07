from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from legend_data_monitor.monitoring import get_calib_data_dict


@pytest.fixture
def calib_data_empty():
    return {
        "fep": [],
        "fep_err": [],
        "cal_const": [],
        "cal_const_err": [],
        "run_start": [],
        "run_end": [],
        "res": [],
        "res_quad": [],
    }


def test_get_calib_data_dict(calib_data_empty):
    fake_pars_dict = {"ch1": {"dummy": "data"}}

    with (
        patch("lgdo.lh5.LH5Store", return_value=MagicMock()),
        patch(
            "legend_data_monitor.monitoring.get_calibration_file",
            return_value=fake_pars_dict,
        ),
        patch(
            "legend_data_monitor.monitoring.extract_fep_peak",
            return_value=(10.0, 0.1, 0.5, 0.01),
        ),
        patch(
            "legend_data_monitor.monitoring.extract_resolution_at_q_bb",
            return_value=(2.5, 3.5),
        ),
        patch(
            "legend_data_monitor.monitoring.evaluate_fep_cal", return_value=(100.0, 1.0)
        ),
        patch(
            "legend_data_monitor.monitoring.get_run_start_end_times",
            return_value=(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")),
        ),
    ):

        calib_data = get_calib_data_dict(
            calib_data_empty,
            channel_info=["ch1", "Channel1"],
            tiers=["tier_hit", "tier_phy"],
            pars=["p0", "p1", "p2", "p3"],
            period="p01",
            run="r001",
            tier="hit",
            key_result="energy_res",
            fit="linear",
        )

    assert calib_data["fep"] == [0.5]
    assert calib_data["fep_err"] == [0.01]
    assert calib_data["cal_const"] == [100.0]
    assert calib_data["cal_const_err"] == [1.0]
    assert calib_data["res"] == [2.5]
    assert calib_data["res_quad"] == [3.5]
    assert calib_data["run_start"][0] == pd.Timestamp("2020-01-01")
    assert calib_data["run_end"][0] == pd.Timestamp("2020-01-02")


def test_channel_name_used_if_not_ch_key(calib_data_empty):
    fake_pars_dict = {"not_ch_key": {"dummy": "data"}}

    with (
        patch("lgdo.lh5.LH5Store", return_value=MagicMock()),
        patch(
            "legend_data_monitor.monitoring.get_calibration_file",
            return_value=fake_pars_dict,
        ),
        patch(
            "legend_data_monitor.monitoring.extract_fep_peak",
            return_value=(np.nan, np.nan, np.nan, np.nan),
        ),
        patch(
            "legend_data_monitor.monitoring.extract_resolution_at_q_bb",
            return_value=(np.nan, np.nan),
        ),
        patch(
            "legend_data_monitor.monitoring.evaluate_fep_cal",
            return_value=(np.nan, np.nan),
        ),
        patch(
            "legend_data_monitor.monitoring.get_run_start_end_times",
            return_value=(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01")),
        ),
    ):

        calib_data = get_calib_data_dict(
            calib_data_empty,
            channel_info=["ch1", "Channel1"],
            tiers=["tier_hit", "tier_phy"],
            pars=["p0", "p1", "p2", "p3"],
            period="p01",
            run="r001",
            tier="hit",
            key_result="energy_res",
            fit="linear",
        )

    # still append nan values
    assert len(calib_data["fep"]) == 1
    assert np.isnan(calib_data["fep"][0])
