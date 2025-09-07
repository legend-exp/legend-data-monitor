import numpy as np

import legend_data_monitor
from legend_data_monitor.monitoring import evaluate_fep_cal


def fake_get_energy_key(operations):
    return operations["cuspEmax_ctc_cal"]


legend_data_monitor.get_energy_key = fake_get_energy_key


def test_channel_not_in_dict():
    pars = {}
    result = evaluate_fep_cal(pars, "ch1", 1000, 1.0)
    assert result == (np.nan, np.nan)


def test_valid_linear_expression():
    pars = {
        "ch1": {
            "pars": {
                "operations": {
                    "cuspEmax_ctc_cal": {
                        "expression": "a * cuspEmax_ctc + b",
                        "parameters": {"a": 2.0, "b": 5.0},
                    }
                }
            }
        }
    }
    fep_peak_pos = 1000.0
    fep_peak_pos_err = 1.0
    fep_cal, fep_cal_err = evaluate_fep_cal(pars, "ch1", fep_peak_pos, fep_peak_pos_err)

    assert fep_cal == float(2 * 1000 + 5)
    assert fep_cal_err == float(2 * 1 + 5)


def test_expression_with_offset_only():
    pars = {
        "ch1": {
            "pars": {
                "operations": {
                    "cuspEmax_ctc_cal": {
                        "expression": "cuspEmax_ctc + a",
                        "parameters": {"a": 10.0},
                    }
                }
            }
        }
    }
    fep_cal, fep_cal_err = evaluate_fep_cal(pars, "ch1", 200.0, 0.5)
    assert fep_cal == float(200 + 10)
    assert fep_cal_err == float(0.5 + 10)
