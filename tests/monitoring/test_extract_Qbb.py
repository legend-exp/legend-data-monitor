import numpy as np

from legend_data_monitor.monitoring import extract_resolution_at_q_bb


def test_channel_not_in_dict():
    pars = {}
    result = extract_resolution_at_q_bb(pars, "ch1", "ecal")
    assert result == (np.nan, np.nan)


def test_no_cuspemax_ctc_cal():
    pars = {"ch1": {"results": {"ecal": {}}}}
    result = extract_resolution_at_q_bb(pars, "ch1", "ecal")
    assert result == (np.nan, np.nan)


def test_no_qbb_key():
    pars = {
        "ch1": {
            "results": {
                "ecal": {
                    "cuspEmax_ctc_cal": {
                        "eres_linear": {"fake_key": 1.23},
                        "eres_quadratic": {"fake_key": 2.34},
                    }
                }
            }
        }
    }
    result = extract_resolution_at_q_bb(pars, "ch1", "ecal")
    assert result == (np.nan, np.nan)


def test_linear_fit():
    pars = {
        "ch1": {
            "results": {
                "ecal": {
                    "cuspEmax_ctc_cal": {
                        "eres_linear": {"Qbb_fwhm_in_keV": 5.67},
                        "eres_quadratic": {"Qbb_fwhm_in_keV": 8.9},
                    }
                }
            }
        }
    }
    result = extract_resolution_at_q_bb(pars, "ch1", "ecal", fit="linear")
    assert result == (5.67, np.nan)


def test_quadratic_fit():
    pars = {
        "ch1": {
            "results": {
                "ecal": {
                    "cuspEmax_ctc_cal": {
                        "eres_linear": {"Qbb_fwhm_in_keV": 5.67},
                        "eres_quadratic": {"Qbb_fwhm_in_keV": 8.9},
                    }
                }
            }
        }
    }
    result = extract_resolution_at_q_bb(pars, "ch1", "ecal", fit="quadratic")
    assert result == (5.67, 8.9)
