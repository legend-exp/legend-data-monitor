import numpy as np

import legend_data_monitor
from legend_data_monitor.monitoring import extract_fep_peak


def fake_get_energy_key(ecal_results):
    return ecal_results


legend_data_monitor.get_energy_key = fake_get_energy_key


def test_channel_not_in_dict():
    pars = {}
    result = extract_fep_peak(pars, "ch1")
    assert result == (np.nan, np.nan, np.nan, np.nan)

    pars = {"ch2": {"results": {"ecal": {}}}}
    result = extract_fep_peak(pars, "ch1")
    assert result == (np.nan, np.nan, np.nan, np.nan)


def test_no_pk_fits():
    pars = {"ch1": {"results": {"ecal": {}}}}
    result = extract_fep_peak(pars, "ch1")
    assert result == (np.nan, np.nan, np.nan, np.nan)


def test_with_parameters_in_adc():
    pars = {
        "ch1": {
            "results": {
                "ecal": {
                    "cuspEmax_ctc_cal": {
                        "pk_fits": {
                            "2614.2": {
                                "parameters_in_ADC": {"mu": 1234},
                                "uncertainties_in_ADC": {"mu": 1.1},
                            }
                        }
                    }
                }
            }
        }
    }
    result = extract_fep_peak(pars, "ch1")
    fep_peak_pos, fep_peak_pos_err, fep_gain, fep_gain_err = result
    assert fep_peak_pos == 1234
    assert fep_peak_pos_err == 1.1
    assert fep_gain == 1234 / 2614.5
    assert fep_gain_err == 1.1 / 2614.5


def test_with_parameters_with_no_units():
    pars = {
        "ch1": {
            "results": {
                "ecal": {
                    "cuspEmax_ctc_cal": {
                        "pk_fits": {
                            "2614.2": {
                                "parameters": {"mu": 4321},
                                "uncertainties": {"mu": 2.2},
                            }
                        }
                    }
                }
            }
        }
    }
    result = extract_fep_peak(pars, "ch1")
    fep_peak_pos, fep_peak_pos_err, fep_gain, fep_gain_err = result
    assert fep_peak_pos == 4321
    assert fep_peak_pos_err == 2.2
    assert fep_gain == 4321 / 2614.5
    assert fep_gain_err == 2.2 / 2614.5


def test_no_fep_energy_in_range():
    pars = {
        "ch1": {
            "results": {
                "ecal": {
                    "cuspEmax_ctc_cal": {
                        "pk_fits": {
                            "2600": {
                                "parameters_in_ADC": {"mu": 100},
                                "uncertainties_in_ADC": {"mu": 1},
                            }
                        }
                    }
                }
            }
        }
    }
    result = extract_fep_peak(pars, "ch1")
    assert result == (np.nan, np.nan, np.nan, np.nan)
