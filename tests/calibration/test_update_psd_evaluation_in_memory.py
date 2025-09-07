import numpy as np

from legend_data_monitor.calibration import update_psd_evaluation_in_memory


def test_update_psd_evaluation_in_memory_creates_keys():
    data = {}
    det_name = "DET1"
    value = True

    update_psd_evaluation_in_memory(data, det_name, value)

    assert det_name in data
    assert "cal" in data[det_name]
    assert data[det_name]["cal"]["PSD"] is True


def test_update_psd_evaluation_in_memory_overwrites():
    data = {"DET1": {"cal": {"PSD": False}}}
    det_name = "DET1"
    value = np.nan

    update_psd_evaluation_in_memory(data, det_name, value)

    assert np.isnan(data[det_name]["cal"]["PSD"])


def test_update_psd_evaluation_in_memory_partial_existing():
    data = {"DET1": {}}
    det_name = "DET1"
    value = True

    update_psd_evaluation_in_memory(data, det_name, value)

    assert "cal" in data[det_name]
    assert data[det_name]["cal"]["PSD"] is True
