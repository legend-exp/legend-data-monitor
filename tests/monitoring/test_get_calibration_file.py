import json
import os
import tempfile

import pytest
import yaml

from legend_data_monitor.monitoring import get_calibration_file


def test_get_calibration_file_json():
    data = {"param1": 123}
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "calib.json")
        with open(filepath, "w") as f:
            json.dump(data, f)

        returned_data = get_calibration_file(tmpdir)
        assert returned_data == data


def test_get_calibration_file_yaml():
    data = {"param1": 456}
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "calib.yaml")
        with open(filepath, "w") as f:
            yaml.dump(data, f)

        returned_data = get_calibration_file(tmpdir)
        assert returned_data == data


# shouldn't happen - worth checking
def test_get_calibration_file_priority_json_over_yaml():
    json_data = {"param": "json"}
    yaml_data = {"param": "yaml"}
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "calib.json")
        yaml_path = os.path.join(tmpdir, "calib.yaml")
        with open(json_path, "w") as f:
            json.dump(json_data, f)
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f)

        returned_data = get_calibration_file(tmpdir)
        assert returned_data == json_data


def test_get_calibration_file_not_found():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            get_calibration_file(tmpdir)
