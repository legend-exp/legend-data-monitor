import os
import tempfile

import yaml

from legend_data_monitor.utils import load_yaml_or_default


def test_load_yaml_or_default():
    detectors = {"ged1": {}, "ged2": {}}

    # file does not exist
    non_existing_path = "non_existing_file.yaml"
    result = load_yaml_or_default(non_existing_path, detectors)
    for ged in detectors:
        assert ged in result
        assert result[ged]["cal"]["npeak"] is None
        assert result[ged]["phy"]["pulser_stab"] is None

    # file exists but empty
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmpfile:
        empty_path = tmpfile.name
    try:
        result = load_yaml_or_default(empty_path, detectors)
        for ged in detectors:
            assert ged in result
            assert result[ged]["cal"]["npeak"] is None
    finally:
        os.remove(empty_path)

    # file exists
    content = {"ged1": {"cal": {"npeak": 123}, "phy": {}}}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmpfile:
        yaml.dump(content, tmpfile)
        tmpfile_path = tmpfile.name

    try:
        result = load_yaml_or_default(tmpfile_path, detectors)
        assert result["ged1"]["cal"]["npeak"] == 123
    finally:
        os.remove(tmpfile_path)
