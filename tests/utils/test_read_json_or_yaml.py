import json
import os
import tempfile

import pytest
import yaml

from legend_data_monitor import errors
from legend_data_monitor.utils import read_json_or_yaml


def test_read_json_or_yaml():
    # json
    json_content = {"key": "value"}
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".json", delete=False
    ) as json_file:
        json.dump(json_content, json_file)
        json_file_path = json_file.name

    result_json = read_json_or_yaml(json_file_path)
    assert result_json == json_content
    os.remove(json_file_path)

    # yaml
    yaml_content = {"another_key": [1, 2, 3]}
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".yaml", delete=False
    ) as yaml_file:
        yaml.dump(yaml_content, yaml_file)
        yaml_file_path = yaml_file.name

    result_yaml = read_json_or_yaml(yaml_file_path)
    assert result_yaml == yaml_content
    os.remove(yaml_file_path)

    # other stuff
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".txt", delete=False
    ) as txt_file:
        txt_file.write("some text")
        txt_file_path = txt_file.name

    with pytest.raises(errors.ConfigError):
        read_json_or_yaml(txt_file_path)

    os.remove(txt_file_path)
