import json
import os
import tempfile

import pytest
import yaml

from legend_data_monitor.utils import load_config


def test_load_config_dict():
    config = {"key": "value"}
    result = load_config(config)
    assert result == config  # returns the config


def test_load_config_json_string():
    config_dict = {"foo": "bar", "number": 42}
    json_str = json.dumps(config_dict)
    result = load_config(json_str)
    assert result == config_dict  # returns again the config


def test_load_config_yaml_file():
    config_dict = {"name": "test", "enabled": True}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_dict, f)
        temp_path = f.name

    try:
        result = load_config(temp_path)
        assert result == config_dict  # returns the config as read from a yaml file
    finally:
        os.remove(temp_path)


def test_load_config_invalid_json_string():
    with pytest.raises(ValueError):
        load_config("{invalid dict}")


def test_load_config_invalid_type():
    with pytest.raises(TypeError):
        load_config(123)
