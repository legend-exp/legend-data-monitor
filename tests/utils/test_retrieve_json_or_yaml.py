import pytest

from legend_data_monitor import errors
from legend_data_monitor.utils import retrieve_json_or_yaml


def test_retrieve_json_or_yaml(tmp_path):
    base_path = tmp_path
    filename = "testfile"
    yaml_path = base_path / f"{filename}.yaml"
    json_path = base_path / f"{filename}.json"

    # yaml
    yaml_path.write_text("some yaml content")
    result = retrieve_json_or_yaml(str(base_path), filename)
    assert result == str(yaml_path)
    # clean up the yaml file
    yaml_path.unlink()

    # json (but no yaml)
    json_path.write_text('{"key": "value"}')
    result = retrieve_json_or_yaml(str(base_path), filename)
    assert result == str(json_path)
    # clean up the json file
    json_path.unlink()

    # no yaml/json
    with pytest.raises(errors.ConfigError):
        retrieve_json_or_yaml(str(base_path), filename)
