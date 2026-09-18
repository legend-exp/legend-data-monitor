import yaml

from legend_data_monitor.excel.core import get_periods


def test_get_periods_basic(tmp_path, monkeypatch):
    datasets_path = tmp_path

    fake_yaml = {"mykey": {"cal": {"p01": ["r000", "r002"]}, "phy": {"p01": ["r001"]}}}

    yaml_file = datasets_path / "runlists.yaml"
    yaml_file.write_text(yaml.dump(fake_yaml))

    # mock expand_run_list
    monkeypatch.setattr("legend_data_monitor.excel.core.expand_run_list", lambda x: x)

    result = get_periods("mykey", datasets_path)

    assert result == {
        "p01": [
            ("cal", "r000"),
            ("cal", "r002"),
            ("phy", "r001"),
        ]
    }
