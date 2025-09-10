import pytest

from legend_data_monitor.utils import build_detector_info

# mock channelmap
mock_chmap = {
    "D1": {
        "system": "geds",
        "name": "D1",
        "daq": {"rawid": 123},
        "location": {"string": 1, "position": 10},
        "analysis": {"processable": True},
    },
    "D2": {
        "system": "geds",
        "name": "D2",
        "daq": {"rawid": 124},
        "location": {"string": 1, "position": 11},
        "analysis": {"processable": False},
    },
    "D3": {
        "system": "other",
        "name": "D3",
        "daq": {"rawid": 125},
        "location": {"string": 2, "position": 12},
    },
}


class MockLegendMetadata:
    def __init__(self, path):
        pass

    def channelmap(self, start_key=None):
        return mock_chmap


@pytest.fixture(autouse=True)
def patch_legendmetadata(monkeypatch):
    # patch LegendMetadata inside legend_data_monitor.utils
    monkeypatch.setattr("legend_data_monitor.utils.LegendMetadata", MockLegendMetadata)


def test_build_detector_info():
    result = build_detector_info("dummy_path")

    # check top level keys
    assert "detectors" in result
    assert "str_chns" in result

    detectors = result["detectors"]
    str_chns = result["str_chns"]

    # check detectors dict
    assert "D1" in detectors
    assert "D2" in detectors
    assert "D3" not in detectors

    # fields
    d1 = detectors["D1"]
    assert d1["daq_rawid"] == 123
    assert d1["channel_str"] == "ch123"
    assert d1["string"] == 1
    assert d1["position"] == 10
    assert d1["processable"] is True

    d2 = detectors["D2"]
    assert d2["processable"] is False

    # only processable detectors are kept in str_chns
    assert str_chns == {1: ["D1"]}
