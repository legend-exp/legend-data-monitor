import pytest

from legend_data_monitor.utils import build_detector_info

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


class MockProduction:
    def __init__(self, mass_in_g):
        self.mass_in_g = mass_in_g


class MockDiode:
    def __init__(self, mass_in_g):
        self.production = MockProduction(mass_in_g)


class MockGermanium:
    def __init__(self):
        self.diodes = {
            "D1": MockDiode(2000),  # 2 kg
            "D2": MockDiode(1800),  # 1.8 kg
        }


class MockDetectors:
    def __init__(self):
        self.germanium = MockGermanium()


class MockHardware:
    def __init__(self):
        self.detectors = MockDetectors()


class MockLegendMetadata:
    def __init__(self, path):
        self.hardware = MockHardware()

    def channelmap(self, start_key=None):
        return mock_chmap


@pytest.fixture(autouse=True)
def patch_legendmetadata(monkeypatch):
    monkeypatch.setattr(
        "legend_data_monitor.utils.LegendMetadata",
        MockLegendMetadata,
    )

def test_build_detector_info():
    result = build_detector_info("dummy_path")

    assert "detectors" in result
    assert "str_chns" in result

    detectors = result["detectors"]
    str_chns = result["str_chns"]

    # existing detectors
    assert set(detectors) == {"D1", "D2"}

    d1 = detectors["D1"]
    assert d1["daq_rawid"] == 123
    assert d1["channel_str"] == "ch123"
    assert d1["string"] == 1
    assert d1["position"] == 10
    assert d1["processable"] is True
    assert d1["mass_in_kg"] == 2.0

    d2 = detectors["D2"]
    assert d2["processable"] is False
    assert d2["mass_in_kg"] == 1.8

    # only processable detectors are kept in str_chns
    assert str_chns == {1: ["D1"]}
