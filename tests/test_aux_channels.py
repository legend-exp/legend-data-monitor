"""Auxiliary channels come from the channel map, not from hardcoded rawids."""

from typing import ClassVar

import pytest

from legend_data_monitor import utils


class _FakeMeta:
    """LegendMetadata stand-in returning a channel map like production's."""

    MAP: ClassVar[dict] = {
        "PULS01": {"system": "puls", "name": "PULS01", "daq": {"rawid": 1027201}},
        "PULS01ANA": {"system": "puls", "name": "PULS01ANA", "daq": {"rawid": 1027203}},
        "MUON01": {"system": "auxs", "name": "MUON01", "daq": {"rawid": 1027202}},
        "BSLN01": {"system": "bsln", "name": "BSLN01", "daq": {"rawid": 1027200}},
        # disconnected placeholders and real detectors must be ignored
        "BF862-12": {"system": "auxs", "name": "BF862-12", "daq": {"rawid": 1116805}},
        "DUMMY01": {"system": "auxs", "name": "DUMMY01", "daq": {"rawid": 999}},
        "V01234A": {"system": "geds", "name": "V01234A", "daq": {"rawid": 1104000}},
    }

    def __init__(self, _path):
        pass

    def channelmap(self, start_key=None):
        return dict(self.MAP)


@pytest.fixture(autouse=True)
def _clear_cache():
    utils._aux_channels_cached.cache_clear()
    yield
    utils._aux_channels_cached.cache_clear()


def test_aux_channels_resolved_by_system_and_name(monkeypatch):
    monkeypatch.setattr(utils, "LegendMetadata", _FakeMeta)
    assert utils.aux_channels("meta") == {
        "pulser": 1027201,
        "pulser01ana": 1027203,
        "muon": 1027202,
        "FCbsln": 1027200,
    }


def test_rawids_are_not_assumed(monkeypatch):
    """A cycle that renumbers the DAQ must still resolve."""

    class _Renumbered(_FakeMeta):
        MAP: ClassVar[dict] = {
            name: {**info, "daq": {"rawid": info["daq"]["rawid"] + 100000}}
            for name, info in _FakeMeta.MAP.items()
        }

    monkeypatch.setattr(utils, "LegendMetadata", _Renumbered)
    assert utils.aux_channels("meta")["pulser01ana"] == 1127203


def test_missing_systems_are_simply_absent(monkeypatch):
    class _GedsOnly(_FakeMeta):
        MAP: ClassVar[dict] = {"V01234A": _FakeMeta.MAP["V01234A"]}

    monkeypatch.setattr(utils, "LegendMetadata", _GedsOnly)
    assert utils.aux_channels("meta") == {}
