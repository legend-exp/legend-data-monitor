"""Runs with calibration pars but no physics data must not abort a period.

check_escale builds its run list from the cal par directory, which legitimately
contains runs that never took physics data (p16/r007). Resolving a phy start
key for those raised FileNotFoundError and took the whole period's calibration
analysis with it -- the p16 cal backfill returned rc=1 for r000-r006 until the
consumer dropped r007 from its corpus.
"""

import pytest

from legend_data_monitor import utils


class _FakeMeta:
    """LegendMetadata stand-in: one geds channel, same map for every key."""

    def __init__(self, _path):
        self.hardware = None

    def channelmap(self, start_key):
        return {
            "V01234A": {
                "system": "geds",
                "name": "V01234A",
                "daq": {"rawid": 1104000},
                "location": {"string": 1, "position": 1},
                "analysis": {"processable": True, "usability": "on"},
            }
        }


def _tree(tmp_path, runs_with_data):
    for run in runs_with_data:
        run_dir = tmp_path / "generated/tier/dsp/phy/p16" / run
        run_dir.mkdir(parents=True)
        (run_dir / f"l200-p16-{run}-phy-20250828T033011Z-tier_dsp.lh5").touch()
    return str(tmp_path)


def test_cal_only_run_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "LegendMetadata", _FakeMeta)
    path = _tree(tmp_path, ["r000", "r001"])
    # r007 has cal pars but no tier data, as on p16
    status = utils.build_detector_info_per_period(
        path, {"p16": ["r000", "r001", "r007"]}, "p16"
    )
    assert set(status["V01234A"]["usability"]) == {"p16-r000", "p16-r001"}
    assert status["V01234A"]["processable"]["p16-r000"] is True


def test_every_run_resolvable(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "LegendMetadata", _FakeMeta)
    path = _tree(tmp_path, ["r000", "r001"])
    status = utils.build_detector_info_per_period(
        path, {"p16": ["r000", "r001"]}, "p16"
    )
    assert set(status["V01234A"]["usability"]) == {"p16-r000", "p16-r001"}


def test_no_resolvable_run_yields_an_empty_map(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "LegendMetadata", _FakeMeta)
    status = utils.build_detector_info_per_period(
        _tree(tmp_path, []), {"p16": ["r007"]}, "p16"
    )
    assert status == {}


def test_empty_run_directory_is_skipped_too(tmp_path, monkeypatch):
    """get_start_key raises ValueError, not FileNotFoundError, for an empty dir."""
    monkeypatch.setattr(utils, "LegendMetadata", _FakeMeta)
    path = _tree(tmp_path, ["r000"])
    (tmp_path / "generated/tier/dsp/phy/p16/r008").mkdir(parents=True)
    status = utils.build_detector_info_per_period(
        path, {"p16": ["r000", "r008"]}, "p16"
    )
    assert set(status["V01234A"]["usability"]) == {"p16-r000"}


def test_get_start_key_still_raises_for_a_direct_caller(tmp_path):
    with pytest.raises(FileNotFoundError):
        utils.get_start_key(str(tmp_path), "phy", "p16", "r007")
