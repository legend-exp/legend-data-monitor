"""get_valid_path falls back across the processed tiers.

Uses real directories rather than patching the filesystem API, so the test
pins the behaviour instead of the implementation.
"""

import pytest

from legend_data_monitor import errors, utils


def test_base_path_exists(tmp_path):
    dsp = tmp_path / "generated/tier/dsp"
    dsp.mkdir(parents=True)
    assert utils.get_valid_path(str(dsp)) == str(dsp)


def test_fallback_path_exists(tmp_path):
    (tmp_path / "generated/tier/psp").mkdir(parents=True)
    dsp = str(tmp_path / "generated/tier/dsp")
    assert utils.get_valid_path(dsp) == str(tmp_path / "generated/tier/psp")


def test_no_valid_path(tmp_path, caplog):
    with pytest.raises(errors.ConfigError):
        utils.get_valid_path(str(tmp_path / "generated/tier/dsp"))
    assert "The path of dsp/hit/evt/psp/pht/pet/skm files is not valid" in caplog.text
