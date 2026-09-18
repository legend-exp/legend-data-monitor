import pytest

from legend_data_monitor import errors, utils


def test_base_path_exists(monkeypatch):
    monkeypatch.setattr("os.path.exists", lambda path: path == "/valid/path/dsp")

    result = utils.get_valid_path("/valid/path/dsp")
    assert result == "/valid/path/dsp"


def test_fallback_path_exists(monkeypatch):
    def fake_exists(path):
        if path == "/valid/path/dsp":
            return False
        elif path == "/valid/path/psp":
            return True
        return False

    monkeypatch.setattr("os.path.exists", fake_exists)

    result = utils.get_valid_path("/valid/path/dsp")
    assert result == "/valid/path/psp"


def test_no_valid_path(monkeypatch, caplog):
    monkeypatch.setattr("os.path.exists", lambda path: False)

    with pytest.raises(errors.ConfigError):
        utils.get_valid_path("/invalid/path/dsp")

    assert "The path of dsp/hit/evt/psp/pht/pet/skm files is not valid" in caplog.text
