from unittest.mock import MagicMock, patch

from legend_data_monitor import utils


def test_make_output_paths_success(monkeypatch):
    """Test successful folder creation and return value."""
    mock_make_dir = MagicMock()

    config = {
        "output": "/base/output",
        "dataset": {"version": "v1", "type": "phy", "period": "2025A"},
    }

    user_time_range = {}

    with patch("legend_data_monitor.utils.make_dir", mock_make_dir):
        result = utils.make_output_paths(config, user_time_range)

    expected_path = "/base/output/v1/generated/plt/hit/phy/2025A/"
    assert result == expected_path

    # check that all required directories were created
    expected_calls = [
        ("/base/output",),
        ("/base/output/v1",),
        ("/base/output/v1/generated",),
        ("/base/output/v1/generated/plt",),
        ("/base/output/v1/generated/plt/hit",),
        ("/base/output/v1/generated/plt/hit/phy",),
        ("/base/output/v1/generated/plt/hit/phy/2025A/",),
    ]
    actual_calls = [call.args for call in mock_make_dir.call_args_list]
    assert actual_calls == expected_calls


def test_missing_output_field(caplog):
    """Test missing 'output' in config triggers error and returns None."""
    config = {"dataset": {"version": "v1", "type": "phy", "period": "2025A"}}
    result = utils.make_output_paths(config, {})
    assert result is None
    assert "Provide output folder path in your config" in caplog.text


def test_make_dir_raises(monkeypatch, caplog):
    """Test failure during initial output dir creation."""

    def failing_make_dir(path):
        raise PermissionError("No permission")

    monkeypatch.setattr("legend_data_monitor.utils.make_dir", failing_make_dir)

    config = {
        "output": "/forbidden/path",
        "dataset": {"version": "v1", "type": "phy", "period": "2025A"},
    }

    result = utils.make_output_paths(config, {})
    assert result is None
    assert "Cannot make output folder" in caplog.text
    assert "Maybe you don't have rights" in caplog.text


def test_type_as_list(monkeypatch):
    """Test when dataset['type'] is a list, creates 'cal_phy' directory."""
    mock_make_dir = MagicMock()

    config = {
        "output": "/data/output",
        "dataset": {"version": "v1", "type": ["cal", "phy"], "period": "2025A"},
    }

    with patch("legend_data_monitor.utils.make_dir", mock_make_dir):
        result = utils.make_output_paths(config, {})

    assert result == "/data/output/v1/generated/plt/hit/cal_phy/2025A/"
