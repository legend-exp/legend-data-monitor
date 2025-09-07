from unittest.mock import patch

from legend_data_monitor.utils import get_time_name


# Timestamp with start and end
def test_timestamp_start_end():
    user_time_range = {
        "timestamp": {"start": "20220928T080000Z", "end": "20220928T093000Z"}
    }
    result = get_time_name(user_time_range)
    assert result == "20220928T080000Z_20220928T093000Z"


# Timestamp single element
def test_timestamp_single():
    user_time_range = {"timestamp": ["20230207T103123Z"]}
    result = get_time_name(user_time_range)
    assert result == "20230207T103123Z"


# Timestamp multiple elements
def test_timestamp_multiple():
    user_time_range = {
        "timestamp": [
            "20230207T103123Z",
            "20230207T141123Z",
            "20230207T083323Z",
        ]
    }
    result = get_time_name(user_time_range)
    # min_max timestamps
    assert result == "20230207T083323Z_20230207T141123Z"


# Single run
def test_run_single():
    user_time_range = {"run": ["r010"]}
    with patch(
        "legend_data_monitor.utils.get_multiple_run_id", return_value="r010"
    ) as mock_run:
        result = get_time_name(user_time_range)
        mock_run.assert_called_once_with(user_time_range)
        assert result == "r010"


# Multiple runs
def test_run_multiple():
    user_time_range = {"run": ["r010", "r014"]}
    with patch(
        "legend_data_monitor.utils.get_multiple_run_id", return_value="r010_r014"
    ) as mock_run:
        result = get_time_name(user_time_range)
        mock_run.assert_called_once_with(user_time_range)
        assert result == "r010_r014"


# Invalid input
def test_invalid_time_selection():
    user_time_range = {"invalid_key": ["foo"]}
    with patch("legend_data_monitor.utils.logger") as mock_logger:
        result = get_time_name(user_time_range)
        mock_logger.error.assert_called_once_with(
            "\033[91mInvalid time selection!\033[0m"
        )
        assert result is None
