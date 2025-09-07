from unittest.mock import patch

import pytest

from legend_data_monitor.utils import check_scdb_settings


# Test missing 'slow_control' key
def test_missing_slow_control_key():
    conf = {}
    with patch("legend_data_monitor.utils.logger") as mock_logger:
        with pytest.raises(SystemExit):
            check_scdb_settings(conf)
        mock_logger.warning.assert_called_with(
            "\033[93mThere is no 'slow_control' key in the config file. Try again if you want to retrieve slow control data.\033[0m"
        )


# Test missing 'parameters' key
def test_missing_parameters_key():
    conf = {"slow_control": {}}
    with patch("legend_data_monitor.utils.logger") as mock_logger:
        with pytest.raises(SystemExit):
            check_scdb_settings(conf)
        mock_logger.warning.assert_called_with(
            "\033[93mThere is no 'parameters' key in config 'slow_control' entry. Try again if you want to retrieve slow control data.\033[0m"
        )


# Test invalid type for 'parameters'
def test_invalid_parameters_type():
    conf = {"slow_control": {"parameters": 123}}  # not str or list
    with patch("legend_data_monitor.utils.logger") as mock_logger:
        with pytest.raises(SystemExit):
            check_scdb_settings(conf)
        mock_logger.error.assert_called_with(
            "\033[91mSlow control parameters must be a string or a list of strings. Try again if you want to retrieve slow control data.\033[0m"
        )


# Test valid 'parameters'
def test_valid_parameters():
    # parameters as string
    conf_str = {"slow_control": {"parameters": "param1"}}
    # parameters as list
    conf_list = {"slow_control": {"parameters": ["param1", "param2"]}}
    with patch("legend_data_monitor.utils.logger") as mock_logger:
        assert check_scdb_settings(conf_str) is None  # function returns None if ok
        assert check_scdb_settings(conf_list) is None
        mock_logger.warning.assert_not_called()
        mock_logger.error.assert_not_called()
