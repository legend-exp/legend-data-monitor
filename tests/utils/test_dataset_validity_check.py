from unittest.mock import patch

from legend_data_monitor.utils import dataset_validity_check


# Test missing keys
def test_missing_experiment_key():
    data_info = {"type": "phy", "period": "p01", "path": "/tmp", "version": "v1"}
    with patch("legend_data_monitor.utils.logger") as mock_logger:
        dataset_validity_check(data_info)
        mock_logger.error.assert_called_with("\033[91mProvide experiment name!\033[0m")


def test_missing_type_key():
    data_info = {"experiment": "l200", "period": "p01", "path": "/tmp", "version": "v1"}
    with patch("legend_data_monitor.utils.logger") as mock_logger:
        dataset_validity_check(data_info)
        mock_logger.error.assert_called_with("\033[91mProvide data type!\033[0m")


def test_missing_period_key():
    data_info = {"experiment": "l200", "type": "phy", "path": "/tmp", "version": "v1"}
    with patch("legend_data_monitor.utils.logger") as mock_logger:
        dataset_validity_check(data_info)
        mock_logger.error.assert_called_with("\033[91mProvide period!\033[0m")


# Test invalid type
def test_invalid_type():
    data_info = {
        "experiment": "l200",
        "type": "invalid",
        "period": "p01",
        "path": "/tmp",
        "version": "v1",
    }
    with patch("legend_data_monitor.utils.logger") as mock_logger:
        dataset_validity_check(data_info)
        mock_logger.error.assert_called_with(
            "\033[91mInvalid data type provided!\033[0m"
        )


# Test path checks
def test_missing_path_key():
    data_info = {"experiment": "l200", "type": "phy", "period": "p01", "version": "v1"}
    with patch("legend_data_monitor.utils.logger") as mock_logger:
        dataset_validity_check(data_info)
        mock_logger.error.assert_called_with("\033[91mProvide path to data!\033[0m")


def test_nonexistent_path():
    data_info = {
        "experiment": "l200",
        "type": "phy",
        "period": "p01",
        "path": "/nonexistent",
        "version": "v1",
    }
    with (
        patch("legend_data_monitor.utils.logger") as mock_logger,
        patch("os.path.exists", return_value=False),
    ):
        dataset_validity_check(data_info)
        mock_logger.error.assert_called_with(
            "\033[91mThe data path you provided does not exist!\033[0m"
        )


# Test version checks
def test_missing_version_key(tmp_path):
    data_info = {
        "experiment": "l200",
        "type": "phy",
        "period": "p01",
        "path": str(tmp_path),
    }
    with patch("legend_data_monitor.utils.logger") as mock_logger:
        dataset_validity_check(data_info)
        mock_logger.error.assert_called_with(
            '\033[91mProvide processing version! If not needed, just put an empty string, "".\033[0m'
        )


def test_invalid_version(tmp_path):
    # create path but not version folder
    path = tmp_path / "data"
    path.mkdir()
    data_info = {
        "experiment": "l200",
        "type": "phy",
        "period": "p01",
        "path": str(path),
        "version": "v1",
    }
    with (
        patch("legend_data_monitor.utils.logger") as mock_logger,
        patch("os.path.exists", side_effect=lambda p: p != str(path / "v1")),
    ):
        dataset_validity_check(data_info)
        mock_logger.error.assert_called_with(
            "\033[91mProvide valid processing version!\033[0m"
        )


# Test valid input
def test_valid_input(tmp_path):
    path = tmp_path / "data"
    version_folder = path / "v1"
    version_folder.mkdir(parents=True)
    data_info = {
        "experiment": "l200",
        "type": "phy",
        "period": "p01",
        "path": str(path),
        "version": "v1",
    }
    with patch("legend_data_monitor.utils.logger") as mock_logger:
        dataset_validity_check(data_info)
        # for valid inputs, logger.error should NOT be called
        mock_logger.error.assert_not_called()
