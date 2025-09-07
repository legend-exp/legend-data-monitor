from unittest.mock import patch

import pytest

from legend_data_monitor.utils import get_output_path


# Ok path
def test_get_output_path_success():
    config = {"dataset": {"type": "phy", "experiment": "LEGEND", "period": "p01"}}

    with (
        patch(
            "legend_data_monitor.utils.get_query_timerange",
            return_value={
                "timestamp": {"start": "20220928T080000Z", "end": "20220928T093000Z"}
            },
        ),
        patch(
            "legend_data_monitor.utils.make_output_paths", return_value="/tmp/output/"
        ),
        patch(
            "legend_data_monitor.utils.get_run_name",
            return_value="20220928T080000Z_20220928T093000Z",
        ),
        patch(
            "legend_data_monitor.utils.get_time_name",
            return_value="20220928T080000Z_20220928T093000Z",
        ),
        patch("legend_data_monitor.utils.make_dir") as mock_make_dir,
    ):

        out_path = get_output_path(config)
        assert out_path.startswith("/tmp/output/20220928T080000Z_20220928T093000Z/")
        assert out_path.endswith("-phy")
        mock_make_dir.assert_called_once()


# Missing dataset field
def test_get_output_path_missing_dataset():
    config = {}  # dataset missing
    with patch("legend_data_monitor.utils.logger") as mock_logger:
        with pytest.raises(SystemExit):
            get_output_path(config)
        mock_logger.error.assert_called_once_with(
            "\033[91mSomething is missing or wrong in your 'dataset' field of the config.\033[0m"
        )


# get_query_timerange returns None
def test_get_output_path_query_timerange_none():
    config = {"dataset": {"type": "phy", "experiment": "LEGEND", "period": "p01"}}
    with patch("legend_data_monitor.utils.get_query_timerange", return_value=None):
        result = get_output_path(config)
        assert result is None
