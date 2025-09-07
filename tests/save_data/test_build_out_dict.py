from unittest.mock import MagicMock, patch

import pytest

from legend_data_monitor.save_data import build_out_dict


@pytest.fixture
def plot_settings_overwrite():
    return {"parameters": "param1", "saving": "overwrite", "plt_path": "/fake/path"}


@pytest.fixture
def plot_settings_append():
    return {
        "parameters": ["param1", "param2"],
        "saving": "append",
        "plt_path": "/fake/path",
    }


@pytest.fixture
def par_dict_content():
    return {
        "plot_info": {"subsystem": "subsys1", "title": "Title"},
        "data": "fake_data",
    }


def test_build_out_dict_overwrite(plot_settings_overwrite, par_dict_content):
    out_dict = {}

    with patch(
        "legend_data_monitor.save_data.build_dict", return_value={"key": "value"}
    ) as mock_build:
        result = build_out_dict(plot_settings_overwrite, par_dict_content, out_dict)

    mock_build.assert_called_once()
    assert result == {"key": "value"}


def test_build_out_dict_append_file_not_exists(plot_settings_append, par_dict_content):
    out_dict = {}

    with (
        patch("os.path.exists", return_value=False),
        patch(
            "legend_data_monitor.save_data.build_dict", return_value={"key": "value"}
        ) as mock_build,
    ):

        result = build_out_dict(plot_settings_append, par_dict_content, out_dict)

    mock_build.assert_called_once()
    assert result == {"key": "value"}


def test_build_out_dict_append_file_exists(plot_settings_append, par_dict_content):
    out_dict = {}
    fake_old_dict = {"param1": [1], "param2": [2]}
    mock_shelf = MagicMock()
    mock_shelf.__enter__.return_value = fake_old_dict

    with (
        patch("os.path.exists", return_value=True),
        patch("shelve.open", return_value=mock_shelf),
        patch(
            "legend_data_monitor.save_data.append_new_data",
            side_effect=lambda *a, **kw: {"updated": True},
        ) as mock_append,
    ):

        result = build_out_dict(plot_settings_append, par_dict_content, out_dict)

    assert mock_append.call_count == 2
    assert result == {"updated": True}


def test_build_out_dict_append_file_exists_one_param(par_dict_content):
    # Test append mode with single parameter (string case)
    plot_settings = {
        "parameters": "single_param",
        "saving": "append",
        "plt_path": "/fake/path",
    }
    out_dict = {}
    fake_old_dict = {"single_param": [1]}
    mock_shelf = MagicMock()
    mock_shelf.__enter__.return_value = fake_old_dict

    with (
        patch("os.path.exists", return_value=True),
        patch("shelve.open", return_value=mock_shelf),
        patch(
            "legend_data_monitor.save_data.append_new_data",
            return_value={"updated": True},
        ) as mock_append,
        patch("legend_data_monitor.save_data.utils.logger") as mock_logger,
    ):

        result = build_out_dict(plot_settings, par_dict_content, out_dict)

    # verify append_new_data was called with the correct parameter
    mock_append.assert_called_once()
    call_args = mock_append.call_args[0]
    assert call_args[0] == "single_param"

    # verify debug message was logged for one-parameter case
    mock_logger.debug.assert_any_call(
        "... appending new data for the one-parameter case"
    )

    assert result == {"updated": True}


def test_build_out_dict_append_file_exists_single_item_list(par_dict_content):
    # Test append mode with single parameter (list with one item case)
    plot_settings = {
        "parameters": ["single_param"],
        "saving": "append",
        "plt_path": "/fake/path",
    }
    out_dict = {}
    fake_old_dict = {"single_param": [1]}
    mock_shelf = MagicMock()
    mock_shelf.__enter__.return_value = fake_old_dict

    with (
        patch("os.path.exists", return_value=True),
        patch("shelve.open", return_value=mock_shelf),
        patch(
            "legend_data_monitor.save_data.append_new_data",
            return_value={"updated": True},
        ) as mock_append,
        patch("legend_data_monitor.save_data.utils.logger") as mock_logger,
    ):

        result = build_out_dict(plot_settings, par_dict_content, out_dict)

    mock_append.assert_called_once()
    call_args = mock_append.call_args[0]
    assert call_args[0] == "single_param"

    mock_logger.debug.assert_any_call(
        "... appending new data for the one-parameter case"
    )

    assert result == {"updated": True}


def test_build_out_dict_append_file_exists_multi_param(par_dict_content):
    # Test append mode with multiple parameters
    plot_settings = {
        "parameters": ["param1", "param2", "param3"],
        "saving": "append",
        "plt_path": "/fake/path",
    }
    out_dict = {}
    fake_old_dict = {"param1": [1], "param2": [2], "param3": [3]}
    mock_shelf = MagicMock()
    mock_shelf.__enter__.return_value = fake_old_dict

    with (
        patch("os.path.exists", return_value=True),
        patch("shelve.open", return_value=mock_shelf),
        patch(
            "legend_data_monitor.save_data.append_new_data",
            side_effect=lambda *a, **kw: {"updated": True},
        ) as mock_append,
        patch("legend_data_monitor.save_data.utils.logger") as mock_logger,
    ):

        result = build_out_dict(plot_settings, par_dict_content, out_dict)

    # verify append_new_data was called for each parameter
    assert mock_append.call_count == 3

    # verify debug message was logged for multi-parameter case
    mock_logger.debug.assert_any_call(
        "... appending new data for the multi-parameters case"
    )

    # check that each parameter was passed correctly
    call_args = [call[0][0] for call in mock_append.call_args_list]
    assert call_args == ["param1", "param2", "param3"]

    assert result == {"updated": True}
