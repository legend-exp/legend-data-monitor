from unittest.mock import patch

import pandas as pd
import pytest

from legend_data_monitor import utils
from legend_data_monitor.save_data import get_param_df


@pytest.fixture
def sample_dataframe():
    data = {
        "param1_value": [1, 2, 3],
        "param1_error": [0.1, 0.2, 0.3],
        "param2_value": [10, 20, 30],
        "param2_error": [1, 2, 3],
        "index": [0, 1, 2],
        "channel": [1, 2, 3],
        "datetime": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
        "name": ["det1", "det2", "det3"],
        "status": ["good", "good", "bad"],
        "other_col": ["a", "b", "c"],  # column not in keep_cols
    }
    return pd.DataFrame(data)


def test_get_param_df_regular_parameter(sample_dataframe):
    # Test with a parameter not in SPECIAL_PARAMETERS
    result = get_param_df("param1", sample_dataframe)

    # should contain param1 columns and keep_cols
    expected_columns = {
        "param1_value",
        "param1_error",
        "index",
        "channel",
        "datetime",
        "name",
        "status",
    }
    assert set(result.columns) == expected_columns
    assert "param2_value" not in result.columns  # other pars should be excluded
    assert (
        "other_col" not in result.columns
    )  # column not in keep_cols should be excluded


def test_get_param_df_special_parameter_single_col(sample_dataframe):
    # Test with special parameter that requires one additional column, by mocking SPECIAL_PARAMETERS
    with patch.dict(utils.SPECIAL_PARAMETERS, {"special_param": "other_col"}):
        df = sample_dataframe.copy()
        df["special_param_value"] = [100, 200, 300]

        result = get_param_df("special_param", df)

        # should contain special_param columns, keep_cols, and the required additional column
        expected_columns = {
            "special_param_value",
            "index",
            "channel",
            "datetime",
            "name",
            "status",
            "other_col",
        }
        assert set(result.columns) == expected_columns


def test_get_param_df_special_parameter_multiple_cols(sample_dataframe):
    # Test with special parameter that requires multiple additional columns
    with patch.dict(
        utils.SPECIAL_PARAMETERS, {"special_param": ["other_col", "param1_value"]}
    ):
        df = sample_dataframe.copy()
        df["special_param_value"] = [100, 200, 300]

        result = get_param_df("special_param", df)

        # should contain special_param columns, keep_cols, and the required additional columns
        expected_columns = {
            "special_param_value",
            "index",
            "channel",
            "datetime",
            "name",
            "status",
            "other_col",
            "param1_value",
        }
        assert set(result.columns) == expected_columns


def test_get_param_df_special_parameter_none(sample_dataframe):
    # Test with special parameter that requires no additional columns
    with patch.dict(utils.SPECIAL_PARAMETERS, {"special_param": None}):
        df = sample_dataframe.copy()
        df["special_param_value"] = [100, 200, 300]

        result = get_param_df("special_param", df)

        # should contain only special_param columns and keep_cols
        expected_columns = {
            "special_param_value",
            "index",
            "channel",
            "datetime",
            "name",
            "status",
        }
        assert set(result.columns) == expected_columns
        assert "other_col" not in result.columns


def test_get_param_df_special_parameter_empty_list(sample_dataframe):
    # Test with special parameter that requires empty list of additional columns
    with patch.dict(utils.SPECIAL_PARAMETERS, {"special_param": []}):
        df = sample_dataframe.copy()
        df["special_param_value"] = [100, 200, 300]

        result = get_param_df("special_param", df)

        expected_columns = {
            "special_param_value",
            "index",
            "channel",
            "datetime",
            "name",
            "status",
        }
        assert set(result.columns) == expected_columns


def test_get_param_df_parameter_not_found(sample_dataframe):
    # Test when parameter doesn't exist in DataFrame
    result = get_param_df("non_existent_param", sample_dataframe)

    # should contain only keep_cols (no parameter columns)
    expected_columns = {"index", "channel", "datetime", "name", "status"}
    assert set(result.columns) == expected_columns
    assert len(result.columns) == len(expected_columns)


def test_get_param_df_empty_dataframe():
    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    result = get_param_df("param1", empty_df)

    assert result.empty
    assert len(result.columns) == 0


def test_get_param_df_no_keep_cols_present(sample_dataframe):
    # Test when some keep_cols are missing from DataFrame
    # remove some keep_cols
    df = sample_dataframe.drop(columns=["channel", "name"])

    result = get_param_df("param1", df)

    # should contain param1 columns and available keep_cols
    expected_columns = {"param1_value", "param1_error", "index", "datetime", "status"}
    assert set(result.columns) == expected_columns


def test_get_param_df_additional_col_not_found(sample_dataframe):
    # Test when special parameter requires additional column that doesn't exist
    with patch.dict(utils.SPECIAL_PARAMETERS, {"special_param": "non_existent_col"}):
        df = sample_dataframe.copy()
        df["special_param_value"] = [100, 200, 300]

        result = get_param_df("special_param", df)

        # should contain special_param columns and keep_cols, but not the non-existent additional column
        expected_columns = {
            "special_param_value",
            "index",
            "channel",
            "datetime",
            "name",
            "status",
        }
        assert set(result.columns) == expected_columns


def test_get_param_df_verify_copy():
    # Test that the original DataFrame is not modified
    df = pd.DataFrame({"param1_value": [1, 2, 3], "index": [0, 1, 2]})
    original_cols = set(df.columns)

    result = get_param_df("param1", df)

    assert set(df.columns) == original_cols
    assert result is not df
