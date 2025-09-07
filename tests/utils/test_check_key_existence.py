import os
import re
import tempfile
from unittest.mock import patch

import pandas as pd

from legend_data_monitor.utils import check_key_existence


def test_check_key_existence_key_found():
    # Test that function returns True when key exists in HDF file
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        hdf_path = tmp_file.name

    try:
        # HDF file with test data
        with pd.HDFStore(hdf_path, mode="w") as store:
            store.put(
                "/test_key", pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
            )
            store.put("/another_key", pd.DataFrame({"col1": [4, 5, 6]}))

        result = check_key_existence(hdf_path, "/test_key")
        assert result is True

        result = check_key_existence(hdf_path, "/another_key")
        assert result is True

    finally:
        # clean up
        if os.path.exists(hdf_path):
            os.unlink(hdf_path)


def test_check_key_existence_key_not_found(caplog):
    # Test that function returns False and logs debug when key doesn't exist
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        hdf_path = tmp_file.name

    try:
        with pd.HDFStore(hdf_path, mode="w") as store:
            store.put("/existing_key", pd.DataFrame({"col1": [1, 2, 3]}))

        result = check_key_existence(hdf_path, "/non_existing_key")
        assert result is False

        # check that debug message was logged
        assert any(
            "Key '/non_existing_key' not found" in rec.message for rec in caplog.records
        )

    finally:
        if os.path.exists(hdf_path):
            os.unlink(hdf_path)


def test_check_key_existence_empty_file(caplog):
    # Test empty HDF file
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        hdf_path = tmp_file.name

    try:
        with pd.HDFStore(hdf_path, mode="w"):
            pass

        result = check_key_existence(hdf_path, "/any_key")
        assert result is False

        # check that debug message was logged
        assert any("Key '/any_key' not found" in rec.message for rec in caplog.records)

    finally:
        if os.path.exists(hdf_path):
            os.unlink(hdf_path)


def test_check_key_existence_file_not_found(caplog):
    # HDF file doesn't exist
    non_existent_path = "/non/existent/path/file.h5"

    result = check_key_existence(non_existent_path, "/any_key")
    assert result is False

    assert any(
        re.search(rf"HDF file '{non_existent_path}' does not exist", message)
        for message in caplog.messages
    )


@patch("pandas.HDFStore")
def test_check_key_existence_store_exception(mock_hdfstore, caplog):
    # Test behavior when HDFStore raises an exception
    mock_hdfstore.side_effect = Exception("Mocked HDFStore error")

    result = check_key_existence("any_path.h5", "/any_key")
    assert result is False
    assert any("Error accessing HDF file" in rec.message for rec in caplog.records)
