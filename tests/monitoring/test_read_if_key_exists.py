import pandas as pd

from legend_data_monitor.monitoring import read_if_key_exists


def test_read_if_key_exists_found(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    hdf_path = tmp_path / "test.h5"
    df.to_hdf(hdf_path, key="mydata", mode="w")

    result = read_if_key_exists(str(hdf_path), "mydata")
    pd.testing.assert_frame_equal(result, df)


def test_read_if_key_exists_not_found(tmp_path):
    df = pd.DataFrame({"x": [10, 20]})
    hdf_path = tmp_path / "test.h5"
    df.to_hdf(hdf_path, key="exists", mode="w")

    result = read_if_key_exists(str(hdf_path), "other")
    assert result is None


def test_read_if_key_exists_empty_file(tmp_path):
    hdf_path = tmp_path / "empty.h5"
    with pd.HDFStore(hdf_path, mode="w"):
        pass

    result = read_if_key_exists(str(hdf_path), "anything")
    assert result is None


def test_read_if_key_exists_with_leading_slash(tmp_path):
    df = pd.DataFrame({"a": [7, 8, 9]})
    hdf_path = tmp_path / "test_slash.h5"
    df.to_hdf(hdf_path, key="/mydata", mode="w")

    # read without specifying the /
    result = read_if_key_exists(str(hdf_path), "mydata")
    pd.testing.assert_frame_equal(result, df)

    # read with the /
    result2 = read_if_key_exists(str(hdf_path), "/mydata")
    pd.testing.assert_frame_equal(result2, df)
