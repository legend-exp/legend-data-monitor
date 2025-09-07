import pytest

from legend_data_monitor.utils import get_key


def test_get_key():
    # valid
    fname = "somefile-20240717T153000Z.lh5"
    expected_key = "20240717T153000Z"
    assert get_key(fname) == expected_key

    # multiple timestamps (should return the first one)
    fname_multi = "prefix-20240101T000000Z-other-20241231T235959Z.lh5"
    expected_key_multi = "20240101T000000Z"
    assert get_key(fname_multi) == expected_key_multi

    # invalid, eg no timestamp
    fname_invalid = "file_without_timestamp.lh5"
    with pytest.raises(AttributeError):
        get_key(fname_invalid)
