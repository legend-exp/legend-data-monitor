import pytest

from legend_data_monitor.utils import get_query_timerange


def test_get_query_timerange_start_end():
    result = get_query_timerange(start="2022-09-28 08:00:00", end="2022-09-28 09:30:00")
    assert result == {
        "timestamp": {"start": "20220928T080000Z", "end": "20220928T093000Z"}
    }


def test_get_query_timerange_timestamps_single():
    result = get_query_timerange(timestamps="20220928T080000Z")
    assert result == {"timestamp": ["20220928T080000Z"]}


def test_get_query_timerange_timestamps_list():
    timestamps = ["20220928T080000Z", "20220928T093000Z"]
    result = get_query_timerange(timestamps=timestamps)
    assert result == {"timestamp": timestamps}


def test_get_query_timerange_runs_single():
    result = get_query_timerange(runs=10)
    assert result == {"run": ["r010"]}


def test_get_query_timerange_runs_list():
    result = get_query_timerange(runs=[9, 10])
    assert result == {"run": ["r009", "r010"]}


def test_get_query_timerange_dataset_wrapper():
    ds = {"start": "2022-09-28 08:00:00", "end": "2022-09-28 09:30:00"}
    result = get_query_timerange(dataset=ds)
    assert result == {
        "timestamp": {"start": "20220928T080000Z", "end": "20220928T093000Z"}
    }


def test_get_query_timerange_invalid_date_format(caplog):
    result = get_query_timerange(start="2022-09-28", end="invalid")
    assert result is None
    assert any("Invalid date format!" in m for m in caplog.text.splitlines())


def test_get_query_timerange_invalid_run_type(caplog):
    with pytest.raises(SystemExit):
        get_query_timerange(runs=["not_an_int"])


def test_get_query_timerange_invalid_mode(caplog):
    result = get_query_timerange(foo="bar")
    assert result is None
    assert any("Invalid time selection" in m for m in caplog.text.splitlines())
