import pytest

from legend_data_monitor.utils import deep_get


def test_deep_get_basic():
    data = {"a": {"b": {"c": 123}, "d": None}, "x": 10}

    assert deep_get(data, ["a", "b", "c"]) == 123
    assert deep_get(data, ["a", "b", "z"], default="missing") == "missing"
    assert deep_get(data, ["a", "y", "c"], default="something") == "something"
    assert deep_get(data, ["x", "y"], default=0) == 0
    assert deep_get(data, ["a", "d"], default="default") is None
    assert deep_get(data, []) == data
