"""True/False command-line flags arrive as strings; "False" must not mean True."""

import pytest

from legend_data_monitor.run import _flag


@pytest.mark.parametrize("value", [False, "False", "false", "0", "no", ""])
def test_false_flags(value):
    assert _flag(value) is False


@pytest.mark.parametrize("value", [True, "True", "true", "1", "yes"])
def test_true_flags(value):
    assert _flag(value) is True
