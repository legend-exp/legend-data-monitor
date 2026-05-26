from legend_data_monitor.excel.core import expand_run_list


def test_expand_run_list_single_string():
    assert expand_run_list("r001") == ["r001"]


def test_expand_run_list_single_range():
    assert expand_run_list("r000..r003") == ["r000", "r001", "r002", "r003"]


def test_expand_run_list_list_mixed():
    input_value = ["r000", "r002..r004", "r006"]

    assert expand_run_list(input_value) == [
        "r000",
        "r002",
        "r003",
        "r004",
        "r006",
    ]
