from legend_data_monitor.monitoring import find_hdf_file


def test_find_hdf_file_includes(tmp_path):
    # fake files
    (tmp_path / "data_geds_run1.hdf").write_text("some_text")
    (tmp_path / "data_puls_run1.hdf").write_text("some_text")
    (tmp_path / "other.txt").write_text("not an hdf")

    result = find_hdf_file(str(tmp_path), include=["geds"])
    assert result is not None
    assert result.endswith("data_geds_run1.hdf")


def test_find_hdf_file_includes_and_excludes(tmp_path):
    (tmp_path / "data_geds_res_run1.hdf").write_text("some_text")
    (tmp_path / "data_geds_run2.hdf").write_text("some_text")

    result = find_hdf_file(str(tmp_path), include=["geds"], exclude=["res"])
    assert result is not None
    assert result.endswith("data_geds_run2.hdf")


def test_find_hdf_file_no_match(tmp_path):
    (tmp_path / "data_other_run1.hdf").write_text("some_text")

    result = find_hdf_file(str(tmp_path), include=["geds"])
    assert result is None


def test_find_hdf_file_empty_dir(tmp_path):
    # no files
    result = find_hdf_file(str(tmp_path), include=["geds"])
    assert result is None
