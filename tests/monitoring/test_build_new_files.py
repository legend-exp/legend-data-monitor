import pandas as pd
import pytest
import yaml

from legend_data_monitor import errors
from legend_data_monitor.monitoring import build_new_files


def make_hdf(path, key, df, mode="w"):
    df.to_hdf(path, key=key, mode=mode)


def test_build_new_files_creates_outputs(tmp_path):
    period = "p01"
    run = "r001"
    base_dir = tmp_path / "generated/plt/hit/phy" / period / run
    base_dir.mkdir(parents=True, exist_ok=True)

    # fake HDF file
    data_file = base_dir / f"l200-{period}-{run}-phy-geds.hdf"
    df = pd.DataFrame(
        {"val": [1, 2, 3]}, index=pd.date_range("2025-01-01", periods=3, freq="min")
    )
    info_df = pd.DataFrame(
        {"Value": ["subsys", "unit", "label", "type", 0, 10, 0, 10]},
        index=[
            "subsystem",
            "unit",
            "label",
            "event_type",
            "lower_lim_var",
            "upper_lim_var",
            "lower_lim_abs",
            "upper_lim_abs",
        ],
    )
    info_df = info_df.astype(str)
    make_hdf(data_file, "data", df)
    make_hdf(data_file, "info", info_df, mode="a")

    build_new_files(str(tmp_path), period, run)

    # check resampled files exist
    for resample_unit in ["10min", "60min"]:
        res_file = base_dir / f"l200-{period}-{run}-phy-geds-res_{resample_unit}.hdf"
        assert res_file.exists()

        # check keys inside the resampled file are untouched
        with pd.HDFStore(res_file, mode="r") as store:
            keys = store.keys()
            assert "/data" in keys

    # check YAML file -- it is named .yaml and is now written as YAML, not as
    # the JSON that happened to parse as YAML before
    info_file = base_dir / f"l200-{period}-{run}-phy-geds-info.yaml"
    assert info_file.exists()
    with open(info_file) as f:
        info = yaml.safe_load(f)
    assert "keys" in info
    assert "info" in info
    assert not info_file.read_text().lstrip().startswith("{")


def test_build_new_files_missing_file(tmp_path):
    # the file does not exist
    period = "p01"
    run = "r001"
    with pytest.raises(errors.DataError):
        build_new_files(str(tmp_path), period, run)
