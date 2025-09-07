import pandas as pd

from legend_data_monitor.monitoring import get_dfs


def make_hdf(path, key, df):
    # write a dataframe into a HDF file for a given key
    df.to_hdf(path, key=key, mode="w")


def test_get_dfs_with_valid_files(tmp_path):
    # create directory structure
    phy_mtg_data = tmp_path
    period = "p01"
    run = "run001"
    run_dir = phy_mtg_data / period / run
    run_dir.mkdir(parents=True)

    # geds file
    geds_df = pd.DataFrame({"val": [1, 2, 3]})
    geds_path = run_dir / "file_geds.hdf"
    make_hdf(geds_path, "IsPulser_TrapemaxCtcCal", geds_df)

    # pulser file
    puls_df = pd.DataFrame({"val": [10, 20]})
    puls_path = run_dir / "file_pulser01ana.hdf"
    make_hdf(puls_path, "IsPulser_TrapemaxCtcCal", puls_df)

    geds_abs, geds_corr, puls_abs = get_dfs(
        str(phy_mtg_data), period, [run], "TrapemaxCtcCal"
    )

    pd.testing.assert_frame_equal(geds_abs.reset_index(drop=True), geds_df)
    pd.testing.assert_frame_equal(puls_abs.reset_index(drop=True), puls_df)
    assert geds_corr.empty  # there is no _pulser01anaDiff key


def test_get_dfs_missing_geds_file(tmp_path):
    phy_mtg_data = tmp_path
    period = "p01"
    run = "run001"
    run_dir = phy_mtg_data / period / run
    run_dir.mkdir(parents=True)

    # no geds files, then 3 None
    result = get_dfs(str(phy_mtg_data), period, [run], "TrapemaxCtcCal")
    assert result == (None, None, None)


def test_get_dfs_runs_not_avail(tmp_path):
    phy_mtg_data = tmp_path
    period = "p01"
    run = "run001"
    run_dir = phy_mtg_data / period / run
    run_dir.mkdir(parents=True)

    # run not present in the list of avail runs
    result = get_dfs(str(phy_mtg_data), period, ["run100"], "TrapemaxCtcCal")
    assert result == (None, None, None)


def test_get_dfs_skip_non_listed_runs(tmp_path):
    phy_mtg_data = tmp_path
    period = "p01"
    (phy_mtg_data / period).mkdir(parents=True)

    # no HDF files, no list runs: expected None
    result = get_dfs(str(phy_mtg_data), period, [], "TrapemaxCtcCal")

    assert result == (None, None, None)


def test_geds_pulser_correction_and_empty_pulser(tmp_path):
    phy_mtg_data = tmp_path
    period = "p01"
    run = "run001"
    run_dir = phy_mtg_data / period / run
    run_dir.mkdir(parents=True)

    # geds HDF with pulser01anaDiff
    df_geds_abs = pd.DataFrame({"val": [1, 2, 3]})
    df_geds_corr = pd.DataFrame({"val": [10, 20, 30]})
    geds_path = run_dir / "file_geds.hdf"
    df_geds_abs.to_hdf(geds_path, key="IsPulser_TrapemaxCtcCal", mode="w")
    df_geds_corr.to_hdf(
        geds_path, key="IsPulser_TrapemaxCtcCal_pulser01anaDiff", mode="a"
    )

    geds_abs, geds_corr, puls_abs = get_dfs(
        str(phy_mtg_data), period, [run], "TrapemaxCtcCal"
    )

    pd.testing.assert_frame_equal(geds_corr.reset_index(drop=True), df_geds_corr)
    assert puls_abs.empty
