import pandas as pd

from legend_data_monitor.monitoring import get_traptmax_tp0est


def make_hdf(path, key, df, mode="w"):
    df.to_hdf(path, key=key, mode=mode)


def test_get_traptmax_tp0est(tmp_path):
    period = "p01"
    run = "run001"
    run_dir = tmp_path / period / run
    run_dir.mkdir(parents=True)

    # geds
    geds_trapTmax = pd.DataFrame({"val": [1, 2, 3]})
    geds_tp0est = pd.DataFrame({"val": [10, 20, 30]})
    geds_path = run_dir / "geds_file.hdf"
    make_hdf(geds_path, "IsPulser_TrapTmax", geds_trapTmax)
    make_hdf(geds_path, "IsPulser_Tp0Est", geds_tp0est, mode="a")

    # pulser
    puls_trapTmax = pd.DataFrame({"val": [100, 200]})
    puls_tp0est = pd.DataFrame({"val": [1000, 2000]})
    puls_path = run_dir / "pulser01ana_file.hdf"
    make_hdf(puls_path, "IsPulser_TrapTmax", puls_trapTmax)
    make_hdf(puls_path, "IsPulser_Tp0Est", puls_tp0est, mode="a")

    geds_out_trapTmax, geds_out_tp0est, puls_out_trapTmax, puls_out_tp0est = (
        get_traptmax_tp0est(str(tmp_path), period, [run])
    )

    pd.testing.assert_frame_equal(
        geds_out_trapTmax.reset_index(drop=True), geds_trapTmax
    )
    pd.testing.assert_frame_equal(geds_out_tp0est.reset_index(drop=True), geds_tp0est)
    pd.testing.assert_frame_equal(
        puls_out_trapTmax.reset_index(drop=True), puls_trapTmax
    )
    pd.testing.assert_frame_equal(puls_out_tp0est.reset_index(drop=True), puls_tp0est)


def test_get_traptmax_tp0est_missing_files(tmp_path):
    # no HDF file
    period = "p01"
    run = "run001"
    run_dir = tmp_path / period / run
    run_dir.mkdir(parents=True)

    geds_out_trapTmax, geds_out_tp0est, puls_out_trapTmax, puls_out_tp0est = (
        get_traptmax_tp0est(str(tmp_path), period, [run])
    )

    assert geds_out_trapTmax.empty
    assert geds_out_tp0est.empty
    assert puls_out_trapTmax.empty
    assert puls_out_tp0est.empty


def test_get_dfs_skip_non_listed_runs(tmp_path):
    period = "p01"
    run = "run001"
    run_dir = tmp_path / period / run
    run_dir.mkdir(parents=True)

    # no HDF files, no list runs: expected None
    geds_out_trapTmax, geds_out_tp0est, puls_out_trapTmax, puls_out_tp0est = (
        get_traptmax_tp0est(str(tmp_path), period, ["r1000"])
    )

    assert geds_out_trapTmax.empty
    assert geds_out_tp0est.empty
    assert puls_out_trapTmax.empty
    assert puls_out_tp0est.empty
