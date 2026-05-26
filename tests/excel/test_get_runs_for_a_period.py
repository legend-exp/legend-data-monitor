import pytest

from legend_data_monitor.excel.core import get_runs_for_a_period


def test_get_runs_for_a_period_only_cal(tmp_path):
    auto_dir = tmp_path / "auto"
    cal_dir = auto_dir / "generated/tier/dsp/cal/p01"
    cal_dir.mkdir(parents=True)

    # fake runs
    (cal_dir / "r000").mkdir()
    (cal_dir / "r001").mkdir()

    result = get_runs_for_a_period(str(auto_dir), "p01")

    assert "p01" in result
    assert result["p01"] == [
        ("cal", "r000"),
        ("cal", "r001"),
    ]


def test_get_runs_for_a_period_with_phy(tmp_path):
    auto_dir = tmp_path / "auto"

    cal_dir = auto_dir / "generated/tier/dsp/cal/p01"
    phy_dir = auto_dir / "generated/tier/dsp/phy/p01"

    cal_dir.mkdir(parents=True)
    phy_dir.mkdir(parents=True)

    (cal_dir / "r000").mkdir()
    (cal_dir / "r001").mkdir()
    # only r000 has phy
    (phy_dir / "r000").mkdir()

    result = get_runs_for_a_period(str(auto_dir), "p01")

    assert result["p01"] == [
        ("cal", "r000"),
        ("phy", "r000"),
        ("cal", "r001"),
    ]


def test_get_runs_for_a_period_missing_cal(tmp_path):
    auto_dir = tmp_path / "auto"

    with pytest.raises(SystemExit) as e:
        get_runs_for_a_period(str(auto_dir), "p01")

    assert "No cal DSP data found" in str(e.value)
