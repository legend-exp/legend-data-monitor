from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from legend_data_monitor.monitoring import get_run_start_end_times


@pytest.fixture
def fake_sto():
    sto = MagicMock()
    # simulate sto.read returning a numpy array of timestamps
    sto.read.side_effect = lambda path, fname: np.array([1000, 2000, 3000])
    return sto


def test_special_case(fake_sto, tmp_path):
    # fake directory structure
    folder_tier = tmp_path / "tier_hit" / "cal" / "p01" / "r001"
    folder_tier.mkdir(parents=True)
    file1 = "l200-p01-r001-cal-20240101T120000Z-tier_hit.lh5"
    file2 = "l200-p01-r001-cal-20240101T130000Z-tier_hit.lh5"
    (folder_tier / file1).write_text("dummy")
    (folder_tier / file2).write_text("dummy")

    # the phy period folder exists but holds no such run
    dir_path = tmp_path / "tier_phy" / "phy" / "p01"
    dir_path.mkdir(parents=True)

    start, end = get_run_start_end_times(
        sto=fake_sto,
        tiers=[str(tmp_path / "tier_hit"), str(tmp_path / "tier_phy")],
        period="p01",
        run="r001",
        tier="hit",
        pulser_rawid=1027201,
    )
    # the timestamps are read off the channel the caller named
    assert fake_sto.read.call_args[0][0] == "ch1027201/dsp/timestamp"

    # both should equal last timestamp
    expected = pd.to_datetime(3000, unit="s")
    assert start == expected
    assert end == expected


def test_normal_case(fake_sto, tmp_path):
    folder_tier = tmp_path / "tier_hit" / "cal" / "p01" / "r002"
    folder_tier.mkdir(parents=True)
    file1 = "l200-p01-r002-cal-20240101T120000Z-tier_hit.lh5"
    file2 = "l200-p01-r002-cal-20240101T130000Z-tier_hit.lh5"
    (folder_tier / file1).write_text("dummy")
    (folder_tier / file2).write_text("dummy")

    # the phy period folder contains the run: normal case
    dir_path = tmp_path / "tier_phy" / "phy" / "p01"
    (dir_path / "r002").mkdir(parents=True)

    start, end = get_run_start_end_times(
        sto=fake_sto,
        tiers=[str(tmp_path / "tier_hit"), str(tmp_path / "tier_phy")],
        period="p01",
        run="r002",
        tier="hit",
        pulser_rawid=1027201,
    )
    assert fake_sto.read.call_args[0][0] == "ch1027201/dsp/timestamp"

    # start = first timestamp, end = last timestamp
    expected_start = pd.to_datetime(1000, unit="s")
    expected_end = pd.to_datetime(3000, unit="s")
    assert start == expected_start
    assert end == expected_end


def test_falls_back_to_the_first_channel_in_the_file(fake_sto, tmp_path):
    """Without a pulser rawid the first channel in the file supplies the times."""
    import numpy as np
    from lgdo import lh5
    from lgdo.types import Array, Table

    folder_tier = tmp_path / "tier_hit" / "cal" / "p01" / "r003"
    folder_tier.mkdir(parents=True)
    fname = "l200-p01-r003-cal-20240101T120000Z-tier_hit.lh5"
    lh5.write(
        Table({"timestamp": Array(np.array([1000.0, 2000.0, 3000.0]))}),
        "dsp",
        str(folder_tier / fname),
        group="ch1084803",
    )
    get_run_start_end_times(
        sto=fake_sto,
        tiers=[str(tmp_path / "tier_hit"), str(tmp_path / "tier_phy")],
        period="p01",
        run="r003",
        tier="hit",
    )
    assert fake_sto.read.call_args[0][0] == "ch1084803/dsp/timestamp"
