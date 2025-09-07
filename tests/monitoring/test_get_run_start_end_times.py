import os
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


def test_special_case(monkeypatch, fake_sto, tmp_path):
    # fake directory structure
    folder_tier = tmp_path / "tier_hit" / "cal" / "p01" / "r001"
    folder_tier.mkdir(parents=True)
    (folder_tier / "file1.lh5").write_text("dummy")
    (folder_tier / "file2.lh5").write_text("dummy")

    dir_path = tmp_path / "tier_phy" / "phy" / "p01"
    dir_path.mkdir(parents=True)
    # os.listdir(dir_path) does NOT contain run
    monkeypatch.setattr(
        os,
        "listdir",
        lambda path: [] if str(path).endswith("p01") else ["file1.lh5", "file2.lh5"],
    )
    monkeypatch.setattr(os.path, "isdir", lambda path: True)

    start, end = get_run_start_end_times(
        sto=fake_sto,
        tiers=[str(tmp_path / "tier_hit"), str(tmp_path / "tier_phy")],
        period="p01",
        run="r001",
        tier="hit",
    )

    # both should equal last timestamp
    expected = pd.to_datetime(3000, unit="s")
    assert start == expected
    assert end == expected


def test_normal_case(monkeypatch, fake_sto, tmp_path):
    folder_tier = tmp_path / "tier_hit" / "cal" / "p01" / "r002"
    folder_tier.mkdir(parents=True)
    (folder_tier / "file1.lh5").write_text("dummy")
    (folder_tier / "file2.lh5").write_text("dummy")

    # dir_path exists but contains run: normal case
    dir_path = tmp_path / "tier_phy" / "phy" / "p01"
    dir_path.mkdir(parents=True)
    monkeypatch.setattr(
        os,
        "listdir",
        lambda path: (
            ["r002"] if str(path).endswith("p01") else ["file1.lh5", "file2.lh5"]
        ),
    )
    monkeypatch.setattr(os.path, "isdir", lambda path: True)

    start, end = get_run_start_end_times(
        sto=fake_sto,
        tiers=[str(tmp_path / "tier_hit"), str(tmp_path / "tier_phy")],
        period="p01",
        run="r002",
        tier="hit",
    )

    # start = first timestamp, end = last timestamp
    expected_start = pd.to_datetime(1000, unit="s")
    expected_end = pd.to_datetime(3000, unit="s")
    assert start == expected_start
    assert end == expected_end
