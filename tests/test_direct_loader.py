"""Equivalence of the direct LH5 loader with the DataLoader path it replaces.

Monitoring loads whole channels for a known file list, so pygama's per-key
entry-list construction is pure overhead; these tests pin the replacement to
byte-identical output. The end-to-end comparison needs the production tree and
is skipped when it is not reachable (e.g. off-cluster CI).
"""

import os

import numpy as np
import pandas as pd
import pytest

from legend_data_monitor.loading import phy_files

PROD = "/data2/public/prodenv/prod-blind/"
PERIOD, RUN, VERSION = "p22", "r012", "auto/v2.0.0"
RUN_DIR = os.path.join(PROD, VERSION, "generated/tier/dsp/phy", PERIOD, RUN)
needs_prod = pytest.mark.skipif(
    not os.path.isdir(RUN_DIR), reason="production tree not available"
)


def test_merge_tiers_joins_on_channel_and_timestamp():
    dsp = pd.DataFrame(
        {"channel": [1, 1, 2], "timestamp": [10.0, 11.0, 10.0], "baseline": [1, 2, 3]}
    )
    hit = pd.DataFrame(
        {
            "channel": [2, 1, 1],
            "timestamp": [10.0, 11.0, 10.0],
            "cal": [30.0, 20.0, 10.0],
        }
    )
    merged = phy_files.merge_tiers({"dsp": dsp, "hit": hit}).sort_values(
        ["channel", "timestamp"]
    )
    # rows pair up by key, not by position (the old positional concat would
    # have mismatched these two orderings silently)
    assert list(merged["baseline"]) == [1, 2, 3]
    assert list(merged["cal"]) == [10.0, 20.0, 30.0]


def test_merge_tiers_handles_empty_and_single():
    assert phy_files.merge_tiers({}).empty
    assert phy_files.merge_tiers({"dsp": pd.DataFrame()}).empty
    only = pd.DataFrame({"channel": [1], "timestamp": [1.0], "x": [2.0]})
    assert phy_files.merge_tiers({"dsp": only}).equals(only)


@needs_prod
def test_resolve_files_by_key_and_by_run():
    keys = ["20260731T181831Z", "20260731T191833Z"]
    by_key = phy_files.resolve_files(
        PROD, VERSION, "dsp", "phy", PERIOD, {"timestamp": keys}
    )
    assert len(by_key) == 2
    assert all(any(k in f for k in keys) for f in by_key)

    by_run = phy_files.resolve_files(
        PROD, VERSION, "dsp", "phy", PERIOD, {"run": [RUN]}
    )
    assert len(by_run) > len(by_key)
    assert all(f.endswith("-tier_dsp.lh5") for f in by_run)


@needs_prod
def test_direct_loader_matches_dataloader_frame():
    """The two loaders must return the same rows, columns, dtypes and values."""
    import glob

    from legend_data_monitor import subsystem

    files = sorted(glob.glob(f"{RUN_DIR}/*.lh5"))[:3]
    keys = [os.path.basename(f).split("-tier")[0].split("phy-")[1] for f in files]
    dataset = {
        "experiment": "L200",
        "period": PERIOD,
        "version": VERSION,
        "path": PROD,
        "type": "phy",
        "timestamps": keys,
    }
    params = ["baseline", "bl_std", "cuspEmax"]

    previous = os.environ.get("LMON_LOADER")
    try:
        os.environ["LMON_LOADER"] = "dataloader"
        reference = subsystem.Subsystem("geds", dataset=dataset)
        reference.get_data(params)
        os.environ["LMON_LOADER"] = "direct"
        direct = subsystem.Subsystem("geds", dataset=dataset)
        direct.get_data(params)
    finally:
        if previous is None:
            os.environ.pop("LMON_LOADER", None)
        else:
            os.environ["LMON_LOADER"] = previous

    order = ["channel", "datetime"]
    left = reference.data.sort_values(order).reset_index(drop=True)
    right = direct.data.sort_values(order).reset_index(drop=True)

    assert list(left.columns) == list(right.columns)
    assert len(left) == len(right)
    for column in left.columns:
        assert left[column].dtype == right[column].dtype, column
        assert left[column].equals(right[column]), column


def test_missing_field_yields_no_data_rather_than_uninitialised_values(tmp_path):
    """A channel without a requested field must not contribute invented values.

    The DataLoader path this replaces filled such channels with uninitialised
    memory: on p22/r012, six detectors that carry is_valid_bl_poly_rms but not
    is_valid_bl_poly_rms_classifier came out as denormals (~1.5e-319) against a
    real range of -5.19..6315.89, and that garbage reached the v1 file, the
    contract and the dashboard.
    """
    import lgdo
    from lgdo import lh5

    path = str(tmp_path / "l200-p22-r012-phy-20260101T000000Z-tier_dsp.lh5")
    # ch0001 has both fields, ch0002 only one of them
    lh5.write(
        lgdo.Table(
            {
                "timestamp": lgdo.Array(np.array([1.0, 2.0])),
                "baseline": lgdo.Array(np.array([10.0, 11.0])),
                "special": lgdo.Array(np.array([100.0, 101.0])),
            }
        ),
        "dsp",
        path,
        group="ch0001",
    )
    lh5.write(
        lgdo.Table(
            {
                "timestamp": lgdo.Array(np.array([3.0, 4.0])),
                "baseline": lgdo.Array(np.array([12.0, 13.0])),
            }
        ),
        "dsp",
        path,
        group="ch0002",
    )

    frame = phy_files.load_channel_frame(
        [path], "dsp", ["ch0001", "ch0002"], ["baseline", "special"]
    )

    assert set(frame["channel"]) == {1, 2}
    # the channel that has the field keeps its real values
    have = frame[frame["channel"] == 1]["special"]
    assert sorted(have.tolist()) == [100.0, 101.0]
    # the channel that lacks it is NaN, never a denormal or a zero
    lack = frame[frame["channel"] == 2]["special"]
    assert lack.isna().all(), lack.tolist()
