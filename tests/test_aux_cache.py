"""The aux channel is loaded once per (dataset, parameter), not per plot entry.

include_aux runs once per configured plot, and each call used to build a fresh
Subsystem and re-read pulser01ana -- 6 redundant loads per chunk on the
production config.
"""

import numpy as np
import pytest

from legend_data_monitor import subsystem


@pytest.fixture(autouse=True)
def _clean_cache():
    subsystem.clear_aux_cache()
    yield
    subsystem.clear_aux_cache()


def _dataset(timestamps=("20260731T181831Z",)):
    return {
        "experiment": "L200",
        "period": "p22",
        "version": "auto/v2.0.0",
        "path": "/prod/",
        "type": "phy",
        "timestamps": list(timestamps),
    }


def test_prewarm_serves_every_parameter_from_one_load(monkeypatch):
    built = []

    class FakeSubsystem:
        def __init__(self, name, dataset=None):
            built.append(name)

        def get_data(self, param):
            self.data = [param] if isinstance(param, str) else list(param)

    monkeypatch.setattr(subsystem, "Subsystem", FakeSubsystem)
    subsystem.prewarm_aux("pulser01ana", _dataset(), ["baseline", "bl_std"])
    first = subsystem._aux_subsystem("pulser01ana", _dataset(), "baseline")
    second = subsystem._aux_subsystem("pulser01ana", _dataset(), "bl_std")

    # one load serves both parameters (the old per-parameter key never hit:
    # every plot asks for a different parameter)
    assert first is second
    assert built == ["pulser01ana"]


def test_parameter_outside_the_prewarm_falls_back_to_its_own_load(monkeypatch):
    built = []

    class FakeSubsystem:
        def __init__(self, name, dataset=None):
            built.append(name)

        def get_data(self, param):
            self.data = [param] if isinstance(param, str) else list(param)

    monkeypatch.setattr(subsystem, "Subsystem", FakeSubsystem)
    subsystem.prewarm_aux("pulser01ana", _dataset(), ["baseline"])
    subsystem._aux_subsystem("pulser01ana", _dataset(), "baseline")
    subsystem._aux_subsystem("pulser01ana", _dataset(), "not_prewarmed")
    assert len(built) == 2


def test_a_different_keylist_is_a_different_entry(monkeypatch):
    """Chunks must not reuse the previous chunk's data."""
    built = []

    class FakeSubsystem:
        def __init__(self, name, dataset=None):
            built.append(name)

        def get_data(self, param):
            self.data = [param] if isinstance(param, str) else list(param)

    monkeypatch.setattr(subsystem, "Subsystem", FakeSubsystem)
    subsystem.prewarm_aux("pulser01ana", _dataset(["k1"]), ["baseline"])
    subsystem.prewarm_aux("pulser01ana", _dataset(["k2"]), ["baseline"])
    assert len(built) == 2


def test_cache_key_is_stable_and_parameter_independent():
    a = subsystem._aux_cache_key("pulser01ana", _dataset())
    b = subsystem._aux_cache_key("pulser01ana", _dataset())
    assert a == b
    assert a != subsystem._aux_cache_key("pulser01ana", _dataset(["other"]))


def test_clear_aux_cache_forces_a_reload(monkeypatch):
    built = []

    class FakeSubsystem:
        def __init__(self, name, dataset=None):
            built.append(name)

        def get_data(self, param):
            self.data = [param] if isinstance(param, str) else list(param)

    monkeypatch.setattr(subsystem, "Subsystem", FakeSubsystem)
    subsystem.prewarm_aux("pulser01ana", _dataset(), ["baseline"])
    subsystem.clear_aux_cache()
    subsystem.prewarm_aux("pulser01ana", _dataset(), ["baseline"])
    assert len(built) == 2


def test_prewarm_skips_parameters_the_aux_merge_never_uses(monkeypatch):
    """hit-tier, special and quality-cut parameters never reach the aux channel."""
    built = []

    class FakeSubsystem:
        def __init__(self, name, dataset=None):
            built.append(name)

        def get_data(self, param):
            self.data = [param] if isinstance(param, str) else list(param)

    monkeypatch.setattr(subsystem, "Subsystem", FakeSubsystem)
    # trapEmax_ctc_cal is hit tier, AoE_Custom is special, quality_cuts is a
    # pseudo-parameter: none of them should trigger a load
    subsystem.prewarm_aux(
        "pulser01ana", _dataset(), ["trapEmax_ctc_cal", "AoE_Custom", "quality_cuts"]
    )
    assert built == []


def test_prewarm_loads_the_union_in_one_call(monkeypatch):
    loaded = []

    class FakeSubsystem:
        def __init__(self, name, dataset=None):
            pass

        def get_data(self, param):
            loaded.append(param)
            self.data = [param] if isinstance(param, str) else list(param)

    monkeypatch.setattr(subsystem, "Subsystem", FakeSubsystem)
    subsystem.prewarm_aux("pulser01ana", _dataset(), ["baseline", "bl_std", "baseline"])
    # a single call carrying the de-duplicated union, not one call per parameter
    assert loaded == [["baseline", "bl_std"]]


# -------------------------------------------------------------------------
# channel-map metadata is compacted after the join
# -------------------------------------------------------------------------


def test_channel_map_columns_are_compacted():
    """Per-channel constants repeated over millions of rows dominated the frame.

    On a 5-file p22 load they were 85% of 446 MB against 3.6% for the
    parameters; downcasting/categorising them is ~13x smaller.
    """
    import pandas as pd

    n = 10_000
    df = pd.DataFrame(
        {
            "baseline": np.linspace(0, 1, n),
            "name": ["V01234A"] * n,  # few distinct values, many rows
            "location": [str(i % 3 + 1) for i in range(n)],  # ints stored as objects
            "det_type": ["bege"] * n,
        }
    )
    meta = ["name", "location", "det_type"]
    before = df.memory_usage(deep=True)[meta].sum()
    out = subsystem.compact_channel_map_columns(df.copy(), meta)
    after = out.memory_usage(deep=True)[meta].sum()

    # the metadata is what dominated the frame; it must shrink sharply
    assert after < before / 5, (before, after)
    # numeric-looking metadata becomes numeric, the rest categorical
    assert out["location"].dtype.kind in "iu"
    assert str(out["name"].dtype) == "category"
    # values survive, and the parameter column is untouched
    assert out["name"].astype(str).tolist() == df["name"].tolist()
    assert out["location"].astype(int).tolist() == [int(v) for v in df["location"]]
    assert out["baseline"].equals(df["baseline"])


def test_compaction_ignores_missing_and_already_typed_columns():
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "name": ["x", "y", "z"]})
    out = subsystem.compact_channel_map_columns(df.copy(), ["name", "not_present"])
    assert str(out["name"].dtype) == "category"
    assert out["a"].dtype == df["a"].dtype  # numeric column left alone
