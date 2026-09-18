"""Repairing one parameter's keys without re-running the pipeline."""

import os

import numpy as np
import pandas as pd
import pytest

from legend_data_monitor import repair


def test_chunk_lists_are_in_processing_order(tmp_path):
    mtg = tmp_path / "generated/tmp/mtg/p22/r012"
    mtg.mkdir(parents=True)
    for i in (1, 10, 2):  # lexical order would put 10 before 2
        (mtg / f"new_keys_part_{i}.filekeylist").write_text("k\n")
    (mtg / "new_keys.filekeylist").write_text("k\n")
    found = repair.chunk_lists(str(tmp_path), "p22", "r012")
    assert [os.path.basename(p) for p in found] == [
        "new_keys_part_1.filekeylist",
        "new_keys_part_2.filekeylist",
        "new_keys_part_10.filekeylist",
    ]


def test_unsplit_run_falls_back_to_the_whole_list(tmp_path):
    mtg = tmp_path / "generated/tmp/mtg/p22/r012"
    mtg.mkdir(parents=True)
    (mtg / "new_keys.filekeylist").write_text("k\n")
    assert [
        os.path.basename(p) for p in repair.chunk_lists(str(tmp_path), "p22", "r012")
    ] == ["new_keys.filekeylist"]
    assert repair.chunk_lists(str(tmp_path), "p22", "r999") == []


def test_config_entries_select_the_parameter():
    entries = repair.config_entries("bl_mean")
    assert entries and all(e["parameters"] == "bl_mean" for e in entries.values())
    assert repair.config_entries("no_such_parameter") == {}


def _v1(path, fill):
    idx = pd.date_range("2026-01-01", periods=30, freq="min")
    for key in [
        "IsPulser_Baseline",
        "IsPulser_BlMean",
        "IsPulser_BlMean_var",
        "IsPulser_BlMean_pulser01anaRatio",
        "IsPulser_BlMeanFoo",
    ]:
        pd.DataFrame(fill, index=idx, columns=[1, 2]).to_hdf(path, key=key, mode="a")
    pd.DataFrame.from_dict({"unit": "ADC"}, orient="index", columns=["Value"]).to_hdf(
        path, key="IsPulser_BlMean_info", mode="a"
    )


def test_family_keys_match_the_camel_token_exactly(tmp_path):
    path = str(tmp_path / "v1.hdf")
    _v1(path, 1.0)
    keys = repair.family_keys(path, "BlMean")
    assert sorted(keys) == [
        "IsPulser_BlMean",
        "IsPulser_BlMean_info",
        "IsPulser_BlMean_pulser01anaRatio",
        "IsPulser_BlMean_var",
    ]  # neither Baseline nor BlMeanFoo


def test_transplant_replaces_only_the_named_keys(tmp_path):
    target, source = str(tmp_path / "target.hdf"), str(tmp_path / "source.hdf")
    _v1(target, 1.0)
    _v1(source, 7.0)
    keys = repair.family_keys(source, "BlMean")
    repair.transplant_keys(target, source, keys)
    assert float(pd.read_hdf(target, "IsPulser_BlMean").iloc[0, 0]) == 7.0
    assert float(pd.read_hdf(target, "IsPulser_Baseline").iloc[0, 0]) == 1.0
    assert pd.read_hdf(target, "IsPulser_BlMean_info").loc["unit", "Value"] == "ADC"
    assert not [p for p in os.listdir(tmp_path) if ".repack" in p]


def test_transplant_failure_leaves_target_untouched(tmp_path, monkeypatch):
    target, source = str(tmp_path / "target.hdf"), str(tmp_path / "source.hdf")
    _v1(target, 1.0)
    _v1(source, 7.0)
    before = os.path.getsize(target)

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_hdf", boom)
    with pytest.raises(RuntimeError):
        repair.transplant_keys(target, source, ["IsPulser_BlMean"])
    assert os.path.getsize(target) == before
    assert np.isclose(float(pd.read_hdf(target, "IsPulser_BlMean").iloc[0, 0]), 1.0)
