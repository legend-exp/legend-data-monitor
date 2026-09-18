"""Per-SiPM single-photoelectron spectra from the hit tier."""

import numpy as np
import pytest
from lgdo import lh5
from lgdo.types import Array, Table, VectorOfVectors

from legend_data_monitor import monitoring
from legend_data_monitor.contract import reader, writer

NAMES = {1064000: "S060", 1064001: "S061"}
KEY = "20260731T181831Z"


def _files(tmp_path, n=400, seed=3):
    """One hit + one evt file, row-aligned, with a known SPE spectrum."""
    rng = np.random.default_rng(seed)
    forced = rng.uniform(size=n) < 0.25
    evt = Table(
        {
            "trigger": Table(
                {
                    "timestamp": Array(1_700_000_000.0 + np.arange(n, dtype=float)),
                    "is_forced": Array(forced),
                }
            )
        }
    )
    evt_path = str(tmp_path / f"l200-p22-r012-phy-{KEY}-tier_evt.lh5")
    lh5.write(evt, "evt", evt_path, wo_mode="overwrite")

    hit_path = str(tmp_path / f"l200-p22-r012-phy-{KEY}-tier_hit.lh5")
    for rawid, centroid in ((1064000, 1.0), (1064001, 0.9)):
        # two pulses per event: one near the 1 p.e. peak, one sub-threshold
        pulses = [
            [float(rng.normal(centroid, 0.03)), float(rng.uniform(0.1, 0.4))]
            for _ in range(n)
        ]
        table = Table(
            {
                "energy_in_pe": VectorOfVectors(pulses, dtype=np.float32),
                "is_valid_hit": VectorOfVectors([[True, False]] * n, dtype=bool),
            }
        )
        lh5.write(table, "hit", hit_path, group=f"ch{rawid}", wo_mode="append")
    return hit_path, evt_path, forced


def test_spe_spectra_split_by_trigger_type(tmp_path):
    hit, evt, forced = _files(tmp_path)
    hists = monitoring.read_spe_spectra([hit], [evt], NAMES)
    assert set(hists) == {"IsBsln", "IsPhysics"}
    # 2 pulses per event per SiPM, split by the forced flag
    assert hists["IsBsln"].sum() == 2 * 2 * forced.sum()
    assert hists["IsPhysics"].sum() == 2 * 2 * (~forced).sum()

    hist = hists["IsBsln"]
    assert hist.axes[0].size == monitoring.SPE_BINS
    assert (hist.axes[0].edges[0], hist.axes[0].edges[-1]) == monitoring.SPE_RANGE
    assert sorted(hist.axes[1]) == ["S060", "S061"]
    # each SiPM's peak sits at its own centroid: the gain drift is visible
    for name, centroid in (("S060", 1.0), ("S061", 0.9)):
        counts = hist[:, hist.axes[1].index(name)].view()
        peak = hist.axes[0].centers[
            np.argmax(counts[int(0.6 / 0.02) :]) + int(0.6 / 0.02)
        ]
        assert peak == pytest.approx(centroid, abs=0.03)


def test_spe_spectra_keep_sub_threshold_pulses(tmp_path):
    """is_valid_hit is deliberately not applied: its threshold is below 1 p.e."""
    hit, evt, _ = _files(tmp_path)
    hist = monitoring.read_spe_spectra([hit], [evt], NAMES)["IsPhysics"]
    counts = hist[:, hist.axes[1].index("S060")].view()
    below = counts[: int(0.5 / 0.02)].sum()
    assert below > 0  # the masked-out pulses are in the histogram


def test_write_spe_spectrum_keys(tmp_path):
    hit, evt, _ = _files(tmp_path)
    run_dir = tmp_path / "p22" / "r012"
    run_dir.mkdir(parents=True)
    contract = str(run_dir / "l200-p22-r012-phy-spms-schema2.hdf")
    writer.write_frame(
        contract,
        "detector_map",
        __import__("pandas").DataFrame(
            [{"name": "S060", "rawid": 1064000}, {"name": "S061", "rawid": 1064001}]
        ),
    )
    written = monitoring.write_spe_spectrum(
        str(tmp_path), "p22", "r012", [hit], [evt], rawid_to_name=NAMES
    )
    assert written == [
        "hist/IsBsln_EnergyInPe_dist2d",
        "hist/IsPhysics_EnergyInPe_dist2d",
    ]
    hist, _, _, attrs = reader.read_hist(contract, "hist/IsBsln_EnergyInPe_dist2d")
    assert attrs["unit"] == "p.e." and "is_valid_hit" in attrs["selection"]
    assert hist.axes[0].size == 250


def test_write_spe_spectrum_without_contract(tmp_path):
    hit, evt, _ = _files(tmp_path)
    assert (
        monitoring.write_spe_spectrum(str(tmp_path), "p22", "r012", [hit], [evt]) == []
    )


def test_spe_spectra_skip_misaligned_tiers(tmp_path, caplog):
    hit, evt, _ = _files(tmp_path)
    short = Table(
        {
            "trigger": Table(
                {
                    "timestamp": Array(np.arange(5.0)),
                    "is_forced": Array(np.zeros(5, bool)),
                }
            )
        }
    )
    evt_short = str(tmp_path / f"l200-p22-r012-phy-{KEY}-tier_evt2.lh5")
    lh5.write(short, "evt", evt_short, wo_mode="overwrite")
    hists = monitoring.read_spe_spectra([hit], [evt_short], NAMES)
    # the evt file for this key has a different row count: nothing is filled
    assert hists["IsBsln"].sum() == 0 and hists["IsPhysics"].sum() == 0


def test_spe_keys_reach_the_manifest(tmp_path):
    """Keys added after the contract build must show up in the inventory."""
    import pandas as pd

    from legend_data_monitor.contract import build
    from legend_data_monitor.contract import reader as contract_reader

    hit, evt, _ = _files(tmp_path)
    root = tmp_path / "root"
    run_dir = root / "generated/plt/hit/phy/p22/r012"
    run_dir.mkdir(parents=True)
    contract = str(run_dir / "l200-p22-r012-phy-spms-schema2.hdf")
    writer.write_frame(
        contract,
        "detector_map",
        pd.DataFrame([{"name": "S060", "rawid": 1064000}]),
    )
    # a manifest that predates the SPE pass
    build.refresh_manifest(str(root), "p22", "r012")
    before = contract_reader.read_manifest(str(run_dir), "p22", "r012")
    assert before["files"]["l200-p22-r012-phy-spms-schema2.hdf"]["keys"] == [
        "detector_map"
    ]

    monitoring.write_spe_spectrum(
        str(root / "generated/plt/hit/phy"),
        "p22",
        "r012",
        [hit],
        [evt],
        rawid_to_name=NAMES,
    )
    after = contract_reader.read_manifest(str(run_dir), "p22", "r012")
    keys = after["files"]["l200-p22-r012-phy-spms-schema2.hdf"]["keys"]
    assert "hist/IsBsln_EnergyInPe_dist2d" in keys
    assert "hist/IsPhysics_EnergyInPe_dist2d" in keys
    assert "detector_map" in keys
