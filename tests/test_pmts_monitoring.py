"""Muon-veto (pmts) flavour: contract build, muon summary, plot config."""

import numpy as np
import pandas as pd
import pytest
import yaml
from lgdo import lh5
from lgdo.types import Array, Table

from legend_data_monitor import monitoring, utils
from legend_data_monitor.contract import build, reader, writer
from legend_data_monitor.processing import binning

PERIOD, RUN = "p22", "r012"
PMTS = {
    "PMT101": {
        "daq_rawid": 2001604,
        "location": "pillbox",
        "processable": True,
        "usability": "on",
    },
    "PMT201": {
        "daq_rawid": 2001605,
        "location": "floor",
        "processable": True,
        "usability": "on",
    },
}


def _write_dsp(path, rawids, t0=1.7e9, n=50):
    rng = np.random.default_rng(1)
    for i, rawid in enumerate(rawids):
        table = Table(
            {
                "timestamp": Array(t0 + np.arange(n) * 12.0),
                "pulseHeight": Array(rng.normal(20 + 10 * i, 3, n).astype("float32")),
                "containsPulse": Array(np.tile([True, True, False], n)[:n]),
            }
        )
        lh5.write(table, "dsp", str(path), group=f"ch{rawid}", wo_mode="append")
    return str(path)


def _write_evt(path, t0=1.7e9, n=200):
    # one nested Table, matching the production evt layout
    table = Table(
        {
            "trigger": Table(
                {
                    "timestamp": Array(t0 + np.arange(n) * 3.0),
                    "is_forced": Array(np.zeros(n, dtype=bool)),
                }
            ),
            "coincident": Table(
                {
                    "muon": Array(np.arange(n) % 100 == 0),
                    "muon_offline": Array(np.zeros(n, dtype=bool)),
                }
            ),
        }
    )
    lh5.write(table, "evt", str(path), wo_mode="append")
    return str(path)


def _pmts_contract(run_dir):
    path = str(run_dir / f"l200-{PERIOD}-{RUN}-phy-pmts-schema2.hdf")
    idx = pd.date_range("2026-07-01", periods=30, freq="1min", tz="UTC")
    frame = pd.DataFrame(1000.0, index=idx, columns=list(PMTS))
    frame.index.name = "datetime"
    binned = binning.frame_to_binned(frame)
    writer.write_binned_series(path, "All", "BlMean", binned)
    writer.write_detector_map(path, PMTS, subsystem="pmts")
    return path


def test_pmts_contract_flavour(tmp_path, monkeypatch):
    run_dir = tmp_path / "generated/plt/hit/phy" / PERIOD / RUN
    run_dir.mkdir(parents=True)
    v1 = run_dir / f"l200-{PERIOD}-{RUN}-phy-pmts.hdf"
    idx = pd.date_range("2026-07-01", periods=40, freq="1min", tz="UTC")
    idx.name = "datetime"
    df = pd.DataFrame(
        np.random.default_rng(0).normal(1000, 1, (40, 2)),
        index=idx,
        columns=[i["daq_rawid"] for i in PMTS.values()],
    )
    df.to_hdf(v1, key="All_BlMean", mode="a")
    df.mean().to_frame().T.to_hdf(v1, key="All_BlMean_mean", mode="a")
    monkeypatch.setattr(utils, "build_pmts_info", lambda path: dict(PMTS))
    build.build_contract_files(
        str(tmp_path), PERIOD, RUN, metadata_path="meta", subsystem="pmts"
    )
    manifest = reader.read_manifest(str(run_dir), PERIOD, RUN)
    name = f"l200-{PERIOD}-{RUN}-phy-pmts-schema2.hdf"
    assert name in manifest["files"]
    keys = manifest["files"][name]["keys"]
    assert {"hist/All_BlMean/1min", "detector_map"} <= set(keys)
    det_map = pd.read_hdf(str(run_dir / name), "detector_map")
    assert (
        list(det_map.columns) == ["name", "rawid"] + writer.DETECTOR_MAP_COLUMNS["pmts"]
    )
    assert list(det_map["location"]) == ["pillbox", "floor"]
    binned = reader.read_binned_series(str(run_dir / name), "All", "BlMean", "1min")
    assert binned.detectors == ["PMT101", "PMT201"]


def test_muon_summary_keys_and_spectrum(tmp_path):
    dsp = _write_dsp(tmp_path / "dsp.lh5", [i["daq_rawid"] for i in PMTS.values()])
    evt = _write_evt(tmp_path / "evt.lh5")
    out = tmp_path / "out"
    run_dir = out / PERIOD / RUN
    run_dir.mkdir(parents=True)
    contract = _pmts_contract(run_dir)
    names = {i["daq_rawid"]: n for n, i in PMTS.items()}
    written = monitoring.write_muon_summary(
        str(out), PERIOD, RUN, [dsp], [evt], rawid_to_name=names
    )
    assert written == [
        f"muon_veto/{RUN}",
        "hist/All_Pulseheight_dist2d",
        "hist/All_MuonMultiplicity_dist",
        "hist/All_MuonLightSum_dist",
    ]
    frame = reader.read_frame(
        monitoring.period_contract_path(str(out), PERIOD), f"muon_veto/{RUN}"
    )
    # 50 triggers over 588 s in one hourly bin; both PMTs pulse 2/3 of triggers
    assert frame["muon_rate_hz"].iloc[0] == pytest.approx(50 / 3600, rel=1e-3)
    assert frame["multiplicity_median"].iloc[0] == 2.0
    assert frame["ge_coincidence_frac"].iloc[0] == pytest.approx(0.01)
    hist, _, _, _ = reader.read_hist(contract, "hist/All_Pulseheight_dist2d")
    v = np.asarray(hist.view())
    assert v.sum() == 100  # 50 triggers x 2 PMTs, all heights in (0, 100)
    assert list(hist.axes[1]) == ["PMT101", "PMT201"]


def test_muon_summary_without_pmts_rows(tmp_path):
    """A valid dsp file that simply holds no PMT tables yields no summary."""
    evt = _write_evt(tmp_path / "evt.lh5")
    dsp = tmp_path / "dsp.lh5"
    lh5.write(
        Table({"timestamp": Array(np.arange(3.0))}),
        "dsp",
        str(dsp),
        group="ch1104000",  # a geds table, no PMT groups
    )
    assert (
        monitoring.write_muon_summary(
            str(tmp_path), PERIOD, RUN, [str(dsp)], [evt], {2001604: "PMT101"}
        )
        == []
    )


def test_pmts_dict_is_valid():
    import importlib.resources

    pkg = importlib.resources.files("legend_data_monitor")
    with open(pkg / "settings" / "pmts-dict.yaml") as f:
        conf = {"subsystems": yaml.load(f, Loader=yaml.CLoader)}
    assert utils.check_plot_settings(conf)
    for plot in conf["subsystems"]["pmts"].values():
        assert plot["event_type"] == "all"  # pmts timestamps match no flags
