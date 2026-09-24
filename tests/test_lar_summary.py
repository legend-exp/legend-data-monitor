"""LAr veto performance keys from a synthetic evt tier."""

import numpy as np
import pandas as pd
from lgdo import lh5
from lgdo.types import Array, Table, VectorOfVectors

from legend_data_monitor import monitoring
from legend_data_monitor.contract import reader

RAWIDS = [1064000, 1064001, 1056002]


def _evt_file(path, n=600):
    rng = np.random.default_rng(7)
    t = 1_700_000_000.0 + np.sort(rng.uniform(0, 3 * 3600, n))
    forced = rng.uniform(size=n) < 0.1
    puls = ~forced & (rng.uniform(size=n) < 0.1)
    vetoed = np.where(forced, rng.uniform(size=n) < 0.05, rng.uniform(size=n) < 0.8)
    # channel 0 participates in 60 % of vetoed events, channel 2 never
    coin = [
        [[bool(vetoed[i] and rng.uniform() < p)] for p in (0.6, 0.4, 0.0)]
        for i in range(n)
    ]
    table = Table(
        {
            "trigger": Table({"timestamp": Array(t), "is_forced": Array(forced)}),
            "coincident": Table(
                {
                    "spms": Array(vetoed),
                    "puls": Array(puls),
                    "muon": Array(np.zeros(n, bool)),
                }
            ),
            "geds": Table({"multiplicity": Array(np.ones(n, np.int32))}),
            "spms": Table(
                {
                    "energy_sum": Array(rng.exponential(30, n).astype("float32")),
                    "multiplicity": Array(rng.integers(1, 20, n).astype("int32")),
                    "first_t0": Array(
                        np.where(rng.uniform(size=n) < 0.9, 1e4, np.nan).astype(
                            "float32"
                        )
                    ),
                    "geds_coincidence_classifier": Array(
                        rng.normal(2, 1, n).astype("float32")
                    ),
                    "is_trig_coin_pulse": VectorOfVectors(coin, dtype=bool),
                    "rawid": VectorOfVectors([RAWIDS] * n, dtype=np.uint32),
                }
            ),
        }
    )
    lh5.write(table, "evt", str(path), wo_mode="overwrite")
    return str(path)


def test_lar_summary_keys(tmp_path):
    evt = _evt_file(tmp_path / "evt.lh5")
    out = tmp_path / "out"
    written = monitoring.write_lar_summary(
        str(out),
        "p22",
        "r012",
        [evt],
        rawid_to_name={1064000: "S060", 1064001: "S061", 1056002: "S055"},
    )
    assert written == ["lar_veto/r012", "lar_occupancy/r012"]
    path = monitoring.period_contract_path(str(out), "p22")
    veto = reader.read_frame(path, "lar_veto/r012")
    assert isinstance(veto.index, pd.DatetimeIndex) and len(veto) == 4
    assert abs(veto["veto_frac"].mean() - 0.8) < 0.08
    assert abs(veto["accidental_frac"].mean() - 0.05) < 0.06
    assert veto["n_phys"].sum() < 600  # forced and pulser events excluded
    occ = reader.read_frame(path, "lar_occupancy/r012")
    assert list(occ.columns) == ["S060", "S061", "S055"]
    assert occ["S055"].eq(0).all() and (occ["S060"] > occ["S061"]).all()


def test_lar_summary_without_evt_data(tmp_path):
    assert monitoring.write_lar_summary(str(tmp_path), "p22", "r012", []) == []


def test_lar_thresholds_grade_period_keys(tmp_path):
    import yaml

    evt = _evt_file(tmp_path / "evt.lh5")
    root = tmp_path / "generated/plt/hit/phy"
    monitoring.write_lar_summary(
        str(root),
        "p22",
        "r012",
        [evt],
        rawid_to_name={1064000: "S060", 1064001: "S061", 1056002: "S055"},
    )
    (root / "p22/r012").mkdir(parents=True)
    graded = monitoring.check_spms_thresholds(str(root), "p22", "r012")
    assert graded["LAr"] == {"lar_veto_frac": True, "lar_accidental_frac": True}
    assert graded["S055"]["spms_occupancy"] is False  # never participates
    assert graded["S060"]["spms_occupancy"] is True
    with open(root / "p22/r012/l200-p22-r012-qcp_summary.yaml") as f:
        summary = yaml.safe_load(f)
    assert summary["LAr"]["phy"]["lar_veto_frac"] is True
