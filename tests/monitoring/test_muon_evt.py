import lh5
import numpy as np
from lgdo import Array, Table

from legend_data_monitor.monitoring import (
    apply_muon_cuts,
    get_muon_evt_data,
    muon_cuts_from_config,
    pe_per_unit,
)

CUTS = {
    "low_multiplicity": {"multiplicity": 11, "energy_sum": 7},
    "flashing": {"multiplicity": 13, "energy_sum": 98},
}


def test_apply_muon_cuts():
    multiplicity = np.array([2, 2, 15, 5, 20, 12, 11])
    energy_sum = np.array([3.0, 20.0, 3.0, 200.0, 200.0, 50.0, 1.0])

    passed = apply_muon_cuts(multiplicity, energy_sum, CUTS)

    # low-multiplicity bump, enough light, enough PMTs, flashing PMT,
    # bright muon, normal muon, multiplicity exactly at the low cut
    np.testing.assert_array_equal(passed, [False, True, True, False, True, True, True])


def test_get_muon_evt_data(tmp_path):
    f1, f2, f3 = (tmp_path / f"c{i}.lh5" for i in range(3))
    for path, mult in [(f1, [1, 12]), (f2, [20])]:
        tbl = Table(
            col_dict={
                "multiplicity": Array(np.array(mult)),
                "energy_sum": Array(np.array(mult, dtype=float) * 2),
                "is_muon": Array(np.array(mult) > 10),
                "timestamp": Array(np.arange(len(mult)) * 100.0),
            }
        )
        lh5.write(tbl, "evt_muon", str(path))
    # file without muon-DAQ events
    lh5.write(Table(col_dict={"x": Array(np.zeros(2))}), "evt", str(f3))

    data = get_muon_evt_data([str(f1), str(f2), str(f3)])

    np.testing.assert_array_equal(data["multiplicity"], [1, 12, 20])
    np.testing.assert_allclose(data["energy_sum"], [2.0, 24.0, 40.0])
    np.testing.assert_array_equal(data["is_muon"], [False, True, True])
    # only the file with two events spans some time
    assert data["duration"] == 100.0


def test_get_muon_evt_data_no_files():
    data = get_muon_evt_data([])

    assert all(len(data[f]) == 0 for f in ["multiplicity", "energy_sum", "is_muon"])
    assert data["duration"] == 0


def write_config(folder, expression):
    """Minimal dataflow config with a muon evt config valid from 2023."""
    (folder / "tier" / "evt").mkdir(parents=True)
    (folder / "validity.yaml").write_text(
        "- valid_from: 20230101T000000Z\n  mode: reset\n  apply:\n    - all-config.yaml\n"
    )
    (folder / "all-config.yaml").write_text(
        "snakemake_rules:\n  tier_evt:\n    inputs:\n"
        "      evt_config: $_/tier/evt/geds.yaml\n"
        "      muon_config:\n        evt_config: $_/tier/evt/muon.yaml\n"
    )
    (folder / "tier" / "evt" / "geds.yaml").write_text("operations: {}\n")
    (folder / "tier" / "evt" / "muon.yaml").write_text(
        "operations:\n  is_muon:\n    expression: >-\n      " + expression + "\n"
    )


def test_muon_cuts_from_config(tmp_path):
    write_config(
        tmp_path,
        "((evt.multiplicity >= 11) | (evt.energy_sum >= 7)) & "
        "((evt.multiplicity >= 13) | (evt.energy_sum <= 98)) & "
        "((evt.muon_noise_peaks>=4) | (evt._num_1pe>=0.5))",
    )

    cuts = muon_cuts_from_config(str(tmp_path), "20260101T000000Z")

    assert cuts["low_multiplicity"] == {"multiplicity": 11, "energy_sum": 7}
    assert cuts["flashing"] == {"multiplicity": 13, "energy_sum": 98}


def test_muon_cuts_from_config_unparsable(tmp_path):
    write_config(tmp_path, "evt.multiplicity > 3")

    assert muon_cuts_from_config(str(tmp_path), "20260101T000000Z") is None


def test_pe_per_unit(tmp_path):
    f = tmp_path / "c.lh5"
    # p18-p22: pulseHeight_in_pe = pulseHeight / 30, pulseHeight_trigd = pulseHeight / 15
    tbl = Table(
        col_dict={
            "pulseHeight_in_pe": Array(np.array([0.5, 1.0, 2.0])),
            "pulseHeight_trigd": Array(np.array([1.0, 2.0, 4.0])),
        }
    )
    lh5.write(tbl, "ch1/hit", str(f))
    assert pe_per_unit(str(f)) == 2.0

    g = tmp_path / "d.lh5"
    lh5.write(
        Table(col_dict={"pulseHeight_in_pe": Array(np.array([1.0]))}), "ch1/hit", str(g)
    )
    assert pe_per_unit(str(g)) == 1.0
