"""spms_noise / spms_calibration period keys from production by-products."""

import os

import pandas as pd
import pytest
import yaml

from legend_data_monitor import monitoring
from legend_data_monitor.contract import reader


def _mock_prod(tmp_path):
    par = tmp_path / "generated/par/dsp/phy/p22/r012"
    par.mkdir(parents=True)
    for key, fwhm in [("20260731T181831Z", 0.40), ("20260731T191833Z", 0.42)]:
        (par / f"l200-p22-r012-phy-{key}-par_dsp_spms.yaml").write_text(
            yaml.safe_dump({"S002": {"baseline_curr_fwhm": fwhm}, "S003": {}})
        )
    ovr = tmp_path / "inputs/dataprod/overrides/hit"
    (ovr / "lar/p19/r005").mkdir(parents=True)
    (ovr / "lar/p19/r005/l200-p19-r005-phy-lar-T%-par_hit-overwrite.yaml").write_text(
        yaml.safe_dump(
            {
                "S002": {
                    "pars": {
                        "operations": {
                            "energy_in_pe": {"parameters": {"a": 0.02, "m": 0.7}},
                            "is_valid_hit": {"parameters": {"a": 0.5}},
                        }
                    }
                }
            }
        )
    )
    (ovr / "lar/p15/r004").mkdir(parents=True)
    (ovr / "lar/p15/r004/l200-p15-r004-lar-T%-par_hit-overwrite.yaml").write_text(
        yaml.safe_dump(
            {
                "S003": {
                    "pars": {
                        "operations": {
                            "energy_in_pe": {"parameters": {"a": 0.01, "m": 0.6}},
                            "is_valid_hit": {"parameters": {"a": 0.4}},
                        }
                    }
                }
            }
        )
    )
    (ovr / "validity.yaml").write_text(
        yaml.safe_dump(
            [
                {
                    "valid_from": "20240101T000000Z",
                    "mode": "reset",
                    "apply": [
                        "lar/p15/r004/l200-p15-r004-lar-T%-par_hit-overwrite.yaml"
                    ],
                },
                {
                    "valid_from": "20260116T191157Z",
                    "mode": "append",
                    "apply": [
                        "lar/p19/r005/l200-p19-r005-phy-lar-T%-par_hit-overwrite.yaml"
                    ],
                },
            ]
        )
    )
    return str(tmp_path)


def test_spms_production_keys(tmp_path):
    prod = _mock_prod(tmp_path)
    out = tmp_path / "out"
    keys = monitoring.write_spms_production_keys(
        str(out), "p22", "r012", prod, start_key="20260731T181831Z"
    )
    assert keys == ["spms_noise/r012", "spms_calibration/r012"]
    path = monitoring.period_contract_path(str(out), "p22")
    noise = reader.read_frame(path, "spms_noise/r012")
    assert list(noise.columns) == ["S002", "S003"]
    assert noise["S002"].tolist() == pytest.approx([0.40, 0.42])
    assert noise["S003"].isna().all() and isinstance(noise.index, pd.DatetimeIndex)
    calib = reader.read_frame(path, "spms_calibration/r012")
    assert calib.loc["S002", "pe_m"] == 0.7 and calib.loc["S002", "threshold_a"] == 0.5
    assert calib.loc["S002", "source"].startswith("lar/p19/r005/")


def test_spms_production_keys_absent_inputs(tmp_path):
    assert (
        monitoring.write_spms_production_keys(
            str(tmp_path), "p22", "r012", str(tmp_path)
        )
        == []
    )
    assert not os.path.exists(monitoring.period_contract_path(str(tmp_path), "p22"))


def test_calibration_source_is_per_sipm(tmp_path):
    """An override touches only the channels it lists, so sources differ per SiPM."""
    prod = _mock_prod(tmp_path)
    calib = monitoring.read_spms_calibration(prod, "20260731T181831Z")
    # S002 is defined only by the newest file, S003 only by the p15 reset file
    assert calib.loc["S002", "source"].startswith("lar/p19/r005/")
    assert calib.loc["S003", "source"].startswith("lar/p15/r004/")
    # values still come from the merged parameter database
    assert calib.loc["S003", "pe_m"] == 0.6 and calib.loc["S002", "pe_m"] == 0.7


def test_calibration_ignores_out_of_root_validity_entries(tmp_path):
    """The hit validity list references ../raw/... files that hold no SiPM pars.

    Following them is what made the parameter-database lookup raise
    ``TypeError: 'NoneType' object is not iterable`` on every p16/p18 run,
    failing build_monitoring_hdf and so the whole run with rc=1.
    """
    prod = _mock_prod(tmp_path)
    validity = tmp_path / "inputs/dataprod/overrides/hit/validity.yaml"
    entries = yaml.safe_load(validity.read_text())
    entries.append(
        {
            "valid_from": "20260201T000000Z",
            "mode": "append",
            "apply": [
                "../raw/cal/p15/r002/l200-p15-r002-cal-T%-par_raw-overwrite.yaml",
                "../raw/cal/p16/r000/l200-p16-r000-cal-T%-par_raw-overwrite.yaml",
            ],
        }
    )
    validity.write_text(yaml.safe_dump(entries))
    calib = monitoring.read_spms_calibration(prod, "20260731T181831Z")
    assert sorted(calib.index) == ["S002", "S003"]
    assert calib.loc["S002", "pe_m"] == 0.7


def test_calibration_merges_split_definitions_deeply(tmp_path):
    """A later file may redefine only part of a channel's pars.

    Real case: S054 and S087 on p22 take energy_in_pe from p15/r004 and
    is_valid_hit from a later override; a shallow merge loses the gain.
    """
    prod = _mock_prod(tmp_path)
    ovr = tmp_path / "inputs/dataprod/overrides/hit"
    (ovr / "lar/p16/r000").mkdir(parents=True)
    (ovr / "lar/p16/r000/l200-p16-r000-phy-lar-T%-par_hit-overwrite.yaml").write_text(
        yaml.safe_dump(
            {
                "S003": {
                    "pars": {"operations": {"is_valid_hit": {"parameters": {"a": 0.9}}}}
                }
            }
        )
    )
    validity = ovr / "validity.yaml"
    entries = yaml.safe_load(validity.read_text())
    entries.append(
        {
            "valid_from": "20250101T000000Z",
            "mode": "append",
            "apply": ["lar/p16/r000/l200-p16-r000-phy-lar-T%-par_hit-overwrite.yaml"],
        }
    )
    validity.write_text(yaml.safe_dump(entries))
    calib = monitoring.read_spms_calibration(prod, "20260731T181831Z")
    # the newer file supplies the threshold, the older one still supplies the gain
    assert calib.loc["S003", "threshold_a"] == 0.9
    assert calib.loc["S003", "pe_m"] == 0.6
    assert calib.loc["S003", "source"].startswith("lar/p16/r000/")


def test_production_keys_survive_a_broken_override_tree(tmp_path, monkeypatch):
    """A malformed override tree must not fail the task: geds output is done."""
    prod = _mock_prod(tmp_path)

    def boom(*args, **kwargs):
        raise TypeError("'NoneType' object is not iterable")

    monkeypatch.setattr(monitoring, "read_spms_calibration", boom)
    out = tmp_path / "out"
    keys = monitoring.write_spms_production_keys(
        str(out), "p22", "r012", prod, start_key="20260731T181831Z"
    )
    assert keys == ["spms_noise/r012"]  # the noise key still lands


def test_wildcard_entry_resolves_its_own_basename(tmp_path):
    """A T% validity entry must never pick up an unrelated yaml in the directory."""
    prod = _mock_prod(tmp_path)
    d = tmp_path / "inputs/dataprod/overrides/hit/lar/p19/r005"
    # an unrelated file that sorts first, and an older timestamped variant
    (d / "aaa-unrelated.yaml").write_text(yaml.safe_dump({"S002": {"pars": {}}}))
    real = d / "l200-p19-r005-phy-lar-T%-par_hit-overwrite.yaml"
    older = d / "l200-p19-r005-phy-lar-20260101T000000Z-par_hit-overwrite.yaml"
    newer = d / "l200-p19-r005-phy-lar-20260201T000000Z-par_hit-overwrite.yaml"
    older.write_text(real.read_text().replace("0.7", "0.1"))
    newer.write_text(real.read_text())
    real.unlink()  # force the wildcard fallback
    calib = monitoring.read_spms_calibration(prod, "20260731T181831Z")
    assert calib.loc["S002", "pe_m"] == 0.7  # the newest matching variant
