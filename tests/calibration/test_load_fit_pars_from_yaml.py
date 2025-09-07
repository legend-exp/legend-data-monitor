import pytest

from legend_data_monitor.calibration import load_fit_pars_from_yaml


@pytest.fixture
def fake_read_json_or_yaml(monkeypatch):
    # mock utils.read_json_or_yaml

    def fake_loader(file_path):
        return {
            "V11925A": {
                "results": {
                    "aoe": {
                        "1000-1300keV": {
                            "0": {
                                "mean": 1.0,
                                "mean_err": 0.1,
                                "sigma": 0.5,
                                "sigma_err": 0.05,
                            }
                        }
                    }
                }
            }
        }

    monkeypatch.setattr("legend_data_monitor.utils.read_json_or_yaml", fake_loader)
    return fake_loader


@pytest.fixture
def fake_deep_get(monkeypatch):
    # mock deep_get
    def fake_get(d, keys, default=None):
        return {
            "mean": 1.0,
            "mean_err": 0.1,
            "sigma": 0.5,
            "sigma_err": 0.05,
        }

    monkeypatch.setattr("legend_data_monitor.utils.deep_get", fake_get)
    return fake_get


def test_load_fit_pars_from_yaml_basic(fake_read_json_or_yaml, fake_deep_get, tmp_path):
    fake_file = tmp_path / "pars" / "r004" / "calibration_pars.yaml"
    fake_file.parent.mkdir(parents=True, exist_ok=True)
    fake_file.write_text("something")

    pars_files_list = [str(fake_file)]
    detectors_list = ["ch10000"]
    detectors_name = ["V11925A"]
    avail_runs = ["r004"]

    result = load_fit_pars_from_yaml(
        pars_files_list, detectors_list, detectors_name, avail_runs
    )

    assert "V11925A" in result
    assert "r004" in result["V11925A"]
    entry = result["V11925A"]["r004"]
    assert entry["mean"] == 1.0
    assert entry["mean_err"] == 0.1
    assert entry["sigma"] == 0.5
    assert entry["sigma_err"] == 0.05


def test_skips_runs_not_in_list(fake_read_json_or_yaml, fake_deep_get, tmp_path):
    fake_file = tmp_path / "pars" / "r010" / "calibration_pars.yaml"
    fake_file.parent.mkdir(parents=True, exist_ok=True)
    fake_file.write_text("something")

    pars_files_list = [str(fake_file)]
    detectors_list = ["ch10000"]
    detectors_name = ["V11925A"]
    avail_runs = ["r004"]

    result = load_fit_pars_from_yaml(
        pars_files_list, detectors_list, detectors_name, avail_runs
    )
    assert result is None
