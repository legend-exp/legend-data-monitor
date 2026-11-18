import os

import lh5
import numpy as np
import pytest
import yaml
from lgdo import Array, Table

from legend_data_monitor.monitoring import (
    classify_pmt_rates,
    compute_pmt_rates,
    find_cycle_outliers,
    find_pmt_bursts,
    find_threshold_crossings,
    get_pmt_counts,
    is_pmt_dead_in_run,
    limits_label,
    load_pmt_rate_history,
    pmt_cycle_history,
    pmt_rate_history,
    pmt_rate_limits,
    pmt_rates_file,
    pmt_run_mean_rates,
    pmt_window_references,
)


def write_pmt_file(path, channels):
    for ch, (pulses, timestamps) in channels.items():
        tbl = Table(
            col_dict={
                "containsPulse": Array(np.array(pulses, dtype=np.uint8)),
                "timestamp": Array(np.array(timestamps, dtype=float)),
            }
        )
        lh5.write(tbl, f"{ch}/hit", str(path), wo_mode="append")


def test_get_pmt_counts(tmp_path):
    f1, f2 = tmp_path / "c1.lh5", tmp_path / "c2.lh5"
    write_pmt_file(
        f1,
        {
            "ch1": ([1, 0, 1], [0.0, 50.0, 100.0]),
            "ch2": ([0, 0, 1], [0.0, 50.0, 100.0]),
        },
    )
    # ch2 missing from the second file
    write_pmt_file(f2, {"ch1": ([1, 1], [200.0, 400.0])})
    pmts = {"PMT1": {"channel_str": "ch1"}, "PMT2": {"channel_str": "ch2"}}

    counts, durations = get_pmt_counts([str(f1), str(f2)], pmts)

    np.testing.assert_array_equal(counts["PMT1"], [2, 2])
    np.testing.assert_array_equal(counts["PMT2"], [1, np.nan])
    np.testing.assert_allclose(durations, [100.0, 200.0])


def test_compute_pmt_rates():
    durations = np.array([360.0, 360.0])
    counts = {
        "PMT1": np.array([10.0, 20.0]),  # 100 and 200 pulses/h
        "PMT2": np.array([30.0, 30.0]),  # 300 pulses/h
        "PMT3": np.array([0.0, 0.0]),  # not usable
    }

    rates, mean_rate = compute_pmt_rates(counts, durations, usable=["PMT1", "PMT2"])

    np.testing.assert_allclose(rates["PMT1"], [100.0, 200.0])
    np.testing.assert_allclose(rates["PMT2"], [300.0, 300.0])
    np.testing.assert_allclose(rates["PMT3"], [0.0, 0.0])
    assert mean_rate == pytest.approx((150.0 + 300.0) / 2)


def test_compute_pmt_rates_missing_data():
    durations = np.array([360.0, np.nan, 360.0])
    counts = {
        "PMT1": np.array([10.0, 10.0, np.nan]),
        "PMT2": np.full(3, np.nan),
    }

    rates, mean_rate = compute_pmt_rates(counts, durations, ["PMT1", "PMT2"])

    np.testing.assert_allclose(rates["PMT1"], [100.0])
    assert rates["PMT2"] is None
    assert mean_rate == pytest.approx(100.0)


def test_compute_pmt_rates_no_usable():
    rates, mean_rate = compute_pmt_rates({"PMT1": np.array([10.0])}, [360.0], [])

    np.testing.assert_allclose(rates["PMT1"], [100.0])
    assert np.isnan(mean_rate)


def test_is_pmt_dead_in_run():
    durations = np.full(10, 3600.0)
    # ~1 pulse per minute, then silent for the last 4 cycles
    counts = np.array([60, 58, 61, 59, 60, 62, 0, 0, 0, 0], dtype=float)

    assert is_pmt_dead_in_run(counts, durations)


def test_is_pmt_dead_in_run_alive_or_too_short():
    durations = np.full(10, 3600.0)
    alive = np.full(10, 60.0)
    short_gap = np.array([60, 58, 61, 59, 60, 62, 60, 59, 0, 0], dtype=float)

    assert not is_pmt_dead_in_run(alive, durations)
    assert not is_pmt_dead_in_run(short_gap, durations)


def test_is_pmt_dead_in_run_low_rate():
    durations = np.full(10, 3600.0)
    # ~1 pulse per hour: 3 empty hours are not significant
    counts = np.array([1, 2, 0, 1, 1, 2, 1, 0, 0, 0], dtype=float)

    assert not is_pmt_dead_in_run(counts, durations)


def test_is_pmt_dead_in_run_drop_to_noise():
    durations = np.full(30, 3600.0)
    # ~120 pulses per hour, a burst, then only dark counts
    counts = np.array([120] * 8 + [1440] + [5] * 21, dtype=float)

    assert is_pmt_dead_in_run(counts, durations)


def test_is_pmt_dead_in_run_burst_at_start():
    durations = np.full(10, 3600.0)
    counts = np.array([6000] + [100] * 9, dtype=float)

    assert not is_pmt_dead_in_run(counts, durations)


def test_is_pmt_dead_in_run_never_alive():
    assert not is_pmt_dead_in_run(np.zeros(5), np.full(5, 3600.0))


def test_compute_pmt_rates_median_robust():
    durations = np.array([3600.0])
    # one broken PMT still 'on' must not pull the reference down
    counts = {"A": np.array([100.0]), "B": np.array([110.0]), "C": np.array([0.0])}

    _, ref = compute_pmt_rates(counts, durations, usable=["A", "B", "C"])

    assert ref == pytest.approx(100.0)


def test_compute_pmt_rates_short_cycles():
    durations = np.array([3600.0, 3600.0, 300.0])
    counts = {"A": np.array([100.0, 110.0, 30.0])}

    rates, ref = compute_pmt_rates(counts, durations, ["A"], min_fraction=0.5)

    # the 5-min cycle is left out; the reference is the median cycle rate
    assert rates["A"] == pytest.approx([100.0, 110.0])
    assert ref == pytest.approx(105.0)


def test_compute_pmt_rates_burst_robust():
    durations = np.full(5, 3600.0)
    # a flasher cycle must not pull the reference up
    counts = {"A": np.array([100.0, 100.0, 6000.0, 100.0, 100.0])}

    _, ref = compute_pmt_rates(counts, durations, ["A"])

    assert ref == pytest.approx(100.0)


def test_pmt_run_mean_rates():
    rates = pmt_run_mean_rates(
        {"A": np.array([10.0, 30.0]), "B": np.full(2, np.nan)}, np.array([360.0, 360.0])
    )

    assert rates["A"] == pytest.approx(200.0)
    assert np.isnan(rates["B"])


def test_pmt_rate_limits():
    limits = pmt_rate_limits(
        {"Wall": 100.0, "Floor": np.nan}, [3600.0, 3600.0, 60.0], 5
    )

    assert limits["Wall"] == pytest.approx((50.0, 150.0))
    assert np.isnan(limits["Floor"][0])


def test_classify_pmt_rates():
    pmts = {
        "ok": {"string": "Wall", "usability": "on", "died_in_run": False},
        "low": {"string": "Wall", "usability": "on", "died_in_run": False},
        "high": {"string": "Wall", "usability": "on", "died_in_run": False},
        "nodata": {"string": "Wall", "usability": "on", "died_in_run": False},
        "dropout": {"string": "Wall", "usability": "on", "died_in_run": True},
        "off": {"string": "Wall", "usability": "off", "died_in_run": False},
    }
    means = {
        "ok": 100,
        "low": 40,
        "high": 160,
        "nodata": np.nan,
        "dropout": 30,
        "off": 0,
    }

    status = classify_pmt_rates(means, pmts, {"Wall": (50.0, 150.0)})

    assert status == {
        "ok": "ok",
        "low": "warning",
        "high": "warning",
        "nodata": "no data",
        "dropout": "dropped out",
        "off": "off",
    }


def test_limits_label():
    assert limits_label(5) == r"$\pm$5$\sigma$ threshold"


def test_find_threshold_crossings():
    values = np.array([100, 80, 70, 60, 90, np.nan, 70, 140])
    low = 0.75 * np.full(len(values), 100.0)

    assert find_threshold_crossings(values, low, 125.0) == [2, 6]
    assert find_threshold_crossings([50, 60, 100], 75.0, 125.0) == [0]


def test_find_cycle_outliers():
    durations = np.full(10, 3600.0)
    steady = np.full(10, 100.0)
    burst = steady.copy()
    burst[3] = 400
    dip = steady.copy()
    dip[5] = 20

    assert find_cycle_outliers(steady, durations, 50, 150) == (False, False)
    assert find_cycle_outliers(burst, durations, 50, 150) == (False, True)
    assert find_cycle_outliers(dip, durations, 50, 150) == (True, False)
    # a short last cycle with few counts is not checked
    short = np.append(steady, 0.0)
    assert find_cycle_outliers(short, np.append(durations, 60.0), 50, 150) == (
        False,
        False,
    )


def test_find_pmt_bursts():
    durations = np.full(5, 3600.0)
    counts = {
        "A": np.array([100, 100, 1000, 100, 100], dtype=float),
        "B": np.array([110, 110, 1100, 110, 110], dtype=float),
        "C": np.array([90, 95, 100, 92, 91], dtype=float),
    }
    pmts = {"A": {"string": "Floor"}, "B": {"string": "Floor"}, "C": {"string": "Wall"}}

    bursts = find_pmt_bursts(counts, durations, pmts, ["A", "B", "C"])

    assert bursts["Floor"] == pytest.approx(10.0)
    assert bursts["Wall"] < 1.2


def write_summary(folder, period, run, rate_403):
    summary = {
        "period": period,
        "run": run,
        "last_cycle": "20260101T000000Z",
        "references": {"Wall": 115.0},
        "threshold_sigma": 5,
        "limits": {"Wall": [61.4, 168.6]},
        "cycles": {
            "keys": [f"20260101T{h:02d}0000Z" for h in range(3)],
            "durations": [3600.0] * 3,
        },
        "bursts": {"Wall": 1.1},
        "pmts": {
            "PMT403": {
                "location": "Wall",
                "usability": "on",
                "died_in_run": False,
                "status": "ok",
                "run_mean": rate_403,
                "min": rate_403 - 20,
                "max": rate_403 + 20,
                "cycle_rates": [rate_403 - 5, rate_403, rate_403 + 5],
            },
            "PMT404": {
                "location": "Wall",
                "usability": "off",
                "died_in_run": False,
                "status": "off",
                "run_mean": 0.0,
                "min": 0.0,
                "max": 0.0,
            },
        },
    }
    path = pmt_rates_file(str(folder), period, run, "phy")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(summary, f)


def test_load_pmt_rate_history(tmp_path):
    for i, run in enumerate(["r000", "r001", "r002", "r003"]):
        write_summary(tmp_path, "p22", run, 110.0 - 10 * i)
    write_summary(tmp_path, "p21", "r009", 120.0)

    hist = load_pmt_rate_history(str(tmp_path), "p22", "r002", "phy", n_runs=3)

    assert [f"{h['period']}-{h['run']}" for h in hist] == [
        "p21-r009",
        "p22-r000",
        "p22-r001",
        "p22-r002",
    ][-3:]


def test_pmt_rate_history(tmp_path):
    for i, run in enumerate(["r000", "r001", "r002"]):
        write_summary(tmp_path, "p22", run, 110.0 - 10 * i)

    shown = pmt_rate_history(str(tmp_path), "p22", "r002", save_pdf=True)

    assert shown == ["p22-r000", "p22-r001", "p22-r002"]
    pdf = tmp_path / "p22" / "r002" / "mtg" / "pdf"
    assert (pdf / "p22_r002_pmt_rate_history.pdf").exists()
    assert (pdf / "p22_r002_pmt_rate_history_per_pmt.pdf").exists()


def test_pmt_cycle_history(tmp_path):
    for i, run in enumerate(["r000", "r001", "r002"]):
        write_summary(tmp_path, "p22", run, 110.0 - 10 * i)

    shown = pmt_cycle_history(str(tmp_path), "p22", "r002", save_pdf=True)

    # all 3 x 3 cycles (less than one week)
    assert shown == 9
    pdf = tmp_path / "p22" / "r002" / "mtg" / "pdf"
    assert (pdf / "p22_r002_pmt_cycle_history_per_pmt.pdf").exists()


def test_pmt_cycle_history_without_cycles(tmp_path):
    write_summary(tmp_path, "p22", "r000", 110.0)
    path = pmt_rates_file(str(tmp_path), "p22", "r000", "phy")
    with open(path) as f:
        summary = yaml.safe_load(f)
    del summary["cycles"]
    with open(path, "w") as f:
        yaml.safe_dump(summary, f)

    assert pmt_cycle_history(str(tmp_path), "p22", "r000") is None


def test_pmt_window_references(tmp_path):
    # previous run: 3 stored cycles of PMT403 at 105, 110, 115 /h
    write_summary(tmp_path, "p22", "r000", 110.0)
    pmts = {"PMT403": {"string": "Wall"}, "PMT404": {"string": "Wall"}}
    current = {"PMT403": np.array([60.0, 60.0]), "PMT404": None}

    refs, durs = pmt_window_references(
        str(tmp_path),
        "p22",
        "r001",
        "phy",
        current,
        [3600.0, 3600.0],
        pmts,
        ["PMT403", "PMT404"],
    )

    # window = 3 previous + 2 current cycles: median of 105, 110, 115, 60, 60
    assert refs["Wall"] == pytest.approx(105.0)
    assert len(durs) == 5
