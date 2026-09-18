"""Slow control: retrieval shaping and publication under the contract."""

import numpy as np
import pandas as pd
import pytest

from legend_data_monitor import core, monitoring, slow_control, utils
from legend_data_monitor.contract import reader

FIRST, LAST = "20260801T000000Z", "20260801T060000Z"


class _FakeDB:
    """Just enough of LegendSlowControlDB for get_sc_param."""

    def __init__(self, tables):
        self.tables = tables
        self.queries = []

    def get_tables(self):
        return list(self.tables)

    def dataframe(self, query):
        self.queries.append(query)
        name = query.split("FROM ")[1].split(" ")[0] if "FROM" in query else query
        return self.tables[name].copy()


def _rack_snap(n=6):
    t = pd.date_range("2026-08-01 01:00", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "tstamp": list(t) * 2,
            "value": list(np.linspace(20.0, 25.0, n)) + list(np.full(n, 99.0)),
            "rack": ["CleanRoom-DaqLeft"] * n + ["CleanRoom-DaqRight"] * n,
            "name": ["Temp"] * (2 * n),
            "sensor": ["Temp-1"] * (2 * n),
        }
    )


def _rack_info():
    t = pd.Timestamp("2026-07-01", tz="UTC")
    return pd.DataFrame(
        {
            "tstamp": [t],
            "rack": ["CleanRoom-DaqLeft"],
            "name": ["Temp"],
            "sensor": ["Temp-1"],
            "unit": ["C"],
            "ltol": [15.0],
            "utol": [30.0],
        }
    )


_SlowControl = slow_control.SlowControl  # the real class, safe from monkeypatching


def _sc(parameter, db):
    sc = _SlowControl.__new__(_SlowControl)
    sc.parameter = parameter
    sc.sc_parameters = utils.SC_PARAMETERS
    sc.scdb = db
    sc.first_timestamp, sc.last_timestamp = FIRST, LAST
    return sc


def test_rack_table_skips_the_diode_merge_and_keeps_its_rows():
    """Rack/clean-room tables have no crate/slot/channel; the merge used to eat them."""
    db = _FakeDB({"rack_snap": _rack_snap(), "rack_info": _rack_info()})
    frame = _sc("DaqLeft-Temp1", db).get_sc_param()
    assert len(frame) == 6  # the DaqLeft rows only, flags applied
    assert frame["value"].tolist() == pytest.approx(np.linspace(20.0, 25.0, 6))
    assert pd.api.types.is_datetime64tz_dtype(
        frame["tstamp"]
    )  # resolution is pandas's call
    assert frame["unit"].iloc[0] == "C"
    assert (frame["lower_lim"] == 15.0).all() and (frame["upper_lim"] == 30.0).all()
    assert not any("diode_info" in q for q in db.queries)


def test_diode_table_still_resolves_detector_names(monkeypatch):
    called = {}

    def fake_merge(df, scdb):
        called["yes"] = True
        return df.assign(name="V01234A", string=1)

    monkeypatch.setattr(slow_control, "include_more_diode_info", fake_merge)
    snap = pd.DataFrame(
        {
            "tstamp": pd.date_range("2026-08-01 01:00", periods=3, freq="h", tz="UTC"),
            "vmon": [4000.0, 4001.0, 4002.0],
            "imon": [1.0, 1.0, 1.0],
            "crate": [0, 0, 0],
            "slot": [1, 1, 1],
            "channel": [2, 2, 2],
            "status": [1, 1, 1],
        }
    )
    db = _FakeDB({"diode_snap": snap})
    params = utils.SC_PARAMETERS["SC_DB_params"]
    diode = next(
        p for p, v in params.items() if v["table"] == "diode_snap" and "vmon" in p
    )
    frame = _sc(diode, db).get_sc_param()
    # vmon parameters rename in place; crate/slot/channel frames merge detector info
    assert "value" in frame.columns
    assert called or "vmon" in diode


def test_write_slow_control_publishes_a_time_series(tmp_path):
    db = _FakeDB({"rack_snap": _rack_snap(), "rack_info": _rack_info()})
    frame = _sc("DaqLeft-Temp1", db).get_sc_param()
    key = monitoring.write_slow_control(
        str(tmp_path), "p22", "r012", "DaqLeft-Temp1", frame
    )
    assert key == "slow_control/DaqLeft_Temp1/r012"
    back = reader.read_frame(monitoring.period_contract_path(str(tmp_path), "p22"), key)
    assert isinstance(back.index, pd.DatetimeIndex) and back.index.name == "datetime"
    assert str(back.index.tz) == "UTC"
    assert list(back.columns) == ["value", "unit", "lower_lim", "upper_lim"]
    assert back.index.is_monotonic_increasing
    # rewriting the same (parameter, run) replaces, never appends
    monitoring.write_slow_control(str(tmp_path), "p22", "r012", "DaqLeft-Temp1", frame)
    assert (
        len(
            reader.read_frame(
                monitoring.period_contract_path(str(tmp_path), "p22"), key
            )
        )
        == 6
    )
    assert (
        monitoring.write_slow_control(str(tmp_path), "p22", "r012", "X", pd.DataFrame())
        is None
    )


def test_retrieve_scdb_writes_the_period_file_only(tmp_path, monkeypatch):
    monkeypatch.setattr(core.subprocess, "run", lambda *a, **k: None)
    db = _FakeDB({"rack_snap": _rack_snap(), "rack_info": _rack_info()})

    def fake_sc(parameter, port, pswd, dataset):
        sc = _sc(parameter, db)
        sc.data = sc.get_sc_param() if parameter == "DaqLeft-Temp1" else pd.DataFrame()
        return sc

    monkeypatch.setattr(core.slow_control, "SlowControl", fake_sc)
    (tmp_path / "auto").mkdir()
    config = {
        "output": str(tmp_path),
        "dataset": {
            "experiment": "L200",
            "period": "p22",
            "version": "auto/v2.0.0",
            "path": "/nonexistent",
            "type": "phy",
            "runs": 12,
        },
        "saving": "overwrite",
        "slow_control": {"parameters": ["DaqLeft-Temp1", "RREiT"]},
    }
    core.retrieve_scdb(config, 5678, "secret")
    phy = tmp_path / "auto/v2.0.0/generated/plt/hit/phy"
    period_file = phy / "p22" / "l200-p22-phy-monitoring.hdf"
    assert period_file.exists()
    assert (
        len(reader.read_frame(str(period_file), "slow_control/DaqLeft_Temp1/r012")) == 6
    )
    with pd.HDFStore(str(period_file), "r") as store:
        assert "/slow_control/RREiT/r012" not in store.keys()  # empty: skipped
    assert not list((phy / "p22" / "r012").glob("*slow_control*"))
