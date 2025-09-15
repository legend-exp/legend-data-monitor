import pandas as pd
from legend_data_monitor.utils import check_threshold

def make_series(values, start="2023-01-01", freq="D"):
    # fake datetime-indexed series
    idx = pd.date_range(start=start, periods=len(values), freq=freq, tz="UTC")
    return pd.Series(values, index=idx)

def test_check_threshold_within_limits():
    series = make_series([0.5, 0.6, 0.7])
    output = {"ch1": {"cal": {"fwhm_ok": True}, "phy": {}}}

    check_threshold(
        data_series=series,
        channel_name="ch1",
        last_checked=None,
        t0=[pd.Timestamp("2023-01-01")],
        threshold=[0.0, 1.0], 
        parameter="gain_stab",
        output=output,
    )

    assert output["ch1"]["phy"]["gain_stab"] is True


def test_check_threshold_out_of_bounds():
    series = make_series([2.5, 0.6, -1.0])
    output = {"ch1": {"cal": {"fwhm_ok": True}, "phy": {}}}

    check_threshold(
        data_series=series,
        channel_name="ch1",
        last_checked=None,
        t0=[pd.Timestamp("2023-01-01")],
        threshold=[-0.5, 2.0],  
        parameter="gain_stab",
        output=output,
    )

    assert output["ch1"]["phy"]["gain_stab"] is False


def test_check_threshold_pulser_stab_fwhm_fail():
    series = make_series([0.1, 0.2, 0.3]) 
    output = {"ch1": {"cal": {"fwhm_ok": False}, "phy": {}}}

    check_threshold(
        data_series=series,
        channel_name="ch1",
        last_checked=None,
        t0=[pd.Timestamp("2023-01-01")],
        threshold=[0.0, 1.0],
        parameter="pulser_stab",
        output=output,
    )

    assert output["ch1"]["phy"]["pulser_stab"] is False

    

def test_check_one_none_threshold():
    series = make_series([0.1, 0.2, 0.3]) 
    output = {"ch1": {"cal": {"fwhm_ok": True}, "phy": {}}}

    check_threshold(
        data_series=series,
        channel_name="ch1",
        last_checked=None,
        t0=[pd.Timestamp("2023-01-01")],
        threshold=[None, 1.0],
        parameter="pulser_stab",
        output=output,
    )

    assert output["ch1"]["phy"]["pulser_stab"] is True
    
    series = make_series([0.1, 0.2, 1.3]) 

    check_threshold(
        data_series=series,
        channel_name="ch1",
        last_checked=None,
        t0=[pd.Timestamp("2023-01-01")],
        threshold=[None, 1.0],
        parameter="pulser_stab",
        output=output,
    )

    assert output["ch1"]["phy"]["pulser_stab"] is False

def test_check_two_none_threshold():
    series = make_series([0.1, 0.2, 0.3]) 
    output = {"ch1": {"cal": {"fwhm_ok": True}, "phy": {}}}

    check_threshold(
        data_series=series,
        channel_name="ch1",
        last_checked=None,
        t0=[pd.Timestamp("2023-01-01")],
        threshold=[None, None],
        parameter="pulser_stab",
        output=output,
    )

    assert output["ch1"]["phy"]["pulser_stab"] is True
    
    series = make_series([0.1, 0.2, 1.3]) 

    check_threshold(
        data_series=series,
        channel_name="ch1",
        last_checked=None,
        t0=[pd.Timestamp("2023-01-01")],
        threshold=[None, None],
        parameter="pulser_stab",
        output=output,
    )

    assert output["ch1"]["phy"]["pulser_stab"] is True
    
