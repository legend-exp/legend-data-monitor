import pandas as pd

from legend_data_monitor.utils import check_threshold


def test_check_threshold_early_exit():
    # should return the original email_message due to None entries
    email_msg = []
    result = check_threshold(
        data_series=None,
        pswd_email=None,
        last_checked=None,
        t0=[],
        pars_data={},
        threshold=[0, 1],
        period="P03",
        current_run="r001",
        channel_name="ch1",
        string="s01",
        email_message=email_msg,
        parameter="gain",
    )
    assert result == email_msg

    email_msg = []
    result = check_threshold(
        data_series=None,
        pswd_email="abc",
        last_checked=None,
        t0=[],
        pars_data={},
        threshold=[0, 1],
        period="P03",
        current_run="r001",
        channel_name="ch1",
        string="s01",
        email_message=email_msg,
        parameter="gain",
    )
    assert result == email_msg

    email_msg = []
    result = check_threshold(
        data_series=None,
        pswd_email="abc",
        last_checked="0123",
        t0=[],
        pars_data={},
        threshold=[0, 1],
        period="P03",
        current_run="r001",
        channel_name="ch1",
        string="s01",
        email_message=email_msg,
        parameter="gain",
    )
    assert result == email_msg

    email_msg = []
    result = check_threshold(
        data_series="data",
        pswd_email="abc",
        last_checked="None",
        t0=[],
        pars_data={},
        threshold=[0, 1],
        period="P03",
        current_run="r001",
        channel_name="ch1",
        string="s01",
        email_message=email_msg,
        parameter="gain",
    )
    assert result == email_msg


def test_check_threshold_no_points_over_threshold():
    now = pd.Timestamp.utcnow()
    index = pd.date_range(now, periods=5, freq="D")
    series = pd.Series([0.5, 0.6, 0.7, 0.8, 0.9], index=index)

    email_msg = []
    t0 = [now - pd.Timedelta(days=1)]

    result = check_threshold(
        data_series=series,
        pswd_email="dummy",
        last_checked=(now - pd.Timedelta(days=2)).timestamp(),
        t0=t0,
        pars_data={},
        threshold=[-1, 2],  # no point outside
        period="P03",
        current_run="r001",
        channel_name="ch1",
        string="s01",
        email_message=email_msg,
        parameter="gain",
    )
    assert result == email_msg

    email_msg = []
    t0 = [now - pd.Timedelta(days=1)]

    result = check_threshold(
        data_series=series,
        pswd_email="dummy",
        last_checked=(now - pd.Timedelta(days=2)).timestamp(),
        t0=t0,
        pars_data={},
        threshold=[None, None],  # no threshold
        period="P03",
        current_run="r001",
        channel_name="ch1",
        string="s01",
        email_message=email_msg,
        parameter="gain",
    )
    assert result == email_msg


def test_check_threshold_points_over_threshold():
    now = pd.Timestamp.utcnow()
    index = pd.date_range(now, periods=5, freq="D")
    series = pd.Series([0.5, 1.5, 0.7, 2.0, 0.9], index=index)

    email_msg = []
    t0 = [now - pd.Timedelta(days=1)]

    result = check_threshold(
        data_series=series,
        pswd_email="dummy",
        last_checked=(now - pd.Timedelta(days=2)).timestamp(),
        t0=t0,
        pars_data={},
        threshold=[None, 1],  # points >1
        period="P03",
        current_run="r001",
        channel_name="ch1",
        string="s01",
        email_message=email_msg,
        parameter="gain",
    )
    # email_message should be updated
    assert len(result) > 0
    assert "ALERT" in result[0] or "gain over threshold" in result[1]
