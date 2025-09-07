import pytest

from legend_data_monitor import plot_styles, plotting
from legend_data_monitor.utils import check_plot_settings


def test_check_plot_settings_valid_config():
    # pick any real valid keys from the repo
    plot_structure_key = next(iter(plotting.PLOT_STRUCTURE.keys()))
    plot_style_key = next(iter(plot_styles.PLOT_STYLE.keys()))

    conf = {
        "subsystems": {
            "subsys1": {
                "plot1": {
                    "parameters": "some_param",
                    "plot_structure": plot_structure_key,
                    "plot_style": plot_style_key,
                    "time_window": "1d",
                }
            }
        }
    }

    assert check_plot_settings(conf) is True


def test_check_plot_settings_missing_subsystems(caplog):
    # Config with no subsystems
    conf = {}
    with pytest.raises(SystemExit):
        check_plot_settings(conf)


def test_check_plot_settings_invalid_option(caplog):
    # Config with an invalid plot_style
    plot_structure_key = next(iter(plotting.PLOT_STRUCTURE.keys()))

    conf = {
        "subsystems": {
            "subsys1": {
                "plot1": {
                    "parameters": "param",
                    "plot_structure": plot_structure_key,
                    "plot_style": "not_a_real_style",
                }
            }
        }
    }

    result = check_plot_settings(conf)
    assert result is False
    assert any("does not exist" in rec.message for rec in caplog.records)


def test_check_plot_settings_exposure_invalid_event_type():
    # Exposure plots must use event_type pulser or all
    plot_structure_key = next(iter(plotting.PLOT_STRUCTURE.keys()))

    conf = {
        "subsystems": {
            "subsys1": {
                "plot1": {
                    "parameters": "exposure",
                    "plot_structure": plot_structure_key,
                    "plot_style": "whatever",
                    "event_type": "bad_event",
                }
            }
        }
    }

    result = check_plot_settings(conf)
    assert result is False


def test_check_plot_settings_vs_time_missing_time_window():
    # Plots with vs time style must provide time_window
    plot_structure_key = next(iter(plotting.PLOT_STRUCTURE.keys()))
    # find a style that equals 'vs time' if it exists
    if "vs time" in plot_styles.PLOT_STYLE:
        style_key = "vs time"
    else:
        pytest.skip("No 'vs time' style available in plot_styles.PLOT_STYLE")

    conf = {
        "subsystems": {
            "subsys1": {
                "plot1": {
                    "parameters": "param",
                    "plot_structure": plot_structure_key,
                    "plot_style": style_key,
                }
            }
        }
    }

    result = check_plot_settings(conf)
    assert result is False


def test_check_plot_settings_skips_validation():
    # Test that exposure parameter skips further validation
    plot_structure_key = next(iter(plotting.PLOT_STRUCTURE.keys()))
    plot_style_key = next(iter(plot_styles.PLOT_STYLE.keys()))

    conf = {
        "subsystems": {
            "subsys1": {
                "plot1": {
                    "parameters": "exposure",
                    "event_type": "all",
                    "plot_structure": plot_structure_key,
                    "plot_style": plot_style_key,
                    # no time_window which would normally fail for vs time plots but should be skipped for exposure
                }
            }
        }
    }

    assert check_plot_settings(conf) is True

    conf = {
        "subsystems": {
            "subsys1": {
                "plot1": {
                    "parameters": "quality_cuts",
                    "plot_structure": plot_structure_key,
                    "plot_style": plot_style_key,
                }
            }
        }
    }

    assert check_plot_settings(conf) is True


def test_check_plot_settings_missing_field_for_regular_parameter(caplog):
    plot_structure_key = next(iter(plotting.PLOT_STRUCTURE.keys()))

    conf = {
        "subsystems": {
            "subsys1": {
                "plot1": {
                    "parameters": "regular_param",
                    # Missing "plot_style" field
                    "plot_structure": plot_structure_key,
                }
            }
        }
    }

    result = check_plot_settings(conf)
    assert result is False
    assert any(
        "Provide plot_style in plot settings of 'plot1' for subsys1!" in rec.message
        for rec in caplog.records
    )
    assert any("Available options:" in rec.message for rec in caplog.records)


def test_check_plot_settings_missing_field_does_not_trigger_for_exposure(caplog):
    # Test that missing fields don't trigger errors for exposure parameters
    conf = {
        "subsystems": {
            "subsys1": {
                "plot1": {
                    "parameters": "exposure",
                    "plot_structure": "per channel",
                    # Missing "plot_style" field
                    "event_type": "all",
                }
            }
        }
    }

    result = check_plot_settings(conf)
    assert result is True


def test_check_plot_settings_missing_field_does_not_trigger_for_quality_cuts(caplog):
    # Test that missing fields don't trigger errors for quality_cuts parameters
    conf = {
        "subsystems": {
            "subsys1": {
                "plot1": {
                    "parameters": "quality_cuts",
                    "plot_structure": "per channel",
                }
            }
        }
    }

    result = check_plot_settings(conf)
    assert result is True


def test_check_plot_settings_missing_field_with_valid_exposure(caplog):
    # Test exposure with valid settings but missing optional fields
    plot_structure_key = next(iter(plotting.PLOT_STRUCTURE.keys()))

    conf = {
        "subsystems": {
            "subsys1": {
                "plot1": {
                    "parameters": "exposure",
                    "plot_structure": plot_structure_key,
                    "event_type": "pulser",
                }
            }
        }
    }

    result = check_plot_settings(conf)
    assert result is True
    assert not any(
        "Provide plot_style in plot settings" in rec.message for rec in caplog.records
    )


def test_check_plot_settings_missing_field_with_valid_quality_cuts(caplog):
    # Test quality_cuts with valid settings but missing optional fields
    plot_structure_key = next(iter(plotting.PLOT_STRUCTURE.keys()))

    conf = {
        "subsystems": {
            "subsys1": {
                "plot1": {
                    "parameters": "quality_cuts",
                    "plot_structure": plot_structure_key,
                    # Missing "plot_style", but it's ok for quality_cuts
                }
            }
        }
    }

    result = check_plot_settings(conf)
    assert result is True
    assert not any(
        "Provide plot_style in plot settings" in rec.message for rec in caplog.records
    )


def test_check_plot_settings_missing_plot_structure_for_regular_param(caplog):
    # Test missing plot_structure for regular parameter
    plot_style_key = next(iter(plot_styles.PLOT_STYLE.keys()))

    conf = {
        "subsystems": {
            "subsys1": {
                "plot1": {
                    "parameters": "regular_param",  # NOT special parameter
                    # Missing "plot_structure"
                    "plot_style": plot_style_key,
                }
            }
        }
    }

    result = check_plot_settings(conf)
    assert result is False
    assert any(
        "Provide plot_structure in plot settings of 'plot1' for subsys1!" in rec.message
        for rec in caplog.records
    )
    assert any("Available options:" in rec.message for rec in caplog.records)


def test_check_plot_settings_missing_both_fields_for_regular_param(caplog):
    # Test missing both fields for regular parameter
    conf = {
        "subsystems": {
            "subsys1": {
                "plot1": {
                    "parameters": "regular_param",  # NOT special parameter
                    # Missing both "plot_structure" and "plot_style"
                }
            }
        }
    }

    result = check_plot_settings(conf)
    assert result is False
    assert any(
        "Provide " in rec.message
        and " in plot settings of 'plot1' for subsys1!" in rec.message
        for rec in caplog.records
    )
