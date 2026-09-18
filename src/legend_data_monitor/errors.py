"""Typed exceptions for legend-data-monitor.

Library code raises these instead of calling ``sys.exit()``: a host process
(Panel dashboard, task runner, notebook) must receive a traceback it can log
or display, never a silent interpreter exit. Only the CLI translates them
into exit codes.
"""


class MonitoringError(Exception):
    """Base class for all legend-data-monitor errors."""


class ConfigError(MonitoringError):
    """Invalid or inconsistent user configuration / settings."""


class DataError(MonitoringError):
    """Requested data is missing, malformed, or could not be loaded."""


class CalibrationError(MonitoringError):
    """Calibration inputs are missing or inconsistent."""
