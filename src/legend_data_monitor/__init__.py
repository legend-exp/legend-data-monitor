"""LEGEND data monitoring package.

Submodules and the public classes are imported lazily: importing
``legend_data_monitor`` must stay cheap (no matplotlib/pygama/YAML parsing)
so that consumers embedding it pay only for what they use.
"""

from importlib import import_module

from legend_data_monitor._version import version as __version__

_LAZY_MODULES = {
    "analysis_data",
    "automatic_run",
    "calibration",
    "config",
    "contract",
    "core",
    "errors",
    "issues",
    "loading",
    "logs",
    "monitoring",
    "orchestration",
    "plot_styles",
    "plotting",
    "processing",
    "save_data",
    "slow_control",
    "string_visualization",
    "subsystem",
    "tasks",
    "utils",
}
_LAZY_ATTRS = {
    "AnalysisData": ("legend_data_monitor.analysis_data", "AnalysisData"),
    "control_plots": ("legend_data_monitor.core", "control_plots"),
    "SlowControl": ("legend_data_monitor.slow_control", "SlowControl"),
    "Subsystem": ("legend_data_monitor.subsystem", "Subsystem"),
}

__all__ = [
    "__version__",
    "AnalysisData",
    "SlowControl",
    "Subsystem",
    "automatic_run",
    "calibration",
    "control_plots",
    "monitoring",
    "plot_styles",
    "plotting",
    "utils",
]


def __getattr__(name):
    if name in _LAZY_MODULES:
        return import_module(f"legend_data_monitor.{name}")
    if name in _LAZY_ATTRS:
        module, attr = _LAZY_ATTRS[name]
        return getattr(import_module(module), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
