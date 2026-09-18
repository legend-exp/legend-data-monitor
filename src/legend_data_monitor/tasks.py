"""Compatibility shim (Phase 4): moved to :mod:`legend_data_monitor.orchestration.tasks`."""

from .orchestration.tasks import (  # noqa: F401
    EXIT_CONFIG_ERROR,
    EXIT_OK,
    EXIT_TASK_FAILED,
    Task,
    TaskResult,
    run_tasks,
)
