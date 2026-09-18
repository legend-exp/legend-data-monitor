"""Logging contract for unattended runs (auto-giorgio).

Layout, one tree per invocation:

    <output>/generated/tmp/log/<YYYYMMDDTHHMMSSZ>/
        orchestrator.log                      one line per task start/end + ISSUES lines
        <task>/<task>-<period>-<run>.log      per-(task, run) full log

Machine-parseable blocks emitted into task logs:

    ERROR in task <task> (period=<p>, run=<r>):
    Traceback (most recent call last):
      ...
    <ExceptionClass>: <message>
    END ERROR

    ISSUE detector=<det> metric=<metric> severity=<sev> (period=<p>, run=<r>, datatype=<dt>):
      ...details...
    END ISSUE

Every saved figure is announced as ``SAVED_PLOT <absolute path>`` so a log
scanner can associate plots with the task that produced them.
"""

import logging
import os
import traceback
from datetime import datetime, timezone

LOG_FORMAT = "%(asctime)sZ %(levelname)s %(name)s %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def log_tree_root(output_folder: str, invocation_key: str | None = None) -> str:
    """Return (and create) the log-tree root for one invocation.

    Parameters
    ----------
    output_folder : str
        Pipeline output root (the tree containing ``generated/``).
    invocation_key : str, optional
        Timestamp key of this invocation; generated (UTC, compact ISO) if not
        given.
    """
    if invocation_key is None:
        invocation_key = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = os.path.join(output_folder, "generated/tmp/log", invocation_key)
    os.makedirs(root, exist_ok=True)
    return root


def _make_file_logger(name: str, file_path: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    # replace stale handlers from a previous task with the same name
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    formatter.converter = __import__("time").gmtime
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def orchestrator_logger(log_root: str) -> logging.Logger:
    """Logger writing one line per task start/end into orchestrator.log."""
    return _make_file_logger(
        "legend_data_monitor.orchestrator", os.path.join(log_root, "orchestrator.log")
    )


def task_logger(log_root: str, task: str, period: str, run: str) -> logging.Logger:
    """Logger with a dedicated per-(task, run) log file under the log tree."""
    task_dir = os.path.join(log_root, task)
    os.makedirs(task_dir, exist_ok=True)
    return _make_file_logger(
        f"legend_data_monitor.task.{task}.{period}.{run}",
        os.path.join(task_dir, f"{task}-{period}-{run}.log"),
    )


def format_error_block(task: str, period: str, run: str, exc: BaseException) -> str:
    """Render the parseable error block for a failed task."""
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return (
        f"ERROR in task {task} (period={period}, run={run}):\n"
        f"{tb.rstrip()}\n"
        "END ERROR"
    )


def log_saved_plot(logger: logging.Logger, path: str) -> None:
    """Announce a saved figure so log scanners can pick it up."""
    logger.info("SAVED_PLOT %s", os.path.abspath(path))


def save_figure(fig, path: str, logger: logging.Logger | None = None, **savefig_kwargs):
    """Save a matplotlib figure and announce it with a SAVED_PLOT line."""
    fig.savefig(path, **savefig_kwargs)
    if logger is not None:
        log_saved_plot(logger, path)
