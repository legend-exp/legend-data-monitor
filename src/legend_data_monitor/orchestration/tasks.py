"""Task registry and runner for unattended pipeline invocations.

Each orchestration unit (subsystem plots, monitoring HDF build, calibration
checks, ...) runs as an isolated task: it gets its own log file under the
invocation's log tree, its exceptions are caught, rendered as a parseable
error block, and do not stop the remaining tasks.

Exit-code policy (owned by the CLI, exposed here as the run result):
    0  all tasks succeeded
    1  at least one task failed (the others were still attempted)
    2  configuration/environment error before any task started
"""

import dataclasses
from collections.abc import Callable

from .. import logs

EXIT_OK = 0
EXIT_TASK_FAILED = 1
EXIT_CONFIG_ERROR = 2


@dataclasses.dataclass
class Task:
    name: str
    func: Callable  # called as func(logger=<task logger>)
    period: str
    run: str


@dataclasses.dataclass
class TaskResult:
    task: Task
    error: BaseException | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def run_tasks(tasks: list, log_root: str) -> tuple[list, int]:
    """Run tasks in order with per-task logging and isolation.

    Returns (results, exit_code). A failing task logs a parseable
    ``ERROR in task ... END ERROR`` block into its own log file and a FAILED
    line into orchestrator.log; the remaining tasks still run.
    """
    orchestrator = logs.orchestrator_logger(log_root)
    results = []

    for task in tasks:
        task_log = logs.task_logger(log_root, task.name, task.period, task.run)
        orchestrator.info(
            "START task=%s period=%s run=%s", task.name, task.period, task.run
        )
        try:
            task.func(logger=task_log)
        except BaseException as exc:  # noqa: B036 - isolation is the point here
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            task_log.error(
                logs.format_error_block(task.name, task.period, task.run, exc)
            )
            orchestrator.error(
                "FAILED task=%s period=%s run=%s error=%s",
                task.name,
                task.period,
                task.run,
                f"{type(exc).__name__}: {exc}",
            )
            results.append(TaskResult(task, exc))
        else:
            orchestrator.info(
                "END task=%s period=%s run=%s status=ok",
                task.name,
                task.period,
                task.run,
            )
            results.append(TaskResult(task))

    exit_code = EXIT_OK if all(r.ok for r in results) else EXIT_TASK_FAILED
    return results, exit_code
