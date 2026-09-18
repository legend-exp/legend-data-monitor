"""Detector-issue contract for auto-giorgio.

A detector issue is a physics-level anomaly (e.g. a parameter jumping out of
its threshold range) raised by the monitoring pipeline for an unattended
agent to triage: spurious/transient blip vs a persistent problem that
warrants a metadata change (e.g. usability -> 'ac').

Artifacts per (period, run, datatype):

    <output>/generated/mon/issues/<period>/<run>/l200-<period>-<run>-<datatype>-issues.jsonl

one JSON object per line, plus a parseable ``ISSUE ... END ISSUE`` block in
the task log so existing log scanners detect issues with no extra polling
path. The JSONL payload carries the triage fields (excursion stats,
persistence, provenance into the raw LH5 tree).

STABILITY: the ``ISSUES <absolute path> count=<n>`` orchestrator-log line is
the discovery contract for external monitors (see
docs/auto-giorgio-integration.md) — do not change its format.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# metric details
#
# The pass/fail booleans in the qcp summary carry no magnitudes, so a triage
# agent cannot tell a 0.5 % wobble from a real excursion. The evaluators know
# the numbers at the moment they decide, so they stash them here and the issue
# emitter (utils.check_cal_phy_thresholds) pops them when it builds records.
# Keyed by (period, run, datatype, detector, metric) so runs cannot mix.
# ---------------------------------------------------------------------------
_DETAILS: dict = {}


def record_detail(period, run, datatype, detector, metric, **fields) -> None:
    """Stash the magnitudes behind one metric evaluation (see module note)."""
    _DETAILS[(period, run, datatype, detector, metric)] = {
        k: v for k, v in fields.items() if v is not None
    }


def pop_detail(period, run, datatype, detector, metric) -> dict:
    """Return (and forget) the details recorded for one metric evaluation."""
    return _DETAILS.pop((period, run, datatype, detector, metric), {})


def clear_details() -> None:
    """Drop all stashed details (used by tests and between invocations)."""
    _DETAILS.clear()


def classify_severity(
    observed: float | None,
    threshold: list | None,
    excursion: "Excursion | None",
    *,
    reference: float | None = None,
    min_frac_out: float = 0.05,
    band_multiple: float = 1.0,
) -> str:
    """Grade an issue as ``warning`` or ``alert``.

    A failed threshold alone is not newsworthy: the cal metrics are two-sided
    consistency bands of a few times the fit error, so a detector whose
    resolution *improved* trips them exactly like one that degraded, and on
    real data ~30 % of the array trips one per run. Only clearly significant
    departures become ``alert``; the rest stay ``warning`` for the consumer's
    severity gate to filter.

    Time series (phy) are graded on the excursion: sustained (at least
    ``min_frac_out`` of the window) and still out of range at the end.
    Single-value checks (cal) have no excursion, so they are graded on how far
    past the band the value sits, in units of the band half-width. A one-sided
    band (e.g. resolution, where only degradation is a problem) has no width of
    its own: the ``reference`` the evaluator recorded — the band centre —
    supplies it, and without one the issue stays ``warning``.
    """
    if excursion is not None:
        if excursion.frac_out < min_frac_out:
            return "warning"
        return "warning" if excursion.recovered else "alert"

    if observed is None or not threshold:
        return "warning"
    low, high = (list(threshold) + [None, None])[:2]
    if low is None or high is None:
        bound = high if low is None else low
        if bound is None or reference is None:
            return "warning"
        if not np.isfinite([bound, reference]).all():
            return "warning"
        half_width = abs(bound - reference)
        outside = max(observed - high if low is None else low - observed, 0.0)
    else:
        if not np.isfinite([low, high]).all():
            return "warning"
        half_width = (high - low) / 2.0
        outside = max(low - observed, observed - high, 0.0)
    if half_width <= 0:
        return "warning"
    return "alert" if outside >= band_multiple * half_width else "warning"


@dataclasses.dataclass
class Excursion:
    """How a series violated its thresholds — the spurious-vs-real triage data."""

    frac_out: float  # fraction of samples out of range
    max_deviation: float  # worst value beyond the violated threshold
    longest_s: float  # longest contiguous out-of-range stretch, seconds
    recovered: bool  # back in range by the end of the window?


@dataclasses.dataclass
class Issue:
    detector: str
    metric: str
    severity: str  # "warning" | "alert"
    period: str
    run: str
    datatype: str
    observed: float | None = None
    threshold: list | None = None  # [low, high]; None entries = unbounded
    unit: str | None = None
    reference: float | None = None  # band centre, when the evaluator has one
    window: list | None = None  # [start_iso, end_iso]
    excursion: Excursion | None = None
    first_seen_run: str | None = None
    rawid: int | None = None
    string: int | None = None
    position: int | None = None
    # array-level records (see collapse_correlated): who was caught up in it
    affected_detectors: list | None = None
    affected_frac: float | None = None
    data_ref: dict | None = None  # {"file": ..., "key": ...}
    raw_ref: dict | None = None  # provenance into the raw LH5 production tree
    plots: list | None = None
    suggested_action: str | None = None

    #: version of the issue-record schema itself (bump on breaking changes)
    RECORD_SCHEMA = 1

    @property
    def issue_id(self) -> str:
        return f"{self.period}-{self.run}-{self.datatype}-{self.detector}-{self.metric}"

    def to_dict(self) -> dict:
        out = {"issue_id": self.issue_id, "schema": self.RECORD_SCHEMA}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if value is None:
                continue
            if isinstance(value, Excursion):
                value = dataclasses.asdict(value)
            if field.name == "plots":
                # consumers (auto-giorgio) attach these files directly:
                # always publish absolute paths
                value = [str(Path(p).absolute()) for p in value]
            out[field.name] = value
        return out


def evaluate_excursion(
    data_series: pd.Series, low: float | None, high: float | None
) -> Excursion | None:
    """Evaluate how a time-indexed series violates [low, high]; None if it doesn't.

    Parameters
    ----------
    data_series : pd.Series
        Values indexed by datetime (or with a datetime column as index).
    low, high : float or None
        Threshold bounds; None means unbounded on that side.
    """
    if data_series is None or len(data_series) == 0 or (low is None and high is None):
        return None

    values = data_series.to_numpy(dtype=float)
    out = np.zeros(len(values), dtype=bool)
    deviation = np.zeros(len(values), dtype=float)
    if high is not None:
        over = values > high
        out |= over
        deviation = np.where(over, values - high, deviation)
    if low is not None:
        under = values < low
        out |= under
        deviation = np.where(under, low - values, deviation)

    if not out.any():
        return None

    # longest contiguous out-of-range stretch, in seconds when the index is
    # datetime-like, else in samples
    idx = data_series.index
    longest = 0.0
    start = None
    for i, flag in enumerate(out):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            longest = max(longest, _span_seconds(idx, start, i - 1))
            start = None
    if start is not None:
        longest = max(longest, _span_seconds(idx, start, len(out) - 1))

    return Excursion(
        frac_out=float(out.mean()),
        max_deviation=float(np.nanmax(deviation)),
        longest_s=float(longest),
        recovered=bool(not out[-1]),
    )


def _span_seconds(idx, i0: int, i1: int) -> float:
    try:
        return max((idx[i1] - idx[i0]).total_seconds(), 0.0)
    except (AttributeError, TypeError):
        return float(i1 - i0)


def issues_file_path(output_folder: str, period: str, run: str, datatype: str) -> str:
    """Path of the issues JSONL for one (period, run, datatype)."""
    return str(
        Path(output_folder)
        / "generated/mon/issues"
        / period
        / run
        / f"l200-{period}-{run}-{datatype}-issues.jsonl"
    )


#: a metric failing on at least this fraction of the detectors it was evaluated
#: on (and on at least MIN_CORRELATED_DETECTORS of them) is one array-wide
#: event, not that many independent detector problems. Calibrated on p22:
#: per-detector cal metrics fail on 10-25 % of the array in a normal run, while
#: the SiPM noise bursts that motivated this hit 34-100 % of the channels.
CORRELATED_FRACTION = 0.3
MIN_CORRELATED_DETECTORS = 5


def _deviation(issue: "Issue") -> float:
    """How badly one record violated its band (comparable within a metric)."""
    if issue.excursion is not None:
        return issue.excursion.frac_out
    if issue.observed is None or not issue.threshold:
        return 0.0
    low, high = (list(issue.threshold) + [None, None])[:2]
    return max(
        low - issue.observed if low is not None else 0.0,
        issue.observed - high if high is not None else 0.0,
        0.0,
    )


def collapse_correlated(
    records: list,
    evaluated: dict | None = None,
    subsystems: dict | None = None,
    *,
    min_fraction: float = CORRELATED_FRACTION,
    min_detectors: int = MIN_CORRELATED_DETECTORS,
    max_plots: int = 6,
) -> list:
    """
    Replace an array-wide failure of one metric with a single array-level record.

    A common-mode event — a noise burst across the SiPMs, a DAQ hiccup — trips
    the same metric on most of the array at once, and emitting one record per
    channel buries the signal it carries. Those records collapse into one,
    keyed on a pseudo-detector (``spms-array``), representing the worst
    channel and listing the rest in ``affected_detectors``. Metrics that fail
    on only a few detectors are untouched: those really are independent.

    Parameters
    ----------
    records : list
        ``Issue`` records for one (period, run, datatype).
    evaluated : dict, optional
        metric -> number of detectors the metric was evaluated on; the
        fraction is taken against the number of failures when not given.
    subsystems : dict, optional
        metric -> subsystem label used to name the pseudo-detector
        (``<label>-array``); unlisted metrics collapse onto ``array``.
    min_fraction, min_detectors : float, int
        Collapse only above both of these (see CORRELATED_FRACTION).
    max_plots : int
        Cap on the figures carried over from the members.

    Returns
    -------
    list
        Records in their original order, with each collapsed group replaced by
        its array-level record at the position of its first member.
    """
    by_metric: dict = {}
    for record in records:
        by_metric.setdefault(record.metric, []).append(record)

    collapsed = {}
    for metric, group in by_metric.items():
        total = (evaluated or {}).get(metric) or len(group)
        if len(group) < min_detectors or len(group) < min_fraction * total:
            continue
        worst = max(group, key=lambda r: (r.severity == "alert", _deviation(r)))
        plots = list(dict.fromkeys(p for r in group for p in (r.plots or [])))
        label = (subsystems or {}).get(metric)
        collapsed[metric] = dataclasses.replace(
            worst,
            detector=f"{label}-array" if label else "array",
            rawid=None,
            string=None,
            position=None,
            affected_detectors=sorted(r.detector for r in group),
            affected_frac=round(len(group) / total, 3),
            first_seen_run=min(r.first_seen_run or r.run for r in group),
            plots=plots[:max_plots],
            suggested_action=(
                f"{len(group)} of {total} detectors failed {metric} in the same "
                "run: treat as common mode (DAQ, HV, run conditions) rather "
                "than reviewing channels individually; "
                f"worst was {worst.detector}"
            ),
        )

    out, done = [], set()
    for record in records:
        if record.metric not in collapsed:
            out.append(record)
        elif record.metric not in done:
            done.add(record.metric)
            out.append(collapsed[record.metric])
    return out


def write_issues(path: str, issues: list) -> str:
    """Write issues as JSONL (one object per line), replacing any previous file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for issue in issues:
            f.write(json.dumps(issue.to_dict(), default=str) + "\n")
    return path


def format_issue_block(issue: Issue, payload_path: str) -> str:
    """Render the parseable ISSUE block for the task log."""
    lines = [
        f"ISSUE detector={issue.detector} metric={issue.metric} "
        f"severity={issue.severity} "
        f"(period={issue.period}, run={issue.run}, datatype={issue.datatype}):"
    ]
    if issue.observed is not None:
        unit = f" {issue.unit}" if issue.unit else ""
        lines.append(f"  observed={issue.observed}{unit} outside {issue.threshold}")
    if issue.excursion is not None:
        e = issue.excursion
        lines.append(
            f"  frac_out={e.frac_out:.3g} longest={e.longest_s:.0f}s "
            f"recovered={str(e.recovered).lower()} "
            f"first_seen={issue.first_seen_run or issue.run}"
        )
    lines.append(f"  payload={payload_path}")
    for plot in issue.plots or []:
        lines.append(f"  plot={plot}")
    lines.append("END ISSUE")
    return "\n".join(lines)
