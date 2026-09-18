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
import os

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
    past the band the value sits, in units of the band half-width.
    """
    if excursion is not None:
        if excursion.frac_out < min_frac_out:
            return "warning"
        return "warning" if excursion.recovered else "alert"

    if observed is None or not threshold:
        return "warning"
    low, high = (list(threshold) + [None, None])[:2]
    if low is None or high is None or not np.isfinite([low, high]).all():
        return "warning"
    half_width = (high - low) / 2.0
    if half_width <= 0:
        return "warning"
    outside = max(low - observed, observed - high, 0.0)
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
    window: list | None = None  # [start_iso, end_iso]
    excursion: Excursion | None = None
    first_seen_run: str | None = None
    rawid: int | None = None
    string: int | None = None
    position: int | None = None
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
                value = [os.path.abspath(p) for p in value]
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
    return os.path.join(
        output_folder,
        "generated/mon/issues",
        period,
        run,
        f"l200-{period}-{run}-{datatype}-issues.jsonl",
    )


def write_issues(path: str, issues: list) -> str:
    """Write issues as JSONL (one object per line), replacing any previous file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
