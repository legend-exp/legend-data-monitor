"""Histogram binning for the file contract: event-level data in, binned stats out.

The base product is a (time × detector) boost-histogram with Mean storage —
count/mean/variance per bin from one fill pass — plus per-bin min/max
sidecars (numpy bincount-style reductions, elementwise-mergeable). File size
is independent of event count; the raw series is dropped after filling.
"""

import boost_histogram as bh
import numpy as np
import pandas as pd

from ..contract import schema


class BinnedTimeSeries:
    """A (time × detector) Mean-storage histogram with min/max sidecars."""

    def __init__(self, hist: bh.Histogram, mins: np.ndarray, maxs: np.ndarray):
        self.hist = hist
        self.mins = mins  # shape (n_time_bins, n_detectors), NaN where empty
        self.maxs = maxs

    @property
    def detectors(self) -> list:
        return list(self.hist.axes[1])

    def __add__(self, other: "BinnedTimeSeries") -> "BinnedTimeSeries":
        if list(self.hist.axes[1]) != list(other.hist.axes[1]) or (
            self.hist.axes[0] != other.hist.axes[0]
        ):
            raise ValueError("cannot merge binned series with different axes")
        return BinnedTimeSeries(
            self.hist + other.hist,
            np.fmin(self.mins, other.mins),
            np.fmax(self.maxs, other.maxs),
        )

    def rebin(self, factor: int) -> "BinnedTimeSeries":
        """Lossless rebin of the time axis by an integer factor."""
        hist = self.hist[:: bh.rebin(factor), :]
        n_new = self.mins.shape[0] // factor
        mins = np.fmin.reduceat(
            self.mins[: n_new * factor], np.arange(0, n_new * factor, factor), axis=0
        )
        maxs = np.fmax.reduceat(
            self.maxs[: n_new * factor], np.arange(0, n_new * factor, factor), axis=0
        )
        return BinnedTimeSeries(hist, mins, maxs)

    def to_frame(self, stat: str = "mean") -> pd.DataFrame:
        """Return one statistic as a tidy frame (UTC DatetimeIndex × detector columns).

        stat: 'mean' | 'count' | 'variance' | 'min' | 'max'
        """
        edges = self.hist.axes[0].edges
        idx = pd.to_datetime(edges[:-1], unit="s", utc=True)
        view = self.hist.view()
        if stat == "mean":
            values = np.where(view["count"] > 0, view["value"], np.nan)
        elif stat == "count":
            values = view["count"]
        elif stat == "variance":
            values = np.where(
                view["count"] > 1,
                view["_sum_of_deltas_squared"] / np.maximum(view["count"] - 1, 1),
                np.nan,
            )
        elif stat == "min":
            values = self.mins
        elif stat == "max":
            values = self.maxs
        else:
            raise ValueError(f"unknown stat {stat!r}")
        return pd.DataFrame(values, index=idx, columns=self.detectors)


def fill_time_series(
    timestamps: np.ndarray,
    detectors: np.ndarray,
    values: np.ndarray,
    detector_names: list,
    t_start: float,
    t_stop: float,
    cadence: str = schema.BASE_CADENCE,
) -> BinnedTimeSeries:
    """Single-pass fill of the base-cadence (time × detector) histogram.

    Parameters
    ----------
    timestamps : array of float
        Event unix timestamps (seconds, UTC).
    detectors : array of str
        Detector name per event (same length as timestamps).
    values : array of float
        Parameter value per event.
    detector_names : list of str
        Full detector axis (defines column order; superset of what appears).
    t_start, t_stop : float
        Window bounds (unix seconds); bin edges are aligned to the cadence.
    cadence : str
        One of schema.CADENCES; the base fill cadence.
    """
    step = schema.CADENCE_SECONDS[cadence]
    # align the window to the coarsest cadence so every rebin between the
    # contract cadences is exact (identical outer edges, integral factors)
    align = max(schema.CADENCE_SECONDS.values())
    t0 = np.floor(t_start / align) * align
    t1 = np.ceil(t_stop / align) * align
    if t1 <= t0:
        t1 = t0 + align
    n_bins = int(round((t1 - t0) / step))

    time_ax = bh.axis.Regular(n_bins, t0, t1)
    det_ax = bh.axis.StrCategory(list(detector_names))
    hist = bh.Histogram(time_ax, det_ax, storage=bh.storage.Mean())

    finite = np.isfinite(values)
    ts, ds, vs = timestamps[finite], detectors[finite], values[finite]
    if len(ts):
        hist.fill(ts, ds, sample=vs)

    # per-bin min/max sidecars
    mins = np.full((n_bins, len(detector_names)), np.nan)
    maxs = np.full((n_bins, len(detector_names)), np.nan)
    if len(ts):
        t_idx = np.clip(((ts - t0) // step).astype(int), 0, n_bins - 1)
        d_idx = det_ax.index(ds)
        flat = t_idx * len(detector_names) + d_idx
        order = np.argsort(flat, kind="stable")
        flat_sorted, v_sorted = flat[order], vs[order]
        starts = np.flatnonzero(np.r_[True, np.diff(flat_sorted) > 0])
        group_min = np.fmin.reduceat(v_sorted, starts)
        group_max = np.fmax.reduceat(v_sorted, starts)
        mins.ravel()[flat_sorted[starts]] = group_min
        maxs.ravel()[flat_sorted[starts]] = group_max

    return BinnedTimeSeries(hist, mins, maxs)


def fill_distribution(
    values: np.ndarray, n_bins: int = 100, value_range: tuple | None = None
) -> bh.Histogram:
    """1-D distribution histogram of all samples (replaces pickled figures).

    The default range covers the 0.5-99.5 percentiles: a single outlier
    (a 3000 ADC noise burst among 10-50 ADC values) used to stretch the axis
    so far that the bulk collapsed into one bin. The flow bins keep whatever
    falls outside, so nothing is lost, only re-placed.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if value_range is None:
        if len(values) == 0:
            value_range = (0.0, 1.0)
        else:
            lo, hi = (float(v) for v in np.percentile(values, [0.5, 99.5]))
            if lo == hi:
                lo, hi = lo - 0.5, hi + 0.5
            # upper bin edges are exclusive: nudge so the top sample is included
            value_range = (lo, float(np.nextafter(hi, np.inf)))
    hist = bh.Histogram(bh.axis.Regular(n_bins, *value_range))
    if len(values):
        hist.fill(values)
    return hist


def fill_distribution_2d(
    df: pd.DataFrame, n_bins: int = 100, value_range: tuple | None = None
) -> bh.Histogram:
    """
    Per-detector distribution histogram of a pivot frame.

    Same idea as :func:`fill_distribution` but with a detector category axis,
    so consumers can draw one histogram per detector from a single key. All
    detectors share one binning; out-of-range samples land in the flow bins.

    Parameters
    ----------
    df : pandas.DataFrame
        Pivot frame, one column per detector.
    n_bins : int
        Number of value bins.
    value_range : tuple, optional
        (lo, hi) value axis range; derived from the finite data when omitted.

    Returns
    -------
    hist: boost_histogram.Histogram
        Regular(value) x StrCategory(detector) count histogram.
    """
    names = [str(c) for c in df.columns]
    values = df.to_numpy(dtype=float).ravel()
    labels = np.tile(np.array(names, dtype=object), len(df))
    finite = np.isfinite(values)
    values, labels = values[finite], labels[finite]
    if value_range is None:
        if len(values) == 0:
            value_range = (0.0, 1.0)
        else:
            lo, hi = float(np.min(values)), float(np.max(values))
            if lo == hi:
                lo, hi = lo - 0.5, hi + 0.5
            value_range = (lo, float(np.nextafter(hi, np.inf)))
    hist = bh.Histogram(
        bh.axis.Regular(n_bins, *value_range),
        bh.axis.StrCategory(names, growth=False),
    )
    if len(values):
        hist.fill(values, labels)
    return hist


def empty_distribution_2d(
    n_bins: int = 100, value_range: tuple = (0.0, 1.0)
) -> bh.Histogram:
    """
    Empty Regular x StrCategory histogram to fill incrementally.

    Same layout as :func:`fill_distribution_2d`, for callers that see their
    values a detector at a time (and never hold them all at once) rather than
    as one pivot frame.

    Parameters
    ----------
    n_bins : int
        Number of value bins.
    value_range : tuple
        (lo, hi) value axis range.

    Returns
    -------
    hist: boost_histogram.Histogram
        Empty histogram; fill with ``hist.fill(values, name)``.
    """
    return bh.Histogram(
        bh.axis.Regular(n_bins, *value_range),
        bh.axis.StrCategory([], growth=True),
    )


def frame_to_binned(
    df: pd.DataFrame,
    cadence: str = schema.BASE_CADENCE,
    t_start: float | None = None,
    t_stop: float | None = None,
) -> BinnedTimeSeries:
    """Bin a tidy frame (DatetimeIndex x detector columns).

    The bridge from the existing pandas pipeline into the contract.
    """
    ts_all = df.index.tz_convert("UTC") if df.index.tz else df.index.tz_localize("UTC")
    unix = ts_all.asi8 / 1e9
    detector_names = list(df.columns)

    step = schema.CADENCE_SECONDS[cadence]
    align = max(schema.CADENCE_SECONDS.values())
    t0 = np.floor((t_start if t_start is not None else unix.min()) / align) * align
    t1 = np.ceil((t_stop if t_stop is not None else unix.max()) / align) * align
    if t1 <= t0:
        t1 = t0 + align
    n_bins = int(round((t1 - t0) / step))

    time_ax = bh.axis.Regular(n_bins, t0, t1)
    det_ax = bh.axis.StrCategory(list(detector_names))
    hist = bh.Histogram(time_ax, det_ax, storage=bh.storage.Mean())
    mins = np.full((n_bins, len(detector_names)), np.nan)
    maxs = np.full((n_bins, len(detector_names)), np.nan)

    # fill one detector at a time: the long-format form of this (repeat the
    # timestamps and tile an object-dtype detector array over every column)
    # materialises three n_events x n_detectors arrays at once, which is what
    # drove the build's memory peak on a full run
    for col, detector in enumerate(detector_names):
        values = df[detector].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if not finite.any():
            continue
        ts, vs = unix[finite], values[finite]
        hist.fill(ts, detector, sample=vs)

        t_idx = np.clip(((ts - t0) // step).astype(int), 0, n_bins - 1)
        order = np.argsort(t_idx, kind="stable")
        t_sorted, v_sorted = t_idx[order], vs[order]
        starts = np.flatnonzero(np.r_[True, np.diff(t_sorted) > 0])
        mins[t_sorted[starts], col] = np.fmin.reduceat(v_sorted, starts)
        maxs[t_sorted[starts], col] = np.fmax.reduceat(v_sorted, starts)

    return BinnedTimeSeries(hist, mins, maxs)
