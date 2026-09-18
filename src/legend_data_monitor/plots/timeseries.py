"""Diagnostic PNG plots from binned contract data.

Pure consumer of BinnedTimeSeries: draws bin means with variance bands and
min/max whisker envelopes, saves PNGs under ``figs/``, and announces each
file with a ``SAVED_PLOT`` log line so unattended agents can attach them.
"""

import os

from .. import logs
from ..monitoring import apply_monitoring_style


def plot_binned_series(
    binned,
    out_dir: str,
    name: str,
    title: str | None = None,
    unit: str | None = None,
    detectors: list | None = None,
    logger=None,
) -> list:
    """One PNG per detector group: mean ± std band + min/max envelope.

    Parameters
    ----------
    binned : BinnedTimeSeries
        The binned data to draw.
    out_dir : str
        Directory for ``figs/``; created if needed.
    name : str
        File-name stem (e.g. ``IsPulser_Trapemax``).
    detectors : list, optional
        Subset of detector names to draw (default: all).
    logger : logging.Logger, optional
        Where SAVED_PLOT lines are announced.
    """
    import matplotlib.pyplot as plt

    apply_monitoring_style()

    mean = binned.to_frame("mean")
    std = binned.to_frame("variance") ** 0.5
    lo = binned.to_frame("min")
    hi = binned.to_frame("max")
    detectors = detectors or list(mean.columns)

    figs_dir = os.path.join(out_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    saved = []

    fig, ax = plt.subplots(figsize=(10, 4))
    for det in detectors:
        if det not in mean.columns or mean[det].dropna().empty:
            continue
        line = ax.plot(mean.index, mean[det], lw=0.8, label=det)[0]
        color = line.get_color()
        ax.fill_between(
            mean.index,
            mean[det] - std[det],
            mean[det] + std[det],
            alpha=0.2,
            color=color,
            linewidth=0,
        )
        ax.plot(lo.index, lo[det], color=color, lw=0.3, alpha=0.5)
        ax.plot(hi.index, hi[det], color=color, lw=0.3, alpha=0.5)
    ax.set_title(title or name)
    if unit:
        ax.set_ylabel(unit)
    if len(detectors) <= 12:
        ax.legend(fontsize=6, ncol=2)
    fig.autofmt_xdate()

    path = os.path.join(figs_dir, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if logger is not None:
        logs.log_saved_plot(logger, path)
    saved.append(path)
    return saved
