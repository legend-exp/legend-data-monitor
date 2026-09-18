"""QC figures drawn from contract data (ports of the monitoring.qc_* figures).

Pure consumers of the period contract file (``qc_rate_series``, ``qc_average``,
``dead_time``, ``qc_classifier_frac`` keys) and the run contract's per-detector
classifier ``_dist2d`` histograms. PDF names and directories replicate the
legacy ``qc_time_series`` / ``qc_average`` / ``qc_distributions`` savefig calls
verbatim; they are a frozen interface consumed by a cloud-upload script.
"""

import itertools
import math
import os

import h5py
import numpy as np
import pandas as pd

from .. import logs, utils
from ..contract import reader as contract_reader
from ..monitoring import apply_monitoring_style, period_contract_path, read_dead_time

# legacy pars_to_inspect order (drives the shared tab20 color cycle)
DEFAULT_QC_FLAGS = [
    "IsHighlyPositivePolarityCandidate",
    "IsValidBlSlope",
    "IsValidBlSlopeRms",
    "IsValidBlPolyRms",
    "IsValidTailRms",
    "IsNotNoiseBurst",
    "IsValidCuspemin",
    "IsValidCuspemax",
    "IsValidTrapTpmax",
    "IsLowCuspemax",
    "IsDischarge",
    "IsSaturated",
]

# legacy listed IsValidBlSlopeRmsClassifier twice; the repeat only re-saved
# the same PDFs, so it is deduplicated here
CLASSIFIER_PARS = [
    "IsValidBlSlopeClassifier",
    "IsValidTailRmsClassifier",
    "IsValidPzSlopeClassifier",
    "IsValidBlSlopeRmsClassifier",
    "IsValidBlPolyRmsClassifier",
    "IsValidCuspeminClassifier",
    "IsValidCuspemaxClassifier",
]

# event-type flag -> legacy legend wording, in legacy drawing order
CLASSIFIER_FLAG_LABELS = {
    "All": "All events",
    "IsPulser": "TP",
    "IsBsln": "FT",
    "IsPhysics": "~TP, ~FT, E>25 keV",
}

_THRESHOLDED_FLAGS = ("IsDischarge", "IsSaturated")


def _run_contract_path(output_folder, period, run, data_type):
    return os.path.join(
        output_folder,
        period,
        run,
        f"l200-{period}-{run}-{data_type}-geds-schema2.hdf",
    )


def _load_detector_map(output_folder, period, run, data_type, detector_map, logger):
    """Detector map as given, or read from the run contract's detector_map key."""
    if detector_map is not None:
        return detector_map
    path = _run_contract_path(output_folder, period, run, data_type)
    if not os.path.isfile(path):
        logger.warning("no run contract file at %s; cannot map detectors", path)
        return None
    try:
        return contract_reader.read_frame(path, "detector_map")
    except (KeyError, OSError):
        logger.warning("no detector_map key in %s; cannot map detectors", path)
        return None


def _detector_groups(detector_map):
    """Ordered ``{string: sub-frame}`` of processable detectors, by position."""
    frame = detector_map
    if "processable" in frame.columns:
        frame = frame[frame["processable"].fillna(False).astype(bool)]
    frame = frame.sort_values(["string", "position"])
    return {int(s): grp for s, grp in frame.groupby("string", sort=True)}


def _draw_qc_threshold(ax, flag):
    """Gray band + dashed line at the MTG upper rate limit (legacy styling)."""
    if flag not in _THRESHOLDED_FLAGS:
        return
    limit = utils.MTG_PLOT_INFO[flag]["limits"][1]
    upper = ax.get_ylim()[1] if ax.get_ylim()[1] > 5 else limit * 1.1
    ax.axhspan(limit, upper, color="gray", alpha=0.25)
    ax.axhline(limit, ls="--", color="black", label=f"{limit} mHz upper threshold")


def _save_figure(fig, pdf_dir, stem, save_pdf, png_dir, logger, **savefig_kwargs):
    """Save a figure as PDF (frozen legacy name) and optional PNG; return paths."""
    paths = []
    if save_pdf:
        os.makedirs(pdf_dir, exist_ok=True)
        path = os.path.abspath(os.path.join(pdf_dir, f"{stem}.pdf"))
        logs.save_figure(fig, path, logger, **savefig_kwargs)
        paths.append(path)
    if png_dir is not None:
        os.makedirs(png_dir, exist_ok=True)
        path = os.path.abspath(os.path.join(png_dir, f"{stem}.png"))
        logs.save_figure(fig, path, logger, **savefig_kwargs)
        paths.append(path)
    return paths


# -------------------------------------------------------------------------
# 1. QC rate vs time, one figure per (flag, string)
# -------------------------------------------------------------------------
def _rate_series_figure(period, run, flag, string, dets, rates, avg_rates, color_cycle):
    """Build one per-string QC rate-vs-time figure (legacy qc_time_series).

    Parameters
    ----------
    period, run : str
        Run being drawn.
    flag : str
        QC flag name.
    string : int
        String number (title only).
    dets : list
        (detector name, position) pairs in string order.
    rates : pandas.DataFrame
        1h-cadence rate frame (mHz), detector-name columns.
    avg_rates : dict
        Detector name -> integrated rate in mHz (legend annotation).
    color_cycle : iterator
        Shared tab20 color cycle (advances across flags/strings as in legacy).

    Returns
    -------
    fig: matplotlib.figure.Figure
        The built figure (caller saves and closes it).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    for name, pos in dets:
        if name not in rates.columns:
            continue
        true_rate_mhz = round(float(avg_rates.get(name, float("nan"))), 2)
        rates[name].plot(
            ax=ax,
            drawstyle="steps-mid",
            label=f"{name} - pos {pos} - {true_rate_mhz} mHz",
            color=next(color_cycle),
        )
    ax.grid(False)
    ax.set_ylabel(f"{period} {run} - 1h {flag} rate (mHz)")
    fig.suptitle(f"{period} {run} - String: {string}")
    _draw_qc_threshold(ax, flag)
    ax.legend()  # legacy adds title "Last cycle: <cycle>"; not in the contract
    fig.tight_layout()
    return fig


def plot_qc_rate_series(
    output_folder,
    period,
    run,
    *,
    detector_map=None,
    data_type="phy",
    save_pdf=True,
    png_dir=None,
    logger=None,
):
    """Per-string QC flag rate vs time figures (port of qc_time_series drawing).

    Parameters
    ----------
    output_folder : str
        Monitoring root (the folder containing ``<period>/``).
    period, run : str
        Run to draw.
    detector_map : pandas.DataFrame, optional
        name/rawid/string/position frame; read from the run contract when None.
    data_type : str
        Data type of the contract files; default ``phy``.
    save_pdf : bool
        Write the legacy-named PDFs under ``<period>/<run>/mtg/pdf/st<string>/``.
    png_dir : str, optional
        Also save each figure there as PNG (PDF basename with ``.png``).
    logger : logging.Logger, optional
        Where warnings and SAVED_PLOT lines go; default package logger.

    Returns
    -------
    saved: list
        Absolute paths of the files written.
    """
    import matplotlib.pyplot as plt

    apply_monitoring_style()
    logger = logger or utils.logger
    path = period_contract_path(output_folder, period, data_type)
    if not os.path.isfile(path):
        logger.warning("no period contract file at %s; nothing to draw", path)
        return []
    detector_map = _load_detector_map(
        output_folder, period, run, data_type, detector_map, logger
    )
    if detector_map is None:
        return []
    groups = _detector_groups(detector_map)

    with pd.HDFStore(path, "r") as store:
        found = {
            key.split("/")[2]
            for key in store.keys()
            if key.startswith("/qc_rate_series/") and key.endswith(f"/{run}")
        }
    if not found:
        logger.warning("no qc_rate_series/*/%s keys in %s; nothing to draw", run, path)
        return []
    # legacy pars_to_inspect order first: the shared color cycle depends on it
    flags = [f for f in DEFAULT_QC_FLAGS if f in found]
    flags += sorted(found - set(DEFAULT_QC_FLAGS))

    avg = {}
    try:
        frame = contract_reader.read_frame(path, f"qc_average/{run}")
        avg = {(r.flag, r.detector): float(r.rate_mhz) for r in frame.itertuples()}
    except (KeyError, OSError):
        logger.warning("no qc_average/%s in %s; legend rates will be nan", run, path)

    color_cycle = itertools.cycle(plt.cm.tab20.colors)
    saved = []
    for flag in flags:
        rates = contract_reader.read_frame(path, f"qc_rate_series/{flag}/{run}")
        for string, grp in groups.items():
            dets = list(zip(grp["name"], grp["position"]))
            avg_rates = {name: avg.get((flag, name), float("nan")) for name, _ in dets}
            fig = _rate_series_figure(
                period, run, flag, string, dets, rates, avg_rates, color_cycle
            )
            if flag in _THRESHOLDED_FLAGS:
                # "_rate" already in the title
                stem = f"{period}_{run}_string{string}_{utils.MTG_PLOT_INFO[flag]['title']}"
            else:
                stem = f"{period}_{run}_string{string}_{flag}_rate"
            pdf_dir = os.path.join(
                output_folder, period, run, "mtg", "pdf", f"st{string}"
            )
            saved += _save_figure(fig, pdf_dir, stem, save_pdf, png_dir, logger)
            plt.close(fig)
    return saved


# -------------------------------------------------------------------------
# 2. QC average rate across the array, one figure per flag
# -------------------------------------------------------------------------
def _average_figure(period, run, flag, rate_by_rawid, string_groups, dead_time):
    """Build the array-wide average QC rate figure (legacy qc_average).

    Parameters
    ----------
    period, run : str
        Run being drawn.
    flag : str
        QC flag name.
    rate_by_rawid : dict
        rawid -> integrated rate in mHz.
    string_groups : dict
        string -> list of (detector name, rawid), in display order.
    dead_time : dict or None
        ``read_dead_time`` result; annotates the IsDischarge title.

    Returns
    -------
    fig: matplotlib.figure.Figure
        The built figure (caller saves and closes it).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4), sharex=True)
    title = f"period: {period} - run: {run} - passing {flag}"
    if flag == "IsDischarge":
        if dead_time is None:
            title += " - tot dead time unavailable"
        else:
            title += f" - tot dead time {dead_time['dead_time_pct']:.3f}%"
    ax.set_title(title)

    x_labels, xs, ys = [], [], []
    string_indices = {}
    ct = -1
    for string, dets in string_groups.items():
        indices = []
        for name, rawid in dets:
            ct += 1
            x_labels.append(name)
            indices.append(ct)
            if rawid not in rate_by_rawid:
                continue
            ys.append(rate_by_rawid[rawid])
            xs.append(ct)
        string_indices[string] = indices

    ax.scatter(xs, ys, color="dodgerblue", marker="o")
    ax.set_ylabel(f"Average rate {flag}=True (mHz)")
    ax.set_yscale("log")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=90)
    ax.grid(False)

    ymin, ymax = ax.get_ylim()
    label_y = ymin * (ymax / ymin) ** 0.05 if ymin > 0 else 0.1
    for string, indices in string_indices.items():
        if not indices:
            continue
        left, right = min(indices), max(indices)
        if string == 1:
            ax.axvline(left - 0.5, ls="--", color="k", alpha=0.5)
        ax.axvline(right + 0.5, ls="--", color="k", alpha=0.5)
        ax.text(
            left,
            label_y,
            f"String {string}",
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    _draw_qc_threshold(ax, flag)
    ax.legend()  # legacy adds title "Last cycle: <cycle>"; not in the contract
    fig.tight_layout()
    return fig


def plot_qc_average(
    output_folder,
    period,
    run,
    *,
    detector_map=None,
    data_type="phy",
    save_pdf=True,
    png_dir=None,
    logger=None,
):
    """Array-wide average QC rate figures (port of qc_average drawing).

    Parameters
    ----------
    output_folder : str
        Monitoring root (the folder containing ``<period>/``).
    period, run : str
        Run to draw.
    detector_map : pandas.DataFrame, optional
        name/rawid/string/position frame; read from the run contract when None.
    data_type : str
        Data type of the contract files; default ``phy``.
    save_pdf : bool
        Write the legacy-named PDFs under ``<period>/<run>/mtg/pdf/``.
    png_dir : str, optional
        Also save each figure there as PNG (PDF basename with ``.png``).
    logger : logging.Logger, optional
        Where warnings and SAVED_PLOT lines go; default package logger.

    Returns
    -------
    saved: list
        Absolute paths of the files written.
    """
    import matplotlib.pyplot as plt

    apply_monitoring_style()
    logger = logger or utils.logger
    path = period_contract_path(output_folder, period, data_type)
    if not os.path.isfile(path):
        logger.warning("no period contract file at %s; nothing to draw", path)
        return []
    try:
        frame = contract_reader.read_frame(path, f"qc_average/{run}")
    except (KeyError, OSError):
        logger.warning("no qc_average/%s in %s; nothing to draw", run, path)
        return []
    detector_map = _load_detector_map(
        output_folder, period, run, data_type, detector_map, logger
    )
    if detector_map is None:
        return []
    string_groups = {
        string: list(zip(grp["name"], grp["rawid"].astype(int)))
        for string, grp in _detector_groups(detector_map).items()
    }

    dead_time = read_dead_time(output_folder, period, run, data_type)
    if dead_time is None:
        logger.warning(
            "no dead time recorded for %s-%s; plotting IsDischarge without it",
            period,
            run,
        )

    saved = []
    # groupby(sort=False) keeps write order = the legacy pars_to_inspect order
    for flag, sub in frame.groupby("flag", sort=False):
        rate_by_rawid = dict(zip(sub["rawid"].astype(int), sub["rate_mhz"]))
        fig = _average_figure(
            period, run, flag, rate_by_rawid, string_groups, dead_time
        )
        if flag in _THRESHOLDED_FLAGS:
            stem = f"{period}_{run}_{utils.MTG_PLOT_INFO[flag]['title']}_avg"
        else:
            stem = f"{period}_{run}_{flag}_avg"
        pdf_dir = os.path.join(output_folder, period, run, "mtg", "pdf")
        saved += _save_figure(fig, pdf_dir, stem, save_pdf, png_dir, logger)
        plt.close(fig)
    return saved


# -------------------------------------------------------------------------
# 3. QC classifier distributions, one per-string grid per classifier
# -------------------------------------------------------------------------
def _read_dist2d_group(group):
    """(edges, {detector: counts}) from a uhi dist2d group, flow bins stripped."""
    values = group["storage/values"][...]
    names = [
        c.decode() if isinstance(c, bytes) else str(c)
        for c in group["ref_axes/axis_1/categories"][...]
    ]
    attrs = group["ref_axes/axis_0"].attrs
    edges = np.linspace(
        float(attrs["lower"]), float(attrs["upper"]), int(attrs["bins"]) + 1
    )
    # rows: [underflow, bins..., overflow]; trailing column is the flow category
    counts = {name: values[1:-1, i] for i, name in enumerate(names)}
    return edges, counts


def _classifier_figure(
    period, run, par, string, dets, edges, counts_by_flag, frac_lookup
):
    """Build one per-string classifier distribution grid (legacy qc_distributions).

    Parameters
    ----------
    period, run : str
        Run being drawn.
    par : str
        Classifier parameter name.
    string : int
        String number.
    dets : list
        (detector name, position) pairs in string order.
    edges : numpy.ndarray
        Shared classifier bin edges.
    counts_by_flag : dict
        event-type flag -> {detector: bin counts}.
    frac_lookup : dict
        (detector, flag) -> percent in [-5, 5] shown in the legend.

    Returns
    -------
    fig: matplotlib.figure.Figure
        The built figure (caller saves and closes it).
    """
    import matplotlib.pyplot as plt

    n_dets = len(dets)
    ncols = math.ceil(math.sqrt(n_dets))
    nrows = math.ceil(n_dets / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, (name, pos) in enumerate(dets):
        ax = axes[i]
        if name not in counts_by_flag.get("All", {}):
            continue  # not processed: leave the empty axes, as legacy did
        for flag, label in CLASSIFIER_FLAG_LABELS.items():
            counts = counts_by_flag.get(flag, {}).get(name)
            if counts is None:
                counts = np.zeros(len(edges) - 1)
            perc = frac_lookup.get((name, flag), float("nan"))
            # step outline, prop-cycle colored: matches legacy histtype="step"
            ax.stairs(counts, edges, label=f"{label} - {perc:.1f}%")
        ax.axvline(-5, color="k", linestyle="--")
        ax.axvline(5, color="k", linestyle="--")
        ax.axvspan(-15, -5, color="darkgray", alpha=0.2)
        ax.axvspan(5, 15, color="darkgray", alpha=0.2)
        ax.set_ylabel("Counts")
        ax.set_xlabel("Classifiers")
        ax.legend(title=f"{name} (pos {pos})", loc="upper right")
        ax.set_yscale("log")
        ax.grid(False)
        ax.set_xlim(-10, 10)

    for j in range(n_dets, len(axes)):
        axes[j].axis("off")

    # legacy appends " - last cycle: <cycle>"; not in the contract
    fig.suptitle(f"{period} {run} - string {string} - {par}")
    fig.tight_layout()
    return fig


def plot_classifier_distributions(
    output_folder,
    period,
    run,
    *,
    detector_map=None,
    data_type="phy",
    save_pdf=True,
    png_dir=None,
    logger=None,
):
    """Per-string QC classifier histogram grids (port of qc_distributions drawing).

    Parameters
    ----------
    output_folder : str
        Monitoring root (the folder containing ``<period>/``).
    period, run : str
        Run to draw.
    detector_map : pandas.DataFrame, optional
        name/rawid/string/position frame; read from the run contract when None.
    data_type : str
        Data type of the contract files; default ``phy``.
    save_pdf : bool
        Write the legacy-named PDFs under ``<period>/<run>/mtg/pdf/st<string>/``.
    png_dir : str, optional
        Also save each figure there as PNG (PDF basename with ``.png``).
    logger : logging.Logger, optional
        Where warnings and SAVED_PLOT lines go; default package logger.

    Returns
    -------
    saved: list
        Absolute paths of the files written.
    """
    import matplotlib.pyplot as plt

    apply_monitoring_style()
    logger = logger or utils.logger
    contract_file = _run_contract_path(output_folder, period, run, data_type)
    if not os.path.isfile(contract_file):
        logger.warning("no run contract file at %s; nothing to draw", contract_file)
        return []
    detector_map = _load_detector_map(
        output_folder, period, run, data_type, detector_map, logger
    )
    if detector_map is None:
        return []
    groups = _detector_groups(detector_map)

    frac_lookup = {}
    period_file = period_contract_path(output_folder, period, data_type)
    try:
        frac = contract_reader.read_frame(period_file, f"qc_classifier_frac/{run}")
        frac_lookup = {
            (r.classifier, r.detector, r.event_type): float(r.percent_in_range)
            for r in frac.itertuples()
        }
    except (KeyError, OSError):
        logger.warning(
            "no qc_classifier_frac/%s in %s; legend percentages will be nan",
            run,
            period_file,
        )

    saved = []
    with h5py.File(contract_file, "r") as f:
        for par in CLASSIFIER_PARS:
            edges = None
            counts_by_flag = {}
            for flag in CLASSIFIER_FLAG_LABELS:
                key = f"hist/{flag}_{par}_dist2d"
                if key not in f:
                    continue
                edges, counts_by_flag[flag] = _read_dist2d_group(f[key])
            if "All" not in counts_by_flag:
                logger.warning(
                    "no hist/All_%s_dist2d in %s; skipping", par, contract_file
                )
                continue
            par_fracs = {
                (det, flag): pct
                for (cls, det, flag), pct in frac_lookup.items()
                if cls == par
            }
            for string, grp in groups.items():
                dets = list(zip(grp["name"], grp["position"]))
                fig = _classifier_figure(
                    period, run, par, string, dets, edges, counts_by_flag, par_fracs
                )
                stem = f"{period}_{run}_string{string}_{par}"
                pdf_dir = os.path.join(
                    output_folder, period, run, "mtg", "pdf", f"st{string}"
                )
                saved += _save_figure(
                    fig,
                    pdf_dir,
                    stem,
                    save_pdf,
                    png_dir,
                    logger,
                    bbox_inches="tight",
                )
                plt.close(fig)
    return saved
