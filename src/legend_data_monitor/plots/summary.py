"""Run-summary figures re-drawn from the period contract file.

Pure consumers of the numbers published by ``monitoring.write_ft_series``,
``monitoring.write_event_rate_qc`` and ``monitoring.write_detector_summary``:
the forced-trigger failure/survival figures, the QC-split event-rate figure
and the per-detector box summary, saved with the exact legacy PDF names.
"""

import itertools
import os
from functools import partial

import numpy as np
import pandas as pd

from .. import logs, utils
from ..contract import reader as contract_reader
from ..monitoring import (
    apply_monitoring_style,
    mhz_to_percent,
    percent_to_mhz,
    period_contract_path,
)

# legend order/colors of the event-rate figure (legacy)
_EVENT_RATE_SERIES = [
    ("all_events", "All events", "dimgrey"),
    ("delayed_discharges", "Delayed discharges", "darkorange"),
    ("failing_qc", "Failing QC", "crimson"),
    ("surviving_qc", "Surviving QC", "dodgerblue"),
]


def _read_key(path, key, logger):
    """One contract frame, or None (with a warning) when it is missing."""
    if not os.path.isfile(path):
        (logger or utils.logger).warning("missing contract file %s", path)
        return None
    try:
        return contract_reader.read_frame(path, key)
    except (KeyError, OSError):
        (logger or utils.logger).warning("missing contract key %s in %s", key, path)
        return None


def _read_detector_map(output_folder, period, run, data_type, logger):
    """Read the run contract's detector map (name, rawid, string, position)."""
    path = os.path.join(
        output_folder,
        period,
        run,
        f"l200-{period}-{run}-{data_type}-geds-schema2.hdf",
    )
    return _read_key(path, "detector_map", logger)


def _save_figure(fig, pdf_dir, stem, save_pdf, png_dir, logger, saved):
    """Save one figure as PDF (legacy name/dir) plus optional PNG copy."""
    if save_pdf:
        os.makedirs(pdf_dir, exist_ok=True)
        path = os.path.abspath(os.path.join(pdf_dir, f"{stem}.pdf"))
        fig.savefig(path, bbox_inches="tight")
        if logger is not None:
            logs.log_saved_plot(logger, path)
        saved.append(path)
    if png_dir is not None:
        os.makedirs(png_dir, exist_ok=True)
        path = os.path.abspath(os.path.join(png_dir, f"{stem}.png"))
        fig.savefig(path, bbox_inches="tight")
        if logger is not None:
            logs.log_saved_plot(logger, path)
        saved.append(path)


def _legend_title(last_cycle):
    """Legacy 'Last cycle' legend title, omitted when the cycle is unknown."""
    return f"Last cycle: {last_cycle}" if last_cycle is not None else None


def _ft_string_figure(
    period, run, string, rates, avg_total_forced_mhz, last_cycle, color_cycle
):
    """Per-string FT failure figure: one steps-mid line per detector."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    for det in rates.columns:
        rates[det].plot(
            ax=ax, drawstyle="steps-mid", label=det, color=next(color_cycle)
        )

    if avg_total_forced_mhz is not None:
        m2p = partial(mhz_to_percent, avg_total_forced_mhz=avg_total_forced_mhz)
        p2m = partial(percent_to_mhz, avg_total_forced_mhz=avg_total_forced_mhz)
        secax = ax.secondary_yaxis("right", functions=(m2p, p2m))
        secax.set_ylabel("FT failure fraction (%)")

    ax.set_ylabel("Normalized FT failure rate (mHz/kg)")
    ax.legend(
        ncol=2, fontsize="small", loc="upper left", title=_legend_title(last_cycle)
    )
    ax.grid(False)
    fig.suptitle(f"{period} - {run} - string {string}")
    fig.tight_layout()
    return fig


def _ft_all_strings_figure(period, run, per_string, last_cycle):
    """Build the combined FT failure figure: one steps-mid line per string."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    color_cycle = itertools.cycle(plt.cm.tab20.colors)
    for string in per_string.columns:
        per_string[string].plot(
            ax=ax,
            drawstyle="steps-mid",
            label=f"String {string}",
            color=next(color_cycle),
        )
    ax.set_ylabel("Normalized FT failure rate (mHz/kg)")
    ax.set_title(f"{period} - {run} - All strings")
    ax.legend(
        ncol=2, fontsize="small", loc="upper left", title=_legend_title(last_cycle)
    )
    ax.grid(False)
    fig.tight_layout()
    return fig


def _ft_sf_figure(period, surviving_frac):
    """FT survival-fraction figure over all strings combined."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    surviving_frac.plot(ax=ax, drawstyle="steps-mid", color="red")
    ax.set_ylabel("FT surviving events (%)")
    ax.set_title(f"{period} - All strings combined")
    ax.grid(False)
    fig.tight_layout()
    return fig


def _event_rate_figure(frame, last_cycle):
    """QC-split hourly event-rate stairs figure (rates already mHz/kg)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3.5))
    on_mass = float(frame["on_mass_kg"].iloc[0])
    for column, label, color in _EVENT_RATE_SERIES:
        if column not in frame.columns:
            continue
        rate = frame[column].dropna()  # columns are union-aligned in the frame
        if rate.empty:
            continue
        edges = rate.index.append(
            pd.DatetimeIndex([rate.index[-1] + pd.Timedelta(hours=1)])
        )
        ax.stairs(rate.to_numpy(), edges, label=label, color=color)

    ax.set_ylabel("Hourly rate normalized by ON mass (mHz/kg)")
    mass_line = f"ON mass = {on_mass:.1f} kg"
    title = (
        f"Last cycle: {last_cycle}\n{mass_line}"
        if last_cycle is not None
        else mass_line
    )
    ax.legend(title=title, loc="upper right")
    ax.grid(False)
    fig.tight_layout()
    return fig


def _metric_info(metric):
    """MTG_PLOT_INFO entry for a metric given as its key or its title."""
    if metric in utils.MTG_PLOT_INFO:
        return utils.MTG_PLOT_INFO[metric]
    for entry in utils.MTG_PLOT_INFO.values():
        if entry.get("title") == metric:
            return entry
    return None


def _detector_summary_figure(period, run, frame, info, last_cycle):
    """Per-detector box summary figure (legacy box_summary_plot drawing)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    df = frame.sort_values(["string", "pos"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))

    if info["title"] in ["FEP_gain_stab", "pulser_stab", "baseln_stab"]:
        ax.axhline(0, color="gray", lw=0.5)

    if not df["fwhm"].isna().all():
        fwhm_label = (
            r"$\pm$FWHM/2"
            if info["title"] == "FEP_gain_stab"
            else r"$\pm$FWHM (threshold)"
        )
        ax.bar(
            x,
            df["fwhm"],
            bottom=-df["fwhm"] / 2,
            width=0.4,
            color="orange",
            alpha=0.2,
            label=fwhm_label,
        )

    ax.bar(
        x,
        2 * df["std"],  # total height = twice 1 std
        bottom=df["mean"] - df["std"],  # center bar on mean
        width=0.6,
        color="skyblue",
        alpha=0.7,
        label="±1σ",
    )

    ax.scatter(x, df["mean"], color="black", zorder=3, label="Mean")

    ax.errorbar(
        x,
        df["mean"],
        yerr=[df["mean"] - df["min"], df["max"] - df["mean"]],
        fmt="none",
        ecolor="#0266c9" if info["title"] != "FEP_gain_stab" else "red",
        capsize=4,
        label="Min/Max",
    )

    ax.set_xticks(x)
    xtick_labels = ax.set_xticklabels(df["ged"], rotation=90)
    for i, label in enumerate(xtick_labels):
        if df.iloc[i]["usability"] in ["off", "false", False]:
            label.set_color("red")
        if df.iloc[i]["usability"] in ["ac"]:
            label.set_color("darkorange")

    ax.axvline(-0.5, color="gray", ls="--", alpha=0.5)
    ymin, ymax = ax.get_ylim()
    label_y = ymin * (ymax / ymin) ** 0.05 if ymin > 0 else -4
    label_y = label_y if info["title"] != "baseln_spike" else 1
    for s in df["string"].unique():
        idx = df.index[df["string"] == s]
        ax.axvline(idx.max() + 0.5, color="gray", ls="--", alpha=0.5)
        ax.text(idx.min(), label_y, f"String {s}", rotation=90)

    ax.set_ylabel(info.get("ylabel", ""))
    ax.set_title(f"{period} {run}")

    if info["title"] in ["baseln_stab"]:
        ax.axhline(
            -10,
            ls="--",
            color="black",
            label=r"$\pm$" + f"{info['limits'][1]}% threshold",
        )
        ax.axhline(10, ls="--", color="black")
        ax.axhspan(10, 500, color="gray", alpha=0.25)
        ax.axhspan(-10, -500, color="gray", alpha=0.25)
    if info["title"] in ["baseln_spike"]:
        ax.axhline(
            50, ls="--", color="black", label=f"{info['limits'][1]} ADC upper threshold"
        )
        ax.axhspan(50, 500, color="gray", alpha=0.25)

    if info["title"] == "FEP_gain_stab":
        ax.axhline(-2, ls="--", color="black", label=r"$\pm$2 keV threshold")
        ax.axhline(2, ls="--", color="black")
        ax.axhspan(2, 500, color="gray", alpha=0.25)
        ax.axhspan(-2, -500, color="gray", alpha=0.25)
        ax.set_ylim(-6, 6)
    if info["title"] == "pulser_stab":
        ax.set_ylim(-6, 6)
    if info["title"] in ["baseln_stab"]:
        ax.set_ylim(-20, 20)
    if info["title"] in ["baseln_spike"]:
        ax.set_ylim(0, 100)

    handles, _ = ax.get_legend_handles_labels()
    legend_patches = [
        Patch(color="red", label="Usability: off"),
        Patch(color="darkorange", label="Usability: ac"),
    ]
    ax.legend(
        handles=handles + legend_patches,
        loc="upper right",
        title=_legend_title(last_cycle),
    )
    ax.grid(False)
    fig.tight_layout()
    return fig


def plot_ft_summary(
    output_folder,
    period,
    run,
    *,
    detector_map=None,
    data_type="phy",
    save_pdf=True,
    png_dir=None,
    logger=None,
    last_cycle=None,
):
    """
    Draw the forced-trigger figures of a run from the period contract file.

    One FT-failure figure per string (detectors as steps-mid lines, mHz/kg,
    with a secondary percent axis), one combined all-strings figure, and the
    FT survival-fraction figure, from the ``ft_summary/*`` contract keys.

    Parameters
    ----------
    output_folder : str
        Monitoring root, i.e. the folder containing ``<period>/``.
    period : str
        Period to draw (e.g. ``p22``).
    run : str
        Run to draw (e.g. ``r000``).
    detector_map : pandas.DataFrame, optional
        Frame with columns name/rawid/string/position; read from the run
        contract's ``detector_map`` key when None.
    data_type : str
        Data type of the contract files (default ``phy``).
    save_pdf : bool
        Save the PDFs with the legacy names under ``<period>/<run>/mtg/pdf``.
    png_dir : str, optional
        Also save each figure there as PNG (PDF basename with ``.png``).
    logger : logging.Logger, optional
        Where SAVED_PLOT lines and warnings are announced.
    last_cycle : str, optional
        Last processed cycle; the 'Last cycle' legend title is omitted
        when None (the contract does not carry it).

    Returns
    -------
    saved : list[str]
        Absolute paths of every file saved (may be empty).
    """
    import matplotlib.pyplot as plt

    apply_monitoring_style()
    contract = period_contract_path(output_folder, period, data_type)
    per_detector = _read_key(contract, f"ft_summary/per_detector/{run}", logger)
    per_string = _read_key(contract, f"ft_summary/per_string/{run}", logger)
    total_forced = _read_key(contract, f"ft_summary/total_forced/{run}", logger)
    survival = _read_key(contract, f"ft_summary/survival_fraction/{run}", logger)

    saved = []
    pdf_root = os.path.join(output_folder, period, run, "mtg/pdf")

    if per_detector is not None:
        if detector_map is None:
            detector_map = _read_detector_map(
                output_folder, period, run, data_type, logger
            )
        if detector_map is None:
            (logger or utils.logger).warning(
                "no detector map: skipping per-string FT figures"
            )
        else:
            avg_mhz = None
            if total_forced is not None:
                avg_mhz = float(total_forced.iloc[:, 0].mean()) / 3600 * 1000
            # one tab20 cycle across ALL string figures, as in the legacy loop
            color_cycle = itertools.cycle(plt.cm.tab20.colors)
            ordered = detector_map.sort_values(["string", "position"])
            for string, group in ordered.groupby("string", sort=True):
                dets = [d for d in group["name"] if d in per_detector.columns]
                fig = _ft_string_figure(
                    period,
                    run,
                    string,
                    per_detector[dets],
                    avg_mhz,
                    last_cycle,
                    color_cycle,
                )
                _save_figure(
                    fig,
                    os.path.join(pdf_root, f"st{string}"),
                    f"{period}_{run}_string{string}_FT_failure",
                    save_pdf,
                    png_dir,
                    logger,
                    saved,
                )
                plt.close(fig)

    if per_string is not None:
        fig = _ft_all_strings_figure(period, run, per_string, last_cycle)
        _save_figure(
            fig,
            pdf_root,
            f"{period}_{run}_all_strings_FT_failure",
            save_pdf,
            png_dir,
            logger,
            saved,
        )
        plt.close(fig)

    if survival is not None:
        fig = _ft_sf_figure(period, survival.iloc[:, 0])
        _save_figure(
            fig,
            pdf_root,
            f"{period}_{run}_all_strings_FT_SF",
            save_pdf,
            png_dir,
            logger,
            saved,
        )
        plt.close(fig)

    return saved


def plot_event_rate_qc(
    output_folder,
    period,
    run,
    *,
    detector_map=None,
    data_type="phy",
    save_pdf=True,
    png_dir=None,
    logger=None,
    last_cycle=None,
):
    """
    Draw the QC-split event-rate figure from the ``event_rate_qc`` key.

    Stairs of All events / Delayed discharges / Failing QC / Surviving QC
    hourly rates (already mHz/kg in the contract; absent columns skipped).

    Parameters
    ----------
    output_folder : str
        Monitoring root, i.e. the folder containing ``<period>/``.
    period : str
        Period to draw.
    run : str
        Run to draw.
    detector_map : pandas.DataFrame, optional
        Unused; kept for the uniform renderer signature.
    data_type : str
        Data type of the period contract file (default ``phy``).
    save_pdf : bool
        Save ``{period}_{run}_event_rate_qc.pdf`` under
        ``<period>/<run>/mtg/pdf``.
    png_dir : str, optional
        Also save the figure there as PNG.
    logger : logging.Logger, optional
        Where SAVED_PLOT lines and warnings are announced.
    last_cycle : str, optional
        Last processed cycle; its legend line is omitted when None.

    Returns
    -------
    saved : list[str]
        Absolute paths of every file saved (may be empty).
    """
    import matplotlib.pyplot as plt

    apply_monitoring_style()
    contract = period_contract_path(output_folder, period, data_type)
    frame = _read_key(contract, f"event_rate_qc/{run}", logger)
    if frame is None or frame.empty:
        return []

    saved = []
    fig = _event_rate_figure(frame, last_cycle)
    _save_figure(
        fig,
        os.path.join(output_folder, period, run, "mtg/pdf"),
        f"{period}_{run}_event_rate_qc",
        save_pdf,
        png_dir,
        logger,
        saved,
    )
    plt.close(fig)
    return saved


def plot_detector_summary(
    output_folder,
    period,
    run,
    metric,
    *,
    detector_map=None,
    data_type="phy",
    save_pdf=True,
    png_dir=None,
    logger=None,
    last_cycle=None,
):
    """
    Draw the per-detector box summary of one metric from the contract file.

    Mean/std/min-max per detector ordered by string and position, with the
    FWHM band, usability tick colors and per-metric thresholds of the legacy
    ``box_summary_plot``.

    Parameters
    ----------
    output_folder : str
        Monitoring root, i.e. the folder containing ``<period>/``.
    period : str
        Period to draw.
    run : str
        Run to draw.
    metric : str
        MTG_PLOT_INFO key (e.g. ``FEP_variation``) or its title (e.g.
        ``FEP_gain_stab``); the contract key is always
        ``detector_summary/<title>/<run>`` since the title is what
        ``write_detector_summary`` stores under.
    detector_map : pandas.DataFrame, optional
        Unused; kept for the uniform renderer signature (string/position
        come from the summary frame itself).
    data_type : str
        Data type of the period contract file (``phy``/``cal``/``lac``/
        ``ssc``); the PDF name and directory do not depend on it.
    save_pdf : bool
        Save ``{period}_{run}_{title}.pdf`` under ``<period>/<run>/mtg/pdf``.
    png_dir : str, optional
        Also save the figure there as PNG.
    logger : logging.Logger, optional
        Where SAVED_PLOT lines and warnings are announced.
    last_cycle : str, optional
        Last processed cycle; the legend title is omitted when None.

    Returns
    -------
    saved : list[str]
        Absolute paths of every file saved (may be empty).
    """
    import matplotlib.pyplot as plt

    apply_monitoring_style()
    info = _metric_info(metric)
    if info is None:
        (logger or utils.logger).warning("unknown summary metric %r", metric)
        return []
    contract = period_contract_path(output_folder, period, data_type)
    frame = _read_key(contract, f"detector_summary/{info['title']}/{run}", logger)
    if frame is None or frame.empty:
        return []

    saved = []
    fig = _detector_summary_figure(period, run, frame, info, last_cycle)
    _save_figure(
        fig,
        os.path.join(output_folder, period, run, "mtg/pdf"),
        f"{period}_{run}_{info['title']}",
        save_pdf,
        png_dir,
        logger,
        saved,
    )
    plt.close(fig)
    return saved
