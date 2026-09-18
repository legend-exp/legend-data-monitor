"""Calibration-stability figures redrawn from the period cal contract.

Pure consumers of the frames written by ``calibration.write_escale_summary``
(key ``escale/<run>``) and ``calibration.write_psd_stability`` (keys
``psd_stability/<run>/<detector>``): no shelves, no pickles, no HDF writes.
PDF names and directories replicate the legacy ``savefig`` calls verbatim.
"""

import os

import numpy as np
import pandas as pd

from .. import logs, utils
from ..contract import reader as contract_reader
from ..monitoring import apply_monitoring_style, period_contract_path

E_583 = 583.191
E_SEP = 2103.511
E_FEP = 2614.511


def _warn(logger, msg):
    (logger or utils.logger).warning(msg)


def _load_detector_map(output_folder, period, run, data_type, logger):
    """Read /detector_map from the run contract, or None when unavailable."""
    path = os.path.join(
        output_folder, period, run, f"l200-{period}-{run}-{data_type}-geds-schema2.hdf"
    )
    try:
        return contract_reader.read_frame(path, "detector_map")
    except (KeyError, OSError):
        _warn(logger, f"no detector_map readable at {path}")
        return None


def _det_location(detector_map, det_name):
    """Return (string, position) of a detector, or None when not mapped."""
    rows = detector_map[detector_map["name"] == det_name]
    if rows.empty:
        return None
    row = rows.iloc[0]
    try:
        return int(row["string"]), int(row["position"])
    except (TypeError, ValueError):
        return None


def _series(frame, parameter, peak=None):
    """One escale parameter as a {period_run: value} dict.

    Parameters
    ----------
    frame : pandas.DataFrame
        Long escale frame (detector, parameter, peak, period_run, value).
    parameter : str
        Parameter name to select.
    peak : float, optional
        Peak energy for peak-resolved parameters (matched numerically).

    Returns
    -------
    values : dict
        Mapping period_run -> value for the selection.
    """
    sel = frame[frame["parameter"] == parameter]
    if peak is not None:
        peaks = pd.to_numeric(sel["peak"], errors="coerce").to_numpy(dtype=float)
        sel = sel[np.isclose(peaks, peak, atol=1e-6)]
    return dict(zip(sel["period_run"], sel["value"]))


def _aligned(all_keys, mapping):
    return np.array([mapping.get(k, np.nan) for k in all_keys], dtype=float)


def _period_prefixes(all_keys):
    return sorted({str(k).split("-")[0] for k in all_keys})


def _shade_status(ax, all_keys, usability):
    """Shade 'ac' (grey) and 'off' (red) runs, as the legacy overlay did."""
    for j, k in enumerate(all_keys):
        status = usability.get(k)
        if status == "ac":
            ax.axvspan(j - 0.5, j + 0.5, alpha=0.15, color="grey")
        elif status == "off":
            ax.axvspan(j - 0.5, j + 0.5, alpha=0.15, color="r")


def _plot_metric(
    ax,
    all_keys,
    values,
    errors=None,
    *,
    title="",
    units="keV",
    alpha=1.0,
    usability=None,
    shade=True,
    plot_mean=True,
    fixed_thr=None,
    err_thr=None,
    exclude_period=None,
):
    """
    One escale panel: per-period series, error band, mean and threshold lines.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    all_keys : list
        Ordered period_run keys defining the x axis.
    values : dict
        Mapping period_run -> value.
    errors : dict, optional
        Mapping period_run -> uncertainty (drawn as a band).
    title, units : str
        Panel title and y-axis units.
    alpha : float
        Line transparency.
    usability : dict, optional
        Mapping period_run -> 'on'/'ac'/'off'; enables status shading and
        restricts the mean to usable runs (legacy behavior). When None the
        mean uses every valid point.
    shade : bool
        Draw the usability shading (needs ``usability``).
    plot_mean : bool
        Draw the per-period mean line.
    fixed_thr : float, optional
        Fixed +- threshold around the mean (red dashed lines).
    err_thr : float, optional
        Multiplier on the mean error for +- thresholds (red dashed lines).
    exclude_period : list, optional
        Period prefixes to skip entirely.
    """
    import matplotlib.pyplot as plt

    x = np.arange(len(all_keys))
    keys_arr = np.asarray(all_keys, dtype=object)
    if shade and usability is not None:
        _shade_status(ax, all_keys, usability)
    vals = _aligned(all_keys, values)
    errs = _aligned(all_keys, errors) if errors is not None else None
    colors = plt.cm.tab10.colors

    for i, prefix in enumerate(_period_prefixes(all_keys)):
        if exclude_period and prefix in exclude_period:
            continue
        mask = np.array([str(k).startswith(prefix) for k in all_keys])
        x0 = x[mask]
        vals0 = vals[mask]
        valid = ~np.isnan(vals0)
        x0, vals0 = x0[valid], vals0[valid]
        if len(x0) == 0:
            continue
        color = colors[i % len(colors)]
        ax.plot(x0, vals0, ls="--", lw=1, marker="*", color=color, alpha=alpha)
        if errs is not None:
            errs0 = errs[mask][valid]
            ax.fill_between(x0, vals0 - errs0, vals0 + errs0, alpha=0.3, color=color)

        lim0, lim1 = x0[0] - 0.5, x0[-1] + 0.5
        if usability is not None:  # legacy: mean over runs where detector is on
            usab0 = np.array([usability.get(k) for k in keys_arr[mask]], dtype=object)
            good = usab0[valid] == "on"
        else:
            good = np.ones(len(vals0), dtype=bool)
        vals_good = vals0[good]
        if len(vals_good) == 0:
            continue
        mean_p = np.nanmean(vals_good)
        if plot_mean:
            ax.hlines(mean_p, lim0, lim1, color="k", ls=":", lw=1.2)
        if fixed_thr is not None:
            for thr in (mean_p + fixed_thr, mean_p - fixed_thr):
                ax.hlines(thr, lim0, lim1, color="r", ls="--", lw=1.2)
        if err_thr is not None and errs is not None:
            mean_err = np.nanmean(errs[mask][valid][good])
            for thr in (mean_p - err_thr * mean_err, mean_p + err_thr * mean_err):
                ax.hlines(thr, lim0, lim1, color="r", ls="--", lw=1.2)

    ax.set_title(title, fontsize=14)
    ax.set_ylabel(f"{title} ({units})")
    ax.set_xticks(x)
    ax.set_xticklabels(all_keys, rotation=90, fontsize=11)
    ax.grid(False)


def _plot_usability(ax, all_keys, usability):
    """Draw the legacy Usability panel: status steps plus the shared legend."""
    import matplotlib.pyplot as plt

    x = np.arange(len(all_keys))
    colors = plt.cm.tab10.colors
    if usability:
        mapping = {"off": 0, "ac": 1, "on": 2}
        for i, prefix in enumerate(_period_prefixes(all_keys)):
            mask = np.array([str(k).startswith(prefix) for k in all_keys])
            vals0 = np.array(
                [
                    mapping.get(usability.get(k), np.nan)
                    for k in np.array(all_keys)[mask]
                ]
            )
            x0 = x[mask][~np.isnan(vals0)]
            vals0 = vals0[~np.isnan(vals0)]
            ax.plot(x0, vals0, ls="-", marker="o", color=colors[i % len(colors)])
    ax.set_title("Usability", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(all_keys, rotation=90, fontsize=11)
    ax.grid(False)
    ax.plot([], [], color="r", ls="--", label="Thresholds")
    ax.plot([], [], color="k", ls=":", label="Mean")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["OFF", "AC", "ON"])
    ax.set_ylabel("Status")
    ax.legend(fontsize=11)


def _build_escale_figure(
    det_name, string, frame, all_keys, usability=None, exclude_period=None
):
    """
    Build the 4x3 energy-scale summary figure for one detector.

    Parameters
    ----------
    det_name : str
        Detector name (figure suptitle).
    string : int
        Detector string (figure suptitle).
    frame : pandas.DataFrame
        Rows of the escale contract frame for this detector.
    all_keys : list
        Ordered period_run keys defining the x axis of every panel.
    usability : dict, optional
        Mapping period_run -> 'on'/'ac'/'off' for this detector.
    exclude_period : list, optional
        Period prefixes to skip.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The assembled figure (caller saves and closes it).
    """
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(14, 14), facecolor="white")
    common = {"usability": usability, "exclude_period": exclude_period}

    _plot_usability(axs[0][0], all_keys, usability)
    _plot_metric(
        axs[0][1],
        all_keys,
        _series(frame, "fwhms_peaks", E_FEP),
        _series(frame, "fwhms_err_peaks", E_FEP),
        title="FWHM at FEP",
        units="keV",
        err_thr=3,
        **common,
    )
    _plot_metric(
        axs[0][2],
        all_keys,
        _series(frame, "fwhms_peaks", E_583),
        _series(frame, "fwhms_err_peaks", E_583),
        title="FWHM at 583 keV",
        units="keV",
        err_thr=3,
        **common,
    )
    _plot_metric(
        axs[1][0],
        all_keys,
        _series(frame, "mus_keV_first_cal_peaks", E_FEP),
        _series(frame, "mus_keV_first_cal_err_peaks", E_FEP),
        title="FEP position in keV using first cal",
        units="keV",
        fixed_thr=0.65375,
        **common,
    )
    _plot_metric(
        axs[1][1],
        all_keys,
        _series(frame, "residuals", E_SEP),
        title="SEP residuals",
        units="keV",
        fixed_thr=0.65375,
        **common,
    )
    # legacy overlays cusp sigma (unshaded) and etrap rise on one panel
    _plot_metric(
        axs[1][2],
        all_keys,
        _series(frame, "cusp_sigma"),
        title="",
        shade=False,
        usability=usability,
        exclude_period=exclude_period,
    )
    _plot_metric(
        axs[1][2],
        all_keys,
        _series(frame, "etrap_rise"),
        title="cusp sigma / etrap rise",
        units=r"$\mu$s",
        alpha=0.3,
        **common,
    )
    _plot_metric(
        axs[2][0],
        all_keys,
        _series(frame, "bl_std"),
        title="bl std",
        units="ADC",
        **common,
    )
    _plot_metric(
        axs[2][1],
        all_keys,
        _series(frame, "bl_max"),
        title="bl max",
        units="ADC",
        **common,
    )
    _plot_metric(
        axs[2][2],
        all_keys,
        _series(frame, "pz_tau"),
        title="PZ const",
        units=r"$\mu$s",
        **common,
    )
    _plot_metric(
        axs[3][0],
        all_keys,
        _series(frame, "ctc_alpha_par"),
        title="alpha ctc",
        units="ns^-1",
        **common,
    )
    _plot_metric(
        axs[3][1],
        all_keys,
        _series(frame, "aoe_mu"),
        _series(frame, "aoe_mu_err"),
        title="AoE mu",
        units="a. u.",
        **common,
    )
    _plot_metric(
        axs[3][2],
        all_keys,
        _series(frame, "aoe_sigma"),
        _series(frame, "aoe_sigma_err"),
        title="AoE sigma",
        units="a. u.",
        **common,
    )

    fig.suptitle(f"{det_name}, String {string}", fontsize=16)
    fig.tight_layout()
    return fig


def _save_figure(fig, pdf_dir, pdf_name, save_pdf, png_dir, logger, **savefig_kwargs):
    """Save one figure as PDF (legacy path) and/or PNG; return saved paths."""
    saved = []
    if save_pdf:
        os.makedirs(pdf_dir, exist_ok=True)
        pdf_path = os.path.abspath(os.path.join(pdf_dir, pdf_name))
        fig.savefig(pdf_path, **savefig_kwargs)
        if logger is not None:
            logs.log_saved_plot(logger, pdf_path)
        saved.append(pdf_path)
    if png_dir is not None:
        os.makedirs(png_dir, exist_ok=True)
        png_path = os.path.abspath(
            os.path.join(png_dir, pdf_name[: -len(".pdf")] + ".png")
        )
        fig.savefig(png_path, **savefig_kwargs)
        if logger is not None:
            logs.log_saved_plot(logger, png_path)
        saved.append(png_path)
    return saved


def plot_escale_panels(
    output_folder,
    period,
    run,
    *,
    detector_map=None,
    detector_status=None,
    exclude_period=None,
    data_type="cal",
    save_pdf=True,
    png_dir=None,
    logger=None,
):
    """
    Redraw the per-detector energy-scale summary figures from contract data.

    Port of ``plotting.plot_all_detector_info``: a 4x3 grid of energy-scale
    quantities across runs, one figure per detector found in the period
    contract key ``escale/<run>``.

    Parameters
    ----------
    output_folder : str
        Monitoring root (the folder containing ``<period>/``).
    period, run : str
        Period and run whose escale summary is drawn.
    detector_map : pandas.DataFrame, optional
        Columns name, rawid, string, position; read from the run contract
        key ``detector_map`` when None.
    detector_status : dict, optional
        Legacy-shaped {detector: {"usability": {period_run: status}}}; not
        reconstructible from the escale frame. When given it drives the
        Usability panel, the status shading and the usable-only means; when
        None those elements are omitted and means use every valid point.
    exclude_period : list, optional
        Period prefixes to exclude from every panel.
    data_type : str
        Data type of the period contract file (escale lives in ``cal``).
    save_pdf : bool
        Write the legacy PDF files (frozen names, see below).
    png_dir : str, optional
        Also save each figure there as PNG (PDF basename with .png).
    logger : logging.Logger, optional
        Where SAVED_PLOT lines are announced.

    Returns
    -------
    saved : list
        Absolute paths of every file written.
    """
    import matplotlib.pyplot as plt

    apply_monitoring_style()
    path = period_contract_path(output_folder, period, data_type)
    try:
        frame = contract_reader.read_frame(path, f"escale/{run}")
    except (KeyError, OSError):
        _warn(logger, f"no escale/{run} frame in {path}; skipping escale figures")
        return []
    if frame.empty:
        return []
    if detector_map is None:
        detector_map = _load_detector_map(output_folder, period, run, data_type, logger)
    if detector_map is None:
        return []

    frame_keys = sorted(frame["period_run"].unique())
    saved = []
    for det_name in sorted(frame["detector"].unique()):
        location = _det_location(detector_map, det_name)
        if location is None:
            _warn(logger, f"{det_name} not in detector_map; skipping its figure")
            continue
        string, position = location
        usability = (detector_status or {}).get(det_name, {}).get("usability")
        all_keys = sorted(usability.keys()) if usability else frame_keys
        fig = _build_escale_figure(
            det_name,
            string,
            frame[frame["detector"] == det_name],
            all_keys,
            usability=usability,
            exclude_period=exclude_period,
        )
        saved += _save_figure(
            fig,
            os.path.join(output_folder, period, "mtg/pdf", f"st{string}"),
            f"{period}_string{string}_pos{position}_{det_name}_ESCALEusability.pdf",
            save_pdf,
            png_dir,
            logger,
        )
        plt.close(fig)
    return saved


def _build_psd_figure(
    det_name, run_labels, mean_vals, mean_errs, sigma_vals, sigma_errs, eval_result
):
    """
    Build the 2x2 A/E PSD-stability figure for one detector.

    Parameters
    ----------
    det_name : str
        Detector name (figure suptitle).
    run_labels : list
        Run labels, one per point.
    mean_vals, mean_errs, sigma_vals, sigma_errs : array-like
        A/E fit results per run.
    eval_result : dict
        ``calibration.evaluate_psd_performance`` output (shift metrics).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The assembled figure (caller saves and closes it).
    """
    import matplotlib.pyplot as plt

    run_positions = list(range(len(run_labels)))
    fig, axs = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    (ax1, ax3), (ax2, ax4) = axs

    mean_avg, mean_std = np.nanmean(mean_vals), np.nanstd(mean_vals)
    ax1.errorbar(
        run_positions,
        mean_vals,
        yerr=mean_errs,
        fmt="s",
        color="blue",
        capsize=4,
        label=r"$\mu_i$",
    )
    ax1.axhline(
        mean_avg,
        linestyle="--",
        color="steelblue",
        label=rf"$\bar{{\mu}} = {mean_avg:.5f}$",
    )
    ax1.fill_between(
        run_positions,
        mean_avg - mean_std,
        mean_avg + mean_std,
        color="steelblue",
        alpha=0.2,
        label="±1 std dev",
    )
    ax1.set_ylabel("Mean stability")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)

    sigma_avg, sigma_std = np.nanmean(sigma_vals), np.nanstd(sigma_vals)
    ax2.errorbar(
        run_positions,
        sigma_vals,
        yerr=sigma_errs,
        fmt="s",
        color="darkorange",
        capsize=4,
        label=r"$\sigma_i$",
    )
    ax2.axhline(
        sigma_avg,
        linestyle="--",
        color="peru",
        label=rf"$\bar{{\sigma}} = {sigma_avg:.5f}$",
    )
    ax2.fill_between(
        run_positions,
        sigma_avg - sigma_std,
        sigma_avg + sigma_std,
        color="peru",
        alpha=0.2,
        label="±1 std dev",
    )
    ax2.set_ylabel("Sigma stability")
    ax2.set_xlabel("Run")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)

    ax3.plot(
        run_positions,
        eval_result["slow_shifts"],
        marker="^",
        markersize=10,
        linestyle="-",
        color="darkorchid",
        label="Slow shifts",
    )
    ax3.axhline(0, color="black", linestyle="--")
    ax3.axhline(0.5, color="crimson", linestyle="--")
    ax3.axhline(-0.5, color="crimson", linestyle="--")
    ax3.set_ylabel(r"$(\mu_i - \mu_0)/\bar{\sigma}$")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper left", bbox_to_anchor=(0, 0.95), fontsize=12)

    ax4.plot(
        run_positions,
        np.array(eval_result["sudden_shifts"]),
        marker="^",
        markersize=10,
        linestyle="-",
        color="green",
        label="Sudden shifts",
    )
    ax4.axhline(0, color="black", linestyle="--")
    ax4.axhline(0.25, color="crimson", linestyle="--")
    ax4.set_ylabel(r"$|(\mu_{i}-\mu_{i-1})/\sigma_i|$")
    ax4.set_xlabel("Run")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="upper left", bbox_to_anchor=(0, 0.95), fontsize=12)

    for ax in axs.flatten():
        ax.set_xticks(run_positions)
        ax.set_xticklabels(run_labels, rotation=0)

    fig.suptitle(det_name, fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def plot_psd_stability(
    output_folder,
    period,
    run,
    *,
    detector_map=None,
    data_type="cal",
    save_pdf=True,
    png_dir=None,
    logger=None,
):
    """
    Redraw the per-detector A/E PSD-stability figures from contract data.

    Port of ``calibration.evaluate_psd_usability_and_plot``: the 2x2 figure
    of mean/sigma stability plus slow- and sudden-shift metrics, one figure
    per contract key ``psd_stability/<run>/<detector>``. Shift metrics are
    recomputed from the stored means/sigmas with
    ``calibration.evaluate_psd_performance``.

    Parameters
    ----------
    output_folder : str
        Monitoring root (the folder containing ``<period>/``).
    period, run : str
        Period and run whose PSD stability frames are drawn.
    detector_map : pandas.DataFrame, optional
        Columns name, rawid, string, position; read from the run contract
        key ``detector_map`` when None.
    data_type : str
        Data type of the period contract file (psd_stability lives in ``cal``).
    save_pdf : bool
        Write the legacy PDF files (frozen names, see below).
    png_dir : str, optional
        Also save each figure there as PNG (PDF basename with .png).
    logger : logging.Logger, optional
        Where SAVED_PLOT lines are announced.

    Returns
    -------
    saved : list
        Absolute paths of every file written.
    """
    import matplotlib.pyplot as plt

    from ..calibration import evaluate_psd_performance

    apply_monitoring_style()
    path = period_contract_path(output_folder, period, data_type)
    if not os.path.isfile(path):
        _warn(logger, f"no period contract file at {path}; skipping PSD figures")
        return []
    prefix = f"/psd_stability/{run}/"
    with pd.HDFStore(path, mode="r") as store:
        detectors = sorted(
            k[len(prefix) :] for k in store.keys() if k.startswith(prefix)
        )
    if not detectors:
        _warn(logger, f"no psd_stability/{run} keys in {path}; skipping PSD figures")
        return []
    if detector_map is None:
        detector_map = _load_detector_map(output_folder, period, run, data_type, logger)
    if detector_map is None:
        return []

    saved = []
    for det_name in detectors:
        frame = contract_reader.read_frame(
            path, f"psd_stability/{run}/{det_name}"
        ).sort_values("run")
        run_labels = list(frame["run"])
        mean_vals = frame["mean"].to_numpy(dtype=float)
        sigma_vals = frame["sigma"].to_numpy(dtype=float)
        eval_result = evaluate_psd_performance(
            list(mean_vals), list(sigma_vals), run_labels, run, det_name
        )
        if eval_result.get("status") is None:
            _warn(logger, f"{det_name}: PSD shift metrics not computable; skipping")
            continue
        location = _det_location(detector_map, det_name)
        if location is None:
            _warn(logger, f"{det_name} not in detector_map; skipping its figure")
            continue
        fig = _build_psd_figure(
            det_name,
            run_labels,
            mean_vals,
            frame["mean_err"].to_numpy(dtype=float),
            sigma_vals,
            frame["sigma_err"].to_numpy(dtype=float),
            eval_result,
        )
        saved += _save_figure(
            fig,
            os.path.join(output_folder, period, "mtg", "pdf", f"st{location[0]}"),
            f"{period}_string{location[0]}_pos{location[1]}_{det_name}_AoE_stab.pdf",
            save_pdf,
            png_dir,
            logger,
            bbox_inches="tight",
        )
        plt.close(fig)
    return saved
