"""Per-detector stability figures redrawn from period contract data.

Pure consumer of the period monitoring contract file: the gain-shift and
parameter-stability traces (``gain_shift/...``, ``param_stability/...``,
``pul_cusp/kevdiff/...``, ``cal_points/...``) and the FEP gain-stability bins
(``fep_gain_stab/...``). The PDF names and directories replicate the legacy
``monitoring.plot_time_series`` and ``calibration.fep_gain_variation`` savefig
calls verbatim -- they are a frozen shifter interface.
"""

import os

import numpy as np
import pandas as pd

from .. import logs, utils
from ..contract import reader as contract_reader
from ..monitoring import apply_monitoring_style, period_contract_path

#: parameter figures ported from monitoring.plot_time_series, in legacy order
STABILITY_PARAMETERS = ["BlStd", "TrapemaxCtcCal", "Baseline", "Trapemax"]

QBB_LIN_LABEL = r"Q$_{\beta\beta}$ $\pm$FWHM/2 lin. (threshold)"
QBB_QUAD_LABEL = r"Q$_{\beta\beta}$ $\pm$FWHM/2 quad. (threshold)"
FEP_MIN_COUNTS = 5  # legacy fep_gain_variation constant, shown in the legend


def _read_frame(path, key, log, warn=True):
    """Read one contract key; None (with a log line) when it is missing."""
    try:
        return contract_reader.read_frame(path, key)
    except (KeyError, OSError) as exc:  # FileNotFoundError is an OSError
        (log.warning if warn else log.debug)(
            "missing contract key %s in %s (%s)", key, path, exc
        )
        return None


def _run_contract_path(output_folder, period, run, data_type):
    """Path of the schema2 run contract file (source of ``detector_map``)."""
    return os.path.join(
        output_folder, period, run, f"l200-{period}-{run}-{data_type}-geds-schema2.hdf"
    )


def _load_detector_map(output_folder, period, run, data_type, detector_map, log):
    """Given map, else the run contract's ``detector_map`` key, else None."""
    if detector_map is not None:
        return detector_map
    path = _run_contract_path(output_folder, period, run, data_type)
    return _read_frame(path, "detector_map", log, warn=False)


def _plain(value):
    """Integral values as plain ints so file names read st1/pos1, not st1.0."""
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return value
    return int(as_float) if as_float == int(as_float) else value


def _string_position(detector_map, cal, detector):
    """Return (string, position) from the detector map or cal_points."""
    for frame, name_col in ((detector_map, "name"), (cal, "detector")):
        if frame is None or name_col not in frame:
            continue
        rows = frame[frame[name_col] == detector]
        if len(rows):
            return _plain(rows.iloc[0]["string"]), _plain(rows.iloc[0]["position"])
    return None, None


def _detector_cal(cal, detector):
    """Return the detector's cal_points rows, run_start-ordered; None when absent."""
    if cal is None or not len(cal):
        return None
    rows = cal[cal["detector"] == detector].sort_values("run_start")
    return rows if len(rows) else None


def _naive_ts(value):
    """Timestamp without timezone, so xlim mixes cleanly with the traces."""
    ts = pd.Timestamp(value)
    return ts.tz_localize(None) if ts.tz is not None else ts


def _save_figure(fig, pdf_dir, pdf_name, save_pdf, png_dir, log, **savefig_kwargs):
    """Save the legacy-named PDF (and optional PNG twin); announce both."""
    saved = []
    if save_pdf:
        os.makedirs(pdf_dir, exist_ok=True)
        path = os.path.abspath(os.path.join(pdf_dir, pdf_name))
        fig.savefig(path, **savefig_kwargs)
        logs.log_saved_plot(log, path)
        saved.append(path)
    if png_dir is not None:
        os.makedirs(png_dir, exist_ok=True)
        path = os.path.abspath(os.path.join(png_dir, pdf_name[:-4] + ".png"))
        fig.savefig(path, **savefig_kwargs)
        logs.log_saved_plot(log, path)
        saved.append(path)
    return saved


def _draw_cal_points(ax, cal, quadratic):
    """FEP/cal-const markers, run boundaries and stepped FWHM/2 band segments.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes of the gain-shift figure.
    cal : pandas.DataFrame
        This detector's cal_points rows, ordered by run_start.
    quadratic : bool
        Also draw the quadratic-fit resolution band (legacy flag).

    Returns
    -------
    t0 : list
        Naive run-start timestamps, used for the x limits.
    """
    t0 = [_naive_ts(t) for t in cal["run_start"]]
    # frames written before the res columns existed still draw, minus the bands
    nan = np.full(len(cal), np.nan)
    res = cal["res"].astype(float).values if "res" in cal else nan
    res_quad = cal["res_quad"].astype(float).values if "res_quad" in cal else nan
    shifted = [t - pd.Timedelta(hours=5) for t in t0]  # legacy -5 h marker shift
    ax.plot(shifted, cal["fep_diff"].astype(float).values, "kx", label="FEP gain")
    ax.plot(
        shifted,
        cal["cal_const_diff"].astype(float).values,
        "rx",
        label="cal. const. diff",
    )
    for ti in t0:
        ax.axvline(ti, color="dimgrey", ls="--")
    for i in range(len(t0)):
        if np.isnan(res[i]):
            continue
        t_end = t0[i] + pd.Timedelta(days=7) if i == len(t0) - 1 else t0[i + 1]
        ax.plot([t0[i], t_end], [res[i] / 2, res[i] / 2], "b-")
        ax.plot([t0[i], t_end], [-res[i] / 2, -res[i] / 2], "b-")
        if quadratic:
            ax.plot(
                [t0[i], t_end],
                [res_quad[i] / 2, res_quad[i] / 2],
                color="dodgerblue",
                linestyle="-",
            )
            ax.plot(
                [t0[i], t_end],
                [-res_quad[i] / 2, -res_quad[i] / 2],
                color="dodgerblue",
                linestyle="-",
            )
        if not np.isnan(res[i]):
            ax.text(t0[i], res[i] / 2 * 1.1, f"{res[i]:.2f}", color="b")
        if quadratic and not np.isnan(res_quad[i]):
            ax.text(
                t0[i],
                res_quad[i] / 2 * 1.5,
                f"{res_quad[i]:.2f}",
                color="dodgerblue",
            )
    return t0


def _band(trace, trace_std):
    """x, values and ±1 sigma arrays aligned on the trace index."""
    x = trace.index.values
    vals = trace.values.astype(float)
    if trace_std is not None:
        sig = trace_std.reindex(trace.index).values.astype(float)
    else:
        sig = np.zeros(len(trace))
    return x, vals, sig


def _build_gain_shift_figure(
    period, detector, string, position, trace, trace_std, pul, cal, corrected, quadratic
):
    """Build one detector's gain-shift figure (legacy plot_time_series, part 1).

    Parameters
    ----------
    period : str
        Period being drawn (title only).
    detector : str
        Detector name.
    string : int | str
        Detector string (title only).
    position : int | str
        Detector position in the string (title only).
    trace : pandas.Series
        The kevdiff_av series (time-indexed).
    trace_std : pandas.Series | None
        Companion kevdiff_std series.
    pul : pandas.Series | None
        PULS01ANA kevdiff trace, drawn only when ``corrected``.
    cal : pandas.DataFrame | None
        This detector's cal_points rows (run_start-ordered).
    corrected : bool
        True for the pulser-corrected trace (C4), else uncorrected dodgerblue.
    quadratic : bool
        Also draw the quadratic resolution band.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The assembled figure (caller saves and closes it).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    x, vals, sig = _band(trace, trace_std)
    ax.fill_between(
        x, vals - sig, vals + sig, color="k", alpha=0.2, label=r"±1$\sigma$"
    )
    if corrected:
        if pul is not None and not pul.dropna().empty:
            ax.plot(pul.index.values, pul.values.astype(float), "C2", label="PULS01ANA")
        ax.plot(x, vals, "C4", label="GED corrected")
    else:
        ax.plot(x, vals, color="dodgerblue", label="GED uncorrected")
    t0 = _draw_cal_points(ax, cal, quadratic) if cal is not None else []
    fig.suptitle(
        f"period: {period} - string: {string} - position: {position} - ged: {detector}"
    )
    ax.set_ylabel(r"Energy diff / keV")
    ax.plot([], [], "b", label=QBB_LIN_LABEL)  # legend proxy: empty, never autoscales
    if quadratic:
        ax.plot([1, 2], [1, 2], "dodgerblue", label=QBB_QUAD_LABEL)
    if t0:
        time_difference = _naive_ts(trace.index.max()) - t0[-1]
        ax.set_xlim(t0[0] - pd.Timedelta(hours=8), t0[-1] + time_difference * 1.5)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def _build_param_figure(
    period, parameter, detector, string, position, trace, trace_std, pul, t0, res0
):
    """Build one detector's parameter figure (legacy plot_time_series, part 2).

    Parameters
    ----------
    period : str
        Period being drawn (title only).
    parameter : str
        One of ``STABILITY_PARAMETERS`` (keys of utils.MTG_PLOT_INFO).
    detector : str
        Detector name.
    string : int | str
        Detector string (title only).
    position : int | str
        Detector position in the string (title only).
    trace : pandas.Series
        The parameter series (time-indexed, already legacy-scaled).
    trace_std : pandas.Series | None
        Companion std series.
    pul : pandas.Series | None
        PULS01ANA trace, drawn only for the corrected TrapemaxCtcCal figure.
    t0 : pandas.Timestamp | None
        Run start of the drawn run (naive).
    res0 : float
        FWHM resolution of the drawn run (NaN when unknown).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The assembled figure (caller saves and closes it).
    """
    import matplotlib.pyplot as plt

    info = utils.MTG_PLOT_INFO[parameter]
    corrected = parameter == "TrapemaxCtcCal" and pul is not None
    fig, ax = plt.subplots(figsize=(12, 4))
    x, vals, sig = _band(trace, trace_std)
    if corrected:
        if not pul.dropna().empty:
            ax.plot(pul.index.values, pul.values.astype(float), "C2", label="PULS01ANA")
        ax.plot(x, vals, "C4", label="GED corrected")
    else:
        ax.plot(x, vals, color=info["colors"][0], label="GED uncorrected")
    ax.fill_between(
        x, vals - sig, vals + sig, color="k", alpha=0.2, label=r"±1$\sigma$"
    )

    threshold = [-res0 / 2, res0 / 2] if "Trapemax" in parameter else info["limits"]
    if t0 is not None:
        span = [t0, t0 + pd.Timedelta(days=7)]
        if parameter == "TrapemaxCtcCal":
            ax.plot(span, [res0 / 2, res0 / 2], color=info["colors"][1], ls="-")
            ax.plot(span, [-res0 / 2, -res0 / 2], color=info["colors"][1], ls="-")
            if not np.isnan(res0):
                ax.text(t0, res0 / 2 * 1.1, f"{res0:.2f}", color=info["colors"][1])
            ax.plot([], [], color=info["colors"][1], label=QBB_LIN_LABEL)
        else:
            if threshold[1] is not None:
                ax.plot(
                    span,
                    [threshold[1], threshold[1]],
                    color=info["colors"][1],
                    ls="-",
                    label="Threshold",
                )
            if threshold[0] is not None:
                ax.plot(
                    span,
                    [threshold[0], threshold[0]],
                    color=info["colors"][1],
                    ls="-",
                    label=None if threshold[1] is not None else "Threshold",
                )
    ax.set_ylabel(info["ylabel"])
    fig.suptitle(
        f"period: {period} - string: {string} - position: {position} - ged: {detector}"
    )
    if t0 is not None:
        time_difference = _naive_ts(trace.index.max()) - t0
        ax.set_xlim(t0 - pd.Timedelta(hours=0.5), t0 + time_difference * 1.1)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def _fep_baseline(rows):
    """Recover the run baseline energy from the stored drift definition.

    ``drift_kev = (mean - baseline) / baseline * 2039`` in
    calibration.compute_fep_gain_variation, inverted on the first valid bin.

    Parameters
    ----------
    rows : pandas.DataFrame
        One detector's ``fep_gain_stab`` rows.

    Returns
    -------
    baseline : float | None
        Baseline FEP energy; None when no bin has a valid mean and drift.
    """
    valid = rows.dropna(subset=["mean", "drift_kev"])
    if valid.empty:
        return None
    first = valid.iloc[0]
    return float(first["mean"]) / (1.0 + float(first["drift_kev"]) / 2039.0)


def _build_fep_gain_figure(period, run, detector, string, position, rows):
    """Build one detector's FEP gain-stability figure.

    Ported from the drawing section of ``calibration.fep_gain_variation``;
    the raw-event 2D histogram is not reproducible from the binned contract
    data and is therefore not drawn.

    Parameters
    ----------
    period : str
        Period being drawn.
    run : str
        Run being drawn.
    detector : str
        Detector name.
    string : int | str
        Detector string (title only).
    position : int | str
        Detector position in the string (title only).
    rows : pandas.DataFrame
        This detector's ``fep_gain_stab/<run>`` rows, time-ordered.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The assembled figure (caller saves and closes it).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    baseline = _fep_baseline(rows)
    if baseline is not None:
        drift = rows["drift_kev"].astype(float)
        band = rows["std"].astype(float) / baseline * 2039.0
        ax.plot(rows["time_s"], drift, "x-", color="red", label="10 min mean")
        ax.fill_between(
            rows["time_s"],
            drift - band,
            drift + band,
            color="red",
            alpha=0.15,
            label="±1 std",
        )
    ax.axhline(-2, ls="--", color="black", label=r"$\pm$2 keV threshold")
    ax.axhline(2, ls="--", color="black")
    ax.axhspan(2, 500, color="gray", alpha=0.25)
    ax.axhspan(-2, -500, color="gray", alpha=0.25)
    ax.set_ylim(-10, 10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("FEP gain variation (keV)")
    ax.set_title(f"{period} {run} string {string} position {position} {detector}")
    ax.legend(loc="lower left", title=f"Minimum counts = {FEP_MIN_COUNTS}")
    fig.tight_layout()
    return fig


def plot_stability_series(
    output_folder,
    period,
    run,
    *,
    detector_map=None,
    data_type="phy",
    save_pdf=True,
    png_dir=None,
    quadratic=False,
    logger=None,
):
    """Redraw the per-detector stability figures from the period contract.

    Covers the legacy ``monitoring.plot_time_series`` per-detector PDFs: the
    corrected/uncorrected gain-shift figures (period level,
    ``<period>/mtg/pdf/st<string>/{period}_string{string}_pos{pos}_{det}_{corr|uncorr}_gain_shift.pdf``)
    and the four parameter figures (run level,
    ``<period>/<run>/mtg/pdf/st<string>/{period}_{run}_string{string}_pos{pos}_{det}_{title}.pdf``).

    Parameters
    ----------
    output_folder : str
        Monitoring root, the folder containing ``<period>/``.
    period : str
        Period to draw.
    run : str
        Run whose contract keys are drawn.
    detector_map : pandas.DataFrame, optional
        Columns name/string/position; read from the run contract when None.
    data_type : str
        Data type routing the period contract file (default ``phy``).
    save_pdf : bool
        Write the legacy-named PDF files (default True).
    png_dir : str, optional
        Also save each figure there as PNG (PDF basename with ``.png``).
    quadratic : bool
        Also draw the quadratic resolution band on gain-shift figures.
    logger : logging.Logger, optional
        Where SAVED_PLOT lines and skip warnings go (default utils.logger).

    Returns
    -------
    saved : list
        Absolute paths of every file saved.
    """
    import matplotlib.pyplot as plt

    apply_monitoring_style()
    log = logger if logger is not None else utils.logger
    path = period_contract_path(output_folder, period, data_type)
    if not os.path.exists(path):
        log.warning("no period contract file %s", path)
        return []
    detector_map = _load_detector_map(
        output_folder, period, run, data_type, detector_map, log
    )
    cal = _read_frame(path, f"cal_points/{run}", log)
    # absent whenever no detector had a usable pulser trace: not a warning
    pul = _read_frame(path, f"pul_cusp/kevdiff/{run}", log, warn=False)
    saved = []

    for plot_type in ("corr", "uncorr"):
        trace_frame = _read_frame(path, f"gain_shift/{plot_type}/{run}", log)
        if trace_frame is None:
            continue
        std_frame = _read_frame(path, f"gain_shift/{plot_type}_std/{run}", log)
        for detector in trace_frame.columns:
            trace = trace_frame[detector]
            if trace.dropna().empty:
                continue
            string, position = _string_position(detector_map, cal, detector)
            if string is None:
                log.warning("no string/position for %s; figure skipped", detector)
                continue
            # a detector missing from pul_cusp was stored uncorrected (legacy)
            corrected = (
                plot_type == "corr" and pul is not None and detector in pul.columns
            )
            fig = _build_gain_shift_figure(
                period,
                detector,
                string,
                position,
                trace,
                std_frame[detector] if std_frame is not None else None,
                pul[detector] if corrected else None,
                _detector_cal(cal, detector),
                corrected,
                quadratic,
            )
            pdf_dir = os.path.join(output_folder, period, "mtg", "pdf", f"st{string}")
            pdf_name = f"{period}_string{string}_pos{position}_{detector}_{plot_type}_gain_shift.pdf"
            saved += _save_figure(fig, pdf_dir, pdf_name, save_pdf, png_dir, log)
            plt.close(fig)

    for parameter in STABILITY_PARAMETERS:
        trace_frame = _read_frame(path, f"param_stability/{parameter}/{run}", log)
        if trace_frame is None:
            continue
        std_frame = _read_frame(path, f"param_stability/{parameter}_std/{run}", log)
        title = utils.MTG_PLOT_INFO[parameter]["title"]
        for detector in trace_frame.columns:
            trace = trace_frame[detector]
            if trace.dropna().empty:
                continue
            string, position = _string_position(detector_map, cal, detector)
            if string is None:
                log.warning("no string/position for %s; figure skipped", detector)
                continue
            det_cal = _detector_cal(cal, detector)
            t0, res0 = None, float("nan")
            if det_cal is not None:
                # cal_points has no run column: the drawn run is the latest point
                last = det_cal.iloc[-1]
                t0 = _naive_ts(last["run_start"])
                res0 = float(last["res"]) if "res" in det_cal else float("nan")
            pul_trace = None
            if (
                parameter == "TrapemaxCtcCal"
                and pul is not None
                and detector in pul.columns
            ):
                # the stored pulser trace spans the period; clip to this run
                pul_trace = pul[detector]
                pul_trace = pul_trace[
                    (pul_trace.index >= trace.index.min())
                    & (pul_trace.index <= trace.index.max())
                ]
            fig = _build_param_figure(
                period,
                parameter,
                detector,
                string,
                position,
                trace,
                std_frame[detector] if std_frame is not None else None,
                pul_trace,
                t0,
                res0,
            )
            pdf_dir = os.path.join(
                output_folder, period, run, "mtg", "pdf", f"st{string}"
            )
            pdf_name = (
                f"{period}_{run}_string{string}_pos{position}_{detector}_{title}.pdf"
            )
            saved += _save_figure(fig, pdf_dir, pdf_name, save_pdf, png_dir, log)
            plt.close(fig)
    return saved


def plot_fep_gain(
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
    """Redraw the FEP gain-stability figures from the cal period contract.

    Covers the legacy ``calibration.fep_gain_variation`` PDFs:
    ``<period>/<run>/mtg/pdf/st<string>/{period}_{run}_string{string}_pos{position}_{ged}_FEP_gain_stab.pdf``.

    Parameters
    ----------
    output_folder : str
        Monitoring root, the folder containing ``<period>/``.
    period : str
        Period to draw.
    run : str
        Run whose ``fep_gain_stab/<run>`` key is drawn.
    detector_map : pandas.DataFrame, optional
        Columns name/string/position; read from the run contract when None.
    data_type : str
        Data type routing the period contract file (default ``cal``, matching
        calibration.write_fep_gain_contract; also lac/ssc/rdc).
    save_pdf : bool
        Write the legacy-named PDF files (default True).
    png_dir : str, optional
        Also save each figure there as PNG (PDF basename with ``.png``).
    logger : logging.Logger, optional
        Where SAVED_PLOT lines and skip warnings go (default utils.logger).

    Returns
    -------
    saved : list
        Absolute paths of every file saved.
    """
    import matplotlib.pyplot as plt

    apply_monitoring_style()
    log = logger if logger is not None else utils.logger
    path = period_contract_path(output_folder, period, data_type)
    frame = _read_frame(path, f"fep_gain_stab/{run}", log)
    if frame is None or frame.empty:
        return []
    detector_map = _load_detector_map(
        output_folder, period, run, data_type, detector_map, log
    )
    saved = []
    for detector, rows in frame.groupby("detector", sort=True):
        string, position = _string_position(detector_map, None, detector)
        if string is None:
            log.warning("no string/position for %s; figure skipped", detector)
            continue
        fig = _build_fep_gain_figure(
            period, run, detector, string, position, rows.sort_values("time_s")
        )
        pdf_dir = os.path.join(output_folder, period, run, "mtg/pdf", f"st{string}")
        pdf_name = (
            f"{period}_{run}_string{string}_pos{position}_{detector}_FEP_gain_stab.pdf"
        )
        saved += _save_figure(
            fig, pdf_dir, pdf_name, save_pdf, png_dir, log, bbox_inches="tight"
        )
        plt.close(fig)
    return saved
