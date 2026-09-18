import glob
import os

import awkward as ak
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from lgdo.lh5 import read_as

from . import errors, utils
from .contract import reader as contract_reader
from .contract import writer as contract_writer

# --- Phase 4 re-export shims: these functions moved to loading/ and processing/;
# import them here so existing ``monitoring.X`` references keep working. ---
from .loading.calib_files import (  # noqa: F401
    _first_run_key,
    _load_validity_file,
    _run_times_cache,
    add_calibration_runs,
    evaluate_fep_cal,
    extract_fep_peak,
    extract_resolution_at_q_bb,
    find_energy_key,
    get_calib_data_dict,
    get_calib_pars,
    get_calibration_file,
    get_energy_key,
    get_run_start_end_times,
    get_tier_keyresult,
    uncalibrated_variable,
)
from .processing.series import (  # noqa: F401
    compute_diff,
    compute_diff_and_rescaling,
    filter_by_period,
    filter_series_by_ignore_keys,
    find_hdf_file,
    get_dfs,
    get_pulser_data,
    get_spike_veto_series,
    read_if_key_exists,
    resample_series,
)

# -------------------------------------------------------------------------

SMALL_SIZE = 8


def period_contract_path(
    output_folder: str, period: str, data_type: str = "phy"
) -> str:
    """Path of the period-level monitoring contract file.

    One file per (period, datatype) holding the numbers the monitoring figures
    are drawn from, so consumers no longer have to unpickle a matplotlib
    figure out of a shelve to reach them.
    """
    return os.path.join(
        output_folder, period, f"l200-{period}-{data_type}-monitoring.hdf"
    )


def write_dead_time(
    output_folder: str,
    period: str,
    run: str,
    dead_time_s: float,
    dead_time_pct: float,
    data_type: str = "phy",
) -> str:
    """Record the discharge dead time of a run in the period contract file."""
    path = period_contract_path(output_folder, period, data_type)
    contract_writer.write_frame(
        path,
        f"dead_time/{run}",
        pd.DataFrame(
            [{"run": run, "dead_time_s": dead_time_s, "dead_time_pct": dead_time_pct}]
        ),
    )
    return path


def read_dead_time(
    output_folder: str, period: str, run: str, data_type: str = "phy"
) -> dict | None:
    """Dead time of a run, or None when it has not been computed yet.

    Callers must handle None: the value comes from qc_and_evt_summary_plots,
    which may not have run for this run yet.
    """
    path = period_contract_path(output_folder, period, data_type)
    if not os.path.isfile(path):
        return None
    try:
        frame = contract_reader.read_frame(path, f"dead_time/{run}")
    except (KeyError, OSError):
        return None
    if frame is None or frame.empty:
        return None
    row = frame.iloc[0]
    return {
        "dead_time_s": float(row["dead_time_s"]),
        "dead_time_pct": float(row["dead_time_pct"]),
    }


def apply_monitoring_style():
    """Apply the monitoring plot style to matplotlib's global rcParams.

    Called by the plot-generating functions; importing this module must not
    restyle the host application's matplotlib.
    """
    plt.rc("font", size=SMALL_SIZE)
    plt.rc("axes", titlesize=SMALL_SIZE)
    plt.rc("axes", labelsize=SMALL_SIZE)
    plt.rc("xtick", labelsize=SMALL_SIZE)
    plt.rc("ytick", labelsize=SMALL_SIZE)
    plt.rc("legend", fontsize=SMALL_SIZE)
    plt.rc("figure", titlesize=SMALL_SIZE)
    plt.rcParams["font.family"] = "serif"
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    plt.rc("axes", facecolor="white", edgecolor="black", axisbelow=True, grid=True)


IGNORE_KEYS = utils.IGNORE_KEYS
CALIB_RUNS = utils.CALIB_RUNS


# -------------------------------------------------------------------------
def write_qc_classifier_fractions(
    output_folder: str, period: str, run: str, rows: list, data_type: str = "phy"
) -> str | None:
    """Write the in-range fractions behind the QC classifier distributions.

    The distributions themselves are already published by the main pipeline as
    contract ``_dist`` histograms; what only existed inside these figures were
    the per-(classifier, detector, event type) percentages, so those are what
    this records.
    """
    if not rows:
        return None
    path = period_contract_path(output_folder, period, data_type)
    contract_writer.write_frame(path, f"qc_classifier_frac/{run}", pd.DataFrame(rows))
    return path


def qc_distributions(
    auto_dir_path: str,
    phy_mtg_data: str,
    output_folder: str,
    start_key: str,
    period: str,
    run: str,
    det_info: dict,
):
    """
    Publish the in-range fractions of every QC classifier per event type.

    Data-only: the per-detector histograms are in the run contract as
    ``_dist2d`` keys and the grids are drawn by
    ``plots.qc.plot_classifier_distributions``.

    Parameters
    ----------
    auto_dir_path : str
        Path to tmp-auto public data files.
    phy_mtg_data : str
        Path to the folder holding the run's v1 monitoring files.
    output_folder : str
        Monitoring output root (period contract file location).
    start_key : str
        First cycle of the run (detector-info vintage).
    period : str
        Period to inspect.
    run : str
        Run under inspection.
    det_info : dict
        Dictionary with channel names, IDs, and mapping to string and position.
    """
    pars_to_inspect = [
        "IsValidBlSlopeClassifier",
        "IsValidTailRmsClassifier",
        "IsValidPzSlopeClassifier",
        "IsValidBlSlopeRmsClassifier",
        "IsValidBlPolyRmsClassifier",
        "IsValidCuspeminClassifier",
        "IsValidCuspemaxClassifier",
    ]

    my_file = os.path.join(
        output_folder, f"{period}/{run}/l200-{period}-{run}-phy-geds.hdf"
    )
    str_chns = det_info["str_chns"]
    utils.logger.debug("...inspecting QC classifiers")
    if not os.path.exists(my_file):
        utils.logger.warning(f"...file not found: {my_file}. Return!")
        return

    def safe_perc(vals, lo=-5, hi=5):
        if len(vals) == 0:
            return float("nan")
        return 100 * np.mean((vals >= lo) & (vals <= hi))

    classifier_rows = []
    with pd.HDFStore(my_file, "r") as store:
        # the mask must come from the *unfiltered* frame: load_and_filter loads
        # its target unfiltered too, and pandas' where() needs both shapes to
        # match. Every frame below is passed through the ignore-keys filter
        # right after masking, so nothing survives that should have been dropped
        mask = store["/IsPhysics_TrapemaxCtcCal"] > 25

        for par in pars_to_inspect:
            frames = {
                "All": utils.load_and_filter(store, f"/All_{par}"),
                "IsPulser": utils.load_and_filter(store, f"/IsPulser_{par}"),
                "IsBsln": utils.load_and_filter(store, f"/IsBsln_{par}"),
                "IsPhysics": utils.load_and_filter(
                    store, f"/IsPhysics_{par}", mask=mask
                ),
            }
            if frames["All"].empty:
                continue
            for flag, frame in frames.items():
                if not frame.empty:
                    frames[flag] = filter_series_by_ignore_keys(
                        frame, utils.IGNORE_KEYS, period
                    )

            for string, det_list in str_chns.items():
                for det in det_list:
                    if det not in det_info["detectors"]:
                        continue
                    if not det_info["detectors"][det]["processable"]:
                        continue
                    ch = det_info["detectors"][det]["daq_rawid"]
                    if ch not in frames["All"].keys():
                        continue
                    for flag, frame in frames.items():
                        vals = utils.get_vals(frame, ch)
                        vals = vals[~np.isnan(vals)]
                        classifier_rows.append(
                            {
                                "run": run,
                                "classifier": par,
                                "detector": det,
                                "string": string,
                                "event_type": flag,
                                "percent_in_range": float(safe_perc(vals)),
                                "n_events": int(len(vals)),
                            }
                        )

    write_qc_classifier_fractions(output_folder, period, run, classifier_rows)


def mhz_to_percent(mhz, avg_total_forced_mhz):
    return (mhz / avg_total_forced_mhz) * 100


def percent_to_mhz(pct, avg_total_forced_mhz):
    return (pct / 100) * avg_total_forced_mhz


def write_ft_series(
    output_folder: str,
    period: str,
    run: str,
    name: str,
    frame,
    data_type: str = "phy",
) -> str | None:
    """Write a forced-trigger monitoring series into the period contract file.

    ``name`` distinguishes the quantities behind the FT figures:
    ``per_detector`` / ``per_string`` (hourly rates, mHz/kg), ``total_forced``
    (hourly counts over the array) and ``survival_fraction`` (%).
    """
    if frame is None or len(frame) == 0:
        return None
    if isinstance(frame, pd.Series):
        frame = frame.to_frame(name=name)
    path = period_contract_path(output_folder, period, data_type)
    contract_writer.write_frame(path, f"ft_summary/{name}/{run}", frame)
    return path


def qc_and_evt_summary_plots(
    auto_dir_path: str,
    phy_mtg_data: str,
    output_folder: str,
    start_key: str,
    period: str,
    run: str,
    det_info: dict,
):
    """
    Publish FT failure rates, event rates and the discharge dead time.

    Data-only: the figures are drawn from the contract by
    ``plots.summary.plot_ft_summary`` / ``plot_event_rate_qc``.

    Parameters
    ----------
    auto_dir_path : str
        Path to tmp-auto public data files.
    phy_mtg_data : str
        Path to the folder holding the run's v1 monitoring files.
    output_folder : str
        Monitoring output root (period contract file location).
    start_key : str
        First cycle of the run (detector-info vintage).
    period : str
        Period to inspect.
    run : str
        Run under inspection.
    det_info : dict
        Dictionary with channel names, IDs, and mapping to string and position.
    """
    utils.logger.debug("...inspecting FT failure rates")
    evt_files_phy = sorted(
        glob.glob(f"{auto_dir_path}/generated/tier/evt/phy/{period}/{run}/*.lh5")
    )

    if not evt_files_phy:
        evt_files_phy = sorted(
            glob.glob(f"{auto_dir_path}/generated/tier/pet/phy/{period}/{run}/*.lh5")
        )

    ged_pul = read_as(
        "evt/coincident", evt_files_phy, "ak", field_mask=["geds", "puls"]
    )
    forced = read_as(
        "evt/trigger", evt_files_phy, "ak", field_mask=["is_forced", "timestamp"]
    )
    is_bb = read_as(
        "evt/geds/quality",
        evt_files_phy,
        "ak",
        field_mask=["is_bb_like", "is_good_channel"],
    )
    is_dis = read_as(
        "evt/geds/quality/is_not_bb_like",
        evt_files_phy,
        "ak",
        field_mask=["is_delayed_discharge"],
    )
    is_fail = read_as(
        "evt/geds/quality/is_not_bb_like",
        evt_files_phy,
        "ak",
        field_mask=["is_empty_bits", "rawid"],
    )

    # build dataframe for FT FAILING events (vectorized: one count matrix fill
    # instead of a python loop over every event)
    mask = forced.is_forced & ~is_bb.is_bb_like & ~is_dis.is_delayed_discharge
    temp = is_fail.rawid[mask]
    n_events = len(temp)
    flat_ch = ak.to_numpy(ak.flatten(temp))
    channels = np.unique(flat_ch)
    counts = np.zeros((n_events, len(channels)))
    if flat_ch.size:
        event_idx = np.repeat(np.arange(n_events), ak.to_numpy(ak.num(temp)))
        np.add.at(counts, (event_idx, np.searchsorted(channels, flat_ch)), 1)
    y = {ch: counts[:, j] for j, ch in enumerate(channels)}
    y["timestamp"] = ak.to_numpy(forced.timestamp[mask])

    df = pd.DataFrame(y)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df.set_index("timestamp", inplace=True)
    daily_cnt = df.resample("h").sum()

    str_counts = {}
    det_rates = {}
    on_mass = 0

    for string, det_list in det_info["str_chns"].items():
        string_counts = None
        string_mass = 0
        for det in det_list:
            if not det_info["detectors"][det]["processable"]:
                continue
            ch = det_info["detectors"][det]["daq_rawid"]
            if ch not in daily_cnt.columns:
                continue

            mass = det_info["detectors"][det]["mass_in_kg"]
            if det_info["detectors"][det]["usability"] == "on":
                on_mass += mass

            det_rates[det] = daily_cnt[ch] / 3600 * 1000 / mass
            det_counts = daily_cnt[ch]
            string_counts = (
                det_counts
                if string_counts is None
                else string_counts.add(det_counts, fill_value=0)
            )
            string_mass += mass

        if string_counts is not None and string_mass > 0:
            str_counts[string] = string_counts / 3600 * 1000 / string_mass
        else:
            str_counts[string] = None

    # the numbers behind the FT figures, published as data
    write_ft_series(output_folder, period, run, "per_detector", pd.DataFrame(det_rates))
    write_ft_series(
        output_folder,
        period,
        run,
        "per_string",
        pd.DataFrame({str(k): v for k, v in str_counts.items() if v is not None}),
    )

    # --- FT survival fraction ---
    mask_forced = forced.is_forced
    mask_survived = mask_forced & is_bb.is_bb_like & ~is_dis.is_delayed_discharge
    ts_all = pd.to_datetime(forced.timestamp[mask_forced], unit="s")
    ts_survived = pd.to_datetime(forced.timestamp[mask_survived], unit="s")
    df_all = pd.DataFrame({"count": 1}, index=ts_all)
    df_survived = pd.DataFrame({"count": 1}, index=ts_survived)
    total_forced = df_all.resample("h").sum()["count"]
    surviving = df_survived.resample("h").sum()["count"]
    surviving_frac = surviving / total_forced * 100
    write_ft_series(output_folder, period, run, "total_forced", total_forced)
    write_ft_series(output_folder, period, run, "survival_fraction", surviving_frac)

    # --- Event rates ---
    base = (
        ged_pul.geds & ~ged_pul.puls & ~forced.is_forced & ~is_dis.is_delayed_discharge
    )
    ser = pd.to_datetime(
        forced.timestamp[ged_pul.geds & ~ged_pul.puls & ~forced.is_forced], unit="s"
    )
    ser_dis = pd.to_datetime(
        forced.timestamp[
            ged_pul.geds
            & ~ged_pul.puls
            & ~forced.is_forced
            & is_dis.is_delayed_discharge
        ],
        unit="s",
    )
    ser_pass = pd.to_datetime(forced.timestamp[base & is_bb.is_bb_like], unit="s")
    ser_fail = pd.to_datetime(forced.timestamp[base & ~is_bb.is_bb_like], unit="s")

    write_event_rate_qc(
        output_folder,
        period,
        run,
        {
            "All events": ser,
            "Delayed discharges": ser_dis,
            "Failing QC": ser_fail,
            "Surviving QC": ser_pass,
        },
        on_mass,
    )

    # --- Dead time from discharge windows ---
    mask_puls = ged_pul.puls
    mask_puls_no_dis = ged_pul.puls & ~is_dis.is_delayed_discharge

    length = len(ak.flatten(ak.where(mask_puls)))
    length_no_dis = len(ak.flatten(ak.where(mask_puls_no_dis)))

    # pulser period is assumed to be of 20 s
    livetime_total = length * 20
    livetime_no_dis = length_no_dis * 20

    dead_time_s = livetime_total - livetime_no_dis
    dead_time_pct = (dead_time_s / livetime_total * 100) if livetime_total > 0 else 0.0
    write_dead_time(output_folder, period, run, dead_time_s, dead_time_pct)

    utils.logger.info(
        f"...dead time from discharges: {dead_time_s:.1f} s ({dead_time_pct:.4f} %)"
    )


def compute_detector_summary(results: dict, det_info: dict, pars: dict) -> pd.DataFrame:
    """Per-detector summary of a monitoring parameter (the box-plot data).

    One row per detector: the mean/std/min/max of its values over the run, its
    Qbb resolution from the calibration pars, and its position and usability
    from the channel map. No matplotlib involved, so the numbers can be
    written to the contract and re-read without a figure.
    """
    detectors = det_info["detectors"]
    rows = []
    for ged, item in results.items():
        if ged not in detectors:
            continue
        meta_info = detectors[ged]

        if item is None or len(item) == 0:
            mean = std = min_val = max_val = np.nan
        else:
            mean = np.nanmean(item)
            std = np.nanstd(item)
            min_val = np.nanmin(item)
            max_val = np.nanmax(item)
        try:
            fwhm = get_energy_key(pars[ged]["results"]["ecal"])["eres_linear"][
                "Qbb_fwhm_in_kev"
            ]
        except (KeyError, TypeError):
            fwhm = np.nan

        rows.append(
            {
                "ged": ged,
                "string": meta_info["string"],
                "pos": meta_info["position"],
                "mean": mean,
                "std": std,
                "min": min_val,
                "max": max_val,
                "fwhm": fwhm,
                "usability": meta_info.get("usability", None),
            }
        )
    return pd.DataFrame(rows)


def write_detector_summary(
    output_folder: str,
    period: str,
    run: str,
    metric: str,
    frame: pd.DataFrame,
    data_type: str = "phy",
) -> str | None:
    """Write a per-detector summary table into the period contract file."""
    if frame is None or frame.empty:
        return None
    path = period_contract_path(output_folder, period, data_type)
    contract_writer.write_frame(path, f"detector_summary/{metric}/{run}", frame)
    return path


def box_summary_plot(
    period: str,
    run: str,
    pars: dict,
    det_info: dict,
    results: dict,
    info: dict,
    output_dir: str,
    data_type: str,
    run_to_apply=None,
):
    """
    Publish the per-detector summary of one monitoring parameter.

    Data-only (the name survives from the figure it used to draw): the box
    figure is drawn from the contract by ``plots.summary.plot_detector_summary``.

    Parameters
    ----------
    period : str
        Period to inspect.
    run : str
        Run to inspect.
    pars : dict
        Calibration results for each detector.
    det_info : dict
        Dictionary with channel names, IDs, and mapping to string and position.
    results : dict
        Dictionary with arrays values (per detector); None if invalid.
    info : dict
        Dictionary containing info on a parameter basis (label, title, limits).
    output_dir : str
        Monitoring output root (period contract file location).
    data_type : str
        Type of data, either 'cal' or 'phy' (or lac/ssc/rdc).
    run_to_apply :
        Run to apply (eg see ssc data).
    """
    utils.logger.debug("...summarizing %s per detector", info["title"])
    df_plot = compute_detector_summary(results, det_info, pars)
    write_detector_summary(
        output_dir, period, run, info["title"], df_plot, data_type=data_type
    )
    if df_plot.empty:
        raise errors.DataError(
            f"box_summary_plot: no detector results for '{info['title']}' "
            "(empty or missing input data)"
        )


def compute_qc_rate_mhz(frame: pd.DataFrame, period: str) -> pd.Series | None:
    """Per-detector rate in mHz over a QC flag frame's time span.

    ``frame`` is a (time x rawid) frame of per-event flags, as stored in the
    v1 monitoring HDF; IGNORE_KEYS ranges are dropped first. Returns None when
    the frame carries no usable time span.
    """
    filtered = filter_series_by_ignore_keys(frame, utils.IGNORE_KEYS, period)
    if filtered.empty:
        return None
    span = (filtered.index.max() - filtered.index.min()).total_seconds()
    if not span > 0:
        return None
    return filtered.sum(axis=0) / span * 1000


def write_event_rate_qc(
    output_folder: str,
    period: str,
    run: str,
    series_by_label: dict,
    on_mass: float,
    data_type: str = "phy",
) -> str | None:
    """
    Write the QC-split hourly event rates behind the event-rate figure.

    Parameters
    ----------
    output_folder : str
        Monitoring output root (the folder containing ``<period>/``).
    period, run : str
        Run the rates belong to.
    series_by_label : dict
        Label -> DatetimeIndex of event times; each is histogrammed hourly and
        normalised to mHz/kg with ``on_mass``.
    on_mass : float
        Total ON detector mass in kg (kept as its own column, so consumers can
        undo the normalisation).
    data_type : str
        Data type key of the period contract file.

    Returns
    -------
    key: str or None
        The key written, or None when every series is empty.
    """
    columns = {}
    for label, times in (series_by_label or {}).items():
        if times is None or len(times) == 0:
            continue
        counts, edges = np.histogram(
            times, bins=pd.date_range(start=times.min(), end=times.max(), freq="h")
        )
        rate = pd.Series(
            counts / 3600 * 1000 / on_mass, index=pd.DatetimeIndex(edges[:-1])
        )
        columns[label.lower().replace(" ", "_")] = rate
    if not columns:
        return None
    frame = pd.DataFrame(columns)
    frame["on_mass_kg"] = on_mass
    path = period_contract_path(output_folder, period, data_type)
    return contract_writer.write_frame(path, f"event_rate_qc/{run}", frame)


def write_slow_control(
    output_folder: str,
    period: str,
    run: str,
    parameter: str,
    frame: pd.DataFrame,
    data_type: str = "phy",
) -> str | None:
    """
    Publish one slow-control parameter for a run to the period contract file.

    Parameters
    ----------
    output_folder : str
        Monitoring output root (the folder containing ``<period>/``).
    period, run : str
        Run the readings were queried for.
    parameter : str
        SC parameter as named in ``SC-params.yaml`` (``DaqLeft-Temp1``); the
        key uses underscores, matching the dashboard's selector vocabulary.
    frame : pandas.DataFrame
        ``SlowControl.data``: ``tstamp``, ``value``, ``unit``, ``lower_lim``,
        ``upper_lim`` columns.
    data_type : str
        Data type key of the period contract file.

    Returns
    -------
    key: str or None
        The key written, or None when the frame is empty.
    """
    if frame is None or frame.empty:
        return None
    series = frame.set_index(pd.DatetimeIndex(frame["tstamp"], name="datetime"))
    series = series[["value", "unit", "lower_lim", "upper_lim"]].sort_index()
    path = period_contract_path(output_folder, period, data_type)
    key = f"slow_control/{parameter.replace('-', '_')}/{run}"
    return contract_writer.write_frame(path, key, series)


def write_qc_rates(
    output_folder: str,
    period: str,
    run: str,
    rates_by_par: dict,
    detectors: dict,
    data_type: str = "phy",
) -> str | None:
    """Write per-(flag, detector) QC rates into the period contract file."""
    rawid_to_name = {info.get("daq_rawid"): name for name, info in detectors.items()}
    rows = []
    for par, rates in rates_by_par.items():
        if rates is None:
            continue
        for rawid, rate in rates.items():
            rows.append(
                {
                    "run": run,
                    "flag": par,
                    "rawid": int(rawid),
                    "detector": rawid_to_name.get(int(rawid)),
                    "rate_mhz": float(rate),
                }
            )
    if not rows:
        return None
    path = period_contract_path(output_folder, period, data_type)
    contract_writer.write_frame(path, f"qc_average/{run}", pd.DataFrame(rows))
    return path


def qc_average(
    auto_dir_path: str,
    output_folder: str,
    det_info: dict,
    period: str,
    run: str,
    pars_to_inspect: list | None = None,
):
    """
    Evaluate the average QC rates and publish them to the period contract.

    Data-only: the figures are drawn from the contract by
    ``plots.qc.plot_qc_average``. The IsDischarge/IsSaturated rate limits and
    the total discharge dead-time limit still land in ``qcp_summary.yaml``.

    Parameters
    ----------
    auto_dir_path : str
        Path to tmp-auto public data files.
    output_folder : str
        Path to generated monitoring hdf files.
    det_info : dict
        Dictionary with channel names, IDs, and mapping to string and position.
    period : str
        Period to inspect.
    run : str
        Run under inspection.
    pars_to_inspect : list
        List of parameters (boolean flags) to inspect.
    """
    if pars_to_inspect is None:
        pars_to_inspect = [
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

    my_file = os.path.join(
        output_folder, f"{period}/{run}/l200-{period}-{run}-phy-geds.hdf"
    )
    detectors = det_info["detectors"]
    str_chns = det_info["str_chns"]
    utils.logger.debug("...inspecting QC average values")
    if not os.path.exists(my_file):
        utils.logger.warning(f"...file not found: {my_file}. Return!")
        return

    usability_map_file = os.path.join(
        output_folder,
        period,
        run,
        f"l200-{period}-{run}-qcp_summary.yaml",
    )
    output = utils.load_yaml_or_default(usability_map_file, detectors)

    dead_time = read_dead_time(output_folder, period, run)
    if dead_time is None:
        utils.logger.warning(
            "\033[93mno dead time recorded for %s-%s; "
            "evaluating IsDischarge without it\033[0m",
            period,
            run,
        )
    dt_condition = bool(
        dead_time is not None
        and dead_time["dead_time_pct"]
        > utils.MTG_PLOT_INFO["tot_discharge_dead_time"]["limits"][1]
    )

    rates_by_par = {}
    with pd.HDFStore(my_file, "r") as store:
        for par in pars_to_inspect:
            key = f"/IsPhysics_{par}"
            if key not in store:
                utils.logger.debug("...skipping %s (not found in HDF)", par)
                continue
            rates = compute_qc_rate_mhz(store[key], period)
            if rates is None:
                utils.logger.debug("...no usable time span for %s. Skip it!", par)
                continue
            rates_by_par[par] = rates

            if par not in ("IsDischarge", "IsSaturated"):
                continue
            info = utils.MTG_PLOT_INFO[par]
            limit = info["limits"][1]  # no lower limit for rates
            # the hourly series behind the verdict, for the issue's excursion stats
            hourly = compute_qc_rate_series(store[key], period, detectors=detectors)
            dt_info = utils.MTG_PLOT_INFO["tot_discharge_dead_time"]
            for det_list in str_chns.values():
                for det_name in det_list:
                    rawid = detectors[det_name]["daq_rawid"]
                    if rawid not in rates:
                        utils.logger.debug(
                            f"{det_name} ({rawid}) missing in dataframe for {par}"
                        )
                        continue
                    condition = bool((rates[rawid] > limit).any())
                    utils.update_evaluation_in_memory(
                        output, det_name, "phy", info["title"], not condition
                    )
                    if condition:
                        series = (
                            hourly[det_name]
                            if hourly is not None and det_name in hourly
                            else None
                        )
                        utils.issues.record_detail(
                            period,
                            run,
                            "phy",
                            det_name,
                            info["title"],
                            observed=float(rates[rawid]),
                            threshold=[None, limit],
                            unit=info.get("unit"),
                            window=(
                                [str(series.index[0]), str(series.index[-1])]
                                if series is not None and len(series)
                                else None
                            ),
                            excursion=utils.issues.evaluate_excursion(
                                series, None, limit
                            ),
                        )
                    utils.update_evaluation_in_memory(
                        output, det_name, "phy", dt_info["title"], not dt_condition
                    )
                    if dt_condition:
                        utils.issues.record_detail(
                            period,
                            run,
                            "phy",
                            det_name,
                            dt_info["title"],
                            observed=float(dead_time["dead_time_pct"]),
                            threshold=list(dt_info["limits"]),
                            unit=dt_info.get("unit"),
                        )

    write_qc_rates(output_folder, period, run, rates_by_par, detectors)

    with open(usability_map_file, "w") as f:
        yaml.dump(output, f)


def compute_qc_rate_series(
    frame: pd.DataFrame, period: str, cadence: str = "1h", detectors: dict | None = None
) -> pd.DataFrame | None:
    """Per-detector QC flag rate versus time, in mHz.

    Resamples the whole (time x rawid) frame at once — equivalent to the
    per-detector resampling the figure does, column by column. Columns are
    renamed to detector names when a channel map is given.
    """
    filtered = filter_series_by_ignore_keys(frame, utils.IGNORE_KEYS, period)
    if filtered.empty:
        return None
    seconds = pd.Timedelta(cadence).total_seconds()
    rates = filtered.resample(cadence).sum() / seconds * 1000
    if detectors:
        rawid_to_name = {
            info.get("daq_rawid"): name for name, info in detectors.items()
        }
        rates = rates.rename(columns=lambda c: rawid_to_name.get(int(c), c))
    return rates


def write_qc_rate_series(
    output_folder: str,
    period: str,
    run: str,
    flag: str,
    rates: pd.DataFrame,
    data_type: str = "phy",
) -> str | None:
    """Write a QC rate-versus-time frame into the period contract file."""
    if rates is None or rates.empty:
        return None
    path = period_contract_path(output_folder, period, data_type)
    contract_writer.write_frame(path, f"qc_rate_series/{flag}/{run}", rates)
    return path


def qc_time_series(
    auto_dir_path: str,
    output_folder: str,
    det_info: dict,
    period: str,
    run: str,
    pars_to_inspect: list | None = None,
):
    """
    Publish the rate-vs-time of each QC flag to the period contract file.

    Data-only: the per-string figures are drawn from the contract by
    ``plots.qc.plot_qc_rate_series``.

    Parameters
    ----------
    auto_dir_path : str
        Path to tmp-auto public data files.
    output_folder : str
        Path to generated monitoring hdf files.
    det_info : dict
        Dictionary with channel names, IDs, and mapping to string and position.
    period : str
        Period to inspect.
    run : str
        Run under inspection.
    pars_to_inspect : list
        List of parameters (boolean flags) to inspect.
    """
    if pars_to_inspect is None:
        pars_to_inspect = [
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
    my_file = os.path.join(
        output_folder, f"{period}/{run}/l200-{period}-{run}-phy-geds.hdf"
    )
    detectors = det_info["detectors"]
    utils.logger.debug("...inspecting QC time series")
    if not os.path.exists(my_file):
        utils.logger.warning(f"...file not found: {my_file}. Return!")
        return

    with pd.HDFStore(my_file, "r") as store:
        for par in pars_to_inspect:
            key = f"/IsPhysics_{par}"
            if key not in store:
                utils.logger.debug("...skipping %s (not found in HDF)", key)
                continue
            write_qc_rate_series(
                output_folder,
                period,
                run,
                par,
                compute_qc_rate_series(store[key], period, detectors=detectors),
            )


def build_new_files(generated_path: str, period: str, run: str, data_type="phy"):
    """
    Generate and store resampled HDF files for a given data run and extract summary info.

    This function:

      - loads the original `.hdf` file for the specified `period` and `run`
      - extracts available keys from the HDF file
      - resamples all applicable time series data into multiple time intervals (10min, 60min)
      - stores each resampled dataset into a separate HDF file
      - extracts metadata from the 'info' key and saves it as a .yaml file

    Parameters
    ----------
    generated_path : str
        Root directory where the data is stored and where new files will be written.
    period : str
        Period (e.g. 'p03') used to construct paths.
    run : str
        Run (e.g. 'r001') used to construct paths.
    data_type : str
        Data type to load; default: 'phy'.
    """
    data_file = os.path.join(
        generated_path,
        "generated/plt/hit",
        data_type,
        period,
        run,
        f"l200-{period}-{run}-{data_type}-geds.hdf",
    )

    if not os.path.exists(data_file):
        utils.logger.debug(f"File not found: {data_file}. Exit here.")
        raise errors.DataError("build_new_files failed (see log for details)")

    with h5py.File(data_file, "r") as f:
        my_keys = list(f.keys())

    info_dict = {"keys": my_keys}

    resampling_times = ["10min", "60min"]

    for idx, resample_unit in enumerate(resampling_times):
        new_file = os.path.join(
            generated_path,
            "generated/plt/hit",
            data_type,
            period,
            run,
            f"l200-{period}-{run}-{data_type}-geds-res_{resample_unit}.hdf",
        )
        # remove it if already exists so we can start again to append resampled data
        if os.path.exists(new_file):
            os.remove(new_file)

        for k in my_keys:
            if "info" in k:
                # do it once
                if idx == 0:
                    original_df = pd.read_hdf(data_file, key=k)
                    original_df = original_df.astype(str)
                    info_dict.update(
                        {
                            k: {
                                "subsystem": original_df.loc["subsystem", "Value"],
                                "unit": original_df.loc["unit", "Value"],
                                "label": original_df.loc["label", "Value"],
                                "event_type": original_df.loc["event_type", "Value"],
                                "lower_lim_var": original_df.loc[
                                    "lower_lim_var", "Value"
                                ],
                                "upper_lim_var": original_df.loc[
                                    "upper_lim_var", "Value"
                                ],
                                "lower_lim_abs": original_df.loc[
                                    "lower_lim_abs", "Value"
                                ],
                                "upper_lim_abs": original_df.loc[
                                    "upper_lim_abs", "Value"
                                ],
                            }
                        }
                    )
                continue

            original_df = pd.read_hdf(data_file, key=k)

            # mean dataframe is kept
            if "_mean" in k:
                original_df.to_hdf(new_file, key=k, mode="a", **utils.HDF_COMPRESSION)
                continue

            original_df.index = pd.to_datetime(original_df.index)
            # resample
            resampled_df = original_df.resample(resample_unit).mean()
            # substitute the original df with the resampled one
            original_df = resampled_df
            # append resampled data to the new file
            resampled_df.to_hdf(new_file, key=k, mode="a", **utils.HDF_COMPRESSION)

        if idx == 0:
            info_output = os.path.join(
                generated_path,
                "generated/plt/hit",
                data_type,
                period,
                run,
                f"l200-{period}-{run}-{data_type}-geds-info.yaml",
            )
            with open(info_output, "w") as file:
                yaml.dump(info_dict, file, sort_keys=False)


def write_stability_series(
    output_folder: str,
    period: str,
    run: str,
    group: str,
    name: str,
    series: dict,
    data_type: str = "phy",
) -> str | None:
    """Write per-detector monitoring series into the period contract file.

    ``series`` maps detector name -> the pandas Series the figure plots, so the
    frame written here is exactly what was drawn (time x detector).
    """
    series = {det: s for det, s in (series or {}).items() if s is not None and len(s)}
    if not series:
        return None
    path = period_contract_path(output_folder, period, data_type)
    contract_writer.write_frame(path, f"{group}/{name}/{run}", pd.DataFrame(series))
    return path


def write_cal_points(
    output_folder: str, period: str, run: str, rows: list, data_type: str = "phy"
) -> str | None:
    """Write the per-run calibration points marked on the stability figures."""
    if not rows:
        return None
    path = period_contract_path(output_folder, period, data_type)
    contract_writer.write_frame(path, f"cal_points/{run}", pd.DataFrame(rows))
    return path


def collect_stability_series(
    auto_dir_path: str,
    phy_mtg_data: str,
    output_folder: str,
    data_type: str,
    period: str,
    runs: list,
    current_run: str,
    det_info: dict,
    escale_val: float,
    last_checked: float | None,
    partition: bool,
    quadratic: bool,
):
    """
    Collect the gain/parameter stability series and publish them as data.

    The data side of the retired ``plot_time_series``: pulser-corrected and
    uncorrected gain series over the period, the four per-run parameter
    series, the calibration points with their resolution thresholds, and the
    threshold verdicts for ``qcp_summary.yaml``. The per-detector figures are
    drawn from the contract by ``plots.stability.plot_stability_series``.

    Parameters
    ----------
    auto_dir_path : str
        Path to tmp-auto public data files.
    phy_mtg_data : str
        Path to generated monitoring hdf files.
    output_folder : str
        Path to output folder.
    data_type : str
        Data type routing the period contract file.
    period : str
        Period to inspect.
    runs : list
        Available runs to inspect for a given period.
    current_run : str
        Run under inspection.
    det_info : dict
        Dictionary containing detector metadata.
    escale_val : float
        Energy scale at which evaluating the gain differences.
    last_checked : float | None
        Timestamp of the last check.
    partition : bool
        False if not partition data.
    quadratic : bool
        Use the quadratic resolution fit.

    Returns
    -------
    results : dict
        parameter -> {detector: value array}, consumed by box_summary_plot.
    """
    avail_runs = [
        entry.replace(",", "").replace("[", "").replace("]", "") for entry in runs
    ]
    dataset = {period: avail_runs}
    period_list = list(dataset.keys())
    fit_flag = "quadratic" if quadratic is True else "linear"

    detectors = det_info["detectors"]
    str_chns = det_info["str_chns"]
    usability_map_file = os.path.join(
        output_folder,
        period,
        current_run,
        f"l200-{period}-{current_run}-qcp_summary.yaml",
    )
    output = utils.load_yaml_or_default(usability_map_file, detectors)

    # skip detectors with no pulser entries
    flag_expr = " or ".join(
        f'(channel == "{channel}" and period in {periods})'
        for channel, periods in utils.NO_PULS_DETS.items()
    )

    def no_pulser(channel, period):
        return bool(eval(flag_expr)) if flag_expr else False  # noqa: S307

    results = {}
    gain_shift_series = {}
    gain_shift_std_series = {}
    param_series = {}
    param_std_series = {}
    pul_cusp_series = {}
    cal_point_rows = []

    # --- gain over the whole period ---
    for period in period_list:
        run_list = dataset[period]
        (
            geds_df_abs,
            geds_df_abs_corr,
            puls_df_abs,
        ) = get_dfs(phy_mtg_data, period, run_list, "Trapemax")
        spike_veto = get_spike_veto_series(phy_mtg_data, period, run_list)
        if geds_df_abs is None or geds_df_abs_corr is None:
            utils.logger.debug("Dataframes are None for %s!", period)
            continue
        if geds_df_abs.empty:
            utils.logger.debug("Dataframes are empty for %s!", period)
            continue

        utils.logger.debug(f"...inspecting gain over {period}")
        for string, det_list in str_chns.items():
            for channel_name in det_list:
                channel = detectors[channel_name]["channel_str"]
                rawid = np.int64(detectors[channel_name]["daq_rawid"])
                pos = detectors[channel_name]["position"]
                if rawid not in set(geds_df_abs.columns):
                    utils.logger.debug(f"{channel} is not present in the dataframe!")
                    continue

                pulser_data = get_pulser_data(
                    "1h",
                    period,
                    geds_df_abs,
                    rawid,
                    escale=escale_val,
                    puls_abs=puls_df_abs,
                    spike_veto=spike_veto,
                    variations=True,
                )
                pars_data = get_calib_pars(
                    auto_dir_path,
                    period,
                    run_list,
                    [channel, channel_name],
                    partition,
                    data_type,
                    escale=escale_val,
                    fit=fit_flag,
                )
                cal_point_rows += [
                    {
                        "detector": channel_name,
                        "string": string,
                        "position": pos,
                        "run_start": start,
                        "fep_diff": fep,
                        "cal_const_diff": const,
                        "res": res,
                        "res_quad": res_quad,
                    }
                    for start, fep, const, res, res_quad in zip(
                        pars_data["run_start"],
                        pars_data["fep_diff"],
                        pars_data["cal_const_diff"],
                        pars_data["res"],
                        pars_data["res_quad"],
                    )
                ]
                if no_pulser(channel, period):
                    continue
                # corrected series when PULS01ANA has a signal, else uncorrected
                if pulser_data["pul"]["kevdiff_av"] is not None:
                    gain_shift_series.setdefault("corr", {})[channel_name] = (
                        pulser_data["diff"]["kevdiff_av"]
                    )
                    gain_shift_std_series.setdefault("corr", {})[channel_name] = (
                        pulser_data["diff"]["kevdiff_std"]
                    )
                    pul_cusp_series[channel_name] = pulser_data["pul"]["kevdiff_av"]
                else:
                    gain_shift_series.setdefault("corr", {})[channel_name] = (
                        pulser_data["ged"]["kevdiff_av"]
                    )
                    gain_shift_std_series.setdefault("corr", {})[channel_name] = (
                        pulser_data["ged"]["kevdiff_std"]
                    )
                gain_shift_series.setdefault("uncorr", {})[channel_name] = pulser_data[
                    "ged"
                ]["kevdiff_av"]
                gain_shift_std_series.setdefault("uncorr", {})[channel_name] = (
                    pulser_data["ged"]["kevdiff_std"]
                )

    # --- parameters (bsln, gain, ...) variations over the current run ---
    utils.logger.debug("...inspecting gain/bsln/etc time series")
    info = utils.MTG_PLOT_INFO
    last_checked = None

    for inspected_parameter in ["BlStd", "TrapemaxCtcCal", "Baseline", "Trapemax"]:
        escale_par = escale_val if inspected_parameter == "TrapemaxCtcCal" else 1
        results.update({inspected_parameter: {}})

        for period in period_list:
            (
                geds_df_abs,
                geds_df_abs_corr,
                puls_df_abs,
            ) = get_dfs(phy_mtg_data, period, [current_run], inspected_parameter)
            spike_veto = get_spike_veto_series(phy_mtg_data, period, [current_run])
            if geds_df_abs is None or geds_df_abs_corr is None:
                utils.logger.debug(
                    "Dataframes are None for %s-%s!", period, current_run
                )
                continue
            if geds_df_abs.empty:
                utils.logger.debug(
                    "Dataframes are empty for %s-%s!", period, current_run
                )
                continue

            utils.logger.debug(
                f"...inspecting {info[inspected_parameter]['title']} over {current_run}"
            )
            for _string, det_list in str_chns.items():
                for channel_name in det_list:
                    channel = detectors[channel_name]["channel_str"]
                    rawid = np.int64(detectors[channel_name]["daq_rawid"])
                    if rawid not in set(geds_df_abs.columns):
                        utils.logger.debug(
                            f"{channel} is not present in the dataframe!"
                        )
                        continue

                    pulser_data = get_pulser_data(
                        "1h",
                        period,
                        geds_df_abs,
                        rawid,
                        escale=escale_par,
                        puls_abs=puls_df_abs,
                        spike_veto=spike_veto,
                        variations=info[inspected_parameter]["percentage"],
                    )
                    pars_data = get_calib_pars(
                        auto_dir_path,
                        period,
                        [current_run],
                        [channel, channel_name],
                        partition,
                        data_type,
                        escale=escale_par,
                        fit=fit_flag,
                    )
                    threshold = (
                        [-pars_data["res"][0] / 2, pars_data["res"][0] / 2]
                        if "Trapemax" in inspected_parameter
                        else info[inspected_parameter]["limits"]
                    )
                    t0 = pars_data["run_start"]
                    if no_pulser(channel, period):
                        continue

                    if (
                        info[inspected_parameter]["percentage"] is True
                        and float(escale_par) == 1.0
                    ):
                        check_kevdiff = pulser_data["ged"]["kevdiff_av"] * 100
                    else:
                        check_kevdiff = pulser_data["ged"]["kevdiff_av"]
                    # check threshold and update YAML summary file
                    # (for energy, only TrapemaxCtcCal and not Trapemax for now)
                    if inspected_parameter != "Trapemax":
                        utils.check_threshold(
                            check_kevdiff,
                            channel_name,
                            last_checked,
                            t0,
                            threshold,
                            info[inspected_parameter]["title"],
                            output,
                            period=period,
                            run=current_run,
                        )

                    # PULS01ANA correction applies to energy parameters only
                    if (
                        pulser_data["pul"]["kevdiff_av"] is not None
                        and inspected_parameter == "TrapemaxCtcCal"
                    ):
                        param_series.setdefault(inspected_parameter, {})[
                            channel_name
                        ] = pulser_data["diff"]["kevdiff_av"]
                        param_std_series.setdefault(inspected_parameter, {})[
                            channel_name
                        ] = pulser_data["diff"]["kevdiff_std"]
                        results[inspected_parameter].update(
                            {
                                channel_name: pulser_data["pul"][
                                    "kevdiff_av"
                                ].values.astype(float)
                            }
                        )
                    else:
                        if (
                            info[inspected_parameter]["percentage"] is True
                            and float(escale_par) == 1.0
                        ):
                            pulser_data["ged"]["kevdiff_av"] *= 100
                            pulser_data["ged"]["kevdiff_std"] *= 100
                        param_series.setdefault(inspected_parameter, {})[
                            channel_name
                        ] = pulser_data["ged"]["kevdiff_av"]
                        param_std_series.setdefault(inspected_parameter, {})[
                            channel_name
                        ] = pulser_data["ged"]["kevdiff_std"]
                        results[inspected_parameter].update(
                            {
                                channel_name: pulser_data["ged"][
                                    "kevdiff_av"
                                ].values.astype(float)
                            }
                        )

    for plot_type, series in gain_shift_series.items():
        write_stability_series(
            output_folder,
            period,
            current_run,
            "gain_shift",
            plot_type,
            series,
            data_type=data_type,
        )
    for plot_type, series in gain_shift_std_series.items():
        write_stability_series(
            output_folder,
            period,
            current_run,
            "gain_shift",
            f"{plot_type}_std",
            series,
            data_type=data_type,
        )
    for parameter, series in param_series.items():
        write_stability_series(
            output_folder,
            period,
            current_run,
            "param_stability",
            parameter,
            series,
            data_type=data_type,
        )
    for parameter, series in param_std_series.items():
        write_stability_series(
            output_folder,
            period,
            current_run,
            "param_stability",
            f"{parameter}_std",
            series,
            data_type=data_type,
        )
    write_stability_series(
        output_folder,
        period,
        current_run,
        "pul_cusp",
        "kevdiff",
        pul_cusp_series,
        data_type=data_type,
    )
    write_cal_points(
        output_folder, period, current_run, cal_point_rows, data_type=data_type
    )

    with open(usability_map_file, "w") as f:
        yaml.dump(output, f)

    return results


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SiPM production by-products: per-cycle noise and the PE calibration in force
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def read_spms_noise(prod_path: str, period: str, run: str, data_type: str = "phy"):
    """
    Collect ``baseline_curr_fwhm`` per SiPM from the per-cycle ``par_dsp_spms`` files.

    Parameters
    ----------
    prod_path : str
        Production version root (the folder containing ``generated/``).
    period, run : str
        Run to read.
    data_type : str
        Data type of the par files.

    Returns
    -------
    pandas.DataFrame
        Cycle timestamp (``datetime`` index) x SiPM name, float32; empty when
        the run has no par files.
    """
    folder = os.path.join(prod_path, "generated/par/dsp", data_type, period, run)
    rows = {}
    for path in sorted(glob.glob(os.path.join(folder, "*-par_dsp_spms.yaml"))):
        key = os.path.basename(path).split("-")[4]
        with open(path) as f:
            pars = yaml.load(f, Loader=yaml.CLoader) or {}
        rows[pd.Timestamp(key, tz="UTC")] = {
            name: entry.get("baseline_curr_fwhm")
            for name, entry in pars.items()
            if isinstance(entry, dict)
        }
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame.from_dict(rows, orient="index").astype("float32")
    frame.index.name = "datetime"
    return frame.sort_index()


def _spms_overrides_in_force(overrides: str, validity: str, start_key: str) -> list:
    """
    Load the SiPM override files in force at ``start_key``, in merge order.

    Replays ``validity.yaml`` the way the parameter database does
    (reset/remove/append). Entries that point outside the overrides directory
    -- the hit validity list references ``../raw/...`` files -- or that do not
    parse to a mapping are skipped: they carry no SiPM calibration, and
    following them is what made the database lookup raise on p16/p18.

    Parameters
    ----------
    overrides : str
        ``inputs/dataprod/overrides/hit`` directory.
    validity : str
        The ``validity.yaml`` inside it.
    start_key : str
        Timestamp key the run starts at.

    Returns
    -------
    list
        ``(relative path, parsed mapping)`` pairs, in the order they apply.
    """
    with open(validity) as f:
        entries = yaml.load(f, Loader=yaml.CLoader) or []
    in_force: list = []
    for entry in sorted(entries, key=lambda e: str(e.get("valid_from", ""))):
        if str(entry.get("valid_from", "")) > start_key:
            break
        paths = [p for p in entry.get("apply", []) if p.startswith("lar/")]
        mode = entry.get("mode")
        if mode == "reset":
            in_force = list(paths)
        elif mode == "remove":
            in_force = [p for p in in_force if p not in entry.get("apply", [])]
        else:
            in_force += paths

    loaded = []
    for path in in_force:
        full = os.path.join(overrides, path)
        if not os.path.exists(full):
            # the file names carry a T% wildcard for the validity timestamp:
            # glob the entry's own basename pattern (newest match wins), so an
            # unrelated yaml sharing the directory is never picked up
            pattern = os.path.basename(path).replace("T%", "*")
            candidates = sorted(
                glob.glob(os.path.join(overrides, os.path.dirname(path), pattern))
            )
            if not candidates:
                continue
            full = candidates[-1]
        with open(full) as f:
            pars = yaml.load(f, Loader=yaml.CLoader)
        if isinstance(pars, dict):
            loaded.append((path, pars))
    return loaded


def _deep_update(base: dict, extra: dict) -> dict:
    """Merge ``extra`` into ``base`` recursively; later values win."""
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def read_spms_calibration(prod_path: str, start_key: str) -> pd.DataFrame:
    """
    Resolve the SiPM PE calibration (``energy_in_pe``/``is_valid_hit`` overrides) in force at ``start_key``.

    Parameters
    ----------
    prod_path : str
        Production version root (the folder containing ``inputs/``).
    start_key : str
        Timestamp key (``20260731T181831Z``) the run starts at.

    Returns
    -------
    pandas.DataFrame
        One row per SiPM: ``pe_a``, ``pe_m``, ``threshold_a`` and the
        ``source`` override file **that last defined that SiPM** (its
        period/run tell how stale its calibration is -- they differ per SiPM,
        so the newest file in force says nothing about most of them); empty
        when no override resolves.
    """
    overrides = os.path.join(prod_path, "inputs/dataprod/overrides/hit")
    validity = os.path.join(overrides, "validity.yaml")
    if not os.path.exists(validity):
        return pd.DataFrame()
    # an override touches only the channels it lists, so a run's SiPMs are
    # calibrated by a spread of files: merge them in order (deeply -- a later
    # file may supply only part of one channel's pars) and remember which file
    # last defined each channel
    merged: dict = {}
    owner: dict = {}
    for path, pars in _spms_overrides_in_force(overrides, validity, start_key):
        for name, entry in pars.items():
            if not str(name).startswith("S") or not isinstance(entry, dict):
                continue
            owner[name] = path
            _deep_update(merged.setdefault(name, {}), entry)

    rows = {}
    for name, entry in merged.items():
        ops = entry.get("pars", {}).get("operations", {})
        rows[name] = {
            "pe_a": ops.get("energy_in_pe", {}).get("parameters", {}).get("a"),
            "pe_m": ops.get("energy_in_pe", {}).get("parameters", {}).get("m"),
            "threshold_a": ops.get("is_valid_hit", {}).get("parameters", {}).get("a"),
            "source": owner.get(name),
        }
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    frame.index.name = "name"
    return frame


def write_spms_production_keys(
    output_folder: str,
    period: str,
    run: str,
    prod_path: str,
    start_key: str | None = None,
    data_type: str = "phy",
) -> list:
    """
    Publish ``spms_noise/<run>`` and ``spms_calibration/<run>`` to the period contract.

    Parameters
    ----------
    output_folder : str
        Monitoring output root (the folder containing ``<period>/``).
    period, run : str
        Run to publish.
    prod_path : str
        Production version root (``generated/`` and ``inputs/`` live there).
    start_key : str, optional
        Run start key for the calibration lookup; skipped when None.
    data_type : str
        Data type key of the period contract file.

    Returns
    -------
    list
        Keys written.
    """
    path = period_contract_path(output_folder, period, data_type)
    written = []
    noise = read_spms_noise(prod_path, period, run, data_type)
    if not noise.empty:
        written.append(contract_writer.write_frame(path, f"spms_noise/{run}", noise))
    if start_key is not None:
        try:
            calib = read_spms_calibration(prod_path, start_key)
        except Exception as exc:  # noqa: BLE001 - auxiliary key, never fatal
            # the geds output of this task is already complete; a malformed
            # override tree must not fail the run (it did, with rc=1, on p16/p18)
            utils.logger.warning(
                "could not resolve the SiPM calibration for %s-%s: %s", period, run, exc
            )
            calib = pd.DataFrame()
        if not calib.empty:
            written.append(
                contract_writer.write_frame(path, f"spms_calibration/{run}", calib)
            )
    return written


#: (contract key, qcp metric) pairs checked by check_spms_thresholds
SPMS_THRESHOLD_KEYS = [
    key
    for key, info in utils.MTG_PLOT_INFO.items()
    if isinstance(info, dict)
    and str(info.get("title", "")).startswith("spms_")
    and not str(key).startswith("lar_")
]


#: period-file keys (``group`` or ``group/column``) graded by check_spms_thresholds
LAR_THRESHOLD_KEYS = [
    key
    for key, info in utils.MTG_PLOT_INFO.items()
    if isinstance(info, dict) and str(key).startswith("lar_")
]


def check_spms_thresholds(
    output_folder: str, period: str, run: str, data_type: str = "phy"
) -> dict:
    """
    Grade every SiPM (and the LAr veto) against the bands and record the verdicts in ``qcp_summary.yaml``.

    Reads the 60min bins of the run's spms contract file for each key listed
    in ``mtg-plot-settings.yaml`` with an ``spms_*`` title, and the
    ``lar_*`` period keys (optionally averaged over ``rolling`` bins first);
    verdicts land
    under ``<sipm>/phy/<metric>`` next to the geds ones, and the magnitudes
    behind a failing verdict are stashed for the issue records exactly as
    :func:`utils.check_threshold` does for geds.

    Parameters
    ----------
    output_folder : str
        Monitoring output root (the folder containing ``<period>/``).
    period, run : str
        Run to grade.
    data_type : str
        Data type of the contract file.

    Returns
    -------
    dict
        ``{sipm: {metric: verdict}}`` for the SiPMs graded (plus ``LAr`` for
        the run-level veto fractions); empty when nothing could be read.
    """
    run_dir = os.path.join(output_folder, period, run)
    contract = os.path.join(
        run_dir, f"l200-{period}-{run}-{data_type}-spms-schema2.hdf"
    )
    qcp_path = os.path.join(run_dir, f"l200-{period}-{run}-qcp_summary.yaml")
    output = utils.load_yaml_or_default(qcp_path, {})
    graded = {}
    if not os.path.isfile(contract):
        utils.logger.debug("no spms contract at %s; no SiPM thresholds", contract)
    for key in SPMS_THRESHOLD_KEYS if os.path.isfile(contract) else []:
        info = utils.MTG_PLOT_INFO[key]
        flag, _, param = key.partition("_")
        try:
            frame = contract_reader.read_binned_series(
                contract, flag, param, "60min"
            ).to_frame("mean")
        except KeyError:
            utils.logger.debug("...no %s in %s, skip", key, contract)
            continue
        t0 = [frame.index[0]]
        if info.get("rolling"):
            frame = frame.rolling(int(info["rolling"]), min_periods=1).mean()
        for sipm in frame.columns:
            series = frame[sipm].dropna()
            if series.empty:
                continue
            output.setdefault(sipm, {}).setdefault("phy", {})
            utils.check_threshold(
                series,
                sipm,
                None,
                t0,
                list(info["limits"]),
                info["title"],
                output,
                period=period,
                run=run,
            )
            graded.setdefault(sipm, {})[info["title"]] = output[sipm]["phy"][
                info["title"]
            ]
    # LAr veto keys live in the period file: per-SiPM occupancy and, under
    # the pseudo-detector "LAr", the run-level veto/accidental fractions
    period_file = period_contract_path(output_folder, period, data_type)
    for key in LAR_THRESHOLD_KEYS:
        info = utils.MTG_PLOT_INFO[key]
        group, _, column = key.partition("/")
        try:
            frame = contract_reader.read_frame(period_file, f"{group}/{run}")
        except (KeyError, OSError):
            continue
        if column:
            frame = frame[[column]].rename(columns={column: "LAr"})
        if info.get("rolling"):
            frame = frame.rolling(int(info["rolling"]), min_periods=1).mean()
        for name in frame.columns:
            series = frame[name].dropna()
            if series.empty:
                continue
            output.setdefault(name, {}).setdefault("phy", {})
            utils.check_threshold(
                series,
                name,
                None,
                [frame.index[0]],
                list(info["limits"]),
                info["title"],
                output,
                period=period,
                run=run,
            )
            graded.setdefault(name, {})[info["title"]] = output[name]["phy"][
                info["title"]
            ]
    if graded:
        with open(qcp_path, "w") as f:
            yaml.dump(output, f)
    return graded


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LAr veto performance from the evt tier
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

_LAR_EVT_FIELDS = [
    "trigger/timestamp",
    "trigger/is_forced",
    "coincident/spms",
    "coincident/puls",
    "coincident/muon",
    "geds/multiplicity",
    "spms/energy_sum",
    "spms/multiplicity",
    "spms/first_t0",
    "spms/geds_coincidence_classifier",
    "spms/is_trig_coin_pulse",
    "spms/rawid",
]


def read_lar_events(files: list) -> tuple:
    """
    Reduce the evt tier of a run to per-event LAr scalars and per-SiPM participation.

    Parameters
    ----------
    files : list
        evt-tier LH5 files of one run.

    Returns
    -------
    tuple
        ``(events, occupancy)``: a frame of per-event scalars (``datetime``
        index; ``is_forced``, ``is_phys`` (geds trigger, no pulser/muon),
        ``vetoed``, ``energy_sum``, ``multiplicity``, ``first_t0``,
        ``classifier``) and a boolean frame (same index x SiPM rawid) of
        whether each SiPM had a pulse coincident with the trigger. Both empty
        when nothing could be read.
    """
    from lgdo import lh5

    frames, parts = [], []
    for path in files:
        try:
            evt = lh5.read_as("evt/", path, library="ak", field_mask=_LAR_EVT_FIELDS)
        except (KeyError, ValueError, OSError) as exc:
            utils.logger.debug("skipping evt file %s: %s", os.path.basename(path), exc)
            continue
        if len(evt) == 0:
            continue
        index = pd.to_datetime(ak.to_numpy(evt.trigger.timestamp), unit="s", utc=True)
        forced = ak.to_numpy(evt.trigger.is_forced)
        frames.append(
            pd.DataFrame(
                {
                    "is_forced": forced,
                    "is_phys": ~forced
                    & ~ak.to_numpy(evt.coincident.puls)
                    & ~ak.to_numpy(evt.coincident.muon)
                    & (ak.to_numpy(evt.geds.multiplicity) > 0),
                    "vetoed": ak.to_numpy(evt.coincident.spms),
                    "energy_sum": ak.to_numpy(evt.spms.energy_sum).astype("float32"),
                    "multiplicity": ak.to_numpy(evt.spms.multiplicity).astype("int16"),
                    "first_t0": ak.to_numpy(evt.spms.first_t0).astype("float32"),
                    "classifier": ak.to_numpy(
                        evt.spms.geds_coincidence_classifier
                    ).astype("float32"),
                },
                index=pd.DatetimeIndex(index, name="datetime"),
            )
        )
        # the per-event rawid list is dense (every SiPM, every event), so a
        # plain 2-D array is the channel axis
        rawids = ak.to_numpy(evt.spms.rawid[0])
        has_pulse = ak.to_numpy(
            ak.fill_none(ak.any(evt.spms.is_trig_coin_pulse, axis=2), False)
        )
        parts.append(
            pd.DataFrame(has_pulse, index=frames[-1].index, columns=rawids.tolist())
        )
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(frames).sort_index(), pd.concat(parts).sort_index()


def lar_veto_series(events: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    """
    Hourly LAr veto performance from the per-event scalars of :func:`read_lar_events`.

    Columns: ``n_phys``, ``veto_frac`` (physics geds events flagged by the
    LAr veto), ``accidental_frac`` (forced triggers flagged: the veto's random
    coincidence probability, i.e. its dead time), ``energy_sum_median``,
    ``multiplicity_median`` and ``first_t0_frac`` over vetoed physics events,
    ``classifier_median`` over physics events.
    """
    if events.empty:
        return pd.DataFrame()
    phys = events[events["is_phys"]]
    vetoed = phys[phys["vetoed"]]
    forced = events[events["is_forced"]]
    grouped = phys.resample(freq)
    out = pd.DataFrame(
        {
            "n_phys": grouped.size(),
            "veto_frac": grouped["vetoed"].mean(),
            "accidental_frac": forced.resample(freq)["vetoed"].mean(),
            "energy_sum_median": vetoed.resample(freq)["energy_sum"].median(),
            "multiplicity_median": vetoed.resample(freq)["multiplicity"].median(),
            "first_t0_frac": vetoed.resample(freq)["first_t0"].apply(
                lambda s: float(np.isfinite(s).mean()) if len(s) else np.nan
            ),
            "classifier_median": grouped["classifier"].median(),
        }
    )
    out.index.name = "datetime"
    return out.astype("float32")


def write_lar_summary(
    output_folder: str,
    period: str,
    run: str,
    files: list,
    rawid_to_name: dict | None = None,
    data_type: str = "phy",
) -> list:
    """
    Publish the run's LAr veto performance to the period contract and the spms contract.

    Parameters
    ----------
    output_folder : str
        Monitoring output root (the folder containing ``<period>/``).
    period, run : str
        Run to summarise.
    files : list
        evt-tier LH5 files of the run.
    rawid_to_name : dict, optional
        SiPM rawid -> name for the occupancy columns (rawids kept otherwise).
    data_type : str
        Data type key of the contract files.

    Returns
    -------
    list
        Keys written: ``lar_veto/<run>`` and ``lar_occupancy/<run>`` in the
        period file, ``hist/IsPhysics_Lar{EnergySum,Multiplicity,Classifier}_dist``
        in the run's spms contract (when it exists).
    """
    from .processing import binning

    events, participation = read_lar_events(files)
    if events.empty:
        utils.logger.warning("no evt data for %s/%s; no LAr summary", period, run)
        return []
    written = []
    path = period_contract_path(output_folder, period, data_type)
    written.append(
        contract_writer.write_frame(path, f"lar_veto/{run}", lar_veto_series(events))
    )
    phys = participation[events["is_phys"].to_numpy()]
    occupancy = phys.resample("1h").mean().astype("float32")
    occupancy.columns = [
        (rawid_to_name or {}).get(c, str(c)) for c in occupancy.columns
    ]
    occupancy.index.name = "datetime"
    written.append(contract_writer.write_frame(path, f"lar_occupancy/{run}", occupancy))

    spms_contract = os.path.join(
        output_folder, period, run, f"l200-{period}-{run}-{data_type}-spms-schema2.hdf"
    )
    if os.path.isfile(spms_contract):
        vetoed = events[events["is_phys"] & events["vetoed"]]
        for column, param, unit in [
            ("energy_sum", "LarEnergySum", "p.e."),
            ("multiplicity", "LarMultiplicity", "SiPMs"),
            ("classifier", "LarClassifier", "a.u."),
        ]:
            values = vetoed[column].to_numpy(dtype=float)
            if not len(values):
                continue
            written.append(
                contract_writer.write_distribution(
                    spms_contract,
                    "IsPhysics",
                    param,
                    binning.fill_distribution(values),
                    {"unit": unit, "label": param, "event_type": "IsPhysics"},
                )
            )
        _refresh_run_manifest(output_folder, period, run, data_type)
    return written


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SiPM single-photoelectron spectra (PE calibration validation)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#: per-pulse energy_in_pe histogram: 0-5 p.e. at 0.02 p.e./bin, fine enough to
#: locate the 1 p.e. centroid (a valid calibration puts it at exactly 1.0)
SPE_BINS = 250
SPE_RANGE = (0.0, 5.0)


def _refresh_run_manifest(
    output_folder: str, period: str, run: str, data_type: str
) -> None:
    """Re-inventory the run manifest after adding keys to a contract file."""
    from .contract import build as contract_build

    generated_path = output_folder.split("/generated/")[0]
    try:
        contract_build.refresh_manifest(
            generated_path, period, run, data_type=data_type
        )
    except (OSError, KeyError) as exc:  # never lose the keys over the inventory
        utils.logger.warning("could not refresh the manifest for %s: %s", run, exc)


def read_spe_spectra(
    hit_files: list, evt_files: list, rawid_to_name: dict | None = None
) -> dict:
    """
    Histogram every SiPM pulse energy, split by trigger type.

    ``energy_in_pe = pe_a + pe_m * energy``, so a correct calibration puts the
    1/2/3 p.e. peaks at exactly 1.0/2.0/3.0 and the 1 p.e. centroid offset is
    the gain drift in p.e. units. Pulses are taken **unmasked** (without
    ``is_valid_hit``): the threshold sits below 1 p.e., so masking clips the
    low side of the very peak being measured. Forced triggers (the random
    sample) give the cleanest single-photoelectron spectrum; physics events
    are kept separately.

    Parameters
    ----------
    hit_files : list
        hit-tier LH5 files of the run.
    evt_files : list
        evt-tier files of the same run, for the forced-trigger flag; the
        tiers are row-aligned per file.
    rawid_to_name : dict, optional
        SiPM rawid -> name for the category axis (rawids kept otherwise).

    Returns
    -------
    dict
        flag (``IsBsln``/``IsPhysics``) -> boost_histogram.Histogram
        (Regular(energy) x StrCategory(SiPM)).
    """
    import awkward as ak
    from lgdo import lh5

    from .processing import binning

    evt_by_key = {os.path.basename(f).split("-")[4]: f for f in evt_files}
    hists = {
        flag: binning.empty_distribution_2d(SPE_BINS, SPE_RANGE)
        for flag in ("IsBsln", "IsPhysics")
    }
    channels = None
    for path in hit_files:
        key = os.path.basename(path).split("-")[4]
        evt_path = evt_by_key.get(key)
        if evt_path is None:
            utils.logger.debug("no evt file for %s; skipping SPE fill", key)
            continue
        if channels is None:
            names = set(rawid_to_name or {})
            channels = [
                c
                for c in lh5.ls(path)
                if c.startswith("ch")
                and (not names or int(c[2:]) in (rawid_to_name or {}))
            ]
        forced = ak.to_numpy(
            lh5.read_as(
                "evt/", evt_path, library="ak", field_mask=["trigger/is_forced"]
            ).trigger.is_forced
        )
        for channel in channels:
            try:
                pulses = lh5.read_as(
                    f"{channel}/hit/",
                    [path],
                    library="ak",
                    field_mask=["energy_in_pe"],
                ).energy_in_pe
            except (KeyError, ValueError, TypeError):
                continue
            if len(pulses) != len(forced):
                utils.logger.warning(
                    "%s: %d hit rows vs %d evt rows in %s; skipping SPE fill",
                    channel,
                    len(pulses),
                    len(forced),
                    key,
                )
                continue
            name = (rawid_to_name or {}).get(int(channel[2:]), channel)
            counts = ak.to_numpy(ak.num(pulses))
            values = ak.to_numpy(ak.flatten(pulses))
            per_pulse = np.repeat(forced, counts)
            finite = np.isfinite(values)
            for flag, mask in (("IsBsln", per_pulse), ("IsPhysics", ~per_pulse)):
                selected = values[finite & mask]
                if len(selected):
                    hists[flag].fill(selected, name)
    return hists


def write_spe_spectrum(
    output_folder: str,
    period: str,
    run: str,
    hit_files: list,
    evt_files: list,
    rawid_to_name: dict | None = None,
    data_type: str = "phy",
) -> list:
    """
    Publish the per-SiPM single-photoelectron spectra to the run's spms contract.

    Parameters
    ----------
    output_folder : str
        Monitoring output root (the folder containing ``<period>/``).
    period, run : str
        Run to summarise.
    hit_files, evt_files : list
        hit- and evt-tier LH5 files of the run.
    rawid_to_name : dict, optional
        SiPM rawid -> name for the category axis.
    data_type : str
        Data type key of the contract file.

    Returns
    -------
    list
        Keys written (``hist/<flag>_EnergyInPe_dist2d``); empty when the run
        has no spms contract file or no pulses.
    """
    contract = os.path.join(
        output_folder, period, run, f"l200-{period}-{run}-{data_type}-spms-schema2.hdf"
    )
    if not os.path.isfile(contract):
        utils.logger.warning("no spms contract at %s; no SPE spectra", contract)
        return []
    written = []
    for flag, hist in read_spe_spectra(hit_files, evt_files, rawid_to_name).items():
        if not hist.sum():
            continue
        written.append(
            contract_writer.write_distribution_2d(
                contract,
                flag,
                "EnergyInPe",
                hist,
                {
                    "unit": "p.e.",
                    "label": "Pulse energy",
                    "event_type": flag,
                    "selection": "all pulses (is_valid_hit not applied)",
                },
            )
        )
    if written:
        _refresh_run_manifest(output_folder, period, run, data_type)
    return written


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Muon veto (pmts) from the dsp tier: this production has no evt_muon group,
# and the muon DAQ triggers independently of the geds stream (its timestamps
# match nothing else), so the per-trigger quantities are derived from the
# per-PMT dsp rows themselves and the ge-coincidence from evt/coincident.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#: per-PMT pulse-height spectrum: fixed range so the flow bins absorb outliers
PMT_SPEC_BINS = 500
PMT_SPEC_RANGE = (0.0, 100.0)


def read_pmt_events(dsp_files: list, rawid_to_name: dict) -> tuple:
    """
    Collect the muon-DAQ triggers and fill the per-PMT pulse-height spectra.

    One dsp row per PMT per muon-DAQ trigger (~5/min): small enough to read
    whole. The spectrum takes every pulse height unmasked (no ``containsPulse``
    cut) so the single-photoelectron region is not clipped; the per-trigger
    ``light_sum`` counts only PMTs that actually saw a pulse.

    Parameters
    ----------
    dsp_files : list
        dsp-tier LH5 files of the run.
    rawid_to_name : dict
        PMT rawid -> name, from :func:`utils.build_pmts_info`.

    Returns
    -------
    tuple
        ``(events, spectrum)``: per-trigger frame (datetime index;
        ``multiplicity`` = PMTs with a pulse, ``light_sum`` = summed pulse
        height in LSB) and the Regular(pulse height) x StrCategory(PMT)
        histogram of every pulse height.
    """
    from lgdo import lh5
    from lh5.io.exceptions import LH5DecodeError

    from .processing import binning

    spectrum = binning.empty_distribution_2d(PMT_SPEC_BINS, PMT_SPEC_RANGE)
    per_channel = []
    for rawid, name in sorted(rawid_to_name.items()):
        frames = []
        for path in dsp_files:
            try:
                frames.append(
                    lh5.read_as(
                        f"ch{rawid}/dsp/",
                        [path],
                        library="pd",
                        field_mask=["timestamp", "pulseHeight", "containsPulse"],
                    )
                )
            except (KeyError, ValueError, TypeError, LH5DecodeError):
                # a PMT can be absent from a file (e.g. switched off)
                continue
        if not frames:
            continue
        frame = pd.concat(frames, ignore_index=True)
        heights = frame["pulseHeight"].to_numpy()
        has_pulse = frame["containsPulse"].astype(bool).to_numpy()
        finite = np.isfinite(heights)
        if finite.any():
            spectrum.fill(heights[finite], name)
        per_channel.append(
            pd.DataFrame(
                {
                    "timestamp": frame["timestamp"],
                    "has_pulse": has_pulse,
                    # only PMTs that saw a pulse contribute to the summed light
                    "height": np.where(finite & has_pulse, heights, 0.0),
                }
            )
        )
    if not per_channel:
        return pd.DataFrame(), spectrum
    stacked = pd.concat(per_channel, ignore_index=True)
    grouped = stacked.groupby("timestamp")
    events = pd.DataFrame(
        {
            "multiplicity": grouped["has_pulse"].sum(),
            "light_sum": grouped["height"].sum(),
        }
    )
    events.index = pd.to_datetime(events.index, unit="s", utc=True)
    events.index.name = "datetime"
    return events.sort_index(), spectrum


def muon_veto_series(
    events: pd.DataFrame, coincidence: pd.DataFrame, freq: str = "1h"
) -> pd.DataFrame:
    """
    Hourly muon-veto performance.

    Parameters
    ----------
    events : pandas.DataFrame
        Per-trigger frame from :func:`read_pmt_events`.
    coincidence : pandas.DataFrame
        Per-geds-event booleans ``muon``/``muon_offline`` (datetime index)
        from the evt tier's ``coincident`` group.
    freq : str
        Resampling cadence.

    Returns
    -------
    pandas.DataFrame
        ``muon_rate_hz`` (muon-DAQ triggers), ``multiplicity_median`` and
        ``light_sum_median`` per trigger, and the fraction of geds events in
        coincidence with a muon (``ge_coincidence_frac``, ``_offline``).
    """
    if events.empty:
        return pd.DataFrame()
    seconds = pd.Timedelta(freq).total_seconds()
    grouped = events.resample(freq)
    out = pd.DataFrame(
        {
            "muon_rate_hz": grouped.size() / seconds,
            "multiplicity_median": grouped["multiplicity"].median(),
            "light_sum_median": grouped["light_sum"].median(),
        }
    )
    if coincidence is not None and not coincidence.empty:
        resampled = coincidence.resample(freq)
        out["ge_coincidence_frac"] = resampled["muon"].mean()
        if "muon_offline" in coincidence.columns:
            out["ge_coincidence_frac_offline"] = resampled["muon_offline"].mean()
    out.index.name = "datetime"
    return out.astype("float32")


def write_muon_summary(
    output_folder: str,
    period: str,
    run: str,
    dsp_files: list,
    evt_files: list,
    rawid_to_name: dict,
    data_type: str = "phy",
) -> list:
    """
    Publish the run's muon-veto performance to the period and pmts contracts.

    Parameters
    ----------
    output_folder : str
        Monitoring output root (the folder containing ``<period>/``).
    period, run : str
        Run to summarise.
    dsp_files, evt_files : list
        dsp- and evt-tier LH5 files of the run.
    rawid_to_name : dict
        PMT rawid -> name, from :func:`utils.build_pmts_info`.
    data_type : str
        Data type key of the contract files.

    Returns
    -------
    list
        Keys written: ``muon_veto/<run>`` in the period file and, in the
        run's pmts contract, ``hist/All_Pulseheight_dist2d`` (the SPP
        spectrum per PMT) plus ``_dist`` keys for the per-trigger
        multiplicity and summed light.
    """
    from lgdo import lh5

    from .processing import binning

    events, spectrum = read_pmt_events(dsp_files, rawid_to_name)
    if events.empty:
        utils.logger.warning(
            "...no pmts dsp rows for %s/%s; no muon summary", period, run
        )
        return []

    coincidence = pd.DataFrame()
    if evt_files:
        frames = []
        for path in evt_files:
            try:
                frame = lh5.read_as(
                    "evt/",
                    path,
                    library="pd",
                    field_mask=[
                        "trigger/timestamp",
                        "coincident/muon",
                        "coincident/muon_offline",
                    ],
                )
            except (KeyError, ValueError, TypeError):
                continue
            frames.append(
                frame.rename(
                    columns={
                        "coincident_muon": "muon",
                        "coincident_muon_offline": "muon_offline",
                    }
                )
            )
        if frames:
            coincidence = pd.concat(frames, ignore_index=True)
            coincidence = coincidence.set_index(
                pd.to_datetime(coincidence.pop("trigger_timestamp"), unit="s", utc=True)
            )

    written = []
    path = period_contract_path(output_folder, period, data_type)
    written.append(
        contract_writer.write_frame(
            path, f"muon_veto/{run}", muon_veto_series(events, coincidence)
        )
    )

    contract = os.path.join(
        output_folder, period, run, f"l200-{period}-{run}-{data_type}-pmts-schema2.hdf"
    )
    if os.path.isfile(contract):
        if spectrum.sum():
            written.append(
                contract_writer.write_distribution_2d(
                    contract,
                    "All",
                    "Pulseheight",
                    spectrum,
                    {
                        "unit": "LSB",
                        "label": "Pulse height",
                        "event_type": "All",
                        "selection": "all triggers (containsPulse not applied)",
                    },
                )
            )
        for name, series in (
            ("MuonMultiplicity", events["multiplicity"]),
            ("MuonLightSum", events["light_sum"]),
        ):
            values = series.to_numpy(dtype=float)
            if len(values):
                written.append(
                    contract_writer.write_distribution(
                        contract,
                        "All",
                        name,
                        binning.fill_distribution(values),
                        {"event_type": "All", "label": name},
                    )
                )
        _refresh_run_manifest(output_folder, period, run, data_type)
    else:
        utils.logger.debug("no pmts contract at %s; period key only", contract)
    return written
