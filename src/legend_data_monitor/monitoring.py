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
        df_energy_IsPhysics = store["/IsPhysics_TrapemaxCtcCal"]
        df_energy_IsPhysics = filter_series_by_ignore_keys(
            df_energy_IsPhysics, utils.IGNORE_KEYS, period
        )

        for par in pars_to_inspect:
            mask = df_energy_IsPhysics > 25
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
