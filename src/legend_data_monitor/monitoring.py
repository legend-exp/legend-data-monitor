import json
import os
import pickle
import shelve
import sys

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import yaml
from legendmeta import LegendMetadata
from lgdo import lh5
from tqdm.notebook import tqdm

from . import utils

# -------------------------------------------------------------------------

IPython_default = plt.rcParams.copy()
SMALL_SIZE = 8

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


def get_energy_key(
    ecal_results: dict,
) -> dict:
    """
    Retrieve the energy calibration results from a given dictionary.

    This function searches for specific keys ('cuspEmax_ctc_runcal' or 'cuspEmax_ctc_cal') in the input `ecal_results` dictionary.
    It returns a sub-dictionary if one of the keys is found, otherwise an empty dictionary is returned.

    Parameters
    ----------
    ecal_results : dict
        Dictionary containing energy calibration results.
    """
    cut_dict = {}
    for key in ["cuspEmax_ctc_runcal", "cuspEmax_ctc_cal"]:
        if key in ecal_results:
            cut_dict = ecal_results[key]
            break
    else:
        utils.logger.debug("No cuspEmax key")
        return cut_dict

    return cut_dict


def get_calibration_file(folder_par: str) -> dict:
    """
    Return the content of the JSON/YAML calibration file in folder_par.

    Parameters
    ----------
    folder_par : str
        Path to the folder containing calibration summary files.
    """
    files = os.listdir(folder_par)
    json_files = [f for f in files if f.endswith(".json")]
    yaml_files = [f for f in files if f.endswith((".yaml", ".yml"))]

    if json_files:
        filepath = os.path.join(folder_par, json_files[0])
        with open(filepath) as f:
            pars_dict = json.load(f)
    elif yaml_files:
        filepath = os.path.join(folder_par, yaml_files[0])
        with open(filepath) as f:
            pars_dict = yaml.load(f, Loader=yaml.CLoader)
    else:
        raise FileNotFoundError(f"No JSON or YAML file found in {folder_par}")

    return pars_dict


def extract_fep_peak(pars_dict: dict, channel: str):
    """
    Return fep_peak_pos, fep_peak_pos_err, fep_gain, fep_gain_err.

    Parameters
    ----------
    pars_dict : dict
        Dictionary containing calibration outputs.
    channel : str
        Channel name or IDs.
    """
    if channel not in pars_dict:
        return np.nan, np.nan, np.nan, np.nan

    # for FEP peak, we want to look at the behaviour over time; take 'ecal' results (not partition ones!)
    ecal_results = pars_dict[channel]["results"]["ecal"]
    pk_fits = get_energy_key(ecal_results).get("pk_fits", {})

    try:
        fep_energy = [p for p in sorted(pk_fits) if 2613 < float(p) < 2616][0]
        try:
            fep_peak_pos = pk_fits[fep_energy]["parameters_in_ADC"]["mu"]
            fep_peak_pos_err = pk_fits[fep_energy]["uncertainties_in_ADC"]["mu"]
        except (KeyError, TypeError):
            fep_peak_pos = pk_fits[fep_energy]["parameters"]["mu"]
            fep_peak_pos_err = pk_fits[fep_energy]["uncertainties"]["mu"]

        fep_gain = fep_peak_pos / 2614.5
        fep_gain_err = fep_peak_pos_err / 2614.5

    except (KeyError, TypeError, IndexError):
        return np.nan, np.nan, np.nan, np.nan

    return fep_peak_pos, fep_peak_pos_err, fep_gain, fep_gain_err


def extract_resolution_at_q_bb(
    pars_dict: dict, channel: str, key_result: str, fit: str = "linear"
):
    """
    Return Qbb_fwhm (linear resolution) and Qbb_fwhm_quad (quadratic resolution).

    Parameters
    ----------
    pars_dict : dict
        Dictionary containing calibration outputs.
    channel : str
        Channel name or IDs.
    key_result : str
        Key name used to extract the resolution results from the parsed file.
    fit : str
        Fitting method used for energy resolution, either 'linear' or 'quadratic'.
    """
    if channel not in pars_dict:
        return np.nan, np.nan

    result = pars_dict[channel]["results"][key_result].get("cuspEmax_ctc_cal", {})
    Qbb_keys = [k for k in result.get("eres_linear", {}) if "Qbb_fwhm_in_" in k]
    if not Qbb_keys:
        return np.nan, np.nan

    Qbb_fwhm = result["eres_linear"][Qbb_keys[0]]
    Qbb_fwhm_quad = result["eres_quadratic"][Qbb_keys[0]] if fit != "linear" else np.nan

    return Qbb_fwhm, Qbb_fwhm_quad


def evaluate_fep_cal(
    pars_dict: dict, channel: str, fep_peak_pos: float, fep_peak_pos_err: float
):
    """
    Return calibrated FEP position (fep_cal) and error (fep_cal_err).

    Parameters
    ----------
    pars_dict : dict
        Dictionary containing calibration outputs.
    channel : str
        Channel name or IDs.
    fep_peak_pos : float
        Uncalibrated FEP position.
    fep_peak_pos_err : float
        Uncalibrated FEP position error.
    """
    if channel not in pars_dict:
        return np.nan, np.nan

    ecal_results = get_energy_key(pars_dict[channel]["pars"]["operations"])
    expr = ecal_results["expression"]
    params = ecal_results["parameters"]

    fep_cal = eval(expr, {}, {**params, "cuspEmax_ctc": fep_peak_pos})
    fep_cal_err = eval(expr, {}, {**params, "cuspEmax_ctc": fep_peak_pos_err})

    return fep_cal, fep_cal_err


def get_run_start_end_times(
    sto,
    tiers: list,
    period: str,
    run: str,
    tier: str,
):
    """
    Determine the start and end timestamps for a given run, including the special case for additional final calibration runs.

    Parameters
    ----------
    sto
        Store object to read timestamps from LH5 files.
    tiers : list of str
        Paths to tier data folders based on the inspected processed version.
    period : str
        Period to inspect.
    run : str
        Run to inspect.
    tier : str
        Tier level for the analysis ('hit', 'phy', etc.).
    """
    folder_tier = os.path.join(tiers[0 if tier == "hit" else 1], "cal", period, run)
    dir_path = os.path.join(tiers[-1], "phy", period)

    # for when we have a calib run but zero phy runs for a given period
    if os.path.isdir(dir_path) and run not in os.listdir(dir_path):
        run_files = sorted(os.listdir(folder_tier))
        run_end_time = pd.to_datetime(
            sto.read(
                "ch1027201/dsp/timestamp", os.path.join(folder_tier, run_files[-1])
            )[-1],
            unit="s",
        )
        run_start_time = run_end_time
    else:
        run_files = sorted(os.listdir(folder_tier))
        run_start_time = pd.to_datetime(
            sto.read(
                "ch1027201/dsp/timestamp", os.path.join(folder_tier, run_files[0])
            )[0],
            unit="s",
        )
        run_end_time = pd.to_datetime(
            sto.read(
                "ch1027201/dsp/timestamp", os.path.join(folder_tier, run_files[-1])
            )[-1],
            unit="s",
        )

    return run_start_time, run_end_time


def get_calib_data_dict(
    calib_data: dict,
    channel_info: list,
    tiers: list,
    pars: list,
    period: str,
    run: str,
    tier: str,
    key_result: str,
    fit: str,
):
    """
    Extract calibration information for a given run and appends it to the provided dictionary.

    This function loads calibration parameters for a specific detector channel and run,
    parses energy calibration results and resolution information, and evaluates
    derived values such as gain and calibration constants. It appends the extracted data
    to the provided `calib_data` dictionary, which is expected to contain keys like
    "fep", "fep_err", "cal_const", "cal_const_err", "run_start", "run_end", "res", and "res_quad".

    Parameters
    ----------
    calib_data : dict
        Dictionary that accumulates calibration results across runs.
    channel_info : list
        List of [channel ID, channel name].
    tiers : list of str
        Paths to tier data folders based on the inspected processed version.
    pars : list of str
        Paths to parameter .yaml/.json files.
    period : str
        Period to inspect.
    run : str
        Run to inspect.
    tier : str
        Tier level for the analysis ('hit', 'phy', etc.).
    key_result : str
        Key name used to extract the resolution results from the parsed file.
    fit : str
        Fitting method used for energy resolution, either 'linear' or 'quadratic'.
    """
    sto = lh5.LH5Store()
    channel = channel_info[0]
    channel_name = channel_info[1]

    folder_par = os.path.join(pars[2 if tier == "hit" else 3], "cal", period, run)
    pars_dict = get_calibration_file(folder_par)

    if not all(k.startswith("ch") for k in pars_dict.keys()):
        channel = channel_name

    # retrieve calibration parameters
    fep_peak_pos, fep_peak_pos_err, fep_gain, fep_gain_err = extract_fep_peak(
        pars_dict, channel
    )
    Qbb_fwhm, Qbb_fwhm_quad = extract_resolution_at_q_bb(
        pars_dict, channel, key_result, fit
    )
    fep_cal, fep_cal_err = evaluate_fep_cal(
        pars_dict, channel, fep_peak_pos, fep_peak_pos_err
    )

    # get timestamp for additional-final cal run (only for FEP gain display)
    run_start_time, run_end_time = get_run_start_end_times(
        sto, tiers, period, run, tier
    )

    calib_data["fep"].append(fep_gain)
    calib_data["fep_err"].append(fep_gain_err)
    calib_data["cal_const"].append(fep_cal)
    calib_data["cal_const_err"].append(fep_cal_err)
    calib_data["run_start"].append(run_start_time)
    calib_data["run_end"].append(run_end_time)
    calib_data["res"].append(Qbb_fwhm)
    calib_data["res_quad"].append(Qbb_fwhm_quad)

    return calib_data


def add_calibration_runs(period: str | list, run_list: list | dict) -> list:
    """
    Add special calibration runs to the run list for a given period.

    Parameters
    ----------
        period : str | list
            Either a string or list of periods
        run_list : list | dict
            Either a list of runs or a dictionary with period keys
    """
    if isinstance(period, list) and isinstance(run_list, dict):
        # multiple periods
        for p in period:
            if p in CALIB_RUNS and p in run_list:
                run_list[p] = run_list[p] + CALIB_RUNS[p]
    else:
        # single period case
        if period in CALIB_RUNS:
            if isinstance(run_list, list):
                run_list.extend(CALIB_RUNS[period])
            else:
                # run_list might be a dict but period is a string
                if period in run_list:
                    run_list[period] = run_list[period] + CALIB_RUNS[period]

    return run_list


def get_tier_keyresult(tiers: list):
    """
    Retrieve proper tier name (pht or hit) and key_result (partition_ecal or ecal) depending if partitioning data exists or not.

    Parameters
    ----------
    tiers : list
        Base directory containing the tier and parameter folders.
    """
    tier = "hit"
    key_result = "ecal"
    if os.path.isdir(tiers[1]):
        if os.listdir(tiers[1]) != []:
            tier = "pht"
            key_result = "partition_ecal"

    return tier, key_result


def compute_diff(
    values: np.ndarray, initial_value: float | int, scale: float | int
) -> np.ndarray:
    """
    Compute relative differences with respect to an initial value. If the initial value is zero, returns an array of nan values.

    Parameters
    ----------
    values : np.ndarray
        Array of values to compute the differences for.
    initial_value : float
        Reference value for computing relative differences.
    scale : float
        Scaling factor.
    """
    if initial_value == 0:
        return np.full_like(values, np.nan, dtype=float)

    return (values - initial_value) / initial_value * scale


def get_calib_pars(
    path: str,
    period: str | list,
    run_list: list,
    channel_info: list,
    partition: bool,
    escale: float,
    fit="linear",
) -> dict:
    """
    Retrieve and process calibration parameters across a list of runs for a given channel.

    This function loads calibration data from JSON/YAML files for each specified run, computes gain and calibration constant evolution over time, and returns a dictionary of relevant quantities, including their relative changes with respect to the initial values.
    It optionally appends special calibration runs at the end of a period, if available.

    Parameters
    ----------
    path : str
        Base directory containing the tier and parameter folders.
    period : str or list
        Period to inspect. Can be a list if multiple periods are inspected.
    run_list : list
        List of run to inspect, or a dictionary mapping periods to lists of runs.
    channel_info : list
        List containing [channel ID, channel name].
    partition : bool
        True if you want to retrieve partition calibration results.
    escale : float
        Scaling factor used to compute relative differences in gain and calibration constant.
    fit : str, optional
        Fit method used for energy resolution ("linear" or "quadratic"), by default "linear".
    """
    # add special calib runs at the end of a period
    run_list = add_calibration_runs(period, run_list)

    calib_data = {
        "fep": [],
        "fep_err": [],
        "cal_const": [],
        "cal_const_err": [],
        "run_start": [],
        "run_end": [],
        "res": [],
        "res_quad": [],
    }

    tiers, pars = utils.get_tiers_pars_folders(path)

    tier, key_result = get_tier_keyresult(tiers)

    for run in run_list:
        calib_data = get_calib_data_dict(
            calib_data, channel_info, tiers, pars, period, run, tier, key_result, fit
        )

    for key, item in calib_data.items():
        calib_data[key] = np.array(item)

    init_cal_const, init_fep = 0, 0
    for cal_, fep_ in zip(calib_data["cal_const"], calib_data["fep"]):
        if init_fep == 0 and fep_ != 0:
            init_fep = fep_
        if init_cal_const == 0 and cal_ != 0:
            init_cal_const = cal_

    calib_data["cal_const_diff"] = compute_diff(
        calib_data["cal_const"], init_cal_const, escale
    )
    calib_data["fep_diff"] = compute_diff(calib_data["fep"], init_fep, escale)

    return calib_data


def find_hdf_file(
    directory: str, include: list[str], exclude: list[str] = None
) -> str | None:
    """
    Find the original HDF monitoring file in a given directory, matching inclusion/exclusion filters.

    Parameters
    ----------
    directory : str
        Path to the folder containing the HDF monitoring files.
    include: list[str]
        List of words that the HDF monitoring file to retrieve must contain.
    exclude: list[str] = None
        List of words that the HDF monitoring file to retrieve must NOT contain.
    """
    exclude = exclude or []
    files = os.listdir(directory)
    candidates = [
        f
        for f in files
        if f.endswith(".hdf")
        and all(tag in f for tag in include)
        and not any(tag in f for tag in exclude)
    ]

    return os.path.join(directory, candidates[0]) if candidates else None


def read_if_key_exists(hdf_path: str, key: str) -> pd.DataFrame | None:
    """
    Read an HDF dataset if the key exists, otherwise return None; handle the case where the parameter is saved under either '/key' or 'key'.

    Parameters
    ----------
    hdf_path : str
        Path to the HDF file.
    key : str
        Key to inspect.
    """
    with pd.HDFStore(hdf_path, mode="r") as f:
        try:
            return f[key]
        except KeyError:
            try:
                return f["/" + key]
            except KeyError:
                return None


def get_dfs(phy_mtg_data: str, period: str, run_list: list, parameter: str):
    """
    Load and concatenate monitoring data from HDF files for a given period and list of runs.

    Parameters
    ----------
    phy_mtg_data : str
        Path to the base directory containing monitoring HDF5 files (typically ending in `/mtg/phy`).
    period : str
        Period to inspect.
    run_list : list
        List of available runs.
    parameter : str
        Parameter name used to construct the HDF key for loading specific datasets (e.g., 'TrapemaxCtcCal' looks for 'IsPulser_TrapemaxCtcCal').
    """
    # lists to accumulate dataframes, concatenated at the endo only
    geds_df_cuspEmax_abs = []
    geds_df_cuspEmax_abs_corr = []
    puls_df_cuspEmax_abs = []

    base_dir = os.path.join(phy_mtg_data, period)
    runs = os.listdir(base_dir)

    for r in runs:
        if r not in run_list:
            continue
        run_dir = os.path.join(base_dir, r)

        # geds file
        hdf_geds = find_hdf_file(run_dir, include=["geds"], exclude=["res", "min"])
        if not hdf_geds:
            utils.logger.debug("hdf_geds is empty")
            return None, None, None

        geds_abs = read_if_key_exists(hdf_geds, f"IsPulser_{parameter}")
        if geds_abs is not None:
            geds_df_cuspEmax_abs.append(geds_abs)

        geds_puls_abs = read_if_key_exists(
            hdf_geds, f"IsPulser_{parameter}_pulser01anaDiff"
        )
        if geds_puls_abs is not None:
            geds_df_cuspEmax_abs_corr.append(geds_puls_abs)

        # pulser file
        hdf_puls = find_hdf_file(
            run_dir, include=["pulser01ana"], exclude=["res", "min"]
        )
        if not hdf_puls:
            utils.logger.debug("hdf_puls is empty")
            # there's no need to return None, as the code will automatically handle the case of missing pulser file later on
        else:
            puls_abs = read_if_key_exists(hdf_puls, f"IsPulser_{parameter}")
            if puls_abs is not None:
                puls_df_cuspEmax_abs.append(puls_abs)

    if (
        not geds_df_cuspEmax_abs
        and not geds_df_cuspEmax_abs_corr
        and not puls_df_cuspEmax_abs
    ):
        return None, None, None
    else:
        return (
            (
                pd.concat(geds_df_cuspEmax_abs, ignore_index=False, axis=0)
                if geds_df_cuspEmax_abs
                else pd.DataFrame()
            ),
            (
                pd.concat(geds_df_cuspEmax_abs_corr, ignore_index=False, axis=0)
                if geds_df_cuspEmax_abs_corr
                else pd.DataFrame()
            ),
            (
                pd.concat(puls_df_cuspEmax_abs, ignore_index=False, axis=0)
                if puls_df_cuspEmax_abs
                else pd.DataFrame()
            ),
        )


def get_traptmax_tp0est(phy_mtg_data: str, period: str, run_list: list):
    """
    Load and concatenate trapTmax and tp0est data from HDF files for a given period and list of runs.

    Parameters
    ----------
    phy_mtg_data : str
        Path to the base directory containing monitoring HDF5 files (typically ending in `/mtg/phy`).
    period : str
        Period to inspect.
    run_list : list
        List of available runs.
    """
    geds_df_trapTmax, geds_df_tp0est = [], []
    puls_df_trapTmax, puls_df_tp0est = [], []

    base_dir = os.path.join(phy_mtg_data, period)
    for r in os.listdir(base_dir):
        if r not in run_list:
            continue
        run_dir = os.path.join(base_dir, r)

        # geds
        hdf_geds = find_hdf_file(run_dir, include=["geds"], exclude=["res", "min"])
        if hdf_geds:
            trapTmax = read_if_key_exists(hdf_geds, "IsPulser_TrapTmax")
            if trapTmax is not None:
                geds_df_trapTmax.append(trapTmax)

            tp0est = read_if_key_exists(hdf_geds, "IsPulser_Tp0Est")
            if tp0est is not None:
                geds_df_tp0est.append(tp0est)
        else:
            utils.logger.debug("hdf_geds is empty")

        # pulser
        hdf_puls = find_hdf_file(
            run_dir, include=["pulser01ana"], exclude=["res", "min"]
        )
        if hdf_puls:
            trapTmax = read_if_key_exists(hdf_puls, "IsPulser_TrapTmax")
            if trapTmax is not None:
                puls_df_trapTmax.append(trapTmax)

            tp0est = read_if_key_exists(hdf_puls, "IsPulser_Tp0Est")
            if tp0est is not None:
                puls_df_tp0est.append(tp0est)
        else:
            utils.logger.debug("hdf_puls is empty")

    return (
        (
            pd.concat(geds_df_trapTmax, ignore_index=False)
            if geds_df_trapTmax
            else pd.DataFrame()
        ),
        (
            pd.concat(geds_df_tp0est, ignore_index=False)
            if geds_df_tp0est
            else pd.DataFrame()
        ),
        (
            pd.concat(puls_df_trapTmax, ignore_index=False)
            if puls_df_trapTmax
            else pd.DataFrame()
        ),
        (
            pd.concat(puls_df_tp0est, ignore_index=False)
            if puls_df_tp0est
            else pd.DataFrame()
        ),
    )


def filter_series_by_ignore_keys(
    series_to_filter: pd.Series, skip_keys: dict, period: str
):
    """
    Remove data from a time-indexed pandas Series that falls within time ranges specified by start and stop timestamps for a given period.

    Parameters
    ----------
    series_to_filter : pd.Series
        The time-indexed pandas Series to be filtered.
    skip_keys : dict
        Dictionary mapping periods to sub-dictionaries containing 'start_keys' and 'stop_keys' lists with timestamp strings in the format '%Y%m%dT%H%M%S%z'.
    period : str
        The period to check for keys to ignore. If not present, the series is returned unmodified.
    """
    if period not in skip_keys:
        return series_to_filter

    start_keys = skip_keys[period]["start_keys"]
    stop_keys = skip_keys[period]["stop_keys"]

    for ki, kf in zip(start_keys, stop_keys):
        isolated_ki = pd.to_datetime(ki.replace("Z", "+0000"), format="%Y%m%dT%H%M%S%z")
        isolated_kf = pd.to_datetime(kf.replace("Z", "+0000"), format="%Y%m%dT%H%M%S%z")
        series_to_filter = series_to_filter[
            (series_to_filter.index < isolated_ki)
            | (series_to_filter.index > isolated_kf)
        ]

    return series_to_filter


def filter_by_period(series: pd.Series, period: str | list) -> pd.Series:
    """Filter a series by ignore keys for the given period(s)."""
    if isinstance(period, list):
        for p in period:
            series = filter_series_by_ignore_keys(series, IGNORE_KEYS, p)
    else:
        series = filter_series_by_ignore_keys(series, IGNORE_KEYS, period)

    return series.dropna()


def compute_diff_and_rescaling(
    series: pd.Series, reference: float, escale: float, variations: bool
):
    """Compute relative differences (if 'variations' is True) and rescale values by 'escale'."""
    if variations:
        diff = (series - reference) / reference
    else:
        diff = series.copy()

    return diff, diff * escale


def resample_series(series: pd.Series, resampling_time: str, mask: pd.Series):
    """Calculate mean/std for resampled time ranges to which a mask is then applied. The function already adds UTC timezones to the series."""
    mean = series.resample(resampling_time).mean()
    std = series.resample(resampling_time).std()

    # add UTC timezone
    if mean.index.tz is None:
        mean = mean.tz_localize("UTC")
        std = std.tz_localize("UTC")
    # different timezone, convert to UTC
    elif mean.index.tz != pytz.UTC:
        mean = mean.tz_convert("UTC")
        std = std.tz_convert("UTC")

    # ensure mask has the same timezone as the resampled series
    if not mask.index.tz:
        mask = mask.tz_localize("UTC")

    # set to nan when the mask is False
    mean[~mask] = std[~mask] = np.nan

    return mean, std


def get_pulser_data(
    resampling_time: str,
    period: str | list,
    dfs: list,
    channel: str,
    escale: float,
    variations=False,
) -> dict:
    """
    Return a dictionary of geds and pulser filtered dataframes for which a time resampling is performed.

    Parameters
    ----------
    resampling_time : str
        Resampling time, eg '1HH' or '10T'.
    period : str | list
        Period or list of periods to inspect.
    dfs : list
        List of dataframes for geds and pulser events.
    channel : str
        Channel to inspect.
    escale : float
        Scaling factor used to compute relative differences in gain and calibration constant.
    variations : bool
        True if you want to retrieve % variations (default: False).
    """
    # geds
    ser_ged_cusp = dfs[0][channel].sort_index()
    ser_ged_cusp = filter_by_period(ser_ged_cusp, period)

    if ser_ged_cusp.empty:
        utils.logger.debug("...geds series is empty after filtering")
        return None

    # compute average over the first 10% of elements
    utils.logger.debug("...computing geds average")
    n_elements = max(int(len(ser_ged_cusp) * 0.10), 1)
    ged_cusp_av = np.nanmean(ser_ged_cusp.iloc[:n_elements])
    if np.isnan(ged_cusp_av):
        utils.logger.debug("...the geds average is NaN")
        return None

    ser_ged_cuspdiff, ser_ged_cuspdiff_kev = compute_diff_and_rescaling(
        ser_ged_cusp, ged_cusp_av, escale, variations
    )

    # hour counts masking
    mask = ser_ged_cusp.resample(resampling_time).count() > 0

    # resample geds series
    ged_cusp_hr_av_, ged_cusp_hr_std = resample_series(
        ser_ged_cuspdiff_kev, resampling_time, mask
    )

    # pulser series
    ser_pul_cusp = ser_pul_cuspdiff = ser_pul_cuspdiff_kev = pul_cusp_hr_av_ = (
        pul_cusp_hr_std
    ) = None
    ged_cusp_corr = ged_cusp_corr_kev = ged_cusp_cor_hr_av_ = ged_cusp_cor_hr_std = None
    # ...if pulser iis available:
    if not dfs[2].empty:
        ser_pul_cusp = dfs[2][1027203].sort_index()
        ser_pul_cusp = filter_by_period(ser_pul_cusp, period)

        # pulser average and diffs
        if not ser_pul_cusp.empty:
            n_elements_pul = max(int(len(ser_pul_cusp) * 0.10), 1)
            pul_cusp_av = np.nanmean(ser_pul_cusp.iloc[:n_elements_pul])
            ser_pul_cuspdiff, ser_pul_cuspdiff_kev = compute_diff_and_rescaling(
                ser_pul_cusp, pul_cusp_av, escale, variations
            )

            pul_cusp_hr_av_, pul_cusp_hr_std = resample_series(
                ser_pul_cuspdiff_kev, resampling_time, mask
            )

            # corrected GED
            common_index = ser_ged_cuspdiff.index.intersection(ser_pul_cuspdiff.index)
            ged_cusp_corr = (
                ser_ged_cuspdiff[common_index] - ser_pul_cuspdiff[common_index]
            )
            ged_cusp_corr_kev = ged_cusp_corr * escale
            ged_cusp_cor_hr_av_, ged_cusp_cor_hr_std = resample_series(
                ged_cusp_corr_kev, resampling_time, mask
            )

    return {
        "ged": {
            "cusp": ser_ged_cusp,
            "cuspdiff": ser_ged_cuspdiff,
            "cuspdiff_kev": ser_ged_cuspdiff_kev,
            "kevdiff_av": ged_cusp_hr_av_,
            "kevdiff_std": ged_cusp_hr_std,
        },
        "pul_cusp": {
            "raw": ser_pul_cusp,
            "rawdiff": ser_pul_cuspdiff,
            "kevdiff": ser_pul_cuspdiff_kev,
            "kevdiff_av": pul_cusp_hr_av_,
            "kevdiff_std": pul_cusp_hr_std,
        },
        "diff": {
            "raw": None,
            "rawdiff": ged_cusp_corr,
            "kevdiff": ged_cusp_corr_kev,
            "kevdiff_av": ged_cusp_cor_hr_av_,
            "kevdiff_std": ged_cusp_cor_hr_std,
        },
    }


def build_new_files(generated_path: str, period: str, run: str):
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
    """
    data_file = os.path.join(
        generated_path,
        "generated/plt/hit/phy",
        period,
        run,
        f"l200-{period}-{run}-phy-geds.hdf",
    )

    if not os.path.exists(data_file):
        utils.logger.debug(f"File not found: {data_file}. Exit here.")
        sys.exit()

    with h5py.File(data_file, "r") as f:
        my_keys = list(f.keys())

    info_dict = {"keys": my_keys}

    resampling_times = ["10min", "60min"]

    for idx, resample_unit in enumerate(resampling_times):
        new_file = os.path.join(
            generated_path,
            "generated/plt/hit/phy",
            period,
            run,
            f"l200-{period}-{run}-phy-geds-res_{resample_unit}.hdf",
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
                original_df.to_hdf(new_file, key=k, mode="a")
                continue

            original_df.index = pd.to_datetime(original_df.index)
            # resample
            resampled_df = original_df.resample(resample_unit).mean()
            # substitute the original df with the resampled one
            original_df = resampled_df
            # append resampled data to the new file
            resampled_df.to_hdf(new_file, key=k, mode="a")

        if idx == 0:
            json_output = os.path.join(
                generated_path,
                "generated/plt/hit/phy",
                period,
                run,
                f"l200-{period}-{run}-phy-geds-info.yaml",
            )
            with open(json_output, "w") as file:
                json.dump(info_dict, file, indent=4)


def plot_time_series(
    auto_dir_path: str,
    phy_mtg_data: str,
    output_folder: str,
    start_key: str,
    period: str,
    runs: list,
    current_run: str,
    pswd_email: str | None,
    save_pdf: bool,
    escale_val: float,
    last_checked: float | None,
    partition: bool,
    quadratic: bool,
    zoom: bool,
):
    """
    Generate and save time-series plots of calibration and monitoring data for germanium detectors across multiple runs.

    This function collects physics and calibration data from HDF5 monitoring files and visualizes stability over time.
    Channels with no pulser entries are automatically skipped.
    Corrections are applied to the gain if pulser data is available ('GED corrected'), otherwise uncorrected data is plotted.
    The plots are saved as pickled objects for later retrieval (eg. in the online Dashboard) and optionally as PDFs:

    - plots saved in shelve database files under ``<output_folder>/<period>/mtg/l200-<period>-phy-monitoring``;
    - if `save_pdf=True`, PDF copies saved under ``<output_folder>/<period>/mtg/pdf/st<string>/``.

    Parameters
    ----------
    auto_dir_path : str
        Path to tmp-auto public data files (eg /data2/public/prodenv/prod-blind/tmp-auto).
    phy_mtg_data : str
        Path to generated monitoring hdf files.
    output_folder : str
        Path to output folder.
    start_key : str
        First timestamp of the inspected range.
    period : str
        Period to inspect.
    runs : list
        Available runs to inspect for a given period.
    current_run : str
        Run under inspection.
    pswd_email : str | None
        Password to access the legend.data.monitoring@gmail.com account for sending alert messages.
    save_pdf : bool
        True if you want to save pdf files too; default: False.
    escale_val : float
        Energy scale at which evaluating the gain differences; default: 2039 keV (76Ge Qbb).
    last_checked : float | None
        Timestamp of the last check.
    partition : bool
        False if not partition data; default: False.
    quadratic : bool
        True if you want to plot the quadratic resolution too; default: False.
    zoom : bool
        True to zoom over y axis; default: False.
    """
    avail_runs = []
    for entry in runs:
        new_entry = entry.replace(",", "").replace("[", "").replace("]", "")
        avail_runs.append(new_entry)

    dataset = {period: avail_runs}
    utils.logger.debug(f"Available phy data: {dataset}")

    xlim_idx = 1

    fit_flag = "quadratic" if quadratic is True else "linear"

    meta = LegendMetadata(os.path.join(auto_dir_path, "inputs/"))
    # get channel map
    chmap = meta.channelmap(start_key)
    # get string info
    str_chns = {}
    string_numbers = [
        int(item.get("location", {}).get("string"))
        for item in chmap.values()
        if "location" in item and "string" in item["location"]
    ]
    string_numbers = list(dict.fromkeys(string_numbers))  # unique values
    for string in string_numbers:
        channels = [
            f"ch{chmap[ged].daq.rawid}"
            for ged, dic in chmap.items()
            if dic["system"] == "geds"
            and dic["location"]["string"] == string
            and dic["analysis"]["processable"]
            is True  # prevent to load non-processable detectors
        ]
        if len(channels) > 0:
            str_chns[string] = channels

    email_message = []

    # skip detectors with no pulser entries
    no_puls_dets = utils.NO_PULS_DETS
    flag_expr = " or ".join(
        f'(channel == "{channel}" and period in {periods})'
        for channel, periods in no_puls_dets.items()
    )
    period_list = list(dataset.keys())

    # gain over period
    for index_i in tqdm(range(len(period_list))):
        period = period_list[index_i]
        run_list = dataset[period]

        (
            geds_df_cuspEmax_abs,
            geds_df_cuspEmax_abs_corr,
            puls_df_cuspEmax_abs,
        ) = get_dfs(phy_mtg_data, period, run_list, "TrapemaxCtcCal")
        geds_df_trapTmax, geds_df_tp0est, puls_df_trapTmax, puls_df_tp0est = (
            get_traptmax_tp0est(phy_mtg_data, period, run_list)
        )

        if (
            geds_df_cuspEmax_abs is None
            or geds_df_cuspEmax_abs_corr is None
            # no need to exit if pulser01ana does not exits, handled it properly now
            # or puls_df_cuspEmax_abs is None
        ):
            utils.logger.debug("Dataframes are None for %s!", period)
            continue

        # check if geds df is empty; if pulser is, means we do not apply any correction
        # (and thus geds_corr is also empty - the code will handle the case)
        if (
            geds_df_cuspEmax_abs.empty
            # or geds_df_cuspEmax_abs_corr.empty
            # or puls_df_cuspEmax_abs.empty
        ):
            utils.logger.debug("Dataframes are empty for %s!", period)
            continue

        dfs = [
            geds_df_cuspEmax_abs,
            geds_df_cuspEmax_abs_corr,
            puls_df_cuspEmax_abs,
            geds_df_trapTmax,
            geds_df_tp0est,
            puls_df_trapTmax,
            puls_df_tp0est,
        ]

        string_list = list(str_chns.keys())
        for index_j in tqdm(range(len(string_list))):
            string = string_list[index_j]

            channel_list = str_chns[string]
            for index_k in range(len(channel_list)):
                channel = channel_list[index_k]
                channel_name = chmap.map("daq.rawid")[int(channel[2:])]["name"]
                resampling_time = "1h"  # if len(runs)>1 else "10T"
                if int(channel.split("ch")[-1]) not in list(dfs[0].columns):
                    utils.logger.debug(f"{channel} is not present in the dataframe!")
                    continue

                utils.logger.debug(f"Inspecting {channel_name}")
                pulser_data = get_pulser_data(
                    resampling_time,
                    period,
                    dfs,
                    int(channel.split("ch")[-1]),
                    escale=escale_val,
                    variations=True,
                )

                fig, ax = plt.subplots(figsize=(12, 4))
                utils.logger.debug("...getting calibration data")
                pars_data = get_calib_pars(
                    auto_dir_path,
                    period,
                    run_list,
                    [channel, channel_name],
                    partition,
                    escale=escale_val,
                    fit=fit_flag,
                )

                t0 = pars_data["run_start"]
                if not eval(flag_expr):
                    kevdiff = (
                        pulser_data["ged"]["kevdiff_av"]
                        if pulser_data["diff"]["kevdiff_av"] is None
                        else pulser_data["diff"]["kevdiff_av"]
                    )

                    # PULS01ANA has a signal - we can correct GEDS energies for it!
                    if pulser_data["pul_cusp"]["kevdiff_av"] is not None:
                        plt.plot(
                            pulser_data["pul_cusp"]["kevdiff_av"],
                            "C2",
                            label="PULS01ANA",
                        )
                        plt.plot(
                            pulser_data["diff"]["kevdiff_av"],
                            "C4",
                            label="GED corrected",
                        )
                        plt.fill_between(
                            pulser_data["diff"]["kevdiff_av"].index.values,
                            y1=[
                                float(i) - float(j)
                                for i, j in zip(
                                    pulser_data["diff"]["kevdiff_av"].values,
                                    pulser_data["diff"]["kevdiff_std"].values,
                                )
                            ],
                            y2=[
                                float(i) + float(j)
                                for i, j in zip(
                                    pulser_data["diff"]["kevdiff_av"].values,
                                    pulser_data["diff"]["kevdiff_std"].values,
                                )
                            ],
                            color="k",
                            alpha=0.2,
                            label=r"±1$\sigma$",
                        )
                    # else, no correction is applied
                    else:
                        plt.plot(
                            pulser_data["ged"]["kevdiff_av"].sort_index(),
                            "dodgerblue",
                            label="GED uncorrected",
                        )
                        plt.fill_between(
                            pulser_data["ged"]["kevdiff_av"].index.values,
                            y1=[
                                float(i) - float(j)
                                for i, j in zip(
                                    pulser_data["ged"]["kevdiff_av"].values,
                                    pulser_data["ged"]["kevdiff_std"].values,
                                )
                            ],
                            y2=[
                                float(i) + float(j)
                                for i, j in zip(
                                    pulser_data["ged"]["kevdiff_av"].values,
                                    pulser_data["ged"]["kevdiff_std"].values,
                                )
                            ],
                            color="k",
                            alpha=0.2,
                            label=r"±1$\sigma$",
                        )

                plt.plot(
                    pars_data["run_start"] - pd.Timedelta(hours=5),
                    pars_data["fep_diff"],
                    "kx",
                    label="FEP gain",
                )
                plt.plot(
                    pars_data["run_start"] - pd.Timedelta(hours=5),
                    pars_data["cal_const_diff"],
                    "rx",
                    label="cal. const. diff",
                )

                for ti in pars_data["run_start"]:
                    plt.axvline(ti, color="dimgrey", ls="--")

                for i in range(len(t0)):
                    if i == len(pars_data["run_start"]) - 1:
                        plt.plot(
                            [t0[i], t0[i] + pd.Timedelta(days=7)],
                            [pars_data["res"][i] / 2, pars_data["res"][i] / 2],
                            "b-",
                        )
                        plt.plot(
                            [t0[i], t0[i] + pd.Timedelta(days=7)],
                            [-pars_data["res"][i] / 2, -pars_data["res"][i] / 2],
                            "b-",
                        )
                        if quadratic:
                            plt.plot(
                                [t0[i], t0[i] + pd.Timedelta(days=7)],
                                [
                                    pars_data["res_quad"][i] / 2,
                                    pars_data["res_quad"][i] / 2,
                                ],
                                color="dodgerblue",
                                linestyle="-",
                            )
                            plt.plot(
                                [t0[i], t0[i] + pd.Timedelta(days=7)],
                                [
                                    -pars_data["res_quad"][i] / 2,
                                    -pars_data["res_quad"][i] / 2,
                                ],
                                color="dodgerblue",
                                linestyle="-",
                            )
                    else:
                        plt.plot(
                            [t0[i], t0[i + 1]],
                            [pars_data["res"][i] / 2, pars_data["res"][i] / 2],
                            "b-",
                        )
                        plt.plot(
                            [t0[i], t0[i + 1]],
                            [-pars_data["res"][i] / 2, -pars_data["res"][i] / 2],
                            "b-",
                        )
                        if quadratic:
                            plt.plot(
                                [t0[i], t0[i + 1]],
                                [
                                    pars_data["res_quad"][i] / 2,
                                    pars_data["res_quad"][i] / 2,
                                ],
                                color="dodgerblue",
                                linestyle="-",
                            )
                            plt.plot(
                                [t0[i], t0[i + 1]],
                                [
                                    -pars_data["res_quad"][i] / 2,
                                    -pars_data["res_quad"][i] / 2,
                                ],
                                color="dodgerblue",
                                linestyle="-",
                            )

                    if str(pars_data["res"][i] / 2 * 1.1) != "nan" and i < len(
                        pars_data["res"]
                    ) - (xlim_idx - 1):
                        plt.text(
                            t0[i],
                            pars_data["res"][i] / 2 * 1.1,
                            "{:.2f}".format(pars_data["res"][i]),
                            color="b",
                        )

                    if quadratic:
                        if str(pars_data["res_quad"][i] / 2 * 1.5) != "nan" and i < len(
                            pars_data["res"]
                        ) - (xlim_idx - 1):
                            plt.text(
                                t0[i],
                                pars_data["res_quad"][i] / 2 * 1.5,
                                "{:.2f}".format(pars_data["res_quad"][i]),
                                color="dodgerblue",
                            )

                fig.suptitle(
                    f'period: {period} - string: {string} - position: {chmap.map("daq.rawid")[int(channel[2:])]["location"]["position"]} - ged: {channel_name}'
                )
                plt.ylabel(r"Energy diff / keV")
                plt.plot([0, 1], [0, 1], "b", label="Qbb FWHM keV lin.")
                if quadratic:
                    plt.plot([1, 2], [1, 2], "dodgerblue", label="Qbb FWHM keV quadr.")

                if zoom:
                    if flag_expr:
                        plt.ylim(-3, 3)
                    else:
                        bound = np.average(pulser_data["ged"]["cusp_av"].dropna())
                        plt.ylim(-2.5 * bound, 2.5 * bound)
                max_date = pulser_data["ged"]["kevdiff_av"].index.max()
                time_difference = max_date.tz_localize(None) - t0[
                    -xlim_idx
                ].tz_localize(None)
                plt.xlim(
                    t0[0] - pd.Timedelta(hours=8), t0[-xlim_idx] + time_difference * 1.5
                )  # pd.Timedelta(days=7))# --> change me to resize the width of the last run
                plt.legend(loc="lower left")
                plt.tight_layout()
                end_folder = os.path.join(
                    output_folder,
                    period,
                    "mtg",
                )
                os.makedirs(end_folder, exist_ok=True)

                if save_pdf:
                    mgt_folder = os.path.join(end_folder, "pdf", f"st{string}")
                    os.makedirs(mgt_folder, exist_ok=True)

                    pdf_name = os.path.join(
                        mgt_folder,
                        f"{period}_string{string}_pos{chmap.map('daq.rawid')[int(channel[2:])]['location']['position']}_{channel_name}_gain_shift.pdf",
                    )
                    plt.savefig(pdf_name)

                # pickle and save calibration inputs retrieved ots in a shelve file
                # serialize the plot
                serialized_plot = pickle.dumps(plt.gcf())
                # store the serialized plot in a shelve object under key
                with shelve.open(
                    os.path.join(end_folder, f"l200-{period}-phy-monitoring"),
                    "c",
                    protocol=pickle.HIGHEST_PROTOCOL,
                ) as shelf:
                    shelf[
                        f'{period}_string{string}_pos{chmap.map("daq.rawid")[int(channel[2:])]["location"]["position"]}_{channel_name}_gain_shift'
                    ] = serialized_plot
                plt.close(fig)

                # structure of pickle files:
                #  - p08_string1_pos1_V02160A
                #  - p08_string1_pos2_V02160B
                #  - ...
                #  - p08_string2_pos1_B00035C
                #  - p08_string2_pos2_C000RG1
                #  - ...

    # parameters (bsln, gain, ...) variations over run
    ylabels = {
        "TrapemaxCtcCal": "Energy diff / keV",
        "Baseline": "Baseline % variations",
        "BlStd": "Baseline std [ADC]",
    }
    colors = {
        "TrapemaxCtcCal": ["dodgerblue", "b"],
        "Baseline": ["r", "firebrick"],
        "BlStd": ["peru", "saddlebrown"],
    }
    percentage = {
        "TrapemaxCtcCal": True,
        "Baseline": True,
        "BlStd": False,
    }
    titles = {
        "TrapemaxCtcCal": "Gain",
        "Baseline": "FPGA baseline",
        "BlStd": "Baseline std",
    }
    limits = {
        "TrapemaxCtcCal": [None, None],
        "Baseline": [-10, 10],
        "BlStd": [None, 100],
    }
    for inspected_parameter in ["Baseline", "TrapemaxCtcCal", "BlStd"]:
        for index_i in tqdm(range(len(period_list))):
            period = period_list[index_i]

            (
                geds_df_cuspEmax_abs,
                geds_df_cuspEmax_abs_corr,
                puls_df_cuspEmax_abs,
            ) = get_dfs(phy_mtg_data, period, [current_run], inspected_parameter)
            geds_df_trapTmax, geds_df_tp0est, puls_df_trapTmax, puls_df_tp0est = (
                get_traptmax_tp0est(phy_mtg_data, period, [current_run])
            )

            if (
                geds_df_cuspEmax_abs is None
                or geds_df_cuspEmax_abs_corr is None
                or puls_df_cuspEmax_abs is None
            ):
                utils.logger.debug("Dataframes are None for %s!", period)
                continue
            if geds_df_cuspEmax_abs.empty:
                utils.logger.debug("Dataframes are empty for %s!", period)
                continue
            dfs = [
                geds_df_cuspEmax_abs,
                geds_df_cuspEmax_abs_corr,
                puls_df_cuspEmax_abs,
                geds_df_trapTmax,
                geds_df_tp0est,
                puls_df_trapTmax,
                puls_df_tp0est,
            ]

            string_list = list(str_chns.keys())
            for index_j in tqdm(range(len(string_list))):
                string = string_list[index_j]

                channel_list = str_chns[string]
                for index_k in range(len(channel_list)):
                    channel = channel_list[index_k]
                    channel_name = chmap.map("daq.rawid")[int(channel[2:])]["name"]
                    resampling_time = "1h"
                    if int(channel.split("ch")[-1]) not in list(dfs[0].columns):
                        utils.logger.debug(
                            f"{channel} is not present in the dataframe!"
                        )
                        continue

                    utils.logger.debug(
                        f"Inspecting {channel_name} for {inspected_parameter}"
                    )
                    pulser_data = get_pulser_data(
                        resampling_time,
                        period,
                        dfs,
                        int(channel.split("ch")[-1]),
                        escale=(
                            escale_val if inspected_parameter == "TrapemaxCtcCal" else 1
                        ),
                        variations=percentage[inspected_parameter],
                    )

                    fig, ax = plt.subplots(figsize=(12, 4))
                    utils.logger.debug("...getting calibration data")
                    pars_data = get_calib_pars(
                        auto_dir_path,
                        period,
                        [current_run],
                        [channel, channel_name],
                        partition,
                        escale=(
                            escale_val if inspected_parameter == "TrapemaxCtcCal" else 1
                        ),
                        fit=fit_flag,
                    )
                    threshold = (
                        [pars_data["res"][0], pars_data["res"][0]]
                        if inspected_parameter == "TrapemaxCtcCal"
                        else limits[inspected_parameter]
                    )

                    t0 = pars_data["run_start"]
                    if not eval(flag_expr):
                        kevdiff = (
                            pulser_data["ged"]["kevdiff_av"]
                            if pulser_data["diff"]["kevdiff_av"] is None
                            else pulser_data["diff"]["kevdiff_av"]
                        )

                        # check threshold and send automatic mail
                        email_message = utils.check_threshold(
                            kevdiff,
                            pswd_email,
                            last_checked,
                            t0,
                            pars_data,
                            threshold,
                            period,
                            current_run,
                            channel_name,
                            string,
                            email_message,
                            titles[inspected_parameter],
                        )

                        # PULS01ANA has a signal - we can correct GEDS energies for it!
                        # only in the case of energy parameters
                        if (
                            pulser_data["pul_cusp"]["kevdiff_av"] is not None
                            and inspected_parameter == "TrapemaxCtcCal"
                        ):
                            plt.plot(
                                pulser_data["pul_cusp"]["kevdiff_av"],
                                "C2",
                                label="PULS01ANA",
                            )
                            plt.plot(
                                pulser_data["diff"]["kevdiff_av"],
                                "C4",
                                label="GED corrected",
                            )
                            plt.fill_between(
                                pulser_data["diff"]["kevdiff_av"].index.values,
                                y1=[
                                    float(i) - float(j)
                                    for i, j in zip(
                                        pulser_data["diff"]["kevdiff_av"].values,
                                        pulser_data["diff"]["kevdiff_std"].values,
                                    )
                                ],
                                y2=[
                                    float(i) + float(j)
                                    for i, j in zip(
                                        pulser_data["diff"]["kevdiff_av"].values,
                                        pulser_data["diff"]["kevdiff_std"].values,
                                    )
                                ],
                                color="k",
                                alpha=0.2,
                                label=r"±1$\sigma$",
                            )
                        # else, no correction is applied
                        else:
                            if percentage[inspected_parameter] is True:
                                pulser_data["ged"]["kevdiff_av"] *= 100
                                pulser_data["ged"]["kevdiff_std"] *= 100

                            plt.plot(
                                pulser_data["ged"]["kevdiff_av"].sort_index(),
                                color=colors[inspected_parameter][0],
                                label="GED uncorrected",
                            )
                            plt.fill_between(
                                pulser_data["ged"]["kevdiff_av"].index.values,
                                y1=[
                                    float(i) - float(j)
                                    for i, j in zip(
                                        pulser_data["ged"]["kevdiff_av"].values,
                                        pulser_data["ged"]["kevdiff_std"].values,
                                    )
                                ],
                                y2=[
                                    float(i) + float(j)
                                    for i, j in zip(
                                        pulser_data["ged"]["kevdiff_av"].values,
                                        pulser_data["ged"]["kevdiff_std"].values,
                                    )
                                ],
                                color="k",
                                alpha=0.2,
                                label=r"±1$\sigma$",
                            )

                    # plot resolution only for the energy parameters
                    if inspected_parameter == "TrapemaxCtcCal":
                        plt.plot(
                            [t0[0], t0[0] + pd.Timedelta(days=7)],
                            [pars_data["res"][0] / 2, pars_data["res"][0] / 2],
                            color=colors[inspected_parameter][1],
                            ls="-",
                        )
                        plt.plot(
                            [t0[0], t0[0] + pd.Timedelta(days=7)],
                            [-pars_data["res"][0] / 2, -pars_data["res"][0] / 2],
                            color=colors[inspected_parameter][1],
                            ls="-",
                        )

                        if str(pars_data["res"][0] / 2 * 1.1) != "nan" and 0 < len(
                            pars_data["res"]
                        ) - (xlim_idx - 1):
                            plt.text(
                                t0[0],
                                pars_data["res"][0] / 2 * 1.1,
                                "{:.2f}".format(pars_data["res"][0]),
                                color=colors[inspected_parameter][1],
                            )
                        plt.plot(
                            [0, 1],
                            [0, 1],
                            color=colors[inspected_parameter][1],
                            label="Qbb FWHM keV lin.",
                        )
                    else:
                        if limits[inspected_parameter][1] is not None:
                            plt.plot(
                                [t0[0], t0[0] + pd.Timedelta(days=7)],
                                [
                                    limits[inspected_parameter][1],
                                    limits[inspected_parameter][1],
                                ],
                                color=colors[inspected_parameter][1],
                                ls="-",
                            )
                        if limits[inspected_parameter][0] is not None:
                            plt.plot(
                                [t0[0], t0[0] + pd.Timedelta(days=7)],
                                [
                                    limits[inspected_parameter][0],
                                    limits[inspected_parameter][0],
                                ],
                                color=colors[inspected_parameter][1],
                                ls="-",
                            )

                    plt.ylabel(ylabels[inspected_parameter])
                    fig.suptitle(
                        f'period: {period} - string: {string} - position: {chmap.map("daq.rawid")[int(channel[2:])]["location"]["position"]} - ged: {channel_name}'
                    )

                    if zoom is True:
                        bound = np.average(pulser_data["ged"]["kevdiff_std"].dropna())
                        plt.ylim(-3.5 * bound, 3.5 * bound)

                    max_date = pulser_data["ged"]["kevdiff_av"].index.max()
                    time_difference = max_date.tz_localize(None) - t0[
                        -xlim_idx
                    ].tz_localize(None)
                    plt.xlim(
                        t0[0] - pd.Timedelta(hours=0.5),
                        t0[-xlim_idx] + time_difference * 1.1,
                    )
                    plt.legend(loc="lower left")
                    plt.tight_layout()

                    end_folder = os.path.join(
                        output_folder,
                        period,
                        current_run,
                        "mtg",
                        inspected_parameter,
                    )
                    os.makedirs(end_folder, exist_ok=True)

                    if save_pdf:
                        mgt_folder = os.path.join(end_folder, "pdf", f"st{string}")
                        os.makedirs(mgt_folder, exist_ok=True)

                        pdf_name = os.path.join(
                            mgt_folder,
                            f"{period}_string{string}_pos{chmap.map('daq.rawid')[int(channel[2:])]['location']['position']}_{channel_name}_{inspected_parameter}.pdf",
                        )
                        plt.savefig(pdf_name)

                    # pickle and save calibration inputs retrieved ots in a shelve file
                    # serialize the plot
                    serialized_plot = pickle.dumps(plt.gcf())
                    # store the serialized plot in a shelve object under key
                    with shelve.open(
                        os.path.join(
                            end_folder,
                            f"l200-{period}-phy-{inspected_parameter}",
                        ),
                        "c",
                        protocol=pickle.HIGHEST_PROTOCOL,
                    ) as shelf:
                        shelf[
                            f'{period}_string{string}_pos{chmap.map("daq.rawid")[int(channel[2:])]["location"]["position"]}_{channel_name}_{inspected_parameter}'
                        ] = serialized_plot
                    plt.close(fig)

    if len(email_message) > 1 and pswd_email is not None:
        with open("message.txt", "w") as f:
            for line in email_message:
                f.write(line + "\n")
        utils.send_email_alert(
            pswd_email, ["sofia.calgaro@physik.uzh.ch"], "message.txt"
        )
        os.remove("message.txt")
