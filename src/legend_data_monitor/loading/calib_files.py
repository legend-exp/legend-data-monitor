"""Plot-free loaders for calibration summary files and run time metadata."""

import json
import re
from functools import cache
from pathlib import Path

import lh5
import numpy as np
import pandas as pd
import yaml

from .. import utils
from ..processing.series import compute_diff

CALIB_RUNS = utils.CALIB_RUNS


def find_energy_key(ecal_results: dict) -> tuple:
    """
    Find the energy-estimator entry of a calibration dictionary.

    The estimators are listed in ``settings/experiment.yaml`` in order of
    preference (run calibration before the partition one); the first one
    present wins.

    Parameters
    ----------
    ecal_results : dict
        Dictionary containing energy calibration results.

    Returns
    -------
    tuple
        ``(key, sub_dict)`` for the estimator found, ``(None, {})`` otherwise.
    """
    for key in utils.EXPERIMENT["energy"]["estimators"]:
        if key in ecal_results:
            return key, ecal_results[key]
    utils.logger.debug("No energy estimator key in %s", list(ecal_results)[:5])
    return None, {}


def get_energy_key(ecal_results: dict) -> dict:
    """
    Retrieve the energy calibration results from a given dictionary.

    Parameters
    ----------
    ecal_results : dict
        Dictionary containing energy calibration results.

    Returns
    -------
    dict
        The estimator's sub-dictionary, empty when none is present.
    """
    return find_energy_key(ecal_results)[1]


def uncalibrated_variable(estimator: str) -> str:
    """
    Return the uncalibrated variable a calibration expression is written in.

    ``cuspEmax_ctc_cal`` and ``cuspEmax_ctc_runcal`` are both calibrations of
    ``cuspEmax_ctc``, which is the name their ``expression`` binds.

    Parameters
    ----------
    estimator : str
        Calibrated estimator key.

    Returns
    -------
    str
        The uncalibrated variable name.
    """
    for suffix in ("_runcal", "_cal"):
        if estimator.endswith(suffix):
            return estimator[: -len(suffix)]
    return estimator


@cache
def get_calibration_file(folder_par: str) -> dict:
    """
    Return the content of the JSON/YAML calibration file in folder_par.

    The result is cached per folder (these files are re-read once per channel
    otherwise); treat the returned dict as read-only.

    Parameters
    ----------
    folder_par : str
        Path to the folder containing calibration summary files.
    """
    files = [p.name for p in Path(folder_par).iterdir()]
    json_files = [f for f in files if f.endswith(".json")]
    yaml_files = [f for f in files if f.endswith((".yaml", ".yml"))]

    if json_files:
        filepath = str(Path(folder_par) / json_files[0])
        with open(filepath) as f:
            pars_dict = json.load(f)
    elif yaml_files:
        filepath = str(Path(folder_par) / yaml_files[0])
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
        Channel name or IDs (eg ch10000).
    key_result : str
        Key name used to extract the resolution results from the parsed file.
    fit : str
        Fitting method used for energy resolution, either 'linear' or 'quadratic'.
    """
    if channel not in pars_dict:
        return np.nan, np.nan

    result = get_energy_key(pars_dict[channel]["results"][key_result])
    eres_linear = result.get("eres_linear") or {}
    Qbb_keys = [k for k in eres_linear if "Qbb_fwhm_in_" in k]
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

    estimator, ecal_results = find_energy_key(pars_dict[channel]["pars"]["operations"])
    if not ecal_results:
        return np.nan, np.nan
    expr = ecal_results["expression"]
    params = ecal_results["parameters"]
    variable = uncalibrated_variable(estimator)

    fep_cal = eval(expr, {}, {**params, variable: fep_peak_pos})
    fep_cal_err = eval(expr, {}, {**params, variable: fep_peak_pos_err})

    return fep_cal, fep_cal_err


_run_times_cache: dict = {}


def get_run_start_end_times(
    sto,
    tiers: list,
    period: str,
    run: str,
    tier: str,
    pulser_rawid: int | None = None,
):
    """
    Determine the start and end timestamps for a given run, including the special case for additional final calibration runs.

    The pulser fires steadily for the whole run, so its first and last
    timestamps bracket the run better than a detector that may have been off;
    its rawid comes from the channel map (see :func:`utils.aux_channels`).

    Results are cached per (tiers, period, run, tier, channel) — the reads
    were previously repeated for every channel.

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
    pulser_rawid : int, optional
        Pulser rawid to read the timestamps from; the first channel in the
        file is used when omitted.
    """
    cache_key = (tuple(tiers), period, run, tier, pulser_rawid)
    if cache_key in _run_times_cache:
        return _run_times_cache[cache_key]

    folder_tier = str(Path(tiers[0 if tier == "hit" else 1]) / "cal" / period / run)
    dir_path = str(Path(tiers[-1]) / "phy" / period)
    pattern = re.compile(
        r"^l\d+-p\d+-r\d+-(cal|hit|raw)-\d{8}T\d{6}Z-tier_(dsp|hit|raw)\.lh5$"
    )

    run_files = sorted(
        f for f in [p.name for p in Path(folder_tier).iterdir()] if pattern.match(f)
    )

    if pulser_rawid is None:
        channels = [
            c
            for c in lh5.ls(str(Path(folder_tier) / run_files[0]))
            if c.startswith("ch")
        ]
        timestamp_key = f"{channels[0]}/dsp/timestamp"
        utils.logger.debug("...no pulser rawid given, timing off %s", timestamp_key)
    else:
        timestamp_key = f"ch{pulser_rawid}/dsp/timestamp"

    # for when we have a calib run but zero phy runs for a given period
    if Path(dir_path).is_dir() and run not in [
        p.name for p in Path(dir_path).iterdir()
    ]:
        run_end_time = pd.to_datetime(
            sto.read(timestamp_key, str(Path(folder_tier) / run_files[-1]))[-1],
            unit="s",
        )
        run_start_time = run_end_time
    else:
        run_start_time = pd.to_datetime(
            sto.read(timestamp_key, str(Path(folder_tier) / run_files[0]))[0],
            unit="s",
        )
        run_end_time = pd.to_datetime(
            sto.read(timestamp_key, str(Path(folder_tier) / run_files[-1]))[-1],
            unit="s",
        )

    _run_times_cache[cache_key] = (run_start_time, run_end_time)
    return run_start_time, run_end_time


@cache
def _load_validity_file(validity_file: str) -> tuple:
    """Load and cache a validity.yaml file (read once per run, not per channel)."""
    with open(validity_file) as f:
        return tuple(yaml.load(f, Loader=yaml.CLoader))


@cache
def _first_run_key(run_path: str) -> str:
    """Return the timestamp key of the first file in a run directory (cached)."""
    return sorted([p.name for p in Path(run_path).iterdir()])[0].split("-")[4]


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
    data_type: str,
    pulser_rawid: int | None = None,
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
    data_type : str
    """
    sto = lh5.LH5Store()
    channel = channel_info[0]
    channel_name = channel_info[1]

    validity_file = str(Path(pars[2 if tier == "hit" else 3]) / "validity.yaml")
    validity_dict = _load_validity_file(validity_file)

    # find first key of current run
    run_path = str(Path(tiers[2 if tier == "hit" else 3]) / data_type / period / run)
    if not Path(run_path).exists():
        return calib_data
    start_key = _first_run_key(run_path)
    # use key to load the right yaml file
    valid_entries = [e for e in validity_dict if e["valid_from"] <= start_key]
    if valid_entries:
        apply = max(valid_entries, key=lambda e: e["valid_from"])["apply"][0]
        run_to_apply = apply.split("/")[-1].split("-")[2]
    else:
        utils.logger.debug(
            f"No valid calibration was found for {period}-{run}. Return."
        )
        return calib_data

    folder_par = str(
        Path(pars[2 if tier == "hit" else 3]) / "cal" / period / run_to_apply
    )
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
        sto, tiers, period, run_to_apply, tier, pulser_rawid
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
    partition_dir = Path(tiers[1])
    if partition_dir.is_dir() and any(partition_dir.iterdir()):
        return "pht", "partition_ecal"
    return "hit", "ecal"


def get_calib_pars(
    path: str,
    period: str | list,
    run_list: list,
    channel_info: list,
    partition: bool,
    data_type: str,
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
    run_list = [r for r in run_list if "old" not in str(r)]

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
    # the pulser brackets the run best; the channel map says which one it is.
    # Only a hint: without reachable metadata the first channel in the file is
    # used instead, so a mock or partial tree still works.
    try:
        pulser_rawid = utils.aux_channels(str(Path(path) / "inputs")).get("pulser")
    except (FileNotFoundError, KeyError, ValueError) as exc:
        utils.logger.debug("...no channel map for the run timing (%s)", exc)
        pulser_rawid = None

    for run in run_list:
        calib_data = get_calib_data_dict(
            calib_data,
            channel_info,
            tiers,
            pars,
            period,
            run,
            tier,
            key_result,
            fit,
            data_type,
            pulser_rawid,
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
