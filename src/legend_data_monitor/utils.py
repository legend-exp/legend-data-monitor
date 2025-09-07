import glob
import importlib.resources
import json
import logging
import os
import re
import smtplib
import sys
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import h5py
import numpy as np
import pandas as pd
import yaml
from legendmeta import JsonDB
from lgdo import lh5
from pandas import DataFrame

# -------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# stream handler (console)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)

# format
formatter = logging.Formatter("%(asctime)s:  %(message)s")
stream_handler.setFormatter(formatter)

# add to logger
logger.addHandler(stream_handler)

# ------------------------------------------------------------------------- SOME DICTIONARIES LOADING/DEFINITION

pkg = importlib.resources.files("legend_data_monitor")

# load dictionary with plot info (= units, thresholds, label, ...)
with open(pkg / "settings" / "par-settings.yaml") as f:
    PLOT_INFO = yaml.load(f, Loader=yaml.CLoader)

# which parameter belongs to which tier
with open(pkg / "settings" / "parameter-tiers.yaml") as f:
    PARAMETER_TIERS = yaml.load(f, Loader=yaml.CLoader)

# which lh5 parameters are needed to be loaded from lh5 to calculate them
with open(pkg / "settings" / "special-parameters.yaml") as f:
    SPECIAL_PARAMETERS = yaml.load(f, Loader=yaml.CLoader)

# flag renames for evt type
with open(pkg / "settings" / "flags.yaml") as f:
    FLAGS_RENAME = yaml.load(f, Loader=yaml.CLoader)

# list of detectors that have no pulser signal in a given period
with open(pkg / "settings" / "no-pulser-dets.yaml") as f:
    NO_PULS_DETS = yaml.load(f, Loader=yaml.CLoader)

# dictionary of keys to ignore
with open(pkg / "settings" / "ignore-keys.yaml") as f:
    IGNORE_KEYS = yaml.load(f, Loader=yaml.CLoader)

# convert all to lists for convenience
for param in SPECIAL_PARAMETERS:
    if isinstance(SPECIAL_PARAMETERS[param], str):
        SPECIAL_PARAMETERS[param] = [SPECIAL_PARAMETERS[param]]

# load SC params and corresponding flags to get specific parameters from big dfs that are stored in the database
with open(pkg / "settings" / "SC-params.yaml") as f:
    SC_PARAMETERS = yaml.load(f, Loader=yaml.CLoader)

# load final calibration run for each period
with open(pkg / "settings" / "final-calibrations.yaml") as f:
    CALIB_RUNS = yaml.load(f, Loader=yaml.CLoader)

# load list of columns to load for a dataframe
COLUMNS_TO_LOAD = [
    "name",
    "location",
    "channel",
    "position",
    "cc4_id",
    "cc4_channel",
    "daq_crate",
    "daq_card",
    "HV_card",
    "HV_channel",
    "det_type",
]

# map position/location for special systems
SPECIAL_SYSTEMS = {"pulser": 0, "pulser01ana": -1, "FCbsln": -2, "muon": -3}

# dictionary with timestamps to remove for specific channels
with open(pkg / "settings" / "remove-keys.yaml") as f:
    REMOVE_KEYS = yaml.load(f, Loader=yaml.CLoader)["remove-keys"]

# dictionary with detectors to remove
with open(pkg / "settings" / "remove-dets.yaml") as f:
    REMOVE_DETS = yaml.load(f, Loader=yaml.CLoader)["remove-dets"]

# -------------------------------------------------------------------------
# Subsystem related functions (for getting channel map & status)
# -------------------------------------------------------------------------


def get_valid_path(base_path):
    if os.path.exists(base_path):
        return base_path

    fallback_subdirs = ["psp", "hit", "pht", "pet", "evt", "skm"]
    for fallback_subdir in fallback_subdirs:
        fallback_path = base_path.replace("dsp", fallback_subdir)

        if os.path.exists(fallback_path):
            return fallback_path

    logger.warning(
        "\033[93mThe path of dsp/hit/evt/psp/pht/pet/skm files is not valid, check config['dataset'] and try again.\033[0m",
    )
    sys.exit()


def get_query_times(**kwargs):
    """
    Get time ranges for DataLoader query from user input, as well as first/last timestamp for channel map / status / SC query.

    Parameters
    ----------
    dataset : dict, optional
        Dictionary with the following keys (note: can provide the same keys as in `dataset` but separately, i.e. `path=...`, `version=...`, `type=...`, and one of `start=...&end=...`, `window=...`, `timestamps=...`, or `runs=...`):

            - 'path' : str
                Base path to the dataset.
            - 'version' : str
                Dataset version.
            - 'type' : str
                Type of dataset. Note: multiple types are not currently supported.
            - Time selection keys (choose one):

                1. 'start' : str, 'end' : str
                    Start and end datetime in the format `'YYYY-MM-DD hh:mm:ss'`.
                2. 'window' : str
                    Time window from the current time, e.g., `'1d 2h 30m'` for 1 day, 2 hours, 30 minutes.
                3. 'timestamps' : str or list of str
                    Timestamps in the format `'YYYYMMDDThhmmssZ'`.
                4. 'runs' : int or list of ints
                    Run number(s), e.g., `10` corresponds to run `'r010'`.

    Notes
    -----
    - `path`, `version`, and `type` are required because channel map and status cannot be retrieved by run directly. These are used to determine the first timestamp available in the data path.
    - Designed in such a way to accommodate Subsystem init kwargs.

    Examples
    --------
    >>> get_query_times(..., start='2022-09-28 08:00:00', end='2022-09-28 09:30:00')
    {'timestamp': {'start': '20220928T080000Z', 'end': '20220928T093000Z'}}, '20220928T080000Z'

    >> get_query_times(..., runs=27)
    ({'run': ['r027']}, '20220928T091135Z')
    """
    # get time range query for DataLoader
    timerange = get_query_timerange(**kwargs)

    first_timestamp = ""
    # get first/last timestamp in case keyword is timestamp
    if "timestamp" in timerange:
        if "start" in timerange["timestamp"]:
            first_timestamp = timerange["timestamp"]["start"]
        if "end" in timerange["timestamp"]:
            last_timestamp = timerange["timestamp"]["end"]
        if (
            "start" not in timerange["timestamp"]
            and "end" not in timerange["timestamp"]
        ):
            first_timestamp = min(timerange["timestamp"])
            last_timestamp = max(timerange["timestamp"])
    # look in path to find first timestamp if keyword is run
    else:
        # currently only list of runs and not 'start' and 'end', so always list
        # find earliest/latest run, format rXXX
        first_run = min(timerange["run"])
        last_run = max(timerange["run"])

        # --- get dsp filelist of this run
        # if setup= keyword was used, get dict; otherwise kwargs is already the dict we need
        path_info = kwargs["dataset"] if "dataset" in kwargs else kwargs
        path = os.path.join(path_info["path"], path_info["version"])
        tiers, _ = get_tiers_pars_folders(path)

        first_glob_path = os.path.join(
            tiers[0],
            path_info["type"],
            path_info["period"],
            first_run,
        )
        last_glob_path = os.path.join(
            tiers[0],
            path_info["type"],
            path_info["period"],
            last_run,
        )

        first_glob_path = get_valid_path(first_glob_path)
        last_glob_path = get_valid_path(last_glob_path)

        # format to search /path_to_prod-ref[/vXX.XX]/generated/tier/dsp/phy/pXX/rXXX (version 'vXX.XX' might not be there).
        # NOTICE that we fixed the tier, otherwise it picks the last one it finds (eg tcm).
        # NOTICE that this is PERIOD SPECIFIC (unlikely we're gonna inspect two periods together, so we fix it)
        first_glob_path = os.path.join(
            first_glob_path,
            "*.lh5",
        )
        last_glob_path = os.path.join(
            last_glob_path,
            "*.lh5",
        )
        first_dsp_files = glob.glob(first_glob_path)
        last_dsp_files = glob.glob(last_glob_path)
        # find earliest
        first_dsp_files.sort()
        first_file = first_dsp_files[0]
        # find latest
        last_dsp_files.sort()
        last_file = last_dsp_files[-1]
        # extract timestamps
        first_timestamp = get_key(first_file)
        # last timestamp is not the key of last file: it's the last timestamp saved in the last file
        last_timestamp = get_last_timestamp(last_file)

    return timerange, first_timestamp, last_timestamp


def get_query_timerange(**kwargs):
    """
    Get DataLoader compatible time range.

    The function accepts either a `dataset` dictionary or keyword arguments.
    Only one type of time selection should be provided at a time.
    Designed in such a way to accommodate Subsystem init kwargs.

    Parameters
    ----------
    dataset : dict, optional
        Dictionary specifying the time selection. Choose one of the following (or enter kwargs separately):
            1. 'start' : str, 'end' : str
                Start and end datetime in the format `'YYYY-MM-DD hh:mm:ss'`.
            2. 'window' : str
                Time window relative to the current time, formatted as `'Xd Xh Xm'`
                for days, hours, and minutes.
            3. 'timestamps' : str or list of str
                Specific timestamps in `'YYYYMMDDThhmmssZ'` format.
            4. 'runs' : int or list of ints
                Run number(s), e.g., `10` corresponds to `'r010'`


    Examples
    --------
    >>> get_query_timerange(start='2022-09-28 08:00:00', end='2022-09-28 09:30:00')
    {'timestamp': {'start': '20220928T080000Z', 'end': '20220928T093000Z'}}

    >>> get_query_timerange(window='1d 5h 0m')
    {'timestamp': {'end': '20230220T114337Z', 'start': '20230219T064337Z'}}

    >>> get_query_timerange(timestamps=['20220928T080000Z', '20220928093000Z'])
    {'timestamp': ['20220928T080000Z', '20220928093000Z']}
    >>> get_query_timerange(timestamps='20220928T080000Z')
    {'timestamp': ['20220928T080000Z']}

    >> get_query_timerange(runs=[9,10])
    {'run': ['r009', 'r010']}
    >>> get_query_timerange(runs=10)
    {'run': ['r010']}

    >>> get_query_timerange(dataset={'start': '2022-09-28 08:00:00', 'end':'2022-09-28 09:30:00'})
    {'timestamp': {'start': '20220928T080000Z', 'end': '20220928T093000Z'}}
    """
    # if dataset= kwarg was provided, get the 'selection' part
    # otherwise kwargs itself is already the dict we need with start= & end=; or timestamp= etc
    user_selection = kwargs["dataset"] if "dataset" in kwargs else kwargs

    #  in these cases, DataLoader will be called with (timestamp >= ...) and (timestamp <= ...)
    if "start" in user_selection:
        time_range = {"timestamp": {}}
        for point in ["start", "end"]:
            try:
                time_range["timestamp"][point] = datetime.strptime(
                    user_selection[point], "%Y-%m-%d %H:%M:%S"
                ).strftime("%Y%m%dT%H%M%SZ")
            except ValueError:
                logger.error("\033[91mInvalid date format!\033[0m")
                return

    elif "window" in user_selection:
        time_range = {"timestamp": {}}
        time_range["timestamp"]["end"] = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        try:
            days, hours, minutes = re.split(r"d|h|m", user_selection["window"])[
                :-1
            ]  # -1 for trailing ''
        except ValueError:
            logger.error("\033[91mInvalid window format!\033[0m")
            return

        dt = timedelta(days=int(days), hours=int(hours), minutes=int(minutes))
        time_range["timestamp"]["start"] = (
            datetime.strptime(time_range["timestamp"]["end"], "%Y%m%dT%H%M%SZ") - dt
        ).strftime("%Y%m%dT%H%M%SZ")

    #  in these cases, DataLoader will be called with (timestamp/run == ...) or (timestamp/run == ...)
    elif "timestamps" in user_selection:
        time_range = {"timestamp": []}
        time_range["timestamp"] = (
            user_selection["timestamps"]
            if isinstance(user_selection["timestamps"], list)
            else [user_selection["timestamps"]]
        )

    elif "runs" in user_selection:
        runs = (
            user_selection["runs"]
            if isinstance(user_selection["runs"], list)
            else [user_selection["runs"]]
        )
        # check validity
        for run in runs:
            if not isinstance(run, int):
                logger.error("\033[91mInvalid run format!\033[0m")
                sys.exit()

        # format rXXX for DataLoader
        time_range = {"run": []}
        time_range["run"] = ["r" + str(run).zfill(3) for run in runs]

    else:
        logger.error(
            "\033[91mInvalid time selection. Choose among: runs, timestamps, window, start+end - try again!\033[0m"
        )
        return

    return time_range


def dataset_validity_check(data_info: dict):
    """
    Check the validity of the input dictionary and if it contains all required fields and keys to existing paths.

    This function is typically used in `Subsystem` and `SlowControl` classes to ensure that all necessary metadata
    for accessing data is present and correct.
    The function also checks that the provided `path` and the combined `path/version` exist on the filesystem.

    Parameters
    ----------
    data_info : dict
        Dictionary containing dataset metadata. Required keys:

            - 'experiment' : str
                Name of the experiment.
            - 'type' : str
                Type of dataset.
            - 'period' : str
                Period to inspect.
            - 'path' : str
                Path to the base dataset directory.
            - 'version' : str
                Processing version. Can be empty string if not needed.

    Examples
    --------
    >>> dataset_info = {
    ...     'experiment': 'L200',
    ...     'period': 'p03',
    ...     'type': 'phy',
    ...     'path': '/global/cfs/cdirs/m2676/data/lngs/l200/public/prodenv/prod-blind/',
    ...     'version': 'tmp-auto',
    ...     // ... additional time selection keys
    ... }
    >>> dataset_validity_check(dataset_info)
    # No output if all checks pass; errors otherwise
    """
    if "experiment" not in data_info:
        logger.error("\033[91mProvide experiment name!\033[0m")
        return

    if "type" not in data_info:
        logger.error("\033[91mProvide data type!\033[0m")
        return

    if "period" not in data_info:
        logger.error("\033[91mProvide period!\033[0m")
        return

    data_types = ["phy", "cal"]
    if not data_info["type"] in data_types:
        logger.error("\033[91mInvalid data type provided!\033[0m")
        return

    if "path" not in data_info:
        logger.error("\033[91mProvide path to data!\033[0m")
        return
    if not os.path.exists(data_info["path"]):
        logger.error("\033[91mThe data path you provided does not exist!\033[0m")
        return

    if "version" not in data_info:
        logger.error(
            '\033[91mProvide processing version! If not needed, just put an empty string, "".\033[0m'
        )
        return

    if not os.path.exists(os.path.join(data_info["path"], data_info["version"])):
        logger.error("\033[91mProvide valid processing version!\033[0m")
        return


# -------------------------------------------------------------------------
# Plotting related functions
# -------------------------------------------------------------------------


def check_scdb_settings(conf: dict) -> bool:
    """
    Validate the 'slow_control' entry in the config dictionary by checking if it contains a 'slow_control' section with a 'parameters' key. It ensures that the 'parameters' value is either a string or a list of strings. Always returns True if the configuration passes all checks. Exits the program otherwise.

    Parameters
    ----------
    conf : dict
        SC configuration dictionary.

    Examples
    --------
    >>> conf = {
    ...     'slow_control': {
    ...         'parameters': ['RREiT', 'ZUL_T_RR']
    ...     }
    ... }
    >>> check_scdb_settings(conf)
    True
    """
    # there is no "slow_control" key
    if "slow_control" not in conf.keys():
        logger.warning(
            "\033[93mThere is no 'slow_control' key in the config file. Try again if you want to retrieve slow control data.\033[0m"
        )
        sys.exit()
    # there is "slow_control" key, but ...
    else:
        # ... there is no "parameters" key
        if "parameters" not in conf["slow_control"].keys():
            logger.warning(
                "\033[93mThere is no 'parameters' key in config 'slow_control' entry. Try again if you want to retrieve slow control data.\033[0m"
            )
            sys.exit()
        # ... there is "parameters" key, but ...
        else:
            # ... it is not a string or a list (of strings)
            if not isinstance(
                conf["slow_control"]["parameters"], str
            ) and not isinstance(conf["slow_control"]["parameters"], list):
                logger.error(
                    "\033[91mSlow control parameters must be a string or a list of strings. Try again if you want to retrieve slow control data.\033[0m"
                )
                sys.exit()


def check_plot_settings(conf: dict) -> bool:
    from . import plot_styles, plotting

    options = {
        "plot_structure": plotting.PLOT_STRUCTURE.keys(),
        "plot_style": plot_styles.PLOT_STYLE.keys(),
    }

    if "subsystems" not in conf.keys():
        logger.error(
            "\033[91mThere is no 'subsystems' key in the config file. Try again if you want to plot data.\033[0m"
        )
        sys.exit()

    for subsys in conf["subsystems"]:
        for plot in conf["subsystems"][subsys]:
            # settings for this plot
            plot_settings = conf["subsystems"][subsys][plot]

            # ----------------------------------------------------------------------------------------------
            # general check
            # ----------------------------------------------------------------------------------------------
            # check if all necessary fields for param settings were provided
            for field in options:
                # when plot_structure is summary or you simply want to load QCs, plot_style is not needed...
                if plot_settings["parameters"] in ("exposure", "quality_cuts"):
                    continue

                # ...otherwise, it is required
                # if this field is not provided by user, tell them to provide it
                # (if optional to provided, will have been set with defaults before calling set_defaults())
                if field not in plot_settings:
                    logger.error(
                        f"\033[91mProvide {field} in plot settings of '{plot}' for {subsys}!\033[0m"
                    )
                    logger.error(
                        "\033[91mAvailable options: {}\033[0m".format(
                            ",".join(options[field])
                        )
                    )
                    return False

                # check if the provided option is valid
                opt = plot_settings[field]

                if opt not in options[field]:
                    logger.error(
                        f"\033[91mOption {opt} provided for {field} in plot settings of '{plot}' for {subsys} does not exist!\033[0m"
                    )
                    logger.error(
                        "\033[91mAvailable options: {}\033[0m".format(
                            ",".join(options[field])
                        )
                    )
                    return False

            # ----------------------------------------------------------------------------------------------
            # special checks
            # ----------------------------------------------------------------------------------------------

            # exposure check
            if plot_settings["parameters"] == "exposure" and (
                plot_settings["event_type"] not in ["pulser", "all"]
            ):
                logger.error(
                    "\033[91mPulser events are needed to calculate livetime/exposure; choose 'pulser' or 'all' event type\033[0m"
                )
                return False

            # ToDo: neater way to skip the whole loop but still do special checks; break? ugly...
            if plot_settings["parameters"] in ("exposure", "quality_cuts"):
                continue

            # other non-exposure checks

            # if vs time was provided, need time window
            if (
                plot_settings["plot_style"] == "vs time"
                and "time_window" not in plot_settings
            ):
                logger.error(
                    "\033[91mYou chose plot style 'vs time' and did not provide 'time_window'!\033[0m"
                )
                return False

    return True


def make_output_paths(config: dict, user_time_range: dict) -> str:
    """
    Get a dict and return a dict. The function defines output paths and create directories accordingly.

    To use when you want a specific output structure of the following type: [...]/prod-ref/{version}/generated/plt/hit/phy/{period}/{run}
    This does not work if you select more types (eg. both cal and phy) or timestamp intervals (but just runs).
    It can be used for run summary plots, eg during stable data taking.
    Note that monitoring plots are stored under the 'hit' subfolder to replicate the structure of the main prodenv.
    """
    logger.info("Setting up plotting...")

    if "output" not in config:
        logger.error(
            "\033[91mProvide output folder path in your config field 'output'!\033[0m"
        )
        return

    # general output path
    try:
        make_dir(config["output"])
    except Exception:
        logger.error(f"\033[91mCannot make output folder {config['output']}\033[0m")
        logger.error("\033[91mMaybe you don't have rights to create this path?\033[0m")
        return

    # create subfolders for the fixed path
    version_dir = os.path.join(config["output"], config["dataset"]["version"])
    generated_dir = os.path.join(version_dir, "generated")
    plt_dir = os.path.join(generated_dir, "plt")
    hit_dir = os.path.join(plt_dir, "hit")
    # 'phy' or 'cal' if one of the two is specified; if both are specified, store data in 'cal_phy/'
    if isinstance(config["dataset"]["type"], list):
        type_dir = os.path.join(hit_dir, "cal_phy")
    else:
        type_dir = os.path.join(hit_dir, config["dataset"]["type"])
    # period info
    period_dir = os.path.join(type_dir, config["dataset"]["period"]) + "/"

    # output subfolders
    make_dir(version_dir)
    make_dir(generated_dir)
    make_dir(plt_dir)
    make_dir(hit_dir)
    make_dir(type_dir)
    make_dir(period_dir)

    return period_dir


def make_dir(dir_path):
    """Check if directory exists, and if not, make it."""
    message = "Output directory " + dir_path
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
        message += " (created)"
    logger.info(message)


def get_multiple_run_id(user_time_range: dict) -> str:
    time_range = list(user_time_range.values())[0]
    name_time = "{}".format("_".join(time_range))
    return name_time


def get_time_name(user_time_range: dict) -> str:
    """
    Get a name for each available time selection.

    Parameters
    ----------
    user_time_range : dict
        Careful handling of folder name depending on the selected time range


    Examples
    --------
    >>> get_time_name({'timestamp': {'start': '20220928T080000Z', 'end': '20220928T093000Z'}})
    20220928T080000Z_20220928T093000Z

    >>> get_time_name({'timestamp': ['20230207T103123Z']})
    20230207T103123Z

    >>> get_time_name({'timestamp': ['20230207T103123Z', '20230207T141123Z', '20230207T083323Z']})
    20230207T083323Z_20230207T141123Z

    >>> get_time_name({'run': ['r010']})
    r010

    >>> get_time_name({'run': ['r010', 'r014']})
    r010_r014
    """
    name_time = ""
    if "timestamp" in user_time_range.keys():
        time_range = list(user_time_range.values())[0]
        if "start" in time_range:
            name_time += time_range["start"] + "_" + time_range["end"]
        else:
            if len(time_range) == 1:
                name_time += time_range[0]
            else:
                timestamp_range = [
                    datetime.strptime(string, "%Y%m%dT%H%M%SZ").timestamp()
                    for string in time_range
                ]
                min_idx = timestamp_range.index(min(timestamp_range))
                max_idx = timestamp_range.index(max(timestamp_range))
                name_time += time_range[min_idx] + "_" + time_range[max_idx]

    elif "run" in user_time_range.keys():
        name_time = get_multiple_run_id(user_time_range)

    else:
        logger.error("\033[91mInvalid time selection!\033[0m")
        return

    return name_time


def get_timestamp(filename: str):
    """Get the timestamp from a filename. For instance, if file='l200-p04-r000-phy-20230421T055556Z-tier_dsp.lh5', then it returns '20230421T055556Z'."""
    # Assumes that the timestamp is in the format YYYYMMDDTHHMMSSZ
    return filename.split("-")[-2]


def get_run_name(config: dict, user_time_range: dict) -> str:
    """Get the run ID given start/end timestamps. If the timestamps run over multiple run IDs, a list of runs is retrieved, out of which only the first element is returned."""
    # this is the root directory to search in the timestamps
    main_folder = os.path.join(
        config["dataset"]["path"], config["dataset"]["version"], "generated/tier"
    )

    # start/end timestamps of the selected time range of interest
    # if range was given, will have keywords "start" and "end"
    if "start" in user_time_range["timestamp"]:
        start_timestamp = user_time_range["timestamp"]["start"]
        end_timestamp = user_time_range["timestamp"]["end"]
    # if list of timestamps was given (may be not consecutive or in order), it's just a list
    else:
        start_timestamp = min(user_time_range["timestamp"])
        end_timestamp = max(user_time_range["timestamp"])

    run_list = []

    # start to look for timestamps inside subfolders
    def search_for_timestamp(folder):
        run_id = ""
        for idx, subfolder in enumerate(os.listdir(folder)):
            subfolder_path = os.path.join(folder, subfolder)
            if os.path.isdir(subfolder_path):
                files = sorted(glob.glob(os.path.join(subfolder_path, "*")))
                for i, file in enumerate(files):
                    if (
                        get_timestamp(files[i - 1])
                        <= start_timestamp
                        <= get_timestamp(file)
                    ) or (
                        get_timestamp(files[i - 1])
                        <= end_timestamp
                        <= get_timestamp(file)
                    ):
                        run_id = file.split("/")[-2]
                        # avoid duplicates
                        if run_id not in run_list:
                            run_list.append(run_id)

                if len(run_list) == 0:
                    search_for_timestamp(subfolder_path)
                if len(run_list) > 0 and idx == len(os.listdir(folder)) - 1:
                    break
        return

    search_for_timestamp(main_folder)

    if len(run_list) == 0:
        logger.error(
            "\033[91mThe selected timestamps were not find anywhere. Try again with another time range!\033[0m"
        )
        sys.exit()
    if len(run_list) > 1:
        return get_multiple_run_id(user_time_range)

    return run_list[0]


def get_all_plot_parameters(subsystem: str, config: dict):
    """Get list of all parameters needed for all plots for given subsystem."""
    version = config["dataset"]["version"]
    path = config["dataset"]["path"]
    # load hit QC and classifier flags
    possible_dirs = ["tier/hit", "tier_hit"]
    file_patterns = ["*-ICPC-hit_config.yaml", "*-ICPC-hit_config.json"]
    hit_config = None
    for subdir in possible_dirs:
        for pattern in file_patterns:
            filepath_pattern = os.path.join(
                path, version, "inputs/dataprod/config", subdir, pattern
            )
            files = glob.glob(filepath_pattern)
            if files:
                filepath = files[0]
                with open(filepath) as file:
                    if filepath.endswith(".yaml"):
                        hit_config = yaml.load(file, Loader=yaml.CLoader)
                    elif filepath.endswith(".json"):
                        hit_config = json.load(file)
                break
        if hit_config:
            break
    if not hit_config:
        logger.error(
            "No matching config files found in either 'tier/hit' or 'tier_hit'."
        )
        sys.exit()

    is_entries = [
        entry
        for entry in hit_config["outputs"]
        if entry.startswith("is_") and not entry.endswith("_classifier")
    ]
    is_classifiers = [
        entry
        for entry in hit_config["outputs"]
        if entry.startswith("is_") and entry.endswith("_classifier")
    ]

    all_parameters = []
    if subsystem in config["subsystems"]:
        for plot in config["subsystems"][subsystem]:
            parameters = config["subsystems"][subsystem][plot]["parameters"]
            if parameters not in ("quality_cuts"):
                if isinstance(parameters, str):
                    all_parameters.append(parameters)
                else:
                    all_parameters += parameters

            # check if event type asked needs a special parameter (eg K lines need energy)
            event_type = config["subsystems"][subsystem][plot]["event_type"]
            if event_type in SPECIAL_PARAMETERS:
                all_parameters += SPECIAL_PARAMETERS[event_type]

            # check if there is any cut to apply; if so, add it to the list of parameters to load
            if "cuts" in config["subsystems"][subsystem][plot]:
                cuts = config["subsystems"][subsystem][plot]["cuts"]
                # convert to list for convenience
                if isinstance(cuts, str):
                    cuts = [cuts]
                for cut in cuts:
                    if "~" in cut:
                        logger.error(
                            "\033[91mThe cut %s is not available at the moment. Exit here.\033[0m",
                            cut,
                        )
                        sys.exit()
                    all_parameters.append(cut)

            # check if we have to load individual QC and classifiers
            if config["subsystems"][subsystem][plot].get("qc_flags") is True:
                all_parameters.extend(is_entries)
            if config["subsystems"][subsystem][plot].get("qc_classifiers") is True:
                all_parameters.extend(is_classifiers)

    return all_parameters


def get_key(dsp_fname: str) -> str:
    """Extract key from lh5 filename."""
    return re.search(r"-\d{8}T\d{6}Z", dsp_fname).group(0)[1:]


def unix_timestamp_to_string(unix_timestamp):
    """Convert a Unix timestamp to a string in the format 'YYYYMMDDTHHMMSSZ' with the timezone indicating UTC+00."""
    utc_datetime = datetime.utcfromtimestamp(unix_timestamp)
    formatted_string = utc_datetime.strftime("%Y%m%dT%H%M%SZ")
    return formatted_string


def get_last_timestamp(fname: str) -> str:
    """Read a lh5 file and return the last timestamp saved in the file. This works only in case of a global trigger where the whole array is entirely recorded for a given timestamp."""
    # pick a random channel
    channels = lh5.ls(fname, "")
    tier = fname.split("-")[-1].replace(".lh5", "").replace("tier_", "")
    tier_map = {"psp": "dsp", "pht": "hit", "pet": "evt"}
    tier = tier_map.get(tier, tier)
    timestamp = None
    # pick the first channel that has a valid timestamp entry
    for ch in channels:
        try:
            # get array of timestamps stored in the lh5 file
            timestamp = lh5.read(f"{ch}/{tier}/timestamp", fname)
            break
        except (KeyError, FileNotFoundError):
            pass
    if timestamp is None:
        logger.error("\033[91mNo timestamps were found. Exit here.\033[0m")
        sys.exit()
    # get the last entry
    last_timestamp = timestamp[-1]
    # convert from UNIX tstamp to string tstmp of format YYYYMMDDTHHMMSSZ
    last_timestamp = unix_timestamp_to_string(last_timestamp)

    return last_timestamp


def bunch_dataset(config: dict, n_files=None):
    """Bunch the full datasets into smaller pieces, based on the number of files we want to inspect at each iteration.

    It works for "start+end", "runs" and "timestamps" in "dataset" present in the config file.
    """
    # --- get dsp filelist of this run
    path_info = config["dataset"]
    user_time_range = get_query_timerange(dataset=config["dataset"])
    run = (
        get_run_name(config, user_time_range)
        if "timestamp" in user_time_range.keys()
        else get_time_name(user_time_range)
    )
    # format to search /path_to_prod-ref[/vXX.XX]/generated/tier/dsp/phy/pXX/rXXX (version 'vXX.XX' might not be there).
    # NOTICE that we fixed the tier, otherwise it picks the last one it finds (eg tcm).
    # NOTICE that this is PERIOD SPECIFIC (unlikely we're gonna inspect two periods together, so we fix it)
    path = os.path.join(path_info["path"], path_info["version"])
    tiers, _ = get_tiers_pars_folders(path)
    path_to_files = os.path.join(
        tiers[0],  # path to dsp folder
        path_info["type"],
        path_info["period"],
        run,
        "*.lh5",
    )
    # get all dsp files
    dsp_files = glob.glob(path_to_files)
    if not dsp_files:
        dsp_files = glob.glob(path_to_files.replace("dsp", "psp"))
    dsp_files.sort()

    if "timestamp" in user_time_range.keys():
        if isinstance(user_time_range["timestamp"], list):
            # sort in crescent order
            user_time_range["timestamp"].sort()
            start_time = datetime.strptime(
                user_time_range["timestamp"][0], "%Y%m%dT%H%M%SZ"
            )
            end_time = datetime.strptime(
                user_time_range["timestamp"][-1], "%Y%m%dT%H%M%SZ"
            )

        else:
            start_time = datetime.strptime(
                user_time_range["timestamp"]["start"], "%Y%m%dT%H%M%SZ"
            )
            end_time = datetime.strptime(
                user_time_range["timestamp"]["end"], "%Y%m%dT%H%M%SZ"
            )

    if "run" in user_time_range.keys():
        timerange, start_tmstmp, end_tmstmp = get_query_times(dataset=config["dataset"])
        start_time = datetime.strptime(start_tmstmp, "%Y%m%dT%H%M%SZ")
        end_time = datetime.strptime(end_tmstmp, "%Y%m%dT%H%M%SZ")

    # filter files and keep the ones within the time range of interest
    filtered_files = []
    for dsp_file in dsp_files:
        # Extract the timestamp from the file name
        timestamp_str = dsp_file.split("-")[-2]
        file_timestamp = datetime.strptime(timestamp_str, "%Y%m%dT%H%M%SZ")
        # Check if the file timestamp is within the specified range
        if start_time <= file_timestamp <= end_time:
            filtered_files.append(dsp_file)

    filtered_files = [filtered_file.split("-")[-2] for filtered_file in filtered_files]
    filtered_files = [
        filtered_files[i : i + int(n_files)]
        for i in range(0, len(filtered_files), int(n_files))
    ]

    return filtered_files


def check_key_existence(hdf_path: str, key_to_load: str) -> bool:
    """Check if a specific key exists in the specified hdf file path."""
    try:
        with pd.HDFStore(hdf_path, mode="r") as store:
            if key_to_load in store.keys():
                return True
            else:
                logger.debug(f"Key '{key_to_load}' not found in {hdf_path}")
                return False
    except FileNotFoundError:
        logger.debug(f"HDF file '{hdf_path}' does not exist")
        return False
    except Exception as e:
        logger.debug(f"Error accessing HDF file '{hdf_path}': {e}")
        return False


# -------------------------------------------------------------------------
# Config file related functions (for building files)
# -------------------------------------------------------------------------


def add_config_entries(
    config: dict,
    file_keys: str,
    prod_path: str,
    prod_config: dict,
) -> dict:
    """Add missing information (output, dataset) to the configuration file. This function is generally used during automathic data production, where the initiali config file has only the 'subsystem' entry."""
    # check if there is an output folder specified in the config file
    if "output" not in config.keys():
        logger.error(
            "\033[91mThe config file is missing the 'output' key. Add it and try again!\033[0m"
        )
        sys.exit()
    # check if there is the saving option specified in the config file
    if "saving" not in config.keys():
        logger.error(
            "\033[91mThe config file is missing the 'saving' key. Add it and try again!\033[0m"
        )
        sys.exit()

    # Get the keys
    with open(file_keys) as f:
        keys = f.readlines()
    # Remove newline characters from each line using strip()
    keys = [key.strip() for key in keys]
    # get only keys of timestamps
    timestamp = [key.split("-")[-1] for key in keys]

    # Get the experiment
    experiment = (keys[0].split("-"))[0].upper()

    # Get the period
    period = (keys[0].split("-"))[1]

    # Get the run
    run = (keys[0].split("-"))[2]

    # Get the version
    if "dataset" in config.keys():
        if "version" in config["dataset"].keys():
            version = config["dataset"]["version"]
        else:
            # case of rsync when inspecting temp files to plot for the dashboard
            if prod_path == "":
                version = ""
            # prod-ref version where the version is specified
            else:
                clean_path = prod_path.rstrip("/")
                version = os.path.basename(clean_path)
        if "type" in config["dataset"].keys():
            type = config["dataset"]["type"]
        else:
            logger.error("\033[91mYou need to provide data type! Try again.\033[0m")
            sys.exit()
        if "path" in config["dataset"].keys():
            path = config["dataset"]["path"]
        else:
            logger.error(
                "\033[91mYou need to provide path to lh5 files! Try again.\033[0m"
            )
            sys.exit()
    else:
        # get phy/cal lists
        phy_keys = [key for key in keys if "phy" in key]
        cal_keys = [key for key in keys if "cal" in key]
        if len(phy_keys) == 0 and len(cal_keys) == 0:
            logger.error("\033[91mNo keys to load. Try again.\033[0m")
            sys.exit()
        if len(phy_keys) != 0 and len(cal_keys) == 0:
            type = "phy"
        if len(phy_keys) == 0 and len(cal_keys) != 0:
            type = "cal"
            logger.error("\033[91mcal is still under development! Try again.\033[0m")
            sys.exit()
        if len(phy_keys) != 0 and len(cal_keys) != 0:
            type = ["cal", "phy"]
            logger.error(
                "\033[91mBoth cal and phy are still under development! Try again.\033[0m"
            )
            sys.exit()
        # get the production path
        base_path = prod_path.split("prod-ref")[0]
        path = os.path.join(base_path, "prod-ref")

    # create the dataset dictionary
    dataset_dict = {
        "experiment": experiment,
        "period": period,
        "version": version,
        "path": path,
        "type": type,
        "run": run,
        "timestamps": timestamp,
    }

    more_info = {"dataset": dataset_dict}

    # 'saving' and 'subsystem' info must be already there
    config.update(more_info)

    # let's make a check that everything we need is inside the config, otherwise exit
    if not all(key in config for key in ["output", "dataset", "saving", "subsystems"]):
        logger.error(
            '\033[91mThere are missing entries among ["output", "dataset", "saving", "subsystems"] in the config file (found keys: %s). Try again and check you start with "output" and "dataset" info!\033[0m',
            config.keys(),
        )
        sys.exit()

    return config


def get_output_plot_path(plt_path: str, extension: str) -> str:
    """
    Given a path to the plt directory, generate a corresponding output path in the tmp/mtg/ directory.

    Parameters
    ----------
        plt_path : str
            Original plot path (e.g. from 'plt/hit/phy/').
        extension : str
            Extension of the file to save (e.g. 'pdf' or 'log').
    """
    filename = os.path.basename(plt_path)
    save_path = plt_path.replace("plt/hit/phy/", "tmp/mtg/").rsplit("/", 1)[0] + "/"
    os.makedirs(save_path, exist_ok=True)
    plt_file = os.path.join(save_path, f"{filename}.{extension}")

    return plt_file


# -------------------------------------------------------------------------
# Other functions
# -------------------------------------------------------------------------


def load_config(config_file: dict | str):
    """
    Load a configuration from a dictionary, JSON string, or YAML file.

    This function supports three input types:

    - A dictionary, which is returned as-is.
    - A JSON string, which is parsed into a dictionary.
    - A path to a YAML (.yaml/.yml) file, which is read and parsed.

    Parameters
    ----------
    config_file : dict or str
        The configuration input
    """
    if isinstance(config_file, dict):
        return config_file

    if isinstance(config_file, str):
        # Looks like a file path and exists
        if os.path.isfile(config_file) and config_file.endswith((".yaml", ".yml")):
            with open(config_file) as f:
                return yaml.load(f, Loader=yaml.CLoader)
        else:
            # Try to parse as a JSON string
            try:
                return json.loads(config_file)
            except json.JSONDecodeError:
                raise ValueError(
                    "Provided string is not a valid JSON or YAML file path."
                )

    raise TypeError(
        "config_file must be a dict, a JSON string, or a path to a .yaml file."
    )


def get_livetime(tot_livetime: float):
    """
    Get the livetime in a human readable format, starting from livetime in seconds.

    Parameters
    ----------
    tot_livetime : float

        - If tot_livetime is more than 0.1 yr, convert it to years.
        - If tot_livetime is less than 0.1 yr but more than 1 day, convert it to days.
        - If tot_livetime is less than 1 day but more than 1 hour, convert it to hours.
        - If tot_livetime is less than 1 hour but more than 1 minute, convert it to minutes.
    """
    if tot_livetime > 60 * 60 * 24 * 365.25:
        tot_livetime = tot_livetime / 60 / 60 / 24 / 365.25
        unit = " yr"
    elif tot_livetime > 60 * 60 * 24:
        tot_livetime = tot_livetime / 60 / 60 / 24
        unit = " days"
    elif tot_livetime > 60 * 60:
        tot_livetime = tot_livetime / 60 / 60
        unit = " hrs"
    elif tot_livetime > 60:
        tot_livetime = tot_livetime / 60
        unit = " min"
    else:
        unit = " sec"
    logger.info(f"Total livetime: {tot_livetime:.2f}{unit}")

    return tot_livetime, unit


def check_empty_df(df) -> bool:
    """Check if df (DataFrame | analysis_data.AnalysisData) exists and is not empty."""
    # the dataframe is of type DataFrame
    if isinstance(df, DataFrame):
        return df.empty
    # the dataframe is of type analysis_data.AnalysisData
    else:
        return df.data.empty


def convert_to_camel_case(string: str, char: str) -> str:
    """Remove a character from a string and capitalize all initial letters."""
    # Split the string by underscores
    words = string.split(char)
    # Capitalize the initial letters of each word
    words = [word.capitalize() for word in words]
    # Join the words back together without any separator
    camel_case_string = "".join(words)

    return camel_case_string


def get_output_path(config: dict):
    """Get output path provided a 'dataset' from the config file. The path will be used to save and store pdfs/hdf/etc files."""
    try:
        data_types = (
            [config["dataset"]["type"]]
            if isinstance(config["dataset"]["type"], str)
            else config["dataset"]["type"]
        )

        plt_basename = "{}-{}-".format(
            config["dataset"]["experiment"].lower(),
            config["dataset"]["period"],
        )
    except (KeyError, TypeError):
        logger.error(
            "\033[91mSomething is missing or wrong in your 'dataset' field of the config.\033[0m"
        )
        sys.exit()

    user_time_range = get_query_timerange(dataset=config["dataset"])
    # will be returned as None if something is wrong, and print an error message
    if not user_time_range:
        return

    # create output folders for plots
    period_dir = make_output_paths(config, user_time_range)
    # get correct time info for subfolder's name
    name_time = (
        get_run_name(config, user_time_range)
        if "timestamp" in user_time_range.keys()
        else get_time_name(user_time_range)
    )
    output_paths = period_dir + name_time + "/"
    make_dir(output_paths)
    if not output_paths:
        logger.info("%s does not exist!", output_paths)
        return

    # we don't care here about the time keyword timestamp/run -> just get the value
    plt_basename += name_time
    out_path = output_paths + plt_basename
    out_path += "-{}".format("_".join(data_types))

    return out_path


def send_email_alert(app_password: str, recipients: list, text_file_path: str):
    """Send automatic emails with alert messages.

    Parameters
    ----------
    app_password: str
        String password to send mails from legend.data.monitoring@gmail.com
    recipients: list
        List of email addresses to send the alert emails
    text_file_path: str
        String path to the .txt file containing the message to send via email
    """
    sender = "legend.data.monitoring@gmail.com"
    subject = "Automatic message - DATA MONITORING ALARM!"
    try:
        with open(text_file_path) as f:
            text = f.read()
    except FileNotFoundError:
        logger.info("Error: File not found: %s", text_file_path)
        return

    # Create email message
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)

    body = MIMEText(text, "plain")
    msg.attach(body)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.starttls()
            smtp.login(sender, app_password)
            smtp.sendmail(sender, recipients, msg.as_string())
            logger.info("Successfully sent emails from %s", sender)
    except smtplib.SMTPException as e:
        logger.info("Error: unable to send email: %s", e)


def check_threshold(
    data_series: pd.Series,
    pswd_email: str | None,
    last_checked: float | None | str,
    t0: list,
    pars_data: dict,
    threshold: list,
    period: str,
    current_run: str | int,
    channel_name: str,
    string: str | int,
    email_message: list,
    parameter: str,
):
    """Check if a given parameter is over threshold and update the email message list.

    Parameters
    ----------
    data_series : pd.Series
        Series of gain differences indexed by timestamp.
    pswd_email : str or None
        Email password to trigger alert (used as a flag).
    last_checked : float
        Timestamp (in seconds since epoch) of last check.
    t0 : list of pd.Timestamp
        List of start times for time windows.
    pars_data : dict
        Dictionary containing parameters including 'res' for thresholds.
    threshold: list
        Threshold (int or float).
    period : str
        Period string (e.g., "P03").
    current_run : str or int
        Identifier of the current run.
    channel_name : str
        Name of the channel.
    string : str or int
        String identifier for the channel.
    email_message : list
        List of messages to be sent via email.
    parameter : str
        Parameter name under inspection.
    """
    if (
        data_series is None
        or pswd_email is None
        or last_checked == "None"
        or (threshold[0] is None and threshold[1] is None)
    ):
        return email_message

    timestamps = data_series.index
    cutoff = pd.to_datetime(float(last_checked), unit="s", utc=True)
    filtered_series = data_series[data_series.index > cutoff]

    if filtered_series.empty:
        return email_message

    time_range_start = pd.Timestamp(t0[0])
    time_range_end = time_range_start + pd.Timedelta(days=7)

    # ensure UTC awareness
    if time_range_start.tzinfo is None:
        time_range_start = time_range_start.tz_localize("UTC")
    else:
        time_range_start = time_range_start.tz_convert("UTC")

    if time_range_end.tzinfo is None:
        time_range_end = time_range_end.tz_localize("UTC")
    else:
        time_range_end = time_range_end.tz_convert("UTC")

    # filter by time range
    mask_time_range = (timestamps >= time_range_start) & (timestamps < time_range_end)
    filtered_timestamps = timestamps[mask_time_range]
    data_series_in_range = data_series[mask_time_range]

    low, high = threshold  # threshold = [low, high]
    mask = pd.Series(True, index=data_series_in_range.index)  # start with all True

    if low is not None:
        mask &= data_series_in_range < low
    if high is not None:
        mask &= data_series_in_range > high

    over_threshold_timestamps = filtered_timestamps[mask]

    if not over_threshold_timestamps.empty:
        if len(email_message) == 0:
            email_message = [
                f"ALERT: Data monitoring threshold exceeded in {period}-{current_run}.\n"
            ]
        email_message.append(
            f"- {parameter} over threshold for {channel_name} (string {string}) "
            f"for {len(over_threshold_timestamps)} times"
        )

    return email_message


def get_map_dict(data_analysis: DataFrame):
    """
    Map string location and geds position for plotting values vs chs.

    Parameters
    ----------
    data_analysis
        DataFrame with geds data information, in particular 'location' and 'position'
    """
    map_dict = {}
    offset = 0
    for string in sorted(data_analysis["location"].unique()):
        subset = data_analysis[data_analysis["location"] == string]
        positions = sorted(subset["position"].unique())
        map_dict[str(string)] = {}
        for i, position in enumerate(positions):
            map_dict[str(string)][str(position)] = offset + i
        offset += len(positions)

    return map_dict


def get_tiers_pars_folders(path: str):
    """
    Get the absolute path to different tier and par folders.

    Parameters
    ----------
    path : str
        Absolute path to the processed data for a specific version, eg path='/global/cfs/cdirs/m2676/data/lngs/l200/public/prodenv/prod-blind/ref-v2.1.5/'.
    """
    # config file with info on all tier folder
    try:
        with open(os.path.join(path, "config.json")) as f:
            config_proc = yaml.load(f, Loader=yaml.CLoader)
    except FileNotFoundError:
        with open(os.path.join(path, "dataflow-config.yaml")) as f:
            config_proc = yaml.load(f, Loader=yaml.CLoader)

    def clean_path(key, path, setup_paths):
        return os.path.join(path, setup_paths[key].replace("$_/", ""))

    try:
        setup_paths = config_proc["setups"]["l200"]["paths"]
    except KeyError:
        setup_paths = config_proc["paths"]

    # tier paths
    tier_keys = [
        "tier_dsp",
        "tier_psp",
        "tier_hit",
        "tier_pht",
        "tier_raw",
        "tier_evt",
        "tier_pet",
    ]
    tiers = [clean_path(key, path, setup_paths) for key in tier_keys]

    # parameter paths
    par_keys = ["par_dsp", "par_psp", "par_hit", "par_pht"]
    pars = [clean_path(key, path, setup_paths) for key in par_keys]

    return tiers, pars


def get_status_map(path: str, version: str, first_timestamp: str, datatype: str):
    """Return the correct status map, either reading a .json or .yaml file."""
    try:
        map_file = os.path.join(path, version, "inputs/dataprod/config")
        full_status_map = JsonDB(map_file).on(
            timestamp=first_timestamp, system=datatype
        )["analysis"]
    except (KeyError, TypeError):
        # fallback if "analysis" key doesn't exist and structure has changed
        map_file = os.path.join(path, version, "inputs/datasets/statuses")
        full_status_map = JsonDB(map_file).on(
            timestamp=first_timestamp, system=datatype
        )

    return full_status_map


# -------------------------------------------------------------------------
# Build runinfo file with livetime info
# -------------------------------------------------------------------------
def update_runinfo(run_info, period, run, data_type, my_global_path):
    files = os.listdir(my_global_path)
    files = [
        os.path.join(my_global_path, f) for f in files if f"{data_type}-geds.hdf" in f
    ]

    with open("settings/ignore-keys.yaml") as f:
        timestamps_file = yaml.load(f, Loader=yaml.CLoader)[period]
    start_timestamps = timestamps_file["start_keys"]
    end_timestamps = timestamps_file["stop_keys"]

    if files == []:
        return run_info
    files = files[0]

    with h5py.File(files, "r") as f:
        keys = sorted(list(f.keys()))
    my_key = [
        k for k in keys if "IsPulser" in k and "info" not in k and "_pulser" not in k
    ]

    tot_livetime = None
    if my_key is not []:
        my_hdf_file = pd.read_hdf(files, key=my_key[0])

        # filter the hdf file
        for ki, kf in zip(start_timestamps, end_timestamps):
            isolated_ki = pd.to_datetime(ki, format="%Y%m%dT%H%M%S%z")
            isolated_kf = pd.to_datetime(kf, format="%Y%m%dT%H%M%S%z")
            my_hdf_file = my_hdf_file[
                (my_hdf_file.index < isolated_ki) | (my_hdf_file.index >= isolated_kf)
            ]

        no_pulser = my_hdf_file.shape[0]
        tot_livetime = no_pulser * 20  # already in seconds

    if period in run_info.keys():
        if run in run_info[period].keys():
            if data_type in run_info[period][run].keys():
                run_info[period][run][data_type].update({"livetime_in_s": tot_livetime})

    return run_info


def pulser_from_evt_or_mtg(my_dir, period, run, output, run_info):
    """Try to load EVT tier; if not found, attempt to update run info from monitoring path."""
    evt_files = os.path.join(my_dir, f"l200-{period}-{run}-phy-tier_pet.lh5")
    # load from monitoring files if the pet files were not processed
    if not os.path.isfile(evt_files):
        mtg_path = os.path.join(output, f"generated/plt/phy/{period}/{run}/")
        if not os.path.isdir(mtg_path):
            return run_info
        run_info = update_runinfo(run_info, period, run, "phy", mtg_path)
        return run_info


def build_runinfo(path: str, version: str, output: str):
    """Build dictionary with main run information (start key, phy livetime in seconds) for multiple data types (phy, cal, fft, bkg, pzc, pul)."""
    periods = []
    runs = []

    possible_dirs = ["inputs/dataprod", "inputs/datasets"]
    file_patterns = ["runinfo.yaml", "*runinfo.json"]
    run_info = None
    for subdir in possible_dirs:
        for pattern in file_patterns:
            filepath_pattern = os.path.join(path, version, subdir, pattern)
            files = glob.glob(filepath_pattern)
            if files:
                filepath = files[0]
                with open(filepath) as file:
                    if filepath.endswith(".yaml"):
                        run_info = yaml.load(file, Loader=yaml.CLoader)
                    elif filepath.endswith(".json"):
                        run_info = json.load(file)
                break
        if run_info:
            break

    raw_paths = [
        os.path.join(path, "ref-raw/generated/tier/raw"),
        os.path.join(path, "tmp-p14-raw/generated/tier/raw"),
    ]

    # collect starting and ending timestamps
    for raw_path in raw_paths:
        data_types = sorted(os.listdir(raw_path))
        data_types = sorted(data_types, key=lambda x: (x != "phy", x))
        for data_type in data_types:  # cal | fft | bkg | phy | pul | pzc
            data_type_path = os.path.join(raw_path, data_type)
            if not os.listdir(data_type_path):
                continue

            for period in sorted(os.listdir(data_type_path)):  # p03 | p04 | ...
                if period in ["p01", "p02"]:
                    continue
                period_path = os.path.join(raw_path, data_type_path, period)
                if not os.listdir(period_path):
                    logger.warning(
                        "\033[93mThere are no files under the path %s\033[0m",
                        period_path,
                    )
                    continue

                period_runs = []
                for run in sorted(os.listdir(period_path)):  # r000 | r001 | ...
                    period_runs.append(run)

                    global_path = os.path.join(
                        raw_path, data_type_path, period_path, run
                    )
                    if not os.listdir(global_path):
                        logger.warning(
                            "\033[93mThere are no files under the path %s\033[0m",
                            global_path,
                        )
                        continue

                    files = sorted(os.listdir(global_path))
                    files_global_path = sorted(
                        [os.path.join(global_path, f) for f in files]
                    )
                    filtered = [
                        f for f in files_global_path if f.endswith((".orca", ".lh5"))
                    ]
                    first_file = filtered[0]
                    first_timestamp = (
                        (first_file.split("-")[-1]).split(".orca")[0]
                        if "orca" in first_file
                        else (first_file.split("/")[-1]).split("-")[4]
                    )

                    if period in run_info.keys():
                        if run in run_info[period].keys():
                            if data_type in run_info[period][run].keys():
                                run_info[period][run][data_type].update(
                                    {"start_key": first_timestamp}
                                )
                            else:
                                run_info[period][run].update(
                                    {data_type: {"start_key": first_timestamp}}
                                )
                        else:
                            run_info[period].update(
                                {run: {data_type: {"start_key": first_timestamp}}}
                            )
                    else:
                        run_info.update(
                            {period: {run: {data_type: {"start_key": first_timestamp}}}}
                        )

                if period not in periods:
                    periods.append(period)
                    runs.append(period_runs)

    # evaluate and save livetime from pulser events
    data_type = "phy"
    for idx_p, period in enumerate(periods):
        if period in ["p01", "p02"]:
            continue

        for run in runs[idx_p]:
            versions = [version] if version == "tmp-auto" else ["tmp-auto", version]

            for v in versions:
                tiers, _ = get_tiers_pars_folders(os.path.join(path, v))
                my_dir = tiers[5] if os.path.isdir(tiers[5]) else tiers[6]
                my_dir = os.path.join(my_dir, "phy")

                if v != "tmp-auto":
                    run_info = pulser_from_evt_or_mtg(
                        my_dir, period, run, output, run_info
                    )

                if v == "tmp-auto":
                    evt_path = os.path.join(my_dir, period, run)
                    if not os.path.isdir(evt_path):
                        continue
                    evt_files = os.listdir(evt_path)
                    evt_files = [
                        os.path.join(my_dir, period, run, f) for f in evt_files
                    ]

                data = lh5.read("evt/coincident/puls", evt_files)
                df_coincident = pd.DataFrame(data, columns=["puls"])
                df = pd.concat([df_coincident], axis=1)
                is_pulser = df["puls"]

                if not is_pulser.any():
                    run_info = pulser_from_evt_or_mtg(
                        my_dir, period, run, output, run_info
                    )
                else:
                    df = df[is_pulser]
                    no_pulser = len(df)
                    tot_livetime = no_pulser * 20

                    if period in run_info.keys():
                        if run in run_info[period].keys():
                            if data_type in run_info[period][run].keys():
                                run_info[period][run][data_type].update(
                                    {"livetime_in_s": tot_livetime}
                                )

    with open(os.path.join(output, "runinfo.yaml"), "w") as fp:
        yaml.dump(run_info, fp, default_flow_style=False, sort_keys=False)


# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------


def read_json_or_yaml(file_path: str):
    """
    Open either a JSON/YAML file, if not raise an error and exit.

    Parameters
    ----------
    file_path : str
        Path to the JSON/YAML file to read.
    """
    with open(file_path) as f:
        if file_path.endswith((".yaml", ".yml")):
            data_dict = yaml.load(f, Loader=yaml.CLoader)
        elif file_path.endswith(".json"):
            data_dict = json.load(f)
        else:
            logger.error(
                "\033[91mUnsupported file format: expected .json or .yaml/.yml. Exit here\033[0m"
            )
            sys.exit()

    return data_dict


def retrieve_json_or_yaml(base_path: str, filename: str):
    """Return either a yaml or a json file for the specified file looking at the existing available extension."""
    yaml_path = os.path.join(base_path, f"{filename}.yaml")
    json_path = os.path.join(base_path, f"{filename}.json")

    if os.path.isfile(yaml_path):
        path = yaml_path
    elif os.path.isfile(json_path):
        path = json_path
    else:
        logger.error(
            "\033[91mNo file found for %s in YAML or JSON format\033[0m", filename
        )
        sys.exit()

    return path


def deep_get(d, keys, default=None):
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d


def none_to_nan(data: list):
    """Convert None elements into nan values for an input list."""
    return [np.nan if v is None else v for v in data]
