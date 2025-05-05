import sys
from datetime import datetime, timezone
from typing import Tuple

import pandas as pd
from legendmeta import LegendSlowControlDB
from pandas import DataFrame

from . import utils

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SLOW CONTROL LOADING/PLOTTING FUNCTIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class SlowControl:
    """
    Object containing Slow Control database information for a data subselected based on given criteria.

    parameter [str] : diode_vmon | diode_imon | PT114 | PT115 | PT118 | PT202 | PT205 | PT208 | LT01 | RREiT | RRNTe | RRSTe | ZUL_T_RR | DaqLeft-Temp1 | DaqLeft-Temp2 | DaqRight-Temp1 | DaqRight-Temp2

    Options for kwargs

    dataset=
        dict with the following keys:
            - 'experiment' [str]: 'L60' or 'L200'
            - 'period' [str]: period format pXX
            - 'path' [str]: path to prod-ref folder (before version)
            - 'version' [str]: version of pygama data processing format vXX.XX
            - 'type' [str]: 'phy' or 'cal'
            - the following key(s) depending in time selection
                1. 'start' : <start datetime>, 'end': <end datetime> where <datetime> input is of format 'YYYY-MM-DD hh:mm:ss'
                2. 'window'[str]: time window in the past from current time point, format: 'Xd Xh Xm' for days, hours, minutes
                2. 'timestamps': str or list of str in format 'YYYYMMDDThhmmssZ'
                3. 'runs': int or list of ints for run number(s)  e.g. 10 for r010
    Or input kwargs separately experiment=, period=, path=, version=, type=; start=&end=, (or window= - ???), or timestamps=, or runs=
    """

    def __init__(self, parameter: str, port: int, pswd: str, **kwargs):
        # if setup= kwarg was provided, get dict provided
        # otherwise kwargs is itself already the dict we need with experiment= and period=
        data_info = kwargs["dataset"] if "dataset" in kwargs else kwargs

        # validity check of kwarg
        utils.dataset_validity_check(data_info)

        # needed to know for making 'if' statement over different experiments/periods
        self.experiment = data_info["experiment"]
        self.period = data_info["period"]
        # need to remember for channel status query
        # ! now needs to be single !
        self.datatype = data_info["type"]
        # need to remember for DataLoader config
        self.path = data_info["path"]
        self.version = data_info["version"]

        # load info from settings/SC-params.json
        self.parameter = parameter
        self.sc_parameters = utils.SC_PARAMETERS
        self.data = pd.DataFrame()
        self.scdb = LegendSlowControlDB()
        self.scdb.connect(port=port, password=pswd)

        # check if parameter is within the one listed in settings/SC-params.json
        if parameter not in self.sc_parameters["SC_DB_params"].keys():
            utils.logger.error(
                f"\033[91mThe parameter '{self.parameter}' is not present in 'settings/SC-params.json'. Try again with another parameter or update the json file!\033[0m"
            )
            return

        (
            self.timerange,
            self.first_timestamp,
            self.last_timestamp,
        ) = utils.get_query_times(**kwargs)

        # None will be returned if something went wrong
        if not self.timerange:
            utils.logger.error("\033[91m%s\033[0m", self.get_data.__doc__)
            return

        # -------------------------------------------------------------------------
        self.data = self.get_sc_param()

    def get_sc_param(self):
        """Load the corresponding table from SC database for the process of interest and apply already the flags for the parameter under study."""
        # getting the process and flags of interest from 'settings/SC-params.json' for the provided parameter
        table_param = self.sc_parameters["SC_DB_params"][self.parameter]["table"]
        flags_param = self.sc_parameters["SC_DB_params"][self.parameter]["flags"]

        # check if the selected table is present in the SC database. If not, arise an error and exit
        if table_param not in self.scdb.get_tables():
            utils.logger.error(
                "\033[91mThis is not present in the SC database! Try again.\033[0m"
            )
            sys.exit()

        # get the dataframe for the process of interest
        utils.logger.debug(
            f"... getting the dataframe for '{table_param}' in the time range of interest\n"
        )
        # SQL query to filter the dataframe based on the time range
        query = f"SELECT * FROM {table_param} WHERE tstamp >= '{self.first_timestamp}' AND tstamp <= '{self.last_timestamp}'"
        get_table_df = self.scdb.dataframe(query)

        # remove unnecessary columns (necessary when retrieving diode parameters)
        # note: there will be a 'status' column such that ON=1 and OFF=0 - right now we are keeping every detector, without removing the OFF ones as we usually do for geds
        if "vmon" in self.parameter and "imon" in list(get_table_df.columns):
            get_table_df = get_table_df.drop(columns="imon")
            # rename the column of interest to 'value' to be consistent with other parameter dataframes
            get_table_df = get_table_df.rename(columns={"vmon": "value"})
        elif "imon" in self.parameter and "vmon" in list(get_table_df.columns):
            get_table_df = get_table_df.drop(columns="vmon")
            get_table_df = get_table_df.rename(columns={"imon": "value"})
        # in case of geds parameters, add the info about the channel name and channel id (right now, there is only crate&slot info)
        else:
            get_table_df = include_more_diode_info(get_table_df, self.scdb)

        # order by timestamp (not automatically done)
        get_table_df = get_table_df.sort_values(by="tstamp")

        # let's apply the flags for keeping only the parameter of interest
        utils.logger.debug(
            f"... applying flags to get the parameter '{self.parameter}'"
        )
        get_table_df = apply_flags(get_table_df, self.sc_parameters, flags_param)

        # get units and lower/upper limits for the parameter of interest
        if "diode" not in self.parameter:
            unit, lower_lim, upper_lim = get_plotting_info(
                self.parameter,
                self.sc_parameters,
                self.first_timestamp,
                self.last_timestamp,
                self.scdb,
            )
        else:
            lower_lim = upper_lim = (
                None  # there are just 'set values', no actual thresholds
            )
            if "vmon" in self.parameter:
                unit = "V"
            elif "imon" in self.parameter:
                unit = "\u03bcA"
            else:
                unit = None

        # append unit, lower_lim, upper_lim to the dataframe
        get_table_df["unit"] = unit
        get_table_df["lower_lim"] = lower_lim
        get_table_df["upper_lim"] = upper_lim

        # fix time column
        get_table_df["tstamp"] = pd.to_datetime(get_table_df["tstamp"], utc=True)
        # fix value column
        get_table_df["value"] = pd.to_numeric(
            get_table_df["value"], errors="coerce"
        )  # handle errors as NaN

        # remove unnecessary columns
        remove_cols = ["rack", "group", "sensor", "name", "almask"]
        for col in remove_cols:
            if col in list(get_table_df.columns):
                get_table_df = get_table_df.drop(columns={col})

        get_table_df = get_table_df.reset_index(drop=True)

        utils.logger.debug(
            "... final dataframe (after flagging the events):\n%s", get_table_df
        )

        return get_table_df


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Other functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def get_plotting_info(
    parameter: str,
    sc_parameters: dict,
    first_tstmp: str,
    last_tstmp: str,
    scdb: LegendSlowControlDB,
) -> Tuple[str, float, float]:
    """Return units and low/high limits of a given parameter."""
    table_param = sc_parameters["SC_DB_params"][parameter]["table"]
    flags_param = sc_parameters["SC_DB_params"][parameter]["flags"]

    # get info dataframe of the corresponding process under study (do I need to specify the param????)
    get_table_info = scdb.dataframe(table_param.replace("snap", "info"))

    # let's apply the flags for keeping only the parameter of interest
    get_table_info = apply_flags(get_table_info, sc_parameters, flags_param)
    utils.logger.debug(
        "... units and thresholds will be retrieved from the following object:\n%s",
        get_table_info,
    )

    # Convert first_tstmp and last_tstmp to datetime objects in the UTC timezone
    first_tstmp = datetime.strptime(first_tstmp, "%Y%m%dT%H%M%SZ").replace(
        tzinfo=timezone.utc
    )
    last_tstmp = datetime.strptime(last_tstmp, "%Y%m%dT%H%M%SZ").replace(
        tzinfo=timezone.utc
    )

    # Filter the DataFrame based on the time interval, starting to look from the latest entry ('reversed(...)')
    times = list(get_table_info["tstamp"].unique())

    for time in reversed(times):
        if first_tstmp < time < last_tstmp:
            unit = list(get_table_info["unit"].unique())[0]
            lower_lim = upper_lim = False
            utils.logger.warning(
                f"\033[93mParameter {parameter} has no valid range in the time period you selected. Upper and lower thresholds are set to False, while units={unit}\033[0m"
            )
            return unit, lower_lim, upper_lim

        if time < first_tstmp and time < last_tstmp:
            unit = list(
                get_table_info[get_table_info["tstamp"] == time]["unit"].unique()
            )[0]
            lower_lim = get_table_info[get_table_info["tstamp"] == time][
                "ltol"
            ].tolist()[-1]
            upper_lim = get_table_info[get_table_info["tstamp"] == time][
                "utol"
            ].tolist()[-1]
            utils.logger.debug(
                f"... parameter {parameter} must be within [{lower_lim};{upper_lim}] {unit}"
            )
            return unit, lower_lim, upper_lim

        if time > first_tstmp and time > last_tstmp:
            if time == times[0]:
                utils.logger.error(
                    "\033[91mYou're travelling too far in the past, there were no SC data in the time period you selected. Try again!\033[0m"
                )
                sys.exit()

    return unit, lower_lim, upper_lim


def apply_flags(df: DataFrame, sc_parameters: dict, flags_param: list) -> DataFrame:
    """Apply the flags read from 'settings/SC-params.json' to the input dataframe."""
    for flag in flags_param:
        column = sc_parameters["expressions"][flag]["column"]
        entry = sc_parameters["expressions"][flag]["entry"]
        df = df[df[column] == entry]

    # check if the dataframe is empty, if so, skip this plot
    if utils.is_empty(df):
        return  # or exit - depending on how we will include these data in plotting

    return df


def include_more_diode_info(df: DataFrame, scdb: LegendSlowControlDB) -> DataFrame:
    """Include more diode info, such as the channel name and the string number to which it belongs."""
    # get the diode info dataframe from the SC database
    df_info = scdb.dataframe("diode_info")
    # remove duplicates of detector names
    df_info = df_info.drop_duplicates(subset="label")
    # remove unnecessary columns (otherwise, they are repeated after the merging)
    df_info = df_info.drop(columns={"status", "tstamp"})
    # there is a repeated detector! Once with an additional blank space in front of its name: removed in case it is found
    if " V00050B" in list(df_info["label"].unique()):
        df_info = df_info[df_info["label"] != " V00050B"]

    # remove 'HV filter test' and 'no cable' entries
    df_info = df_info[~df_info["label"].str.contains("Ch")]
    # remove other stuff (???)
    if "?" in list(df_info["label"].unique()):
        df_info = df_info[df_info["label"] != "?"]
    if " routed" in list(df_info["label"].unique()):
        df_info = df_info[df_info["label"] != " routed"]
    if "routed" in list(df_info["label"].unique()):
        df_info = df_info[df_info["label"] != "routed"]

    # Merge df_info into df based on 'crate' and 'slot'
    merged_df = df.merge(
        df_info[["crate", "slot", "channel", "label", "group"]],
        on=["crate", "slot", "channel"],
        how="left",
    )
    merged_df = merged_df.rename(columns={"label": "name", "group": "string"})
    # remove "name"=NaN (ie entries for which there was not a correspondence among the two merged dataframes)
    merged_df = merged_df.dropna(subset=["name"])
    # switch from "String X" (str) to "X" (int) for entries of the 'string' column
    merged_df["string"] = merged_df["string"].str.extract(r"(\d+)").astype(int)

    return merged_df
