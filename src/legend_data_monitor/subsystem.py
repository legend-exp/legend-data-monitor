import os
import sys
import typing
from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd
from dbetto import TextDB
from pygama.flow import DataLoader

from . import utils

list_of_str = list[str]
tuple_of_str = tuple[str]


class Subsystem:
    """
    Object containing information for a given subsystem such as channel map, channels status etc.

    sub_type [str]: geds | spms | pulser | pulser01ana | FCbsln | muon

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
                2. 'window' [str]: time window in the past from current time point, format: 'Xd Xh Xm' for days, hours, minutes
                2. 'timestamps': str or list of str in format 'YYYYMMDDThhmmssZ'
                3. 'runs': int or list of ints for run number(s)  e.g. 10 for r010
    Or input kwargs separately experiment=, period=, path=, version=, type=; start=&end=, or window=, or timestamps=, or runs=

    Experiment is needed to know which channel belongs to the pulser Subsystem (and its name), "auxs" ch0 (L60) or "puls" ch1 (L200)
    Period is needed to know channel name ("fcid" or "rawid")
    Selection range is needed for the channel map and status information at that time point, and should be the only information needed,
        however, pylegendmeta only allows query .on(timestamp=...) but not .on(run=...);
        therefore, to be able to get info in case of `runs` selection, we need to know
        path, version, and run type to look up first timestamp of the run.
        If this changes in the future, the path will only be asked when data is requested to be loaded with Subsystem.get_data(),
        but not to just load the channel map and status for given run

    Might set default "latest" for version, but gotta be careful.
    """

    def __init__(self, sub_type: str, **kwargs):

        banner = "\33[95m" + "-" * 50 + "\33[0m"
        utils.logger.info(banner)
        utils.logger.info(f"\33[95m S E T T I N G  UP : {sub_type}\33[0m")
        utils.logger.info(banner)

        self.type = sub_type

        # -------------------------------------------------------------------------
        # input check
        # -------------------------------------------------------------------------

        # if setup= kwarg was provided, get dict provided
        # otherwise kwargs is itself already the dict we need with experiment= and period=
        data_info = kwargs["dataset"] if "dataset" in kwargs else kwargs

        # validity check of kwarg
        utils.dataset_validity_check(data_info)

        # -------------------------------------------------------------------------
        # get channel info for this subsystem
        # -------------------------------------------------------------------------

        # needed to know for making 'if' statement over different experiments/periods
        self.experiment = data_info["experiment"]
        self.period = data_info["period"]
        # need to remember for channel status query
        # ! now needs to be single !
        self.datatype = data_info["type"]
        # need to remember for DataLoader config
        self.path = data_info["path"]
        self.version = data_info["version"]

        # data stored under these folders have been partitioned!
        if "tmp-auto" != self.path:
            self.partition = True
        else:
            self.partition = False

        (
            self.timerange,
            self.first_timestamp,
            self.last_timestamp,
        ) = utils.get_query_times(**kwargs)

        # None will be returned if something went wrong
        if not self.timerange:
            utils.logger.error("\033[91m%s\033[0m", self.get_data.__doc__)
            return

        self.channel_map = self.get_channel_map()

        # add column status to channel map stating on/off
        self.get_channel_status()

        # -------------------------------------------------------------------------
        # have something before get_data() is called just in case
        self.data = pd.DataFrame()

    def get_data(self, parameters: typing.Union[str, list_of_str, tuple_of_str] = ()):
        """
        Get data for requested parameters from DataLoader and "prime" it to be ready for analysis.

        parameters: single parameter or list of parameters to load.
            If empty, only default parameters will be loaded (channel, timestamp; baseline and wfmax for pulser)
        """
        utils.logger.info("... getting data")

        # -------------------------------------------------------------------------
        # Set up DataLoader config
        # -------------------------------------------------------------------------
        utils.logger.info("...... setting up DataLoader")

        # --- construct list of parameters for the data loader
        # depending on special parameters, k lines etc.
        params_for_dataloader = self.get_parameters_for_dataloader(parameters)

        # --- set up DataLoader config
        # separate dsp and hit parameters as they might have different paths
        param_tiers = pd.DataFrame.from_dict(utils.PARAMETER_TIERS.items())
        param_tiers.columns = ["param", "tier"]
        # keep only parameters that have an associated entry in the parameter-tiers.yaml file
        my_params = param_tiers[param_tiers["param"].isin(params_for_dataloader)]
        dsp_params = my_params[my_params["tier"] == "dsp"]["param"].tolist()
        hit_params = my_params[my_params["tier"] == "hit"]["param"].tolist()
        evt_params = my_params[my_params["tier"] == "evt"]["param"].tolist()
        utils.logger.debug("dsp/psp parameters: %s", dsp_params)
        utils.logger.debug("hit/pht parameters: %s", hit_params)
        utils.logger.debug("evt/pet parameters: %s", evt_params)

        # find parameters that are not in the YAML
        missing_params = [
            p for p in params_for_dataloader if p not in param_tiers["param"].values
        ]
        if missing_params:
            utils.logger.warning(
                "\033[93mThe following parameters are not in settings/parameter-tiers.yaml and will be skipped:\033[0m %s",
                ", ".join(missing_params),
            )

        dl_dsp = None
        dl_hit = None
        dl_evt1 = None
        dl_evt2 = None
        if dsp_params != []:
            dlconfig_dsp, dbconfig_dsp = self.construct_dataloader_configs(
                param_tiers, dsp_params, "dsp"
            )
            dl_dsp = DataLoader(dlconfig_dsp, dbconfig_dsp)  # set up DataLoader
        if hit_params != []:
            dlconfig_hit, dbconfig_hit = self.construct_dataloader_configs(
                param_tiers, hit_params, "hit"
            )
            dl_hit = DataLoader(dlconfig_hit, dbconfig_hit)
        if evt_params != []:
            # separate params with different paths
            lists = {}
            for param in evt_params:
                # take the first element *after* "geds/quality/"
                key = param.split("/")[2]
                lists.setdefault(key, []).append(param)

            evt_params1 = lists.get("is_bb_like", [])
            evt_params2 = lists.get("is_not_bb_like", [])

            dlconfig_evt1, dbconfig_evt1 = self.construct_dataloader_configs(
                param_tiers, evt_params1, "evt"
            )
            dl_evt1 = DataLoader(dlconfig_evt1, dbconfig_evt1)

            dlconfig_evt2, dbconfig_evt2 = self.construct_dataloader_configs(
                param_tiers, evt_params2, "evt"
            )
            dl_evt2 = DataLoader(dlconfig_evt2, dbconfig_evt2)

        # -------------------------------------------------------------------------
        # Set up query
        # -------------------------------------------------------------------------

        # if querying by run, time word is 'run'; otherwise 'timestamp'; is the key of the timerange dict
        time_word = list(self.timerange.keys())[0]

        if "start" in self.timerange[time_word]:
            # query by (run/timestamp >= ) and (run/timestamp <=) if format {start: end:} - note: start/end have to be expressed in UTC+00 since timestamps in filenames are expressed in that format too
            # ...this does not enter into files and get potential timestamps that enter into the selected time window;
            # ...for the same reason, you can get timestamps over th selected time range because there is no cut in it (this can potentially be fixed later on by cutting away some rows from the dataframe)
            query = f"({time_word} >= '{self.timerange[time_word]['start']}') and ({time_word} <= '{self.timerange[time_word]['end']}')"
        else:
            # query by (run/timestamp == ) or (run/timestamp == ) if format [list of runs/timestamps]
            query = " or ".join(
                f"({time_word} == '" + run_or_timestamp + "')"
                for run_or_timestamp in self.timerange[time_word]
            )

        # --- cal or phy data or both
        # ! not possible to load both phy and cal for now, based on how channel status works
        # query += (
        #     " and ("
        #     + " or ".join("(type == '" + x + "')" for x in self.type)
        #     + ")"
        # )
        query += f" and (type == '{self.datatype}')"

        # p02 keys (missing ch068)
        query += (
            " and (timestamp != '20230125T222013Z')"
            + " and (timestamp != '20230126T015308Z')"
            + " and (timestamp != '20230222T231553Z')"
        )
        utils.logger.info("...... querying DataLoader - removing %s", query)

        # --- query data loader
        if dsp_params != []:
            dl_dsp.set_files(query)
            dl_dsp.set_output(fmt="pd.DataFrame", columns=dsp_params)
        if hit_params != []:
            dl_hit.set_files(query)
            dl_hit.set_output(fmt="pd.DataFrame", columns=hit_params)
        if evt_params != []:
            # from "geds/quality/is_X" to "is_X"
            elems1 = evt_params1[0].split("/")
            evt_params1 = [elems1[-1]]
            dl_evt1.set_files(query)
            dl_evt1.set_output(fmt="pd.DataFrame", columns=evt_params1)

            elems2 = evt_params2[0].split("/")
            evt_params2 = [elems2[-1]]
            dl_evt2.set_files(query)
            dl_evt2.set_output(fmt="pd.DataFrame", columns=evt_params2)

        now = datetime.now()

        # --- create self.data object
        dsp_data = None
        hit_data = None
        evt_data1 = None
        evt_data2 = None
        if dsp_params != []:
            dsp_data = dl_dsp.load()
        if hit_params != []:
            hit_data = dl_hit.load()
        if evt_params != []:
            evt_data1 = dl_evt1.load()
            evt_data2 = dl_evt2.load()

        valid_data = [
            df for df in [dsp_data, hit_data, evt_data1, evt_data2] if df is not None
        ]
        if not valid_data:
            utils.logger.error(
                "\033[91mdsp_data, hit_data, evt_data are all None. Exit here.\033[0m"
            )
            sys.exit()
        elif len(valid_data) == 1:
            self.data = valid_data[0]
        else:
            self.data = pd.concat(valid_data, axis=1)

        utils.logger.info(f"Total time to load data: {(datetime.now() - now)}")

        # -------------------------------------------------------------------------
        # polish things up
        # -------------------------------------------------------------------------
        # drop useless columns
        self.data = self.data.drop(
            columns=[
                col
                for col in self.data.columns
                if col.endswith("_idx") or col == "file"
            ]
        )
        # rename "table" column to "channel" and drop duplicates
        table_cols = [col for col in self.data.columns if col.endswith("_table")]
        if table_cols:
            keep_col = table_cols[0]
            drop_cols = [col for col in table_cols if col != keep_col]
            self.data = self.data.drop(columns=drop_cols)
            self.data = self.data.rename(columns={keep_col: "channel"})

        # -------------------------------------------------------------------------
        # create datetime column based on initial key and timestamp
        # -------------------------------------------------------------------------
        # convert UTC timestamp to datetime (unix epoch time)
        self.data["datetime"] = pd.to_datetime(
            self.data["timestamp"], origin="unix", utc=True, unit="s"
        )
        self.data = self.data.drop("timestamp", axis=1)

        # -------------------------------------------------------------------------
        # add detector name, location and position from map
        # -------------------------------------------------------------------------
        utils.logger.info("... mapping to name and string/fiber position")
        self.data = self.data.set_index("channel")
        # expand channel map index to match that of data with repeating channels
        ch_map_reindexed = self.channel_map.set_index("channel").reindex(
            self.data.index
        )
        # append the channel map columns to the data
        self.data = pd.concat([self.data, ch_map_reindexed], axis=1)
        self.data = self.data.reset_index()
        for col in ["location", "position"]:
            # ignore string values for fibers ('I/OB-XXX-XXX') and positions ('top/bottom') for SiPMs
            if isinstance(self.data[col].iloc[0], float):
                self.data[col] = self.data[col].astype(int)
        utils.logger.info("... appended channel map to the data dataframe")

        # -------------------------------------------------------------------------
        # if this subsystem is pulser, flag pulser timestamps
        # -------------------------------------------------------------------------

        if self.type == "pulser":
            self.flag_pulser_events()
        if self.type == "FCbsln":
            self.flag_fcbsln_events()
        if self.type == "muon":
            self.flag_muon_events()
        utils.logger.info("... flagge pulser | FC bsl | muon events")

    def include_aux(
        self, params: Union[str, list], dataset: dict, plot: dict, aux_ch: str
    ):
        """Include in a new column data coming from PULS01ANA aux channel, to either compute a ratio or a difference with data coming from the inspected subsystem."""
        # auxiliary channel of reference (fixed for the moment)
        aux_channel = "pulser01ana"
        # both options (diff and ratio) are present -> BAD! For this parameter we do not subtract/divide for any AUX entry
        if "AUX_ratio" in plot.keys() and "AUX_diff" in plot.keys():
            utils.logger.error(
                "\033[91mYou selected both 'AUX_ratio' and 'AUX_diff' for %s. Pick one!\033[0m",
                plot["parameters"],
            )
            sys.exit()
        # one option (either diff or ratio) is present
        if "AUX_ratio" in plot.keys() or "AUX_diff" in plot.keys():
            # check if the selected AUX channel exists, otherwise continue
            if "AUX_ratio" in plot.keys() and plot["AUX_ratio"] is True:
                utils.logger.debug(
                    "... you are going to plot the parameter accounting for the ratio wrt PULS01ANA data"
                )
            if "AUX_diff" in plot.keys() and plot["AUX_diff"] is True:
                utils.logger.debug(
                    "... you are going to plot the parameter accounting for the difference wrt PULS01ANA data"
                )

            utils.logger.debug(
                "... but now we are going to perform diff/ratio with PULS01ANA entries"
            )

        def add_aux(param):
            aux_subsys = Subsystem(aux_channel, dataset=dataset)
            # get data for these parameters and time range given in the dataset
            # (if no parameters given to plot, baseline and wfmax will always be loaded to flag pulser events anyway)
            aux_subsys.get_data(param)

            # Merge the dataframes based on the 'datetime' column
            utils.logger.debug(
                "... merging the PULS01ANA dataframe with the original one"
            )
            self.data = self.data.merge(
                aux_subsys.data[["datetime", param]], on="datetime", how="left"
            )

            # ratio
            self.data[f"{param}_{aux_ch}Ratio"] = (
                self.data[f"{param}_x"] / self.data[f"{param}_y"]
            )
            # diff
            self.data[f"{param}_{aux_ch}Diff"] = (
                self.data[f"{param}_x"] - self.data[f"{param}_y"]
            )
            # rename columns (absolute values)
            self.data = self.data.rename(
                columns={f"{param}_x": param, f"{param}_y": f"{param}_{aux_ch}"}
            )

        # one-parameter case
        if (isinstance(params, list) and len(params) == 1) or isinstance(params, str):
            param = params if isinstance(params, str) else params[0]
            # check if the parameter under study is special; if so, skip it
            if param in utils.SPECIAL_PARAMETERS.keys():
                utils.logger.warning(
                    "\033[93m'%s' is a special parameter. "
                    + "For the moment, we skip the ratio/diff wrt the AUX channel and plot the parameter as it is.\033[0m",
                    params,
                )
                return
            if param in [
                "quality_cuts",
                "geds/quality/is_not_bb_like/is_delayed_discharge",
                "geds/quality/is_bb_like",
            ]:
                utils.logger.warning(
                    "\033[93m'%s' does not require the ratio/diff wrt the AUX channel. Skip this step.\033[0m",
                    params,
                )
                return
            # check if the parameter under study is from 'hit' tier; if so, skip it
            if (
                param in utils.PARAMETER_TIERS.keys()
                and utils.PARAMETER_TIERS[param] == "hit"
            ):
                utils.logger.warning(
                    "\033[93m'%s' is saved in hit tier, for which no AUX channel is present. "
                    + "We skip the ratio/diff wrt the AUX channel and plot the parameter as it is.\033[0m",
                    params,
                )
                return
            if f"{param}_{aux_channel}" not in list(self.data.columns):
                add_aux(params)

        # multiple-parameters case
        if isinstance(params, list) and len(params) > 1:
            for param in params:
                if param in utils.SPECIAL_PARAMETERS.keys():
                    utils.logger.warning(
                        "\033[93m'%s' is a special parameter. "
                        + "For the moment, we skip the ratio/diff wrt the AUX channel and plot the parameter as it is.\033[0m",
                        params,
                    )
                    return
                if utils.PARAMETER_TIERS[param] == "hit":
                    utils.logger.warning(
                        "\033[93m'%s' is saved in hit tier, for which no AUX channel is present. "
                        + "We skip the ratio/diff wrt the AUX channel and plot the parameter as it is.\033[0m",
                        param,
                    )
                    continue
                if f"{param}_{aux_channel}" not in list(self.data.columns):
                    add_aux(params)

    def flag_pulser_events(self, pulser=None):
        """Flag pulser events. If a pulser object was provided, flag pulser events in data based on its flag."""
        utils.logger.info("... flagging pulser events")

        # --- if a pulser object was provided, flag pulser events in data based on its flag
        if pulser:
            try:
                pulser_timestamps = pulser.data[pulser.data["flag_pulser"]][
                    "datetime"
                ]  # .set_index('datetime').index
                self.data["flag_pulser"] = False
                self.data = self.data.set_index("datetime")
                self.data.loc[pulser_timestamps, "flag_pulser"] = True
            except KeyError:
                utils.logger.warning(
                    "\033[93mWarning: cannot flag pulser events, timestamps don't match!\n \
                    If you are you looking at calibration data, it's not possible to flag pulser events in it this way.\n \
                    Contact the developers if you would like them to focus on advanced flagging methods.\033[0m"
                )
                utils.logger.warning(
                    "\033[93m! Proceeding without pulser flag !\033[0m"
                )

        else:
            # --- if no object was provided, it's understood that this itself is a pulser
            trapTmax = self.data["trapTmax"]
            pulser_timestamps = self.data[trapTmax > 200].index
            # flag them
            self.data["flag_pulser"] = False
            self.data.loc[pulser_timestamps, "flag_pulser"] = True

        self.data = self.data.reset_index()

    def flag_fcbsln_events(self, fc_bsln=None):
        """Flag FC baseline events, keeping the ones that are in correspondence with a pulser event too. If a FC baseline object was provided, flag FC baseline events in data based on its flag."""
        utils.logger.info("... flagging FC baseline events")

        # --- if a FC baseline object was provided, flag FC baseline events in data based on its flag
        if fc_bsln:
            try:
                fc_bsln_timestamps = fc_bsln.data[fc_bsln.data["flag_fc_bsln"]][
                    "datetime"
                ]  # .set_index('datetime').index
                self.data["flag_fc_bsln"] = False
                self.data = self.data.set_index("datetime")
                self.data.loc[fc_bsln_timestamps, "flag_fc_bsln"] = True
            except KeyError:
                utils.logger.warning(
                    "\033[93mWarning: cannot flag FC baseline events, timestamps don't match!\n \
                    If you are you looking at calibration data, it's not possible to flag FC baseline events in it this way.\n \
                    Contact the developers if you would like them to focus on advanced flagging methods.\033[0m"
                )
                utils.logger.warning(
                    "\033[93m! Proceeding without FC baseline flag !\033[0m"
                )

        else:
            # --- if no object was provided, it's understood that this itself is a FC baseline
            # find timestamps over threshold
            high_thr = 3000
            self.data = self.data.set_index("datetime")
            wf_max_rel = self.data["wf_max"] - self.data["baseline"]
            fc_bsln_timestamps = self.data[wf_max_rel > high_thr].index
            # flag them
            self.data["flag_fc_bsln"] = False
            self.data.loc[fc_bsln_timestamps, "flag_fc_bsln"] = True

        self.data = self.data.reset_index()

    def flag_fcbsln_only_events(self, fc_bsln=None):
        """Flag FC baseline events. If a FC baseline object was provided, flag FC baseline events in data based on its flag."""
        utils.logger.info("... flagging FC baseline ONLY events")

        # --- if a FC baseline object was provided, flag FC baseline events in data
        if fc_bsln:
            self.data = self.data.merge(
                fc_bsln.data[["datetime", "flag_fc_bsln"]], on="datetime"
            )

        # in any case, define FC bsln events as FC bsln events for which there was not a pulser event
        self.data["flag_fc_bsln"] = (
            self.data["flag_fc_bsln"] & ~self.data["flag_pulser"]
        )

        self.data = self.data.reset_index()

    def flag_muon_events(self, muon=None):
        """Flag muon events. If a muon object was provided, flag muon events in data based on its flag."""
        utils.logger.info("... flagging muon events")

        # --- if a muon object was provided, flag muon events in data based on its flag
        if muon:
            try:
                muon_timestamps = muon.data[muon.data["flag_muon"]][
                    "datetime"
                ]  # .set_index('datetime').index
                self.data["flag_muon"] = False
                self.data = self.data.set_index("datetime")
                self.data.loc[muon_timestamps, "flag_muon"] = True
            except KeyError:
                utils.logger.warning(
                    "\033[93mWarning: cannot flag muon events, timestamps don't match!\n \
                    If you are you looking at calibration data, it's not possible to flag muon events in it this way.\n \
                    Contact the developers if you would like them to focus on advanced flagging methods.\033[0m"
                )
                utils.logger.warning("\033[93m! Proceeding without muon flag !\033[0m")

        else:
            # --- if no object was provided, it's understood that this itself is a muon
            # find timestamps over threshold
            high_thr = 500
            self.data = self.data.set_index("datetime")
            wf_max_rel = self.data["wf_max"] - self.data["baseline"]
            muon_timestamps = self.data[wf_max_rel > high_thr].index
            # flag them
            self.data["flag_muon"] = False
            self.data.loc[muon_timestamps, "flag_muon"] = True

        self.data = self.data.reset_index()

    def get_channel_map(self):
        """
        Build channel map for given subsystem with info like name, position, cc4, HV, DAQ, detector type, ... for each channel.

        setup_info: dict with the keys 'experiment' and 'period'

        Later will probably be changed to get channel map by run, if possible
        Planning to add:
            - barrel column for SiPMs special case
        """
        utils.logger.info("... getting channel map")

        # -------------------------------------------------------------------------
        # load full channel map of this exp and period (and version)
        # -------------------------------------------------------------------------

        map_file = os.path.join(
            self.path, self.version, "inputs/hardware/configuration/channelmaps"
        )
        full_channel_map = TextDB(map_file).on(timestamp=self.first_timestamp)

        df_map = pd.DataFrame(columns=utils.COLUMNS_TO_LOAD)
        df_map = df_map.set_index("channel")

        # -------------------------------------------------------------------------
        # helper function to determine which channel map entry belongs to this subsystem
        # -------------------------------------------------------------------------

        # for L60-p01 and L200-p02, keep using 'fcid' as channel
        if int(self.period.split("p")[-1]) < 3:
            ch_flag = "fcid"
        # from L200-p03 included, uses 'rawid' as channel
        if int(self.period.split("p")[-1]) >= 3:
            ch_flag = "rawid"

        # dct_key is the subdict corresponding to one chmap entry
        def is_subsystem(entry):
            # special case for pulser
            if self.type == "pulser":
                if self.experiment == "L60":
                    return entry["system"] == "auxs" and entry["daq"]["fcid"] == 0
                if self.experiment == "L200":
                    # we get PULS01
                    if self.below_period_3_excluded():
                        return entry["system"] == "puls" and entry["daq"][ch_flag] == 1
                    # we get PULS01ANA
                    if self.above_period_3_included():
                        return (
                            entry["system"] == "puls"
                            # and entry["daq"][ch_flag] == 1027203
                            and entry["daq"][ch_flag] == 1027201
                        )
            # special case for pulser AUX
            if self.type == "pulser01ana":
                if self.experiment == "L60":
                    utils.logger.error(
                        "\033[91mThere is no pulser AUX channel in L60. Remove this subsystem!\033[0m"
                    )
                    exit()
                if self.experiment == "L200":
                    if self.below_period_3_excluded():
                        return entry["system"] == "puls" and entry["daq"][ch_flag] == 3
                    if self.above_period_3_included():
                        return (
                            entry["system"] == "puls"
                            and entry["daq"][ch_flag] == 1027203
                        )
            # special case for baseline
            if self.type == "FCbsln":
                if self.experiment == "L60":
                    return entry["system"] == "auxs" and entry["daq"]["fcid"] == 0
                if self.experiment == "L200":
                    if self.below_period_3_excluded():
                        return entry["system"] == "bsln" and entry["daq"][ch_flag] == 0
                    if self.above_period_3_included():
                        return (
                            entry["system"] == "bsln"
                            and entry["daq"][ch_flag] == 1027200
                        )
            # special case for muon channel
            if self.type == "muon":
                if self.experiment == "L60":
                    return entry["system"] == "auxs" and entry["daq"]["fcid"] == 1
                if self.experiment == "L200":
                    if self.below_period_3_excluded():
                        return entry["system"] == "auxs" and entry["daq"][ch_flag] == 2
                    if self.above_period_3_included():
                        return (
                            entry["system"] == "auxs"
                            and entry["daq"][ch_flag] == 1027202
                        )
            # for geds or spms
            return entry["system"] == self.type

        # name of location in the channel map
        loc_code = {"geds": "string", "spms": "fiber"}

        # detector type for geds in the channel map
        type_code = {"B": "bege", "C": "coax", "V": "icpc", "P": "ppc"}

        # systems for which the location/position has to be handled carefully; values were chosen arbitrarily to avoid conflicts
        special_systems = utils.SPECIAL_SYSTEMS

        # -------------------------------------------------------------------------
        # loop over entries and find out subsystem
        # -------------------------------------------------------------------------

        # config.channel_map is already a dict read from the channel map
        for entry in full_channel_map:
            # skip dummy channels
            if "BF" in entry or "DUMMY" in entry:
                continue

            entry_info = full_channel_map[entry]

            # skip if this is not our system
            if not is_subsystem(entry_info):
                continue

            # --- add info for this channel - Raw/FlashCam ID, unique for geds/spms/pulser/pulser01ana/FCbsln/muon
            ch = entry_info["daq"][ch_flag]

            df_map.at[ch, "name"] = entry_info["name"]
            # number/name of string/fiber for geds/spms, dummy for pulser/pulser01ana/FCbsln/muon
            df_map.at[ch, "location"] = (
                special_systems[self.type]
                if self.type in special_systems
                else entry_info["location"][loc_code[self.type]]
            )
            # position in string/fiber for geds/spms, dummy for pulser/pulser01ana/FCbsln/muon
            df_map.at[ch, "position"] = (
                special_systems[self.type]
                if self.type in special_systems
                else entry_info["location"]["position"]
            )
            # CC4 information - will be None for L60 (set to 'null') or spms (there, but no CC4s)
            df_map.at[ch, "cc4_id"] = (
                entry_info["electronics"]["cc4"]["id"] if self.type == "geds" else None
            )
            df_map.at[ch, "cc4_channel"] = (
                entry_info["electronics"]["cc4"]["channel"]
                if self.type == "geds"
                else None
            )
            # DAQ information - present even in L60 and spms
            df_map.at[ch, "daq_crate"] = entry_info["daq"]["crate"]
            df_map.at[ch, "daq_card"] = entry_info["daq"]["card"]["id"]
            # voltage = not for pulser/spms (just daq and electronics)
            df_map.at[ch, "HV_card"] = (
                entry_info["voltage"]["card"]["id"] if self.type == "geds" else None
            )
            df_map.at[ch, "HV_channel"] = (
                entry_info["voltage"]["channel"] if self.type == "geds" else None
            )
            # detector type for geds (based on channel's name)
            if self.type == "geds":
                df_map.at[ch, "det_type"] = (
                    type_code[entry_info["name"][0]]
                    if entry_info["name"][0] in type_code.keys()
                    else None
                )
            else:
                df_map.at[ch, "det_type"] = None

        df_map = df_map.reset_index()

        # -------------------------------------------------------------------------
        # stupid dataframe, can use dtype somehow to fix it?
        for col in [
            "channel",
            "location",
            "position",
            "cc4_channel",
            "daq_crate",
            "daq_card",
            "HV_card",
            "HV_channel",
        ]:
            if isinstance(df_map[col].loc[0], float):
                df_map[col] = df_map[col].astype(int)

        # sort by channel -> do we really need to?
        df_map = df_map.sort_values("channel")
        return df_map

    def get_channel_status(self):
        """
        Add status column to channel map with on/off for software status.

        setup_info: dict with the keys 'experiment' and 'period'

        Later will probably be changed to get channel status by timestamp (or hopefully run, if possible)
        """
        utils.logger.info("... getting channel status")

        # -------------------------------------------------------------------------
        # load full status map of this time selection (and version)
        # -------------------------------------------------------------------------
        full_status_map = utils.get_status_map(
            self.path, self.version, self.first_timestamp, self.datatype
        )

        # AUX channels are not in status map, so at least for pulser/pulser01ana/FCbsln/muon need default on
        self.channel_map["status"] = "on"

        self.channel_map = self.channel_map.set_index("name")
        # 'channel_name' has the format 'DNNXXXS' (= "name" column)
        for channel_name in full_status_map:
            # status map contains all channels, check if this channel is in our subsystem
            if channel_name in self.channel_map.index:
                self.channel_map.at[channel_name, "status"] = full_status_map[
                    channel_name
                ]["usability"]

        # -------------------------------------------------------------------------
        # quick-fix to remove detectors while status maps are not updated
        # -------------------------------------------------------------------------
        for channel_name in utils.REMOVE_DETS:
            # status map contains all channels, check if this channel is in our subsystem
            if channel_name in self.channel_map.index:
                self.channel_map.at[channel_name, "status"] = "off"

        self.channel_map = self.channel_map.reset_index()

    def get_parameters_for_dataloader(self, parameters: typing.Union[str, list_of_str]):
        """
        Construct list of parameters to query from the DataLoader.

        - parameters that are always loaded (+ pulser special case)
        - parameters that are already in lh5
        - parameters needed for calculation, if special parameter(s) asked (e.g. wf_max_rel)
        """
        # --- always read timestamp
        params = ["timestamp"]
        # --- always get wf_max & baseline for pulser for flagging
        if self.type in ["pulser", "pulser01ana", "FCbsln", "muon"]:
            params += ["wf_max", "baseline", "trapTmax"]

        # --- add user requested parameters
        # change to list for convenience, if input was single
        if isinstance(parameters, str):
            parameters = [parameters]

        for param in parameters:
            if param in utils.SPECIAL_PARAMETERS:
                # for special parameters, look up which parameters are needed to be loaded for their calculation
                # if none, ignore
                params += (
                    utils.SPECIAL_PARAMETERS[param]
                    if utils.SPECIAL_PARAMETERS[param]
                    else []
                )
            else:
                # otherwise just add the parameter directly
                params.append(param)

        # some parameters might be repeated twice - remove
        return list(np.unique(params))

    def construct_dataloader_configs(
        self, param_tiers, params: list_of_str, tier_key: str
    ):
        """
        Construct DL and DB configs for DataLoader based on parameters and which tiers they belong to.

        params: list of parameters to load
        """
        tiers, _ = utils.get_tiers_pars_folders(os.path.join(self.path, self.version))
        data_dir = os.path.join(self.path, self.version, "generated", "tier")

        if tier_key == "dsp":
            tier_folder = tiers[0]
            tier_folder_part = tiers[1]
        if tier_key == "hit":
            tier_folder = tiers[2]
            tier_folder_part = tiers[3]
        if tier_key == "evt":
            tier_folder = tiers[5]
            tier_folder_part = tiers[6]
        tier_key_new = tier_key

        if self.partition:
            # check if the psp/pht folder exists (ie is not empty)
            if os.path.isdir(tier_folder_part) and os.listdir(tier_folder_part):
                if tier_key == "dsp":
                    tier_key_new = "psp"
                if tier_key == "hit":
                    tier_key_new = "pht"
                if tier_key == "evt":
                    tier_key_new = "pet"
                param_tiers["tier"] = param_tiers["tier"].replace(
                    tier_key, tier_key_new
                )
                data_dir = os.path.join(
                    tier_folder_part.split("generated")[0], "generated", "tier"
                )
            else:
                data_dir = os.path.join(
                    tier_folder.split("generated")[0], "generated", "tier"
                )

        # which of these are requested by user
        param_tiers = param_tiers[param_tiers["param"].isin(params)]
        utils.logger.info("...... loading parameters from the following tiers:")
        utils.logger.debug(param_tiers)

        # -------------------------------------------------------------------------
        # set up config templates
        # -------------------------------------------------------------------------
        dict_dbconfig = {
            "data_dir": data_dir,
            "tier_dirs": {},
            "file_format": {},
            "table_format": {},
            "tables": {},
            "columns": {},
        }

        dict_dlconfig = {"channel_map": {}, "levels": {}}

        # -------------------------------------------------------------------------
        # set up tiers depending on what parameters we need
        # -------------------------------------------------------------------------

        # only load channels that are on or ac
        chlist = list(
            self.channel_map[
                (self.channel_map["status"] == "on")
                | (self.channel_map["status"] == "ac")
                | (self.channel_map["status"] == "True")
                | (self.channel_map["status"] is True)
            ]["channel"]
        )
        # remove off channels
        removed_chs = list(
            self.channel_map[self.channel_map["status"] == "off"]["name"]
        )
        utils.logger.info(f"...... not loading channels with status off: {removed_chs}")

        # for L60-p01 and L200-p02, keep using 3 digits
        if int(self.period.split("p")[-1]) < 3:
            ch_format = "ch:03d"
        # from L200-p03 included, uses 7 digits
        if int(self.period.split("p")[-1]) >= 3:
            ch_format = "ch:07d"

        # --- settings for each tier
        for tier, tier_params in param_tiers.groupby("tier"):
            dict_dbconfig["tier_dirs"][tier] = f"/{tier}"
            # type not fixed and instead specified in the query
            dict_dbconfig["file_format"][tier] = (
                "/{type}/"
                + self.period  # {period}
                + "/{run}/{exp}-"
                + self.period  # {period}
                + "-{run}-{type}-{timestamp}-tier_"
                + tier
                + ".lh5"
            )

            dict_dbconfig["table_format"][tier] = "ch{" + ch_format + "}/"
            if tier == "pht":
                dict_dbconfig["table_format"][tier] += "hit"
            elif tier in ["evt", "pet"]:
                elems = list(tier_params["param"])[0].split("/")
                core = elems[:-1]
                flag = "/".join(core)
                dict_dbconfig["table_format"][tier] = f"evt/{flag}/"
                tier_params["param"] = elems[-1]
            elif tier == "psp":
                dict_dbconfig["table_format"][tier] += "dsp"
            else:
                dict_dbconfig["table_format"][tier] += tier

            dict_dbconfig["tables"][tier] = chlist

            dict_dbconfig["columns"][tier] = list(tier_params["param"])

        # --- settings based on tier hierarchy
        order = {tier_key_new: 2, "raw": 1}
        param_tiers = param_tiers.copy()
        param_tiers["order"] = param_tiers["tier"].apply(lambda x: order[x])
        # find highest tier
        max_tier = param_tiers[param_tiers["order"] == param_tiers["order"].max()][
            "tier"
        ].iloc[0]
        # format {"hit": {"tiers": ["dsp", "hit"]}}
        dict_dlconfig["levels"][max_tier] = {
            "tiers": list(param_tiers["tier"].unique())
        }

        return dict_dlconfig, dict_dbconfig

    def remove_timestamps(self, remove_keys: dict):
        """Remove timestamps from the dataframes for a given channel.

        The time interval in which to remove the channel is provided through an external YAML file.
        """
        # all timestamps we are considering are expressed in UTC0
        utils.logger.debug("... removing timestamps from the following detectors:")

        # loop over channels for which we want to remove timestamps
        for detector in remove_keys:
            if detector in self.data["name"].unique():
                utils.logger.debug(f".... {detector}")
                # remove timestamps from self.data that are within time_from and time_to, for a given channel
                for chunk in remove_keys[detector]:
                    utils.logger.debug(f"from {chunk['from']} to {chunk['to']}")
                    # times are in format YYYYMMDDTHHMMSSZ, convert them into a UTC0 timestamp
                    for point in ["from", "to"]:
                        # convert UTC timestamp to datetime (unix epoch time)
                        chunk[point] = pd.to_datetime(
                            chunk[point], utc=True, format="%Y%m%dT%H%M%SZ"
                        )

                    # entries to drop for this chunk
                    rows_to_drop = self.data[
                        (self.data["name"] == detector)
                        & (self.data["datetime"] >= chunk["from"])
                        & (self.data["datetime"] <= chunk["to"])
                    ]
                    self.data = self.data.drop(rows_to_drop.index)

        self.data = self.data.reset_index()

    def below_period_3_excluded(self) -> bool:
        if int(self.period.split("p")[-1]) < 3:
            return True
        else:
            return False

    def above_period_3_included(self) -> bool:
        if int(self.period.split("p")[-1]) >= 3:
            return True
        else:
            return False
