import os
import re
import subprocess
import sys

import yaml
from legendmeta import JsonDB

from . import analysis_data, plotting, slow_control, subsystem, utils


def retrieve_exposure(
    period: str, runs: str | list[str], runinfo_path: str, path: str, version: str
):

    runinfo = utils.read_json_or_yaml(runinfo_path)

    tot_liv = 0
    ac_exposure = 0
    on_validPSD_NONvalidPSD_exposure = 0
    on_validPSD_exposure = 0
    mass = 0
    on_validPSD_dets_exposure = {"BEGe": 0, "ICPC": 0, "COAX": 0, "PPC": 0}
    on_dets_exposure = {"BEGe": 0, "ICPC": 0, "COAX": 0, "PPC": 0}
    ac_dets_exposure = {"BEGe": 0, "ICPC": 0, "COAX": 0, "PPC": 0}

    for run in runs:
        if "phy" not in runinfo[period][run].keys():
            utils.logger.debug(f"No 'phy' key present in {runinfo_path}. Exit here")
            return
        livetime_in_d = runinfo[period][run]["phy"]["livetime_in_s"] / (60 * 60 * 24)
        tot_liv += livetime_in_d

        first_timestamp = runinfo[period][run]["phy"]["start_key"]
        full_status_map = utils.get_status_map(path, version, first_timestamp, "geds")

        map_file = os.path.join(
            path, version, "inputs/hardware/configuration/channelmaps"
        )
        full_channel_map = JsonDB(map_file).on(timestamp=first_timestamp)

        for hpge in full_channel_map.group("system").geds.map("name").keys():
            diode_path = utils.retrieve_json_or_yaml(
                os.path.join(
                    path, version, "inputs/hardware/detectors/germanium/diodes"
                ),
                hpge,
            )
            diode = utils.read_json_or_yaml(diode_path)
            usability = full_status_map[hpge]["usability"]
            psd = full_status_map[hpge]["psd"]

            mass += diode["production"]["mass_in_g"] / 1000

            expo = livetime_in_d / 365.25 * diode["production"]["mass_in_g"] / 1000

            if usability == "ac":
                ac_exposure += expo
                if hpge[0] == "B":
                    ac_dets_exposure["BEGe"] += expo
                if hpge[0] == "V":
                    ac_dets_exposure["ICPC"] += expo
                if hpge[0] == "C":
                    ac_dets_exposure["COAX"] += expo
                if hpge[0] == "P":
                    ac_dets_exposure["PPC"] += expo

            if usability == "on":
                on_validPSD_NONvalidPSD_exposure += expo
                if hpge[0] == "B":
                    on_dets_exposure["BEGe"] += expo
                if hpge[0] == "V":
                    on_dets_exposure["ICPC"] += expo
                if hpge[0] == "C":
                    on_dets_exposure["COAX"] += expo
                if hpge[0] == "P":
                    on_dets_exposure["PPC"] += expo

                if "is_bb_like" in psd and psd["is_bb_like"] != "missing":
                    if all(
                        [
                            psd["status"][p.strip()] == "valid"
                            for p in psd["is_bb_like"].split("&")
                        ]
                    ):
                        on_validPSD_exposure += expo
                        if hpge[0] == "B":
                            on_validPSD_dets_exposure["BEGe"] += expo
                        if hpge[0] == "V":
                            on_validPSD_dets_exposure["ICPC"] += expo
                        if hpge[0] == "C":
                            on_validPSD_dets_exposure["COAX"] += expo
                        if hpge[0] == "P":
                            on_validPSD_dets_exposure["PPC"] += expo

    utils.logger.info("mass: %s", round(mass, 3))
    utils.logger.info("period: %s", period)
    utils.logger.info("runs: %s", runs)
    utils.logger.info("Total livetime: %.4f d", tot_liv)
    utils.logger.info("AC exposure: %.4f kg-yr", round(ac_exposure, 4))
    formatted = {k: f"{v:.4f}" for k, v in ac_dets_exposure.items()}
    utils.logger.info("--- per detector: %s", formatted)
    utils.logger.info(
        "ON (valid PSD + non-valid PSD) exposure: %.4f kg-yr",
        round(on_validPSD_NONvalidPSD_exposure, 4),
    )
    formatted = {k: f"{v:.4f}" for k, v in on_dets_exposure.items()}
    utils.logger.info("--- per detector: %s", formatted)
    utils.logger.info(
        "ON (valid PSD) exposure: %.4f kg-yr", round(on_validPSD_exposure, 4)
    )
    formatted = {k: f"{v:.4f}" for k, v in on_validPSD_dets_exposure.items()}
    utils.logger.info("--- per detector: %s", formatted)


def retrieve_scdb(config: str, port: int, pswd: str):
    """Set the configuration file and the output paths when a user config file is provided. The function to retrieve Slow Control data from database is then automatically called."""
    # -------------------------------------------------------------------------
    # SSH tunnel to the Slow Control database
    # -------------------------------------------------------------------------
    # for the settings, see instructions on Confluence
    try:
        subprocess.run("ssh -T -N -f ugnet-proxy", shell=True, check=True)
        utils.logger.debug(
            "SSH tunnel to Slow Control database established successfully."
        )
    except subprocess.CalledProcessError as e:
        utils.logger.error(
            f"\033[91mError running SSH tunnel to Slow Control database command: {e}\033[0m"
        )
        sys.exit()

    # -------------------------------------------------------------------------
    # Read user settings
    # -------------------------------------------------------------------------
    config = utils.load_config(config)

    # check validity of scdb settings
    utils.check_scdb_settings(config)

    # -------------------------------------------------------------------------
    # Define PDF file basename
    # -------------------------------------------------------------------------

    # Format: l200-{period}-{run}-{data_type}; one pdf/log/shelve file for each subsystem
    out_path = utils.get_output_path(config) + "-slow_control.hdf"

    # -------------------------------------------------------------------------
    # Load and save data
    # -------------------------------------------------------------------------
    for idx, param in enumerate(config["slow_control"]["parameters"]):
        utils.logger.info(
            "\33[34m~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\33[0m"
        )
        utils.logger.info(f"\33[34m~~~ R E T R I E V I N G : {param}\33[0m")
        utils.logger.info(
            "\33[34m~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\33[0m"
        )

        # build a SlowControl object
        # - select parameter of interest from a list of available parameters
        # - apply time interval cuts
        # - get values from SC database (available from LNGS only)
        # - get limits/units/... from SC databasee (available from LNGS only)
        sc_analysis = slow_control.SlowControl(
            param, port, pswd, dataset=config["dataset"]
        )

        # check if the dataframe is empty or not (no data)
        if utils.check_empty_df(sc_analysis):
            utils.logger.warning(
                "\033[93m'%s' is not inspected, we continue with the next parameter (if present).\033[0m",
                param,
            )
            continue

        # remove the slow control hdf file if
        #   1) it already exists
        #   2) we specified "overwrite" as saving option
        #   3) it is the first parameter we want to save (idx==0)
        if os.path.exists(out_path) and config["saving"] == "overwrite" and idx == 0:
            os.remove(out_path)

        # save data to hdf file
        sc_analysis.data.copy().to_hdf(
            out_path,
            key=param.replace("-", "_"),
            mode="a",
        )


def control_plots(user_config_path: str, n_files=None):
    """Set the configuration file and the output paths when a user config file is provided. The function to generate plots is then automatically called."""
    # -------------------------------------------------------------------------
    # Read user settings
    # -------------------------------------------------------------------------
    with open(user_config_path) as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    # check validity of plot settings
    utils.check_plot_settings(config)

    # -------------------------------------------------------------------------
    # Define PDF file basename
    # -------------------------------------------------------------------------

    # Format: l200-p02-{run}-{data_type}; One pdf/log/shelve file for each subsystem
    plt_path = utils.get_output_path(config)

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    generate_plots(config, plt_path, n_files)


def auto_control_plots(
    config: str, file_keys: str, prod_path: str, prod_config: str, n_files=None
):
    """Set the configuration file and the output paths when a config file is provided during automathic plot production."""
    # -------------------------------------------------------------------------
    # Read user settings
    # -------------------------------------------------------------------------
    config = utils.load_config(config)

    # check validity of plot settings
    utils.check_plot_settings(config)

    # -------------------------------------------------------------------------
    # Add missing information (output, dataset) to the config
    # -------------------------------------------------------------------------
    config = utils.add_config_entries(config, file_keys, prod_path, prod_config)

    # -------------------------------------------------------------------------
    # Define PDF file basename
    # -------------------------------------------------------------------------

    # Format: l200-p02-{run}-{data_type}; One pdf/log/shelve file for each subsystem
    plt_path = utils.get_output_path(config)

    # plot
    generate_plots(config, plt_path, n_files)


def generate_plots(config: dict, plt_path: str, n_files=None):
    """Generate plots once the config file is set and once we provide the path and name in which store results. n_files specifies if we want to inspect the entire time window (if n_files is not specified), otherwise we subdivide the time window in smaller datasets, each one being composed by n_files files."""
    # no subdivision of data (useful when the inspected time window is short enough)
    if n_files is None:
        # some output messages, just to warn the user...
        if config["saving"] is None:
            utils.logger.warning(
                "\033[93mData will not be saved, but the pdf will be.\033[0m"
            )
        elif config["saving"] == "append":
            utils.logger.warning(
                "\033[93mYou're going to append new data to already existing data. If not present, you first create the output file as a very first step.\033[0m"
            )
        elif config["saving"] == "overwrite":
            utils.logger.warning(
                "\033[93mYou have accepted to overwrite already generated files, there's no way back until you manually stop the code NOW!\033[0m"
            )
        else:
            utils.logger.error(
                "\033[91mThe selected saving option in the config file is wrong. Try again with 'overwrite', 'append' or nothing!\033[0m"
            )
            sys.exit()
        # do the plots
        make_plots(config, plt_path, config["saving"])

    # for subdivision of data, let's loop over lists of timestamps, each one of length n_files
    else:
        # list of datasets to loop over later on
        bunches = utils.bunch_dataset(config.copy(), n_files)
        if bunches == []:
            utils.logger.info("No bunches of files were found. Exit here.")
            return

        # remove unnecessary keys for precaution - we will replace the time selections with individual timestamps/file keys
        config["dataset"].pop("start", None)
        config["dataset"].pop("end", None)
        config["dataset"].pop("runs", None)

        for idx, bunch in enumerate(bunches):
            utils.logger.debug(
                f"\33[44mYou are inspecting bunch #{idx+1}/{len(bunches)}...\33[0m"
            )
            # if it is the first dataset, just override previous content
            if idx == 0:
                config["saving"] = "overwrite"
            # if we already inspected the first dataset, append the ones coming after
            if idx > 0:
                config["saving"] = "append"

            # get the dataset
            config["dataset"]["timestamps"] = bunch
            # make the plots / load data for the dataset of interest
            make_plots(config.copy(), plt_path, config["saving"])


def make_plots(config: dict, plt_path: str, saving: str):

    # -------------------------------------------------------------------------
    # set up log file for each system
    # -------------------------------------------------------------------------
    log_file = utils.get_output_plot_path(plt_path, "log")
    file_handler = utils.logging.FileHandler(log_file)
    file_handler.setLevel(utils.logging.DEBUG)
    utils.logger.addHandler(file_handler)

    # -------------------------------------------------------------------------
    # flag events - PULSER
    # -------------------------------------------------------------------------
    # put it in a dict, so that later, if pulser is also wanted to be plotted, we don't have to load it twice
    subsystems = {"pulser": subsystem.Subsystem("pulser", dataset=config["dataset"])}
    # get list of all parameters needed for all requested plots, if any
    parameters = utils.get_all_plot_parameters("pulser", config)
    # get data for these parameters and time range given in the dataset
    # (if no parameters given to plot, baseline and wfmax will always be loaded to flag pulser events anyway)
    subsystems["pulser"].get_data(parameters)
    utils.logger.debug(subsystems["pulser"].data)

    # -------------------------------------------------------------------------
    # flag events - FC baseline
    # -------------------------------------------------------------------------
    subsystems["FCbsln"] = subsystem.Subsystem("FCbsln", dataset=config["dataset"])
    parameters = utils.get_all_plot_parameters("FCbsln", config)
    subsystems["FCbsln"].get_data(parameters)
    # the following 3 lines help to tag FC bsln events that are not in coincidence with a pulser
    subsystems["FCbsln"].flag_pulser_events(subsystems["pulser"])
    subsystems["FCbsln"].flag_fcbsln_only_events()
    subsystems["FCbsln"].data.drop(columns={"flag_pulser"})
    utils.logger.debug(subsystems["FCbsln"].data)

    # -------------------------------------------------------------------------
    # flag events - muon
    # -------------------------------------------------------------------------
    subsystems["muon"] = subsystem.Subsystem("muon", dataset=config["dataset"])
    parameters = utils.get_all_plot_parameters("muon", config)
    subsystems["muon"].get_data(parameters)
    utils.logger.debug(subsystems["muon"].data)

    # -------------------------------------------------------------------------
    # What subsystems do we want to plot?
    subsystems_to_plot = list(config["subsystems"].keys())

    for system in subsystems_to_plot:

        # -------------------------------------------------------------------------
        # set up subsystem
        # -------------------------------------------------------------------------

        # set up if wasn't already set up (meaning, not pulser, previously already set up)
        if system not in subsystems:
            # Subsystem: knows its channel map & software status (on/off channels)
            subsystems[system] = subsystem.Subsystem(system, dataset=config["dataset"])
            # get list of parameters needed for all requested plots, if any
            parameters = utils.get_all_plot_parameters(system, config)
            # get data for these parameters and dataset range
            subsystems[system].get_data(parameters)

        # load also aux channel if necessary (FOR ALL SYSTEMS), and add it to the already existing df
        for plot in config["subsystems"][system].keys():
            # !!! add if for sipms...
            subsystems[system].include_aux(
                config["subsystems"][system][plot]["parameters"],
                config["dataset"],
                config["subsystems"][system][plot],
                "pulser01ana",
            )

        utils.logger.debug(subsystems[system].data)

        # -------------------------------------------------------------------------
        # flag events (FOR ALL SYSTEMS)
        # -------------------------------------------------------------------------
        # flag pulser events for future parameter data selection
        subsystems[system].flag_pulser_events(subsystems["pulser"])
        # flag FC baseline events (not in correspondence with any pulser event) for future parameter data selection
        subsystems[system].flag_fcbsln_events(subsystems["FCbsln"])
        # flag muon events for future parameter data selection
        subsystems[system].flag_muon_events(subsystems["muon"])

        # remove timestamps for given detectors (moved here cause otherwise timestamps for flagging don't match)
        subsystems[system].remove_timestamps(utils.REMOVE_KEYS)
        utils.logger.debug(subsystems[system].data)

        # -------------------------------------------------------------------------
        # make subsystem plots
        # -------------------------------------------------------------------------
        plotting.make_subsystem_plots(
            subsystems[system],
            config["subsystems"][system],
            config["dataset"],
            plt_path,
            saving,
        )

        # -------------------------------------------------------------------------
        # save loaded data avoiding to plot it (eg quality cuts)
        # -------------------------------------------------------------------------
        analysis_data.load_subsystem_data(
            subsystems[system],
            config["dataset"],
            config["subsystems"][system],
            plt_path,
            saving,
        )

    # flush and remove the handler before cleaning
    file_handler.flush()
    utils.logger.removeHandler(file_handler)
    file_handler.close()
    # safely clean the log file
    with open(log_file) as f:
        log_text = f.read()

    pattern = re.compile(r"\033\[[0-9;]+m")
    clean_text = pattern.sub("", log_text)
    with open(log_file, "a") as f:
        f.write(clean_text)
