import glob
import os
import re
from pathlib import Path

import yaml

from . import calibration, core, monitoring, utils


def auto_run(
    cluster,
    ref_version,
    output_folder,
    partition,
    pswd,
    get_sc,
    port,
    pswd_email,
    chunk_size,
    input_period,
    input_run,
    save_pdf,
    escale_val,
    data_type,
):
    """Inspect LEGEND HDF5 (LH5) processed data (and Slow Control data from lngs-login cluster) for a specific period and run (if specified; otherwise the latest being processed are used); plots and summary files are saved; automatic alert emails are sent."""
    auto_dir = (
        "/global/cfs/cdirs/m2676/data/lngs/l200/public/prodenv/prod-blind/"
        if cluster == "nersc"
        else "/data2/public/prodenv/prod-blind/"
    )
    auto_dir_path = os.path.join(auto_dir, ref_version)
    found = False
    for tier in ["hit", "pht", "dsp", "psp", "evt", "pet", "ssc", "lac", "rdc", "bkg"]:
        search_directory = os.path.join(
            auto_dir_path, "generated/tier", tier, data_type
        )
        if os.path.isdir(search_directory):
            found = True
            utils.logger.debug(f"Valid folder: {search_directory}")
            break
    if found is False:
        utils.logger.debug(f"No valid folder {search_directory} found. Exiting.")
        exit()

    def search_latest_folder(my_dir):
        directories = [
            d for d in os.listdir(my_dir) if os.path.isdir(os.path.join(my_dir, d))
        ]
        directories.sort(key=lambda x: Path(my_dir, x).stat().st_ctime)
        return directories[-1]

    # Period to monitor
    period = (
        search_latest_folder(search_directory) if input_period is None else input_period
    )
    # Run to monitor
    run = search_latest_folder(search_directory) if input_run is None else input_run
    source_dir = os.path.join(search_directory, period, run)

    # ===========================================================================================
    # START OF THE ANALYSIS
    # ===========================================================================================

    # define slow control dict
    scdb = {
        "output": output_folder,
        "dataset": {
            "experiment": "L200",
            "period": period,
            "version": ref_version,
            "path": auto_dir,
            "type": data_type,
            "runs": int(run.split("r")[-1]),
        },
        "saving": "overwrite",
        "slow_control": {
            "parameters": [
                "DaqLeft-Temp1",
                "DaqLeft-Temp2",
                "DaqRight-Temp1",
                "DaqRight-Temp2",
                "RREiT",
                "RRNTe",
                "RRSTe",
                "ZUL_T_RR",
            ]
        },
    }

    # define geds dict
    my_config = {
        "output": output_folder,
        "dataset": {
            "experiment": "L200",
            "period": period,
            "version": ref_version,
            "path": auto_dir,
            "type": data_type,
            "runs": int(run.split("r")[-1]),
        },
        "saving": "append",
        "subsystems": {
            "geds": {
                "Baselines (dsp/baseline) in pulser events": {
                    "parameters": "baseline",
                    "event_type": "pulser",
                    "plot_structure": "per string",
                    "resampled": "only",
                    "plot_style": "vs time",
                    "AUX_ratio": True,
                    "variation": True,
                    "time_window": "10T",
                },
                "Mean baselines (dsp/bl_mean) in pulser events": {
                    "parameters": "bl_mean",
                    "event_type": "pulser",
                    "plot_structure": "per string",
                    "resampled": "only",
                    "plot_style": "vs time",
                    "AUX_ratio": True,
                    "variation": True,
                    "time_window": "10T",
                },
                "trapTmax gain (dsp/trapTmax) in pulser events": {
                    "parameters": "trapTmax",
                    "event_type": "pulser",
                    "plot_structure": "per string",
                    "resampled": "only",
                    "plot_style": "vs time",
                    "AUX_ratio": True,
                    "variation": True,
                    "time_window": "10T",
                },
                "tp_0_est gain (dsp/tp_0_est) in pulser events": {
                    "parameters": "tp_0_est",
                    "event_type": "pulser",
                    "plot_structure": "per string",
                    "resampled": "only",
                    "plot_style": "vs time",
                    "AUX_ratio": True,
                    "variation": True,
                    "time_window": "10T",
                },
                "Uncalibrated gain (dsp/trapEmax) in pulser events": {
                    "parameters": "trapEmax",
                    "event_type": "pulser",
                    "plot_structure": "per string",
                    "resampled": "only",
                    "plot_style": "vs time",
                    "AUX_ratio": True,
                    "variation": True,
                    "time_window": "10T",
                },
                "Calibrated gain (hit/trapEmax_ctc_cal) in physics events": {
                    "parameters": "trapEmax_ctc_cal",
                    "event_type": "phy",
                    "plot_structure": "per string",
                    "resampled": "only",
                    "plot_style": "vs time",
                    "variation": True,
                    "time_window": "10T",
                },
                "Calibrated gain (hit/trapEmax_ctc_cal) in pulser events": {
                    "parameters": "trapEmax_ctc_cal",
                    "event_type": "pulser",
                    "plot_structure": "per string",
                    "resampled": "only",
                    "plot_style": "vs time",
                    "variation": True,
                    "time_window": "10T",
                },
                "Noise (dsp/bl_std) in pulser events": {
                    "parameters": "bl_std",
                    "event_type": "pulser",
                    "plot_structure": "per string",
                    "resampled": "only",
                    "plot_style": "vs time",
                    "AUX_ratio": True,
                    "variation": True,
                    "time_window": "10T",
                },
                "A/E (from dsp) in pulser events": {
                    "parameters": "AoE_Custom",
                    "event_type": "pulser",
                    "plot_structure": "per string",
                    "resampled": "only",
                    "plot_style": "vs time",
                    "variation": True,
                    "time_window": "10T",
                },
                "Quality cuts and classifiers in physics events": {
                    "parameters": "quality_cuts",
                    "event_type": "phy",
                    "qc_flags": True,
                    "qc_classifiers": True,
                },
                "QC classifiers in all events": {
                    "parameters": "quality_cuts",
                    "event_type": "all",
                    "qc_flags": False,
                    "qc_classifiers": True,
                },
                "QC classifiers in pulser events": {
                    "parameters": "quality_cuts",
                    "event_type": "pulser",
                    "qc_flags": False,
                    "qc_classifiers": True,
                },
                "QC classifiers in FCbsln events": {
                    "parameters": "quality_cuts",
                    "event_type": "FCbsln",
                    "qc_flags": False,
                    "qc_classifiers": True,
                },
            }
        },
    }

    # ===========================================================================================
    # Check calibration stability and create summary files
    # ===========================================================================================

    phy_folder = os.path.join(
        output_folder, ref_version, "generated/plt/hit", data_type
    )
    os.makedirs(os.path.join(phy_folder, period, run), exist_ok=True)
    if os.path.isfile(
        os.path.join(phy_folder, period, run, f"l200-{period}-{run}-qcp_summary.yaml")
    ):
        pass
    else:
        os.makedirs(os.path.join(phy_folder, period, run, "mtg/pdf"), exist_ok=True)
        utils.logger.info("...inspecting calibration data!")
        check_calib(
            auto_dir_path=auto_dir_path,
            output_folder=phy_folder,
            period=period,
            current_run=run,
            pswd_email=pswd_email,
            data_type=data_type,
            partition=partition,
            save_pdf=save_pdf,
        )
        utils.logger.info("...done!")

    # ===========================================================================================
    # Get not-analyzed files
    # ===========================================================================================

    # File to store the timestamp of the last check
    rsync_path = os.path.join(
        output_folder, ref_version, "generated", "tmp", "mtg", period, run
    )
    os.makedirs(rsync_path, exist_ok=True)
    timestamp_file = os.path.join(rsync_path, "last_checked_timestamp.txt")

    # Read the last checked timestamp
    last_checked = None
    if os.path.exists(timestamp_file):
        with open(timestamp_file) as file:
            last_checked = file.read().strip()

    # Get the current timestamp
    if not os.path.isdir(source_dir):
        utils.logger.debug(f"Error: folder '{source_dir}' does not exist.")
        exit()
    else:
        utils.logger.debug(f"Found folder {source_dir}")
    current_files = os.listdir(source_dir)
    new_files = []

    # Compare the timestamps of files and find new files
    for file in current_files:
        file_path = os.path.join(source_dir, file)
        current_timestamp = os.path.getmtime(file_path)
        if last_checked is None or current_timestamp > float(last_checked):
            new_files.append(file)

    # If new files are found, check if they are ok or not
    if new_files:
        pattern = r"\d+"
        correct_files = []

        for new_file in new_files:
            matches = re.findall(pattern, new_file)
            # get only files with correct ending (and discard the ones that are still under processing)
            if len(matches) == 6:
                correct_files.append(new_file)
        new_files = correct_files
    new_files = sorted(new_files)

    if new_files:
        utils.logger.info(f"New files found: {' '.join(new_files)}")

        # create the file containing the keys with correct format to be later used by legend-data-monitor (it must be created every time with the new keys; NOT APPEND)
        utils.logger.debug("Creating the file containing the keys to inspect...")
        with open(os.path.join(rsync_path, "new_keys.filekeylist"), "w") as f:
            for new_file in new_files:
                new_file = new_file.split("-tier")[0]
                f.write(new_file + "\n")
        utils.logger.debug("...done!")

        # run the plot production
        utils.logger.debug("Running the generation of plots...")
        keys_file = os.path.join(rsync_path, "new_keys.filekeylist")

        # read all lines from the original file
        with open(keys_file) as f:
            lines = f.readlines()
        num_lines = len(lines)

        if num_lines > chunk_size:
            # split lines into chunks and write to multiple files
            for idx, i in enumerate(range(0, num_lines, chunk_size), start=1):
                chunk = lines[i : i + chunk_size]
                output_file = os.path.join(
                    rsync_path, f"new_keys_part_{i // chunk_size + 1}.filekeylist"
                )

                with open(output_file, "w") as out_f:
                    out_f.writelines(chunk)

                total_parts = (num_lines + chunk_size - 1) // chunk_size
                utils.logger.debug(
                    f"[{idx}/{total_parts}] Created file: {output_file} with {len(chunk)} lines."
                )
                utils.logger.debug(
                    "...running command for generating hdf monitoring files"
                )
                core.auto_control_plots(my_config, output_file, "", {})
        else:
            utils.logger.debug(f"... file has {num_lines} lines. No need to split.")
            utils.logger.debug("...running command for generating hdf monitoring files")
            core.auto_control_plots(my_config, keys_file, "", {})

        utils.logger.debug("...done!")

        # compute resampling + info yaml
        utils.logger.debug("Resampling outputs...")
        files_folder = os.path.join(output_folder, ref_version)
        monitoring.build_new_files(files_folder, period, run, data_type=data_type)
        utils.logger.debug("...done!")

        # ===========================================================================================
        # Analyze Slow Control data
        # ===========================================================================================
        if cluster == "lngs" and get_sc is True:
            try:
                utils.logger.debug("Retrieving Slow Control data...")
                core.retrieve_scdb(scdb, port, pswd)
                utils.logger.debug("...SC done!")
            except Exception as e:
                utils.logger.error(
                    f"Failed to retrieve Slow Control data: {e}"
                )

        # ===========================================================================================
        # Generate Monitoring Summary Plots
        # ===========================================================================================
        mtg_folder = os.path.join(
            output_folder, ref_version, "generated/plt/hit", data_type
        )
        os.makedirs(mtg_folder, exist_ok=True)
        utils.logger.info(f"Folder {mtg_folder} ensured")

        # define dataset depending on the (latest) monitored period/run
        avail_runs = sorted(os.listdir(os.path.join(mtg_folder, period)))
        avail_runs = [
            ar for ar in avail_runs if "mtg" not in ar and ar != ".ipynb_checkpoints"
        ]
        dataset = {period: avail_runs}
        if dataset[period] != []:
            # per-period & per-run monitoring plots
            utils.logger.debug("...generating monitoring plots")
            start_key = (
                sorted(
                    os.listdir(os.path.join(search_directory, period, avail_runs[0]))
                )[0]
            ).split("-")[4]

            summary_plots(
                auto_dir_path=auto_dir_path,
                phy_mtg_data=mtg_folder,
                output_folder=mtg_folder,
                start_key=start_key,
                period=period,
                current_run=run,
                runs=avail_runs,
                pswd_email=pswd_email,
                last_checked=last_checked,
                data_type=data_type,
                partition=partition,
                escale_val=escale_val,
                save_pdf=save_pdf,
                # zoom=False,  # Optional
                # quadratic=False,  # Optional
            )
            utils.logger.info("...done!")

            # QC - average + time series
            utils.logger.info("...inspecting quality cuts")
            qc_avg_series(
                auto_dir_path=auto_dir_path,
                output_folder=mtg_folder,
                start_key=start_key,
                period=period,
                current_run=run,
                save_pdf=save_pdf,
            )
            utils.logger.info("...done!")

    else:
        utils.logger.debug("No new files were detected.")

    # Update the last checked timestamp
    with open(timestamp_file, "w") as file:
        file.write(
            str(
                os.path.getmtime(
                    max(
                        [os.path.join(source_dir, file) for file in current_files],
                        key=os.path.getmtime,
                    )
                )
            )
        )


def summary_plots(
    auto_dir_path: str,
    phy_mtg_data: str,
    output_folder: str,
    start_key: str,
    period: str,
    current_run: str,
    runs: list,
    pswd_email: str,
    last_checked: str,
    data_type: str = "phy",
    partition: bool = False,
    escale_val: float = 2039.0,
    save_pdf: bool = False,
    zoom: bool = False,
    quadratic: bool = False,
):
    """
    Run function for creating summary plots.
    
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
    current_run : str
        Run under inspection.
    runs : list
        Available runs to inspect for a given period.
    pswd_email : str
        Password to access the legend.data.monitoring@gmail.com account for sending alert messages.
    last_checked : str
        Timestamp of the last check.
    data_type : str
        Data type to load; default: 'phy'.
    partition : bool
        False if not partition data; default: False.
    escale_val : float
        Energy scale at which evaluating the gain differences; default: 2039 keV (76Ge Qbb).
    save_pdf : bool
        True if you want to save pdf files too; default: False.
    zoom : bool
        True to zoom over y axis; default: False.
    quadratic : bool
        True if you want to plot the quadratic resolution too; default: False.
    """
    det_info = utils.build_detector_info(
        os.path.join(auto_dir_path, "inputs"), start_key=start_key
    )

    # stability plots
    results = monitoring.plot_time_series(
        auto_dir_path,
        phy_mtg_data,
        output_folder,
        data_type,
        period,
        runs,
        current_run,
        det_info,
        save_pdf,
        escale_val,
        last_checked,
        partition,
        quadratic,
        zoom,
    )

    # load proper calibration (eg for lac/ssc/rdc data or back-dated calibs)
    tier = "pht" if partition is True else "hit"
    validity_file = os.path.join(auto_dir_path, "generated/par", tier, "validity.yaml")
    with open(validity_file) as f:
        validity_dict = yaml.load(f, Loader=yaml.CLoader)

    # find first key of current run
    start_key = utils.get_start_key(auto_dir_path, data_type, period, current_run)
    # use key to load the right yaml file
    valid_entries = [e for e in validity_dict if e["valid_from"] <= start_key]
    if valid_entries:
        apply = max(valid_entries, key=lambda e: e["valid_from"])["apply"][0]
        run_to_apply = apply.split("/")[-1].split("-")[2]
    else:
        if data_type not in ["lac", "ssc", "rdc"]:
            utils.logger.debug(
                f"No valid calibration was found for {period}-{current_run}. Return."
            )
        return

    # don't run any check if there are no runs
    cal_path = os.path.join(auto_dir_path, "generated/par", tier, "cal", period)
    cal_runs = os.listdir(cal_path)
    if len(cal_runs) == 0:
        utils.logger.debug("No available calibration runs to inspect. Returning.")
        return

    cal_path = os.path.join(auto_dir_path, "generated/par", tier, "cal", period)
    pars_files_list = sorted(glob.glob(f"{cal_path}/*/*.yaml"))
    if not pars_files_list:
        pars_files_list = sorted(glob.glob(f"{cal_path}/*/*.json"))
    det_info = utils.build_detector_info(
        os.path.join(auto_dir_path, "inputs"), start_key=start_key
    )

    pars_path = [p for p in pars_files_list if run_to_apply in p][0]
    pars = utils.read_json_or_yaml(pars_path)
    # phy box summary plots
    for k in results.keys():
        pars_dict = pars if k in ["TrapemaxCtcCal"] else None
        monitoring.box_summary_plot(
            period,
            current_run,
            pars_dict,
            det_info,
            results[k],
            utils.MTG_PLOT_INFO[k],
            output_folder,
            data_type,
            save_pdf,
            run_to_apply=run_to_apply,
        )

    utils.check_cal_phy_thresholds(
        output_folder,
        period,
        current_run,
        data_type,
        det_info["detectors"],
        pswd_email,
    )

    # FT failure rate plots
    if data_type not in ["ssc", "lac", "rdc"]:

        # qc classifier plots
        monitoring.qc_distributions(
            auto_dir_path,
            phy_mtg_data,
            output_folder,
            start_key,
            period,
            current_run,
            det_info,
            save_pdf,
        )

        monitoring.qc_and_evt_summary_plots(
            auto_dir_path,
            phy_mtg_data,
            output_folder,
            start_key,
            period,
            current_run,
            det_info,
            save_pdf,
        )


def check_calib(
    auto_dir_path: str,
    output_folder: str,
    period: str,
    current_run: str,
    pswd_email: str,
    data_type: str = "phy",
    partition: bool = False,
    save_pdf: bool = False,
):
    """
    Check calibration stability in calibration runs and create monitoring summary file.
    
    Parameters
    ----------
    auto_dir_path : str
        Path to tmp-auto public data files (eg /data2/public/prodenv/prod-blind/tmp-auto).
    output_folder : str
        Path to output folder.
    period : str
        Period to inspect.
    current_run : str
        Run under inspection.
    pswd_email : str
        Password to access the legend.data.monitoring@gmail.com account for sending alert messages.
    data_type : str
        Data type to load; default: 'phy'.
    partition : bool
        False if not partition data; default: False.
    save_pdf : bool
        True if you want to save pdf files too; default: False.
    """
    tier = "pht" if partition is True else "hit"
    validity_file = os.path.join(auto_dir_path, "generated/par", tier, "validity.yaml")
    with open(validity_file) as f:
        validity_dict = yaml.load(f, Loader=yaml.CLoader)

    # find first key of current run
    start_key = utils.get_start_key(auto_dir_path, data_type, period, current_run)
    # use key to load the right yaml file
    valid_entries = [e for e in validity_dict if e["valid_from"] <= start_key]
    if valid_entries:
        apply = max(valid_entries, key=lambda e: e["valid_from"])["apply"][0]
        run_to_apply = apply.split("/")[-1].split("-")[2]
    else:
        if data_type not in ["lac", "ssc", "rdc"]:
            utils.logger.debug(
                f"No valid calibration was found for {period}-{current_run}. Return."
            )
        return

    # don't run any check if there are no runs
    cal_path = os.path.join(auto_dir_path, "generated/par", tier, "cal", period)
    cal_runs = os.listdir(cal_path)
    if len(cal_runs) == 0:
        utils.logger.debug("No available calibration runs to inspect. Returning.")
        return
    first_run = len(cal_runs) == 1

    cal_path = os.path.join(auto_dir_path, "generated/par", tier, "cal", period)
    pars_files_list = sorted(glob.glob(f"{cal_path}/*/*.yaml"))
    if not pars_files_list:
        pars_files_list = sorted(glob.glob(f"{cal_path}/*/*.json"))
    det_info = utils.build_detector_info(
        os.path.join(auto_dir_path, "inputs"), start_key=start_key
    )

    if data_type not in ["lac", "ssc", "rdc"]:
        current_run = run_to_apply
        utils.logger.debug(f"...valid run for {current_run} is {run_to_apply}")

        calibration.check_calibration(
            auto_dir_path,
            output_folder,
            period,
            current_run,
            first_run,
            det_info,
            save_pdf,
        )

        calibration.check_psd(
            auto_dir_path,
            cal_path,
            pars_files_list,
            output_folder,
            period,
            current_run,
            det_info,
            save_pdf,
        )
    else:
        calibration.check_calibration_lac_ssc(
            auto_dir_path,
            output_folder,
            period,
            current_run,
            run_to_apply,
            first_run,
            det_info,
            save_pdf=save_pdf,
            data_type=data_type,
        )

        utils.logger.debug(
            f"...we do not inspect PSD time stability in {data_type} data"
        )

    utils.check_cal_phy_thresholds(
        output_folder,
        period,
        current_run,
        "cal",
        det_info["detectors"],
        pswd_email,
    )


def qc_avg_series(
    auto_dir_path: str,
    output_folder: str,
    start_key: str,
    period: str,
    current_run: str,
    save_pdf: bool = False,
):
    """
    Plot quality cuts average values across the array and trends in time.
    
    Parameters
    ----------
    auto_dir_path : str
        Path to tmp-auto public data files (eg /data2/public/prodenv/prod-blind/tmp-auto).
    output_folder : str
        Path to output folder.
    start_key : str
        First timestamp of the inspected range.
    period : str
        Period to inspect.
    current_run : str
        Run under inspection.
    save_pdf : bool
        True if you want to save pdf files too; default: False.
    """
    det_info = utils.build_detector_info(
        os.path.join(auto_dir_path, "inputs/"), start_key=start_key
    )

    monitoring.qc_average(
        auto_dir_path, output_folder, det_info, period, current_run, save_pdf
    )
    monitoring.qc_time_series(
        auto_dir_path, output_folder, det_info, period, current_run, save_pdf
    )
