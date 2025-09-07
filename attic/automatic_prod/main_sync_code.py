import argparse
import json
import logging
import os
import re
import shlex
import subprocess
from pathlib import Path

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

# -------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Main code for automatically load and plot processed data on legend-login or NERSC cluster."
    )
    parser.add_argument(
        "--cluster",
        help="Name of the cluster where you are working; pick among 'lngs' or 'nersc'.",
    )
    parser.add_argument(
        "--ref_version",
        help="Version of processed data to inspect (eg. tmp-auto or ref-v2.1.0).",
    )
    parser.add_argument(
        "--output_folder",
        help="Path where to store the automatic results (plots and summary files).",
    )
    parser.add_argument(
        "--partition",
        default=False,
        help="False (default) if not partition data, else True",
    )
    parser.add_argument(
        "--pswd",
        help="Password to access the Slow Control database (NOT available on NERSC).",
    )
    parser.add_argument(
        "--sc",
        default=False,
        help="Boolean for retrieving Slow Control data (default: False).",
    )
    parser.add_argument(
        "--port",
        default=8282,
        help="Port necessary to retrieve the Slow Control database (default: 8282).",
    )
    parser.add_argument(
        "--pswd_email",
        default=None,
        help="Password to access the legend.data.monitoring@gmail.com account for sending alert messages.",
    )
    parser.add_argument(
        "--chunk_size",
        default=20,
        type=int,
        help="Maximum integer number of files to read at each loop in order to avoid the process to be killed.",
    )
    parser.add_argument(
        "--p",
        default=None,
        help="Period to inspect.",
    )
    parser.add_argument(
        "--r",
        default=None,
        help="Run to inspect.",
    )
    parser.add_argument(
        "--escale",
        default=2039,
        help="Energy sccale at which evaluating the gain differences; default: 2039 keV (76Ge Qbb).",
    )
    parser.add_argument(
        "--pdf",
        default=False,
        help="True if you want to save pdf files too; default: False",
    )

    args = parser.parse_args()
    cluster = args.cluster
    ref_version = args.ref_version
    output_folder = args.output_folder
    partition = False if args.partition is False else True
    pswd = args.pswd
    get_sc = False if args.sc is False else True
    port = args.port
    pswd_email = args.pswd_email
    chunk_size = args.chunk_size
    input_period = args.p
    input_run = args.r
    save_pdf = False if args.pdf is False else True
    escale_val = args.escale

    auto_dir = (
        "/global/cfs/cdirs/m2676/data/lngs/l200/public/prodenv/prod-blind/"
        if cluster == "nersc"
        else "/data2/public/prodenv/prod-blind/"
    )
    auto_dir_path = os.path.join(auto_dir, ref_version)
    found = False
    for tier in ["hit", "pht", "dsp", "psp", "evt", "pet"]:
        search_directory = os.path.join(auto_dir_path, "generated/tier", tier, "phy")
        if os.path.isdir(search_directory):
            found = True
            break
    if found is False:
        logger.debug(f"No valid folder {search_directory} found. Exiting.")
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

    # commands to run the container
    if cluster == "nersc":
        cmd = "shifter --image=legendexp/legend-base:latest --env PATH=$HOME/.local/bin:$PATH"
    else:
        cmd = "apptainer run --env PATH=$HOME/.local/bin:$PATH --cleanenv /data2/public/prodenv/containers/legendexp_legend-base_latest.sif"

    # ===========================================================================================
    # BEGINNING OF THE ANALYSIS
    # ===========================================================================================

    # define slow control dict
    scdb = {
        "output": output_folder,
        "dataset": {
            "experiment": "L200",
            "period": period,
            "version": ref_version,
            "path": auto_dir,
            "type": "phy",
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
            "type": "phy",
            "runs": int(run.split("r")[-1]),
        },
        "saving": "append",
        "subsystems": {
            "geds": {
                "Event rate in pulser events": {
                    "parameters": "event_rate",
                    "event_type": "pulser",
                    "plot_structure": "per string",
                    "resampled": "only",
                    "plot_style": "vs time",
                    "time_window": "20S",
                },
                "Event rate in FCbsln events": {
                    "parameters": "event_rate",
                    "event_type": "FCbsln",
                    "plot_structure": "per string",
                    "resampled": "only",
                    "plot_style": "vs time",
                    "time_window": "20S",
                },
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
                "Baselines (dsp/baseline) in FCbsln events": {
                    "parameters": "baseline",
                    "event_type": "FCbsln",
                    "plot_structure": "per string",
                    "resampled": "only",
                    "plot_style": "vs time",
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
                "Mean baselines (dsp/bl_mean) in FCbsln events": {
                    "parameters": "bl_mean",
                    "event_type": "FCbsln",
                    "plot_structure": "per string",
                    "resampled": "only",
                    "plot_style": "vs time",
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
                "Uncalibrated gain (dsp/trapEmax) in FCbsln events": {
                    "parameters": "trapEmax",
                    "event_type": "FCbsln",
                    "plot_structure": "per string",
                    "resampled": "only",
                    "plot_style": "vs time",
                    "AUX_ratio": True,
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
                "Calibrated gain (hit/trapEmax_ctc_cal) in FCbsln events": {
                    "parameters": "trapEmax_ctc_cal",
                    "event_type": "FCbsln",
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
                "Noise (dsp/bl_std) in FCbsln events": {
                    "parameters": "bl_std",
                    "event_type": "FCbsln",
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
                "A/E (from dsp) in FCbsln events": {
                    "parameters": "AoE_Custom",
                    "event_type": "FCbsln",
                    "plot_structure": "per string",
                    "resampled": "only",
                    "plot_style": "vs time",
                    "variation": True,
                    "time_window": "10T",
                },
                "Quality cuts in physics events": {
                    "parameters": "quality_cuts",
                    "event_type": "phy",
                    "qc_flags": True,
                    "qc_classifiers": False,
                },
            }
        },
    }

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
        logger.debug(f"Error: folder '{source_dir}' does not exist.")
        exit()
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

    # If new files are found, run the shell command
    if new_files:
        # Replace this command with your desired shell command
        command = "echo New files found: \033[91m{}\033[0m".format(" ".join(new_files))
        subprocess.run(command, shell=True)

        # create the file containing the keys with correct format to be later used by legend-data-monitor (it must be created every time with the new keys; NOT APPEND)
        logger.debug("Creating the file containing the keys to inspect...")
        with open(os.path.join(rsync_path, "new_keys.filekeylist"), "w") as f:
            for new_file in new_files:
                new_file = new_file.split("-tier")[0]
                f.write(new_file + "\n")
        logger.debug("...done!")

        # run the plot production
        logger.debug("Running the generation of plots...")
        keys_file = os.path.join(rsync_path, "new_keys.filekeylist")

        # read all lines from the original file
        with open(keys_file) as f:
            lines = f.readlines()
        num_lines = len(lines)

        safe_json_string = shlex.quote(json.dumps(my_config))

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
                logger.debug(
                    f"[{idx}/{total_parts}] Created file: {output_file} with {len(chunk)} lines."
                )
                bash_command = (
                    f"{cmd} legend-data-monitor user_rsync_prod "
                    f"--config {safe_json_string} --keys {output_file}"
                )
                logger.debug("...running command for generating hdf monitoring files")
                subprocess.run(bash_command, shell=True)
        else:
            logger.debug(f"... file has {num_lines} lines. No need to split.")
            bash_command = (
                f"{cmd} legend-data-monitor user_rsync_prod "
                f"--config {safe_json_string} --keys {keys_file}"
            )
            logger.debug("...running command for generating hdf monitoring files")
            subprocess.run(bash_command, shell=True)
        logger.debug("...done!")

        # compute resampling + info yaml
        logger.debug("Resampling outputs...")
        files_folder = os.path.join(output_folder, ref_version)
        bash_command = f"{cmd} python monitoring.py summary_files --path {files_folder} --period {period} --run {run}"
        logger.debug(f"...running command {bash_command}")
        subprocess.run(bash_command, shell=True)
        logger.debug("...done!")

        # ===========================================================================================
        # Analyze Slow Control data
        # ===========================================================================================
        if cluster == "lngs" and get_sc is True:
            try:
                logger.debug("Retrieving Slow Control data...")
                safe_json_string = shlex.quote(json.dumps(scdb))
                bash_command = (
                    f"{cmd} legend-data-monitor user_scdb "
                    f"--config {safe_json_string} --port {port} --pswd {pswd}"
                )
                logger.debug(f"...running command {bash_command}")
                subprocess.run(bash_command, shell=True)
                logger.debug("...SC done!")
            except subprocess.CalledProcessError as e:
                logger.error(f"Slow Control command failed: {e}")
            except Exception as e:
                logger.error(
                    f"Unexpected error while retrieving Slow Control data: {e}"
                )

        # ===========================================================================================
        # Generate Monitoring Summary Plots
        # ===========================================================================================
        mtg_folder = os.path.join(output_folder, ref_version, "generated/plt/hit/phy")
        os.makedirs(mtg_folder, exist_ok=True)
        logger.info(f"Folder {mtg_folder} ensured")

        # define dataset depending on the (latest) monitored period/run

        avail_runs = sorted(os.listdir(os.path.join(mtg_folder, period)))
        avail_runs = [
            ar for ar in avail_runs if "mtg" not in ar and ar != ".ipynb_checkpoints"
        ]
        dataset = {period: avail_runs}
        if dataset[period] != []:
            logger.debug("Generating monitoring plots...")
            # get first timestamp of first run of the given period
            start_key = (
                sorted(
                    os.listdir(os.path.join(search_directory, period, avail_runs[0]))
                )[0]
            ).split("-")[4]

            mtg_bash_command = f"{cmd} python monitoring.py plot --public_data {auto_dir_path} --hdf_files {mtg_folder} --output {mtg_folder} --start {start_key} --p {period} --avail_runs {avail_runs} --pswd_email {pswd_email} --escale {escale_val} --current_run {run} --last_checked {last_checked}"
            if partition is True:
                mtg_bash_command += " --partition True"
            if save_pdf is True:
                mtg_bash_command += " --pdf True"

            logger.debug(f"...running command {mtg_bash_command}")
            subprocess.run(mtg_bash_command, shell=True)
            logger.info("...monitoring plots generated!")

        # ===========================================================================================
        # Calibration checks
        # ===========================================================================================
        cal_bash_command = f"{cmd} python monitoring.py calib_psd --public_data {auto_dir_path} --output {mtg_folder} --p {period} --current_run {run}"
        if save_pdf is True:
            cal_bash_command += " --pdf True"
        logger.debug(f"...running command {cal_bash_command}")
        subprocess.run(cal_bash_command, shell=True)
        logger.info("...calibration data inspected!")

    else:
        logger.debug("No new files were detected.")

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


if __name__ == "__main__":
    main()
