import glob
import importlib.resources
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from . import calibration, core, errors, logs, monitoring, repack, tasks, utils
from .contract import build as contract_build
from .contract import reader as contract_reader
from .contract import schema as contract_schema
from .plots import calib as calib_plots
from .plots import qc as qc_plots_mod
from .plots import stability as stability_plots
from .plots import summary as summary_plots_mod
from .plots import timeseries as contract_plots


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
    prod_root=None,
    render_plots=True,
):
    """Inspect LEGEND HDF5 (LH5) processed data (and Slow Control data from lngs-login cluster) for a specific period and run (if specified; otherwise the latest being processed are used) and save plots and summary files.

    The stages run as isolated tasks with per-task log files under
    ``<output>/<ref_version>/generated/tmp/log/<timestamp>/`` (see ``logs``);
    a failing task does not stop the remaining ones. Returns the exit code
    for the CLI (0 ok, 1 at least one task failed).

    ``prod_root`` overrides the cluster-mapped production root (useful for
    local/mock trees); by default the root is derived from ``cluster``.
    """
    if prod_root is not None:
        auto_dir = prod_root
    else:
        auto_dir = (
            "/global/cfs/cdirs/m2676/data/lngs/l200/public/prodenv/prod-blind/"
            if cluster == "nersc"
            else "/data2/public/prodenv/prod-blind/"
        )
    auto_dir_path = os.path.join(auto_dir, ref_version)
    found = False
    for tier in [
        "hit",
        "pht",
        "dsp",
        "psp",
        "evt",
        "pet",
        "ssc",
        "lac",
        "rdc",
        "bkg",
        "tst",
    ]:
        search_directory = os.path.join(
            auto_dir_path, "generated/tier", tier, data_type
        )
        if os.path.isdir(search_directory):
            found = True
            utils.logger.debug(f"Valid folder: {search_directory}")
            break
    if found is False:
        raise errors.ConfigError(
            f"no valid tier folder found under {auto_dir_path} for '{data_type}'"
        )

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
    search_directory = os.path.join(search_directory, period)
    if not os.path.isdir(search_directory):
        raise errors.ConfigError(f"period directory does not exist: {search_directory}")

    # Run to monitor
    run = search_latest_folder(search_directory) if input_run is None else input_run
    source_dir = os.path.join(search_directory, run)
    if not os.path.isdir(source_dir):
        raise errors.ConfigError(f"run directory does not exist: {source_dir}")
    utils.logger.info(f"You are inspecting {period}-{run}")

    # ===========================================================================================
    # Configuration for the individual tasks
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
            "parameters": list(utils.EXPERIMENT["slow_control_parameters"])
        },
    }

    pkg = importlib.resources.files("legend_data_monitor")
    # one plot dictionary per subsystem; each produces its own v1 + contract file
    subsystems_dict = {}
    for name in ("geds-dict.yaml", "spms-dict.yaml", "pmts-dict.yaml"):
        with open(pkg / "settings" / name) as f:
            subsystems_dict |= yaml.load(f, Loader=yaml.CLoader)

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
        "subsystems": subsystems_dict,
    }

    phy_folder = os.path.join(
        output_folder, ref_version, "generated/plt/hit", data_type
    )
    qcp_path = os.path.join(
        phy_folder, period, run, f"l200-{period}-{run}-qcp_summary.yaml"
    )
    os.makedirs(os.path.join(phy_folder, period, run, "mtg/pdf"), exist_ok=True)

    # ===========================================================================================
    # Detect not-yet-analyzed files (rsync bookkeeping)
    # ===========================================================================================

    rsync_path = os.path.join(
        output_folder, ref_version, "generated", "tmp", "mtg", period, run
    )
    os.makedirs(rsync_path, exist_ok=True)
    timestamp_file = os.path.join(rsync_path, "last_checked_timestamp.txt")

    last_checked = None
    if os.path.exists(timestamp_file):
        with open(timestamp_file) as file:
            last_checked = file.read().strip()

    current_files = os.listdir(source_dir)
    new_files = []
    for file in current_files:
        file_path = os.path.join(source_dir, file)
        current_timestamp = os.path.getmtime(file_path)
        if last_checked is None or current_timestamp > float(last_checked):
            new_files.append(file)

    if new_files:
        # keep only files with correct ending (discard ones still under processing)
        new_files = sorted(f for f in new_files if len(re.findall(r"\d+", f)) == 6)

    last_cycle = new_files[-1].split("-")[-2] if new_files else None

    # ===========================================================================================
    # Task definitions
    # ===========================================================================================

    def task_check_calibration(logger=None):
        if _qcp_file_is_populated(qcp_path):
            utils.logger.info("...qcp summary already populated, skipping")
            return
        utils.logger.info("...inspecting calibration data!")
        check_calib(
            auto_dir_path=auto_dir_path,
            output_folder=phy_folder,
            period=period,
            current_run=run,
            data_type=data_type,
            partition=partition,
            save_pdf=save_pdf,
            render=render_plots,
        )
        utils.logger.info("...done!")

    def task_subsystem_plots(logger=None):
        utils.logger.info(f"New files found: {' '.join(new_files)}")
        # create the file containing the keys with correct format to be later
        # used by legend-data-monitor (recreated every time; NOT append)
        keys_file = os.path.join(rsync_path, "new_keys.filekeylist")
        with open(keys_file, "w") as f:
            for new_file in new_files:
                f.write(new_file.split("-tier")[0] + "\n")

        with open(keys_file) as f:
            key_lines = f.readlines()
        num_lines = len(key_lines)

        if num_lines > chunk_size:
            # split lines into chunks and write to multiple files
            for idx, i in enumerate(range(0, num_lines, chunk_size), start=1):
                chunk = key_lines[i : i + chunk_size]
                output_file = os.path.join(
                    rsync_path, f"new_keys_part_{i // chunk_size + 1}.filekeylist"
                )
                with open(output_file, "w") as out_f:
                    out_f.writelines(chunk)
                total_parts = (num_lines + chunk_size - 1) // chunk_size
                utils.logger.debug(
                    f"[{idx}/{total_parts}] Created file: {output_file} with {len(chunk)} lines."
                )
                core.auto_control_plots(
                    my_config, output_file, "", {}, render=render_plots
                )
                plt.close("all")
        else:
            utils.logger.debug(f"... file has {num_lines} lines. No need to split.")
            core.auto_control_plots(my_config, keys_file, "", {}, render=render_plots)

    def task_build_monitoring_hdf(logger=None):
        files_folder = os.path.join(output_folder, ref_version)
        monitoring.build_new_files(files_folder, period, run, data_type=data_type)
        contract_build.build_all_contract_files(
            files_folder,
            period,
            run,
            metadata_path=os.path.join(auto_dir_path, "inputs"),
            data_type=data_type,
        )
        monitoring.write_spms_production_keys(
            phy_folder,
            period,
            run,
            auto_dir_path,
            start_key=utils.get_start_key(auto_dir_path, data_type, period, run),
            data_type=data_type,
        )

    def task_lar_summary(logger=None):
        """Summarise the LAr veto performance of the run from the evt tier (~20 s)."""
        evt_dir = os.path.join(
            auto_dir_path, "generated/tier/evt", data_type, period, run
        )
        files = sorted(glob.glob(os.path.join(evt_dir, "*.lh5")))
        if not files:
            utils.logger.info("...no evt files for %s/%s; no LAr summary", period, run)
            return
        names = {
            info["daq_rawid"]: name
            for name, info in utils.build_spms_info(
                os.path.join(auto_dir_path, "inputs")
            ).items()
        }
        written = monitoring.write_lar_summary(
            phy_folder, period, run, files, rawid_to_name=names, data_type=data_type
        )
        utils.logger.info("...LAr summary keys written: %s", written)

        # per-SiPM SPE spectra: validates the PE calibration in force (~5 min)
        hit_dir = os.path.join(
            auto_dir_path, "generated/tier/hit", data_type, period, run
        )
        hit_files = sorted(glob.glob(os.path.join(hit_dir, "*.lh5")))
        written = monitoring.write_spe_spectrum(
            phy_folder,
            period,
            run,
            hit_files,
            files,
            rawid_to_name=names,
            data_type=data_type,
        )
        utils.logger.info("...SPE spectrum keys written: %s", written)

    def task_muon_summary(logger=None):
        """Summarise the muon veto from the pmts dsp stream (~1 min).

        This production has no evt_muon group and the muon DAQ triggers
        independently of the geds stream, so the per-trigger multiplicity and
        summed light come from the per-PMT dsp rows and the ge-coincidence
        fractions from evt/coincident.
        """
        dsp_dir = os.path.join(
            auto_dir_path, "generated/tier/dsp", data_type, period, run
        )
        dsp_files = sorted(glob.glob(os.path.join(dsp_dir, "*.lh5")))
        if not dsp_files:
            utils.logger.info("...no dsp files for %s/%s; no muon summary", period, run)
            return
        names = {
            info["daq_rawid"]: name
            for name, info in utils.build_pmts_info(
                os.path.join(auto_dir_path, "inputs")
            ).items()
        }
        if not names:
            utils.logger.info("...no pmts in the channel map; no muon summary")
            return
        evt_dir = os.path.join(
            auto_dir_path, "generated/tier/evt", data_type, period, run
        )
        evt_files = sorted(glob.glob(os.path.join(evt_dir, "*.lh5")))
        written = monitoring.write_muon_summary(
            phy_folder,
            period,
            run,
            dsp_files,
            evt_files,
            rawid_to_name=names,
            data_type=data_type,
        )
        utils.logger.info("...muon summary keys written: %s", written)

    def task_strip_transport(logger=None):
        """Drop the v1 classifier pivots of the period's finished runs.

        They were only the transport to the contract build and the res files
        (qc_plots reads them too, so this runs last). The current run is left
        alone: it is still appending, and stripping mid-run would make the
        contract rebuild bin only post-strip chunks. Earlier runs are closed
        once this run exists, so they are safe -- and each strip re-verifies
        the contract holds every key before removing anything.
        """
        files_folder = os.path.join(output_folder, ref_version)
        period_dir = os.path.join(phy_folder, period)
        for done_run in sorted(os.listdir(period_dir)):
            if done_run >= run or not re.fullmatch(r"r\d+", done_run):
                continue
            for subsystem in contract_schema.SUBSYSTEMS:
                repack.strip_transport_pivots(
                    files_folder, period, done_run, data_type, subsystem=subsystem
                )

    def task_render_plots(logger=None):
        """Draw the run's figures from the contract file.

        Separate from the data tasks on purpose: it reads only the contract,
        so it is cheap, it can be skipped (--plots off) and re-run later with
        `legend-data-monitor plot_run` without touching the production tree.
        """
        saved = render_run_plots(
            os.path.join(output_folder, ref_version),
            period,
            run,
            data_type,
            logger,
        )
        utils.logger.info("...rendered %d figure(s)", len(saved))

    def task_slow_control(logger=None):
        core.retrieve_scdb(scdb, port, pswd)

    mtg_folder = os.path.join(
        output_folder, ref_version, "generated/plt/hit", data_type
    )

    def task_phy_summary_plots(logger=None):
        os.makedirs(mtg_folder, exist_ok=True)
        avail_runs = sorted(os.listdir(os.path.join(mtg_folder, period)))
        avail_runs = [ar for ar in avail_runs if re.fullmatch(r"r\d{3}", ar)]
        if not avail_runs:
            utils.logger.debug("...no available runs to summarize")
            return
        start_key = (
            sorted(os.listdir(os.path.join(search_directory, avail_runs[0])))[0]
        ).split("-")[4]
        summary_plots(
            auto_dir_path=auto_dir_path,
            phy_mtg_data=mtg_folder,
            output_folder=mtg_folder,
            start_key=start_key,
            period=period,
            current_run=run,
            runs=avail_runs,
            last_checked=last_checked,
            last_cycle=last_cycle,
            data_type=data_type,
            partition=partition,
            escale_val=escale_val,
            save_pdf=save_pdf,
            render=render_plots,
        )

    def task_qc_plots(logger=None):
        avail_runs = sorted(os.listdir(os.path.join(mtg_folder, period)))
        avail_runs = [ar for ar in avail_runs if re.fullmatch(r"r\d{3}", ar)]
        if not avail_runs:
            return
        start_key = (
            sorted(os.listdir(os.path.join(search_directory, avail_runs[0])))[0]
        ).split("-")[4]
        qc_avg_series(
            auto_dir_path=auto_dir_path,
            output_folder=mtg_folder,
            start_key=start_key,
            period=period,
            current_run=run,
            save_pdf=save_pdf,
            render=render_plots,
        )

    def task_phy_issues(logger=None):
        """Turn the run's phy verdicts into issue records.

        Last of the phy tasks on purpose: qc_plots writes the discharge /
        saturated / dead-time verdicts after phy_summary_plots, and the
        magnitudes every producer stashed for its verdict live in-process until
        this single emission picks them up.
        """
        avail_runs = sorted(os.listdir(os.path.join(mtg_folder, period)))
        avail_runs = [ar for ar in avail_runs if re.fullmatch(r"r\d{3}", ar)]
        if not avail_runs:
            return
        start_key = (
            sorted(os.listdir(os.path.join(search_directory, avail_runs[0])))[0]
        ).split("-")[4]
        det_info = utils.build_detector_info(
            os.path.join(auto_dir_path, "inputs"), start_key=start_key
        )
        monitoring.check_spms_thresholds(mtg_folder, period, run, data_type=data_type)
        spms_info = utils.build_spms_info(
            os.path.join(auto_dir_path, "inputs"), start_key=start_key
        )
        utils.check_cal_phy_thresholds(
            mtg_folder,
            period,
            run,
            data_type,
            det_info["detectors"],
            detector_info=det_info["detectors"] | spms_info,
            data_type=data_type,
        )

    task_list = [tasks.Task("check_calibration", task_check_calibration, period, run)]
    if new_files:
        task_list.append(
            tasks.Task("build_subsystem_data", task_subsystem_plots, period, run)
        )
        task_list.append(
            tasks.Task("build_monitoring_hdf", task_build_monitoring_hdf, period, run)
        )
        if render_plots:
            task_list.append(tasks.Task("render_plots", task_render_plots, period, run))
        if cluster == "lngs" and get_sc is True:
            task_list.append(tasks.Task("slow_control", task_slow_control, period, run))
        task_list.append(
            tasks.Task("phy_summary_plots", task_phy_summary_plots, period, run)
        )
        task_list.append(tasks.Task("qc_plots", task_qc_plots, period, run))
        task_list.append(tasks.Task("lar_summary", task_lar_summary, period, run))
        task_list.append(tasks.Task("muon_summary", task_muon_summary, period, run))
        task_list.append(tasks.Task("phy_issues", task_phy_issues, period, run))
        task_list.append(
            tasks.Task("strip_transport", task_strip_transport, period, run)
        )
    else:
        utils.logger.debug("No new files were detected.")

    log_root = logs.log_tree_root(os.path.join(output_folder, ref_version))
    results, exit_code = tasks.run_tasks(task_list, log_root)

    # update the last checked timestamp only when everything succeeded, so a
    # failed invocation is retried on the next cron cycle
    if exit_code == tasks.EXIT_OK and current_files:
        with open(timestamp_file, "w") as file:
            file.write(
                str(
                    os.path.getmtime(
                        max(
                            [os.path.join(source_dir, f) for f in current_files],
                            key=os.path.getmtime,
                        )
                    )
                )
            )

    return exit_code


# headline (flag, param, unit) triples rendered as per-string PNGs after each
# contract build; missing keys are skipped so datatype/config changes stay safe.
# Defined per subsystem in settings/experiment.yaml.
HEADLINE_PNG_KEYS = [
    tuple(entry) for entry in utils.EXPERIMENT["headline_plots"]["geds"]
]

# same for the spms contract, grouped by barrel and position instead of string
SPMS_HEADLINE_PNG_KEYS = [
    ("IsBsln", "WfMode_var", "%"),
    ("IsBsln", "CurrFwhm", "ADC"),
    ("IsBsln", "NPulses", "per window"),
    ("All", "HasAnyNoise", "fraction"),
]


def render_run_plots(
    files_folder: str,
    period: str,
    run: str,
    data_type: str = "phy",
    logger=None,
) -> list:
    """Render a run's per-string PNGs from its contract-v2 file.

    Reads only the contract file, so it needs no access to the production
    tree: figures for a run processed with ``--plots off`` can be regenerated
    afterwards in seconds (``legend-data-monitor plot_run``).

    The SAVED_PLOT log lines these emit are the attachment source for
    unattended agents (see docs/auto-giorgio-integration.md).

    Returns
    -------
    list
        Absolute paths of the figures written.
    """
    # SAVED_PLOT lines are a consumer contract, so always announce on some
    # logger; the per-task one when running in the pipeline, else the package's
    logger = logger if logger is not None else utils.logger
    run_dir = os.path.join(files_folder, "generated/plt/hit", data_type, period, run)
    v2_file = os.path.join(run_dir, f"l200-{period}-{run}-{data_type}-geds-schema2.hdf")
    spms_file = v2_file.replace("-geds-schema2.hdf", "-spms-schema2.hdf")
    if not os.path.isfile(v2_file) and not os.path.isfile(spms_file):
        logger.warning("no contract-v2 file to render PNGs from: %s", v2_file)
        return []
    saved = []
    detector_map = None
    if os.path.isfile(v2_file):
        detector_map = contract_reader.read_frame(v2_file, "detector_map")
        for flag, param, unit in HEADLINE_PNG_KEYS:
            try:
                binned = contract_reader.read_binned_series(
                    v2_file, flag, param, "10min"
                )
            except KeyError:
                logger.debug("...no %s_%s in %s, skip PNG", flag, param, v2_file)
                continue
            for string, group in detector_map.groupby("string"):
                saved += contract_plots.plot_binned_series(
                    binned,
                    run_dir,
                    f"{flag}_{param}_st{int(string):02d}",
                    title=f"{flag} {param} — string {string} ({period} {run}, 10min bins)",
                    unit=unit,
                    detectors=list(group["name"]),
                    logger=logger,
                )

    if os.path.isfile(spms_file):
        spms_map = contract_reader.read_frame(spms_file, "detector_map")
        for flag, param, unit in SPMS_HEADLINE_PNG_KEYS:
            try:
                binned = contract_reader.read_binned_series(
                    spms_file, flag, param, "10min"
                )
            except KeyError:
                continue
            for (barrel, position), group in spms_map.groupby(["barrel", "position"]):
                saved += contract_plots.plot_binned_series(
                    binned,
                    run_dir,
                    f"{flag}_{param}_{barrel}_{position}",
                    title=f"{flag} {param} — {barrel} {position} ({period} {run}, 10min bins)",
                    unit=unit,
                    detectors=list(group["name"]),
                    logger=logger,
                    envelope=param != "HasAnyNoise",
                )

    # the full monitoring figure set, from the period contract file(s)
    output_folder = os.path.join(files_folder, "generated/plt/hit", data_type)
    common = dict(
        detector_map=detector_map,
        data_type=data_type,
        save_pdf=True,
        logger=logger,
    )
    saved += qc_plots_mod.plot_qc_rate_series(output_folder, period, run, **common)
    saved += qc_plots_mod.plot_qc_average(output_folder, period, run, **common)
    saved += qc_plots_mod.plot_classifier_distributions(
        output_folder, period, run, **common
    )
    saved += summary_plots_mod.plot_ft_summary(output_folder, period, run, **common)
    saved += summary_plots_mod.plot_event_rate_qc(output_folder, period, run, **common)
    for metric in ["TrapemaxCtcCal", "BlStd", "Baseline", "Trapemax"]:
        saved += summary_plots_mod.plot_detector_summary(
            output_folder, period, run, metric=metric, **common
        )
    saved += stability_plots.plot_stability_series(output_folder, period, run, **common)
    cal_common = dict(detector_map=detector_map, save_pdf=True, logger=logger)
    saved += stability_plots.plot_fep_gain(output_folder, period, run, **cal_common)
    saved += calib_plots.plot_psd_stability(output_folder, period, run, **cal_common)
    saved += calib_plots.plot_escale_panels(output_folder, period, run, **cal_common)
    return saved


# kept as the in-pipeline name; plot_run calls render_run_plots directly
_render_headline_pngs = render_run_plots


def _qcp_file_is_populated(filepath: str) -> bool:
    """Return True if the qcp summary file exists and has at least one non-null cal entry."""
    if not os.path.isfile(filepath):
        return False
    with open(filepath) as f:
        data = yaml.safe_load(f)
    if not data:
        return False
    for det_data in data.values():
        cal = det_data.get("cal", {})
        if any(v is not None for v in cal.values()):
            return True
    return False


def _detector_map_frame(det_info: dict):
    """name/rawid/string/position frame for the plots/ renderers."""
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "name": name,
                "rawid": info.get("daq_rawid"),
                "string": info.get("string"),
                "position": info.get("position"),
                "processable": info.get("processable"),
            }
            for name, info in det_info["detectors"].items()
        ]
    )


def summary_plots(
    auto_dir_path: str,
    phy_mtg_data: str,
    output_folder: str,
    start_key: str,
    period: str,
    current_run: str,
    runs: list,
    last_checked: str,
    last_cycle: str,
    data_type: str = "phy",
    partition: bool = False,
    escale_val: float = 2039.0,
    save_pdf: bool = False,
    quadratic: bool = False,
    render: bool = True,
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
    last_checked : str
        Timestamp of the last check.
    last_cycle : str
        Last cycle of the inspect list; format: YYYYMMDDThhmmssZ.
    data_type : str
        Data type to load; default: 'phy'.
    partition : bool
        False if not partition data; default: False.
    escale_val : float
        Energy scale at which evaluating the gain differences; default: 2039 keV (76Ge Qbb).
    save_pdf : bool
        True if you want to save pdf files too; default: False.
    quadratic : bool
        True if you want to plot the quadratic resolution too; default: False.
    render : bool
        Draw the figures from the contract after the data pass; default: True.
    """
    det_info = utils.build_detector_info(
        os.path.join(auto_dir_path, "inputs"), start_key=start_key
    )

    # stability series (data pass; figures come from the contract below)
    results = monitoring.collect_stability_series(
        auto_dir_path,
        phy_mtg_data,
        output_folder,
        data_type,
        period,
        runs,
        current_run,
        det_info,
        escale_val,
        last_checked,
        partition,
        quadratic,
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
            run_to_apply=run_to_apply,
        )

    # FT failure rate plots
    if data_type not in ["ssc", "lac", "rdc"]:

        # qc classifier fractions + FT/event-rate/dead-time data
        monitoring.qc_distributions(
            auto_dir_path,
            phy_mtg_data,
            output_folder,
            start_key,
            period,
            current_run,
            det_info,
        )

        monitoring.qc_and_evt_summary_plots(
            auto_dir_path,
            phy_mtg_data,
            output_folder,
            start_key,
            period,
            current_run,
            det_info,
        )

    if render:
        detector_map = _detector_map_frame(det_info)
        common = dict(detector_map=detector_map, data_type=data_type, save_pdf=save_pdf)
        stability_plots.plot_stability_series(
            output_folder, period, current_run, quadratic=quadratic, **common
        )
        for k in results.keys():
            summary_plots_mod.plot_detector_summary(
                output_folder, period, current_run, metric=k, **common
            )
        if data_type not in ["ssc", "lac", "rdc"]:
            qc_plots_mod.plot_classifier_distributions(
                output_folder, period, current_run, **common
            )
            summary_plots_mod.plot_ft_summary(
                output_folder, period, current_run, **common
            )
            summary_plots_mod.plot_event_rate_qc(
                output_folder, period, current_run, last_cycle=last_cycle, **common
            )


def check_calib(
    auto_dir_path: str,
    output_folder: str,
    period: str,
    current_run: str,
    data_type: str = "phy",
    partition: bool = False,
    save_pdf: bool = False,
    render: bool = True,
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
        )
        calibration.check_psd(
            auto_dir_path,
            cal_path,
            pars_files_list,
            output_folder,
            period,
            current_run,
            det_info,
        )

        detector_status = calibration.check_escale(
            auto_dir_path,
            cal_path,
            output_folder,
            period,
            current_run,
            det_info,
        )

        if render:
            detector_map = _detector_map_frame(det_info)
            common = dict(detector_map=detector_map, save_pdf=save_pdf)
            stability_plots.plot_fep_gain(output_folder, period, current_run, **common)
            calib_plots.plot_psd_stability(output_folder, period, current_run, **common)
            calib_plots.plot_escale_panels(
                output_folder,
                period,
                current_run,
                detector_status=detector_status,
                exclude_period=["p05", "p10", "p11", "p13", "p15", "p17"],
                **common,
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
            data_type=data_type,
        )
        if render:
            stability_plots.plot_fep_gain(
                output_folder,
                period,
                current_run,
                detector_map=_detector_map_frame(det_info),
                data_type=data_type,
                save_pdf=save_pdf,
            )

        utils.logger.debug(
            f"...we do not inspect PSD time stability in {data_type} data"
        )

    utils.check_cal_phy_thresholds(
        output_folder,
        period,
        current_run,
        data_type if data_type in ["lac", "ssc", "rdc"] else "cal",
        det_info["detectors"],
        detector_info=det_info["detectors"],
        data_type=data_type,
    )


def qc_avg_series(
    auto_dir_path: str,
    output_folder: str,
    start_key: str,
    period: str,
    current_run: str,
    save_pdf: bool = False,
    render: bool = True,
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
    render : bool
        Draw the figures from the contract after the data pass; default: True.
    """
    det_info = utils.build_detector_info(
        os.path.join(auto_dir_path, "inputs/"), start_key=start_key
    )

    monitoring.qc_average(auto_dir_path, output_folder, det_info, period, current_run)
    monitoring.qc_time_series(
        auto_dir_path, output_folder, det_info, period, current_run
    )

    if render:
        detector_map = _detector_map_frame(det_info)
        qc_plots_mod.plot_qc_average(
            output_folder,
            period,
            current_run,
            detector_map=detector_map,
            save_pdf=save_pdf,
        )
        qc_plots_mod.plot_qc_rate_series(
            output_folder,
            period,
            current_run,
            detector_map=detector_map,
            save_pdf=save_pdf,
        )
