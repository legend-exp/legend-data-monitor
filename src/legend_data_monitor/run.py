from __future__ import annotations

import argparse
import os
import sys

import yaml

import legend_data_monitor


def main():
    """legend-data-monitor's starting point.

    Here you define the path to the YAML configuration file you want to use when generating the plots.
    To learn more, have a look at the help section:

    .. code-block:: console
      $ legend-data-monitor --help

    """
    parser = argparse.ArgumentParser(
        prog="legend-data-monitor", description="Software's command-line interface."
    )

    # global options
    parser.add_argument(
        "--version",
        action="store_true",
        help="""Print version and exit.""",
    )

    subparsers = parser.add_subparsers()

    # functions for different purpouses
    add_user_scdb(subparsers)
    add_user_config_parser(subparsers)
    add_user_bunch_parser(subparsers)
    add_user_rsync_parser(subparsers)
    add_auto_prod_parser(subparsers)
    add_auto_run_parser(subparsers)
    add_get_exposure(subparsers)
    add_get_runinfo(subparsers)

    if len(sys.argv) < 2:
        parser.print_usage(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.version:
        legend_data_monitor.utils.logger.info(
            "Version: %s", legend_data_monitor.__version__
        )
        sys.exit()

    args.func(args)


def add_user_scdb(subparsers):
    """Configure :func:`.core.control_plots` command line interface."""
    parser_auto_prod = subparsers.add_parser(
        "user_scdb",
        description="""Retrieve Slow Control data from database by giving a full config file with parameters/subsystems info to plot. Available only when working in LNGS machines.""",
    )
    parser_auto_prod.add_argument(
        "--config",
        help="""Path to config file (e.g. \"some_path/config_L200_r001_phy.yaml\") or dictionary.""",
    )
    parser_auto_prod.add_argument(
        "--port",
        help="""Local port.""",
    )
    parser_auto_prod.add_argument(
        "--pswd",
        help="""Password to get access to the Slow Control database (check on Confluence).""",
    )
    parser_auto_prod.set_defaults(func=user_scdb_cli)


def user_scdb_cli(args):
    """Pass command line arguments to :func:`.core.retrieve_scdb`."""
    config_file = args.config
    port = args.port
    password = args.pswd

    legend_data_monitor.core.retrieve_scdb(config_file, port, password)


def add_user_config_parser(subparsers):
    """Configure :func:`.core.control_plots` command line interface."""
    parser_auto_prod = subparsers.add_parser(
        "user_prod",
        description="""Inspect LEGEND HDF5 (LH5) processed data by giving a full config file with parameters/subsystems info to plot.""",
    )
    parser_auto_prod.add_argument(
        "--config",
        help="""Path to config file (e.g. \"some_path/config_L200_r001_phy.yaml\").""",
    )
    parser_auto_prod.set_defaults(func=user_config_cli)


def user_config_cli(args):
    """Pass command line arguments to :func:`.core.control_plots`."""
    config_file = args.config

    legend_data_monitor.core.control_plots(config_file)


def add_user_bunch_parser(subparsers):
    """Configure :func:`.core.control_plots` command line interface."""
    parser_auto_prod = subparsers.add_parser(
        "user_bunch",
        description="""Inspect LEGEND HDF5 (LH5) processed data by giving a full config file with parameters/subsystems info to plot. Files will be bunched in groups of n_files files each, and every time the code is run you will append new data to the previously generated ones.""",
    )
    parser_auto_prod.add_argument(
        "--config",
        help="""Path to config file (e.g. \"some_path/config_L200_r001_phy.yaml\").""",
    )
    parser_auto_prod.add_argument(
        "--n_files",
        help="""Number (int) of files of a given run you want to inspect at each cycle.""",
    )
    parser_auto_prod.set_defaults(func=user_bunch_cli)


def user_bunch_cli(args):
    """Pass command line arguments to :func:`.core.control_plots`."""
    config_file = args.config
    n_files = args.n_files

    legend_data_monitor.core.control_plots(config_file, n_files)


def add_user_rsync_parser(subparsers):
    """Configure :func:`.core.auto_control_plots` command line interface."""
    parser_auto_prod = subparsers.add_parser(
        "user_rsync_prod",
        description="""Inspect LEGEND HDF5 (LH5) processed data by giving a full config file with parameters/subsystems info to plot, syncing with new produced data.""",
    )
    parser_auto_prod.add_argument(
        "--config",
        help="""Path to the configuration info or dictionary with config info.""",
    )
    parser_auto_prod.add_argument(
        "--keys",
        help="""Path to file containing new keys to inspect (e.g. \"some_path/new_keys.filekeylist\").""",
    )
    parser_auto_prod.set_defaults(func=user_rsync_cli)


def user_rsync_cli(args):
    """Pass command line arguments to :func:`.core.auto_control_plots`."""
    config_file = args.config
    keys_file = args.keys

    legend_data_monitor.core.auto_control_plots(config_file, keys_file, "", {})


def add_auto_prod_parser(subparsers):
    """Configure :func:`.core.auto_control_plots` command line interface."""
    parser_auto_prod = subparsers.add_parser(
        "auto_prod",
        description="""Inspect LEGEND HDF5 (LH5) processed data by giving a partial config file with parameters/subsystems info to plot,\na file with a list of keys to load, and a path to the production environment.""",
    )
    parser_auto_prod.add_argument(
        "--plot_config",
        help="""Path to config file with parameters/subsystems info to plot (e.g. \"some_path/plot_config.yaml\").""",
    )
    parser_auto_prod.add_argument(
        "--filekeylist",
        help="""File-keylist name (e.g. \"all-l200-p02-r001-phy.filekeylist\").""",
    )
    parser_auto_prod.add_argument(
        "--prod_path",
        help="""Path to production environment (e.g. \"/data1/shared/l200/l200-prodenv/prod-ref/vXX.YY/\").\nHere, you should find \"config.yaml\" containing input/output folders info.""",
    ) 
    parser_auto_prod.set_defaults(func=auto_prod_cli)


def auto_prod_cli(args):
    """Pass command line arguments to :func:`.core.auto_control_plots`."""
    file_keys = args.filekeylist
    prod_path = args.prod_path
    plot_config = args.plot_config

    # get the production config file
    prod_config_file = os.path.join(prod_path, "config.yaml")
    with open(prod_config_file) as f:
        prod_config = yaml.load(f, Loader=yaml.CLoader)

    # get the filelist file path
    folder_filelists = prod_config["setups"]["l200"]["paths"]["tmp_filelists"][3:]
    file_keys = os.path.join(prod_path, folder_filelists, args.filekeylist)

    legend_data_monitor.core.auto_control_plots(
        plot_config, file_keys, prod_path, prod_config
    )


def add_auto_run_parser(subparsers):
    """Configure :func:`.core.auto_run` command line interface."""
    parser_auto_run = subparsers.add_parser(
        "auto_run",
        description="""Inspect LEGEND HDF5 (LH5) processed data (and Slow Control data from lngs-login cluster) for a specific period and run (if specified; otherwise the latest being processed are used); plots and summary files are saved; automatic alert emails are sent.""",
    )
    parser_auto_run.add_argument(
        "--cluster",
        default='lngs',
        help="Name of the cluster where you are working; pick among 'lngs' (default) or 'nersc'.",
    )
    parser_auto_run.add_argument(
        "--ref_version",
        help="Version of processed data to inspect (eg. tmp-auto or ref-v2.1.0).",
    )
    parser_auto_run.add_argument(
        "--output_folder",
        help="Path where to store the automatic results (plots and summary files).",
    )
    parser_auto_run.add_argument(
        "--data_type",
        default="phy",
        help="Data type to load. Default: 'phy' (alternatives: 'lac', 'ssc', 'rdc').",
    )
    parser_auto_run.add_argument(
        "--partition",
        default=False,
        help="False (default) if not partition data, else True",
    )
    parser_auto_run.add_argument(
        "--pswd",
        help="Password to access the Slow Control database (NOT available on NERSC).",
    )
    parser_auto_run.add_argument(
        "--sc",
        default=False,
        help="Boolean for retrieving Slow Control data (default: False).",
    )
    parser_auto_run.add_argument(
        "--port",
        default=8282,
        help="Port necessary to retrieve the Slow Control database (default: 8282).",
    )
    parser_auto_run.add_argument(
        "--pswd_email",
        default=None,
        help="Password to access the legend.data.monitoring@gmail.com account for sending alert messages. Optional; if not specified, no automatic alerts will be sent.",
    )
    parser_auto_run.add_argument(
        "--chunk_size",
        default=20,
        type=int,
        help="Maximum integer number of files to read at each loop in order to avoid the process to be killed; default: 20 files.",
    )
    parser_auto_run.add_argument(
        "--p",
        default=None,
        help="Period to inspect. Optional; if not specified, the latest period processed for the given prodenv ref version is used.",
    )
    parser_auto_run.add_argument(
        "--r",
        default=None,
        help="Run to inspect. Optional; if not specified, the latest run processed for the given period is used.",
    )
    parser_auto_run.add_argument(
        "--escale",
        default=2039.0,
        type=float,
        help="Energy scale at which evaluating the gain differences; default: 2039 keV (76Ge Qbb).",
    )
    parser_auto_run.add_argument(
        "--pdf",
        default=False,
        help="True if you want to save pdf files too; default: False",
    )

    parser_auto_run.set_defaults(func=auto_run_cli)


def auto_run_cli(args):
    """Pass command line arguments to :func:`.core.auto_run`."""
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
    data_type = args.data_type

    legend_data_monitor.automatic_run.auto_run(
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
    )


def add_get_exposure(subparsers):
    """Configure :func:`.core.retrieve_exposure` command line interface."""
    parser_auto_prod = subparsers.add_parser(
        "get_exposure",
        description="""Retrieve exposure for active detectors, with or without valid PSD flag.""",
    )
    parser_auto_prod.add_argument("--period", help="Period to inspect.")
    parser_auto_prod.add_argument("--run", nargs="+", help="List of runs.")
    parser_auto_prod.add_argument(
        "--runinfo_path",
        help="Path to the runinfo.yaml/.json file containing summary run information and livetime in seconds.",
    )
    parser_auto_prod.add_argument("--path", help="Path to the processed data.")
    parser_auto_prod.add_argument("--data_version", help="Version of processed data.")
    parser_auto_prod.set_defaults(func=get_exposure_cli)


def get_exposure_cli(args):
    """Pass command line arguments to :func:`.core.retrieve_exposure`."""
    period = args.period
    run = args.run
    runinfo_path = args.runinfo_path
    path = args.path
    data_version = args.data_version

    legend_data_monitor.core.retrieve_exposure(
        period, run, runinfo_path, path, data_version
    )


def add_get_runinfo(subparsers):
    """Configure :func:`.core.build_runinfo` command line interface."""
    parser_auto_prod = subparsers.add_parser(
        "get_runinfo",
        description="""Build runinfo.yaml summary file.""",
    )
    parser_auto_prod.add_argument("--path", help="Path to the processed data.")
    parser_auto_prod.add_argument(
        "--proc_folder", help="Path to the processed folder with data."
    )
    parser_auto_prod.add_argument("--data_version", help="Version of processed data.")
    parser_auto_prod.add_argument(
        "--output",
        default=None,
        help="Path to the output folder where to save runinfo.yaml; default: None (the file will be saved under <proc_folder>/<data_version>/inputs/datasets/).",
    )
    parser_auto_prod.set_defaults(func=get_runinfo_cli)


def get_runinfo_cli(args):
    """Pass command line arguments to :func:`.core.build_runinfo`."""
    path = args.path
    proc_folder = args.proc_folder
    data_version = args.data_version
    output = args.output

    legend_data_monitor.utils.build_runinfo(path, data_version, proc_folder, output)
