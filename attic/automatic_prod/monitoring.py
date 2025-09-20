import argparse
import glob
import os

import legend_data_monitor


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

    func1_parser = subparsers.add_parser(
        "summary_files", help="Run function for creating summary HDF and YAML files."
    )
    func1_parser.add_argument(
        "--path", help="Path to the folder containing the monitoring HDF files."
    )
    func1_parser.add_argument("--period", help="Period to inspect.")
    func1_parser.add_argument("--run", help="Run to inspect.")

    func2_parser = subparsers.add_parser(
        "plot", help="Run function for creating summary plots."
    )
    func2_parser.add_argument(
        "--public_data",
        help="Path to tmp-auto public data files (eg /data2/public/prodenv/prod-blind/tmp-auto).",
        default="/data2/public/prodenv/prod-blind/ref-v1.0.1",
    )
    func2_parser.add_argument(
        "--hdf_files",
        help="Path to generated monitoring hdf files.",
    )
    func2_parser.add_argument("--output", help="Path to output folder.")
    func2_parser.add_argument(
        "--start_key", help="First timestamp of the inspected range."
    )
    func2_parser.add_argument("--period", help="Period to inspect.")
    func2_parser.add_argument(
        "--avail_runs",
        nargs="+",
        type=str,
        help="Available runs to inspect for a given period.",
    )
    func2_parser.add_argument("--current_run", type=str, help="Run under inspection.")
    func2_parser.add_argument(
        "--partition",
        default=False,
        help="False if not partition data; default: False.",
    )
    func2_parser.add_argument(
        "--zoom", default=False, help="True to zoom over y axis; default: False."
    )
    func2_parser.add_argument(
        "--quad_res",
        default=False,
        help="True if you want to plot the quadratic resolution too; default: False.",
    )
    func2_parser.add_argument(
        "--pswd_email",
        default=None,
        help="Password to access the legend.data.monitoring@gmail.com account for sending alert messages.",
    )
    func2_parser.add_argument(
        "--escale",
        default=2039.0,
        type=float,
        help="Energy scale at which evaluating the gain differences; default: 2039 keV (76Ge Qbb).",
    )
    func2_parser.add_argument(
        "--pdf",
        default=False,
        help="True if you want to save pdf files too; default: False.",
    )
    func2_parser.add_argument(
        "--last_checked",
        help="Timestamp of the last check.",
    )

    func3_parser = subparsers.add_parser(
        "check_calib",
        help="Check calibration stability in calibration runs and create monitoring summary file.",
    )
    func3_parser.add_argument(
        "--public_data",
        help="Path to tmp-auto public data files (eg /data2/public/prodenv/prod-blind/tmp-auto).",
        default="/data2/public/prodenv/prod-blind/ref-v1.0.1",
    )
    func3_parser.add_argument("--output", help="Path to output folder.")
    func3_parser.add_argument("--period", help="Period to inspect.")
    func3_parser.add_argument("--current_run", type=str, help="Run under inspection.")
    func3_parser.add_argument(
        "--pdf",
        default=False,
        help="True if you want to save pdf files too; default: False.",
    )
    func3_parser.add_argument(
        "--pswd_email",
        default=None,
        help="Password to access the legend.data.monitoring@gmail.com account for sending alert messages.",
    )

    func4_parser = subparsers.add_parser(
        "qc_avg_series",
        help="Plot and raise warning for PSD stability in calibration runs.",
    )
    func4_parser.add_argument(
        "--public_data",
        help="Path to tmp-auto public data files (eg /data2/public/prodenv/prod-blind/tmp-auto).",
        default="/data2/public/prodenv/prod-blind/ref-v1.0.1",
    )
    func4_parser.add_argument("--output", help="Path to output folder.")
    func4_parser.add_argument(
        "--start_key", help="First timestamp of the inspected range."
    )
    func4_parser.add_argument("--period", help="Period to inspect.")
    func4_parser.add_argument("--current_run", type=str, help="Run under inspection.")
    func4_parser.add_argument(
        "--pdf",
        default=False,
        help="True if you want to save pdf files too; default: False.",
    )

    args = parser.parse_args()

    if args.command == "summary_files":
        legend_data_monitor.monitoring.build_new_files(args.path, args.period, args.run)

    elif args.command == "plot":
        auto_dir_path = args.public_data
        phy_mtg_data = args.hdf_files
        output_folder = args.output
        start_key = args.start_key
        period = args.period
        runs = args.avail_runs
        current_run = args.current_run
        pswd_email = args.pswd_email
        save_pdf = args.pdf
        escale_val = args.escale
        last_checked = args.last_checked
        partition = args.partition
        quadratic = args.quad_res
        zoom = args.zoom

        det_info = legend_data_monitor.utils.build_detector_info(
            os.path.join(auto_dir_path, "inputs"), start_key=start_key
        )

        # qc classifier plots
        legend_data_monitor.monitoring.qc_distributions(
            auto_dir_path,
            phy_mtg_data,
            output_folder,
            start_key,
            period,
            current_run,
            det_info,
            save_pdf,
        )

        # stability plots
        results = legend_data_monitor.monitoring.plot_time_series(
            auto_dir_path,
            phy_mtg_data,
            output_folder,
            start_key,
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

        # load par summary yaml cal files
        found = False
        filename = ""
        for tier in ["hit", "pht"]:
            cal_path = os.path.join(auto_dir_path, "generated/par", tier, "cal", period, current_run)
            if os.path.isdir(cal_path):
                found = True
                filename = [f for f in os.listdir(cal_path) if f.endswith('.yaml') or f.endswith('.json')][0].split('.')[0]
                break
        if not found:
            legend_data_monitor.utils.logger.debug(f"No valid folder {cal_path} found. Exiting.")
            return
        pars_path = legend_data_monitor.utils.retrieve_json_or_yaml(cal_path, filename)
        pars = legend_data_monitor.utils.read_json_or_yaml(pars_path)

        # phy box summary pltos
        for k in results.keys():
            pars_dict = pars if k in ['TrapemaxCtcCal'] else None
            legend_data_monitor.monitoring.box_summary_plot(period, current_run, pars_dict, det_info, results[k], legend_data_monitor.utils.MTG_PLOT_INFO[k], output_folder, 'phy', save_pdf)

        legend_data_monitor.utils.check_cal_phy_thresholds(
            output_folder,
            period,
            current_run,
            "phy",
            det_info["detectors"],
            pswd_email,
        )

    elif args.command == "check_calib":
        auto_dir_path = args.public_data
        output_folder = args.output
        period = args.period
        save_pdf = args.pdf
        current_run = args.current_run
        pswd_email = args.pswd_email

        found = False
        for tier in ["hit", "pht"]:
            cal_path = os.path.join(auto_dir_path, "generated/par", tier, "cal", period)
            if os.path.isdir(cal_path):
                found = True
                break
        if not found:
            legend_data_monitor.utils.logger.debug(f"No valid folder {cal_path} found. Exiting.")
            return

        # don't run any check if there are no runs
        cal_runs = os.listdir(cal_path)
        if len(cal_runs) == 0:
            legend_data_monitor.utils.logger.debug("No available calibration runs to inspect. Exiting.")
            return
        first_run = len(cal_runs) == 1

        pars_files_list = sorted(glob.glob(f"{cal_path}/*/*.yaml"))
        if not pars_files_list:
            pars_files_list = sorted(glob.glob(f"{cal_path}/*/*.json"))
        start_key = pars_files_list[0].split("-")[-2]
        det_info = legend_data_monitor.utils.build_detector_info(
            os.path.join(auto_dir_path, "inputs"), start_key=start_key
        )

        legend_data_monitor.calibration.check_calibration(
            auto_dir_path,
            output_folder,
            period,
            current_run,
            first_run,
            det_info,
            save_pdf,
        )

        legend_data_monitor.calibration.check_psd(
            auto_dir_path,
            cal_path,
            pars_files_list,
            output_folder,
            period,
            current_run,
            det_info,
            save_pdf,
        )

        legend_data_monitor.utils.check_cal_phy_thresholds(
            output_folder,
            period,
            current_run,
            "cal",
            det_info["detectors"],
            pswd_email,
        )

    elif args.command == "qc_avg_series":
        auto_dir_path = args.public_data
        output_folder = args.output
        period = args.period
        save_pdf = args.pdf
        current_run = args.current_run
        start_key = args.start_key

        det_info = legend_data_monitor.utils.build_detector_info(
            os.path.join(auto_dir_path, "inputs/"), start_key=start_key
        )

        legend_data_monitor.monitoring.qc_average(
            auto_dir_path, output_folder, det_info, period, current_run, save_pdf
        )
        legend_data_monitor.monitoring.qc_time_series(
            auto_dir_path, output_folder, det_info, period, current_run, save_pdf
        )


if __name__ == "__main__":
    main()
